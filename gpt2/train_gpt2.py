from __future__ import annotations

from dataclasses import dataclass

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layers: int = 12
    n_heads: int = 12
    n_embed: int = 768


class GPT(nn.Module):
    """
    Attention Is All You Need: https://arxiv.org/pdf/1706.03762
    GPT-2: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
    GPT-3: https://arxiv.org/pdf/2005.14165
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embed),
                "wpe": nn.Embedding(config.block_size, config.n_embed),
                "h": nn.ModuleList((Block(config) for _ in range(config.n_layers))),
                "ln_f": nn.LayerNorm(config.n_embed),
            }
        )
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        self.register_buffer("pos_arange", torch.arange(config.block_size, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        assert T <= self.config.block_size, f"cannot forward seq of length {T} > block_size"

        pos_embed = self.transformer.wpe(self.pos_arange[:T])
        tok_embed = self.transformer.wte(x)
        x = pos_embed + tok_embed

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, prompt: str, n_sequences: int, max_length: int) -> list[str]:
        self.eval()

        tok = tiktoken.get_encoding("gpt2")
        tokens = torch.tensor(tok.encode(prompt), dtype=torch.long, device="cuda")
        tokens = tokens.unsqueeze(0).repeat(n_sequences, 1)

        while tokens.shape[-1] < max_length:
            logits = self(tokens[:, -self.config.block_size :])[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_inds = probs.topk(50, dim=-1)
            inds = torch.multinomial(topk_probs, num_samples=1)
            new_inds = topk_inds.gather(1, inds)
            tokens = torch.cat((tokens, new_inds), dim=-1)

        completions = tok.decode_batch(tokens.tolist())
        return completions

    @torch.no_grad()
    @staticmethod
    def from_pretrained(model_name: str) -> GPT:
        configs = {
            "gpt2": GPTConfig(n_layers=12, n_heads=12, n_embed=768),  # 124M.
            "gpt2-medium": GPTConfig(n_layers=24, n_heads=16, n_embed=1024),  # 350M.
            "gpt2-large": GPTConfig(n_layers=36, n_heads=20, n_embed=1280),  # 774M.
            "gpt2-xl": GPTConfig(n_layers=48, n_heads=25, n_embed=1600),  # 1558M.
        }
        assert model_name in configs, f"model_name must be in {configs.keys()}"
        config = configs[model_name]

        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [
            k
            for k in sd.keys()
            if not (k.endswith(".attn.causal_mask") or k.endswith("pos_arange"))
        ]

        # BUG: hf fails to find cache if not running script in same dir.
        model_hf = GPT2LMHeadModel.from_pretrained(model_name)
        sd_hf = model_hf.state_dict()
        assert sd_keys == list(sd_hf.keys()), "gpt state dict doesn't match huggingface reference"
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        for k in sd_keys:
            if any(k.endswith(name) for name in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape, (
                    "transposed shape doesn't match hugginface"
                )
                sd[k].copy_(sd_hf[k].T)
            else:
                assert sd_hf[k].shape == sd[k].shape, "shape doesn't match huggingface"
                sd[k].copy_(sd_hf[k])

        return model


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embed % config.n_heads == 0, "n_embed must be divisible by n_heads"
        self.n_embed = config.n_embed
        self.n_heads = config.n_heads
        self.head_size = config.n_embed // config.n_heads
        self.c_attn = nn.Linear(config.n_embed, config.n_embed * 3)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.full((1, 1, config.block_size, config.block_size), float("-inf")),
                diagonal=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.c_attn(x).split(self.n_embed, dim=-1)

        B, T, C = x.shape
        q = q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_size).transpose(1, 2)

        affin = (q @ k.transpose(-1, -2)) * (k.shape[-1] ** -0.5)
        affin = affin + self.causal_mask[:, :, :T, :T]
        affin = F.softmax(affin, dim=-1)
        y = affin @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, config.n_embed * 4)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(config.n_embed * 4, config.n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


if __name__ == "__main__":
    model = GPT.from_pretrained("gpt2").to("cuda")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    completions = model.generate("Hello, I'm a language model,", n_sequences=4, max_length=64)
    for c in completions:
        print("<", c)
