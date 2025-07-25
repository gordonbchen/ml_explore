from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests
import tiktoken
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from transformers import GPT2LMHeadModel


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50_257
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
        self.transformer.wte.weight = self.lm_head.weight
        self.register_buffer("pos_arange", torch.arange(config.block_size, dtype=torch.long))
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "RESIDUAL_SCALE_INIT"):
                std *= (2 * self.config.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
        sd_keys = [k for k in sd.keys() if not (k.endswith(".attn.causal_mask") or k.endswith("pos_arange"))]

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
                assert sd_hf[k].shape[::-1] == sd[k].shape, "transposed shape doesn't match hugginface"
                sd[k].copy_(sd_hf[k].T)
            else:
                assert sd_hf[k].shape == sd[k].shape, "shape doesn't match huggingface"
                sd[k].copy_(sd_hf[k])

        return model


def calc_loss(model: GPT, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
    return loss


@torch.no_grad()
def generate(model: GPT, prompt: str, n_sequences: int, max_length: int, device: str) -> list[str]:
    model.eval()

    tok = tiktoken.get_encoding("gpt2")
    tokens = torch.tensor(tok.encode(prompt), dtype=torch.long, device=device)
    tokens = tokens.unsqueeze(0).repeat(n_sequences, 1)

    while tokens.shape[-1] < max_length:
        logits = model(tokens[:, -model.config.block_size :])[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_inds = probs.topk(50, dim=-1)
        inds = torch.multinomial(topk_probs, num_samples=1)
        new_inds = topk_inds.gather(1, inds)
        tokens = torch.cat((tokens, new_inds), dim=-1)

    completions = tok.decode_batch(tokens.tolist())
    return completions


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
        self.c_proj.RESIDUAL_SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q, k, v = self.c_attn(x).split(self.n_embed, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_size).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, config.n_embed * 4)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(config.n_embed * 4, config.n_embed)
        self.c_proj.RESIDUAL_SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class DataLoaderLite:
    def __init__(self, B: int, T: int, device: str, ddp_rank: int, ddp_world_size: int) -> None:
        self.B = B
        self.T = T
        tok = tiktoken.get_encoding("gpt2")
        text = self.get_shakespeare_data()
        self.tokens = torch.tensor(tok.encode(text), device=device)
        self.curr_pos = B * T * ddp_rank
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size

    def get_next_batch(self) -> torch.Tensor:
        buf = self.tokens[self.curr_pos : self.curr_pos + (self.B * self.T) + 1]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)

        self.curr_pos += self.B * self.T * self.ddp_world_size
        if (self.curr_pos + (self.B * self.T * self.ddp_world_size) + 1) > len(self.tokens):
            self.curr_pos = self.B * self.T * self.ddp_rank
        return x, y

    def get_shakespeare_data(self, path: str = "data/tiny_shakespeare.txt") -> str:
        path = Path(path)
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            text = requests.get(
                "https://raw.githubusercontent.com/karpathy/char-rnn/refs/heads/master/data/tinyshakespeare/input.txt"
            ).text
            with open(path, "w") as f:
                f.write(text)
            return text

        with open(path, "r") as f:
            text = f.read()
        return text


def get_lr_scheduler(max_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> Callable[[int], float]:
    def lr_scheduler(step: int) -> float:
        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps
        if step > max_steps:
            return min_lr

        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        assert (0 <= decay_ratio) and (decay_ratio <= 1)
        coeff = 0.5 * (1.0 + math.cos(decay_ratio * math.pi))
        return min_lr + (coeff * (max_lr - min_lr))

    return lr_scheduler


def config_optim(model: GPT, lr: float, weight_decay: float, master_process: bool) -> AdamW:
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    if master_process:
        print(f"Total params: {sum(p.numel() for p in model.parameters())}")
        print(f"Decayed tensors, params: {len(decay_params)}, {sum(p.numel() for p in decay_params)}")
        print(f"Non-decayed tensors, params: {len(no_decay_params)}, {sum(p.numel() for p in no_decay_params)}")

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optim = AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
    return optim


if __name__ == "__main__":
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        device = "cuda"
        master_process = True

    torch.manual_seed(1337)
    torch.set_float32_matmul_precision("high")

    config = GPTConfig(vocab_size=50_304)
    model = torch.compile(GPT(config)).to(device)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model if not ddp else model.module

    optim = config_optim(raw_model, lr=6e-4, weight_decay=0.1, master_process=master_process)
    lr_scheduler = get_lr_scheduler(max_lr=6e-4, min_lr=6e-5, warmup_steps=10, max_steps=50)

    batch_size = 524_288  # 2**19, ~0.5M tokens.
    B = 4  # micro batch size.
    assert batch_size % (B * config.block_size * ddp_world_size) == 0, (
        "batch_size must be divisible by tokens per micro batch multiplied by ddp world size"
    )
    grad_accum_steps = batch_size // (B * config.block_size * ddp_world_size)
    if master_process:
        print(f"batch size (tok): {batch_size}, grad accum steps: {grad_accum_steps}")
    train_data = DataLoaderLite(
        B=B, T=config.block_size, device=device, ddp_rank=ddp_rank, ddp_world_size=ddp_world_size
    )
    if master_process:
        print(f"# tokens: {len(train_data.tokens)}")
        print(f"micro batches per epoch: {len(train_data.tokens) // (B * config.block_size * ddp_world_size)}")

    model.train()
    for step in range(50):
        t0 = time.time()

        loss_accum = torch.tensor(0.0, dtype=torch.float32, device=device)
        for micro_step in range(grad_accum_steps):
            x, y = train_data.get_next_batch()
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                loss = calc_loss(model, x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            if ddp:
                model.require_backward_grad_sync = micro_step == (grad_accum_steps - 1)
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
        lr = lr_scheduler(step)
        for param_group in optim.param_groups:
            param_group["lr"] = lr
        optim.step()

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        tokens_per_sec = (B * config.block_size * ddp_world_size * grad_accum_steps) / dt
        if master_process:
            print(f"step: {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | ", end="")
            print(f"norm: {norm.item():.4f} | dt: {dt * 1000:.2f} ms | tok/sec: {tokens_per_sec:.2f}")

    if ddp:
        dist.destroy_process_group()

    # completions = generate(
    #     model, "Hello, I'm a language model,", n_sequences=4, max_length=64, device=device
    # )
    # for c in completions:
    #     print("<", c)
