"""Pre-norm causal decoder from tensor/module primitives, with inspectable attention."""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 64
    width: int = 64
    layers: int = 2
    heads: int = 4
    dropout: float = 0.1
    positions: bool = True
    residuals: bool = True
    normalization: str = "layer"

    def __post_init__(self):
        if min(self.width, self.layers, self.heads, self.block_size) < 1 or self.width % self.heads:
            raise ValueError("Positive dimensions and width divisible by heads required.")
        if not 0 <= self.dropout < 1 or self.vocab_size < 259:
            raise ValueError("Invalid vocabulary/dropout.")
        if self.normalization not in {"layer", "rms", "none"}:
            raise ValueError("Unsupported normalization.")


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return self.gain * x * (x.square().mean(dim=-1, keepdim=True) + 1e-5).rsqrt()


def make_norm(config):
    if config.normalization == "layer":
        return nn.LayerNorm(config.width)
    if config.normalization == "rms":
        return RMSNorm(config.width)
    return nn.Identity()


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.size = config.heads, config.width // config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.out = nn.Linear(config.width, config.width)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x, inspect=False):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [a.view(b, t, self.heads, self.size).transpose(1, 2) for a in (q, k, v)]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.size)
        future = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
        weights = scores.masked_fill(future, float("-inf")).softmax(dim=-1)
        output = (self.drop(weights) @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.out(output), weights if inspect else None


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1, self.norm2 = make_norm(config), make_norm(config)
        self.residuals = config.residuals
        self.attention = Attention(config)
        self.mlp = nn.Sequential(nn.Linear(config.width, 4 * config.width), nn.GELU(),
                                 nn.Linear(4 * config.width, config.width), nn.Dropout(config.dropout))

    def forward(self, x, inspect=False):
        attended, weights = self.attention(self.norm1(x), inspect)
        x = x + attended if self.residuals else attended
        transformed = self.mlp(self.norm2(x))
        return (x + transformed if self.residuals else transformed), weights


class TinyDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(config.vocab_size, config.width)
        self.positions = nn.Embedding(config.block_size, config.width) if config.positions else None
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = make_norm(config)
        self.head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.head.weight = self.tokens.weight
        self.apply(self.initialize)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids, inspect=False):
        if ids.ndim != 2 or not 0 < ids.shape[1] <= self.config.block_size:
            raise ValueError("Inputs must be [batch, time] within context limit.")
        x = self.tokens(ids)
        if self.positions is not None:
            x = x + self.positions(torch.arange(ids.shape[1], device=ids.device))
        attention = []
        for block in self.blocks:
            x, weights = block(x, inspect)
            if inspect:
                attention.append(weights)
        logits = self.head(self.norm(x))
        # These symbols can condition a prediction, but can never be predicted.
        logits = logits.clone()
        logits[:, :, :2] = torch.finfo(logits.dtype).min
        return (logits, attention) if inspect else logits

    def description(self):
        return {"config": asdict(self.config), "parameters": sum(p.numel() for p in self.parameters()),
                "weights_tied": True, "positions": "learned_absolute" if self.positions is not None else "no_explicit_positions",
                "normalization": self.config.normalization, "residuals": self.config.residuals}


def loss_sum(logits, targets):
    count = int((targets != -100).sum())
    if count == 0:
        raise ValueError("All targets masked; cannot divide by zero.")
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                           ignore_index=-100, reduction="sum")
    return loss, count


def sample_token(logits, temperature=0.8, top_k=20):
    if temperature < 0 or top_k < 1:
        raise ValueError("Temperature >= 0 and top_k >= 1 required.")
    if temperature == 0:
        return int(logits.argmax())
    values, indices = torch.topk(logits / temperature, min(top_k, logits.numel()))
    return int(indices[torch.multinomial(values.softmax(-1), 1)])


@torch.no_grad()
def generate(model, tokenizer, prompt, new_tokens=80, temperature=0.8, top_k=20):
    if not 1 <= new_tokens <= 512:
        raise ValueError("Generation limit is 1–512 tokens.")
    device = next(model.parameters()).device
    model.eval()
    ids = [1] + tokenizer.encode(prompt)
    generated = []
    for _ in range(new_tokens):
        context = torch.tensor([ids[-model.config.block_size:]], device=device)
        token = sample_token(model(context)[0, -1], temperature, top_k)
        if token == 2:
            break
        generated.append(token)
        ids.append(token)
    return tokenizer.decode(generated)
