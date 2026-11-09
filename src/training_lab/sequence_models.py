"""Small sequence encoders and a patch-based vision Transformer for local study."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

SEQUENCE_MODELS = {"rnn", "birnn", "lstm", "bilstm", "gru", "bigru", "cnn1d", "tcn", "embedding_mean", "transformer_encoder"}


def probe_training_device(model, example, device):
    """Check real training operations without changing weights or random streams.

    cuDNN recurrent backward requires a training-mode forward, even for a probe.
    These small encoder families have no running BatchNorm statistics to mutate.
    """
    mode, cpu_state = model.training, torch.get_rng_state()
    backend = torch.cuda if device == "cuda" else torch.mps if device == "mps" else None
    device_state = backend.get_rng_state() if backend is not None else None
    try:
        model.train()
        model(example.to(device)).sum().backward()
        if backend is not None: backend.synchronize()
    finally:
        model.zero_grad(set_to_none=True)
        model.train(mode)
        torch.set_rng_state(cpu_state)
        if backend is not None: backend.set_rng_state(device_state)


class EncoderBlock(nn.Module):
    """Pre-norm bidirectional self-attention, then a position-wise MLP.

    Mask keys before softmax and clear padded queries after the residual path.
    Explicit operations keep the teaching code and CPU/MPS execution aligned.
    """
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.projection = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.drop = nn.Dropout(dropout)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * width, width))

    def forward(self, x, valid):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q @ k.transpose(-2, -1)) / (width // self.heads) ** .5
        weights = scores.masked_fill(~valid[:, None, None, :], float("-inf")).softmax(-1)
        attended = (self.drop(weights) @ v).transpose(1, 2).reshape(batch, length, width)
        x = x + self.drop(self.projection(attended))
        return (x + self.drop(self.ff(self.norm2(x)))) * valid.unsqueeze(-1)


class TemporalBlock(nn.Module):
    """Two causal, dilated convolutions plus an identity shortcut."""
    def __init__(self, width, kernel, dilation, dropout):
        super().__init__()
        self.padding = (kernel - 1) * dilation
        self.convs = nn.ModuleList([nn.Conv1d(width, width, kernel, dilation=dilation) for _ in range(2)])
        self.drop = nn.Dropout(dropout)

    def forward(self, x, valid):
        original = x
        mask = valid[:, None, :]
        for conv in self.convs:
            x = self.drop(torch.relu(conv(nn.functional.pad(x, (self.padding, 0))))) * mask
        return torch.relu(x + original) * mask


class SequencePredictor(nn.Module):
    """Turn tokens, sensor steps or image rows into one class/number per example."""
    def __init__(self, name, p, outputs, spec, image_shape=None):
        super().__init__()
        self.name, self.spec, self.image_shape = name, spec, image_shape
        width, layers, dropout = p.get("width", 64), p.get("layers", 1), p["dropout"]
        self.text = spec["kind"] == "text"
        if self.text:
            inputs = p["embedding_dim"]
            self.embedding = nn.Embedding(spec["vocabulary_size"], inputs, padding_idx=0)
            max_length = spec["sequence_shape"][0]
        else:
            max_length, inputs = spec["sequence_shape"]
        self.drop = nn.Dropout(dropout)
        self.bidirectional = name.startswith("bi")
        if name in {"rnn", "birnn", "lstm", "bilstm", "gru", "bigru"}:
            cell = nn.LSTM if "lstm" in name else nn.GRU if "gru" in name else nn.RNN
            self.encoder = cell(inputs, width, num_layers=layers, batch_first=True,
                                bidirectional=self.bidirectional, dropout=dropout if layers > 1 else 0)
            hidden = width * (2 if self.bidirectional else 1)
        elif name == "embedding_mean":
            hidden = inputs
        else:
            self.input_projection = nn.Linear(inputs, width)
            hidden = width
            if name == "transformer_encoder":
                self.position = nn.Parameter(torch.zeros(1, max_length, width)) if p["positions"] else None
                if self.position is not None: nn.init.normal_(self.position, std=.02)
                self.blocks = nn.ModuleList([EncoderBlock(width, p["heads"], dropout) for _ in range(layers)])
            elif name == "tcn":
                self.blocks = nn.ModuleList([TemporalBlock(width, p["kernel_size"], 2**i, dropout) for i in range(layers)])
            else:
                self.blocks = nn.ModuleList([nn.Conv1d(width, width, p["kernel_size"], padding=p["kernel_size"]//2) for _ in range(layers)])
        self.head = nn.Linear(hidden, outputs)

    def forward(self, flat):
        if self.text:
            valid = flat != 0
            lengths = valid.sum(1).clamp_min(1)
            x = self.embedding(flat.long())
        else:
            if self.image_shape:
                # One time step is one horizontal image row, across all channels.
                x = flat.reshape(-1, *self.image_shape).permute(0, 2, 1, 3).flatten(2)
            else:
                x = flat.reshape(-1, *self.spec["sequence_shape"])
            valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
            lengths = valid.sum(1)
        if self.name in {"rnn", "birnn", "lstm", "bilstm", "gru", "bigru"}:
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, state = self.encoder(packed)
            hidden = state[0] if isinstance(state, tuple) else state
            # Last layer's forward AND backward final states, not last output row.
            pooled = torch.cat([hidden[-2], hidden[-1]], dim=-1) if self.bidirectional else hidden[-1]
        elif self.name == "embedding_mean":
            pooled = (x * valid.unsqueeze(-1)).sum(1) / lengths.unsqueeze(-1)
        else:
            x = self.input_projection(x) * valid.unsqueeze(-1)
            if self.name == "transformer_encoder":
                if self.position is not None: x = (x + self.position[:, :x.shape[1]]) * valid.unsqueeze(-1)
                for block in self.blocks: x = block(x, valid)
            else:
                x = x.transpose(1, 2)
                for block in self.blocks:
                    x = block(x, valid) if self.name == "tcn" else self.drop(torch.relu(block(x))) * valid[:, None, :]
                x = x.transpose(1, 2)
            # Regression reads the final history state; classification pools valid steps.
            pooled = x[torch.arange(len(x), device=x.device), lengths - 1] if self.spec.get("forecast") else (x * valid.unsqueeze(-1)).sum(1) / lengths.unsqueeze(-1)
        return self.head(self.drop(pooled))


class TinyViT(nn.Module):
    """Train a small mean-pooled ViT from scratch, not a pretrained foundation model."""
    def __init__(self, p, outputs, image_shape):
        super().__init__()
        channels, height, width = image_shape
        patch, hidden = p["patch_size"], p["width"]
        if height % patch or width % patch: raise ValueError("Patch size must divide both image dimensions.")
        self.image_shape = image_shape
        self.patches = nn.Conv2d(channels, hidden, patch, stride=patch)
        self.position = nn.Parameter(torch.zeros(1, (height//patch)*(width//patch), hidden)) if p["positions"] else None
        if self.position is not None: nn.init.normal_(self.position, std=.02)
        self.blocks = nn.ModuleList([EncoderBlock(hidden, p["heads"], p["dropout"]) for _ in range(p["layers"])])
        self.norm, self.head = nn.LayerNorm(hidden), nn.Linear(hidden, outputs)

    def forward(self, flat):
        x = self.patches(flat.reshape(-1, *self.image_shape)).flatten(2).transpose(1, 2)
        if self.position is not None: x = x + self.position
        valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        for block in self.blocks: x = block(x, valid)
        return self.head(self.norm(x).mean(1))
