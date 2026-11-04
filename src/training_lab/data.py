"""Document splits are explicit; neither BPE nor model fitting sees dev/test."""
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

import torch

from .tokenization import PAD


def normal(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def load_corpus(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids, contents = set(), {}
    for row in rows:
        if set(row) != {"id", "split", "text"} or row["split"] not in {"train", "dev", "test"}:
            raise ValueError("Each document needs exactly id, split and text.")
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in ids:
            raise ValueError("Document IDs must be unique nonempty strings.")
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError("Empty/non-string document.")
        ids.add(row["id"])
        fingerprint = hashlib.sha256(normal(row["text"]).encode()).hexdigest()
        if fingerprint in contents:
            raise ValueError(f"Normalized duplicate: {row['id']} and {contents[fingerprint]}")
        contents[fingerprint] = row["id"]
    if Counter(r["split"] for r in rows).keys() != {"train", "dev", "test"}:
        raise ValueError("All three document splits must be nonempty.")
    return rows


def examples(rows, tokenizer, split, block_size):
    result = []
    for row in rows:
        if row["split"] != split:
            continue
        sequence = tokenizer.encode(row["text"], boundaries=True)
        # Each target occurs once; chunks never concatenate different documents.
        for start in range(0, len(sequence) - 1, block_size):
            x = sequence[start:start + block_size]
            y = sequence[start + 1:start + block_size + 1]
            x = x[:len(y)]
            result.append((x, y, row["id"]))
    if not result:
        raise ValueError("Empty example split.")
    return result


def batch(items, device="cpu"):
    width = max(len(x) for x, _, _ in items)
    x = torch.full((len(items), width), PAD, dtype=torch.long)
    y = torch.full_like(x, -100)
    for i, (inputs, targets, _) in enumerate(items):
        x[i, :len(inputs)] = torch.tensor(inputs)
        y[i, :len(targets)] = torch.tensor(targets)
    return x.to(device), y.to(device)


class Bigram:
    """Add-one smoothed baseline, on exactly the Transformer's tokenizer/splits."""
    def __init__(self, vocab_size):
        self.counts = torch.ones(vocab_size, vocab_size, dtype=torch.float64)
        self.counts[:, :2] = 0  # PAD/BOS are inputs, never generation targets.

    def fit(self, rows, tokenizer):
        for row in rows:
            if row["split"] == "train":
                sequence = tokenizer.encode(row["text"], boundaries=True)
                for x, y in zip(sequence, sequence[1:]):
                    self.counts[x, y] += 1
        return self

    def evaluate(self, rows, tokenizer, split):
        logp = (self.counts / self.counts.sum(dim=-1, keepdim=True)).log()
        total, n = 0.0, 0
        for row in rows:
            if row["split"] == split:
                sequence = tokenizer.encode(row["text"], boundaries=True)
                for x, y in zip(sequence, sequence[1:]):
                    total -= logp[x, y].item()
                    n += 1
        if not n:
            raise ValueError("No evaluation tokens.")
        return {"nll": total / n, "tokens": n}
