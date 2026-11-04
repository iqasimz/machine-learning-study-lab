"""A small byte-level BPE implementation. Lossless UTF-8; no hidden downloads."""
from collections import Counter

PAD, BOS, EOS = 0, 1, 2


class ByteBPE:
    def __init__(self, merges=()):
        self.merges = [tuple(pair) for pair in merges]
        self.pieces = [b"", b"", b""] + [bytes([i]) for i in range(256)]
        for left, right in self.merges:
            if not (3 <= left < len(self.pieces) and 3 <= right < len(self.pieces)):
                raise ValueError("Malformed BPE merge.")
            self.pieces.append(self.pieces[left] + self.pieces[right])

    @property
    def vocab_size(self):
        return len(self.pieces)

    @staticmethod
    def replace(sequence, pair, new_id):
        result, i = [], 0
        while i < len(sequence):
            if i + 1 < len(sequence) and (sequence[i], sequence[i + 1]) == pair:
                result.append(new_id)
                i += 2
            else:
                result.append(sequence[i])
                i += 1
        return result

    @classmethod
    def fit(cls, documents, count=32):
        if not documents or not any(documents) or not 0 <= count <= 256:
            raise ValueError("Need training text and 0–256 merges.")
        sequences = [[b + 3 for b in text.encode("utf-8")] for text in documents]
        merges = []
        for _ in range(count):
            counts = Counter(pair for sequence in sequences for pair in zip(sequence, sequence[1:]))
            if not counts:
                break
            # Stable tie-breaking, and no pairs spanning documents.
            pair = min(counts, key=lambda p: (-counts[p], p))
            if counts[pair] < 2:
                break
            sequences = [cls.replace(s, pair, 259 + len(merges)) for s in sequences]
            merges.append(pair)
        return cls(merges)

    def encode(self, text, boundaries=False):
        sequence = [b + 3 for b in text.encode("utf-8")]
        for i, pair in enumerate(self.merges):
            sequence = self.replace(sequence, pair, 259 + i)
        return [BOS] + sequence + [EOS] if boundaries else sequence

    def decode(self, ids):
        if any(not 0 <= i < self.vocab_size for i in ids):
            raise ValueError("Token outside vocabulary.")
        return b"".join(self.pieces[i] for i in ids).decode("utf-8", errors="replace")

    def label(self, token):
        if token < 3:
            return ["<PAD>", "<BOS>", "<EOS>"][token]
        piece = self.pieces[token]
        try:
            return piece.decode("utf-8").replace("\n", "↵")
        except UnicodeDecodeError:
            return "0x" + piece.hex()

    def to_dict(self):
        return {"type": "byte_bpe_v1", "merges": [list(pair) for pair in self.merges], "special_ids": {"pad": PAD, "bos": BOS, "eos": EOS}}

    @classmethod
    def from_dict(cls, data):
        if data.get("type") != "byte_bpe_v1":
            raise ValueError("Unsupported tokenizer format.")
        return cls(data["merges"])
