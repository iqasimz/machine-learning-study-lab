"""Fixed public text, sensor and next-hour forecasting tasks; no implicit downloads."""
from collections import Counter
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .artifacts import read_json, sha256, write_json

SOURCES = {
    "sms_spam": ("https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip", 2_000_000),
    "uci_har": ("https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip", 100_000_000),
    "bike_sharing": ("https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip", 2_000_000),
}


def member(archive, name, limit):
    """Read one exact member into memory, never extract archive-supplied paths."""
    item = archive.getinfo(name)
    if item.is_dir() or item.file_size > limit or item.file_size < 1:
        raise ValueError("Unexpected public dataset member size/type: " + name)
    value = archive.read(item)
    if len(value) != item.file_size: raise ValueError("Truncated dataset member.")
    return value


def word_tokens(text):
    return re.findall(r"\w+|[^\w\s]", text.casefold(), flags=re.UNICODE)


def sms_rows(archive):
    texts, labels, seen = [], [], {}
    for line in member(archive, "SMSSpamCollection", 2_000_000).decode("utf-8-sig").splitlines():
        label, text = line.split("\t", 1)
        if label not in {"ham", "spam"} or not text.strip(): raise ValueError("Invalid SMS record.")
        key = " ".join(text.casefold().split())
        if key in seen:
            if seen[key] != label: raise ValueError("Conflicting labels on identical SMS text.")
            continue
        seen[key] = label
        texts.append(text); labels.append(int(label == "spam"))
    if len(texts) < 20 or len(set(labels)) != 2: raise ValueError("Insufficient SMS classes/examples.")
    return np.asarray(texts), np.asarray(labels, dtype=np.int64)


def har_rows(archive):
    # UCI sometimes distributes a fixed nested zip beside the dataset's papers.
    if "UCI HAR Dataset.zip" in archive.namelist():
        with zipfile.ZipFile(io.BytesIO(member(archive, "UCI HAR Dataset.zip", 90_000_000))) as inner:
            return har_rows(inner)
    arrays, labels, subjects, original = [], [], [], []
    channels = [f"{kind}_{axis}" for kind in ["body_acc", "body_gyro", "total_acc"] for axis in "xyz"]
    for split in ["train", "test"]:
        prefix = "UCI HAR Dataset/" + split + "/"
        signals = [np.loadtxt(io.BytesIO(member(archive, prefix + "Inertial Signals/" + c + "_" + split + ".txt", 20_000_000)), dtype=np.float32, ndmin=2) for c in channels]
        x = np.stack(signals, axis=-1)
        y = np.loadtxt(io.BytesIO(member(archive, prefix + "y_" + split + ".txt", 100_000)), dtype=np.int64, ndmin=1) - 1
        people = np.loadtxt(io.BytesIO(member(archive, prefix + "subject_" + split + ".txt", 100_000)), dtype=np.int64, ndmin=1)
        if x.shape != (len(y), 128, 9) or len(people) != len(y) or not np.isfinite(x).all() or (y < 0).any() or (y > 5).any():
            raise ValueError("Invalid HAR sequence shapes, values or labels.")
        arrays.append(x); labels.append(y); subjects.append(people); original.extend([split] * len(y))
    if set(subjects[0]) & set(subjects[1]): raise ValueError("HAR original splits share participants.")
    return np.concatenate(arrays), np.concatenate(labels), np.concatenate(subjects), np.asarray(original), channels


def bike_rows(archive):
    rows = list(csv.DictReader(io.StringIO(member(archive, "hour.csv", 3_000_000).decode("utf-8-sig"))))
    values = {}
    for row in rows:
        stamp = datetime.strptime(row["dteday"], "%Y-%m-%d") + timedelta(hours=int(row["hr"]))
        count = float(row["cnt"])
        if not 0 <= int(row["hr"]) <= 23 or count < 0 or not np.isfinite(count) or stamp in values:
            raise ValueError("Invalid or duplicate Bike Sharing hour.")
        values[stamp] = count
    times, histories, targets = [], [], []
    for stamp in sorted(values):
        past = [stamp - timedelta(hours=i) for i in range(96, 0, -1)]
        # Fixed target cohort across every lookback; gaps are not silently compressed.
        if all(t in values for t in past):
            times.append(stamp.isoformat()); histories.append([values[t] for t in past]); targets.append(values[stamp])
    if len(times) < 30: raise ValueError("Not enough complete 96-hour histories.")
    return np.asarray(histories, dtype=np.float32), np.asarray(targets, dtype=np.float32), times


def prepare(name, root):
    from .model_data import _directory, _download
    folder = _directory(name, root); url, limit = SOURCES[name]
    file = folder / "source.zip"
    if file.is_symlink(): raise ValueError("Dataset archive symlink is not allowed.")
    if not file.is_file(): _download(url, file, limit)
    if file.stat().st_size > limit: raise ValueError("Dataset archive exceeds size limit.")
    with zipfile.ZipFile(file) as archive:
        {"sms_spam": sms_rows, "uci_har": har_rows, "bike_sharing": bike_rows}[name](archive)
    receipt = {"dataset": name, "sources": [url], "files": {"source.zip": sha256(file)},
               "prepared_at": datetime.now(timezone.utc).isoformat(),
               "verification": "Schema and bounded members checked; SHA256 records the retrieved source. UCI does not supply an upstream checksum here."}
    write_json(folder / "receipt.json", receipt)
    return receipt


def encode_text(texts, train, max_tokens, vocabulary_limit):
    counts = Counter(token for i in train for token in word_tokens(str(texts[i])))
    words = sorted(counts, key=lambda word: (-counts[word], word))[:vocabulary_limit-2]
    vocabulary = {word: i + 2 for i, word in enumerate(words)}
    x = np.zeros((len(texts), max_tokens), dtype=np.int64)
    for i, text in enumerate(texts):
        ids = [vocabulary.get(t, 1) for t in word_tokens(str(text))[:max_tokens]] or [1]
        x[i, :len(ids)] = ids
    return x, vocabulary


def load(name, train_limit, root, controls):
    from .model_data import CATALOG, SPLIT_SEED
    folder = Path(root) / name
    if folder.is_symlink(): raise ValueError("Dataset symlink is not allowed.")
    receipt = read_json(folder / "receipt.json"); file = folder / "source.zip"
    if file.is_symlink() or set(receipt["files"]) != {"source.zip"} or sha256(file) != receipt["files"]["source.zip"]:
        raise ValueError("Prepared sequence dataset integrity check failed.")
    info = dict(CATALOG[name]); extras = {}; groups = None
    with zipfile.ZipFile(file) as archive:
        if name == "sms_spam":
            texts, y = sms_rows(archive); indices = np.arange(len(y))
            train_dev, test = train_test_split(indices, test_size=.2, stratify=y, random_state=SPLIT_SEED)
            train, dev = train_test_split(train_dev, test_size=.25, stratify=y[train_dev], random_state=SPLIT_SEED)
            policy = "Identical normalized messages deduplicated before a fixed stratified 60/20/20 split. Vocabulary fitted on selected training texts only."
            info.update(class_names=["ham", "spam"])
            extras["texts"] = texts
        elif name == "uci_har":
            windows, y, groups, original, channels = har_rows(archive)
            available = np.flatnonzero(original == "train"); test = np.flatnonzero(original == "test")
            a, b = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=SPLIT_SEED).split(available, y[available], groups[available]))
            train, dev = available[a], available[b]
            x = windows.reshape(len(y), -1)
            info.update(sequence_shape=[128, 9], class_names=["walking", "walking upstairs", "walking downstairs", "sitting", "standing", "laying"], channel_names=channels)
            policy = "Original subject-disjoint test retained; development uses separate participants from original training. Overlapping windows from one person never cross splits."
            extras["subjects"] = groups
        else:
            histories, y, times = bike_rows(archive); n = len(y)
            a, b = int(n * .6), int(n * .8)
            train, dev, test = np.arange(a), np.arange(a, b), np.arange(b, n)
            lookback = controls.get("lookback", 24)
            x = histories[:, -lookback:].copy()
            info.update(sequence_shape=[lookback, 1], target_units="rentals in the next hour", target_times=times,
                        feature_names=[f"count {i} hours before target" for i in range(lookback, 0, -1)])
            extras.update(persistence=histories[:, -1].copy(), target_times=times)
            policy = "Chronological 60/20/20 target hours. Every target has 96 complete preceding hours; selected lookback uses only those observed past counts. Rolling one-hour forecasts, not recursive multi-step forecasts. Training cap retains the most recent fitting hours."
    full_train = len(train)
    if full_train > train_limit:
        if name == "bike_sharing": train = train[-train_limit:]
        else:
            train, _ = train_test_split(train, train_size=train_limit, stratify=y[train], random_state=SPLIT_SEED + 1)
    input_spec = {"kind": info["kind"]}
    if name == "sms_spam":
        x, vocabulary = encode_text(extras["texts"], train, controls.get("max_tokens", 128), controls.get("vocab_size", 5000))
        info["sequence_shape"] = [x.shape[1], 1]
        input_spec.update(vocabulary_size=len(vocabulary) + 2)
        extras["vocabulary"] = vocabulary
    input_spec.update(sequence_shape=info["sequence_shape"], forecast=name == "bike_sharing")
    split_ids = {"train": train.tolist(), "dev": dev.tolist(), "test": test.tolist()}
    # Identity is the raw source/cohort, not a tokenizer or lookback representation.
    fingerprint = hashlib.sha256((name + receipt["files"]["source.zip"]).encode()).hexdigest()
    info.update(id=name, rows=len(y), features=x.shape[1], image_shape=None, input_kind=info["kind"], input_spec=input_spec,
                fingerprint=fingerprint, split_fingerprint=hashlib.sha256(json.dumps(split_ids, sort_keys=True).encode()).hexdigest(),
                split_seed=SPLIT_SEED, split_policy=policy, counts={k: len(v) for k, v in split_ids.items()}, available_train=full_train,
                receipt=receipt, warning="Test viewing is exploratory. Subject dependence and temporal dependence require appropriate uncertainty estimates, not independent-row intervals.")
    return {"x": x, "y": y, "groups": groups, "indices": {"train": train, "dev": dev, "test": test},
            "split_ids": split_ids, "info": info, **extras}


def neural_inputs(data):
    """Training-only per-channel scaling; never standardize integer token IDs."""
    from sklearn.preprocessing import StandardScaler
    x = data["x"].copy(); info = data["info"]; train = data["indices"]["train"]
    spec = info.get("input_spec", {"kind": "image" if info["image_shape"] else "table"})
    if spec["kind"] == "text":
        return x, {"kind": "train-only vocabulary; PAD=0, UNK=1; right padding masked", "vocabulary": data["vocabulary"]}, spec
    if spec["kind"] == "sequence":
        shaped = x.reshape(-1, *spec["sequence_shape"])
        values = shaped[train].reshape(-1, shaped.shape[-1])
        mean, scale = values.mean(0), np.maximum(values.std(0), 1e-6)
        x = ((shaped - mean) / scale).reshape(len(x), -1).astype(np.float32)
        return x, {"kind": "training-only per-channel standardization", "mean": mean.tolist(), "scale": scale.tolist()}, spec
    if info["image_shape"]:
        c, h, w = info["image_shape"]
        return x, {"kind": "fixed image range 0–1"}, {"kind": "image", "sequence_shape": [h, c*w]}
    scaler = StandardScaler().fit(x[train])
    return scaler.transform(x).astype(np.float32), {"kind": "training-only standardization", "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}, spec
