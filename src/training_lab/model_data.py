"""Bounded, attributed datasets for the ML/DL lab. No user-supplied URLs or paths."""
from datetime import datetime, timezone
import gzip
import hashlib
import io
from pathlib import Path
import struct
from urllib.request import urlopen
import zipfile

import numpy as np
from sklearn import datasets
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .artifacts import ROOT, read_json, sha256, write_json

DATA = ROOT / "model_data"
SPLIT_SEED = 20261102
FASHION_BASE = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/"
FASHION_FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
CATALOG = {
    "iris": {"name": "Iris", "task": "classification", "source": "https://archive.ics.uci.edu/dataset/53/iris", "license": "CC BY 4.0 · R. A. Fisher / UCI", "note": "150 flower measurements; a small teaching dataset, not a deployment benchmark."},
    "wine": {"name": "Wine cultivars", "task": "classification", "source": "https://archive.ics.uci.edu/dataset/109/wine", "license": "CC BY 4.0 · S. Aeberhard, M. Forina / UCI", "note": "178 chemistry rows and three cultivars. This is not Wine Quality."},
    "digits": {"name": "Handwritten digits", "task": "classification", "source": "https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits", "license": "CC BY 4.0 · E. Alpaydin, C. Kaynak / UCI", "note": "1,797 bundled 8×8 images. New stratified splits are not guaranteed writer-disjoint."},
    "wine_red": {"name": "Wine Quality · red", "task": "regression", "source": "https://archive.ics.uci.edu/dataset/186/wine+quality", "license": "CC BY 4.0 · P. Cortez et al. / UCI (2009)", "note": "1,599 rows, 11 features and an ordered quality rating. Exact duplicate feature rows stay in one split."},
    "wine_white": {"name": "Wine Quality · white", "task": "regression", "source": "https://archive.ics.uci.edu/dataset/186/wine+quality", "license": "CC BY 4.0 · P. Cortez et al. / UCI (2009)", "note": "4,898 rows; separate from red wine. Regression error is in quality-rating units."},
    "fashion_mnist": {"name": "Fashion-MNIST", "task": "classification", "source": "https://github.com/zalandoresearch/fashion-mnist", "license": "MIT · Zalando SE (2017), Han Xiao, Kashif Rasul, Roland Vollgraf", "note": "60,000 train / 10,000 official test 28×28 clothing images. Development is drawn only from the original training set."},
    "curve": {"name": "Noisy nonlinear regression", "task": "regression", "source": None, "license": "Original teaching fixture · CC0-1.0", "note": "2,400 generated rows: y = 2 sin(x₀) + 0.4 x₁² − x₂ + Gaussian noise. Known synthetic mechanism, not real-world evidence."},
    "clusters": {"name": "Unequal clusters and noise", "task": "clustering", "source": None, "license": "Original teaching fixture · CC0-1.0", "note": "1,200 generated 2D points in three Gaussian groups plus uniform noise. Reference labels are diagnostic only, never used for fitting."},
}


def _directory(name, root=DATA):
    if name not in {"fashion_mnist", "wine_quality"}:
        raise ValueError("Only Fashion-MNIST and Wine Quality need preparation.")
    path = Path(root) / name
    if path.is_symlink():
        raise ValueError("Dataset symlink is not allowed.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download(url, destination, limit):
    if destination.is_symlink():
        raise ValueError("Dataset file symlink is not allowed.")
    temp = destination.with_suffix(destination.suffix + ".download")
    if temp.is_symlink():
        raise ValueError("Temporary dataset symlink is not allowed.")
    try:
        with urlopen(url, timeout=45) as response, temp.open("wb") as output:
            if not response.url.startswith("https://"):
                raise ValueError("An HTTPS dataset source is required.")
            size = 0
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise ValueError("Dataset exceeds the download limit.")
                output.write(chunk)
        temp.replace(destination)
    finally:
        if temp.exists():
            temp.unlink()


def prepare(name, root=DATA):
    path = _directory(name, root)
    files, sources = {}, []
    if name == "fashion_mnist":
        for filename, expected in FASHION_FILES.items():
            file = path / filename
            if file.is_symlink():
                raise ValueError("Dataset file symlink is not allowed.")
            if not file.is_file() or hashlib.md5(file.read_bytes()).hexdigest() != expected:
                _download(FASHION_BASE + "data/fashion/" + filename, file, 32_000_000)
            if hashlib.md5(file.read_bytes()).hexdigest() != expected:
                raise ValueError("Fashion-MNIST source checksum mismatch: " + filename)
            files[filename] = sha256(file)
            sources.append(FASHION_BASE + "data/fashion/" + filename)
        license_file = path / "LICENSE"
        _download(FASHION_BASE + "LICENSE", license_file, 20_000)
        files["LICENSE"] = sha256(license_file)
    else:
        file = path / "wine_quality.zip"
        if file.is_symlink():raise ValueError("Dataset archive symlink is not allowed.")
        url = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
        if not file.is_file():
            _download(url, file, 2_000_000)
        # Read two fixed members in memory. Never extract arbitrary archive paths.
        with zipfile.ZipFile(file) as archive:
            for variant, count in [("red", 1599), ("white", 4898)]:
                member = archive.getinfo(f"winequality-{variant}.csv")
                if member.file_size > 1_500_000:
                    raise ValueError("Unexpected CSV size.")
                raw = archive.read(member)
                array = np.loadtxt(io.BytesIO(raw), delimiter=";", skiprows=1)
                if array.shape != (count, 12) or not np.isfinite(array).all():
                    raise ValueError("Unexpected Wine Quality content.")
        files[file.name] = sha256(file)
        sources = [url]
    receipt = {"dataset": name, "prepared_at": datetime.now(timezone.utc).isoformat(), "files": files, "sources": sources,
               "license": CATALOG["fashion_mnist" if name == "fashion_mnist" else "wine_red"]["license"]}
    write_json(path / "receipt.json", receipt)
    return receipt


def ready(name, root=DATA):
    prepared = "wine_quality" if name.startswith("wine_") and name != "wine" else name
    if prepared not in {"wine_quality", "fashion_mnist"}:
        return True
    path = Path(root) / prepared
    if path.is_symlink():
        return False
    try:
        receipt = read_json(path / "receipt.json")
        expected = set(FASHION_FILES) | {"LICENSE"} if prepared == "fashion_mnist" else {"wine_quality.zip"}
        return set(receipt["files"]) == expected and all(not (path / f).is_symlink() and (path / f).is_file() for f in expected)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def catalog():
    return [{"id": key, **value, "ready": ready(key),
             "prepare_id": "wine_quality" if key in {"wine_red", "wine_white"} else key}
            for key, value in CATALOG.items()]


def _idx(path, images):
    with gzip.open(path, "rb") as stream:
        raw = stream.read(50_000_001)
    if len(raw) > 50_000_000:
        raise ValueError("Uncompressed image data exceeds the limit.")
    magic, n = struct.unpack(">II", raw[:8])
    if images:
        rows, cols = struct.unpack(">II", raw[8:16])
        if magic != 2051 or (rows, cols) != (28, 28) or n not in {60000, 10000} or len(raw) != 16 + n * 784:
            raise ValueError("Unexpected IDX image data.")
        return np.frombuffer(raw[16:], dtype=np.uint8).reshape(n, 784).astype(np.float32) / 255
    if magic != 2049 or n not in {60000, 10000} or len(raw) != 8 + n:
        raise ValueError("Unexpected IDX label data.")
    return np.frombuffer(raw[8:], dtype=np.uint8).astype(np.int64)


def load(name, train_limit=5000, root=DATA):
    if name not in CATALOG or type(train_limit) is not int or train_limit not in {500, 1500, 5000, 15000, 60000}:
        raise ValueError("Unknown dataset or training-row limit.")
    info = dict(CATALOG[name]); shape = None; groups = None; official = False
    if not ready(name, root):
        raise ValueError("Prepare this public dataset before training.")
    if name in {"fashion_mnist", "wine_red", "wine_white"}:
        folder = Path(root) / ("fashion_mnist" if name == "fashion_mnist" else "wine_quality")
        receipt = read_json(folder / "receipt.json")
        for filename, expected in receipt["files"].items():
            file = folder / filename
            if Path(filename).name != filename or file.is_symlink() or sha256(file) != expected:
                raise ValueError("Prepared dataset integrity check failed.")
        info["receipt"] = receipt
    if name in {"iris", "wine", "digits"}:
        item = getattr(datasets, "load_" + name)()
        x, y = item.data.astype(np.float32), item.target.astype(np.int64)
        info.update(feature_names=list(item.feature_names), class_names=[str(t) for t in item.target_names])
        if name == "digits":
            x = x / 16; shape = [1, 8, 8]
    elif name.startswith("wine_"):
        with zipfile.ZipFile(folder / "wine_quality.zip") as archive:
            raw = archive.read("winequality-" + name.removeprefix("wine_") + ".csv")
        values = np.loadtxt(io.BytesIO(raw), delimiter=";", skiprows=1).astype(np.float32)
        x, y = values[:, :-1], values[:, -1]
        _, groups = np.unique(x, axis=0, return_inverse=True)
        info["feature_names"] = raw.decode().splitlines()[0].replace('"', '').split(";")[:-1]
    elif name == "fashion_mnist":
        x = np.concatenate([_idx(folder / "train-images-idx3-ubyte.gz", True), _idx(folder / "t10k-images-idx3-ubyte.gz", True)])
        y = np.concatenate([_idx(folder / "train-labels-idx1-ubyte.gz", False), _idx(folder / "t10k-labels-idx1-ubyte.gz", False)])
        shape = [1, 28, 28]; official = True
        info["class_names"] = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    else:
        rng = np.random.default_rng(SPLIT_SEED)
        if name == "curve":
            x = rng.uniform(-3, 3, (2400, 6)).astype(np.float32)
            y = (2 * np.sin(x[:, 0]) + .4 * x[:, 1]**2 - x[:, 2] + rng.normal(0, .4, 2400)).astype(np.float32)
        else:
            centers = np.array([[-2, -1], [2, 1], [0, 3]])
            y = np.concatenate([np.repeat(np.arange(3), [400, 350, 330]), np.repeat(-1, 120)])
            x = np.concatenate([rng.normal(centers[k], [.5, .65, .4][k], (n, 2)) for k, n in enumerate([400, 350, 330])] + [rng.uniform(-5, 5, (120, 2))]).astype(np.float32)
        info["feature_names"] = ["x" + str(i) for i in range(x.shape[1])]
    indices = np.arange(len(x))
    if official:
        train, dev = train_test_split(indices[:60000], test_size=10000, random_state=SPLIT_SEED, stratify=y[:60000])
        test = indices[60000:]
        policy = "50,000 fitting / 10,000 development from original train; 10,000 original test untouched by fitting."
    elif groups is not None:
        train_dev, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SPLIT_SEED).split(x, y, groups))
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=SPLIT_SEED).split(x[train_dev], y[train_dev], groups[train_dev]))
        train, dev = train_dev[a], train_dev[b]
        policy = "Approximately 60/20/20; exact identical feature rows grouped, including differing ratings."
    else:
        stratify = y if info["task"] == "classification" else None
        train_dev, test = train_test_split(indices, test_size=.2, random_state=SPLIT_SEED, stratify=stratify)
        train, dev = train_test_split(train_dev, test_size=.25, random_state=SPLIT_SEED, stratify=y[train_dev] if stratify is not None else None)
        policy = "Fixed 60/20/20 split; stratified only for labelled classification."
    full_train = len(train)
    if full_train > train_limit:
        train, _ = train_test_split(train, train_size=train_limit, random_state=SPLIT_SEED + 1,
                                    stratify=y[train] if info["task"] == "classification" else None)
    split_ids = {"train": train.tolist(), "dev": dev.tolist(), "test": test.tolist()}
    fingerprint = hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()
    import json
    split_hash = hashlib.sha256(json.dumps(split_ids, sort_keys=True).encode()).hexdigest()
    info.update(id=name, rows=len(x), features=x.shape[1], image_shape=shape, fingerprint=fingerprint,
                split_fingerprint=split_hash, split_seed=SPLIT_SEED, split_policy=policy,
                counts={k: len(v) for k, v in split_ids.items()}, available_train=full_train,
                warning="Repeatedly viewing test scores makes this an exploratory lab, not a fresh confirmatory benchmark.")
    return {"x": x, "y": y, "groups":groups, "indices": {"train": train, "dev": dev, "test": test}, "info": info, "split_ids": split_ids}
