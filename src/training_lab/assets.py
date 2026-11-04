"""Explicit, pinned public downloads. Ordinary app use never downloads a model."""
import base64
import hashlib
import io
import json
from pathlib import Path
import tarfile
from urllib.request import urlopen

from .artifacts import ROOT, read_json, sha256, write_json

ASSETS = ROOT / "assets"
MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
MODEL_REVISION = "a10cc1512eabd3dde888204e902eca88bddb4951"
MODEL_DIR = ASSETS / "models" / "smollm2-360m-instruct"
DATA_ID = "HuggingFaceTB/everyday-conversations-llama3.1-2k"
DATA_REVISION = "14f543216b9ba42b6b951dc5bd199460d193b162"
DATA_DIR = ASSETS / "datasets" / "everyday-conversations"
DATA_FILES = {
    "train_sft-00000-of-00001.parquet": "4750214d12b4b92d07fca953e0b5cc59e313c575282d08b605c639aa87531daa",
    "test_sft-00000-of-00001.parquet": "9a6dd47c31ce940f36ced9eba2a733a0a0035ef514158c68739073e043382adc",
}
VENDOR = ROOT / "src" / "training_lab" / "web" / "vendor" / "katex"
KATEX_VERSION = "0.19.1"


def status():
    model = (MODEL_DIR / "model.safetensors").is_file() and (MODEL_DIR / "tokenizer.json").is_file() and (ASSETS / "receipts" / "model.json").is_file()
    data = all((DATA_DIR / "data" / name).is_file() for name in DATA_FILES) and (ASSETS / "receipts" / "dataset.json").is_file()
    return {"model_ready": model, "data_ready": data, "ready": model and data,
            "model": MODEL_ID, "revision": MODEL_REVISION, "parameters": 361821120,
            "dataset": DATA_ID, "dataset_revision": DATA_REVISION,
            "license": "Apache-2.0", "math_ready": (VENDOR / "katex.min.js").is_file(),
            "note": "The model already knows language; local LoRA training adapts a small set of weights. The synthetic conversation data may have been seen during original model training."}


def prepare_model():
    from huggingface_hub import hf_hub_download
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    names = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "model.safetensors", "README.md"]
    for name in names:
        print("Preparing model file:", name, flush=True)
        hf_hub_download(MODEL_ID, name, revision=MODEL_REVISION, local_dir=MODEL_DIR, token=False)
    write_json(ASSETS / "receipts" / "model.json", {"repo": MODEL_ID, "revision": MODEL_REVISION,
        "license": "Apache-2.0", "files": {n:sha256(MODEL_DIR/n) for n in names}})
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, expected in DATA_FILES.items():
        print("Preparing conversation data:", name, flush=True)
        path = Path(hf_hub_download(DATA_ID, "data/"+name, repo_type="dataset", revision=DATA_REVISION, local_dir=DATA_DIR, token=False))
        if sha256(path) != expected: raise ValueError("Dataset checksum differs from the pinned source.")
        # Keep the upstream data/ layout; reader uses this same validated location.
    write_json(ASSETS / "receipts" / "dataset.json", {"repo": DATA_ID, "revision": DATA_REVISION,
        "license": "Apache-2.0", "synthetic": True, "source_model": "Llama-3.1-70B-Instruct",
        "known_pretraining_overlap": "Part of SmolTalk; may already be seen by SmolLM2. Local holdout is not a new generalization benchmark.",
        "files":DATA_FILES})
    return status()


def prepare_math():
    # Vendor a verified npm release; execute no package scripts and extract only selected files.
    with urlopen("https://registry.npmjs.org/katex/"+KATEX_VERSION, timeout=30) as response:
        metadata=json.load(response)
    archive_url=metadata["dist"]["tarball"]
    if not archive_url.startswith("https://registry.npmjs.org/katex/-/"): raise ValueError("Unexpected asset host.")
    with urlopen(archive_url, timeout=60) as response: payload=response.read(12_000_001)
    if len(payload)>12_000_000: raise ValueError("Math archive exceeds the download ceiling.")
    expected=metadata["dist"]["integrity"]
    digest="sha512-"+base64.b64encode(hashlib.sha512(payload).digest()).decode()
    if digest!=expected: raise ValueError("Math archive integrity check failed.")
    VENDOR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            name=member.name
            selected=name in {"package/dist/katex.min.js","package/dist/katex.min.css","package/dist/contrib/auto-render.min.js","package/LICENSE"} or name.startswith("package/dist/fonts/")
            if not selected or not member.isfile(): continue
            relative=Path("LICENSE") if name=="package/LICENSE" else Path(name.removeprefix("package/dist/"))
            if ".." in relative.parts or relative.is_absolute(): raise ValueError("Unsafe archive path.")
            target=VENDOR/relative; target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())
    write_json(VENDOR/"release.json", {"package":"katex","version":KATEX_VERSION,"integrity":digest,"license":"MIT","source":archive_url})
    return {"math_ready":True,"version":KATEX_VERSION}
