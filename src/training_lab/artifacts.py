"""JSON artifacts and integrity checks. No credentials or external services."""
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$")


def run_path(name, root=RUNS):
    if not NAME.fullmatch(name):
        raise ValueError("Run ID must contain only letters, digits, hyphens or underscores.")
    path = Path(root) / name
    if path.is_symlink():
        raise ValueError("Symlink run directories are not allowed.")
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seal(directory):
    directory = Path(directory)
    hashes = {p.name: sha256(p) for p in sorted(directory.iterdir())
              if p.is_file() and p.name not in {"hashes.json", "worker.log"} and not p.name.endswith(".tmp")}
    write_json(directory / "hashes.json", hashes)


def verify(directory):
    directory = Path(directory)
    hashes = read_json(directory / "hashes.json")
    if not hashes:
        raise ValueError("Empty integrity manifest.")
    for name, expected in hashes.items():
        if Path(name).name != name:
            raise ValueError("Invalid artifact name in hash manifest.")
        path = directory / name
        if path.is_symlink() or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Artifact missing or changed: {name}")
    if "adapter_files.json" in hashes:
        for name,expected in read_json(directory/"adapter_files.json").items():
            relative=Path(name)
            if len(relative.parts)!=2 or relative.parts[0] not in {"adapter","adapter-last"}:
                raise ValueError("Unexpected adapter artifact path.")
            path=directory/relative
            if path.is_symlink() or not path.is_file() or sha256(path)!=expected:
                raise ValueError("Adapter artifact changed: "+name)
    return hashes


def source_hashes():
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "src").rglob("*"))
            if p.is_file() and p.suffix in {".py", ".js", ".html", ".css"}}
