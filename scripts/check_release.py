"""Audit a source checkout; optionally refresh its checksums and create a clean ZIP."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", ".venv", "__pycache__", "runs", "model_runs", "model_data", "assets", "exports", "build", "dist", "site-next", "site-check"}


def files():
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(p in EXCLUDED or p.endswith(".egg-info") for p in relative.parts): continue
        if path.is_symlink(): raise ValueError("Symlinks are not allowed: "+str(relative))
        if not path.is_file() or relative.name in {".DS_Store", "RELEASE_MANIFEST.json"}: continue
        if relative.name.startswith(".env") or path.suffix in {".pt", ".pth", ".safetensors", ".joblib", ".pdf", ".docx", ".parquet", ".zip", ".log", ".pyc"}:
            raise ValueError("Unexpected private/generated artifact: "+str(relative))
        if path.stat().st_size > 20_000_000: raise ValueError("Unexpected large file: "+str(relative))
        if path.suffix in {".py", ".js", ".cjs", ".json", ".md", ".toml", ".yml", ".html"}:
            text = path.read_text()
            if re.search(r"/(?:Users|home)/[A-Za-z0-9._-]+/|sk-(?:proj-)?[A-Za-z0-9_-]{30,}|ghp_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}", text):
                raise ValueError("Potential personal path or credential: "+str(relative))
        yield relative, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--zip", type=Path)
    args = parser.parse_args()
    selected = list(files())
    hashes = {str(relative): hashlib.sha256(path.read_bytes()).hexdigest() for relative, path in selected}
    manifest = ROOT / "RELEASE_MANIFEST.json"
    if args.write_manifest:
        manifest.write_text(json.dumps({"schema": 1, "files": hashes,
            "excluded": sorted(EXCLUDED), "note": "Checksums detect changes, not malicious replacement of both files and manifest."}, indent=2)+"\n")
    else:
        expected = json.loads(manifest.read_text())["files"]
        if expected != hashes: raise ValueError("Release files changed; review them before explicitly refreshing the manifest.")
    if args.zip:
        target = args.zip.resolve()
        if target.exists(): raise ValueError("Archive already exists; choose a new filename.")
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            for relative, path in selected:
                archive.write(path, str(Path(ROOT.name) / relative))
            archive.write(manifest, str(Path(ROOT.name) / manifest.name))
    print(json.dumps({"status": "PASS", "files": len(hashes), "manifest_matches": True, "bytes": sum(p.stat().st_size for _,p in selected)}, indent=2))


if __name__ == "__main__": main()
