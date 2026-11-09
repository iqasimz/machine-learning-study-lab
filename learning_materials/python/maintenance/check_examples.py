"""Check the authored source pack; never used by the website or a learner runner.

Only explicitly marked, locally authored non-training snippets are executed.
Each runs in a separate temporary working directory. No model is downloaded.
Use --record to write derived evidence/catalog records and generated plot assets.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ["01_PYTHON_BASICS.md", "02_DATA_AND_PLOTTING.md", "03_MACHINE_LEARNING.md",
         "04_DEEP_LEARNING.md", "05_NLP.md", "06_LLMS.md", "07_OTHER_LIBRARIES.md",
         "08_DSA.md"]
EXPECTED = {"PY": 14, "DS": 10, "SK": 12, "PT": 12, "NC": 8, "LC": 10, "OL": 9, "DA": 20}
PATTERN = re.compile(r"<!-- example: ([A-Z]+\d+-[a-z0-9-]+) -->\n```python\n(.*?)\n```\n\n\*\*Output\*\*\n```text\n(.*?)\n```", re.S)
FORBIDDEN_CALLS = {"fit", "fit_transform", "partial_fit", "step", "train", "from_pretrained", "load_dataset", "download", "urlopen"}
WALKTHROUGHS = {"SK12", "PT05", "LC03", "LC04"}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def portable_diagnostics(text):
    """Keep the warning/error, not the author's account name or cache paths."""
    text = re.sub(r"/Users/[^/\s]+", "<user-directory>", text)
    text = re.sub(r"/private/tmp/python-course-check-[^/\s]+", "<temporary-directory>", text)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--model-dir", type=Path,
                        default=ROOT.parents[1] / "assets/models/smollm2-360m-instruct")
    args = parser.parse_args()
    metadata = json.loads((ROOT / "metadata.json").read_text())
    cases = []; lessons = []; failures = []; outputs = {}
    for name in FILES:
        path = ROOT / name
        if not path.is_file():
            failures.append(f"Missing chapter: {name}"); continue
        text = path.read_text(encoding="utf-8")
        headings = list(re.finditer(r"^## ([A-Z]+\d+)\. (.+)$", text, re.M))
        for i, heading in enumerate(headings):
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            body = text[heading.end():end]
            identifier = heading[1]
            examples = list(PATTERN.finditer(body))
            if body.count("<!-- example:") != len(examples):
                failures.append(identifier + ": malformed or unmatched example/output pair")
            sources = re.findall(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", body)
            if not sources: failures.append(identifier + ": no primary/book source")
            if not examples and identifier not in WALKTHROUGHS:
                failures.append(identifier + ": missing checked-output example")
            for block in re.findall(r"```python walkthrough\n(.*?)\n```", body, re.S):
                try: ast.parse(block)
                except SyntaxError as error: failures.append(identifier + ": invalid walkthrough: " + str(error))
            lessons.append({"id": identifier, "title": heading[2], "chapter": name,
                            "source_sha256": sha(body), "sources": sources,
                            "examples": [e[1] for e in examples],
                            "has_training_walkthrough": "```python walkthrough" in body,
                            "website_integrated": metadata["status"].startswith("integrated_"),
                            **metadata["lessons"].get(identifier, {})})
            for match in examples:
                case_id, code, expected = match.groups()
                if not case_id.startswith(identifier + "-"):
                    failures.append(case_id + ": wrong owning lesson")
                tree = ast.parse(code)
                forbidden = []
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                        continue
                    if node.func.attr not in FORBIDDEN_CALLS:
                        continue
                    keywords = {k.arg: k.value.value for k in node.keywords if isinstance(k.value, ast.Constant)}
                    if (node.func.attr == "from_pretrained" and keywords.get("local_files_only") is True
                            and keywords.get("trust_remote_code") is False):
                        continue
                    forbidden.append(node.func.attr)
                if forbidden:
                    failures.append(case_id + ": training/download call: " + str(forbidden)); continue
                record = {"id": case_id, "lesson": identifier, "code_sha256": sha(code),
                          "expected_sha256": sha(expected),
                          "kind": "captured_pretrained_inference" if case_id == "LC01-pretrained" else "checked_non_training"}
                env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg",
                       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                       "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}
                env["STUDY_LAB_MODEL_DIR"] = str(args.model_dir.resolve())
                try:
                    with tempfile.TemporaryDirectory(prefix="python-course-check-") as folder:
                        env["MPLCONFIGDIR"] = str(Path(folder) / "mpl-config")
                        cache = Path(folder) / "cache"
                        cache.mkdir()
                        env["XDG_CACHE_HOME"] = str(cache)
                        result = subprocess.run([sys.executable, "-c", code], cwd=folder, env=env,
                                                capture_output=True, text=True, timeout=90)
                        actual = result.stdout.rstrip()
                        record.update(returncode=result.returncode, stderr=portable_diagnostics(result.stderr[-2500:]),
                                      actual_sha256=sha(actual), passed=result.returncode == 0 and actual == expected.rstrip())
                        outputs[case_id] = actual
                        if args.record and record["passed"]:
                            for image in Path(folder).glob("*.png"):
                                destination = ROOT / "outputs" / (case_id + "-" + image.name)
                                destination.parent.mkdir(exist_ok=True)
                                shutil.copy2(image, destination)
                                record.setdefault("plots", []).append({"file":str(destination.relative_to(ROOT)),
                                                                      "sha256":hashlib.sha256(destination.read_bytes()).hexdigest()})
                        if not record["passed"]:
                            failures.append(case_id + ": expected " + repr(expected) + "; got " + repr(actual) + "; stderr=" + portable_diagnostics(result.stderr[-800:]))
                except subprocess.TimeoutExpired:
                    record.update(passed=False, timeout=True); failures.append(case_id + ": timed out")
                cases.append(record)
                print(case_id, "PASS" if record.get("passed") else "FAIL", flush=True)
    counts = Counter(re.sub(r"\d+$", "", l["id"]) for l in lessons)
    if dict(counts) != EXPECTED: failures.append("Lesson counts: " + str(dict(counts)))
    if len({l["id"] for l in lessons}) != len(lessons): failures.append("Duplicate lesson IDs")
    if len({c["id"] for c in cases}) != len(cases): failures.append("Duplicate example IDs")
    if set(metadata["lessons"]) != {l["id"] for l in lessons}: failures.append("Metadata/catalog ID mismatch")
    by_id = {l["id"]: l for l in lessons}
    for lesson in lessons:
        for prerequisite in lesson["prerequisites"]:
            if prerequisite not in by_id: failures.append(lesson["id"] + ": unknown prerequisite " + prerequisite)
        if lesson["has_training_walkthrough"] != (lesson["id"] in WALKTHROUGHS):
            failures.append(lesson["id"] + ": training walkthrough ownership mismatch")
        if lesson["has_training_walkthrough"] and not lesson.get("lab_handoff"):
            failures.append(lesson["id"] + ": missing lab handoff")
    visiting, visited = set(), set()
    def visit(identifier):
        if identifier in visiting: raise ValueError("Prerequisite cycle: " + identifier)
        if identifier in visited: return
        visiting.add(identifier)
        for prerequisite in by_id[identifier]["prerequisites"]: visit(prerequisite)
        visiting.remove(identifier); visited.add(identifier)
    try:
        for identifier in by_id: visit(identifier)
    except (KeyError, ValueError) as error: failures.append(str(error))
    packages = {}
    for name in ["numpy", "pandas", "scipy", "matplotlib", "seaborn", "scikit-learn", "torch", "transformers", "tokenizers", "datasets", "peft", "sentence-transformers", "spacy", "nltk", "pydantic", "httpx", "fastapi", "starlette", "tqdm", "pyarrow", "duckdb", "einops", "safetensors", "mlflow", "hypothesis", "pytest", "networkx", "gensim", "faiss-cpu", "sacrebleu", "rouge-score"]:
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    model_files = {}
    if args.model_dir.is_dir():
        for filename in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "README.md", "model.safetensors"]:
            path = args.model_dir / filename
            if path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                model_files[filename] = digest.hexdigest()
    receipt_matches = False
    receipt_path = args.model_dir.parent.parent / "receipts/model.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        receipt_matches = (receipt.get("repo") == "HuggingFaceTB/SmolLM2-360M-Instruct"
                           and receipt.get("revision") == "a10cc1512eabd3dde888204e902eca88bddb4951"
                           and bool(model_files)
                           and all(receipt.get("files", {}).get(name) == value for name, value in model_files.items()))
    if not receipt_matches:
        failures.append("Local pretrained model files do not match the expected prepared-asset receipt")
    report = {"checked_at":datetime.now(timezone.utc).isoformat(), "python":sys.version.split()[0],
              "packages":packages, "lesson_count":len(lessons), "example_count":len(cases),
              "passed":not failures, "failures":failures, "examples":cases,
              "training_executed":False, "model_downloads":False, "website_integration_tested":False,
              "existing_local_model":{"repo":"HuggingFaceTB/SmolLM2-360M-Instruct",
                                      "revision":"a10cc1512eabd3dde888204e902eca88bddb4951", "file_hashes":model_files,
                                      "prepared_receipt_matches":receipt_matches},
              "limits":["Outputs checked in this CPU environment, not all platforms.",
                        "No full training-loop execution or model-quality evaluation.",
                        "No website or browser integration tested.",
                        "Optional extension recipes have separate readiness status."]}
    if args.record:
        destination = ROOT / "verification"
        destination.mkdir(exist_ok=True)
        (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        (destination / "stdout.json").write_text(json.dumps(outputs, indent=2, ensure_ascii=False) + "\n")
        (ROOT / "catalog.json").write_text(json.dumps({"schema":1, "lessons":lessons}, indent=2) + "\n")
    print(json.dumps({"lessons":len(lessons),"examples":len(cases),"passed":not failures,"failures":failures},indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
