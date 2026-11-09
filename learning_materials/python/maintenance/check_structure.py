"""Read-only checks for source/evidence consistency and future theory linking."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    metadata = json.loads((ROOT / "metadata.json").read_text())
    catalog = json.loads((ROOT / "catalog.json").read_text())
    report = json.loads((ROOT / "verification/report.json").read_text())
    source_examples = {}
    errors = []; source_urls = set(); local_links = 0
    for path in ROOT.glob("*.md"):
        text = path.read_text()
        if len(re.findall(r"^```", text, re.M)) % 2:
            errors.append(path.name + ": unmatched fence")
        if "/Users/" in text: errors.append(path.name + ": private absolute path")
        for target in re.findall(r"\[[^\]]+\]\(([^\s)]+)\)", text):
            if target.startswith(("https://", "http://")):
                source_urls.add(target); continue
            destination = ROOT / target.split("#", 1)[0]
            if not destination.exists(): errors.append(path.name + ": missing local link " + target)
            local_links += 1
        for identifier, code, output in re.findall(r"<!-- example: ([A-Z]+\d+-[a-z0-9-]+) -->\n```python\n(.*?)\n```\n\n\*\*Output\*\*\n```text\n(.*?)\n```", text, re.S):
            source_examples[identifier] = (hashlib.sha256(code.encode()).hexdigest(), hashlib.sha256(output.encode()).hexdigest())
    for case in report["examples"]:
        if source_examples.get(case["id"]) != (case["code_sha256"], case["expected_sha256"]):
            errors.append(case["id"] + ": stale verification")
    if set(source_examples) != {case["id"] for case in report["examples"]}:
        errors.append("Source/evidence example set differs")
    if set(metadata["lessons"]) != {lesson["id"] for lesson in catalog["lessons"]}:
        errors.append("Source/catalog lesson set differs")
    for lesson in catalog["lessons"]:
        text = (ROOT / lesson["chapter"]).read_text()
        pattern = r"^## " + re.escape(lesson["id"]) + r"\. [^\n]+$(.*?)(?=^## [A-Z]+\d+\. |\Z)"
        match = re.search(pattern, text, re.M | re.S)
        if not match or hashlib.sha256(match[1].encode()).hexdigest() != lesson["source_sha256"]:
            errors.append(lesson["id"] + ": stale chapter fingerprint")
    theory = json.loads((ROOT.parent / "reader_profiles.json").read_text())["lessons"]
    linked_theory = set()
    for identifier, lesson in metadata["lessons"].items():
        for linked in lesson["theory_ids"]:
            linked_theory.add(linked)
            if linked not in theory: errors.append(identifier + ": unknown theory lesson " + linked)
    required = {line.split("==", 1)[0]: line.split("==", 1)[1]
                for line in (ROOT / "requirements-examples.txt").read_text().splitlines()
                if line and not line.startswith("#")}
    for package, version in required.items():
        if report["packages"].get(package) != version:
            errors.append(package + ": direct-package record differs")
    if not report["passed"]: errors.append("Execution report contains failures")
    if not report["existing_local_model"].get("prepared_receipt_matches"):
        errors.append("Local model receipt not verified")
    for image in ROOT.glob("outputs/*.png"):
        if image.stat().st_size < 1000: errors.append("Empty plot " + image.name)
    summary = {"lessons":len(catalog["lessons"]), "examples":len(source_examples),
               "unique_reference_urls":len(source_urls), "local_links":local_links,
               "linked_theory_lessons":len(linked_theory), "errors":errors}
    print(json.dumps(summary, indent=2))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
