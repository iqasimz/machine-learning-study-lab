"""Read-only Python lessons and checked outputs; never executes authored code."""
import hashlib
import json
import re
from pathlib import Path

from .dsa_scenes import scene_for
from .dsa_teaching import teaching_for
from .materials import MATERIALS

ROOT = MATERIALS / "python"
EXAMPLES = re.compile(r"<!-- example: ([A-Z]+\d+-[a-z0-9-]+) -->\n```python\n(.*?)\n```\n\n\*\*Output\*\*\n```text\n(.*?)\n```", re.S)
SCENE = re.compile(r"^<!-- dsa-scene: (DA\d{2}) -->$")
PLOTS = {"DS08-main-progress.png": "Reading progress: 0, 4 and 9 records at 0, 10 and 20 minutes.",
         "DS09-main-confusion.png": "Confusion matrix: actual rows and predicted columns; counts 1, 1, 0 and 2."}


def read(relative):
    path = ROOT / relative
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Python course sources must remain inside the course folder.")
    return path.read_text(encoding="utf-8")


def metadata():
    return json.loads(read("metadata.json"))


def rows():
    result = []
    spec = metadata()
    for section in spec["sections"]:
        text = read(section["chapter"])
        for identifier, title in re.findall(r"^## ([A-Z]+\d+)\. (.+)$", text, re.M):
            if identifier not in spec["lessons"] or not identifier.startswith(section["prefix"]):
                raise ValueError("Python lesson metadata differs from its authored chapter.")
            result.append((identifier, section["id"], "python/" + section["chapter"], title,
                           "Read the code, inspect its output and explain what changed.", [], None))
    if len(result) != len(spec["lessons"]):
        raise ValueError("Python course entries must have one authored explanation each.")
    return result


def asset(name):
    if name not in PLOTS:
        raise ValueError("Unknown Python course image.")
    path = ROOT / "outputs" / name
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Invalid Python course image path.")
    return path


def assets():
    return [{"source": "python/outputs/" + name, "url": "./course-assets/python/" + name,
             "alt": alt, "sha256": hashlib.sha256(asset(name).read_bytes()).hexdigest()}
            for name, alt in PLOTS.items()]


def checks():
    result = {}
    for position, row in enumerate(json.loads(read("learning_checks.json"))["rows"]):
        identifier, prompt, correct, explanation, *wrong = row
        options = [{"text": correct, "feedback": explanation}] + [{"text": text, "feedback": feedback} for text, feedback in wrong]
        answer = position % 4
        options = options[-answer:] + options[:-answer] if answer else options
        result[identifier] = [{"id": identifier + "-a", "prompt": prompt, "answer": answer,
                               "options": options, "explanation": explanation,
                               "skill": "Explain code behavior and the learning/execution boundary"}]
    if set(result) != set(metadata()["lessons"]):
        raise ValueError("Every Python lesson needs its authored learning check.")
    return result


def append_prose(text, blocks):
    pending = []

    def flush():
        prose = "\n".join(pending).strip()
        pending.clear()
        if prose:
            blocks.append({"kind": "prose", "markdown": prose})

    for line in text.splitlines():
        match = SCENE.match(line.strip())
        if match:
            flush()
            identifier = match.group(1)
            blocks.append({"kind": "scene", "id": identifier, "scene": scene_for(identifier)})
        else:
            pending.append(line)
    flush()


def document(identifier):
    spec = metadata()
    if identifier not in spec["lessons"]:
        raise ValueError("Unknown Python lesson.")
    row = next(row for row in rows() if row[0] == identifier)
    text = read(row[2].removeprefix("python/"))
    headings = list(re.finditer(r"^## ([A-Z]+\d+)\. (.+)$", text, re.M))
    position = next(i for i, heading in enumerate(headings) if heading[1] == identifier)
    start = headings[position].end()
    end = headings[position + 1].start() if position + 1 < len(headings) else len(text)
    body = text[start:end].strip()
    markdown = "# " + identifier + ". " + row[3] + "\n\n" + body + "\n"
    evidence = json.loads(read("verification/report.json"))
    checked = {entry["id"]: entry for entry in evidence["examples"]}
    blocks = []; cursor = 0
    for match in EXAMPLES.finditer(body):
        if match.start() > cursor:
            append_prose(body[cursor:match.start()], blocks)
        name, code, output = match.groups()
        record = checked.get(name, {})
        if (not record.get("passed") or record.get("code_sha256") != hashlib.sha256(code.encode()).hexdigest()
                or record.get("expected_sha256") != hashlib.sha256(output.encode()).hexdigest()):
            raise ValueError("Recheck the code/output pair before serving " + name + ".")
        plot_assets = []
        for plot in record.get("plots", []):
            filename = Path(plot["file"]).name
            if filename not in PLOTS or hashlib.sha256(asset(filename).read_bytes()).hexdigest() != plot["sha256"]:
                raise ValueError("Python plot differs from its checked data.")
            plot_assets.append(next(a for a in assets() if a["source"].endswith("/" + filename)))
        blocks.append({"kind": "example", "id": name, "code": code, "output": output,
                       "output_kind": record["kind"], "plots": plot_assets})
        cursor = match.end()
    if cursor < len(body): append_prose(body[cursor:], blocks)
    sources = list(dict.fromkeys(re.findall(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", body)))
    lesson = spec["lessons"][identifier]
    reader = {"type": "code", "blocks": blocks, "notes": [], "sources": sources,
              "related": lesson["theory_ids"], "prerequisites": lesson["prerequisites"],
              "lab_handoff": lesson.get("lab_handoff"), "original_sha256": hashlib.sha256(markdown.encode()).hexdigest(),
              "topics": re.findall(r"^### (.+)$", body, re.M), "edition": "2026-11-07"}
    if identifier.startswith("DA"):
        reader["teaching"] = teaching_for(identifier)
    reader["fingerprint"] = hashlib.sha256(json.dumps(reader, sort_keys=True).encode()).hexdigest()
    return {"id": identifier, "title": row[3], "stage": row[1], "source": row[2],
            "source_section": identifier, "chapter_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "goal": row[4], "diagram": [], "lab": None, "parameters": [],
            "prerequisites": lesson["prerequisites"], "markdown": markdown,
            "sha256": reader["original_sha256"], "reader": reader}
