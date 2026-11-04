"""Read-only interactive course: complete authored topics plus explicit learning checks.

This module never trains, downloads, grades free text or writes learner records.
Progress stays in the browser; existing interactive.py remains the inference helper.
"""
import hashlib
import json
import re

from . import curriculum

CONTENT = curriculum.MATERIALS / "interactive_mode"
PATHS = {
    "basics": {"title": "Shared foundations", "description": "Review shapes, gradients, uncertainty, experimental boundaries and metrics once. Skip familiar entries; subjects do not repeat this section.", "stages": ["basics"]},
    "ml": {"title": "Machine learning", "description": "From a prediction question to classical models, uncertainty and a complete experiment.", "stages": ["ml"]},
    "dl": {"title": "Deep learning", "description": "Follow gradients through neural architectures, representations and training systems.", "stages": ["dl"]},
    "nlp": {"title":"Natural language processing", "description":"From linguistic units and lexical features to recurrent models, structured prediction, sentence encoders and grounded NLP evaluation.", "stages":["language","nlp"]},
    "llm": {"title": "LLMs", "description": "Transformer training, adaptation, retrieval, agents and deployment. Shared tokenization and embedding lessons are under NLP.", "stages": ["training", "post"]},
}


def _json(name):
    path = CONTENT / name
    if path.is_symlink() or not path.resolve().is_relative_to(CONTENT.resolve()):
        raise ValueError("Course content must remain inside its authored folder.")
    return json.loads(path.read_text(encoding="utf-8"))


def checks():
    result = {}
    for file in ["foundations_llm_checks.json", "ml_checks.json", "dl_checks.json", "nlp_checks.json"]:
        rows = _json(file)["lessons"]
        if result.keys() & rows.keys():
            raise ValueError("Duplicate assessment lesson.")
        result.update(rows)
    def q(key, prompt, options, answer, explanation):
        return {"id":key,"prompt":prompt,"answer":answer,"explanation":explanation,"skill":"Inspect metric denominators",
                "options":[{"text":text,"feedback":feedback} for text,feedback in options]}
    result["P8"] = [
        q("P8-a","A classifier finds 8 positives, misses 2 and raises 4 false alarms. Which precision/recall pair is correct?",
          [("Precision 8/12, recall 8/10","Precision counts the 12 predicted positives; recall counts the 10 actual positives."),
           ("Precision 8/10, recall 8/12","Those denominators are reversed: precision uses predictions, recall uses actual positives."),
           ("Precision and recall are both 8/14","Adding every positive-related count gives neither metric's denominator."),
           ("Both equal overall accuracy","Accuracy also counts true negatives; no true-negative count is supplied.")],0,
          "Precision is TP/(TP+FP)=8/12; recall is TP/(TP+FN)=8/10. F1 is not the arithmetic mean."),
        q("P8-b","Two language-model evaluations score different numbers of valid targets. How should their combined mean NLL be computed?",
          [("Average the two reported means equally","That only matches token weighting when both evaluations score the same valid count."),
           ("Add the perplexities, then divide by two","Perplexity is exponential in mean NLL; averaging perplexities does not recover this objective."),
           ("Sum loss over valid targets, then divide by their total count","This gives each scored target equal weight under the specified matched scoring protocol."),
           ("Use whichever run has the lower loss","Selecting the best run changes the question and does not calculate a combined mean.")],2,
          "Weight each mean NLL by its valid-target count, or sum target losses directly, then divide by total valid count. Tokenization/context must also match.")
    ]
    active={row[0] for row in curriculum.LESSONS}
    for identifier, rows in _json("advanced_checks.json")["lessons"].items():
        if identifier not in result or identifier not in active:
            raise ValueError("Advanced checks require an existing active lesson.")
        result[identifier].extend(rows)
    return {key:value for key,value in result.items() if key in active}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def source_blocks(markdown):
    """Partition exact source text at safe headings, never inside code/math blocks.

    Source offsets are Python string/character offsets, not UTF-8 byte offsets.
    """
    lines = markdown.splitlines(keepends=True)
    headings = []; fence = None; math = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        marker = re.match(r"^(`{3,}|~{3,})", stripped)
        if marker:
            token = marker[1][0]
            if fence is None: fence = token
            elif token == fence: fence = None
            continue
        if fence is not None: continue
        if stripped in {r"\[", "$$"}: math = not math; continue
        if stripped == r"\]": math = False; continue
        if not math and re.match(r"^#{2,3} ", line):
            headings.append((i, line.strip().lstrip("# ")))
    if not headings:
        return [{"title": "Complete lesson", "markdown": markdown, "source_start": 0, "source_end": len(markdown)}]
    blocks = []; offsets = [0]
    for line in lines: offsets.append(offsets[-1] + len(line))
    for pos, (line_number, title) in enumerate(headings):
        start = 0 if pos == 0 else line_number
        end = headings[pos+1][0] if pos+1 < len(headings) else len(lines)
        blocks.append({"title": re.sub(r"^\d+\.\s*", "", title), "markdown": "".join(lines[start:end]),
                       "source_start": offsets[start], "source_end": offsets[end]})
    return blocks


def lesson(identifier):
    if identifier not in {row[0] for row in curriculum.LESSONS}:
        raise ValueError("Unknown interactive lesson.")
    doc = curriculum.document(identifier)
    questions = checks()[identifier]
    bridge = _json("lesson_bridges.json")["lessons"].get(identifier, {})
    steps = []
    def add(kind, title, markdown="", **extra):
        key = extra.pop("key")
        step = {"id": identifier+":"+key, "kind":kind, "title":title, "markdown":markdown, **extra}
        step["fingerprint"] = fingerprint(step)
        steps.append(step)
    # Both navigation modes render this exact canonical document with one reader.
    add("lesson", doc["title"], key="reader", document=doc)
    for i, question in enumerate(questions):
        title = "Check the idea" if i == 0 else "Use it in a decision" if i == 1 else "Advanced check"
        key = "check-a" if i == 0 else "check-b" if i == 1 else "check-" + question["id"]
        add("question", title, question=question, key=key)
    if bridge.get("lab"):
        add("lab", "Apply in the lab", "## Controlled experiment\n\n"+bridge["lab"]["challenge"],
            lab=bridge["lab"], key="lab")
    for i, step in enumerate(steps[:-1]):
        step["next_hint"] = steps[i+1]["title"]
    return {"id":identifier, "title":doc["title"], "goal":doc["goal"], "source":doc["source"],
            "source_sha256":doc["sha256"], "reader_fingerprint":doc["reader"]["fingerprint"],
            "fingerprint":fingerprint({"reader":doc["reader"]["fingerprint"], "steps":steps}),
            "steps":steps, "prerequisites":doc.get("prerequisites", []),
            "takeaway":bridge.get("takeaway", "Explain the cases, operation and denominator before interpreting a score.")}


def catalog():
    decks={row[0]:lesson(row[0]) for row in curriculum.LESSONS}
    units={key:{"title":deck["title"],"fingerprint":deck["fingerprint"],"steps":[{k:step[k] for k in ["id","kind","title","fingerprint"]} for step in deck["steps"]]} for key,deck in decks.items()}
    paths = []
    for identifier, spec in PATHS.items():
        rows = [row for row in curriculum.LESSONS if row[1] in spec["stages"]]
        paths.append({"id": identifier, "title": spec["title"], "description": spec["description"],
                      "lessons": [{"id": row[0], "title": row[3], "stage": row[1]} for row in rows]})
    return {"schema_version": 2, "paths": paths, "units":units, "shared_foundations": ["P2", "P3", "P4", "P5", "P8"],
            "progress_note": "Reading and answer history are saved only in this browser at this address. They are not evidence of independent mastery. Reading never starts training."}
