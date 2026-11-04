"""One canonical student view; complete authored Markdown remains recoverable.

This is an explicit editorial map, not a similarity model that silently deletes
sentences. Teaching chapters own the connected main explanation; explorers own
worked calculations. Unique source sections become depth entries in the same
reader, rather than a competing copy of the main lesson.
"""
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from .artifacts import ROOT
from .materials import MATERIALS
from .teaching import explanations

TEACHING = explanations()

ADVANCED = {identifier: {row["section"]: row for row in rows}
            for identifier, rows in json.loads((MATERIALS / "ml_dl/metadata/advanced_sections.json").read_text())["lessons"].items()}


@lru_cache(maxsize=1)
def runtime_fingerprint():
    web = Path(__file__).parent / "web"
    files = ["architecture_math.js", "architecture_lessons.js", "architecture_explorer.js",
             "explorer_lessons.js", "explorer_ml.js", "explorer_dl.js", "explorer_llm.js",
             "explorer_figures.js", "metrics.js", "learning.js", "nlp_math.js", "explorer_nlp.js", "nlp_figures.js"]
    return hashlib.sha256(b"".join((web/name).read_bytes() for name in files)).hexdigest()

# Source H2 numbers retained for additional depth. Core walkthroughs and repeated
# examples are intentionally not displayed a second time.
KEEP = {
    "P2": [4, 6], "P3": [2, 4, 5], "P4": [4], "P5": [5, 6],
    "P6": [1, 3, 4, 5, 6], "P7": [3, 4, 6], "P8": [2, 3],
    "T0": [10], "T1": [3, 4, 5], "T2": [2, 3, 5, 6],
    "T3": [6], "T4": [2, 3, 4, 5, 6], "T5": [3, 4, 5, 6],
    "T6": [1, 2, 3, 4, 5, 6], "T7": [2, 3, 4, 5, 6], "T8": [2, 3, 4, 5, 6],
    "R1": [1, 4, 5, 6], "R2": [2, 3, 4, 5, 6, 7, 8, 9], "R3": [1, 2, 3, 4, 5, 6],
    "R4": [2, 3, 4, 5, 6, 7, 8, 9], "R5": [3, 4, 5, 6, 7],
    "R6": [3, 4, 5, 6, 7], "R7": [3, 4, 5, 6, 7],
    "R8": [3, 4, 5, 6, 7, 8], "R9": [2, 3, 4, 5, 6, 7, 8],
}
# Retain deeper subsections without repeating the section's basic introduction.
EXTENSIONS = {
    "T0": (7,"Check the complete update boundary"),
    "T1": (2,"Why one forward pass can learn many next tokens"),
    "T5": (2,"A two-microbatch update, with the denominator made explicit"),
    "R1": (3,"Why zero initial adapter output does not stop learning"),
    "DL12": (2,"Region overlap: IoU and Dice"),
    "ML05": (1,"Learning curves and validation curves"),
    "P4": (6,"A transfer experiment that separates style from definition"),
    "R5": (2,"Follow a consequential tool request without confusing intent and execution"),
    "R9": (5,"Evaluate a steering direction as an intervention, not a label"),
}
for prefix, count in (("ML", 16), ("DL", 14)):
    for i in range(1, count+1):
        KEEP[f"{prefix}{i:02}"] = [3, 4, 5]
# Preserve important qualifications not fully developed in the compact example.
for identifier in ("ML02", "ML03", "ML04", "ML06", "ML07", "ML11", "ML13", "ML15",
                   "DL04", "DL05", "DL07", "DL10", "DL11", "DL12"):
    KEEP[identifier].insert(0, 1)
# Definitions/derivations of shared metrics have one canonical home.
KEEP["ML10"] = [4, 5]
KEEP["ML08"] = [1, 3, 4]
KEEP["ML12"] = [1, 3, 4, 5]
KEEP["DL05"] = [3, 4, 5]
KEEP["DL03"] = [1, 3, 4, 5]
KEEP["DL10"] = [3, 4, 5]
KEEP.update({f"N{i:02}":[1,2] for i in range(1,25)})

# These source paragraphs describe examples already shown in the canonical NLP
# walkthrough. Omit them only from depth notes; retain the complete source text.
NLP_CORE_EXAMPLES = {
    "N01": ("The example agreement calculation is deliberately small:",),
    "N02": ("The example “not good” becomes",),
    "N03": ("The worked example uses raw term frequency",),
    "N04": ("The browser uses authored counts",),
    "N05": ("The example conditions on “like”",),
    "N07": ("The scalar example uses three token features",),
    "N12": ("The browser enumerates all eight paths",),
    "N17": ("The authored example contains",),
    "N19": ("The small diagram uses an exact context-token alignment",),
    "N21": ("The diagram compares two explicitly declared annotation policies",),
    "N22": ("The diagram's 2 × 2 source–target score matrix",),
    "N23": ("Overlapping windows can preserve a span",),
    "N24": ("The authored example has two gold entities",),
}

# The canonical LLM examples already show these arithmetic cases. Keep the
# complete source available while avoiding a duplicate in the deeper notes.
LLM_CORE_EXAMPLES = {
    "R2": ("For positive advantage 2,", "Separate numerical DPO example:",
           "Separate four-response example rewards"),
    "R3": ("**Calculation.** For two logits [2,0]",),
}

RELATED = {
    "P2": ["DL02", "ML04"], "P3": ["P8", "ML10"], "P4": ["P8", "ML15", "ML16"],
    "P5": ["DL01", "DL02", "DL03"], "P6": ["T4", "T2"],
    "P7": ["P8", "R4"], "P8": ["ML10", "P3", "T8", "R4"],
    "ML01": ["P8"], "ML03": ["P5", "P8"], "ML04": ["P8"],
    "ML07": ["P3", "ML15"], "ML10": ["P8", "P3"], "ML14": ["P7", "R2"],
    "ML15": ["ML07", "DL10"], "ML16": ["P8", "DL13", "R8"],
    "DL05": ["DL08", "DL12"], "DL06": ["DL08", "P7", "DL10"],
    "DL08": ["T3"], "DL10": ["DL09", "ML15", "DL12"], "DL11": ["ML07", "P8"],
    "DL12": ["P8", "DL05"], "DL13": ["ML16", "R6", "R8"], "DL14": ["P8"],
    "T0": ["T2", "T3", "T5", "R1"], "T1": ["P6", "P8"],
    "T3": ["T4", "R6", "T8"], "T5": ["DL03"], "T8": ["P8", "P3"],
    "R2": ["P5"], "R3": ["ML10"], "R4": ["P7", "N15", "P8"], "R6": ["T6"], "R9": ["T8"],
}


def parameter_owners(lessons, parameters):
    owners = {}
    for lesson in lessons:
        for key in lesson["parameter_ids"]:
            owners.setdefault(key, lesson["id"])
    for key in ("learning_rate", "optimizer", "momentum", "adam_beta1", "adam_beta2",
                "optimizer_epsilon", "initialization", "normalization", "weight_decay", "schedule",
                "gradient_clip"):
        if key in parameters: owners[key] = "DL03"
    for key in ("batch_size", "epochs", "device", "dtype", "loader_workers", "gradient_accumulation"):
        if key in parameters: owners[key] = "DL13"
    for key in ("threshold", "bootstrap_repeats"):
        if key in parameters: owners[key] = "P8" if key == "threshold" else "P3"
    return owners


def clean_note(text, first_section=False):
    """Remove reader scaffolding, not subject-specific prose or formulae."""
    lines = text.splitlines()
    if lines and lines[0].startswith("## "): lines = lines[1:]
    text = "\n".join(lines).strip()
    text = re.split(r"\*\*Check\.\*\*", text, maxsplit=1)[0].strip()
    paragraphs = re.split(r"\n\s*\n", text)
    kept = []
    for paragraph in paragraphs:
        if paragraph.startswith(("The related settings are defined below", "Before considering this lesson complete",
                                 "The numbers above are illustrative", "Sources are linked above", "**Process.**")):
            continue
        if first_section and paragraph.startswith(("**Key terms.**", "**Process.**")):
            continue
        kept.append(paragraph)
    return "\n\n".join(kept).strip()


def view(doc, metadata, parameters, all_metadata, reading=None):
    identifier = doc["id"]
    notes = []
    for section in doc["sections"]:
        match = re.match(r"^## (\d+)\.\s+([^\n]+)", section["markdown"])
        advanced = ADVANCED.get(identifier, {}).get(int(match[1])) if match else None
        if match and (int(match[1]) in KEEP.get(identifier, []) or advanced):
            text = clean_note(section["markdown"], int(match[1]) == 1)
            prefixes = NLP_CORE_EXAMPLES.get(identifier, ()) + LLM_CORE_EXAMPLES.get(identifier, ())
            if prefixes:
                text = "\n\n".join(p for p in re.split(r"\n\s*\n", text)
                                    if not p.startswith(prefixes)).strip()
            if text:
                if identifier == "ML03" and int(match[1]) == 5:
                    text = ("The diagram uses an unhalved summed squared loss plus a ridge penalty. The derivation below "
                            "explicitly uses a 1/(2n) convention, so its derivatives have a different scale.\n\n" + text)
                title = {"Scenario and check":"Apply the idea", "Definitions and mechanism":"Assumptions and limitations",
                         "Alternatives and controls":"Compare the alternatives"}.get(match[2],match[2])
                note = {"title": title, "markdown": text, "kind": "depth"}
                if advanced:
                    note.update(level="advanced", advanced_id=advanced["id"])
                notes.append(note)
        if (identifier in EXTENSIONS and match and int(match[1]) == EXTENSIONS[identifier][0]
                and int(match[1]) not in KEEP.get(identifier, [])):
            title=EXTENSIONS[identifier][1]
            extension=re.search(r"^### "+re.escape(title)+r"\n(.+)",section["markdown"],re.M|re.S)
            if extension:notes.append({"title":title,"markdown":extension[1].strip(),"kind":"depth"})
        # A real-model extension is different from the width-4 teaching decoder.
        if identifier == "T3" and match and int(match[1]) == 5:
            advanced = re.search(r"^### Read a real grouped-query block by its dimensions\n(.+)", section["markdown"], re.M | re.S)
            if advanced:
                notes.append({"title": "Read the actual local grouped-query model",
                              "markdown": advanced[1].strip(), "kind": "depth"})
    owners = parameter_owners(all_metadata, parameters)
    owned = [parameters[key] for key, owner in owners.items() if owner == identifier]
    for start in range(0, len(owned), 3):
        cards = owned[start:start+3]
        text = []
        for p in cards:
            text.append("### "+p["label"]+"\n\n"+p["definition"]+"\n\n"+p["mathematical_meaning"]+
                        "\n\n"+p["expected_effect"]+"\n\n"+"Interactions: "+"; ".join(p["interactions"])+
                        ".\n\n"+" ".join(p["failure_modes"]))
        notes.append({"title": "Settings: "+", ".join(p["label"] for p in cards),
                      "markdown": "\n\n".join(text), "kind": "settings",
                      "parameter_ids": [p["id"] for p in cards]})
    # Small local application links, never another copy of the definition.
    related = list(metadata.get("related",RELATED.get(identifier, [])))
    for p in metadata.get("parameter_ids", []):
        owner = owners.get(p)
        if owner and owner != identifier and owner not in related: related.append(owner)
    active_ids = {row["id"] for row in all_metadata} | set(KEEP)
    related = [x for x in related if x != identifier and x in active_ids]
    source_links = []
    for title, url in re.findall(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", doc["markdown"]):
        if not any(u == url for _, u in source_links): source_links.append([title, url])
    result = {"version": 1, "explorer": {"DL05": "cnn", "T3": "transformer"}.get(identifier, identifier),
              "notes": notes, "related": related,
              "sources": source_links, "runtime_fingerprint": runtime_fingerprint(),
              "original_sha256": doc["sha256"]}
    if reading:
        teaching = TEACHING[identifier]
        result["reading"] = {**reading, "lesson_markdown": teaching["markdown"],
                             "lesson_source": teaching["source"], "edition": "2026-11-05"}
        for title, url in teaching["sources"]:
            if not any(u == url for _, u in source_links):
                source_links.append([title, url])
    result["fingerprint"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result
