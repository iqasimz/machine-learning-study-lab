"""Load the course's main explanations from explicitly keyed Markdown chapters.

This content is part of the served reader, not an audit or optional resource.
Only fixed files in the bundled materials directory are read; clients cannot
request arbitrary paths. The full text participates in reader fingerprinting.
"""
import re
from .materials import MATERIALS

CHAPTERS = ("foundations.md", "machine_learning.md", "deep_learning.md",
            "natural_language.md", "language_models.md")


def explanations():
    result = {}
    for name in CHAPTERS:
        path = MATERIALS / "teaching" / name
        text = path.read_text(encoding="utf-8")
        headings = list(re.finditer(r"^## ([A-Z]+\d+)\. ([^\n]+)\n", text, re.M))
        if not headings:
            raise ValueError(f"No keyed teaching sections in {name}.")
        for index, match in enumerate(headings):
            identifier = match[1]
            if identifier in result:
                raise ValueError(f"Duplicate teaching explanation: {identifier}.")
            end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
            markdown = text[match.end():end].strip()
            if not markdown:
                raise ValueError(f"Empty teaching explanation: {identifier}.")
            sources = list(dict.fromkeys(re.findall(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", markdown)))
            result[identifier] = {"title": match[2], "markdown": markdown,
                                  "source": "teaching/" + name, "sources": sources}
    return result
