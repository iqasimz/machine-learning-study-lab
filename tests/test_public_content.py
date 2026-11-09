"""Keep served lessons general-purpose rather than tied to a private study plan."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from training_lab import curriculum


class PublicContentTests(unittest.TestCase):
    def test_served_content_has_no_personal_study_references(self):
        index = curriculum.catalog()
        forbidden = ("your thesis", "qasim's thesis", "you previously chose", "bachelors track",
                     "../projects/evidence_desk", "../projects/llm_training_lab")
        records = [(row["id"], curriculum.document(row["id"])["markdown"])
                   for row in index["lessons"] + index["resources"]]
        records += [(row["id"], curriculum.interview(row["id"])["markdown"])
                    for row in index["questions"]]
        for identifier, text in records:
            for phrase in forbidden:
                with self.subTest(document=identifier, phrase=phrase):
                    self.assertNotIn(phrase, text.casefold())

    def test_course_map_contains_every_active_lesson(self):
        text = (curriculum.MATERIALS / "COURSE_MAP.md").read_text()
        for identifier, _, source, title, *_ in curriculum.LESSONS:
            with self.subTest(lesson=identifier):
                self.assertIn(f"| {identifier} | [{title}]({source}) |", text)


if __name__ == "__main__": unittest.main()
