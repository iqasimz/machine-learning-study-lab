"""First-reader edition coverage, scope and computation-graph meaning."""
import copy
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from training_lab import curriculum as c, guided_course as g, lesson_reader as reader


class ReaderEditionTests(unittest.TestCase):
    def test_exact_course_scope(self):
        expected = {"P2", "P3", "P4", "P5", "P6", "P7", "P8"} | {f"ML{i:02}" for i in range(1, 17)} | {f"DL{i:02}" for i in range(1, 15)} | {f"N{i:02}" for i in range(1, 25)} | {f"T{i}" for i in range(9)} | {f"R{i}" for i in range(1, 10)}
        self.assertEqual(set(c.READER_PROFILES), expected)
        for row in c.catalog()["lessons"]:
            doc = c.document(row["id"])
            self.assertEqual("reading" in doc["reader"], row["id"] in expected)
            if row["id"] in expected:
                self.assertEqual(doc["title"], c.READER_PROFILES[row["id"]]["title"])
                self.assertTrue(doc["markdown"].startswith("# " + row["id"] + ". " + doc["title"] + "\n"))
                self.assertIn(c.READER_PROFILES[row["id"]]["definition"], doc["markdown"])

    def test_plain_introduction_and_specific_sections(self):
        for identifier, profile in c.READER_PROFILES.items():
            with self.subTest(lesson=identifier):
                self.assertGreater(len(profile["definition"]), 180)
                self.assertGreater(len(profile["example"]), 100)
                self.assertGreater(len(profile["connection"]), 100)
                self.assertIn(profile["view"], {"reading", "diagram", "architecture"})
                self.assertNotRegex(profile["connection"], r"\b(?:P[23458]|ML\d{2})\b")
                for variant in [profile, *profile.get("variants", {}).values()]:
                    for step in variant["steps"]:
                        self.assertGreater(len(step["prose"]), 100)
                        self.assertGreater(len(step["terms"]), 35)
                        self.assertGreater(len(step["next"]), 35)
                        self.assertNotRegex(step["next"], r"outgoing connection carries|endpoint of the worked")
                for note in c.document(identifier)["reader"]["notes"]:
                    self.assertNotIn(note["title"], {
                        "Assumptions and limitations", "Apply the idea", "Compare the alternatives"})

    def test_dependencies_are_not_reading_order(self):
        p = c.READER_PROFILES
        for identifier in ("P3", "P4", "P8", "ML01", "ML02", "ML05", "ML06",
                           "ML10", "ML14", "ML15", "ML16"):
            self.assertEqual(p[identifier]["view"], "reading")
            self.assertNotIn("diagram", p[identifier])
        logistic = {(a, b) for a, b, *_ in p["ML04"]["diagram"]["links"]}
        self.assertEqual(logistic, {(0, 1), (1, 2), (1, 3)})
        naive_bayes = {(a, b) for a, b, *_ in p["ML07"]["diagram"]["links"]}
        self.assertEqual(naive_bayes, {(0, 2), (1, 2), (2, 3)})
        for identifier, profile in p.items():
            if profile["view"] != "diagram":
                continue
            valid = set(profile["diagram"]["nodes"])
            self.assertTrue(all(a in valid and b in valid and len(note) > 30
                                for a, b, note, *_ in profile["diagram"]["links"]))

    def test_guided_and_sidebar_share_edition(self):
        for identifier in c.READER_PROFILES:
            doc = c.document(identifier)
            deck = g.lesson(identifier)
            self.assertEqual(deck["steps"][0]["document"]["reader"], doc["reader"])
            self.assertEqual(deck["title"], doc["title"])
            self.assertEqual(len([s for s in deck["steps"] if s["kind"] == "question"]), len(g.checks()[identifier]))
        self.assertEqual(set(c.READER_PROFILES["P8"]["variants"]),
                         {"classification", "multiclass", "regression", "language", "retrieval"})

    def test_factual_corrections_and_edition_fingerprinting(self):
        source = c.document("ML11")["markdown"]
        self.assertIn("initial centroids [0,10]", source)
        self.assertNotIn("initial centroids [0,8]", source)
        self.assertIn("a point at 5 is tied", source)
        doc = c.document("P2")
        changed = copy.deepcopy(c.READER_PROFILES["P2"])
        changed["definition"] += " A changed explanation requires review."
        other = reader.view(doc, {}, c.ML_DL_PARAMETERS, c.ML_DL_LESSONS, reading=changed)
        self.assertNotEqual(other["fingerprint"], doc["reader"]["fingerprint"])


if __name__ == "__main__":
    unittest.main()

