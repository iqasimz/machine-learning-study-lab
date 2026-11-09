"""Check delivery and selected derivations; these do not certify learning efficacy."""
import copy
import math
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from training_lab import curriculum, guided_course, lesson_reader
from training_lab.teaching import explanations


class TeachingContentTests(unittest.TestCase):
    def test_all_active_lessons_have_visible_source_backed_explanations(self):
        authored = explanations()
        active = set(curriculum.READER_PROFILES)
        self.assertEqual(set(authored), active)
        for identifier, chapter in authored.items():
            with self.subTest(lesson=identifier):
                self.assertEqual(chapter["title"], curriculum.READER_PROFILES[identifier]["title"])
                self.assertGreater(len(chapter["markdown"].split()), 150)
                self.assertGreaterEqual(len(chapter["sources"]), 2)
                self.assertTrue(all(url.startswith("https://") for _, url in chapter["sources"]))
                for phrase in ("your thesis", "qasim", "the served step", "the lesson should", "grok"):
                    self.assertNotIn(phrase, chapter["markdown"].casefold())
                doc = curriculum.document(identifier)
                self.assertEqual(doc["reader"]["reading"]["lesson_markdown"], chapter["markdown"])
                self.assertEqual(guided_course.lesson(identifier)["steps"][0]["document"]["reader"], doc["reader"])
                self.assertTrue({u for _, u in chapter["sources"]}.issubset(
                    {u for _, u in doc["reader"]["sources"]}))

    def test_fingerprint_includes_the_explanation(self):
        doc = curriculum.document("P3")
        changed = copy.deepcopy(lesson_reader.TEACHING["P3"])
        try:
            lesson_reader.TEACHING["P3"]["markdown"] += "\nA substantive explanation changed."
            other = curriculum.document("P3")
            self.assertNotEqual(other["reader"]["fingerprint"], doc["reader"]["fingerprint"])
        finally:
            lesson_reader.TEACHING["P3"] = changed

    def test_missing_foundation_objects_precede_their_applications(self):
        text = explanations()["P3"]["markdown"]
        for term in ("random variable", "density", "sampling distribution", "standard error",
                     "empirical distribution", "with replacement", "confidence-interval"):
            self.assertIn(term, text)
        self.assertLess(text.index("random variable"), text.index("empirical distribution"))
        self.assertLess(text.index("sampling distribution"), text.index("Why bootstrap"))

    def test_selected_worked_examples_independently(self):
        self.assertAlmostEqual(.4 / (.4 + .05), 8 / 9)
        self.assertAlmostEqual(2 / 102, 1 / 51)
        self.assertAlmostEqual(1 - .95 ** 20, .6415140775914581)
        self.assertAlmostEqual(math.sqrt(.82 * .18 / 100), .03841874542459709)
        w = 1 - .1 * -12
        self.assertAlmostEqual((2 * w - 1 - 4) ** 2, .36)
        self.assertAlmostEqual(.9 * .8 + .2 * -.5, .62)
        self.assertAlmostEqual(.7 * math.tanh(.62), .38578960990909084, places=3)
        self.assertAlmostEqual((2 + 9) / 4, 2.75)

    def test_known_draft_errors_do_not_reach_the_new_main_text(self):
        texts = explanations()
        self.assertIn("8/9", texts["ML07"]["markdown"])
        self.assertIn("negative ELBO", texts["DL10"]["markdown"])
        self.assertIn("categorical", texts["DL12"]["markdown"])
        self.assertIn("preceding contexts", texts["N05"]["markdown"])
        self.assertIn("evaluated targets' mean", texts["P8"]["markdown"])
        self.assertIn("two-thirds", texts["T5"]["markdown"])
        self.assertIn("alone does not prevent", texts["T2"]["markdown"])
        self.assertIn("final input c has a target d", texts["T0"]["markdown"])
        self.assertIn("F1 1 but exact match 0", texts["N19"]["markdown"])
        self.assertIn("Mitchell", texts["T8"]["markdown"])
        self.assertNotIn("P1,", curriculum.document("ML01")["markdown"])


if __name__ == "__main__":
    unittest.main()
