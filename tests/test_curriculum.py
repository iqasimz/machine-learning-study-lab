"""The reader imports complete original files; its API never accepts a path."""
import hashlib
import re
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from training_lab import curriculum as c


class CurriculumTests(unittest.TestCase):
    def test_stage_coverage(self):
        index = c.catalog()
        self.assertEqual({s:sum(l["stage"]==s for l in index["lessons"]) for s in index["stages"]},
                         {"basics":5,"python-basics":14,"python-data":10,"python-ml":12,"python-dl":12,"python-nlp":8,"python-llm":10,"python-libraries":9,"python-dsa":20,"ml":16,"dl":14,"language":2,"nlp":24,"training":9,"post":9})
        self.assertEqual(len({l["id"] for l in index["lessons"]}),174)
        self.assertEqual(len(index["questions"]),62)
        self.assertEqual(len({q["id"] for q in index["questions"]}),62)
        self.assertGreaterEqual(len(index["glossary"]),30)

    def test_all_documents_are_complete_originals(self):
        index = c.catalog()
        for entry in index["lessons"] + index["resources"]:
            with self.subTest(document=entry["id"]):
                doc = c.document(entry["id"])
                original = (c.MATERIALS / entry["source"]).read_text(encoding="utf-8")
                if entry.get("stage", "").startswith("python-"):
                    self.assertIn(doc["markdown"].split("\n",1)[1].strip(), original)
                    self.assertEqual(doc["sha256"],hashlib.sha256(doc["markdown"].encode()).hexdigest())
                    continue
                self.assertEqual(doc["markdown"],original)
                self.assertEqual(doc["sha256"],hashlib.sha256(original.encode()).hexdigest())
                self.assertTrue(doc["sections"])
                self.assertTrue(all(s["markdown"].strip() for s in doc["sections"]))

    def test_interview_answers_are_separate_and_real(self):
        for entry in c.catalog()["questions"]:
            self.assertNotIn("markdown",entry)
            doc = c.interview(entry["id"])
            self.assertGreater(len(doc["markdown"]),100)
            self.assertIn(doc["markdown"],(c.MATERIALS / doc["source"]).read_text())
        self.assertEqual(c.interview("P16")["id"],"B69")

    def test_no_arbitrary_path_endpoint(self):
        for name in ["../../secret", "..%2Fsecret", "README.md", "", "P1/../P2", "P1?path=secret"]:
            with self.subTest(name=name), self.assertRaises(ValueError): c.document(name)
        with self.assertRaises(ValueError): c.interview("F99")

    def test_section_parser_does_not_split_code(self):
        text = "# Title\n\n## First\nText\n```python\n## Not a section\n```\n## 2. Second\nOther"
        parts = c.sections(text)
        self.assertEqual([s["title"] for s in parts],["First","Second"])
        self.assertIn("## Not a section",parts[0]["markdown"])

    def test_navigation_and_runtime_boundary(self):
        self.assertIsNone(c.document("P2")["previous"])
        with self.assertRaises(ValueError):c.document("P1")
        self.assertEqual(c.document("P7")["next"],"N01")
        self.assertEqual(c.document("N24")["next"],"T0")
        self.assertEqual(c.document("T0")["next"],"T1")
        self.assertEqual(c.document("T8")["next"],"R1")
        self.assertIsNone(c.document("R9")["next"])
        self.assertIsNone(c.document("R2")["lab"],"RL theory must not pretend to launch LLM RL")
        self.assertIn("still planned",c.STAGES["post"]["description"])

    def test_advanced_walkthroughs_and_sources_are_in_every_lesson(self):
        for entry in c.catalog()["lessons"]:
            text=c.document(entry["id"])["markdown"]
            if entry["stage"].startswith("python-"):
                self.assertEqual(c.document(entry["id"])["reader"]["type"],"code")
                continue
            with self.subTest(lesson=entry["id"]):
                if entry["stage"]=="nlp":self.assertEqual(len(c.document(entry["id"])["reader"]["notes"]),2)
                elif entry["id"]=="P8":self.assertIn("ROC",text)
                elif not entry["id"].startswith(("ML","DL")):self.assertIn("### ",text)
                else:
                    self.assertIn("**Symbols and assumptions.**",text)
                    self.assertIn("**Answer.**",text)
                    self.assertTrue(c.document(entry["id"])["parameters"])
                self.assertGreaterEqual(len(re.findall(r"https://[^\s)]+",text)),2)
                headings=re.findall(r"^## (.+)$",text,re.M)
                self.assertEqual(len(headings),len(set(headings)),"No duplicated exercise/section headings")
                self.assertNotIn("VISUAL_LAB.html",text)

    def test_definitions_and_process_precede_every_teaching_topic(self):
        count = 0
        for entry in c.catalog()["lessons"]:
            doc = c.document(entry["id"])
            self.assertTrue(doc["markdown"].startswith(f'# {entry["id"]}. {entry["title"]}\n'))
            if entry["stage"].startswith("python-"):continue
            if entry["id"]=="P8" or entry["stage"]=="nlp":continue
            if entry["stage"] in {"ml","dl"}:
                self.assertLess(doc["markdown"].index("**Key terms.**"),doc["markdown"].index("**Process.**"))
                self.assertLess(doc["markdown"].index("**Process.**"),doc["markdown"].index("**Symbols and assumptions.**"))
                continue
            topics = re.split(r"^## \d+\. .+\n",doc["markdown"],flags=re.M)[1:]
            for number, topic in enumerate(topics,1):
                with self.subTest(lesson=entry["id"],topic=number):
                    self.assertRegex(topic,r"^\n\*\*Key terms\.\*\* [^\n]+\n\n\*\*Process\.\*\* [^\n]+\n")
                    self.assertGreater(len(topic.split("**Process.**",1)[1].splitlines()[0]),80)
                    count += 1
        self.assertEqual(count,162)

    def test_worked_examples_declare_their_units_and_limits(self):
        self.assertIn("18/30=0.60",c.document("P4")["markdown"])
        self.assertIn("0.05/2=0.025",c.document("P3")["markdown"])
        self.assertIn("global token mean",c.document("T7")["markdown"])
        self.assertIn("USD 0.0008",c.document("R8")["markdown"])
        self.assertIn("not a current provider quote",c.document("R8")["markdown"])

    def test_common_foundations_and_subject_specific_questions(self):
        self.assertEqual(c.document("P5")["next"],"P8")
        self.assertEqual(c.document("P8")["next"],"PY01")
        self.assertEqual(c.document("LC10")["next"],"OL01")
        self.assertEqual(c.document("OL09")["next"],"DA01")
        self.assertEqual(c.document("DA20")["next"],"ML01")
        self.assertEqual(c.document("ML16")["next"],"DL01")
        self.assertEqual(c.document("DL14")["next"],"P6")
        self.assertEqual(c.interview("IML01")["category"],"ml")
        self.assertEqual(c.interview("IDL12")["id"],"B69")
        self.assertEqual(sum(q["category"]=="ml" for q in c.catalog()["questions"]),9)
        self.assertEqual(sum(q["category"]=="dl" for q in c.catalog()["questions"]),7)


if __name__ == "__main__": unittest.main()
