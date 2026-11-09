"""Content/protocol tests for all guided paths. No fitting, downloads or learner writes."""
import hashlib
import re
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from training_lab import curriculum, guided_course as c


class GuidedCourseTests(unittest.TestCase):
    def test_four_sections_have_one_common_foundation(self):
        index=c.catalog();paths={p["id"]:p for p in index["paths"]}
        self.assertEqual(list(paths),["basics","python","ml","dl","nlp","llm"])
        self.assertEqual({k:len(p["lessons"]) for k,p in paths.items()},{"basics":5,"python":95,"ml":16,"dl":14,"nlp":26,"llm":18})
        flat=[l["id"] for p in paths.values() for l in p["lessons"]]
        self.assertEqual(len(flat),len(set(flat)))
        self.assertEqual(set(flat),{r[0] for r in curriculum.LESSONS})
        self.assertEqual(index["shared_foundations"],["P2","P3","P4","P5","P8"])
        self.assertEqual(len(index["units"]),174)

    def test_guided_and_sidebar_use_the_same_canonical_reader(self):
        for identifier,*_ in curriculum.LESSONS:
            with self.subTest(lesson=identifier):
                doc=curriculum.document(identifier);deck=c.lesson(identifier)
                readings=[s for s in deck["steps"] if s["kind"]=="lesson"]
                self.assertEqual(len(readings),1)
                self.assertEqual(readings[0]["document"],doc)
                self.assertEqual(deck["source_sha256"],hashlib.sha256(doc["markdown"].encode()).hexdigest())
                self.assertEqual(deck["reader_fingerprint"],doc["reader"]["fingerprint"])
                self.assertEqual(deck["steps"][0]["kind"],"lesson")
                self.assertEqual(len({s["id"] for s in deck["steps"]}),len(deck["steps"]))
                self.assertNotIn("settings",{s["kind"] for s in deck["steps"]})
                self.assertNotIn("recap",{s["kind"] for s in deck["steps"]})

    def test_all_checks_have_specific_feedback(self):
        checks=c.checks();self.assertEqual(set(checks),{r[0] for r in curriculum.LESSONS})
        ids=set();prompts=set();positions=[0]*4
        for identifier,rows in checks.items():
            expected_advanced = c._json("advanced_checks.json")["lessons"].get(identifier, [])
            self.assertEqual(len(rows),(1 if identifier in curriculum.python_course.metadata()["lessons"] else 2)+len(expected_advanced))
            for q in rows:
                self.assertEqual(len(q["options"]),4)
                self.assertEqual(len({o["text"] for o in q["options"]}),4)
                self.assertIs(type(q["answer"]),int);self.assertIn(q["answer"],range(4));positions[q["answer"]]+=1
                self.assertNotIn(q["id"],ids);ids.add(q["id"])
                self.assertNotIn(q["prompt"],prompts);prompts.add(q["prompt"])
                self.assertGreater(len(q["prompt"]),25);self.assertGreater(len(q["explanation"]),30)
                self.assertTrue(q["skill"])
                for option in q["options"]:
                    self.assertGreater(len(option["feedback"]),20)
                self.assertEqual(len({o["feedback"] for o in q["options"]}),4)
            deck=c.lesson(identifier)
            self.assertEqual([s["question"]["id"] for s in deck["steps"] if s["kind"]=="question"],[q["id"] for q in rows])
        self.assertEqual(len(ids),278)
        self.assertTrue(all(.15*len(ids)<=p<=.35*len(ids) for p in positions))

    def test_partition_respects_code_and_math(self):
        text="# Example\n\n## One\n```python\n### Not a heading\n```\n\\[\n## Also not a heading\n\\]\n### Real detail\nBody\n## Two\nEnd\n"
        rows=c.source_blocks(text)
        self.assertEqual([r["title"] for r in rows],["One","Real detail","Two"])
        self.assertEqual("".join(r["markdown"] for r in rows),text)

    def test_no_arbitrary_document_or_computation_route(self):
        for identifier in ["../P1","..%2fsecret","resources","", "P1?path=x"]:
            with self.assertRaises(ValueError):c.lesson(identifier)
        for identifier in ["T0","R1","ML16","DL14"]:
            lab=next(s["lab"] for s in c.lesson(identifier)["steps"] if s["kind"]=="lab")
            self.assertIn(lab["page"],{"train","ml-lab","dl-lab"})
            self.assertNotIn("settings",lab)

    def test_selected_numerical_reasoning(self):
        self.assertEqual(2*3*4*4,96)
        self.assertEqual((12*2+4*4)/16,2.5)
        self.assertEqual(4*8*2,64)
        self.assertEqual(2*1*2*10*2*4*2,640)
        self.assertEqual(100*4+4*200,1200)
        self.assertAlmostEqual(2*1000/1_000_000+4*200/1_000_000,.0028)


if __name__=="__main__":unittest.main()
