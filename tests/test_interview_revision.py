"""Realistic prompts and self-contained examples, without changing stable item IDs."""
import math
from pathlib import Path
import re
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from training_lab import curriculum as c


class InterviewRevisionTests(unittest.TestCase):
    def test_all_existing_items_remain(self):
        expected=set()
        for prefix,count in (("F",16),("M",16),("S",12),("D",6),("P",16),("IML",12),("IDL",12),("INLP",12)):
            expected.update(f"{prefix}{i:02}" for i in range(1,count+1))
        questions=c.catalog()["questions"]
        self.assertEqual({q["id"] for q in questions},expected)
        for q in questions:
            body=c.interview(q["id"])["markdown"]
            self.assertGreater(len(body.split()),65,q["id"])
            self.assertNotRegex(body,r"\bGate:|J-104|\bA0[1-9]/|\bL0[1-9]/")
        self.assertEqual(len({q["title"] for q in questions}),102)

    def test_nlp_answers_explain_and_reference_their_own_case(self):
        for i in range(1,13):
            body=c.interview(f"INLP{i:02}")["markdown"]
            self.assertIn("**Follow-up.**",body)
            self.assertIn("https://",body)
            self.assertNotIn("the course uses constant",body)
        self.assertIn("[A,B,C]",c.interview("IDL09")["markdown"])
        self.assertIn("inference-only",c.interview("IDL11")["markdown"])

    def test_practical_examples_include_interpretable_answers(self):
        self.assertIn("[a,b]",c.interview("P02")["markdown"])
        probs=[math.exp(-1)/(1+math.exp(-1)),1/(1+math.exp(-1))]
        self.assertAlmostEqual(probs[0],.26894142137)
        self.assertAlmostEqual((10*2+90*4)/100,3.8)
        self.assertIn("3.8",c.interview("P05")["markdown"])
        self.assertEqual(100+9*20,280);self.assertEqual(50+9*40,410)
        self.assertIn("280 ms",c.interview("P13")["markdown"])
        self.assertAlmostEqual(2*9/(2*9+9+1),.642857142857)
        self.assertIn("0.6429",c.interview("F14")["markdown"])


if __name__=="__main__":unittest.main()
