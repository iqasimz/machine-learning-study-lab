"""Meaningful NLP reading order, dependency branches and visible calculation inputs."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from training_lab import curriculum as c


class NLPReaderTests(unittest.TestCase):
    def test_diagrams_do_not_conflate_alternative_operations(self):
        p = c.READER_PROFILES
        for identifier in ("N01", "N02", "N13", "N20", "N22", "N23", "N24"):
            self.assertEqual(p[identifier]["view"], "reading")
        self.assertEqual(set(p["N11"]["diagram"]["nodes"]), {0, 1, 2})
        edges = {(a, b) for a, b, *_ in p["N12"]["diagram"]["links"]}
        self.assertEqual(edges, {(0, 1), (1, 2), (0, 3)})
        self.assertNotIn((2, 3), edges, "Normalization does not follow best-path decoding")
        self.assertEqual(p["N08"]["variants"]["1"]["diagram"]["nodes"], [0, 1, 3])
        self.assertEqual(set(p["N15"]["variants"]), {"cross", "late"})
        self.assertEqual(p["N04"]["variants"]["logistic"]["steps"][1]["node"],
                         "Coefficients and bias")

    def test_hmm_crf_inputs_and_logistic_equation_are_visible(self):
        script = r'''
const fs=require('fs'),vm=require('vm'),v=vm.createContext({});
for(const f of ['architecture_math','explorer_lessons','nlp_math','explorer_nlp'])
 vm.runInContext(fs.readFileSync(process.argv[1]+'/'+f+'.js','utf8'),v);
console.log(vm.runInContext(`JSON.stringify({
 taggers:['crf','hmm'].map(kind=>ConceptLessons.build('N12',{kind}).steps[0]),
 logistic:ConceptLessons.build('N04',{model:'logistic'}).steps[1]
})`,v));
'''
        data = json.loads(subprocess.check_output(
            ["node", "-e", script, str(ROOT / "src/training_lab/web")], text=True))
        crf, hmm = data["taggers"]
        for row, expected in ((crf, [[.5, -1.], [-.5, .8]]),
                              (hmm, [[.7, .3], [.4, .6]])):
            table = row["values"][0]
            self.assertIn("rows = previous tag, columns = next tag", table["label"])
            actual = [[float(x) for x in r[1:]] for r in table["rows"]]
            self.assertEqual(actual, expected)
        self.assertEqual([float(x) for x in crf["values"][1]["rows"][0][1:]], [0., 0.])
        self.assertEqual([float(x) for x in hmm["values"][1]["rows"][0][1:]], [.6, .4])
        self.assertIn("natural logs", hmm["calculation"])
        self.assertEqual(data["logistic"]["equation"], "z=w^Tx+b")
        self.assertNotIn("NB counts", data["logistic"]["input"])

    def test_unique_depth_and_source_recoverability(self):
        for identifier in (f"N{i:02}" for i in range(1, 25)):
            doc = c.document(identifier)
            self.assertEqual(len(doc["reader"]["notes"]), 2)
            self.assertGreaterEqual(len(doc["reader"]["sources"]), 2)
        notes = lambda identifier: "\n".join(n["markdown"] for n in c.document(identifier)["reader"]["notes"])
        for identifier, phrase in (("N03", "Read a conventional BM25 term contribution"),
                                   ("N10", "Count the actual framework parameters"),
                                   ("N12", "Reconstruct one Viterbi update"),
                                   ("N13", "A constituency chart is another structure"),
                                   ("N24", "A local first coding exercise")):
            self.assertIn(phrase, notes(identifier))
        repeated = "The example agreement calculation is deliberately small:"
        self.assertIn(repeated, c.document("N01")["markdown"])
        self.assertNotIn(repeated, notes("N01"), "Keep the canonical calculation, not a second copy")
        self.assertIn("B→B→A", notes("N12"))


if __name__ == "__main__":
    unittest.main()
