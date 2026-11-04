"""Cross-check browser arithmetic against independent sklearn/numpy calculations."""
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from sklearn import metrics

WEB=Path(__file__).resolve().parents[1]/"src/training_lab/web"
sys.path.insert(0,str(WEB.parent.parent))
from training_lab import curriculum


class MetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script="""
const fs=require('fs'),vm=require('vm');const c=vm.createContext({});
for(const f of ['architecture_math','explorer_lessons','metrics'])vm.runInContext(fs.readFileSync(process.argv[1]+'/'+f+'.js','utf8'),c);
console.log(vm.runInContext('JSON.stringify({binary:[.3,.5,.7,.95].map(t=>MetricsLab.binary(t)),multiclass:MetricsLab.multiclass(),regression:MetricsLab.regression(),language:[MetricsLab.language("all"),MetricsLab.language("response")],retrieval:[1,2,3].map(k=>MetricsLab.retrieval(k))})',c));
"""
        cls.results=json.loads(subprocess.check_output(["node","-e",script,str(WEB)],text=True))

    def test_binary_thresholds_and_zero_division(self):
        y=np.array([1,0,1,1,0,0]);scores=np.array([.9,.8,.6,.4,.3,.1])
        for t,r in zip([.3,.5,.7,.95],self.results["binary"]):
            pred=scores>=t
            tn,fp,fn,tp=metrics.confusion_matrix(y,pred).ravel()
            self.assertEqual([r[k] for k in ["tn","fp","fn","tp"]],[tn,fp,fn,tp])
            self.assertAlmostEqual(r["accuracy"],metrics.accuracy_score(y,pred))
            self.assertAlmostEqual(r["recall"],metrics.recall_score(y,pred))
            self.assertAlmostEqual(r["f1"],metrics.f1_score(y,pred))
            if pred.any():self.assertAlmostEqual(r["precision"],metrics.precision_score(y,pred))
            else:self.assertIsNone(r["precision"])
            self.assertAlmostEqual(r["rocAuc"],metrics.roc_auc_score(y,scores))
            self.assertAlmostEqual(r["averagePrecision"],metrics.average_precision_score(y,scores))

    def test_class_averaging(self):
        r=self.results["multiclass"];truth=[];pred=[]
        for i,row in enumerate(r["matrix"]):
            for j,count in enumerate(row):truth.extend([i]*count);pred.extend([j]*count)
        for kind,key in [("macro","macroF1"),("weighted","weightedF1"),("micro","microF1")]:
            self.assertAlmostEqual(r[key],metrics.f1_score(truth,pred,average=kind))
        self.assertAlmostEqual(r["balancedAccuracy"],metrics.balanced_accuracy_score(truth,pred))
        self.assertAlmostEqual(r["accuracy"],metrics.accuracy_score(truth,pred))

    def test_regression_units(self):
        r=self.results["regression"];y=r["truth"];p=r["prediction"]
        self.assertAlmostEqual(r["mae"],metrics.mean_absolute_error(y,p))
        self.assertAlmostEqual(r["mse"],metrics.mean_squared_error(y,p))
        self.assertAlmostEqual(r["rmse"],math.sqrt(metrics.mean_squared_error(y,p)))
        self.assertAlmostEqual(r["r2"],metrics.r2_score(y,p))

    def test_language_masks_and_byte_alignment(self):
        for start,r in zip([0,1],self.results["language"]):
            losses=-np.log([.5,.25,.1])[start:]
            self.assertAlmostEqual(r["nll"],float(losses.mean()))
            self.assertAlmostEqual(r["perplexity"],math.exp(losses.mean()))
            self.assertEqual(r["count"],len(losses))
            if start:self.assertIsNone(r["bitsPerByte"])
            else:self.assertAlmostEqual(r["bitsPerByte"],losses.sum()/math.log(2)/12)

    def test_retrieval_prefix_and_full_ranking(self):
        relevance=np.array([[0,1,1]]);scores=np.array([[3,2,1]])
        for k,r in enumerate(self.results["retrieval"],1):
            hits=int(relevance[0,:k].sum())
            self.assertAlmostEqual(r["precision"],hits/k)
            self.assertAlmostEqual(r["recall"],hits/2)
            self.assertAlmostEqual(r["averagePrecision"],metrics.average_precision_score(relevance[0],scores[0]))
            self.assertAlmostEqual(r["ndcg"],metrics.ndcg_score(relevance,scores,k=3))
            self.assertEqual(r["reciprocalRank"],.5)

    def test_reader_keeps_depth_and_single_parameter_ownership(self):
        docs=[curriculum.document(row[0]) for row in curriculum.LESSONS]
        parameters=[p for d in docs for n in d["reader"]["notes"] for p in n.get("parameter_ids",[])]
        self.assertEqual(len(parameters),len(set(parameters)))
        self.assertEqual(set(parameters),{p["id"] for d in docs for p in d.get("parameters",[])})
        for d in docs:
            self.assertEqual(d["reader"]["original_sha256"],d["sha256"])
            self.assertTrue(d["reader"]["sources"])
            for n in d["reader"]["notes"]:
                self.assertNotIn("**Check.**",n["markdown"])
                self.assertNotIn("Before considering this lesson complete",n["markdown"])
        t3=curriculum.document("T3")["reader"]
        self.assertTrue(any("actual local grouped-query" in n["title"] for n in t3["notes"]))
        self.assertIn("P8",curriculum.document("R4")["reader"]["related"])


if __name__=="__main__":unittest.main()
