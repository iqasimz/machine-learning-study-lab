import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import torch
from training_lab.artifacts import read_json,write_json
from training_lab.comparison import compare_runs
from training_lab.experiments import DEFAULTS,GUIDES,PARAMETERS,catalog,estimate,lesson,make_plan,validate
from training_lab.model import ModelConfig,TinyDecoder
from training_lab.studies import create_study,custom_plan,execute_study
from training_lab.training import learning_rate_at,make_optimizer,train

torch.set_num_threads(1)


class RegistryTests(unittest.TestCase):
    def test_all_guides_validate(self):
        self.assertEqual(len(GUIDES),20)
        for guide in GUIDES:
            plan=make_plan(guide["id"])
            self.assertTrue(1<=len(plan["runs"])<=3)
            for row in plan["runs"]:self.assertEqual(validate(row["settings"]),row["settings"])

    def test_numeric_json_types_and_bool(self):
        self.assertEqual(validate({"dropout":0,"train_fraction":1})["dropout"],0.0)
        for key,value in [("steps",True),("positions",1),("learning_rate",float("nan")),("width",63),("shell","anything")]:
            with self.assertRaises(ValueError):validate({key:value})

    def test_combination_budget(self):
        with self.assertRaises(ValueError):validate({"warmup_steps":40,"schedule_steps":80,"steps":800,"width":128,"layers":4,"block_size":128,"batch_size":16,"accumulation":4})

    def test_plan_prediction_and_single_variation(self):
        p=make_plan("rate",1,{"steps":20},"I expect slower fitting.")
        self.assertEqual(len(p["runs"]),1)
        self.assertEqual(p["runs"][0]["settings"]["learning_rate"],.003)
        self.assertEqual(p["prediction"],"I expect slower fitting.")
        with self.assertRaises(ValueError):make_plan("rate",False)

    def test_catalog_actual_source_and_lessons(self):
        c=catalog()
        self.assertEqual(set(c["parameters"]),set(PARAMETERS))
        for p in c["parameters"].values():
            self.assertIn("line",p["source"])
            self.assertTrue(p["effects"] and p["interactions"])
        for key in c["lessons"]:self.assertGreater(len(lesson(key)["markdown"]),100)
        with self.assertRaises(ValueError):lesson("../secret")

    def test_estimator_is_upper_bound(self):
        e=estimate(DEFAULTS)
        self.assertEqual(e["maximum_token_exposures"],120*4*64)
        self.assertEqual(e["effective_windows_per_update"],4)


class MechanismTests(unittest.TestCase):
    def test_schedule_warmup_and_floor(self):
        c={**DEFAULTS,"schedule":"cosine","warmup_steps":10,"schedule_steps":80}
        self.assertAlmostEqual(learning_rate_at(1,c),.0003)
        self.assertAlmostEqual(learning_rate_at(10,c),.003)
        self.assertAlmostEqual(learning_rate_at(80,c),.0003)
        self.assertAlmostEqual(learning_rate_at(100,c),.0003)

    def test_supported_ablations_are_real(self):
        for norm in ["layer","rms","none"]:
            m=TinyDecoder(ModelConfig(259,width=16,heads=2,layers=2,positions=False,residuals=False,normalization=norm))
            self.assertIsNone(m.positions)
            logits=m(torch.tensor([[1,4,5]]))
            self.assertTrue(torch.isfinite(logits).all())
            self.assertEqual(logits.shape,(1,3,259))

    def test_optimizer_selection(self):
        m=TinyDecoder(ModelConfig(259,width=16,heads=2,layers=1))
        self.assertIsInstance(make_optimizer(m,{**DEFAULTS,"optimizer":"sgd"}),torch.optim.SGD)
        self.assertIsInstance(make_optimizer(m,DEFAULTS),torch.optim.AdamW)


class StudyTests(unittest.TestCase):
    def test_serial_study_runs_and_preserves_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            plan=make_plan("positions","all",{"steps":20,"width":32,"layers":1,"block_size":16},"Removing positions may hurt.")
            record=create_study(plan,root/"studies")
            result=execute_study(record["id"],root/"studies",root/"runs")
            self.assertEqual(result["status"],"MEASURED")
            self.assertEqual(result["completed_runs"],2)
            ids=[row["id"] for row in result["runs"]]
            report=compare_runs(ids,root/"runs")
            self.assertTrue(report["nll_comparable"])
            self.assertEqual([c["parameter"] for c in report["changes"]],["positions"])
            self.assertEqual(read_json(root/"runs"/ids[0]/"manifest.json")["learning"]["prediction"],plan["prediction"])
            self.assertIn("bits_per_byte",report["runs"][0]["evaluation"]["transformer"]["test"])

    def test_cancelled_study_does_not_invent_remaining_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);record=create_study(make_plan("rate"),root/"studies")
            with patch("training_lab.training.train",side_effect=KeyboardInterrupt("test cancellation")):
                with self.assertRaises(KeyboardInterrupt):execute_study(record["id"],root/"studies",root/"runs")
            saved=read_json((root/"studies"/record["id"]).with_suffix(".json"))
            self.assertEqual(saved["status"],"CANCELLED")
            self.assertEqual([r["status"] for r in saved["runs"]],["CANCELLED","NOT_RUN","NOT_RUN"])

    def test_tokenizer_units_cannot_be_ranked_as_nll(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name,merges in [("bytes",0),("bpe",32)]:
                train(name,steps=2,overrides={**DEFAULTS,"merges":merges},runs_root=root)
            report=compare_runs(["bytes","bpe"],root)
            self.assertFalse(report["nll_comparable"])
            self.assertTrue(report["bits_per_byte_comparable"])
            self.assertTrue(any("different units" in w for w in report["warnings"]))

    def test_train_fraction_keeps_eval_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            train("quarter",steps=1,overrides={**DEFAULTS,"train_fraction":.25},runs_root=root)
            data=read_json(root/"quarter/data.json")
            self.assertEqual(data["counts"],{"train":4,"dev":4,"test":4})


if __name__=="__main__":unittest.main()
