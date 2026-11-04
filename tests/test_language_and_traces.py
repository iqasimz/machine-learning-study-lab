import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from training_lab import adaptation as a
from training_lab.assets import status
from training_lab.artifacts import read_json,write_json,verify,seal,sha256
from training_lab.experiments import DEFAULTS
from training_lab.training import train
from training_lab.traces import capture,capture_update


class LanguageTests(unittest.TestCase):
    def test_validated_bounded_settings(self):
        self.assertEqual(a.validate({"rank":4.0})["rank"],4)
        for value in [{"steps":True},{"rank":64},{"device":"unknown"},{"shell":"anything"},{"learning_rate":float("nan")}]:
            with self.subTest(value=value),self.assertRaises(ValueError):a.validate(value)

    def test_prompt_mask_keeps_real_eos(self):
        class Tokenizer:
            def apply_chat_template(self,messages,tokenize,add_generation_prompt):
                return [1,2,3] if len(messages)==1 else [1,2,3,7,8,9,10,0]
        item=a.encode_example({"id":"x","group":"g","prompt":"q","answer":"a"},Tokenizer(),16)
        self.assertEqual(item["labels"],[-100,-100,-100,7,8,9,10,0])

    def test_template_mismatch_rejected(self):
        class Tokenizer:
            def apply_chat_template(self,messages,**kw):return [1,2] if len(messages)==1 else [7,8,9,10,11,12]
        with self.assertRaises(ValueError):a.encode_example({"id":"x","group":"g","prompt":"q","answer":"a"},Tokenizer(),16)

    @unittest.skipUnless(status()["ready"],"Prepare pinned local model/data for the dataset audit")
    def test_real_grouped_splits_and_provenance(self):
        from transformers import AutoTokenizer
        from training_lab.assets import MODEL_DIR
        tokenizer=AutoTokenizer.from_pretrained(MODEL_DIR,local_files_only=True,trust_remote_code=False)
        items,docs=a.prepared_data(tokenizer,a.validate({"block_size":128}))
        self.assertTrue(all(len(v)>=4 for v in items.values()))
        groups={s:{d["group"] for d in docs if d["split"]==s} for s in items}
        self.assertFalse(groups["train"]&groups["dev"]);self.assertFalse(groups["train"]&groups["test"]);self.assertFalse(groups["dev"]&groups["test"])
        self.assertTrue(all(-100 in item["labels"] and any(x!=-100 for x in item["labels"]) for rows in items.values() for item in rows))

    def test_real_scratch_trace_is_causal_and_read_only(self):
        with tempfile.TemporaryDirectory() as temp:
            train("trace",steps=1,runs_root=temp,overrides={**DEFAULTS,"width":32,"layers":1,"block_size":16})
            before=read_json(Path(temp)/"trace/hashes.json")
            trace=capture("trace","A model",runs_root=temp)
            self.assertEqual(trace["kind"],"measured")
            for i,row in enumerate(trace["attention"]):
                self.assertAlmostEqual(sum(row),1,places=5)
                self.assertTrue(all(v==0 for v in row[i+1:]))
            self.assertEqual(before,read_json(Path(temp)/"trace/hashes.json"));verify(Path(temp)/"trace")
            json.dumps(trace,allow_nan=False)

    def test_update_trace_is_measured_and_does_not_change_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            train("step-trace",steps=1,runs_root=temp,overrides={**DEFAULTS,"width":32,"layers":1,"block_size":16})
            directory=Path(temp)/"step-trace"
            before={p.name:p.read_bytes() for p in directory.iterdir() if p.is_file()}
            trace=capture_update("step-trace","A model learns",runs_root=temp)
            t=trace["training_update"]
            self.assertGreater(t["gradient_norm"],0)
            for old_row,g_row,new_row in zip(t["before"],t["gradient"],t["after"]):
                for old,g,new in zip(old_row,g_row,new_row):self.assertAlmostEqual(new,old-.01*g,places=7)
            self.assertTrue(t["loss_after"]<t["loss_before"])
            self.assertEqual(before,{p.name:p.read_bytes() for p in directory.iterdir() if p.is_file()})
            verify(directory);json.dumps(trace,allow_nan=False)

    @unittest.skipUnless(status()["ready"],"Prepare pinned local weights for the real projection-hook check")
    def test_real_lora_projection_hooks(self):
        import gc
        from peft import LoraConfig,get_peft_model
        base,_=a.load_base("cpu")
        model=get_peft_model(base,LoraConfig(r=4,lora_alpha=8,target_modules=["q_proj","v_proj"],task_type="CAUSAL_LM"))
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)/"projection-fixture"
            model.save_pretrained(directory/"adapter",safe_serialization=True)
            write_json(directory/"manifest.json",{"status":"MEASURED","engine":"lora","note":"Untrained adapter test fixture, not research evidence."})
            write_json(directory/"adapter_files.json",{str(p.relative_to(directory)):sha256(p) for p in (directory/"adapter").iterdir() if p.suffix in {".json",".safetensors"}})
            seal(directory);del model,base;gc.collect()
            trace=capture("projection-fixture","The model learns.",runs_root=temp)
            for key in ["query","key","value"]:
                self.assertEqual(len(trace[key]),len(trace["ids"]))
                self.assertTrue(all(len(row)==8 for row in trace[key]))
            self.assertEqual(trace["model_config"]["num_key_value_heads"],5)
            self.assertIn("before rotary",trace["projection_note"])
            json.dumps(trace,allow_nan=False);verify(directory)


if __name__=="__main__":unittest.main()
