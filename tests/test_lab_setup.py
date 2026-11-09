"""Setup-source fidelity and new local dataset/model boundaries; no downloads."""
import ast
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from training_lab import model_data, model_lab, source_code
from training_lab.artifacts import ROOT


class LabSetupTests(unittest.TestCase):
    def test_sources_are_exact_fixed_symbols_not_pseudocode(self):
        groups=list(source_code.models().values())+list(source_code.language("lora").values())+list(source_code.language("scratch").values())
        for group in groups:
            for part in group["parts"]:
                source=(ROOT/part["file"]).read_text().splitlines()
                self.assertEqual(part["code"],"\n".join(source[part["first_line"]-1:part["last_line"]]))
                ast.parse(part["code"])
        with self.assertRaises(ValueError):source_code.excerpt("../../secret","load")
        with self.assertRaises(ValueError):source_code.excerpt("model_lab","unknown")
        self.assertIn("loss.backward()",source_code.models()["neural_train"]["parts"][0]["code"])
        self.assertIn("pipeline.fit",source_code.models()["classical_train"]["parts"][0]["code"])

    def test_bundled_new_data_and_training_only_splits(self):
        for name,rows,features in [("breast_cancer",569,30),("diabetes",442,10)]:
            data=model_data.load(name,500)
            self.assertEqual(data["x"].shape,(rows,features))
            self.assertTrue(np.isfinite(data["x"]).all())
            self.assertEqual(data["split_ids"],model_data.load(name,500)["split_ids"])
            a,b,c=[set(data["split_ids"][s]) for s in ["train","dev","test"]]
            self.assertFalse(a&b or a&c or b&c)
        with tempfile.TemporaryDirectory() as temp:
            for model,name,task in [("residual_mlp","breast_cancer","classification"),("residual_mlp","diabetes","regression")]:
                config=dict(area="dl",dataset=name,task=task,model=model,controls=dict(epochs=1,device="cpu",width=16,batch_size=128))
                model_lab.fit(name,config,Path(temp))
                self.assertEqual(model_lab.detail(name,Path(temp))["integrity"],"PASS")

    def test_new_and_old_image_heads_accept_rgb_and_grayscale(self):
        import torch
        torch.set_num_threads(2)
        for shape in ([1,8,8],[1,28,28],[3,32,32]):
            features=int(np.prod(shape))
            for name in ["cnn","residual","lenet","conv_autoencoder"]:
                config=model_lab.validate(dict(area="dl",dataset="cifar10" if shape[0]==3 else "digits",task="representation" if name=="conv_autoencoder" else "classification",model=name,controls={"width":16}),require_ready=False)
                network=model_lab.network(name,config["controls"],features,10,shape)
                output=network(torch.zeros(2,features))
                self.assertEqual(tuple(output.shape),(2,features if name=="conv_autoencoder" else 10))
                output.sum().backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in network.parameters()))
        with self.assertRaises(ValueError):model_lab.validate(dict(area="dl",dataset="diabetes",task="regression",model="lenet",controls={}))

    def test_downloads_are_explicit_and_cifar_never_unpickles_or_extracts(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(model_data,"urlopen") as download:
            for name in ["mnist","cifar10"]:
                self.assertFalse(model_data.ready(name,Path(temp)))
                with self.assertRaises(ValueError):model_data.load(name,500,Path(temp))
            download.assert_not_called()
        raw=np.zeros((10000,3073),dtype=np.uint8)
        raw[:,0]=np.arange(10000)%10;raw[:,1]=10;raw[:,1025]=20;raw[:,2049]=30
        member=SimpleNamespace(size=raw.nbytes,isfile=lambda:True)
        archive=SimpleNamespace(getmember=lambda name:member,extractfile=lambda member:io.BytesIO(raw.tobytes()))
        pixels,labels=model_data._cifar_batch(archive,"test_batch.bin")
        self.assertEqual(pixels.shape,(10000,3072));self.assertEqual(labels[9],9)
        self.assertEqual(pixels[0,[0,1024,2048]].tolist(),[10,20,30])
        with self.assertRaises(ValueError):model_data._cifar_batch(archive,"../../payload")
        member.size=1
        with self.assertRaises(ValueError):model_data._cifar_batch(archive,"test_batch.bin")
        text=(ROOT/"src/training_lab/model_data.py").read_text()
        self.assertNotIn("pickle.load",text);self.assertNotIn("extractall",text)


if __name__=="__main__":unittest.main()
