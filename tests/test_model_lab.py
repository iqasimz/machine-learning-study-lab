"""Actual CPU fitting and protocol invariants; temporary artifacts, no downloads."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from training_lab import model_data as data, model_lab as lab
from training_lab.artifacts import verify


def config(model="logistic",dataset="digits",task="classification",area="ml",**controls):
    return dict(area=area,dataset=dataset,task=task,model=model,controls=controls)


class ModelLabTests(unittest.TestCase):
    def test_fixed_splits_and_data_identity(self):
        a=data.load("digits",500);b=data.load("digits",500)
        self.assertEqual(a["info"]["fingerprint"],b["info"]["fingerprint"])
        self.assertEqual(a["split_ids"],b["split_ids"])
        for x,y in [("train","dev"),("train","test"),("dev","test")]:
            self.assertFalse(set(a["split_ids"][x])&set(a["split_ids"][y]))
        larger=data.load("digits",1500)
        self.assertEqual(a["split_ids"]["dev"],larger["split_ids"]["dev"])
        self.assertEqual(a["split_ids"]["test"],larger["split_ids"]["test"])
        self.assertNotEqual(a["info"]["split_fingerprint"],larger["info"]["split_fingerprint"])
        self.assertEqual(len(a["indices"]["train"]),500)

    def test_catalog_and_strict_validation(self):
        catalog=lab.catalog()
        self.assertEqual(len(catalog["methods"]),37)
        self.assertEqual(len(catalog["datasets"]),15)
        self.assertEqual(lab.validate(config(C=1))["controls"]["C"],1)
        invalid=[config(C=True),config(C=float("nan")),config(C=999),config(unknown=2),
                 config("mlp",area="ml"),config("cnn","wine",area="dl"),config("pca","iris","representation",components=10),
                 config("ridge","digits","regression"),config("lasso","curve","regression",strength=0),
                 config("isolation","iris","anomaly"),{**config(),"command":"rm"},config(dataset="../x")]
        for item in invalid:
            with self.subTest(item=item),self.assertRaises(ValueError):lab.validate(item)

    def test_all_classical_method_paths_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for model in ["dummy","logistic","knn","svm","naive_bayes","tree","forest","boost"]:
                with self.subTest(model=model):
                    manifest=lab.fit(model,config(model,"iris",train_limit=500),root)
                    self.assertEqual(manifest["status"],"MEASURED")
                    result=lab.detail(model,root)
                    self.assertEqual(result["integrity"],"PASS")
                    self.assertEqual(len(result["result"]["predictions"]["test"]["truth"]),30)
                    self.assertIn("macro_f1",result["result"]["predictions"]["dev"]["scores"])
            for model in ["dummy","ridge","lasso","elastic","svm","tree","forest","boost"]:
                with self.subTest(model=model):
                    name="reg-"+model;lab.fit(name,config(model,"curve","regression",train_limit=500),root)
                    self.assertIn("mae",lab.detail(name,root)["result"]["predictions"]["test"]["scores"])
            for model,task in [("kmeans","clustering"),("gmm","clustering"),("dbscan","clustering"),("pca","representation"),("isolation","anomaly")]:
                with self.subTest(model=model):
                    lab.fit(model,config(model,"clusters",task,train_limit=500),root)
                    self.assertNotIn("predictions",lab.detail(model,root)["result"])
                    verify(root/model)

    def test_real_neural_architectures_and_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            for model in ["linear_head","mlp","cnn","residual","autoencoder","residual_mlp","lenet","conv_autoencoder"]:
                task="representation" if model in {"autoencoder","conv_autoencoder"} else "classification"
                with self.subTest(model=model):
                    settings=config(model,"digits",task,"dl",train_limit=500,epochs=1,batch_size=128)
                    lab.fit(model,settings,Path(tmp));run=lab.detail(model,Path(tmp))
                    self.assertEqual(run["integrity"],"PASS")
                    self.assertEqual(len(run["history"]),1)
                    self.assertTrue(np.isfinite(run["history"][0]["dev_loss"]))
                    self.assertGreater(run["result"]["parameters"],10)
                    self.assertTrue((Path(tmp)/model/"weights.pt").is_file())
                    self.assertTrue(list((Path(tmp)/model/"tensorboard").glob("events*")))
            lab.fit("numeric",config("mlp","curve","regression","dl",train_limit=500,epochs=1),Path(tmp))
            self.assertIn("mae",lab.detail("numeric",Path(tmp))["result"]["predictions"]["test"]["scores"])

    def test_shared_neural_normalization_activation_and_schedule(self):
        import torch
        for norm in ["none","layer","batch"]:
            for name in ["mlp","cnn","residual"]:
                with self.subTest(norm=norm,model=name):
                    settings=lab.validate(config(name,"digits",area="dl",normalization=norm,activation="gelu",width=16))
                    model=lab.network(name,settings["controls"],64,10,[1,8,8])
                    output=model(torch.zeros(4,64));self.assertEqual(tuple(output.shape),(4,10))
                    output.sum().backward()
        with tempfile.TemporaryDirectory() as tmp:
            lab.fit("schedule",config("mlp","digits",area="dl",epochs=3,train_limit=500,normalization="batch",batch_size=256,schedule="cosine",optimizer="sgd",learning_rate=.01),Path(tmp))
            rows=lab.detail("schedule",Path(tmp))["history"]
            self.assertGreater(rows[0]["learning_rate"],rows[-1]["learning_rate"])

    def test_matched_comparison_integrity_and_immutability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            lab.fit("a",config("logistic","iris",C=1),root)
            lab.fit("b",config("tree","iris"),root)
            self.assertEqual(len(lab.compare(["a","b"],root)["rows"]),2)
            with self.assertRaises(ValueError):lab.fit("a",config(),root)
            with self.assertRaises(ValueError):lab.compare(["a","a"],root)
            lab.fit("c",config("logistic","wine"),root)
            with self.assertRaises(ValueError):lab.compare(["a","c"],root)
            (root/"a"/"result.json").write_text("{}")
            self.assertTrue(lab.detail("a",root)["integrity"].startswith("FAIL"))
            with self.assertRaises(ValueError):lab.compare(["a","b"],root)

    def test_neural_checkpoint_loads_without_arbitrary_code(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);lab.fit("n",config("mlp","digits",area="dl",epochs=1,train_limit=500),root)
            saved=torch.load(root/"n"/"weights.pt",map_location="cpu",weights_only=True)
            model=lab.network(saved["settings"]["model"],saved["settings"]["controls"],saved["features"],saved["outputs"],saved["image_shape"])
            model.load_state_dict(saved["state_dict"])
            self.assertEqual(tuple(model(torch.zeros(2,64)).shape),(2,10))

    def test_metric_definitions_and_bootstrap(self):
        y=np.array([0,0,1,1]);pred=np.array([0,1,1,1])
        result=lab.scores(y,pred,"classification",labels=[0,1])
        self.assertEqual(result["accuracy"],.75)
        self.assertAlmostEqual(result["macro_f1"],(2/3+.8)/2)
        ci=lab.interval(y,pred,"classification")
        self.assertLessEqual(ci["low"],.75);self.assertGreaterEqual(ci["high"],.75)
        grouped=lab.interval(y,pred,"classification",groups=np.array([0,0,1,1]))
        self.assertEqual(grouped["resampling_unit"],"duplicate-feature group")

    @unittest.skipUnless(data.ready("fashion_mnist") and data.ready("wine_red"),"Public datasets are explicitly prepared, never downloaded by a test")
    def test_prepared_public_counts_group_boundary_and_official_test(self):
        for name,count in [("wine_red",1599),("wine_white",4898)]:
            record=data.load(name);self.assertEqual(record["info"]["rows"],count)
            for first,second in [("train","dev"),("train","test"),("dev","test")]:
                a=set(record["groups"][record["indices"][first]])
                b=set(record["groups"][record["indices"][second]])
                self.assertFalse(a&b,"Identical feature rows must not cross evaluation boundaries")
        record=data.load("fashion_mnist",500)
        self.assertEqual(record["info"]["image_shape"],[1,28,28])
        self.assertEqual(record["info"]["rows"],70000)
        self.assertLess(record["indices"]["dev"].max(),60000)
        self.assertGreaterEqual(record["indices"]["test"].min(),60000)
        self.assertEqual(len(record["indices"]["test"]),10000)


if __name__=="__main__":unittest.main()
