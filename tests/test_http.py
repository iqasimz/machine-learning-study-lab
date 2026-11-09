"""Opt-in integration tests: LAB_TEST_URL=http://127.0.0.1:8765 python -m unittest ..."""
import json
import os
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE = os.environ.get("LAB_TEST_URL")


@unittest.skipUnless(BASE, "Set LAB_TEST_URL to the running local dashboard for HTTP integration tests")
class HTTPTests(unittest.TestCase):
    def request(self, path, data=None, origin=True, host=None):
        headers = {}
        if data is not None:
            headers["Content-Type"] = "application/json"
            if origin:
                headers["Origin"] = BASE
        if host:
            headers["Host"] = host
        request = Request(BASE + path, data=json.dumps(data).encode() if data is not None else None, headers=headers)
        try:
            with urlopen(request, timeout=10) as response:
                raw = response.read()
                return response.status, json.loads(raw) if "application/json" in response.headers["Content-Type"] else raw.decode()
        except HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def wait_for(self, predicate, seconds=30):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            value=predicate()
            if value:return value
            time.sleep(.1)
        self.fail("Worker timed out")

    def test_01_static_and_list(self):
        code,html=self.request("/")
        self.assertEqual(code,200);self.assertIn("Machine Learning Study Lab",html);self.assertIn("Learn how models work",html)
        self.assertEqual(self.request("/app.js")[0],200)
        for script in ("readouts.js", "architecture_math.js", "architecture_lessons.js", "architecture_explorer.js", "explorer_lessons.js", "explorer_ml.js", "explorer_dl.js", "explorer_llm.js", "explorer_figures.js", "metrics.js", "nlp_math.js", "explorer_nlp.js", "nlp_figures.js"):
            self.assertIn("/"+script,html)
            code,source=self.request("/"+script)
            self.assertEqual(code,200)
            self.assertIn('"use strict"',source)
        code,runs=self.request("/api/runs");self.assertEqual(code,200);self.assertIsInstance(runs,list)

    def test_02_write_origin_and_host(self):
        self.assertEqual(self.request("/api/train",{"preset":"smoke"},origin=False)[0],403)
        self.assertEqual(self.request("/api/runs",host="evil.invalid")[0],403)

    def test_03_reject_unbounded_inputs(self):
        self.assertEqual(self.request("/api/train",{"preset":"huge"})[0],400)
        self.assertEqual(self.request("/api/train",{"preset":"tiny","steps":999999})[0],400)
        self.assertEqual(self.request("/api/run/..%2Fsecret")[0],400)

    def test_04_train_real_worker(self):
        code,job=self.request("/api/train",{"preset":"smoke"})
        self.assertEqual(code,202)
        name=job["id"]
        self.wait_for(lambda:not self.request("/api/job")[1]["running"])
        code,result=self.request("/api/run/"+name)
        self.assertEqual(code,200);self.assertEqual(result["manifest"]["status"],"MEASURED")
        self.assertEqual(result["integrity"],"PASS");self.assertEqual(result["manifest"]["completed_steps"],40)
        code,tokens=self.request("/api/tokenize",{"run":name,"text":"Hi 🧠 <EOS>"})
        self.assertEqual(code,200);self.assertEqual(tokens["round_trip"],"Hi 🧠 <EOS>")

    def test_05_cancel_owned_worker(self):
        code,job=self.request("/api/train",{"preset":"tiny"})
        self.assertEqual(code,202);name=job["id"]
        self.wait_for(lambda:self.request("/api/run/"+name)[0]==200)
        self.assertEqual(self.request("/api/train",{"preset":"smoke"})[0],400)
        code,_=self.request("/api/cancel",{});self.assertEqual(code,200)
        self.wait_for(lambda:not self.request("/api/job")[1]["running"])
        _,result=self.request("/api/run/"+name)
        self.assertEqual(result["manifest"]["status"],"CANCELLED")
        self.assertNotIn("evaluation",result)

    def test_06_guided_study_comparison_generation_and_journal(self):
        code,catalog=self.request("/api/catalog")
        self.assertEqual(code,200);self.assertEqual(len(catalog["guides"]),20)
        self.assertIn("code",catalog["parameters"]["learning_rate"]["source"])
        payload={"guide":"rate","variation":"all","overrides":{"steps":20,"width":32,"layers":1,"block_size":16},
                 "prediction":"I predict a faster early drop, but not a guaranteed dev gain."}
        code,preview=self.request("/api/preview",payload)
        self.assertEqual(code,200);self.assertEqual(len(preview["runs"]),3)
        code,job=self.request("/api/guided",payload)
        self.assertEqual(code,202);study_id=job["study"]["id"]
        self.wait_for(lambda:not self.request("/api/job")[1]["running"])
        _,studies=self.request("/api/studies")
        study=next(s for s in studies if s["id"]==study_id)
        self.assertEqual(study["status"],"MEASURED")
        ids=[r["id"] for r in study["runs"]]
        code,comparison=self.request("/api/compare",{"runs":ids})
        self.assertEqual(code,200);self.assertTrue(comparison["nll_comparable"])
        self.assertEqual([c["parameter"] for c in comparison["changes"]],["learning_rate"])
        for row in comparison["runs"]:
            self.assertEqual(row["manifest"]["learning"]["prediction"],payload["prediction"])
        code,_=self.request("/api/reflection",{"study":study_id,"reflection":"The aggressive rate was not automatically the best."})
        self.assertEqual(code,200)
        code,result=self.request("/api/generate",{"run":ids[0],"prompt":"The model","temperature":0,
                                                  "top_k":20,"tokens":16,"seed":17})
        self.assertEqual(code,200);self.assertEqual(result["device"],"cpu");self.assertIn("completion",result)
        code,export=self.request("/api/export",{"run":ids[0]})
        self.assertEqual(code,200);self.assertFalse(export["assistant_approved"])
        code,lesson=self.request("/api/lesson/optimization")
        self.assertEqual(code,200);self.assertGreater(len(lesson["markdown"]),100)

    def test_07_editable_workbench_and_repeated_resume(self):
        code,job=self.request("/api/workbench",{"settings":{"steps":20,"width":32,"layers":1,"block_size":16,
                                                            "normalization":"rms","positions":False,"dropout":0},
                                               "prediction":"Check the actual ablated configuration."})
        self.assertEqual(code,202);run=job["study"]["runs"][0]["id"]
        self.wait_for(lambda:not self.request("/api/job")[1]["running"])
        code,result=self.request("/api/run/"+run)
        self.assertEqual(code,200);self.assertEqual(result["manifest"]["config"]["model"]["normalization"],"rms")
        self.assertTrue(result["can_resume"])
        for expected in [60,100]:
            code,_=self.request("/api/resume",{"run":run,"additional_steps":40})
            self.assertEqual(code,202)
            self.wait_for(lambda:not self.request("/api/job")[1]["running"])
            code,result=self.request("/api/run/"+run)
            self.assertEqual(result["manifest"]["status"],"MEASURED")
            self.assertEqual(result["manifest"]["completed_steps"],expected)
        code,history=self.request("/api/history/"+run)
        self.assertEqual(code,200)
        self.assertEqual([s["manifest"]["completed_steps"] for s in history],[20,60])

    def test_08_reject_invalid_guided_and_generation_settings(self):
        bad={"guide":"rate","variation":"all","overrides":{"steps":True},"prediction":""}
        self.assertEqual(self.request("/api/guided",bad)[0],400)
        self.assertEqual(self.request("/api/workbench",{"settings":{"device":"unknown"},"prediction":""})[0],400)
        self.assertEqual(self.request("/api/lesson/..%2Fsecret")[0],400)

    def test_09_integrated_course_and_practice(self):
        code,index=self.request("/api/learning")
        self.assertEqual(code,200);self.assertEqual(len(index["lessons"]),174);self.assertEqual(len(index["questions"]),62)
        code,doc=self.request("/api/learning/document/T3")
        self.assertEqual(code,200);self.assertIn("attention",doc["markdown"].lower());self.assertGreater(len(doc["sections"]),1)
        self.assertEqual(len(doc["sha256"]),64)
        code,q=self.request("/api/learning/interview/F01")
        self.assertEqual(code,200);self.assertGreater(len(q["markdown"]),100)
        self.assertEqual(self.request("/api/learning/document/..%2Fsecret")[0],400)
        self.assertEqual(self.request("/api/learning/interview/unknown")[0],400)
        self.assertEqual(self.request("/learning.js")[0],200)

    def test_10_static_reader_and_strict_policy(self):
        self.assertEqual(self.request("/learning-visuals")[0],404)
        with urlopen(BASE+"/") as response:
            self.assertNotIn("unsafe-inline",response.headers["Content-Security-Policy"])
            page=response.read().decode()
            self.assertIn('/vendor/katex/katex.min.js',page)
            self.assertNotIn('/spatial.js',page)
            self.assertNotIn('id="home-visual"',page)

    def test_11_local_math_visuals_and_language_catalog(self):
        self.assertEqual(self.request("/visuals.js")[0],200)
        self.assertEqual(self.request("/vendor/katex/katex.min.js")[0],200)
        self.assertEqual(self.request("/vendor/katex/../../server.py")[0],404)
        code,language=self.request("/api/adaptation")
        self.assertEqual(code,200);self.assertEqual(language["assets"]["parameters"],361821120)
        self.assertEqual(self.request("/api/adapt",{"settings":{"rank":999}})[0],400)
        self.assertEqual(self.request("/api/trace",{"run":"../x","text":"a"})[0],400)

    def test_12_removed_spatial_and_update_bounds(self):
        for path in ["/spatial.js","/spatial-models.js","/vendor/three/three.module.js"]:
            self.assertEqual(self.request(path)[0],404)
        self.assertEqual(self.request("/api/trace-update",{"run":"../x","text":"a"})[0],400)
        self.assertEqual(self.request("/api/trace-update",{"run":"x","text":"a"},origin=False)[0],403)

    def test_13_model_api_bounds_and_assets(self):
        code,catalog=self.request("/api/models/catalog")
        self.assertEqual(code,200);self.assertEqual(len(catalog["methods"]),37)
        for file in ["/ml_visuals.js","/models.js","/lab_setup.js"]:self.assertEqual(self.request(file)[0],200)
        for payload in [{"settings":{}},{"settings":{"model":"logistic","command":"ls"}}]:
            self.assertEqual(self.request("/api/models/train",payload)[0],400)
        self.assertEqual(self.request("/api/models/train",{"settings":{}},origin=False)[0],403)
        self.assertEqual(self.request("/api/models/prepare",{"dataset":"https://evil.invalid/x"})[0],400)
        self.assertEqual(self.request("/api/models/run/..%2Fsecret")[0],400)
        self.assertEqual(self.request("/model_runs/ml-digits-logistic-demo/estimator.joblib")[0],404)
        for identifier in ["ML16","DL14"]:
            code,doc=self.request("/api/learning/document/"+identifier)
            self.assertEqual(code,200);self.assertTrue(doc["parameters"])
        self.assertEqual(self.request("/api/learning/interview/IML01")[0],200)
        self.assertEqual(self.request("/api/learning/interview/IDL12")[0],200)

    def test_14_real_classical_jobs_and_matched_compare(self):
        identifiers=[]
        for method in ["logistic","tree"]:
            settings={"area":"ml","dataset":"iris","task":"classification","model":method,"controls":{}}
            code,job=self.request("/api/models/train",{"settings":settings});self.assertEqual(code,202)
            identifiers.append(job["id"])
            self.wait_for(lambda:not self.request("/api/job")[1]["running"])
            code,run=self.request("/api/models/run/"+job["id"])
            self.assertEqual(code,200);self.assertEqual(run["manifest"]["status"],"MEASURED")
            self.assertEqual(run["integrity"],"PASS");self.assertIn("macro_f1",run["result"]["predictions"]["dev"]["scores"])
        code,comparison=self.request("/api/models/compare",{"runs":identifiers})
        self.assertEqual(code,200);self.assertEqual(len(comparison["rows"]),2)

    def test_16_guided_courses_are_read_only_and_allowlisted(self):
        self.assertEqual(self.request("/course.js")[0],200)
        code,index=self.request("/api/course/catalog")
        self.assertEqual(code,200);self.assertEqual(len(index["units"]),174)
        self.assertEqual([p["id"] for p in index["paths"]],["basics","python","ml","dl","nlp","llm"])
        self.assertEqual([len(p["lessons"]) for p in index["paths"]],[5,95,16,14,26,18])
        code,deck=self.request("/api/course/lesson/T3")
        self.assertEqual(code,200)
        self.assertEqual(len([s for s in deck["steps"] if s["kind"]=="question"]),2)
        self.assertIn("attention",json.dumps(deck).lower())
        self.assertEqual(self.request("/api/course/lesson/..%2Fsecret")[0],400)
        self.assertEqual(self.request("/api/course/lesson/unknown")[0],400)
        self.assertEqual(self.request("/api/course/catalog",{})[0],400)

    def test_15_neural_worker_cancellation_is_shared(self):
        settings={"area":"dl","dataset":"digits","task":"classification","model":"residual","controls":{"epochs":30,"width":128}}
        code,job=self.request("/api/models/train",{"settings":settings});self.assertEqual(code,202)
        self.wait_for(lambda:self.request("/api/models/run/"+job["id"])[0]==200)
        self.assertEqual(self.request("/api/train",{"preset":"smoke"})[0],400)
        self.assertEqual(self.request("/api/models/train",{"settings":settings})[0],400)
        self.assertEqual(self.request("/api/cancel",{})[0],200)
        self.wait_for(lambda:not self.request("/api/job")[1]["running"])
        code,run=self.request("/api/models/run/"+job["id"])
        self.assertEqual(code,200);self.assertEqual(run["manifest"]["status"],"CANCELLED")
        self.assertNotIn("result",run)


if __name__=="__main__":unittest.main()
