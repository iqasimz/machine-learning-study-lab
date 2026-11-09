"""Independent numerical checks for authored NLP examples, not model-quality tests."""
import itertools
import contextlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from training_lab import curriculum, guided_course


class NLPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script="""
const fs=require('fs'),vm=require('vm');const c=vm.createContext({});
for(const f of ['architecture_math','nlp_math'])vm.runInContext(fs.readFileSync(process.argv[1]+'/'+f+'.js','utf8'),c);
const expression=`(()=>{const recurrent=[];for(const cell of ['rnn','lstm','gru'])for(const directions of [1,2])for(const setting of cell==='lstm'?[.5,.8,.95]:cell==='gru'?[0,.5,1]:[.8])recurrent.push({cell,directions,setting,result:NLPMath.recurrent(cell,directions,cell==='lstm'?setting:.8,cell==='gru'?setting:.5)});return {recurrent,placement:NLPMath.gruPlacement(),tfidf:NLPMath.tfidf(),nb:[.1,1].map(a=>NLPMath.classify('nb',a)),cnn:['max','mean'].map(p=>NLPMath.cnn(p)),tagger:['hmm','crf'].map(k=>NLPMath.tagger(k)),pool:[false,true].map(x=>NLPMath.pooling(x)),sentence:[false,true].map(x=>NLPMath.sentence(.5,x)),translation:['short','repeat','exact'].map(x=>NLPMath.translation(x)),qa:['partial','exact','wide'].map(x=>NLPMath.qa(x)),chunks:[0,2].map(x=>NLPMath.chunks(x)),sgns:NLPMath.skipgram(),ngram:[0,.5,1].map(a=>['cats','dogs','UNK'].map(w=>NLPMath.ngram(w,a)))}})()`;
console.log(vm.runInContext('JSON.stringify('+expression+')',c));
"""
        cls.data=json.loads(subprocess.check_output(["node","-e",script,str(ROOT/"src/training_lab/web")],text=True))

    def test_recurrent_states_and_directional_terminals_match_pytorch(self):
        for row in self.data["recurrent"]:
            with self.subTest(cell=row["cell"],directions=row["directions"],setting=row["setting"]):
                cell=row["cell"];directions=row["directions"];setting=row["setting"]
                model={"rnn":torch.nn.RNN,"lstm":torch.nn.LSTM,"gru":torch.nn.GRU}[cell](1,1,batch_first=True,bidirectional=directions==2,dtype=torch.float64)
                with torch.no_grad():
                    for p in model.parameters():p.zero_()
                    for d in range(directions):
                        suffix="_reverse" if d else ""
                        wi=getattr(model,"weight_ih_l0"+suffix);wh=getattr(model,"weight_hh_l0"+suffix);bi=getattr(model,"bias_ih_l0"+suffix)
                        w=.3 if d else .5;u=.6 if d else .8
                        if cell=="rnn":wi[0,0]=w;wh[0,0]=u
                        elif cell=="lstm":
                            wi[2,0]=w;wh[2,0]=.1;bi[1]=math.log(setting/(1-setting));bi[3]=math.log(3)
                        else:
                            wi[2,0]=w;wh[2,0]=u;bi[0]=-math.inf if setting==0 else math.inf if setting==1 else math.log(setting/(1-setting));bi[1]=math.log(.25/.75)
                    output,state=model(torch.tensor([[[1.],[0.],[-1.]]],dtype=torch.float64))
                hn,cn=state if cell=="lstm" else (state,None)
                np.testing.assert_allclose(output[0].detach().numpy(),row["result"]["output"],rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(hn[:,0,0].detach().numpy(),row["result"]["terminal"],rtol=1e-12,atol=1e-12)
                if cn is not None:
                    expected=[row["result"]["forward"][-1]["c"]]+([row["result"]["backward"][0]["c"]] if directions==2 else [])
                    np.testing.assert_allclose(cn[:,0,0].detach().numpy(),expected,atol=1e-12)

    def test_gru_reset_placement_with_matrices_and_nonzero_bias(self):
        model=torch.nn.GRUCell(1,2,dtype=torch.float64)
        with torch.no_grad():
            for p in model.parameters():p.zero_()
            model.bias_ih[:2]=torch.tensor([math.log(.2/.8),math.log(.8/.2)])
            model.bias_ih[2:4]=-math.inf
            model.weight_ih[4:,0]=torch.tensor([.3,-.2])
            model.weight_hh[4:]=torch.tensor([[1.,2.],[3.,4.]])
            model.bias_hh[4:]=torch.tensor([.5,-.5])
            actual=model(torch.tensor([[1.]],dtype=torch.float64),torch.tensor([[1.,2.]],dtype=torch.float64))
        np.testing.assert_allclose(actual.numpy()[0],self.data["placement"]["after"],atol=1e-7)
        self.assertFalse(np.allclose(self.data["placement"]["after"],self.data["placement"]["before"],atol=1e-6))

    def test_tfidf_and_naive_bayes_match_sklearn(self):
        vectorizer=TfidfVectorizer(vocabulary=["cat","dog","sat","slept"],smooth_idf=True,norm="l2")
        actual=vectorizer.fit_transform(["cat sat","cat slept","dog sat"])
        np.testing.assert_allclose(actual.toarray(),self.data["tfidf"]["vectors"],atol=1e-12)
        np.testing.assert_allclose(vectorizer.idf_,self.data["tfidf"]["idf"],atol=1e-12)
        for alpha,row in zip([.1,1],self.data["nb"]):
            model=MultinomialNB(alpha=alpha).fit([[3,1],[1,3]],[0,1])
            np.testing.assert_allclose(model.predict_proba([[2,1]])[0],row["posterior"],atol=1e-12)

    def test_cnn_and_negative_sampling(self):
        x=torch.tensor([[[1.,-1.,2.]]]);k=torch.tensor([[[.5,-.5]]]);conv=torch.nn.functional.conv1d(x,k)[0,0]
        for pool,row in zip(["max","mean"],self.data["cnn"]):
            np.testing.assert_allclose(conv.numpy(),row["convolution"])
            expected=torch.relu(conv).max() if pool=="max" else torch.relu(conv).mean()
            self.assertAlmostEqual(float(expected),row["pooled"])
        loss=-torch.nn.functional.logsigmoid(torch.tensor(.5,dtype=torch.float64))-torch.nn.functional.logsigmoid(torch.tensor(0.,dtype=torch.float64))
        self.assertAlmostEqual(float(loss),self.data["sgns"]["loss"])

    def test_viterbi_and_partition_against_independent_path_enumeration(self):
        for kind,row in zip(["hmm","crf"],self.data["tagger"]):
            emissions=np.array([[2,1],[0,3],[2,0]]) if kind=="crf" else np.log([[.8,.3],[.2,.7],[.8,.3]])
            transitions=np.array([[.5,-1],[-.5,.8]]) if kind=="crf" else np.log([[.7,.3],[.4,.6]])
            start=np.array([0.,0.]) if kind=="crf" else np.log([.6,.4])
            paths=list(itertools.product(range(2),repeat=3));scores=[]
            for p in paths:
                scores.append(start[p[0]]+sum(emissions[t,p[t]] for t in range(3))+sum(transitions[p[t-1],p[t]] for t in [1,2]))
            self.assertEqual(list(paths[int(np.argmax(scores))]),row["best"])
            self.assertAlmostEqual(max(scores),row["bestScore"])
            self.assertAlmostEqual(float(np.logaddexp.reduce(scores)),row["logZ"])
            self.assertAlmostEqual(math.exp(max(scores)-np.logaddexp.reduce(scores)),row["probability"])

    def test_pooling_flows_into_actual_similarity(self):
        for include,row,pool in zip([False,True],self.data["sentence"],self.data["pool"]):
            states=np.array([[1.,0.],[0.,1.],[2.,-1.]])
            pooled=(states if include else states[:2]).mean(axis=0);q=pooled/np.linalg.norm(pooled)
            np.testing.assert_allclose(pooled,pool["pooled"])
            np.testing.assert_allclose(q,row["q"])
            scores=np.array([[.8,.6],[0.,1.]])@q
            np.testing.assert_allclose(scores,row["scores"])
            loss=float(torch.nn.functional.cross_entropy(torch.from_numpy((scores/.5)[None,:]),torch.tensor([0])))
            self.assertAlmostEqual(loss,row["loss"])
        self.assertFalse(np.allclose(self.data["sentence"][0]["scores"],self.data["sentence"][1]["scores"]))

    def test_generation_metrics_and_chunk_coverage(self):
        for row,p,bp in zip(self.data["translation"],[1,.75,1],[math.exp(1-4/3),1,1]):
            self.assertAlmostEqual(row["precision"],p);self.assertAlmostEqual(row["bp"],bp);self.assertAlmostEqual(row["bleu1"],p*bp)
        for row,f1,em in zip(self.data["qa"],[2/3,1,.8],[0,1,0]):self.assertAlmostEqual(row["f1"],f1);self.assertEqual(row["em"],em)
        for overlap,row in zip([0,2],self.data["chunks"]):
            self.assertEqual(row["exposures"],8 if overlap==0 else 12)
            self.assertEqual(row["coversBoundary"],overlap==2)
            local=np.fromfunction(lambda i,j:abs(i-j)<=1,(8,8),dtype=int)
            global_mask=local.copy();global_mask[0,:]=True;global_mask[:,0]=True
            self.assertEqual(row["localPairs"],int(local.sum()));self.assertEqual(row["globalPairs"],int(global_mask.sum()))
        for alpha,rows in zip([0,.5,1],self.data["ngram"]):
            self.assertAlmostEqual(rows[0]["p"]+rows[1]["p"]+5*rows[2]["p"],1)
            if alpha==0:self.assertIsNone(rows[2]["nll"])

    def test_pytorch_parameter_count_conventions(self):
        for family,gates in [(torch.nn.RNN,1),(torch.nn.LSTM,4),(torch.nn.GRU,3)]:
            for directions in [1,2]:
                model=family(3,4,bidirectional=directions==2)
                self.assertEqual(sum(p.numel() for p in model.parameters()),directions*gates*4*(3+4+2))

    def test_nlp_coverage_and_shared_lesson_ownership(self):
        index=curriculum.catalog();active={l["id"] for l in index["lessons"]}
        self.assertEqual(len(curriculum.NLP_LESSONS),24)
        for row in curriculum.NLP_LESSONS:
            doc=curriculum.document(row["id"])
            self.assertEqual(doc["stage"],"nlp");self.assertEqual(doc["reader"]["explorer"],row["id"])
            self.assertEqual(len(doc["reader"]["notes"]),2)
            self.assertTrue(set(row["related"])<=active);self.assertTrue(set(row["prerequisites"])<=active)
            self.assertNotIn(row["id"],row["prerequisites"])
            self.assertEqual(len(guided_course.checks()[row["id"]]),2)
        paths={p["id"]:p for p in guided_course.catalog()["paths"]}
        self.assertIn("P6",{l["id"] for l in paths["nlp"]["lessons"]})
        self.assertNotIn("P6",{l["id"] for l in paths["llm"]["lessons"]})
        # The unified interview bank consolidated the former twelve NLP prompts.
        self.assertEqual(sum(q["category"]=="nlp" for q in index["questions"]),7)
        for bad in ["N00","N25","N09/../../secret"]:
            with self.assertRaises(ValueError):curriculum.document(bad)

    def test_embedded_practice_code_is_executable_without_downloads(self):
        import re
        for identifier in ["N10","N24"]:
            doc=curriculum.document(identifier)
            blocks=re.findall(r"```python\n(.*?)\n```",doc["markdown"],re.S)
            self.assertEqual(len(blocks),1)
            scope={}
            with contextlib.redirect_stdout(io.StringIO()):exec(compile(blocks[0],doc["source"],"exec"),scope)
            if identifier=="N10":
                self.assertEqual(tuple(scope["output"].shape),(2,5,8))
                self.assertEqual(tuple(scope["sentence"].shape),(2,8))
                self.assertEqual(scope["valid_lengths"].tolist(),[5,3])
            else:
                self.assertEqual(len(scope["pipeline"].predict(scope["probe"])),3)


if __name__=="__main__":unittest.main()
