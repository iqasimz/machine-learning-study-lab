import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from training_lab.alignment import dpo_loss, group_advantages, ppo_policy_loss, response_example
from training_lab.artifacts import ROOT, read_json, run_path, seal, verify, write_json
from training_lab.data import Bigram, batch, examples, load_corpus
from training_lab.model import ModelConfig, TinyDecoder, generate, loss_sum
from training_lab.server import Jobs, detail
from training_lab.tokenization import ByteBPE
from training_lab.training import load_model, train

torch.set_num_threads(1)


class TokenizerTests(unittest.TestCase):
    def test_unicode_roundtrip(self):
        tok = ByteBPE.fit(["abc abc 🧠", "abc def"], 16)
        for text in ("", "🧠 naïve e\u0301 中文", "<EOS> is literal text", "\x00\n"):
            self.assertEqual(tok.decode(tok.encode(text)), text)

    def test_deterministic_serialization(self):
        a = ByteBPE.fit(["abab ab", "baba"], 10)
        b = ByteBPE.fit(["abab ab", "baba"], 10)
        self.assertEqual(a.to_dict(), b.to_dict())
        self.assertEqual(ByteBPE.from_dict(a.to_dict()).encode("baba"), a.encode("baba"))

    def test_no_reserved_token_injection(self):
        self.assertTrue(all(i >= 3 for i in ByteBPE().encode("<PAD><BOS><EOS>")))

    def test_never_merge_across_documents(self):
        tok = ByteBPE.fit(["a", "b", "a", "b"], 20)
        self.assertEqual(tok.merges, [])

    def test_invalid_merge_rejected(self):
        with self.assertRaises(ValueError):
            ByteBPE([(500, 501)])


class DataTests(unittest.TestCase):
    def test_fixture_split_counts(self):
        rows = load_corpus(ROOT / "data/corpus.jsonl")
        self.assertEqual([sum(r["split"] == s for r in rows) for s in ["train", "dev", "test"]], [16,4,4])

    def test_duplicate_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.jsonl"
            p.write_text('\n'.join(json.dumps(r) for r in [
                {"id":"a","split":"train","text":"Hi THERE"},
                {"id":"b","split":"dev","text":" hi  there "},
                {"id":"c","split":"test","text":"Other"}]))
            with self.assertRaises(ValueError):
                load_corpus(p)

    def test_targets_never_cross_boundary(self):
        rows = [{"id":"a","split":"train","text":"abc"}, {"id":"b","split":"train","text":"def"}]
        tok = ByteBPE()
        items = examples(rows, tok, "train", 2)
        self.assertEqual(sum(len(y) for _, y, _ in items), 8)
        self.assertEqual(items[1][1][-1], 2)
        self.assertEqual(items[2][0][0], 1)

    def test_padding_labels_are_ignored(self):
        x, y = batch([([1,4],[4,2],"a"),([1],[2],"b")])
        self.assertEqual(y.tolist(), [[4,2],[2,-100]])
        self.assertEqual(x[1,1], 0)

    def test_bigram_smoothing(self):
        tok = ByteBPE()
        base = Bigram(tok.vocab_size).fit([{"split":"train","text":"ab"}], tok)
        self.assertAlmostEqual(float(base.counts[100,101] / base.counts[100].sum()), 2/258)
        self.assertEqual(float(base.counts[:, :2].sum()), 0)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.model = TinyDecoder(ModelConfig(259, block_size=16, width=16, heads=2, layers=1, dropout=0))

    def test_causal_invariance(self):
        self.model.eval()
        a = torch.tensor([[1,4,5,6]])
        b = torch.tensor([[1,4,80,90]])
        self.assertTrue(torch.equal(self.model(a)[:,:2], self.model(b)[:,:2]))

    def test_shapes_and_mask(self):
        logits, attention = self.model(torch.tensor([[1,4,5]]), inspect=True)
        self.assertEqual(list(logits.shape), [1,3,259])
        self.assertEqual(list(attention[0].shape), [1,2,3,3])
        self.assertEqual(float(attention[0].detach().triu(1).sum()), 0)
        torch.testing.assert_close(attention[0].sum(-1), torch.ones(1,2,3))

    def test_loss_and_all_masked(self):
        logits = torch.zeros(1,2,259)
        loss, n = loss_sum(logits, torch.tensor([[4,-100]]))
        self.assertEqual(n,1)
        self.assertAlmostEqual(loss.item(), math.log(259), places=5)
        with self.assertRaises(ValueError):
            loss_sum(logits, torch.full((1,2), -100))

    def test_accumulation_token_weighting(self):
        first = ([1,4,5],[4,5,2],"a")
        second = ([1],[2],"b")
        self.model.zero_grad()
        x,y = batch([first,second]); loss,n = loss_sum(self.model(x), y)
        (loss/n).backward()
        expected = [p.grad.clone() for p in self.model.parameters()]
        self.model.zero_grad()
        for item in (first,second):
            x,y=batch([item]); loss,_=loss_sum(self.model(x),y); (loss/n).backward()
        for p,g in zip(self.model.parameters(), expected):
            torch.testing.assert_close(p.grad,g,atol=1e-6,rtol=1e-5)

    def test_tied_weights_and_generation(self):
        self.assertIs(self.model.head.weight,self.model.tokens.weight)
        self.assertIsInstance(generate(self.model,ByteBPE(),"a",new_tokens=4,temperature=0),str)


class AlignmentTests(unittest.TestCase):
    def test_response_mask(self):
        x,y=response_example(ByteBPE(),"hi","ok")
        self.assertEqual(len(x),len(y))
        self.assertEqual(y[:2],[-100,-100])
        self.assertEqual(y[2:],[ord("o")+3,ord("k")+3,2])

    def test_dpo_margin_and_gradient(self):
        value=torch.tensor([0.0],requires_grad=True)
        loss=dpo_loss(value,torch.tensor([0.0]),torch.tensor([0.0]),torch.tensor([0.0]))
        self.assertAlmostEqual(loss.item(),math.log(2),places=6)
        loss.backward();self.assertLess(value.grad.item(),0)

    def test_ppo_negative_advantage(self):
        # ratio=2, negative advantage: unclipped is the pessimistic policy term.
        loss=ppo_policy_loss(torch.tensor([math.log(2)]),torch.tensor([0.]),torch.tensor([-1.]))
        self.assertAlmostEqual(loss.item(),2)

    def test_equal_group_rewards(self):
        self.assertTrue(torch.equal(group_advantages(torch.ones(4)),torch.zeros(4)))


class ArtifactTests(unittest.TestCase):
    def test_hashes_detect_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);write_json(path/"a.json",{"x":1});seal(path);verify(path)
            write_json(path/"a.json",{"x":2})
            with self.assertRaises(ValueError):verify(path)

    def test_invalid_run_paths(self):
        for name in ("../secret","/tmp/foo","a/b","x y",""):
            with self.assertRaises(ValueError):run_path(name)

    def test_only_allowlisted_jobs(self):
        with self.assertRaises(ValueError):Jobs().start("tiny; rm anything")


class WorkflowTests(unittest.TestCase):
    def test_train_resume_matches_uninterrupted_cpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            train("full","smoke",steps=6,device="cpu",runs_root=root)
            train("split","smoke",steps=3,device="cpu",runs_root=root)
            train("split","smoke",steps=6,device="cpu",resume=True,runs_root=root)
            a=torch.load(root/"full/last.pt",weights_only=True)
            b=torch.load(root/"split/last.pt",weights_only=True)
            for key in a["model"]:torch.testing.assert_close(a["model"][key],b["model"][key],atol=0,rtol=0)
            self.assertEqual(a["tokens_processed"],b["tokens_processed"])
            self.assertTrue(torch.equal(a["sampler_rng"],b["sampler_rng"]))
            self.assertEqual(read_json(root/"split/manifest.json")["status"],"MEASURED")
            verify(root/"split")
            model,tok,state=load_model(root/"split")
            self.assertIsInstance(generate(model,tok,"A",new_tokens=2),str)

    def test_existing_run_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"exists").mkdir()
            with self.assertRaises(FileExistsError):train("exists","smoke",steps=1,device="cpu",runs_root=root)

    def test_resume_changed_seed_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);train("run","smoke",steps=1,device="cpu",runs_root=root)
            with self.assertRaisesRegex(ValueError,"identical config"):
                train("run","smoke",steps=2,seed=33,device="cpu",resume=True,runs_root=root)


if __name__ == "__main__":
    unittest.main()
