"""Bounded offline training with weighted loss, checkpoints and frozen split evaluation."""
from collections import Counter
from datetime import datetime, timezone
import math
import os
import platform
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .artifacts import ROOT, RUNS, read_json, run_path, seal, sha256, source_hashes, verify, write_json
from .data import Bigram, batch, examples, load_corpus
from .model import ModelConfig, TinyDecoder, generate, loss_sum
from .tokenization import ByteBPE
from .experiments import DEFAULTS, validate
from .telemetry import Telemetry
from .hardware import choose_device

PRESETS = {
    "smoke": {"width": 32, "layers": 1, "heads": 2, "block_size": 48, "steps": 40, "batch_size": 4, "accumulation": 1, "merges": 24},
    "tiny": {"width": 64, "layers": 2, "heads": 4, "block_size": 64, "steps": 240, "batch_size": 8, "accumulation": 2, "merges": 48},
}


def now():
    return datetime.now(timezone.utc).isoformat()


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def learning_rate_at(step, config):
    warmup = config["warmup_steps"]
    peak = config["learning_rate"]
    if warmup and step <= warmup:
        return peak * step / warmup
    if config["schedule"] == "constant":
        return peak
    progress = min(1.0, max(0.0, (step-warmup)/(config["schedule_steps"]-warmup)))
    return peak * (0.1 + 0.9 * (1 + math.cos(math.pi * progress))/2)


def make_optimizer(model, config):
    if config["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    return torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])


def clip_gradients(model, limit):
    return float(torch.nn.utils.clip_grad_norm_(model.parameters(), limit if limit > 0 else float("inf")))


def select_training_data(rows, fraction):
    train_rows = [r for r in rows if r["split"] == "train"]
    selected = {r["id"] for r in train_rows[:max(1, int(len(train_rows)*fraction))]}
    return [r for r in rows if r["split"] != "train" or r["id"] in selected]


@torch.no_grad()
def evaluate(model, items, device):
    model.eval()
    total, count, per_document = 0.0, 0, {}
    for item in items:
        x, y = batch([item], device)
        loss, valid = loss_sum(model(x), y)
        total += loss.item()
        count += valid
        entry = per_document.setdefault(item[2], {"sum_nll": 0.0, "tokens": 0})
        entry["sum_nll"] += loss.item()
        entry["tokens"] += valid
    if not count:
        raise ValueError("No evaluation targets.")
    return {"nll": total / count, "perplexity": math.exp(min(total / count, 80)), "tokens": count,
            "documents": [{"id": doc, "nll": v["sum_nll"] / v["tokens"], "tokens": v["tokens"]}
                          for doc, v in per_document.items()]}


def save_checkpoint(path, model, optimizer, sampler, step, exposures, best_dev, config):
    state = {"format": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "sampler_rng": sampler.get_state(), "torch_rng": torch.get_rng_state(),
             "step": step, "tokens_processed": exposures, "best_dev": best_dev, "config": config}
    device = next(model.parameters()).device.type
    if device == "mps":
        state["mps_rng"] = torch.mps.get_rng_state()
    if device == "cuda":
        state["cuda_rng"] = torch.cuda.get_rng_state()
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def load_model(directory, checkpoint="best.pt", device="cpu"):
    verify(directory)
    # Only load our locally generated state dictionaries. Never arbitrary pickled modules.
    state = torch.load(Path(directory) / checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    tokenizer = ByteBPE.from_dict(read_json(Path(directory) / "tokenizer.json"))
    model = TinyDecoder(ModelConfig(**config["model"])).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, tokenizer, state


def inspect_model(model, tokenizer):
    ids = ([1] + tokenizer.encode("The model learns from text."))[:16]
    with torch.no_grad():
        logits, attention = model(torch.tensor([ids], device=next(model.parameters()).device), inspect=True)
        values, indices = logits[0, -1].softmax(-1).topk(8)
    return {"status": "MEASURED", "layer": 0, "head": 0, "tokens": [tokenizer.label(i) for i in ids],
            "input_shape": [1, len(ids)], "hidden_shape": [1, len(ids), model.config.width],
            "attention_shape": list(attention[0].shape), "logits_shape": list(logits.shape),
            "attention": attention[0][0, 0].cpu().tolist(),
            "next_tokens": [{"token": tokenizer.label(int(i)), "probability": float(p)} for i, p in zip(indices, values)],
            "warning": "Attention weights show information mixing, not proof of causal explanation."}


def train(name, preset="tiny", steps=None, seed=17, device="auto", resume=False,
          corpus=None, runs_root=RUNS, overrides=None, learning=None):
    if preset not in PRESETS:
        raise ValueError("Choose smoke or tiny.")
    settings = {**DEFAULTS, **PRESETS[preset]}
    settings["seed"], settings["device"] = seed, device
    if overrides is not None:
        settings = validate(overrides)
        seed, device = settings["seed"], settings["device"]
    steps = settings["steps"] if steps is None else steps
    if not isinstance(steps, int) or not 1 <= steps <= 2000:
        raise ValueError("This local lab allows 1–2000 optimizer steps per run.")
    if not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1:
        raise ValueError("Seed must be an integer in [0, 2^31-1].")
    corpus = Path(corpus or ROOT / "data" / "corpus.jsonl")
    rows = load_corpus(corpus)
    rows = select_training_data(rows, settings["train_fraction"])
    device = choose_device(device)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    random.seed(seed)
    tokenizer = ByteBPE.fit([r["text"] for r in rows if r["split"] == "train"], settings["merges"])
    config = {"preset": preset, "seed": seed, "learning_rate": settings["learning_rate"],
              "batch_size": settings["batch_size"], "accumulation": settings["accumulation"],
              "optimizer":settings["optimizer"], "weight_decay":settings["weight_decay"],
              "clip_norm":settings["clip_norm"], "schedule":settings["schedule"],
              "warmup_steps":settings["warmup_steps"], "schedule_steps":settings["schedule_steps"],
              "train_fraction":settings["train_fraction"],
              "corpus_sha256": sha256(corpus), "tokenizer": tokenizer.to_dict(),
              "model": asdict(ModelConfig(tokenizer.vocab_size, settings["block_size"], settings["width"],
                                          settings["layers"], settings["heads"], settings["dropout"],
                                          settings["positions"],settings["residuals"],settings["normalization"]))}
    directory = run_path(name, runs_root)
    if resume:
        verify(directory)
        previous = read_json(directory / "manifest.json")
        if previous["config"] != config or previous["hardware"]["device"] != device:
            raise ValueError("Resume requires identical config, data, seed and device.")
        if previous["source_hashes"] != source_hashes():
            raise ValueError("Source changed since checkpoint; start a new run instead of silently resuming.")
    else:
        directory.mkdir(parents=True, exist_ok=False)
    model = TinyDecoder(ModelConfig(**config["model"])).to(device)
    optimizer = make_optimizer(model, config)
    sampler = torch.Generator().manual_seed(seed + 1)
    start_step, exposures, best_dev, metrics = 0, 0, float("inf"), []
    if resume:
        state = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        sampler.set_state(state["sampler_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device == "mps":
            torch.mps.set_rng_state(state["mps_rng"])
        if device == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"])
        start_step, exposures, best_dev = state["step"], state["tokens_processed"], state["best_dev"]
        if steps <= start_step:
            raise ValueError("Resume steps is the TOTAL target, and must exceed the saved step.")
        metrics = read_json(directory / "metrics.json")
    split_items = {s: examples(rows, tokenizer, s, settings["block_size"]) for s in ("train", "dev", "test")}
    baseline = Bigram(tokenizer.vocab_size).fit(rows, tokenizer)
    manifest = {"schema_version": 2, "id": name, "status": "RUNNING", "purpose": "educational_pretraining",
                "started_at": previous["started_at"] if resume else now(), "updated_at": now(),
                "resumed": resume, "requested_steps": steps, "completed_steps": start_step,
                "config": config, "model": model.description(), "source_hashes": source_hashes(),
                "settings":{**settings,"steps":steps,"seed":seed,"device":device}, "learning":learning or {},
                "evaluation_protocol":"document_chunks_v1", "metric_definition":"next-token NLL including EOS; excludes PAD/BOS",
                "hardware": {"device": device, "machine": platform.machine(), "python": platform.python_version(),
                             "torch": str(torch.__version__), "threads": 1},
                "limits": ["Authored miniature corpus, not general language competence.",
                           "Development loss selects checkpoint; test is evaluated only at run completion.",
                           "Same-tokenizer bigram baseline; training compute and model capacity differ.",
                           "Fixed synthetic splits share templates; exact dedup is not semantic dedup.",
                           "CPU restart equivalence tested; accelerator bitwise determinism not promised."]}
    write_json(directory / "manifest.json", manifest)
    write_json(directory / "tokenizer.json", tokenizer.to_dict())
    write_json(directory / "data.json", {"license": "CC0-1.0 authored synthetic learning fixture",
                                        "documents": rows, "counts": dict(Counter(r["split"] for r in rows)),
                                        "duplicates": 0, "audit": "NFKC + casefold + whitespace exact dedup only"})
    write_json(directory / "metrics.json", metrics)
    elapsed_before = previous.get("training_seconds", 0) if resume else 0
    started = time.perf_counter()
    telemetry=Telemetry(directory)
    try:
        for step in range(start_step + 1, steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            current_lr = learning_rate_at(step, config)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            microbatches = []
            for _ in range(config["accumulation"]):
                indices = torch.randint(len(split_items["train"]), (config["batch_size"],), generator=sampler)
                microbatches.append(batch([split_items["train"][i] for i in indices.tolist()], device))
            # Divide by ALL valid tokens, not an average of unequal microbatch means.
            valid_total = sum(int((y != -100).sum()) for _, y in microbatches)
            summed_loss = 0.0
            for x, y in microbatches:
                loss, count = loss_sum(model(x), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss; run stopped.")
                (loss / valid_total).backward()
                summed_loss += loss.item()
            grad_norm = clip_gradients(model, config["clip_norm"])
            if not math.isfinite(grad_norm):
                raise FloatingPointError("Non-finite gradients; run stopped.")
            optimizer.step()
            exposures += valid_total
            row={"step":step,"sampled_train_nll":summed_loss/valid_total,
                 "gradient_norm_pre_clip":grad_norm,"learning_rate":current_lr,"tokens_processed":exposures,
                 "elapsed_seconds":elapsed_before+time.perf_counter()-started,
                 "tokens_per_second":(exposures-state["tokens_processed"] if resume else exposures)/max(time.perf_counter()-started,1e-6)}
            if step == 1 or step % 20 == 0 or step == steps:
                dev = evaluate(model, split_items["dev"], device)
                train_eval = evaluate(model, split_items["train"], device)
                row.update(train_nll=train_eval["nll"],dev_nll=dev["nll"])
                if dev["nll"] < best_dev:
                    best_dev = dev["nll"]
                    save_checkpoint(directory / "best.pt", model, optimizer, sampler, step, exposures, best_dev, config)
            metrics.append(row);telemetry.log(row)
            # A resumable last checkpoint is recorded for EVERY completed optimizer step.
            save_checkpoint(directory / "last.pt", model, optimizer, sampler, step, exposures, best_dev, config)
            manifest.update(completed_steps=step, tokens_processed=exposures, updated_at=now(),
                            elapsed_seconds=elapsed_before+time.perf_counter()-started,phase="Training the decoder",
                            tokens_per_second=row["tokens_per_second"],
                            eta_seconds=(time.perf_counter()-started)/(step-start_step)*(steps-step))
            write_json(directory / "metrics.json", metrics)
            write_json(directory / "manifest.json", manifest)
        synchronize(device)
        manifest["training_seconds"] = elapsed_before + time.perf_counter() - started
        state = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        evaluations = {s: evaluate(model, split_items[s], device) for s in ("train", "dev", "test")}
        for split, result in evaluations.items():
            byte_count = sum(len(r["text"].encode("utf-8")) for r in rows if r["split"] == split)
            result["utf8_bytes"] = byte_count
            result["bits_per_byte"] = result["nll"] * result["tokens"] / (byte_count * math.log(2))
        baselines = {s: baseline.evaluate(rows, tokenizer, s) for s in ("dev", "test")}
        write_json(directory / "evaluation.json", {"checkpoint_step": state["step"], "transformer": evaluations,
                   "bigram": baselines, "selection": "lowest dev NLL; never selected on test",
                   "units": "nats per shared BPE next-token target; lower is better"})
        write_json(directory / "inspection.json", inspect_model(model, tokenizer))
        torch.manual_seed(seed + 100)
        prompts = ["The model", "A source", "Training"]
        write_json(directory / "samples.json", [{"prompt": p, "completion": generate(model, tokenizer, p),
                   "temperature": 0.8, "top_k": 20, "max_new_tokens": 80, "seed": seed + 100,
                   "note": "Sequential samples share one seeded RNG stream; not independent reseeding."} for p in prompts])
        manifest.update(status="MEASURED", best_checkpoint_step=state["step"], finished_at=now(), updated_at=now())
    except BaseException as exc:
        synchronize(device)
        manifest.update(status="CANCELLED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                        error=f"{type(exc).__name__}: {exc}", updated_at=now(),
                        training_seconds=elapsed_before + time.perf_counter() - started)
        raise
    finally:
        telemetry.close()
        write_json(directory / "manifest.json", manifest)
        seal(directory)
    return manifest
