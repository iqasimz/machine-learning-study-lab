"""Bounded post-training operations exposed by the local learning UI."""
from datetime import datetime, timezone
import math
import shutil
import uuid

from .artifacts import ROOT, RUNS, read_json, run_path, sha256, verify, write_json


def generate_saved(payload):
    import torch
    from .model import generate
    from .training import load_model
    required={"run","prompt","temperature","top_k","tokens","seed"}
    if not isinstance(payload,dict) or set(payload)!=required:
        raise ValueError("Generation needs run, prompt, temperature, top_k, tokens and seed.")
    if not isinstance(payload["prompt"],str) or len(payload["prompt"])>1000:
        raise ValueError("Prompt must contain at most 1000 characters.")
    for key,choices in {"temperature":[0.0,.3,.8,1.2],"top_k":[1,10,20,50],"tokens":[16,40,80,128],"seed":[7,17,29,43]}.items():
        if not any(type(payload[key]) in {int,float} and type(payload[key]) is not bool and payload[key]==c for c in choices):
            raise ValueError(f"Unsupported {key}.")
    directory=run_path(payload["run"])
    manifest=read_json(directory/"manifest.json")
    if manifest.get("engine")=="lora":
        from .adaptation import generate_saved as generate_adapted
        return generate_adapted(payload)
    if manifest["status"]!="MEASURED":raise ValueError("Choose a completed run.")
    torch.set_num_threads(1)
    torch.manual_seed(payload["seed"])
    model,tokenizer,state=load_model(directory,device="cpu")
    # load_model initializes an object before loading weights; seed again for sampling only.
    torch.manual_seed(payload["seed"])
    text=generate(model,tokenizer,payload["prompt"],payload["tokens"],payload["temperature"],payload["top_k"])
    identifier="generation-"+uuid.uuid4().hex[:12]
    result={**payload,"id":identifier,"completion":text,"device":"cpu","checkpoint_step":state["step"],
            "checkpoint_sha256":sha256(directory/"best.pt"),"created_at":datetime.now(timezone.utc).isoformat(),
            "explanation":"Sampling changes outputs, not trained weights. Greedy temperature zero ignores top-k randomness."}
    write_json(run_path(identifier,RUNS/"_generations").with_suffix(".json"),result)
    return result


def export_run(identifier):
    directory=run_path(identifier)
    verify(directory)
    manifest=read_json(directory/"manifest.json")
    if manifest.get("engine")=="lora":
        raise ValueError("This run uses a separate pretrained model and local adapters. See its run folder; the scratch-model bundle format does not apply.")
    if manifest["status"]!="MEASURED":raise ValueError("Only completed measured runs can be exported.")
    target=run_path(identifier,ROOT/"exports")
    if target.exists():
        existing=read_json(target/"export.json")
        for name,digest in existing["files"].items():
            if name not in {"best.pt","tokenizer.json","manifest.json","evaluation.json","data.json"}:
                raise ValueError("Unexpected existing export filename.")
            if sha256(target/name)!=digest or sha256(directory/name)!=digest:
                raise ValueError("Existing export differs; it will not be overwritten. Export a new named run instead.")
        return {"run":identifier,"directory":str(target),"status":"EXISTING_VERIFIED","assistant_approved":False}
    target.mkdir(parents=True,exist_ok=False)
    names=["best.pt","tokenizer.json","manifest.json","evaluation.json","data.json"]
    for name in names:shutil.copy2(directory/name,target/name)
    write_json(target/"export.json",{"format":"training_lab_educational_v1","source_run":identifier,
               "usable_as_general_assistant":False,"deployment_status":"NOT_VALIDATED",
               "files":{n:sha256(target/n) for n in names}})
    return {"run":identifier,"directory":str(target),"status":"EXPORTED","assistant_approved":False}
