"""Comparison diagnostics: display differences before inviting a ranking."""
import json

from .artifacts import RUNS, read_json, run_path, verify


def flatten(obj,prefix=""):
    result={}
    for key,value in obj.items():
        name=f"{prefix}.{key}" if prefix else key
        if isinstance(value,dict):result.update(flatten(value,name))
        else:result[name]=value
    return result


def compare_runs(identifiers,runs_root=RUNS):
    if not isinstance(identifiers,list) or not 2<=len(identifiers)<=4 or len(set(identifiers))!=len(identifiers):
        raise ValueError("Select two to four different completed runs.")
    runs=[]
    for identifier in identifiers:
        directory=run_path(identifier,runs_root)
        verify(directory)
        manifest=read_json(directory/"manifest.json")
        if manifest["status"]!="MEASURED":raise ValueError("Only completed measured runs can be compared.")
        data=read_json(directory/"data.json")
        runs.append({"manifest":manifest,"evaluation":read_json(directory/"evaluation.json"),
                     "metrics":read_json(directory/"metrics.json"),"data":data})
    def signature(rows,split):
        return sorted((r["id"],r["text"]) for r in rows["data"]["documents"] if r["split"]==split)
    same_text=all(signature(r,"dev")==signature(runs[0],"dev") and signature(r,"test")==signature(runs[0],"test") for r in runs)
    same_tokenizer=all(r["manifest"]["config"]["tokenizer"]==runs[0]["manifest"]["config"]["tokenizer"] for r in runs)
    protocols={r["manifest"].get("evaluation_protocol","document_chunks_v1") for r in runs}
    nll_comparable=same_text and same_tokenizer and len(protocols)==1
    bpb_comparable=same_text and len(protocols)==1 and all("bits_per_byte" in r["evaluation"]["transformer"]["test"] for r in runs)
    settings=[flatten(r["manifest"].get("settings",r["manifest"]["config"])) for r in runs]
    changes=[]
    for key in sorted(set().union(*(s.keys() for s in settings))):
        values=[s.get(key,"not recorded") for s in settings]
        if any(v!=values[0] for v in values):changes.append({"parameter":key,"values":values})
    warnings=[]
    if not same_text:warnings.append("Held-out text differs. Do not rank these as the same evaluation task.")
    if not same_tokenizer:warnings.append("Tokenizers differ: per-token NLL/perplexity have different units. Their cross-run ranking is disabled.")
    if len(protocols)>1:warnings.append("Evaluation protocols differ; cross-run metric ranking is disabled.")
    if bpb_comparable and not same_tokenizer:warnings.append("Bits/byte uses the same raw text, but includes EOS cost; tokenizer-dependent context/chunking and training budgets still differ.")
    if len(changes)>1:warnings.append("Several settings changed. This is exploratory comparison, not an isolated one-factor effect.")
    exposures=[r["manifest"].get("tokens_processed") for r in runs]
    if len(set(exposures))>1:warnings.append("Token exposure differs. Equal optimizer updates do not establish equal training data exposure or compute.")
    relevant=lambda r:{k:v for k,v in r["manifest"]["source_hashes"].items() if k.endswith(("training.py","model.py","data.py","tokenization.py","adaptation.py"))}
    if any(relevant(r)!=relevant(runs[0]) for r in runs):warnings.append("Training-related source revisions differ; code changes are an additional possible confound.")
    warnings.extend(["No automatic statistical significance or causal claim; this panel does not replace repeated-seed evaluation.",
                     "Using test outcomes to choose your next configuration makes these fixture tests part of development."])
    return {"runs":[{"id":r["manifest"]["id"],"manifest":r["manifest"],"evaluation":r["evaluation"],"metrics":r["metrics"]} for r in runs],
            "same_held_out_text":same_text,"same_tokenizer":same_tokenizer,"nll_comparable":nll_comparable,
            "bits_per_byte_comparable":bpb_comparable,"changes":changes,"warnings":warnings,
            "default_metric":"dev_nll" if nll_comparable else "dev_bits_per_byte" if bpb_comparable else "none"}
