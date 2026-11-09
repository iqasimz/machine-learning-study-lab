"""Offline LoRA adaptation of a pinned, coherent pretrained language model."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import math
import platform
import random
import time

import torch

from .artifacts import ROOT,RUNS,read_json,run_path,seal,sha256,source_hashes,verify,write_json
from .assets import MODEL_DIR,MODEL_ID,MODEL_REVISION,DATA_DIR,DATA_FILES,DATA_ID,DATA_REVISION,status
from .telemetry import Telemetry
from .training import choose_device,synchronize

DEFAULTS={"steps":40,"learning_rate":0.0001,"block_size":256,"accumulation":4,"rank":8,"seed":17,"device":"auto","examples":512}
CHOICES={"steps":[10,20,40,80,160,400],"learning_rate":[0.00003,0.0001,0.0003],"block_size":[128,256],
         "accumulation":[1,2,4],"rank":[4,8,16],"seed":[7,17,29],"device":["auto","cpu","cuda","mps"],"examples":[128,512,1024]}
PROMPTS=["Explain what a language model does in two short sentences.","Why does ice melt when it gets warm?", "Suggest three things to bring for a day hike."]


def validate(settings=None):
    if not isinstance(settings or {},dict) or set(settings or {})-set(DEFAULTS): raise ValueError("Unknown adaptation settings.")
    result={**DEFAULTS,**(settings or {})}
    for key,value in result.items():
        if type(value) is bool or value not in CHOICES[key]: raise ValueError("Unsupported adaptation setting: "+key)
        result[key]=next(choice for choice in CHOICES[key] if type(value)==type(choice) and value==choice) if type(value) is str else next(choice for choice in CHOICES[key] if value==choice)
    return result


def catalog():
    from .source_code import language
    return {"defaults":DEFAULTS,"choices":CHOICES,"assets":status(),"prompts":PROMPTS,
            "sources":language("lora"),
            "model_config":read_json(MODEL_DIR/"config.json") if (MODEL_DIR/"config.json").is_file() else None,
            "method":"LoRA: frozen pretrained weights, small trainable adapters on attention query/value projections.",
            "caveat":"Conversational data is synthetic and may be part of the base model's original training. This is an adaptation exercise, not a novel generalization benchmark."}


def conversations():
    import pyarrow.parquet as pq
    rows=[];seen=set()
    for file,expected in DATA_FILES.items():
        path=DATA_DIR/"data"/file
        if sha256(path)!=expected:raise ValueError("Prepared conversation data changed; prepare it again.")
        for record in pq.read_table(path).to_pylist():
            messages=record["messages"]
            canonical="\n".join(m["content"].strip() for m in messages)
            group=hashlib.sha256(canonical.casefold().encode()).hexdigest()
            if group in seen:continue
            seen.add(group)
            split="test" if file.startswith("test") else "dev" if int(group[:8],16)%10==0 else "train"
            # Group by full conversation; all turns from one conversation stay in one split.
            for i,m in enumerate(messages):
                if m["role"]!="assistant" or i<3:continue # skip the generic greeting exchange
                if messages[i-1]["role"]!="user":continue
                rows.append({"id":group[:16]+"-"+str(i),"group":group,"split":split,
                    "messages":messages[:i+1],"text":messages[i-1]["content"]+"\n"+m["content"],
                    "prompt":messages[i-1]["content"],"answer":m["content"]})
    return rows


def encode_example(row,tokenizer,maximum):
    # Preserve the supplied chat template; mask prompt positions, not EOS token IDs.
    # The immediate exchange is a self-contained bounded task; earlier turns are omitted.
    messages=[{"role":"user","content":row["prompt"]},{"role":"assistant","content":row["answer"]}]
    prompt=tokenizer.apply_chat_template(messages[:1],tokenize=True,add_generation_prompt=True)
    full=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False)
    if full[:len(prompt)]!=prompt:raise ValueError("Chat template prefix is not stable; cannot mask safely.")
    if len(prompt)>=maximum-8:return None
    full=full[:maximum]
    labels=[-100]*len(prompt)+full[len(prompt):]
    if sum(x!=-100 for x in labels)<4:return None
    return {"ids":full,"labels":labels,"id":row["id"],"group":row["group"]}


def prepared_data(tokenizer,settings):
    rows=conversations();encoded={s:[] for s in ["train","dev","test"]};docs=[]
    seen=set()
    # Stable text-level duplicate filtering across all retained examples as well as group separation.
    for row in sorted(rows,key=lambda r:(r["split"]!="test",r["split"]!="dev",r["id"])):
        key=hashlib.sha256(" ".join(row["text"].casefold().split()).encode()).hexdigest()
        if key in seen:continue
        item=encode_example(row,tokenizer,settings["block_size"])
        limit=settings["examples"] if row["split"]=="train" else 16
        if item is None or len(encoded[row["split"]])>=limit:continue
        seen.add(key);encoded[row["split"]].append(item)
        docs.append({k:row[k] for k in ["id","group","split","text","prompt","answer"]})
    if min(map(len,encoded.values()))<4:raise ValueError("Too few valid train/dev/test examples after masking.")
    return encoded,docs


def tensors(item,device):
    return torch.tensor([item["ids"]],device=device),torch.tensor([item["labels"]],device=device)


@torch.no_grad()
def evaluate(model,items,device):
    model.eval();total=0.;count=0
    for item in items:
        x,y=tensors(item,device);valid=int((y[:,1:]!=-100).sum())
        value=model(input_ids=x,labels=y,use_cache=False).loss
        total+=float(value)*valid;count+=valid
    return {"nll":total/count,"perplexity":math.exp(min(80,total/count)),"tokens":count}


@torch.no_grad()
def sample(model,tokenizer,prompt,device,temperature=0.,top_k=20,tokens=72,seed=17):
    model.eval();torch.manual_seed(seed)
    ids=tokenizer.apply_chat_template([{"role":"user","content":prompt}],tokenize=True,add_generation_prompt=True,return_tensors="pt").to(device)
    if ids.shape[1]>512:raise ValueError("Prompt exceeds the local generation limit.")
    kw={"do_sample":temperature>0,"max_new_tokens":tokens,"pad_token_id":tokenizer.pad_token_id,
        "eos_token_id":tokenizer.eos_token_id,"use_cache":True}
    if temperature>0:kw.update(temperature=temperature,top_k=top_k)
    output=model.generate(ids,attention_mask=torch.ones_like(ids),**kw)
    return tokenizer.decode(output[0,ids.shape[1]:],skip_special_tokens=True)


def load_base(device):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    if not status()["ready"]:raise ValueError("Local language model/data are not prepared. Run: python lab.py prepare --model")
    tokenizer=AutoTokenizer.from_pretrained(MODEL_DIR,local_files_only=True,trust_remote_code=False)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(MODEL_DIR,local_files_only=True,trust_remote_code=False,
        use_safetensors=True,dtype=torch.float32,attn_implementation="eager").to(device)
    return model,tokenizer


def train(name,settings=None,runs_root=RUNS):
    from peft import LoraConfig,get_peft_model
    settings=validate(settings);device=choose_device(settings["device"])
    torch.set_num_threads(4);torch.manual_seed(settings["seed"]);random.seed(settings["seed"])
    directory=run_path(name,runs_root);directory.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();metrics=[];writer=None
    manifest={"schema_version":3,"id":name,"engine":"lora","status":"RUNNING","phase":"Loading the pretrained model",
        "purpose":"local_instruction_adaptation","started_at":datetime.now(timezone.utc).isoformat(),"completed_steps":0,
        "requested_steps":settings["steps"],"settings":settings,"source_hashes":source_hashes(),
        "model":{"parameters":361821120,"name":MODEL_ID},"hardware":{"device":device,"torch":str(torch.__version__),"machine":platform.machine()},
        "evaluation_protocol":"assistant_only_conversation_v1","config":{"tokenizer":{"repo":MODEL_ID,"revision":MODEL_REVISION},"model":{"name":MODEL_ID}},
        "limits":["Language competence comes from the pretrained model, not this short local adaptation.",
                  "Conversation data is synthetic and may have been seen during original SmolLM training.",
                  "Only assistant response tokens contribute to loss; reported loss is not directly comparable to scratch runs.",
                  "Adapters may improve or hurt behaviour. Short runs do not establish general quality."]}
    write_json(directory/"manifest.json",manifest);write_json(directory/"metrics.json",metrics)
    def progress(phase,**values):
        elapsed=time.perf_counter()-started
        manifest.update(phase=phase,elapsed_seconds=elapsed,updated_at=datetime.now(timezone.utc).isoformat(),**values)
        write_json(directory/"manifest.json",manifest)
    try:
        model,tokenizer=load_base(device);items,docs=prepared_data(tokenizer,settings)
        write_json(directory/"data.json",{"license":"Apache-2.0","repo":DATA_ID,"revision":DATA_REVISION,"synthetic":True,
            "documents":docs,"counts":dict(Counter(d["split"] for d in docs)),"masking":"Immediate user/assistant exchange; prompt labels -100; real EOS retained; padding never targets.",
            "overlap_warning":"Held out only from local adaptation, not proven absent from original pretraining/SFT."})
        write_json(directory/"tokenizer.json",{"engine":"huggingface","repo":MODEL_ID,"revision":MODEL_REVISION})
        progress("Measuring the original model")
        baseline={s:evaluate(model,items[s],device) for s in ["dev","test"]}
        baseline_samples=[{"prompt":p,"completion":sample(model,tokenizer,p,device,seed=settings["seed"]),"stage":"Before adaptation"} for p in PROMPTS]
        write_json(directory/"baseline.json",{"evaluation":baseline,"samples":baseline_samples})
        model=get_peft_model(model,LoraConfig(r=settings["rank"],lora_alpha=2*settings["rank"],lora_dropout=0.05,
            target_modules=["q_proj","v_proj"],bias="none",task_type="CAUSAL_LM"))
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
        manifest["model"].update(trainable_parameters=trainable,rank=settings["rank"])
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=settings["learning_rate"])
        writer=Telemetry(directory);sampler=random.Random(settings["seed"]);best=baseline["dev"]["nll"]
        # Save the zero-update adapter too: adaptation cannot silently replace a better baseline.
        model.save_pretrained(directory/"adapter",safe_serialization=True)
        best_step=0;exposures=0;train_started=time.perf_counter()
        metrics.append({"step":0,"dev_nll":baseline["dev"]["nll"],"tokens_processed":0,"elapsed_seconds":time.perf_counter()-started})
        for step in range(1,settings["steps"]+1):
            progress("Training the adapters",completed_steps=step-1)
            model.train();optimizer.zero_grad(set_to_none=True)
            chosen=[sampler.choice(items["train"]) for _ in range(settings["accumulation"])]
            valid_total=sum(sum(v!=-100 for v in item["labels"][1:]) for item in chosen)
            total=0.
            for item in chosen:
                x,y=tensors(item,device);valid=int((y[:,1:]!=-100).sum())
                loss=model(input_ids=x,labels=y,use_cache=False).loss
                if not torch.isfinite(loss):raise ValueError("Non-finite loss; training stopped.")
                (loss*valid/valid_total).backward();total+=float(loss.detach())*valid
            norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.))
            if not math.isfinite(norm):raise ValueError("Non-finite gradient; training stopped.")
            optimizer.step();exposures+=valid_total;synchronize(device)
            elapsed=time.perf_counter()-train_started
            row={"step":step,"sampled_train_nll":total/valid_total,"learning_rate":settings["learning_rate"],
                 "gradient_norm_pre_clip":norm,"tokens_processed":exposures,"tokens_per_second":exposures/max(elapsed,1e-6),
                 "elapsed_seconds":time.perf_counter()-started}
            if step==1 or step%10==0 or step==settings["steps"]:
                progress("Checking held-out responses",completed_steps=step)
                dev=evaluate(model,items["dev"],device);row["dev_nll"]=dev["nll"]
                if dev["nll"]<best:
                    best=dev["nll"];best_step=step;model.save_pretrained(directory/"adapter",safe_serialization=True)
            metrics.append(row);writer.log(row);write_json(directory/"metrics.json",metrics)
            progress("Training the adapters",completed_steps=step,tokens_processed=exposures,
                     tokens_per_second=row["tokens_per_second"],eta_seconds=elapsed/step*(settings["steps"]-step))
        model.save_pretrained(directory/"adapter-last",safe_serialization=True)
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model,load_file(str(directory/"adapter"/"adapter_model.safetensors")))
        progress("Evaluating the selected checkpoint",completed_steps=settings["steps"])
        final={s:evaluate(model,items[s],device) for s in ["dev","test"]}
        write_json(directory/"evaluation.json",{"checkpoint_step":best_step,"transformer":final,"pretrained":baseline,
            "selection":"Lowest local development loss, including the zero-update pretrained baseline.","units":"nats per assistant response token"})
        after=[{"prompt":p,"completion":sample(model,tokenizer,p,device,seed=settings["seed"]),"stage":"Selected adapter"} for p in PROMPTS]
        write_json(directory/"samples.json",baseline_samples+after)
        # Seal adapter contents through a top-level digest file as well as the common manifest.
        write_json(directory/"adapter_files.json",{str(p.relative_to(directory)):sha256(p) for p in directory.glob("adapter*/*") if p.is_file()})
        progress("Complete",status="MEASURED",best_checkpoint_step=best_step,training_seconds=time.perf_counter()-started,eta_seconds=0)
    except BaseException as exc:
        progress("Stopped" if isinstance(exc,KeyboardInterrupt) else "Failed",status="CANCELLED" if isinstance(exc,KeyboardInterrupt) else "FAILED",error=str(exc))
        raise
    finally:
        if writer:writer.close()
        seal(directory)
    return manifest


def generate_saved(payload):
    from peft import PeftModel
    directory=run_path(payload["run"]);verify(directory);manifest=read_json(directory/"manifest.json")
    if manifest["status"]!="MEASURED":raise ValueError("Select a completed adaptation.")
    for name,digest in read_json(directory/"adapter_files.json").items():
        if ".." in name.split("/") or sha256(directory/name)!=digest:raise ValueError("Adapter files changed.")
    prompt=payload["prompt"]
    if not isinstance(prompt,str) or not 1<=len(prompt)<=1000:raise ValueError("Use a prompt of 1–1000 characters.")
    for key,allowed in {"temperature":[0.,.3,.8,1.2],"top_k":[1,10,20,50],"tokens":[16,40,80,128],"seed":[7,17,29,43]}.items():
        if type(payload[key]) is bool or payload[key] not in allowed:raise ValueError("Unsupported generation control.")
    device=choose_device("auto");torch.set_num_threads(4)
    model,tokenizer=load_base(device);model=PeftModel.from_pretrained(model,directory/"adapter",is_trainable=False)
    text=sample(model,tokenizer,prompt,device,payload["temperature"],payload["top_k"],payload["tokens"],payload["seed"])
    result={**payload,"completion":text,"explanation":"Uses the selected local adapter. Sampling does not train weights.","device":device}
    import uuid
    write_json(RUNS/"_generations"/("adaptation-"+uuid.uuid4().hex+".json"),result)
    return result
