"""Bounded read-only forward traces. Never modify or train the saved checkpoint."""
import math
import torch
from .artifacts import RUNS,read_json,run_path,verify


@torch.no_grad()
def capture(identifier,text,runs_root=RUNS):
    if not isinstance(text,str) or not 1<=len(text)<=160:raise ValueError("Use 1–160 characters for a trace.")
    directory=run_path(identifier,runs_root);verify(directory);manifest=read_json(directory/"manifest.json")
    if manifest["status"]!="MEASURED":raise ValueError("Select a completed model.")
    torch.set_num_threads(4)
    if manifest.get("engine")=="lora":
        from .adaptation import load_base
        from peft import PeftModel
        model,tokenizer=load_base("cpu")
        model=PeftModel.from_pretrained(model,directory/"adapter",is_trainable=False).eval()
        ids=tokenizer.encode(text,add_special_tokens=False)[:24]
        if not ids:raise ValueError("No text tokens.")
        projected={}
        attention=model.get_base_model().model.layers[0].self_attn
        hooks=[getattr(attention,name).register_forward_hook(lambda module,args,output,name=name:projected.__setitem__(name,output.detach())) for name in ["q_proj","k_proj","v_proj"]]
        x=torch.tensor([ids])
        try:result=model(input_ids=x,output_attentions=True,output_hidden_states=True,use_cache=False)
        finally:
            for hook in hooks:hook.remove()
        weights=result.attentions[0][0,0]
        embedding=result.hidden_states[0][0]
        logits=result.logits[0,-1];p,idx=logits.softmax(-1).topk(8)
        return {"run":identifier,"kind":"measured","architecture":"SmolLM2: rotary positions and grouped-query attention",
            "model_config":{k:getattr(model.config,k) for k in ["hidden_size","intermediate_size","num_hidden_layers","num_attention_heads","num_key_value_heads","rope_theta","rms_norm_eps"]},
            "query":projected["q_proj"][0,:,:8].tolist(),"key":projected["k_proj"][0,:,:8].tolist(),"value":projected["v_proj"][0,:,:8].tolist(),
            "projection_note":"First head, first eight projected components. Query/key slices are before rotary position rotation; the attention map is after the actual model's rotary attention computation.",
            "tokens":[tokenizer.decode([i]) for i in ids],"ids":ids,"embeddings":embedding[:,:8].tolist(),
            "attention":weights.tolist(),"next_tokens":[{"token":tokenizer.decode([int(i)]),"probability":float(v)} for i,v in zip(idx,p)],
            "stages":["Text pieces","Learned vectors","Rotary attention","Decoder layers","Next-token probabilities"],
            "note":"Recorded forward calculation, first layer/head, up to 24 tokens; vectors show their first 8 dimensions. Attention is not proof of causal reasoning. No chat template was added to this raw-text inspection."}
    from .training import load_model
    model,tokenizer,_=load_model(directory);model.eval()
    ids=([1]+tokenizer.encode(text))[:min(12,model.config.block_size)];xids=torch.tensor([ids])
    embeddings=model.tokens(xids);x=embeddings.clone()
    if model.positions is not None:x=x+model.positions(torch.arange(len(ids)))
    block=model.blocks[0];norm=block.norm1(x)
    q,k,v=block.attention.qkv(norm).chunk(3,dim=-1)
    h=model.config.heads;size=model.config.width//h
    q,k,v=[a.view(1,len(ids),h,size).transpose(1,2) for a in [q,k,v]]
    scores=q@k.transpose(-2,-1)/(size**.5)
    mask=torch.ones(len(ids),len(ids),dtype=torch.bool).triu(1)
    probabilities=scores.masked_fill(mask,float("-inf")).softmax(-1)
    logits,_=model(xids,inspect=True);p,idx=logits[0,-1].softmax(-1).topk(8)
    return {"run":identifier,"kind":"measured","architecture":"Local pre-norm decoder",
        "model_config":vars(model.config),
        "tokens":[tokenizer.label(i) for i in ids],"ids":ids,"embeddings":embeddings[0,:,:8].tolist(),
        "query":q[0,0,:,:8].tolist(),"key":k[0,0,:,:8].tolist(),"value":v[0,0,:,:8].tolist(),
        "scores":scores[0,0].tolist(),"attention":probabilities[0,0].tolist(),
        "next_tokens":[{"token":tokenizer.label(int(i)),"probability":float(value)} for i,value in zip(idx,p)],
        "stages":["Text pieces","Token + position vectors","Q / K / V","Causal attention","Next-token probabilities"],
        "note":"Recorded from this checkpoint, first layer/head, up to 12 positions. Vector displays show at most 8 dimensions. The computation uses full vectors. Attention is not a causal explanation."}


def capture_update(identifier,text,runs_root=RUNS):
    """One diagnostic SGD step on a freshly loaded CPU model. Never save it."""
    if not isinstance(text,str) or not 1<=len(text)<=160:
        raise ValueError("Use 1–160 characters for a trace.")
    directory=run_path(identifier,runs_root);verify(directory)
    manifest=read_json(directory/"manifest.json")
    if manifest["status"]!="MEASURED" or manifest.get("engine")=="lora":
        raise ValueError("This bounded update inspector is for completed scratch models only.")
    if manifest["model"]["parameters"]>5_000_000:
        raise ValueError("The update inspector is limited to scratch models below five million parameters.")
    from .training import load_model
    from .model import loss_sum
    trace=capture(identifier,text,runs_root)
    model,tokenizer,_=load_model(directory);model.eval()  # Disable dropout for a matched before/after comparison.
    ids=([1]+tokenizer.encode(text))[:min(13,model.config.block_size+1)]
    if len(ids)<2:raise ValueError("Provide enough text for a next-token target.")
    x=torch.tensor([ids[:-1]]);targets=torch.tensor([ids[1:]])
    weight=model.blocks[0].attention.qkv.weight
    before=weight[:8,:8].detach().clone()
    with torch.enable_grad():
        total,count=loss_sum(model(x),targets);loss=total/count;loss.backward()
        gradient=weight.grad[:8,:8].detach().clone()
        total_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),float("inf")))
        if not torch.isfinite(loss) or not math.isfinite(total_norm):
            raise ValueError("Non-finite diagnostic gradient.")
        optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        optimizer.step()
    with torch.no_grad():
        total,count=loss_sum(model(x),targets);after_loss=total/count
    trace["training_update"]={"method":"SGD", "learning_rate":.01,"dropout":False,
        "target_tokens":len(ids)-1,"loss_before":float(loss.detach()),"loss_after":float(after_loss),
        "gradient_norm":total_norm,"parameter":"blocks.0.attention.qkv.weight",
        "before":before.tolist(),"gradient":gradient.tolist(),"after":weight[:8,:8].detach().tolist(),
        "note":"First 8 output rows × 8 input columns of the first QKV projection. All trainable weights received one diagnostic SGD update on this short text, in evaluation mode. This is not a replay of the run's original optimizer or batch. The model copy is discarded; the checkpoint is unchanged."}
    trace["note"]+=" Includes a one-step diagnostic on a disposable CPU model; no files were written."
    return trace
