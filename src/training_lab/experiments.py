"""One registry for local controls, guided experiments, teaching and validation."""
import ast
import copy
import math
from pathlib import Path

from .artifacts import ROOT
from .materials import MATERIALS

VERSION = "local-lab-0.4"


def parameter(label, choices, default, group, explanation, equation, effects, interactions, code):
    return dict(label=label, choices=choices, default=default, group=group, explanation=explanation,
                equation=equation, effects=effects, interactions=interactions, code=code)


PARAMETERS = {
 "learning_rate": parameter("Learning rate", [.0003,.001,.003,.01,.03], .003, "Optimization",
  "How large an adjustment the optimizer attempts. Smaller is not automatically safer or better within a fixed training budget.",
  "w_next = w - η × update; w = weights, η = learning rate. AdamW rescales the gradient before this update.",
  ["Too small: loss may fall slowly.","Too large: updates may overshoot or become unstable.","The best value depends on the model and data; predict, then measure."],
  ["Batch size and accumulation change gradient noise.","Clipping limits the gradient, not the learning rate itself.","Warmup and the schedule change the rate over time."], "training.learning_rate_at"),
 "steps": parameter("Optimizer updates", [20,40,80,120,240,400,800], 120, "Optimization",
  "How many weight updates to perform. More updates reuse this small corpus; they do not create new information.",
  "token exposures ≈ updates × microbatch × accumulation × context; padding/short windows reduce the actual count.",
  ["More updates may improve fitting first and overfit later.","The best-dev checkpoint can be earlier than the last update."],
  ["Changing context, batch or accumulation also changes tokens seen per update.","At a fixed schedule horizon, extra updates can continue at the minimum rate."], "training.train"),
 "batch_size": parameter("Examples per microbatch", [1,2,4,8,16], 4, "Optimization",
  "How many sampled windows are processed together in one forward/backward pass.",
  "effective batch (windows) = microbatch size × accumulation; valid-token count can differ between windows.",
  ["Larger batches usually need more activation memory.","They can reduce gradient noise, but may not improve held-out performance."],
  ["At fixed steps, a larger batch sees more tokens: this is not a compute-matched comparison.","Accumulation can increase effective batch without holding all activations at once."], "training.train"),
 "accumulation": parameter("Gradient accumulation", [1,2,4], 1, "Optimization",
  "Combine gradients from several microbatches before updating the weights once.",
  "L_update = sum(all valid-token losses) / count(all valid tokens), not mean(microbatch means).",
  ["More accumulation generally reduces the gradient noise per update.","It adds work per update; it is not free extra training."],
  ["Compare equal effective batches when studying accumulation itself.","Dropout and sampling order can prevent bitwise equality in differently arranged real runs."], "training.train"),
 "optimizer": parameter("Optimizer", ["adamw","sgd"], "adamw", "Optimization",
  "The rule that converts gradients into weight changes. AdamW tracks moving statistics; plain SGD does not.",
  "SGD: w_next = w - ηg. AdamW: rescaled moving-gradient update plus decoupled weight decay.",
  ["The same learning rate need not suit both optimizers.","AdamW uses additional state for each parameter."],
  ["A one-rate comparison tests these configurations, not the universally best optimizer.","Weight decay in SGD is coupled L2 here; AdamW decay is decoupled."], "training.make_optimizer"),
 "weight_decay": parameter("Weight decay / L2 coefficient", [0.0,.01,.1], .01, "Optimization",
  "A pressure toward smaller weights. Here it applies to every parameter, including normalization and embeddings.",
  "AdamW decay contribution: w_next ≈ (1 - ηλ)w + gradient update; λ = decay coefficient.",
  ["May reduce overfitting; too much can suppress useful learning.","This is not the same mechanism as randomly dropping activations."],
  ["Its effect depends on learning rate and number of updates.","The implementation differs between AdamW and SGD."], "training.make_optimizer"),
 "clip_norm": parameter("Gradient clipping limit", [0.0,.5,1.0,5.0], 1.0, "Optimization",
  "Limit the total gradient norm before the optimizer step. Zero explicitly disables clipping.",
  "g_clipped = g × min(1, c / ||g||); c = clipping threshold.",
  ["Can limit a large gradient spike.","A small threshold can restrict useful updates; clipping is not a guarantee against divergence."],
  ["Inspect the recorded pre-clip norm.","A high learning rate can still cause trouble after gradients are clipped."], "training.clip_gradients"),
 "schedule": parameter("Learning-rate schedule", ["constant","cosine"], "constant", "Optimization",
  "How the learning rate changes after optional warmup. The horizon is fixed separately so restart does not silently change it.",
  "cosine multiplier = 0.1 + 0.9 × (1 + cos(πp))/2; p = progress after warmup, capped at 1.",
  ["Decay makes later updates smaller, which may stabilize fitting.","It can also slow learning too early."],
  ["Schedule horizon is measured in optimizer updates, not tokens.","Changing batch/context changes how much data is seen before decay."], "training.learning_rate_at"),
 "warmup_steps": parameter("Warmup updates", [0,10,20,40], 0, "Optimization",
  "Start at a fraction of the peak learning rate, then increase linearly.",
  "During warmup: η(step) = peak_rate × step / warmup_updates.",
  ["Can soften early updates.","Too much warmup in a short run can spend most of the run learning slowly."],
  ["Warmup must be shorter than the schedule horizon.","Warmup also applies before a constant post-warmup schedule."], "training.learning_rate_at"),
 "schedule_steps": parameter("Schedule horizon", [80,160,240,400,800], 240, "Optimization",
  "The update at which cosine decay reaches its minimum. It is independent of when a run is interrupted.",
  "p = clamp((step - warmup) / (horizon - warmup), 0, 1).",
  ["A longer horizon decays more slowly.","With a constant schedule and no warmup it does not affect updates."],
  ["Keep the horizon fixed when checking resume equality.","Interpret this together with total updates and token exposure."], "training.learning_rate_at"),
 "dropout": parameter("Dropout", [0.0,.1,.3,.5], .1, "Architecture",
  "Randomly zero some attention/MLP activations during training. It is disabled for evaluation and generation.",
  "training output = mask × activation / (1-p); p = dropout probability.",
  ["Can discourage reliance on particular activations.","Too much can make fitting difficult; it does not add data."],
  ["Larger models/smaller datasets may overfit more.","Training loss with dropout is not directly the same estimator as full evaluation-mode loss."], "model.Block"),
 "block_size": parameter("Context window (tokens)", [16,32,64,128], 64, "Architecture",
  "The maximum number of input tokens in one training window. Longer documents are chunked without crossing document boundaries.",
  "attention score elements = batch × heads × T²; T = context length.",
  ["Longer context can expose more preceding information.","It increases attention cost and changes window/chunk boundaries."],
  ["A tokenizer change alters how much text fits into T tokens.","At fixed updates, context also changes token exposure and position-embedding count."], "model.Attention.forward"),
 "width": parameter("Hidden width", [32,64,128], 64, "Architecture",
  "How many numbers describe a token inside the model.",
  "A dense decoder block has roughly 12d² matrix weights; d = hidden width.",
  ["More width increases capacity and compute, not guaranteed held-out quality.","A tiny corpus may let a wider model memorize faster."],
  ["Width must be divisible by attention heads.","Width and number of layers jointly change parameter count and optimization."], "model.TinyDecoder"),
 "layers": parameter("Decoder layers", [1,2,4], 2, "Architecture",
  "How many attention-plus-MLP blocks process the representation in sequence.",
  "x_(layer+1) = block_layer(x_layer); deeper models repeat this computation.",
  ["Adds representational capacity and training work.","More depth can make optimization harder."],
  ["Residual paths and normalization help signals travel through depth.","This is not a parameter-matched depth comparison unless width is adjusted."], "model.Block"),
 "heads": parameter("Attention heads", [1,2,4,8], 4, "Architecture",
  "Split the hidden representation into several attention calculations.",
  "head width = model width / heads; attention = softmax(QKᵀ/√head_width + causal_mask)V.",
  ["At fixed model width, extra heads mean narrower heads—not a proportionally bigger projection.","Different heads may learn different patterns, but need not be human-interpretable."],
  ["Must divide width exactly.","The explicit attention-score tensor grows with head count."], "model.Attention.forward"),
 "positions": parameter("Position embeddings", [True,False], True, "Architecture",
  "Add a learned vector for each location. Turning this off removes explicit learned position vectors, not every possible order cue.",
  "x_t = token_embedding(token_t) + position_embedding(t), when enabled.",
  ["May help distinguish locations/order.","Without it, the causal attention pattern still depends on position; do not claim the model has no order information."],
  ["Changing context changes the position table size.","Chunking resets learned position indices."], "model.TinyDecoder.forward"),
 "residuals": parameter("Residual connections", [True,False], True, "Architecture",
  "Add each block's input back to its transformed output. Turning them off creates an educational ablation.",
  "with residual: x_next = x + f(x); without residual: x_next = f(x).",
  ["Can make optimization and information flow easier.","Removing them may hurt deeper networks; a tiny run is not a universal proof."],
  ["Effects depend strongly on depth and normalization.","No weights are added simply by the identity connection."], "model.Block.forward"),
 "normalization": parameter("Normalization", ["layer","rms","none"], "layer", "Architecture",
  "Rescale a token's hidden features before each sublayer. LayerNorm also subtracts the feature mean; RMSNorm does not.",
  "RMSNorm(x) = gain × x / √(mean(x²)+ε); LayerNorm centers x before scaling.",
  ["May help keep optimization well behaved.","Removing normalization can alter gradient scales; it is not only a cosmetic architecture choice."],
  ["Learning rate, residuals and depth interact with stability.","LayerNorm and RMSNorm have different learned parameter counts here."], "model.make_norm"),
 "merges": parameter("BPE merge operations", [0,16,32,48,96], 32, "Data",
  "How many frequent byte pairs the tokenizer may merge, fitted on training documents only. Zero means UTF-8 byte tokens.",
  "vocabulary = 259 + completed_merges (three special IDs plus 256 bytes).",
  ["More merges often shorten token sequences but expand the vocabulary.","Changing the tokenizer changes the unit of per-token loss."],
  ["A fixed token context covers different amounts of text.","Compare bits per byte only on identical held-out text and read its boundary/context caveats."], "tokenization.ByteBPE.fit"),
 "train_fraction": parameter("Training document fraction", [0.25,0.5,1.0], 1.0, "Data",
  "Use a deterministic prefix of the authored training documents. Dev/test text remains fixed. This is not a random representative sample.",
  "selected_train_docs = max(1, floor(full_train_docs × fraction)).",
  ["Less data may encourage memorization and reduce topic coverage.","More updates on fewer documents repeat information rather than add it."],
  ["The tokenizer is refitted on the selected train documents.","This intervention changes both data coverage and possibly tokenization; those are not isolated automatically."], "training.select_training_data"),
 "seed": parameter("Random seed", [7,17,29,43], 17, "Reproducibility",
  "Controls initialization, sampling and dropout for this backend. One seed is not an uncertainty estimate.",
  "same code + config + data + restored state → tested CPU repeatability; not universal hardware determinism.",
  ["Different seeds can lead to different outcomes.","Repeated runs help reveal variation but three seeds are still limited evidence."],
  ["Changing architecture consumes random numbers differently.","Changing backend/library may change numerical behavior."], "training.train"),
 "device": parameter("Local device", ["auto","cpu","cuda","mps"], "auto", "Reproducibility",
  "Auto checks NVIDIA CUDA, Apple Metal, then CPU. Select CPU for a reference run. No remote backend, API or cluster is invoked.",
  "measured elapsed time ≠ portable speed; backend kernels and synchronization matter.",
  ["CPU is the tested reference.","GPU support depends on the installed PyTorch build, drivers and model operations; unavailable explicit requests fail."],
  ["Tiny models may spend more time in overhead than arithmetic.","Comparing devices is not purely a model-quality experiment."], "hardware.choose_device"),
}

DEFAULTS = {key: p["default"] for key,p in PARAMETERS.items()}


def guided(id, title, question, parameter_key, values, labels, prediction, lesson, overrides=None):
    return dict(id=id, title=title, question=question, parameter=parameter_key,
                variations=[dict(label=label,value=value) for label,value in zip(labels,values)],
                prediction=prediction, lesson=lesson, overrides=overrides or {})


GUIDES = [
 guided("first","First run: watch a model learn","How do text targets become weight changes?","steps",[20,40,120],["20 updates: quick check","40 updates: short fit","120 updates: longer fit"],"Will held-out loss keep falling as training continues?","training_basics"),
 guided("rate","Learning rate: slow, balanced or aggressive?","What changes when the optimizer takes bigger steps?","learning_rate",[.0003,.003,.03],["Small adjustments","Reference rate","Aggressive adjustments"],"Which run will learn fastest, and will that be the best on dev?","optimization"),
 guided("overfit","More training: learning or memorizing?","When does more fitting stop helping held-out text?","steps",[40,120,240],["40 updates","120 updates","240 updates"],"Sketch the train and dev curves before running.","generalization"),
 guided("dropout","Dropout: remove some training activations","Does random dropping help on this tiny corpus?","dropout",[0.0,.1,.3],["No dropout","Light dropout","Stronger dropout"],"Will dropout make training loss look worse while helping dev? It may not.","optimization"),
 guided("batch","Batch size: more examples per update","Does a less noisy update help at the same number of updates?","batch_size",[2,4,8],["2 windows","4 windows","8 windows"],"Remember: equal updates will NOT mean equal token exposure.","optimization"),
 guided("accumulate","Accumulation: build a bigger effective batch","What changes when gradients are combined before updating?","accumulation",[1,2,4],["One microbatch","Two microbatches","Four microbatches"],"Predict token exposure and runtime before predicting accuracy.","optimization"),
 guided("context","Context: how far can the model look?","Does a larger window help this particular corpus?","block_size",[16,32,64],["16-token window","32-token window","64-token window"],"What happens to attention cost, chunk boundaries and text coverage?","attention"),
 guided("width","Capacity: make the model wider","Does extra capacity learn useful patterns or memorize?","width",[32,64,128],["Width 32","Width 64","Width 128"],"A wider model costs more; a lower train loss alone is not a win.","attention"),
 guided("depth","Depth: stack more decoder blocks","Does depth help when the data is so small?","layers",[1,2,4],["One layer","Two layers","Four layers"],"How might residuals and normalization change this outcome?","attention"),
 guided("heads","Heads: divide attention differently","What do extra heads mean when total width stays fixed?","heads",[1,4,8],["One wider head","Four heads","Eight narrower heads"],"Does head count multiply parameter count at fixed width?","attention"),
 guided("tokenizer","Tokenizer: bytes versus merged pieces","What changes when a token means a different amount of text?","merges",[0,32,96],["UTF-8 bytes","32 learned merges","96 learned merges"],"Do not rank these using per-token loss; inspect bits per byte and context limits.","tokenization"),
 guided("optimizer","Optimizer: SGD versus AdamW","How does the update rule affect this run?","optimizer",["sgd","adamw"],["Plain SGD","AdamW"],"The same rate is not separately tuned for each optimizer.","optimization"),
 guided("schedule","Schedule: constant versus cosine decay","Do smaller late updates help?","schedule",["constant","cosine"],["Keep the rate constant","Cosine decay"],"Which learning-rate curve do you expect before looking at losses?","optimization",{"steps":240,"schedule_steps":240}),
 guided("warmup","Warmup: ease into the updates","Can gentler initial updates help at this scale?","warmup_steps",[0,10,40],["No warmup","10-update warmup","40-update warmup"],"How much of a 120-update run is spent warming up?","optimization"),
 guided("decay","Weight decay: pull weights toward zero","Does stronger regularization help held-out text?","weight_decay",[0.0,.01,.1],["No decay","Light decay","Stronger decay"],"Why does the effect also depend on learning rate and training length?","optimization"),
 guided("positions","Ablation: remove position embeddings","What explicit location signal does the model lose?","positions",[True,False],["Learned positions on","Learned positions off"],"The causal mask still supplies an order-dependent pattern.","attention"),
 guided("residuals","Ablation: remove residual paths","How important are identity paths to optimization?","residuals",[True,False],["Residuals on","Residuals off"],"Expect an interaction with depth; this is not a universal architecture verdict.","attention"),
 guided("normalization","Ablation: LayerNorm, RMSNorm or none","How does rescaling hidden features change training?","normalization",["layer","rms","none"],["Layer normalization","RMS normalization","No normalization"],"Watch gradients and dev loss, not just final training fit.","attention"),
 guided("data","Data: fewer training documents","How much can a small, fixed subset teach the model?","train_fraction",[.25,.5,1.0],["Quarter of train documents","Half of train documents","All train documents"],"Dev/test stay fixed, but the refitted tokenizer can change too.","data"),
 guided("seeds","Repeatability: try different seeds","How sensitive is this setup to randomness?","seed",[7,17,29],["Seed 7","Seed 17","Seed 29"],"A single best run is not an average effect or confidence interval.","generalization"),
]

LESSONS = {
 "training_basics": ("Next-token objectives and lifecycle", "llm_training/01_OBJECTIVES_AND_LIFECYCLE.md"),
 "optimization": ("Stable training and debugging", "llm_training/05_STABLE_TRAINING_AND_DEBUGGING.md"),
 "generalization": ("Learning and generalization", "pre_llm/04_LEARNING_AND_GENERALIZATION.md"),
 "attention": ("Transformer from first principles", "llm_training/03_TRANSFORMER_FROM_FIRST_PRINCIPLES.md"),
 "tokenization": ("Data and tokenizer engineering", "llm_training/02_DATA_AND_TOKENIZER_ENGINEERING.md"),
 "data": ("Data and tokenizer engineering", "llm_training/02_DATA_AND_TOKENIZER_ENGINEERING.md"),
 "generation": ("Prompting, reasoning and structured output", "post_training/03_PROMPTING_REASONING_AND_STRUCTURED_OUTPUT.md"),
}


def validate(overrides=None):
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict) or set(overrides) - set(PARAMETERS):
        raise ValueError("Unknown or malformed settings.")
    result = dict(DEFAULTS)
    for key,value in overrides.items():
        # Python considers True == 1: explicitly separate booleans from numbers.
        choices = PARAMETERS[key]["choices"]
        matches=[c for c in choices if value==c and (type(value) is type(c) or
                 type(value) in {int,float} and type(c) in {int,float})]
        if not matches:
            raise ValueError(f"{key}: choose one of {choices}.")
        result[key] = matches[0]  # JSON has one numeric type; preserve the registry's canonical type.
    if result["width"] % result["heads"]:
        raise ValueError("Width must be divisible by attention heads.")
    if result["warmup_steps"] >= result["schedule_steps"]:
        raise ValueError("Warmup must be shorter than the schedule horizon.")
    # Conservative work proxy, NOT a FLOP or memory measurement.
    work = result["steps"] * result["batch_size"] * result["accumulation"] * result["block_size"] * result["layers"] * result["width"]**2
    if work > 20_000_000_000:
        raise ValueError("This combination exceeds the local work ceiling. Reduce updates, batch, context or model size.")
    return result


def estimate(settings):
    s = validate(settings)
    return {"effective_windows_per_update": s["batch_size"]*s["accumulation"],
            "maximum_token_exposures": s["steps"]*s["batch_size"]*s["accumulation"]*s["block_size"],
            "attention_scores_per_layer_microbatch": s["batch_size"]*s["heads"]*s["block_size"]**2,
            "note": "Upper bounds/proxies, not measured RAM, FLOPs, wall time or unique data. Short windows reduce valid tokens."}


def make_plan(guide_id, variation="all", overrides=None, prediction=""):
    guide = next((g for g in GUIDES if g["id"] == guide_id), None)
    if guide is None:
        raise ValueError("Unknown guided experiment.")
    if not isinstance(prediction,str) or len(prediction)>2000:
        raise ValueError("Prediction must be text of at most 2000 characters.")
    if overrides is not None and not isinstance(overrides,dict):
        raise ValueError("Overrides must be a settings object.")
    base = validate({**guide["overrides"], **(overrides or {})})
    if variation == "all":
        variations = guide["variations"]
    elif type(variation) is int and 0 <= variation < len(guide["variations"]):
        variations = [guide["variations"][variation]]
    else:
        raise ValueError("Choose all or a valid variation index.")
    configs = [{"label": v["label"], "settings": validate({**base,guide["parameter"]:v["value"]})} for v in variations]
    return {"registry_version": VERSION, "guide": guide_id, "question": guide["question"],
            "varied_parameter": guide["parameter"], "prediction": prediction,
            "runs": [{**c,"estimate":estimate(c["settings"])} for c in configs],
            "limits": ["One seed per variation unless studying seeds; not a confidence interval.",
                       "Equal optimizer updates do not necessarily mean equal tokens, compute or capacity.",
                       "This is the authored miniature corpus, not a general-purpose model benchmark."]}


def code_excerpt(reference):
    module, qualified = reference.split(".",1)
    if module not in {"training","model","tokenization","hardware"}:
        raise ValueError("Unsupported source module.")
    file = ROOT / "src" / "training_lab" / (module+".py")
    source = file.read_text(encoding="utf-8")
    nodes = ast.parse(source).body
    node = None
    for part in qualified.split("."):
        node = next((n for n in nodes if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and n.name == part),None)
        if node is None:
            return {"file":str(file.relative_to(ROOT)),"code":"Source section is unavailable in this version."}
        nodes = getattr(node,"body",[])
    lines=source.splitlines()
    return {"file":str(file.relative_to(ROOT)),"line":node.lineno,
            "code":"\n".join(lines[node.lineno-1:node.end_lineno])}


def catalog():
    from .source_code import language
    parameters = copy.deepcopy(PARAMETERS)
    for p in parameters.values():
        p["source"] = code_excerpt(p["code"])
        p["lesson"] = {"Architecture":"attention","Data":"tokenization","Optimization":"optimization",
                       "Reproducibility":"generalization"}[p["group"]]
    return {"version":VERSION,"parameters":parameters,"defaults":DEFAULTS,"guides":GUIDES,"sources":language("scratch"),
            "lessons":{k:{"title":v[0]} for k,v in LESSONS.items()},
            "scope":"This registry configures the local random-initialized decoder. A separate lab supports pretrained LoRA adaptation. Full LLM preference/RL training is not implemented."}


def lesson(identifier):
    if identifier not in LESSONS:
        raise ValueError("Unknown lesson.")
    title,relative = LESSONS[identifier]
    return {"title":title,"markdown":(MATERIALS/relative).read_text(encoding="utf-8")}
