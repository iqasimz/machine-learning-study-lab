"""Read-only, allowlisted access to the existing course and interview Markdown."""
import hashlib
import json
from pathlib import Path
import re

from .artifacts import ROOT

from .materials import MATERIALS
READER_PROFILES = json.loads((MATERIALS/"reader_profiles.json").read_text(encoding="utf-8"))["lessons"]

# Orientation text is new; the actual lessons/answers remain their original files.
LESSONS = [
 ("P2","basics","pre_llm/02_LINEAR_ALGEBRA_AND_CALCULUS.md","Linear algebra and calculus","Understand vectors, matrix shapes and how a small change affects a result.",["Numbers","Vectors and matrices","Prediction","Gradient"],"width"),
 ("P3","basics","pre_llm/03_PROBABILITY_INFORMATION_AND_STATISTICS.md","Probability and statistics","Learn what a probability, loss and uncertain result actually mean.",["Possible answers","Probabilities","Observed answer","Measure the mistake"],"rate"),
 ("P4","basics","pre_llm/04_LEARNING_AND_GENERALIZATION.md","Learning and generalization","Keep fitting, model selection and final checking separate.",["Training text","Practice check","Choose a checkpoint","Final check"],"overfit"),
 ("P5","basics","pre_llm/05_NEURAL_NETWORKS_AND_OPTIMIZATION.md","Neural networks and optimization","Connect a prediction, an error, a gradient and a weight update.",["Guess","Measure error","Find gradients","Adjust learned numbers"],"optimizer"),
 ("P8","basics","pre_llm/08_EVALUATION_METRICS.md","Evaluation metrics","Inspect the cases and denominators behind classification, regression, retrieval and language-model scores.",[],None),
 ("P6","basics","pre_llm/06_LANGUAGE_TOKENIZATION_AND_NLP.md","Language and tokenization","Understand tokens and why a text task needs a clear definition.",["Text","Pieces of text","Token IDs","Model input"],"tokenizer"),
 ("P7","basics","pre_llm/07_EMBEDDINGS_AND_RETRIEVAL_FOUNDATIONS.md","Embeddings and retrieval","See how numerical representations help retrieve useful documents.",["Question","Representation","Search documents","Check relevance"],None),
 ("T0","training","llm_training/00_THE_COMPLETE_TRAINING_LOOP.md","The complete training loop","Follow preparation, prediction, gradients, updates, validation and saved-model use in one place.",["Prepare","Predict and score","Update","Validate and save"],"first"),
 ("T1","training","llm_training/01_OBJECTIVES_AND_LIFECYCLE.md","Training objectives and lifecycle","See where next-token targets come from and what training changes.",["Earlier text","Predict next piece","Correct next piece","Update weights"],"first"),
 ("T2","training","llm_training/02_DATA_AND_TOKENIZER_ENGINEERING.md","Training data and tokenizers","Separate documents, fit a tokenizer and avoid accidental answer leakage.",["Source documents","Check and split","Fit tokenizer on train","Build windows"],"data"),
 ("T3","training","llm_training/03_TRANSFORMER_FROM_FIRST_PRINCIPLES.md","Transformer architecture","Trace embeddings, attention and the next-token prediction.",["Token IDs","Embeddings","Attention and MLP","Next-token scores"],"heads"),
 ("T4","training","llm_training/04_ARCHITECTURES_AND_MODALITIES.md","Architecture families","Compare architecture choices without assuming more complexity is better.",["Architecture choice","Capacity and cost","Train a controlled run","Check the trade-off"],"positions"),
 ("T5","training","llm_training/05_STABLE_TRAINING_AND_DEBUGGING.md","Optimization and training stability","Read learning curves and diagnose training problems.",["Configure","Train","Check loss and gradients","Repair or continue"],"rate"),
 ("T6","training","llm_training/06_SCALING_COMPUTE_AND_MEMORY.md","Compute and memory","Separate estimates from measurements before scaling up.",["Model and data size","Estimate resources","Measure locally","Choose a safe budget"],"width"),
 ("T7","training","llm_training/07_DISTRIBUTED_TRAINING_AND_KERNELS.md","Distributed training","Learn communication and parallelism before attempting a cluster run.",["One correct worker","Split the work","Communicate","Check correctness and speed"],None),
 ("T8","training","llm_training/08_EVALUATION_INTERPRETABILITY_AND_RELEASE.md","Evaluation and interpretability","Evaluate a checkpoint and explain what the evidence does not prove.",["Freeze the comparison","Measure held-out cases","Inspect failures","Report limits"],"seeds"),
 ("R1","post","post_training/01_SFT_AND_PARAMETER_EFFICIENT_ADAPTATION.md","Fine-tuning and LoRA","Understand fine-tuning and adapters before changing a capable model.",["Base model","Reviewed examples","Update selected weights","Compare with the base"],None),
 ("R2","post","post_training/02_PREFERENCES_RLHF_DPO_PPO_AND_GRPO.md","Preference optimization and RLHF","Distinguish DPO, PPO and GRPO by their training signals.",["Candidate responses","Preference or reward","Policy update","Check for reward shortcuts"],None),
 ("R3","post","post_training/03_PROMPTING_REASONING_AND_STRUCTURED_OUTPUT.md","Prompting, reasoning and decoding","See how prompts and decoding affect output without necessarily changing weights.",["Prompt","Model probabilities","Sampling rules","Generated answer"],"generation"),
 ("R4","post","post_training/04_RAG_AND_VECTOR_DATABASES.md","RAG and vector databases","Understand retrieval, sources and vector search.",["Question","Retrieve permitted evidence","Build context","Answer with support"],None),
 ("R5","post","post_training/05_AGENTS_TOOLS_AND_MCP.md","Agents, tools and MCP","Trace requests, permissions and tool results instead of trusting an autonomous label.",["User request","Allowed tool","Validate and authorize","Inspect the result"],None),
 ("R6","post","post_training/06_INFERENCE_AND_SERVING.md","Inference and serving","Understand prompt processing, token generation, caching and waiting time.",["Incoming request","Process the prompt","Generate tokens","Measure response time"],"generation"),
 ("R7","post","post_training/07_COMPRESSION_EDGE_AND_DEPLOYMENT.md","Compression and deployment","Study the quality, memory and speed trade-offs of compression.",["Same model/task","Compression choice","Measure quality and cost","Choose a deployment"],None),
 ("R8","post","post_training/08_PRODUCTION_EVALUATION_SECURITY_AND_OPERATIONS.md","Production evaluation and security","Connect evaluation, security, monitoring and recovery.",["Release checks","Limited rollout","Monitor failures","Repair or roll back"],"compare"),
 ("R9","post","post_training/09_ADVANCED_BRANCHES_AND_RESEARCH_JUDGMENT.md","Advanced research methods","Classify unfamiliar methods by what changes and what evidence would justify them.",["New method","What does it change?","What would test it?","Evidence and limits"],None),
]

# The authored ML/DL metadata is a local allowlist, never a request-provided file path.
ML_DL = MATERIALS / "ml_dl"
ML_DL_LESSONS = json.loads((ML_DL/"metadata/lessons.json").read_text())["lessons"]
ML_DL_PARAMETERS = {p["id"]:p for p in json.loads((ML_DL/"metadata/parameters.json").read_text())["parameters"]}
for row in ML_DL_LESSONS:
    LESSONS.append((row["id"],row["stage"],"ml_dl/"+row["file"],row["title"]," ".join(row["learning_outcomes"]),[],None))
NLP_LESSONS=json.loads((MATERIALS/"nlp/metadata/lessons.json").read_text())["lessons"]
for row in NLP_LESSONS:
    LESSONS.append((row["id"],"nlp","nlp/"+row["file"],row["title"],row["goal"],[],None))
LESSONS = [(i,"language" if i in {"P6","P7"} else s,r,t,g,n,l) for i,s,r,t,g,n,l in LESSONS]
_stage_order = {s:i for i,s in enumerate(["basics","ml","dl","language","nlp","training","post"])}
LESSONS.sort(key=lambda row:_stage_order[row[1]])
LESSONS = [(i,s,r,READER_PROFILES[i]["title"],READER_PROFILES[i]["definition"],n,l)
           if i in READER_PROFILES else (i,s,r,t,g,n,l) for i,s,r,t,g,n,l in LESSONS]

STAGES = {
 "basics":{"title":"Shared foundations","description":"One focused section on mathematical shapes, uncertainty, experimental boundaries, optimization and evaluation metrics."},
 "ml":{"title":"Machine learning","description":"Sixteen lessons connecting targets, preprocessing, classical methods, selection, uncertainty and complete experiments."},
 "dl":{"title":"Deep learning","description":"Fourteen lessons on backpropagation, architectures, optimization, representations and training systems. Executable local architectures are listed in the training lab."},
 "language":{"title":"Language foundations","description":"Canonical tokenization and embedding foundations shared by NLP and LLMs; maintained once under NLP."},
 "nlp":{"title":"Natural language processing","description":"Twenty-four lessons from annotation and lexical baselines through recurrent/structured/contextual models, sentence encoders, tasks and transfer. Calculations are authored examples, not trained benchmark results."},
 "training":{"title":"LLM training","description":"Begin with the complete training loop, then follow the eight detailed training lessons. The scratch and pretrained-adapter paths have different purposes."},
 "post":{"title":"Post-training and systems","description":"Read nine lessons on adaptation and real-world systems. Local LoRA adaptation runs here; actual preference/RL training and deployed assistant systems are still planned."},
}

INTERVIEW_FILES = {
 "nlp":("NLP interviews","nlp/INTERVIEW_QUESTIONS.md"),
 "ml":("ML interviews","ml_dl/INTERVIEW_QUESTIONS.md"),
 "dl":("DL interviews","ml_dl/INTERVIEW_QUESTIONS.md"),
 "foundations":("Foundations and code","interview_preparation/FOUNDATIONS_AND_CODE.md"),
 "models":("Models and training","interview_preparation/MODELS_TRAINING_AND_ALIGNMENT.md"),
 "systems":("Applications and safety","interview_preparation/RAG_AGENTS_AND_SECURITY.md"),
 "design":("System-design conversations","interview_preparation/LLM_SYSTEM_DESIGN_CASEBOOK.md"),
 "practice":("Practical exercises","interview_preparation/PRACTICAL_ASSESSMENTS.md"),
}
RESOURCES = {
 "ml-dl-terms":("ML/DL notation and definitions","ml_dl/DEFINITIONS_AND_NOTATION.md"),
 "ml-dl-comparisons":("Method comparisons","ml_dl/METHOD_COMPARISONS.md"),
 "ml-dl-assessments":("ML/DL exercises","ml_dl/ASSESSMENTS.md"),
 "ml-dl-reading":("ML/DL primary sources","ml_dl/research/READING_ROUTE.md"),
 "math":("Maths and terms reference","MATH_AND_TERMS_REFERENCE.md"),
 "resources":("Books, sources and interactive resources","RESEARCH_AND_RESOURCES.md"),
 "study-plan":("Study plan","STUDY_PLAN.md"),
 "assessments":("Assessments and exercises","ASSESSMENTS.md"),
 "coverage":("Coverage and limits","COVERAGE_AND_LIMITS.md"),
 "course-map":("Full course map","COURSE_MAP.md"),
 "interview-guide":("How to practise for interviews","interview_preparation/README.md"),
 "interview-coverage":("Interview coverage matrix","interview_preparation/COVERAGE_MATRIX.md"),
 "readiness":("Independent readiness tracker","interview_preparation/READINESS_TRACKER.md"),
}

GLOSSARY = {
 "Token":"A piece of text represented by an ID. Depending on the tokenizer it can be part of a word, a whole word, a symbol or a byte.",
 "Weights / model parameters":"Numbers learned during training. They are different from the settings you choose, such as learning rate or batch size.",
 "Training setting / hyperparameter":"A choice made before or during an experiment, such as model size, learning rate or number of updates.",
 "Training":"Repeatedly predict a target, measure the mistake and update learned numbers.",
 "Training split":"The documents used to fit the tokenizer/model. Evaluation documents must stay outside fitting.",
 "Development / validation split":"Practice-check documents used to choose configurations or checkpoints. They are not the final independent check.",
 "Test split":"Held-out documents intended for a final check. Repeatedly choosing experiments from their scores makes them part of development.",
 "Loss / NLL":"How much the model is penalized for giving too little probability to the correct next piece. Lower is better for a matched task/tokenizer; it is not an answer-trustworthiness score.",
 "Perplexity":"The exponential of average next-token loss. Its units depend on tokenization, so do not compare different tokenizers blindly.",
 "Bits per byte":"Total prediction cost measured per raw text byte. It can help compare tokenizers on identical text, with context/boundary/budget caveats.",
 "Gradient":"How a small change in a learned number would locally change the loss. The optimizer uses this signal to choose an update.",
 "Learning rate":"How large an adjustment the optimizer attempts. Larger can learn faster, overshoot, or become unstable.",
 "Optimizer":"The rule that turns gradients into weight updates. This lab offers AdamW and plain SGD.",
 "Batch / microbatch":"Windows of text processed together in one forward/backward calculation.",
 "Gradient accumulation":"Combine several microbatch gradients before changing the weights once.",
 "Update / training step":"One optimizer update. It can include several microbatches and many next-token targets.",
 "Epoch":"One pass over a dataset. This lab samples windows and reports updates/exposures rather than pretending they are exact dataset passes.",
 "Token exposure":"One valid target practised during training. Repeating the same target counts again; this is not unique data.",
 "Context window":"The maximum number of preceding input tokens available in one model window.",
 "Embedding":"A learned numerical vector looked up for a token or position.",
 "Attention":"A calculation that mixes information from allowed token positions. Its weights are not automatically an explanation of correctness.",
 "Causal mask":"A rule blocking future positions so the model cannot read the next answer during prediction.",
 "Checkpoint":"Saved model state. Resumable training also needs optimizer, sampling and random states—not just weights.",
 "Overfitting":"Getting better at the training material while performance on unseen practice-check text stops improving or worsens.",
 "Baseline":"A simpler comparison system. Here a bigram predicts using only the previous token.",
 "Run":"One recorded training experiment with its own configuration, results and files.",
 "Study":"A group of related runs, such as three learning-rate choices, plus your prediction and reflection.",
 "Seed":"A starting value for random generators. It helps repeat a setup, but does not guarantee identical results across hardware/software.",
 "Inference / generation":"Using learned weights to produce predictions. Sampling settings change outputs without retraining weights.",
 "Temperature":"A sampling control that sharpens or flattens next-token probabilities. Zero chooses the highest-scoring token directly.",
 "SFT / fine-tuning":"Continue training on demonstrations to teach behaviour. This lab implements local LoRA response training; that does not imply full LLM preference/RL training is implemented.",
 "RAG":"Retrieve evidence and include it in the model input. Retrieval does not by itself update the model's weights.",
}


def sections(markdown):
    result=[];heading="Overview";buffer=[];fenced=False
    for line in markdown.splitlines():
        if line.startswith("```"):fenced=not fenced
        if line.startswith("## ") and not fenced:
            text="\n".join(buffer).strip()
            # The first H1 is shown by the reader; do not create an empty title-only section.
            if text and not (heading=="Overview" and len([s for s in buffer if s.strip()])==1 and text.startswith("# ")):
                result.append({"title":heading,"markdown":text})
            heading=re.sub(r"^\d+\.\s*","",line[3:].strip());buffer=[line]
        else:buffer.append(line)
    text="\n".join(buffer).strip()
    if text:result.append({"title":heading,"markdown":text})
    return result


def _text(relative):
    path=MATERIALS/relative
    if path.is_symlink() or not path.resolve().is_relative_to(MATERIALS.resolve()):
        raise ValueError("Learning sources must remain inside the authored materials folder.")
    return path.read_text(encoding="utf-8")


def interview_items():
    items=[]
    for category,(label,relative) in INTERVIEW_FILES.items():
        text=_text(relative)
        matches=list(re.finditer(r"^## ((?:INLP|IML|IDL|F|M|S|D|P)\d{2}) — (.+)$",text,re.M))
        for i,m in enumerate(matches):
            if category in {"ml","dl"} and not m[1].startswith("IML" if category=="ml" else "IDL"):continue
            body=text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)].strip()
            # A final unrelated scoring section is not part of P16's answer.
            body=re.split(r"\n## ",body,maxsplit=1)[0].strip()
            items.append({"id":m[1],"title":m[2],"category":category,"category_label":label,
                          "source":relative,"markdown":body})
    return items


def catalog():
    lessons=[]
    for identifier,stage,relative,title,goal,nodes,lab in LESSONS:
        doc=document(identifier)
        reader=doc["reader"]
        # Match the student-facing lesson, not headings retained only in source Markdown.
        topics=list(dict.fromkeys([s["title"] for s in reader.get("reading",{}).get("steps",[])]+
                                  [n["title"] for n in reader["notes"]]))
        lessons.append({"id":identifier,"stage":stage,"source":relative,"title":title,"goal":goal,
                        "diagram":nodes,"lab":lab,"section_titles":[s["title"] for s in doc["sections"]],
                        "topics":topics})
    questions=[{k:v for k,v in q.items() if k!="markdown"} for q in interview_items()]
    resources=[{"id":key,"title":value[0],"source":value[1]} for key,value in RESOURCES.items()]
    return {"stages":STAGES,"lessons":lessons,"questions":questions,"resources":resources,
            "glossary":[{"term":key,"meaning":value} for key,value in GLOSSARY.items()],
            "limits":"Reading/bookmarks are browser-local and self-reported, not proof of independent mastery. Post-training theory does not imply an executable trainer."}


def document(identifier):
    lesson=next((row for row in LESSONS if row[0]==identifier),None)
    if lesson:
        _,stage,relative,title,goal,nodes,lab=lesson
        ids=[row[0] for row in LESSONS];index=ids.index(identifier)
        metadata={"stage":stage,"goal":goal,"diagram":nodes,"lab":lab,
                  "previous":ids[index-1] if index else None,"next":ids[index+1] if index+1<len(ids) else None}
    elif identifier in RESOURCES:
        title,relative=RESOURCES[identifier];metadata={"stage":"reference","goal":"Use this reference when a lesson or experiment raises a question.","diagram":[],"lab":None,"previous":None,"next":None}
    else:raise ValueError("Unknown learning document.")
    text=_text(relative)
    extra=next((row for row in ML_DL_LESSONS if row["id"]==identifier),None)
    if extra:
        metadata["parameters"]=[ML_DL_PARAMETERS[p] for p in extra["parameter_ids"]]
        metadata["prerequisites"]=[p for p in extra["prerequisites"] if p!="P1"]
    nlp_extra=next((row for row in NLP_LESSONS if row["id"]==identifier),None)
    if nlp_extra:metadata["prerequisites"]=nlp_extra["prerequisites"]
    result={"id":identifier,"title":title,**metadata,"source":relative,
            "sha256":hashlib.sha256(text.encode()).hexdigest(),"markdown":text,"sections":sections(text)}
    if lesson:
        from .lesson_reader import view
        result["reader"]=view(result,extra or nlp_extra or {},ML_DL_PARAMETERS,ML_DL_LESSONS,
                              reading=READER_PROFILES.get(identifier))
    return result


def interview(identifier):
    item=next((q for q in interview_items() if q["id"]==identifier),None)
    if item is None:raise ValueError("Unknown practice question.")
    return item
