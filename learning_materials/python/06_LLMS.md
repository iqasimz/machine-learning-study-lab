# LLM code and systems

Prerequisites: tokenization, tensors, forward passes and reading a training loop. An LLM application combines model behavior with data/permissions, retrieval, validation and deployment constraints. Changing a prompt is not training; retrieving evidence is not updating weights; a valid JSON object is not proof of truth.

## LC01. Model, weights and tokenizer

A language-model program has three separate objects. A **configuration** says how wide the network is and how many layers it has. A **checkpoint** is the stored weights. A **tokenizer** turns text into IDs. Building a model from a configuration creates a network with initial weights. Those weights are not a trained language model.

<!-- example: LC01-pieces -->
```python
configuration = {"layers": 1, "width": 8}
checkpoint = "local-weights"
vocabulary = {"hello": 2}
print(configuration["layers"], configuration["width"])
print(checkpoint, vocabulary["hello"])
```

**Output**
```text
1 8
local-weights 2
```

The width 8 is a shape decision. `"local-weights"` names a file of numbers. `2` is the ID for the word `"hello"`. Swapping any two of these changes a different part of the program. The next model is built from a configuration only, so its logits have the configured shape and its weights are initial values.

<!-- example: LC01-main -->
```python
import torch
from transformers import GPT2Config, AutoModelForCausalLM

configuration = GPT2Config(vocab_size=8, n_positions=8, n_embd=8,
                          n_layer=1, n_head=2, bos_token_id=1, eos_token_id=1)
model = AutoModelForCausalLM.from_config(configuration)
model.eval()
with torch.inference_mode():
    result = model(input_ids=torch.tensor([[2, 3, 4]]))
print(type(model).__name__)
print(tuple(result.logits.shape))
```

**Output**
```text
GPT2LMHeadModel
(1, 3, 8)
```

The output contains one example, three input positions and eight vocabulary logits per position. Random weights affect values, but not those configured dimensions. We deliberately do not print an invented sensible completion: this model has no demonstrated language competence.

A real inference example should load a reviewed local/pinned pretrained artifact, its matching tokenizer and documented dtype/device. `from_pretrained` is a loading interface, not a training call; it may nevertheless download assets unless explicitly restricted. Check license, revision and remote-code requirements before using it. A larger architecture alone does not supply learned knowledge.

Greedy decoding selects the largest next-token score; sampling uses a distribution modified by settings such as temperature/top-k/top-p. A completion is a captured sample under those choices, not a deterministic truth. Chat templates serialize message roles into model-specific control tokens. Use the supplied template rather than assuming that a Python list of messages is itself a valid token sequence.

The objects in that call have a fixed order. A message is a dictionary with a `role` and a `content`. `apply_chat_template` turns the list into text, then into a batch. The batch is a mapping whose keys include `input_ids` and `attention_mask`. `model.generate(**batch)` passes every key in that mapping as a named argument. Generation returns a tensor whose first axis is the batch and whose second axis is the prompt length plus the new tokens. The slice after the prompt length is what `decode` turns back into text. The tiny model below has its weights set so token `4`, the word `hello`, is the only next-token choice. That makes the trace readable. It is not a claim about SmolLM2.

<!-- example: LC01-trace -->
```python
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, GPT2Config, PreTrainedTokenizerFast

vocabulary = {"<pad>": 0, "<eos>": 1, "user": 2, "assistant": 3, "hello": 4}
backend = Tokenizer(WordLevel(vocabulary, unk_token="<pad>"))
backend.pre_tokenizer = Whitespace()
tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<pad>")
tokenizer.chat_template = "{% for message in messages %}{{ message['role'] }} {{ message['content'] }} {% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
messages = [{"role": "user", "content": "hello"}]
rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
batch = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
configuration = GPT2Config(vocab_size=5, n_positions=8, n_embd=4, n_layer=1, n_head=1, bos_token_id=1, eos_token_id=1, pad_token_id=0)
model = AutoModelForCausalLM.from_config(configuration)
with torch.no_grad():
    for parameter in model.parameters():
        parameter.zero_()
    model.transformer.ln_f.weight.fill_(1)
    model.transformer.ln_f.bias.fill_(1)
    model.lm_head.weight.zero_()
    model.lm_head.weight[4].fill_(1)
model.eval()
with torch.inference_mode():
    generated = model.generate(**batch, max_new_tokens=1, do_sample=False, pad_token_id=0)
prompt_length = batch["input_ids"].shape[1]
new_ids = generated[0, prompt_length:]
print([message["role"] for message in messages])
print(rendered.split())
print(sorted(batch))
print(batch["input_ids"].tolist())
print(tuple(generated.shape))
print(new_ids.tolist())
print(tokenizer.decode(new_ids))
```

**Output**
```text
['user']
['user', 'hello', 'assistant']
['attention_mask', 'input_ids', 'token_type_ids']
[[2, 4, 3]]
(1, 4)
[4]
hello
```

The only role in the input list is `user`. The template then adds the word `assistant` because generation is about to start. The rendered words become IDs `2`, `4`, and `3`. `**batch` is what hands `input_ids`, `attention_mask`, and `token_type_ids` to `generate`. The returned tensor has shape `(1, 4)`: one example, and the three prompt IDs plus one new ID. That new ID is `4`, which decodes to `hello`. The same sequence of objects appears in the recorded SmolLM2 call below. There the weights are the reviewed checkpoint, and the decoded text is the recorded answer rather than a weight we set by hand.

Sources: [Transformers quickstart](https://huggingface.co/docs/transformers/v4.57.1/quicktour), [chat templates](https://huggingface.co/docs/transformers/v4.57.1/chat_templating).

### A recorded answer from an actual pretrained model

Generation appends new IDs after the prompt IDs. The text you want to read is the slice that starts where the prompt ended.

<!-- example: LC01-slice -->
```python
prompt_ids = [2, 3]
generated_ids = [2, 3, 4, 1]
new_ids = generated_ids[len(prompt_ids):]
print(new_ids)
```

**Output**
```text
[4, 1]
```

The full row still begins with the prompt. `len(prompt_ids)` is 2, so the slice keeps `[4, 1]`. The following example does that slice on IDs produced by the local SmolLM2 checkpoint, then decodes those new IDs back to text.

The earlier width-eight model only illustrates an interface. This second example uses the **already prepared** SmolLM2-360M-Instruct asset from the local lab, revision `a10cc1512eabd3dde888204e902eca88bddb4951`, under Apache-2.0. Its language competence comes from that checkpoint, not from this course. The input dependency is explicit: `STUDY_LAB_MODEL_DIR` must name that reviewed local model folder. No files are downloaded and no weights updated.

`os` comes with Python. `os.environ` reads a variable the process was given when it started. The lesson uses it so the model folder is a setting, not a path written into the example. The call returns the text stored under that name. It does not download the model and it does not check that the folder contains the reviewed checkpoint.

<!-- example: LC01-pretrained -->
```python
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_num_threads(4)
model_dir = os.environ["STUDY_LAB_MODEL_DIR"]
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True,
    trust_remote_code=False, use_safetensors=True, dtype=torch.float32, attn_implementation="eager")
model.eval()
messages = [{"role": "user", "content": "What is 2 + 2? Answer with one number."}]
batch = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
    tokenize=True, return_tensors="pt", return_dict=True)
with torch.inference_mode():
    generated = model.generate(**batch, max_new_tokens=12, do_sample=False,
                               pad_token_id=tokenizer.eos_token_id)
new_ids = generated[0, batch["input_ids"].shape[1]:]
print(tokenizer.decode(new_ids, skip_special_tokens=True))
```

**Output**
```text
2 + 2 = 4
```

This is a recorded CPU inference result, not an across-platform output guarantee or model benchmark. Notice that the answer is arithmetically correct but does not obey “one number” exactly. That distinction is useful: content correctness and instruction adherence are separate evaluation questions. The slice removes prompt tokens so we display only newly generated text. Greedy decoding disables sampling; it does not guarantee identical low-level results across every library/device edition.

The verifier records hashes of the reviewed local model files, but does not copy its weights into this source pack or a public release. A static reader needs no model installed to inspect the recorded answer.

## LC02. Labels

A causal language model predicts a later token from earlier ones. **Labels** say which of those positions count in the loss. The value `-100` means “skip this position.” The model can still read a prompt token while the loss ignores it.

<!-- example: LC02-labels -->
```python
ids = [2, 3, 4, 1]
prompt_length = 2
labels = [-100] * prompt_length + ids[prompt_length:]
print(ids)
print(labels)
```

**Output**
```text
[2, 3, 4, 1]
[-100, -100, 4, 1]
```

The IDs are prompt `2`, prompt `3`, then response `4` and `1`. The labels repeat the two response IDs and mark the prompt positions as ignored. An attention mask would be a different list: it says which positions the model may read.

### Why Datasets

Datasets is the Hugging Face library for a table of examples. People use it because a list of records can be mapped, saved, and streamed without each project inventing a loader, and because the same object can feed a training script. `Dataset.from_list` builds the table from dictionaries already in memory. `.map` runs a function on each row. The library applies that function and keeps the columns. It does not check that the split is honest, that the text is licensed, or that `-100` was placed on the right positions. The next example stores the labels above on a dataset row.

<!-- example: LC02-main -->
```python
from datasets import Dataset

dataset = Dataset.from_list([{"ids": [2, 3, 4, 1], "prompt_length": 2}])
def attach_labels(row):
    return {"labels": [-100] * row["prompt_length"] + row["ids"][row["prompt_length"]:]}
prepared = dataset.map(attach_labels, load_from_cache_file=False)
item = prepared[0]
print(item["ids"])
print(item["labels"])
print(item["ids"][:-1], item["labels"][1:])
print(sum(target != -100 for target in item["labels"][1:]))
```

**Output**
```text
[2, 3, 4, 1]
[-100, -100, 4, 1]
[2, 3, 4] [-100, 4, 1]
2
```

Tokens 2 and 3 are prompt tokens; 4 and 1 are response/EOS targets. After the causal shift, the last prompt position predicts the first response token. The model can read prompt tokens even though those prompt targets do not contribute to the response-only loss. This example prepares records, not weights.

Many model classes shift labels internally; do not shift them a second time when using that interface. Padding must be excluded by position/mask, not by deleting every occurrence of an ID that may also be a legitimate EOS. Split source documents/conversations before making overlapping windows; otherwise related content can cross the boundary. Datasets mapping/streaming/caching help organize records but cannot guarantee an honest split or licensed use.

Sources: [Datasets processing](https://huggingface.co/docs/datasets/process), [causal language modeling](https://huggingface.co/docs/transformers/v4.57.1/tasks/language_modeling).

## LC03. The scratch training loop

This **read-only walkthrough** explains the existing scratch lab's causal learning task. The caller supplies a decoder, optimizer and input/next-token-target tensors already prepared from fitting documents. There is no call or training-output panel here.

```python walkthrough
def scratch_update(model, optimizer, input_ids, next_token_targets):
    import torch.nn.functional as F
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           next_token_targets.reshape(-1), ignore_index=-100)
    loss.backward()
    optimizer.step()
    return loss.detach()
```

Each input position's vocabulary scores are compared with its supplied next-token target. Flattening combines batch and sequence positions for this loss interface without changing which vocabulary dimension represents classes. Targets must already be shifted exactly once for this function. Ignored positions are excluded. The real scratch worker adds causal architecture constraints, token-count-weighted gradient accumulation, clipping/scheduling, validation, checkpoint state and telemetry; this minimal function is not that complete worker.

The tokenizer also has a fitting boundary: a tokenizer trained from this experiment's corpus must use fitting documents only. Validation selects settings/checkpoints; final test results should not repeatedly choose the model. A short scratch run can teach the loop while remaining incapable of useful general conversation.

**Run supported experiments:** **LLMs → Train a model → Build a model from scratch** (`train`). Inspect measured results in `explore` and compatible comparisons in `compare`. The scratch model and the pretrained adaptation route have very different starting knowledge; their raw losses/completions are not a fair universal leaderboard. Local installation is required.

Sources: [D2L language models](https://d2l.ai/chapter_recurrent-neural-networks/language-model.html), [PyTorch optimization](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html). Local references: `src/training_lab/training.py`, `model.py`, `data.py`.

## LC04. LoRA

LoRA keeps the original weight and adds a correction. The correction is built from two smaller matrices. The base weight can stay frozen while those smaller matrices learn.

<!-- example: LC04-add -->
```python
base = [1.0, 2.0]
added = [1.0, 2.0]
print([left + right for left, right in zip(base, added)])
```

**Output**
```text
[2.0, 4.0]
```

The base result is `[1, 2]`. The correction contributes another `[1, 2]`. Adding them position by position gives `[2, 4]`.

### Why PEFT

PEFT is the Hugging Face library for parameter-efficient fine-tuning. LoRA is the method this lab uses: two small matrices stand in for a change to a large weight, and the original weight can stay frozen. People use the library because the rank, the scaling, and the layer names are one configuration object instead of a hand-wired product in every training script. `LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])` stores those choices. PEFT will attach the adapter where those names match a model. It does not decide that a small adapter is enough for the task, and a lower parameter count is not a quality score.

The next example builds the addition above from the two small matrices. If a base linear weight has shape `(output, input)`, a rank-r correction can use `A` with shape `(r, input)` and `B` with shape `(output, r)`.

<!-- example: LC04-main -->
```python
import torch
from peft import LoraConfig

inputs = torch.tensor([[1., 2.]])
base_weight = torch.eye(2)
adapter_a = torch.tensor([[1., 0.]])
adapter_b = torch.tensor([[1.], [2.]])
base_output = inputs @ base_weight.T
adapted_output = base_output + (inputs @ adapter_a.T) @ adapter_b.T
print(base_output.tolist())
print(adapted_output.tolist())
configuration = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])
print(configuration.r, configuration.lora_alpha)
print("adapter parameters:", 4 * 8 + 12 * 4)
```

**Output**
```text
[[1.0, 2.0]]
[[2.0, 4.0]]
4 8
adapter parameters: 80
```

The fixed rank-one arithmetic has scale one and is separate from the constructed rank-four config. Its added result is `[1,2]`; the adapted result is `[2,4]`. In the parameter-count calculation, an 8-input/12-output rank-four adapter has `4×8+12×4=80` parameters, versus 96 base weights. Savings depend on dimensions/rank and do not guarantee better task quality. The alpha/r scaling convention must match the adapter implementation.

The existing lab uses PEFT on prepared SmolLM2 attention query/value projections. Its response-only objective can be read as follows; this function is not executed here:

```python walkthrough
def response_only_adapter_update(adapter_model, optimizer, input_ids, masked_labels):
    adapter_model.train()
    optimizer.zero_grad(set_to_none=True)
    result = adapter_model(input_ids=input_ids, labels=masked_labels, use_cache=False)
    result.loss.backward()
    optimizer.step()
    return result.loss.detach()
```

The configured model class handles shifting internally, unlike LC03's explicitly prepared targets. The real worker also handles conversation splits, prompt masking, evaluation and saved artifacts. It does not expose every full-fine-tuning/PEFT/export feature. **Run it:** **LLMs → Train a model → Adapt a language model**, then use the existing results/comparison pages. No duplicate before/after training score is shown in this lesson.

Sources: [PEFT quicktour](https://huggingface.co/docs/peft/quicktour), [LoRA paper](https://arxiv.org/abs/2106.09685). Local reference: `src/training_lab/adaptation.py`.

## LC05. Preference signals

A preference method needs an explicit signal about two answers, or about a reward. One correct label is the signal used by ordinary supervised learning. Here the signal is the gap between a preferred score and a rejected score.

<!-- example: LC05-pair -->
```python
preferred_score = 1.0
rejected_score = 0.0
print(preferred_score - rejected_score)
```

**Output**
```text
1.0
```

The preferred answer scores 1 and the rejected answer scores 0, so the gap is 1. DPO compares preferred and rejected examples with a reference policy. PPO uses sampled responses, advantages, and a limit on how far the update may move. GRPO compares rewards inside a group. Each one consumes a different signal.

The next block uses two substitutions. For DPO, the policy log-odds are 1 and the reference log-odds are 0, so the difference is 1. `beta` is 0.5, so the margin is `0.5 × 1 = 0.5`. The loss then uses that margin. For PPO, the ratio is 1.4 and epsilon is 0.2, so the allowed ratios run from 0.8 to 1.2. The ratio 1.4 is above 1.2, so the clipped ratio is 1.2. With an advantage of 2, the clipped product is `1.2 × 2 = 2.4`.

<!-- example: LC05-steps -->
```python
beta = 0.5
policy_log_odds = 1.0
reference_log_odds = 0.0
difference = policy_log_odds - reference_log_odds
margin = beta * difference
ratio, advantage, epsilon = 1.4, 2.0, 0.2
lower, upper = 1 - epsilon, 1 + epsilon
clipped_ratio = min(max(ratio, lower), upper)
print(difference)
print(margin)
print(lower, upper, clipped_ratio)
print(clipped_ratio * advantage)
```

**Output**
```text
1.0
0.5
0.8 1.2 1.2
2.4
```

The combined example below starts from these same numbers. Its DPO loss is `log(1 + exp(-0.5))`, which is the margin from this substitution. Its PPO line is the product 2.4.

<!-- example: LC05-main -->
```python
import math
import numpy as np

beta = 0.5
policy_log_odds = 1.0
reference_log_odds = 0.0
margin = beta * (policy_log_odds - reference_log_odds)
dpo_loss = np.logaddexp(0., -margin)
ratio, advantage, epsilon = 1.4, 2., 0.2
ppo_surrogate = min(ratio * advantage, min(max(ratio, 1-epsilon), 1+epsilon) * advantage)
group_rewards = np.array([0., 2., 4.])
advantages = (group_rewards - group_rewards.mean()) / group_rewards.std()
print(f"DPO loss={dpo_loss:.6f}")
print(f"PPO clipped surrogate={ppo_surrogate:.1f}")
print(np.round(advantages, 3).tolist())
```

**Output**
```text
DPO loss=0.474077
PPO clipped surrogate=2.4
[-1.225, 0.0, 1.225]
```

The DPO example uses supplied sequence log-odds, not raw probabilities or a training run. The PPO ratio is clipped at 1.2 in this positive-advantage example, limiting the surrogate to `1.2×2=2.4`. The displayed surrogate is maximized; an implementation minimizing a loss uses an appropriate sign. Negative advantages must use the actual min/clipping expression, not an improvised “always cap the reward” rule.

The group normalization illustrates one reward-relative signal using population standard deviation. It is not the complete GRPO loss; real implementations also define token/sequence aggregation, clipping, masks, reference/KL choices and zero-variance handling. Optimizing a reward can teach shortcuts in that reward. No preference/RL trainer is currently available in the lab, and no fitting loop or improvement table is added here.

Sources: [DPO paper](https://arxiv.org/abs/2305.18290), [PPO paper](https://arxiv.org/abs/1707.06347), [DeepSeekMath/GRPO](https://arxiv.org/abs/2402.03300), [TRL](https://huggingface.co/docs/trl/index).

## LC06. Retrieval

Retrieval selects source rows before a model writes an answer. A row the user is not allowed to see has to be removed before ranking.

<!-- example: LC06-allow -->
```python
documents = [("d1", True), ("d2", False)]
print([name for name, allowed in documents if allowed])
```

**Output**
```text
['d1']
```

`d2` is dropped because its flag is false. The remaining IDs are the only ones a later step may rank or quote. RAG places the selected text into the generation request. A high score says the text matched the query. It does not certify that the text is true, and it does not by itself permit an action.

<!-- example: LC06-main -->
```python
documents = [
    {"id": "d1", "text": "The library opens at nine.", "allowed": True},
    {"id": "d2", "text": "Private staffing records.", "allowed": False},
    {"id": "d3", "text": "The cafe opens at eight.", "allowed": True},
]
query_terms = {"library", "opens"}
permitted = [document for document in documents if document["allowed"]]
ranked = sorted(permitted, key=lambda document: len(query_terms & set(document["text"].lower().split())), reverse=True)
evidence = ranked[0]
print(evidence["id"], evidence["text"])
print("Evidence IDs:", [document["id"] for document in permitted])
```

**Output**
```text
d1 The library opens at nine.
Evidence IDs: ['d1', 'd3']
```

This is a simple lexical-overlap retrieval rule with an explicit allowlist, not an embedding model or a generated answer. Permission filtering happens before evidence is exposed. Keep source IDs and offsets so a later answer can cite the actual supporting passage. A high similarity score is not a probability that a source is correct.

A real pipeline may chunk documents, combine lexical/dense retrieval, rerank candidates and allocate a prompt token budget. Evaluate retrieval separately from answer faithfulness; an answer can be wrong despite retrieving the right document. Conflicting or missing evidence should lead to an appropriate qualified answer or refusal, not a guessed citation. Retrieved text is untrusted content and must not override the application's tool permissions. No RAG training/production service is implied by this fixed example.

Sources: [RAG paper](https://arxiv.org/abs/2005.11401), [Sentence Transformers retrieval](https://sbert.net/docs/quickstart.html), [FAISS](https://github.com/facebookresearch/faiss/wiki/Getting-started).

## LC07. Validating actions

A model can print something that looks like a request. The application has to parse that text into fields before any action is possible.

<!-- example: LC07-json -->
```python
import json

text = '{"query": "argument mining", "limit": 3}'
request = json.loads(text)
print(request["query"], request["limit"])
```

**Output**
```text
argument mining 3
```

`json.loads` turns the text into a dictionary. The query is `"argument mining"` and the limit is the integer 3. Parsing succeeds for any JSON with those keys, including a limit of 30.

### Why Pydantic

Pydantic is the library this course uses to check that a parsed object has the fields and the ranges a program expects. People use it because a class declares the types once, and `model_validate_json` either returns that object or raises a validation error. `Field(ge=1, le=5)` is the limit written next to the field. Pydantic checks the shape you declared. It does not know whether the query is safe to run, and a valid object is still not evidence that the query is true. Permission to run a tool is a further check.

<!-- example: LC07-main -->
```python
from pydantic import BaseModel, ConfigDict, Field, ValidationError

class SearchRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    query: str
    limit: int = Field(ge=1, le=5)

request = SearchRequest.model_validate_json('{"query":"argument mining","limit":3}')
print(request.query, request.limit)
try:
    SearchRequest.model_validate_json('{"query":"x","limit":30}')
except ValidationError:
    print("invalid request rejected")
```

**Output**
```text
argument mining 3
invalid request rejected
```

The validated request is not executed. Strict typing and forbidden extra fields make some unintended inputs explicit, but the query could still contain malicious instructions or sensitive information. Tool execution additionally needs an allowlisted operation, controlled scope, bounded resources and human approval for consequential actions.

Retries should be bounded and respond to a known failure rather than endlessly asking the model again. Store explicit workflow state so an interrupted request can resume without repeating an external action. MCP standardizes communication/discovery between clients and tool/resource servers; it does not automatically make a discovered tool safe or grant it authority. A future concrete SDK recipe needs separate version/permission verification; this core example teaches the validation boundary without creating a server.

Sources: [Pydantic models](https://docs.pydantic.dev/latest/concepts/models/), [MCP specification](https://modelcontextprotocol.io/specification).

### Keep a tool request separate from permission to execute it

<!-- example: LC07-permission -->
```python
proposed_request = {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                    "params": {"name": "send_email", "arguments": {"subject": "Update"}}}
allowed_tools = {"search_documents"}
tool_name = proposed_request["params"]["name"]
print(proposed_request["method"], tool_name)
print("authorized:", tool_name in allowed_tools)
```

**Output**
```text
tools/call send_email
authorized: False
```

This is a constructed JSON-RPC-shaped request for illustrating a boundary, not an initialized MCP client or validated full protocol exchange. Parsing the request does not send an email. A real host negotiates the applicable protocol, validates the tool schema and independently enforces permissions/approval. A model's claimed urgency is not authority to bypass that gate.

## LC08. Service boundaries

An HTTP response has a status and a body. A test can build that pair in memory and read it back. The status says how the call finished. The body is the payload.

<!-- example: LC08-status -->
```python
response = {"status": 200, "body": {"ready": True}}
print(response["status"], response["body"]["ready"])
```

**Output**
```text
200 True
```

Status 200 and body `{"ready": True}` are values the test chose. They do not prove that a model process is healthy.

### Why HTTPX and FastAPI

FastAPI is a Python library for describing an HTTP service as functions. HTTPX is the library that sends HTTP requests, including a test double that never leaves the process. People use the pair because a route and its test can live in the same language as the model code, and because `TestClient` and `MockTransport` exercise status codes and JSON without a public server or a paid API. `@application.get("/health")` names the path. `client.get("/health").json()` reads the body. The libraries return the response your function built. They do not make a model ready, and they do not make a blocking calculation concurrent.

The next example asks a tiny application for that body through the HTTP test client, still without opening a public port. A request also has a method, a path, data, and a deadline. `asyncio`, which comes with Python, lets the program wait for other work during a network call. A long model calculation still occupies the worker for as long as the calculation takes.

<!-- example: LC08-main -->
```python
import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

application = FastAPI()
@application.get("/health")
def health():
    return {"ready": True}

with TestClient(application) as client:
    print(client.get("/health").json())

def mock_provider(request):
    return httpx.Response(200, json={"tokens": 4})

with httpx.Client(transport=httpx.MockTransport(mock_provider), timeout=5) as client:
    print(client.get("https://example.invalid/status").json())
```

**Output**
```text
{'ready': True}
{'tokens': 4}
```

Both calls remain in-process: no server listens publicly and no request reaches that hostname. The health endpoint only reports the fixture's chosen flag; it is not proof that a GPU model is ready or an application secure. Mocking lets a maintainer test error handling without spending API credits.

Use explicit connect/read/write deadlines, bounded concurrency, rate-limit handling and cancellation. Retry only operations whose repetition is safe, especially after ambiguous network failures. Streaming responses need cleanup when a client disconnects. Keep secrets out of source/logs and do not return full prompts in public error messages. HTTPX's AsyncClient can overlap network waiting; a GPU worker still needs appropriate scheduling/backpressure. No new deployed assistant is built by this lesson.

Sources: [HTTPX transports](https://www.python-httpx.org/advanced/transports/), [HTTPX async](https://www.python-httpx.org/async/), [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/).

### Bound concurrent waiting explicitly

<!-- example: LC08-async -->
```python
import asyncio
import httpx

async def main():
    limit = asyncio.Semaphore(2)
    async def mock(request):
        return httpx.Response(200, json={"item": request.url.path.rsplit("/", 1)[-1]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as client:
        async def fetch(number):
            async with limit:
                response = await client.get(f"https://example.invalid/items/{number}")
                response.raise_for_status()
                return response.json()["item"]
        print(await asyncio.gather(*(fetch(number) for number in range(3))))

asyncio.run(main())
```

**Output**
```text
['0', '1', '2']
```

At most two fetch bodies enter the semaphore-protected region together. `gather` returns results in input order, not necessarily completion order. These are mocked requests with no real network or measured speedup. Production code also needs deadlines, bounded total work, exception/cancellation handling and safe retry semantics; concurrency without a resource budget can overload the provider or your process.

## LC09. Memory and latency

Weight memory is the number of parameters times the bytes stored for each one. A 16-bit weight uses 2 bytes.

<!-- example: LC09-bytes -->
```python
parameters = 100
bytes_per_weight = 2
print(parameters * bytes_per_weight)
```

**Output**
```text
200
```

One hundred 2-byte weights occupy 200 bytes. The same multiplication, with a hundred million parameters, is the weight estimate in the next example. Activations and the KV cache occupy more memory on top of the weights. During generation, a **KV cache** stores earlier keys and values so a later token does not recompute them. Prefill reads the prompt. Decode produces the following tokens. Those two stages spend time differently.

<!-- example: LC09-main -->
```python
parameters = 100_000_000
bytes_per_weight = 2
weight_megabytes = parameters * bytes_per_weight / 1_000_000
layers, batch, context, kv_heads, head_width, bytes_per_value = 2, 1, 128, 2, 4, 2
cache_bytes = 2 * layers * batch * context * kv_heads * head_width * bytes_per_value
print(f"weights only={weight_megabytes:.0f} MB")
print(f"KV cache example={cache_bytes / 1024:.0f} KiB")
```

**Output**
```text
weights only=200 MB
KV cache example=8 KiB
```

The leading cache factor two counts keys and values. This simple formula assumes a particular uncompressed cache layout and uses KV-head count, not necessarily query-head count in grouped-query attention. The weight estimate excludes activations, adapter/optimizer states, allocator overhead and metadata. MB and MiB are different units.

Quantization reduces numerical storage but may introduce scales/metadata and backend constraints. A smaller file does not guarantee faster inference on every device. Distillation trains another model and is not an inference-only setting. Batching can improve throughput while increasing waiting time; measure time-to-first-token and inter-token latency as well as tokens/second. MLX is Apple-focused; vLLM/llama.cpp/ONNX paths have their own hardware/model support. Do not present an untested backend as an M4 speedup or add a new training recipe here.

Sources: [Transformers cache](https://huggingface.co/docs/transformers/v4.57.1/kv_cache), [MLX](https://ml-explore.github.io/mlx/build/html/index.html), [PyTorch memory notes](https://docs.pytorch.org/docs/2.9/notes/cuda.html#cuda-memory-management).

## LC10. System evaluation

An evaluation records one question and the numbers that answer it. Token loss is the total penalty divided by the number of scored tokens.

<!-- example: LC10-mean -->
```python
nll_sum = 4 + 3
valid_tokens = 2 + 3
print(nll_sum / valid_tokens)
```

**Output**
```text
1.4
```

The two batches contribute penalties 4 and 3, over 2 and 3 tokens. `7 / 5` is 1.4. That number says how surprised the model was by those target tokens. It does not say the answer was honest, useful, or allowed to call a tool. A release check needs those questions written down separately, with a way to notice a failure and return to a previous version.

<!-- example: LC10-main -->
```python
import math

fixed_batches = [{"nll_sum": 4., "valid_tokens": 2},
                 {"nll_sum": 3., "valid_tokens": 3}]
mean_nll = sum(batch["nll_sum"] for batch in fixed_batches) / sum(batch["valid_tokens"] for batch in fixed_batches)
fixed_cases = [{"supported": True, "tool_allowed": True},
               {"supported": False, "tool_allowed": True}]
print(f"mean NLL={mean_nll:.1f}, perplexity={math.exp(mean_nll):.3f}")
print("support checks passed:", sum(case["supported"] for case in fixed_cases), "/", len(fixed_cases))
print("tool permission does not imply source support")
```

**Output**
```text
mean NLL=1.4, perplexity=4.055
support checks passed: 1 / 2
tool permission does not imply source support
```

The loss uses all five valid tokens: `(4+3)/(2+3)=1.4`. Averaging batch means equally would give a different result. Perplexity comparisons require compatible tokenization, context and scoring boundaries. These constructed cases are neither a new training result nor a benchmark claim.

Keep regression fixtures for empty input, missing evidence, conflicting sources, invalid schema, denied tools, timeout and exhausted budget. Logs/traces should capture useful IDs/timings without leaking sensitive content. A model/data card documents provenance, intended use and known limits; versioned configuration and rollback make failure recovery possible. Activation hooks/steering can inspect or intervene on representations, but a changed output does not establish that an explanation is faithful. Preserve a baseline and a defined intervention/evaluation protocol before making such claims.

For actual trained-run metrics, continue to the lab's existing saved results/comparison pages. This course teaches how to interpret/compute specific quantities; it does not reproduce those pages as another leaderboard.

Sources: [PyTorch loss reductions](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html), [Hugging Face model cards](https://huggingface.co/docs/hub/model-cards), [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework).

### An activation hook is a program intervention

<!-- example: LC10-hook -->
```python
import torch
from torch import nn

layer = nn.Linear(2, 2, bias=False)
with torch.no_grad():
    layer.weight.copy_(torch.eye(2))
inputs = torch.tensor([[1., 2.]])
with torch.inference_mode():
    original = layer(inputs)
handle = layer.register_forward_hook(lambda module, arguments, output: output + torch.tensor([0., 1.]))
with torch.inference_mode():
    changed = layer(inputs)
handle.remove()
print(original.tolist(), changed.tolist())
```

**Output**
```text
[[1.0, 2.0]] [[1.0, 3.0]]
```

The hook adds a supplied vector to this layer's output and is then removed. It demonstrates an intervention at a precise program location, not discovery of a “truth direction” or a human-interpretable concept. For real activation steering, identify layer/position/direction, preserve a baseline, measure intended behavior and side effects, and avoid deriving directions from held-out labels. Source: [PyTorch module hooks](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook).
