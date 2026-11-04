# R3. Prompting, reasoning and output decoding

Prompting changes the input on which a model conditions; decoding changes how its token scores become outputs. Neither ordinarily updates learned weights. Intermediate reasoning, tool use and structured outputs add distinct procedures or constraints that must be evaluated for usefulness, source support and cost rather than assumed to make answers reliable.

Keep logits [2,1,0] fixed while changing positive temperature. The resulting distribution becomes sharper or flatter, but the model's parameters do not change. A separate JSON example shows why valid structure does not establish a correct value.

Training changes the predictor; inference can instead change its input or selection procedure. This lesson separates those controls before retrieval and tool systems add external evidence, actions and permission boundaries.

## 1. Prompt construction

**Key terms.** A **prompt** supplies context/instructions; **few-shot examples** demonstrate a task; **decoding** selects output from model scores.

**Process.** Define and serialize the task, then generate with fixed decoding settings. No weights change: compare prompt conditions without attributing the result to fine-tuning.

A prompt supplies task instructions, examples, context and output requirements. Zero-shot uses no task demonstrations in that prompt; few-shot includes some. A prompt can exploit existing capabilities but cannot guarantee knowledge or authority the system lacks.

Treat a prompt as a versioned experimental input: task definition, allowed evidence, examples, format and refusal rule. Do not change all five at once and attribute the gain to one clever phrase. Examples can clarify a label definition or leak answers. Their order and distribution can affect results.

## 2. Reasoning strategies

**Key terms.** **Chain-of-thought** is generated intermediate reasoning; **self-consistency** aggregates sampled answers; **test-time compute** is extra work during use.

**Process.** Choose direct prediction or a reasoning procedure, generate under a stated budget and verify the final answer. More reasoning text is a treatment to test, not proof of correct reasoning.

**Direct answer:** produce the target immediately. **Free-form reasoning:** generate intermediate text. **Structured intermediate representation:** extract defined fields or steps before answering. **Decomposition:** solve explicit subproblems. **Self-consistency:** sample several paths and aggregate final answers. **Search/tree strategies:** branch, evaluate and continue selected paths. **Verifier-guided selection:** score candidate outputs with a separate check.

These spend different amounts of test-time compute. A fair study reports model, input information, number of candidates, total tokens, tool calls, selection rule and latency. Comparing one direct sample against 64 sampled reasoning paths is not an equal-budget comparison.

Generated reasoning is not a guaranteed faithful account of internal computation. Test usefulness through matched conditions, perturbations and downstream outcomes. More steps can add contradictions or rationalize a prior guess. A short structured field can be more useful than an elaborate explanation if it isolates the right information.

## 3. Decoding is a distribution choice

**Key terms.** **Temperature** rescales logits; **top-k** keeps k candidates; **top-p** keeps a probability-mass subset; **sampling** draws a token.

**Process.** Transform model logits, filter and renormalize probabilities, then choose a token. Append it and repeat; these settings alter selection rather than learned weights.

**Calculation.** For two logits [2,0], temperature 1 gives first-token probability exp(2)/(exp(2)+1)≈0.8808. Temperature 2 divides the logits by 2, giving exp(1)/(exp(1)+1)≈0.7311. Both use the same model weights.

Given logits `z`, temperature sampling uses `p_i∝exp(z_i/T)` for `T>0`. Lower temperature sharpens relative probabilities; higher temperature flattens them. Temperature zero is usually a separate greedy limit/implementation, not division by zero.

- **Greedy:** choose the highest-probability token each step.
- **Top-k:** restrict to the k highest-scoring choices and renormalize.
- **Top-p / nucleus:** retain a smallest leading set whose cumulative probability reaches a threshold, then renormalize.
- **Beam search:** retain several high-scoring partial sequences. Length/coverage scoring changes behaviour; it is not automatically superior for open-ended chat.
- **Penalties:** modify scores for repetition/frequency or other preferences; provider definitions differ.

For probabilities `[0.6,0.25,0.15]`, top-p 0.8 retains the first two, totalling 0.85, then renormalizes to about `[0.7059,0.2941]`. Lower randomness does not make unsupported information true.

Stop conditions include EOS/turn markers, explicit stop sequences, tool-call boundaries and maximum output tokens. A maximum-token cutoff may leave a truncated JSON object; distinguish truncation from semantic error.

## 4. Structured output: three different guarantees

**Key terms.** A **schema** specifies structure/types; **constrained decoding** restricts allowable tokens; **validation** checks an output; **semantic correctness** concerns meaning.

**Process.** Specify the contract, enforce supported structural constraints and validate the result. Check evidence and meaning afterwards: valid JSON can still contain false content.

```text
schema validity → fields/types are allowed
referential validity → IDs/spans point to real input objects
semantic validity → claims are actually supported and correctly interpreted
```

Constrained decoding restricts allowed token continuations using a grammar or schema. It can enforce syntax under supported conditions. It does not establish factuality, complete coverage or correct labels. Validate after generation too: required fields, enums, lengths, source spans, conflicting values and application invariants.

Example teaching record:

```json
{
  "unit_id": "abstract1:sentence2",
  "claim": "Symptoms improved in the reported trial.",
  "population": "adults in the reported trial",
  "uncertainty": "small sample; evidence scope limited",
  "source_span": "Symptoms improved ...",
  "abstain": false,
  "abstain_reason": null,
  "warnings": ["Do not infer effects outside the reported population."]
}
```

An **abstain** flag means the system did not provide a particular decision/extraction under a defined rule; its reason explains why. A **warning** can accompany a produced answer when limitations remain. Define whether abstention applies to the entire record or an individual field. Missing information should be null/absent according to the schema, not invented to fill every slot.

### Test an intermediate representation rather than admire its format

Suppose a system extracts claims, evidence and uncertainty before predicting a label. Three separate comparisons are useful.

1. Compare direct prediction with free-form reasoning and structured extraction, keeping source text, guideline access and model revision fixed. Report additional generation tokens and latency.
2. Compare the structured representation with and without extra fields. More fields can add distractors, correlated mistakes and truncation, not just useful information.
3. Audit intermediate records against source spans. A correct final label does not prove the representation is faithful; the classifier may exploit a shortcut in the generated record.
4. Perturb an allegedly important field while preserving other information as far as possible. A changed output suggests sensitivity, not necessarily the desired reasoning mechanism; unrealistic perturbations can introduce new confounds.
5. Evaluate on datasets with different target definitions. If a roles-only rule transfers only to claim-oriented corpora, report that scope instead of turning a helpful diagnostic into a universal definition.

For constrained output, separate grammar acceptance, source-reference validity and semantic support. A decoder can forbid invalid JSON tokens, but it cannot infer that a nonexistent quotation is a valid source. Original reasoning approaches include [chain-of-thought prompting](https://arxiv.org/abs/2201.11903), [self-consistency](https://arxiv.org/abs/2203.11171) and [ReAct](https://arxiv.org/abs/2210.03629); these are different mechanisms and budgets, not interchangeable labels for “reasoning.”

## 5. Confidence, calibration and refusal

**Key terms.** **Confidence** is stated certainty; **calibration** compares it with observed correctness; **refusal** declines an answer.

**Process.** Check a confidence signal on held-out data and select any refusal policy before final testing. Evaluate answered-case error together with the proportion answered.

Self-reported confidence can be stylistic. Token probability concerns a token given a prefix, not necessarily the correctness of a whole proposition. Train/test selective prediction policies on representative data; report answered coverage and error together. Avoid teaching a system that refusal always earns the safest reward, or it may refuse useful benign tasks.

## 6. Failure diagnosis

**Key terms.** A **failure category** groups similar errors; a **matched contrast** isolates one changed input or procedure.

**Process.** Save failing prompts and raw outputs, locate the earliest wrong assumption and test a focused correction. Separate formatting, missing evidence and reasoning failures.

Bad format → inspect template, stop rules and grammar. Unsupported details → inspect evidence use and supervision, not just temperature. Lost facts → inspect truncation, summarization objective and coverage tests. Prompt injection → enforce system trust boundaries and permissions; a longer warning prompt is not sufficient containment. Inconsistent labels → inspect task definitions and examples.

Prompt caching and in-context learning do not necessarily alter weights. A conversation's apparent memory may be replayed messages or external storage. Distinguish those from continued training.

## Exercises and checks

1. Design direct, free-reasoning and structured conditions with equal source/guideline access and a documented token budget.
2. A schema-valid output cites a nonexistent source. Which guarantee failed? **Referential validity**, and likely semantic support.
3. What would falsify “reasoning improves robustness”? No held-out transfer gain, improved scores explained by more context/budget, or unstable intermediate steps irrelevant to the correct output.

Explore [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) for sampling and [ReAct](https://arxiv.org/abs/2210.03629) for interaction with tools. Do not interpret a successful toy prompt as universal evidence for chain-of-thought.
