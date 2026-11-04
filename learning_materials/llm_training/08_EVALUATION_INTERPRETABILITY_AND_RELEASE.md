# T8. LLM evaluation and interpretability evidence

LLM evaluation tests a specified claim, such as better token prediction, task accuracy, source support, robustness or efficiency. Interpretability asks different questions about computation and mechanism. A score gain, an attention picture and a controlled intervention provide different evidence; none alone establishes every claim about useful or reliable behavior.

Two hypothetical task scores, 0.76 and 0.79, illustrate a difference of 0.03. A separate scalar intervention starts from 0.8 and subtracts a selected amount. These are interpretation exercises, not measured causal effects from a trained model.

After fitting and resource accounting, evaluate the artifact against its intended use. Shared metrics supply the formulas; this lesson connects matched protocols, uncertainty and mechanism claims without treating one as a substitute for another.

## 1. Evaluation protocols

**Key terms.** A **claim** states the proposed improvement; an **evaluation protocol** fixes how it is checked; a **held-out set** is excluded from fitting.

**Process.** Translate the claim into a measurable outcome before choosing a benchmark. Freeze the protocol, collect predictions and inspect failures before generalizing beyond that outcome.

Specify a claim before choosing a benchmark: next-token prediction, task success, source faithfulness, robustness, calibration, safety or efficiency. No single leaderboard measures all of them. Save model revision, tokenizer/template, prompts, examples, decoding, budgets and scoring code.

Likelihood-based multiple choice, free generation and constrained label generation are different protocols. A model can rank a correct choice highest but fail to emit it under a format constraint. Report the interface being tested.

## 2. Build an evaluation ladder

**Key terms.** A **sanity check** tests basic correctness; a **benchmark** measures a defined task; **robustness** tests changed conditions.

**Process.** Begin with small known cases, progress to matched held-out tasks and then test shifts or adversarial cases. Each level builds evidence for a stronger claim rather than replacing the earlier checks.

```text
unit checks → held-out language-model loss → task datasets
→ domain/length/language slices → adversarial and shift tests
→ human-reviewed cases → intended system evaluation
```

Benchmark contamination, reused development examples and repeated selection all weaken interpretation. Freeze a final set. Record unresolved samples rather than dropping malformed generations silently. Human annotation needs instructions, disagreement handling and safeguards for sensitive material.

## 3. Matched comparisons

**Key terms.** A **matched comparison** controls data and budgets; **paired evaluation** compares systems on the same cases; **seed variation** comes from training randomness.

**Process.** Freeze conditions, run comparable systems and save per-example predictions. Compute effects and uncertainty under the declared unit before deciding whether the change helped reliably.

Match available information and state resource budgets. Report uncertainty using appropriate independent units and training seeds where feasible. A confidence interval for one dataset does not establish universal superiority. If comparing many settings, distinguish exploratory screening from confirmatory comparisons.

For generative tasks, assess correctness, completeness, unsupported additions, contradictions and appropriate refusal. A factual statement can be unsupported by the supplied source; a source-faithful summary can repeat a false source. Those are different axes.

LLM judges can exhibit position, verbosity, self-preference and instruction-following biases. Blind system identities where feasible, randomize answer order and compare a sample with independent human judgements. Judge agreement with one annotator does not establish objective truth.

### Build an evaluation table that supports one clear conclusion

For a claimed robustness improvement, use one row per condition and columns for input access, training data, selection rule, token/compute budget and outcomes. A score without those columns can hide an unmatched comparison.

1. Freeze prompts, dataset groups and scoring rules before the confirmatory test. Mark exploratory results separately.
2. Save per-example labels, predictions and failure states. A malformed generation or abstention belongs in a declared denominator, not a silently filtered “successful outputs” subset.
3. Calculate paired differences on the same cases. Use document/group resampling when cases share a source, and report training-seed variation separately where feasible.
4. Evaluate an explicit transfer condition and important slices: domain, label policy, language, length, negation and rare entities. A global mean can improve while a critical slice fails.
5. For a proposed explanation, add a control that could refute it. For example, compare useful intermediate fields with shuffled or task-irrelevant fields under a matched budget; do not infer causality from an attention picture.

For generated answers, score correctness, support and coverage separately. A citation can identify a real source without supporting the claim attached to it. Use an independently reviewed subset to audit automatic judges. References: [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) and [paired bootstrap implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).

## 4. Inspection, attribution and intervention

**Key terms.** **Inspection** observes computation; **attribution** assigns relevance; an **intervention** changes computation to test an effect.

**Process.** Inspect model states to form a hypothesis, then use matched interventions where feasible. Report the difference between a useful visualization and evidence that a component caused behaviour.

**Behavioural analysis** studies inputs/outputs: change a guideline, remove context, vary a counterfactual. **Representational analysis** uses probes or similarities inside the network. **Mechanistic intervention** changes activations/components to test their role in computation.

A probe predicting a property shows it can be decoded under that probe setup, not that the model uses it for the task. A saliency map is local sensitivity, not a complete causal account. Activation patching replaces selected internal values from one run with values from another and measures an outcome change; matched controls and the choice of corruption matter.

A sparse autoencoder attempts to decompose activations using a sparsity-constrained dictionary. Features can be useful hypotheses, but names assigned by humans/models may be incomplete. Superposition means more concepts may be represented than there are clean one-concept-per-neuron axes.

Steering adds or modifies activation directions during inference. Test intended effect, unrelated capabilities, different prompts/languages and distribution shift. “A behaviour changed” does not prove the direction exclusively encodes that behaviour. R9 returns to the method/claim boundary.

## 5. Release as a reproducibility package

**Key terms.** A **model card** documents intended use and limits; a **manifest** lists artifacts; a **revision** fixes a version.

**Process.** Freeze the checkpoint, code, tokenizer and evaluation settings, then package the evidence and limitations. A separate reader should be able to reconstruct the reported comparison without private assumptions.

Include model/config/tokenizer versions, training objective, data provenance and known restrictions, evaluation protocol/results, intended uses, limitations and safety considerations. Release only data/weights you have rights to share. Remove secrets and private examples; prefer safe formats and inspect code before enabling remote execution.

**Open weights** means weights are available under stated terms. It does not necessarily mean open training data, reproducible training, unrestricted use or an open-source-compliant license. Model cards and dataset cards document a system; they do not certify its claims automatically.

## 6. Final evaluation exercise

**Key terms.** An **acceptance criterion** is a stated success condition; **reproducibility** means reconstructing results under declared conditions.

**Process.** Bring the trained artifact and its evidence together, test the declared criteria and document unresolved failures. Release or revise based on those checks, not merely a finished training progress bar.

You have a checkpoint with improved perplexity but worse scientific-summary fidelity. State five hypotheses: corpus shift, alignment loss, overfitting, decoding/template change, or a weak fidelity metric. Design a matched test for each rather than declaring that perplexity is useless or that the model has become “less intelligent.”

**Pass condition:** your report separates observations from explanations, gives exact inputs/budgets, includes failure examples and does not use an attention heatmap as proof of truthfulness.

References: [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), [CS336 evaluation material](https://cs336.stanford.edu/), [NIST AI risk framework](https://www.nist.gov/itl/ai-risk-management-framework). Interpretability is a specialist field: the mechanism overview here is not an exhaustive implementation course.
