# R9. LLM research methods and activation steering

Research methods can change data, parameters, activations, decoding or system operations, and those interventions support different claims. Activation steering changes selected runtime representations while keeping model weights fixed. A local output change demonstrates sensitivity under that intervention, not that a named concept has been isolated or reliable behavior achieved everywhere.

The supplied hidden row vector [0.2,−0.1] is scored by a fixed projection [1,−1]. Adding alpha times direction [1,0] changes its score from 0.3 to 0.3+alpha. These authored values do not identify a real truth, safety or reasoning direction.

Evaluation and operations require evidence beyond plausible mechanism names. This final lesson uses one exact steering example and keeps editing, unlearning, merging, continual learning and privacy as distinct deeper branches.

## 1. Evaluating an unfamiliar method

**Key terms.** An **assumption** is a required condition; an **intervention** is the intended change; **evidence** tests its benefit.

**Process.** Identify what a method changes, what stays fixed and what result would disprove its claim. Read its objective and experiment before adding it to a pipeline.

For every paper or library, ask:

1. What object changes: data, tokenizer, weights, activations, retrieval, decoding or orchestration?
2. What training/evaluation signal is used, and who defines “good”?
3. What assumptions make the method plausible?
4. What is its memory/compute/data/annotation cost?
5. Against which matched baseline is its claim tested?
6. Which failure, domain or budget has not been tested?

This is more durable than memorising a new acronym every week.

## 2. Continued learning and forgetting

**Key terms.** **Continual learning** adapts over time; **catastrophic forgetting** loses earlier capabilities; **replay** reuses old examples.

**Process.** Adapt to new data, then test both new and previous tasks. Balance improvement with retention rather than declaring success on only the newest task.

Domain-adaptive pretraining learns from a narrower corpus; task-adaptive pretraining focuses on task-related unlabelled input. Continual learning repeatedly adapts over time. **Catastrophic forgetting** is loss of earlier capabilities after adaptation. Replay mixes older examples; regularization discourages harmful changes; adapters isolate some updates. Each trades storage, privacy, cost and flexibility.

Evaluate old and new tasks, not only the newest domain. A frozen base plus a selected adapter changes which behaviour is used without retraining the base each time; it does not solve all interference or routing problems.

## 3. Model editing and unlearning

**Key terms.** **Model editing** targets particular behaviour; **unlearning** aims to remove an influence; **locality** checks unrelated behaviour.

**Process.** Specify the change, intervene and test the target. Then test paraphrases and unrelated cases: one corrected answer does not establish broad removal.

Model editing tries to change specific associations or behaviours with limited collateral effects. Evaluate edit success, paraphrase generalization, locality to unrelated facts, persistence and composition with other edits. Updating one question's answer is not the same as updating all dependent knowledge.

Unlearning seeks removal of a training contribution under a specified definition. Refusing to reveal a fact or removing it from retrieval is not proof that trained weights no longer contain it. Exact retraining is a conceptual reference but can be costly; approximate methods need careful guarantees/empirical tests. Privacy/legal obligations require specialist review.

## 4. Merging and ensembles

**Key terms.** **Merging** combines compatible parameters; an **ensemble** combines predictions; **compatibility** includes architecture and parameter meaning.

**Process.** Check origins and architectures before merging, or combine outputs under a declared rule. Evaluate interference and cost against each component and a baseline.

An ensemble combines outputs from multiple models; a weight merge combines compatible parameter tensors into one model. A simple interpolation is `θ=αθ₁+(1−α)θ₂`, but independently trained networks can use different parameter symmetries. Related fine-tunes from one base are a more plausible setting, still requiring evaluation.

Task arithmetic uses parameter differences from a shared base. More advanced merges handle conflicts or sparsify deltas. Adapter composition has its own compatibility issues. Compare against selecting the right single model and against an equal-cost output ensemble. Do not assume combining two strengths preserves both.

## 5. Mechanistic interpretability and steering

**Key terms.** A **circuit** is a proposed computational substructure; **activation patching** substitutes intermediate values; **steering** changes activations; a **control direction** tests nonspecific disruption.

**Process.** Form a behavioural hypothesis, choose a layer/intervention and measure intended and unrelated behaviours. Controls distinguish a useful directional effect from damaging computation.

Activation patching intervenes on internal values to test their role. Circuit analysis tries to identify interacting components supporting behaviour. Sparse autoencoders learn a dictionary representation of activations under a sparsity objective:

\[
\min_{encoder,decoder}E[\|a-\hat a\|_2^2+\lambda\|z\|_1],
\]

where `a` is an activation, `z` the learned code and `â` its reconstruction. Exact SAE objectives vary. Reconstruction loss versus sparsity is a trade-off; a feature label is a hypothesis, not a verified natural-language ground truth.

Steering might apply `a' = a + αv` at a chosen layer/token. `v` could come from contrastive examples or a learned feature. Test dose/strength, specificity, unrelated tasks, prompt diversity, languages and adversarial settings. Representation axes can be entangled. “We changed refusal rate” does not establish isolated control of a safety circuit.

### Evaluate a steering direction as an intervention, not a label

Suppose contrastive examples produce an activation direction `v`, and inference applies `a′=a+αv` at a chosen layer and token position.

1. Define the desired change and independent test prompts before choosing the direction's strength. A lower refusal rate is not necessarily better if harmful requests also become easier.
2. Record how contrast pairs differ. If examples also vary in length, style or topic, their mean activation difference can include those factors.
3. Compare several strengths, the opposite direction, a norm-matched random direction and appropriate layer/token controls. Use development data for choosing these settings.
4. Measure the desired effect alongside unrelated capabilities, benign refusals, language/style slices and adversarial prompts. A large change on one set can come from broad disruption rather than specific control.
5. Inspect failures under shift. An intervention that works for one formatting pattern is not production-grade controllability.

Activation patching asks a related but different question: replacing selected internal values between clean/corrupted computations tests their contribution under that intervention. Probe accuracy merely shows decodability. Sparse-autoencoder reconstruction introduces yet another approximation. Keep these evidential levels separate. References: [activation additions](https://arxiv.org/abs/2308.10248), [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) and [ROME's causal tracing/model-editing study](https://arxiv.org/abs/2202.05262).

## 6. Differential privacy and federated training

**Key terms.** **Differential privacy** bounds result sensitivity to one individual's inclusion; **ε/δ** describe a guarantee; **federated learning** keeps data at participants.

**Process.** Define the privacy unit and threat model, then select training/aggregation and track privacy accounting. Distributed data storage alone does not make updates or outputs private.

Differential privacy limits how much an output distribution can change when one defined data unit changes. A standard statement is:

\[
P[M(D)\in S]\le e^\epsilon P[M(D')\in S]+\delta
\]

for neighbouring datasets under a specified relation. Smaller `ε` generally means stronger privacy under that definition; `δ` is an allowed slack term. Privacy accounting across many steps is essential. DP-SGD clips per-example gradients and adds calibrated noise; naive clipping without noise is not DP. User-level and record-level protection are different.

Federated learning trains across clients and aggregates updates. Non-identical client distributions, unreliable connectivity, malicious updates and privacy leakage are core issues. Secure aggregation hides individual updates under assumptions; it does not itself ensure model robustness or differential privacy. This is specialist material, not a minor configuration change to a local training script.

## 7. Reasoning, world models and emerging directions

**Key terms.** A **world model** predicts environment behaviour; **test-time search** explores continuations; **verification** checks proposed answers/actions.

**Process.** Identify extra state, search and feedback, then compare under matched budgets. Separate better outcomes from simply longer reasoning or more samples.

Test-time search, verifier-guided generation, tool-integrated reasoning and learned process rewards are separate mechanisms. A model can improve a benchmark by spending more sampling/search compute without changing weights. RL can alter how that computation is used. Distinguish pass@k—whether any of k candidates succeeds—from selecting the right answer in practice.

Long-term agents, grounded world models, causal reasoning, multilingual transfer, multimodal reasoning and diffusion-language models remain active areas with unsettled questions. Do not claim their benchmark gains establish human-like understanding or general reliability. Derive a new study from a concrete failure and a controlled comparison.

## 8. Specialist branches not expanded into full courses

**Key terms.** A **specialist branch** has its own assumptions, data and evaluation; **coverage** here means orientation rather than implementation or mastery.

**Process.** Identify prerequisites and read a branch's primary sources. Build a focused experiment before claiming practical expertise in a method merely mentioned here.

Information retrieval theory, compiler/kernel engineering, distributed fault tolerance, Bayesian deep learning, formal verification, privacy accounting, detailed speech/vision generation, advanced RL proofs and comprehensive mechanistic interpretability each require substantial further study. You have enough of their role and assumptions here to recognise the boundary, not to claim expertise.

## Final exercise

A new paper claims “50% cheaper reasoning with better accuracy.” Ask whether it changes the model, token budget, candidate count, retrieval or hardware; whether costs include training; whether tests are contaminated; whether outputs are selected using an oracle; and whether quality persists across domains. Write the strongest claim the evidence actually supports.

Primary examples: [ROME/model editing](https://arxiv.org/abs/2202.05262), [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), [differentially private deep learning](https://arxiv.org/abs/1607.00133), [FrugalGPT routing](https://arxiv.org/abs/2305.05176). These are starting examples, not a complete specialist bibliography.
