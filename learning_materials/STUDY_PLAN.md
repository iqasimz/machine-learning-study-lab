# How to study this course

## Pace without a false mastery deadline

Choose the subject path that matches your prerequisites and goals. A study block can cover a short section, one worked calculation or a small experiment. Attention, distributed systems and reinforcement learning usually require several blocks. Time spent reading is not by itself evidence that you can implement or explain a method.

If Python is new, reserve time for programming practice alongside the mathematical foundations. Adjust the balance using the exercises you can solve independently rather than a fixed completion deadline.

## The six-step study loop

1. **Explain:** read one section and say it aloud without its terminology. “A query asks which earlier representations are relevant.”
2. **Draw:** reproduce the diagram with a pencil. Label inputs, outputs, weights, temporary state and trust boundaries.
3. **Calculate:** work the small numerical example before reading its answer. Keep units and tensor shapes.
4. **Change:** alter a value or assumption and predict the result. What happens if the context doubles? What if all rewards match?
5. **Test:** run the supplied arithmetic script or write a small experiment. An unexpected answer is a debugging opportunity.
6. **Defend:** answer the checkpoint with no generated response in front of you. State a limitation and a falsifying test.

Take one-page notes per lesson: problem → mechanism → calculation → failure → comparison. Do not copy every paragraph into another notebook.

## A 20-hour week after the basic Python gate

| Activity | Hours | Evidence |
| --- | ---: | --- |
| Read and annotate two or three sections | 4 | Questions and one plain-language summary |
| Work maths by hand | 4 | Calculations with units/shapes |
| Implement or modify a small experiment | 6 | Your code, tests and error log |
| Visual exploration | 2 | Predictions versus observations |
| Retrieval practice and oral answers | 2 | Recorded explanation or written answers |
| Review and plan | 2 | Corrected errors, progress entry |

## Gates

**Before T1:** implement functions, use lists/dictionaries, read a traceback, understand a train/test split, multiply a small matrix, differentiate a squared loss, and calculate a conditional probability. Review programming fundamentals and the shared foundations if any of these need practice.

**Before R1:** trace a causal Transformer, identify training targets and masks, estimate memory, explain why loss can decrease while usefulness worsens, and describe one reproducible baseline.

**Before deployment projects:** show held-out evaluation, a permission model, a budget, a timeout, a rollback and an escalation path. A demo that works once is not ready to serve other users.

See [Assessments](ASSESSMENTS.md) for actual prompts and answer criteria.

## Three passes, not three parallel courses

Use the training labs to test model choices and the systems exercises to practise retrieval, permissions and operational decisions. Start with one small experiment and state the comparison it is intended to answer.

Use the existing two hours of retrieval/oral practice for [technical interview work](interview_preparation/README.md): one explanation/calculation, one code/debug variation and one design follow-up tied to the current stage. Every fourth study block, run a short mixed rehearsal. Keep [readiness evidence](interview_preparation/READINESS_TRACKER.md); fluent terminology without an independent test is not a passed gate.

- **Pass A — understand:** do the lesson, hand example and verbal checkpoint. Avoid reading all source papers.
- **Pass B — implement:** tiny model, retrieval system, evaluation harness. Inspect tensors and predictions.
- **Pass C — specialise:** choose research training, application engineering or inference/edge. Deepen the relevant advanced branches, retaining the common core.

For your NLP research route, prioritise fair comparisons, data leakage, SFT, controlled reasoning studies, evaluation, RAG and RL foundations. This is prioritisation, not permission to pretend distributed training or production security is already mastered.

## What to do when a paper feels impossible

Write five sentences before studying its derivation: what was broken, what was changed, what signal trains it, what baseline it beats, and what remains untested. Then follow only the missing mathematics. Reconstruct one equation with a two-example toy case. Return to the full paper afterwards.

## Version discipline

Before a hands-on framework lab, record Python/library versions, model revision, tokenizer revision, chat template and hardware. Documentation may describe a newer release than your environment. Pin compatible versions rather than installing every “latest” package. No example requires access to private systems or spending money.
