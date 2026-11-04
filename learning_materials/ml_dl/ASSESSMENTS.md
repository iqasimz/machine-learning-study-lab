# Assessment plan

An original course needs original work checks. Reading a file or reproducing a model answer is not demonstrated mastery. None of these is marked passed.

## A1 — Data and baseline

Choose a task, define prediction time, identify forbidden fields and split independent units. Calculate mean/median/majority baselines by hand on a small separate set. Deliver a task card, split IDs and one leakage counterexample. Explain why the baseline metric has its stated unit.

## A2 — Linear and probabilistic fitting

Derive squared-loss gradients and binary log-loss gradient. Implement both in NumPy, verify a finite difference on a tiny deterministic problem and compare with a trusted library under matched regularization/reduction. Show why an intercept or scaling mismatch changes the result. Explain a naive Bayes posterior and its independence assumption.

## A3 — Classical method comparison

Use Wine Quality or digits with a clean pipeline. Compare linear, neighbour/kernel and tree/ensemble families under a declared search budget. Save per-case predictions, settings, metric definitions, actual fit/predict timing and the held-out split. Diagnose a planted preprocessing leak. A justified simple winner is acceptable.

## A4 — Unsupervised structure

Reproduce one k-means iteration and one PCA projection. Compare scaled/unscaled inputs, initialization and retained dimensions. Identify why an attractive two-dimensional plot is not evidence of true classes. Do not use withheld labels to select an allegedly unsupervised pipeline.

## A5 — Neural mechanics

Hand-compute XOR features, one scalar backward/update, convolution shape/parameter count and a small attention row. Implement an MLP forward/backward or inspect autograd in detail. Catch missing gradient zeroing, a detached layer and the wrong target shape. Show an actual parameter before/gradient/after.

## A6 — Neural training diagnosis

Overfit a tiny clean batch, then train with a separate development set. Change one of learning rate, optimizer, dropout or batch and predict what could happen before running. Record actual loss/gradient/rate measurements and explain what the results do and do not support. Verify eval mode and bounded save/restore.

## A7 — Matched ML versus DL capstone

Use the same digits/Fashion task to compare linear, MLP and CNN, keeping source scope and split fixed. Include a strong tabular classical contrast elsewhere. Report parameters, data exposure, tuning work, outcomes and costs separately. Explain at least one negative or inconclusive result without inventing success.

## Oral defence prompts

- Why did you choose this task, split, metric and baseline?
- Which information was available at prediction time?
- What exactly changed in the claimed improvement?
- Which fitted transformations and state must be saved?
- Which algorithm assumptions fail in this example?
- What would you check first if training or generalization failed?
- What is the cheapest experiment that could disprove your interpretation?
- Why might a strong tree/linear method beat your neural model?
- What does the uncertainty procedure include and omit?
- Which part did you derive, implement, inspect or merely call from a library?

Assessment rubric: mathematical correctness, executable evidence, controlled design, clear explanation and honest boundaries. Provide an initial attempt, code and diagnostic notes before seeking a full answer. No benchmark cutoff alone proves competence.
