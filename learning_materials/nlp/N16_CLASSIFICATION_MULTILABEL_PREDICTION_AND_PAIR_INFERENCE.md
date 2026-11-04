# N16. Text classification and paired-input inference

Text classification maps a document, token sequence or input pair to a defined label structure. Single-label classification chooses one mutually exclusive class; multilabel classification can assign several labels at once. The target structure determines the loss and decoding rule, not merely how many label names appear in a dataset.

The three supplied logits are [1,−1,0.2] for sports, politics and finance. Switch between one exclusive class and multiple allowed labels to compare softmax with independent sigmoids, then inspect the corresponding prediction rule.

Encoders and lexical pipelines create representations; a classifier turns them into task outputs. This lesson connects the output head to label semantics before the course moves to richer span and relation structures.

## 1. Output units, targets and losses

Single-label classification chooses one mutually exclusive class, usually with a softmax/cross-entropy objective. Multilabel classification permits several labels at once, often using independent sigmoid outputs and binary cross-entropy. A probability sum of one is not appropriate merely because the task has several possible labels.

Token classification predicts one output per token, while document classification pools or aggregates evidence for one output per document. Aspect-based sentiment may require identifying an aspect and judging sentiment toward it; a single document polarity can miss the task. Ordinal labels, hierarchies and severe imbalance may need additional modelling/evaluation choices.

For paired inputs, a cross-encoder can classify a premise–hypothesis or query–answer relation. Entailment is directional: swapping the pair can change the correct label. Similarity, paraphrase and entailment are not synonyms.

## 2. Threshold selection and error analysis

Choose decision thresholds on development data, per label only when the protocol and sample size support it. A sigmoid output is not automatically a calibrated probability; class weighting can change its interpretation. Shared metrics/calibration lessons own the general definitions.

Inspect negation, rare labels, document length and author/source leakage. Distinguish zero-shot label descriptions, frozen sentence-vector classifiers and supervised encoder fine-tuning. Their input access and training budgets differ. A larger encoder should earn its additional cost under a matched task, not by replacing a weak or incorrectly fitted baseline.

## Sources

[Hugging Face sequence classification guide](https://huggingface.co/docs/transformers/tasks/sequence_classification) · [Speech and Language Processing — author-hosted draft](https://web.stanford.edu/~jurafsky/slp3/)
