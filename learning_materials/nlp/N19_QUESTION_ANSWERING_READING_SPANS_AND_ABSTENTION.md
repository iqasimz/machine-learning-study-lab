# N19. Question answering, answer spans and abstention

Question answering predicts an answer to a question under a defined information-access setting. An extractive reader chooses a span in supplied context; a generative system may produce new wording. Answer correctness, source support and deciding when no answer is available are separate requirements, with different evaluation rules.

The context tokens are 'The red car stopped .', and the gold answer is red car. Select car, red car or The red car to compare exact answer matching with partial token credit. All span coordinates here use inclusive token ends.

Translation metrics compare generated wording with references. Question answering additionally needs a question, a permissible evidence source and an answerability decision; it must not reuse an unrelated classification or entity-matching score.

## 1. Answer spans, generated answers and missing answers

An extractive reader assigns start/end scores to context positions. A valid decoder enforces start ≤ end, permitted context positions and any maximum answer length. Independently choosing the two highest logits can produce an illegal span. A retrieval step can fail before the reader ever sees useful context.

A generative QA model may return words not present verbatim in a passage. Correctness, answerability and evidence support then require separate tests. An explicit null/abstain decision should be selected using development data and a declared cost/metric, not hidden test labels.

The small diagram uses an exact context-token alignment and three candidate spans. Its answer-token F1 is not classification macro-F1 or entity span F1. A real SQuAD-style evaluator also specifies normalization and multiple-reference handling.

## 2. Exact match, token overlap and source support

Exact match asks whether the normalized answer matches a reference; token-overlap F1 rewards partial matches under that evaluator's tokenizer/normalizer. A partial entity name may receive answer-token credit while failing exact entity extraction. That is a metric definition, not a contradiction.

SQuAD 2 introduces unanswerable questions; closed-book factual QA and source-grounded QA test different information access. Evaluate retrieval recall, reader correctness, abstention and source support separately. A plausible span in a retrieved passage does not establish that the passage answers the actual question. Keep source offsets and truncated-context coverage visible.

## Sources

[SQuAD 2 original paper](https://aclanthology.org/P18-2124/) · [Speech and Language Processing — author-hosted draft](https://web.stanford.edu/~jurafsky/slp3/)
