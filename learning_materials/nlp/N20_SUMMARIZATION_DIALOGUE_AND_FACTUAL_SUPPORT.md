# N20. Summarization, dialogue and source faithfulness

Summarization selects and compresses source content, either by extracting passages or generating new wording. Source faithfulness asks whether the output preserves what the source supports; external truth asks a different question. Dialogue adds an interaction goal: helping a user may require clarification, correction or abstention rather than simply producing agreeable text.

The supplied source says 'trial improved symptoms'. Replacing improved with worsened leaves two of three reference words unchanged while reversing the main claim. The control reveals why lexical overlap alone cannot certify a faithful summary.

Generation scores and QA matching test limited forms of agreement. This lesson examines their blind spots and connects language-model outputs to source claims and actual interaction usefulness, without treating style as correctness.

## 1. Source faithfulness and summary evaluation

Extractive summarization selects source material; abstractive summarization generates new wording. Compression requires selecting what to omit, while faithfulness requires avoiding unsupported changes. A source-faithful summary may repeat a false source; a true external statement may still be unsupported by the supplied source.

ROUGE variants compare lexical or sequence overlap and use different matching/reduction rules. The example replaces “improved” with “worsened”: two of three unigrams still match even though the direction of the finding reverses. That toy recall is not a full implementation of every ROUGE variant and cannot certify factuality.

Evaluate claims, numbers, entities, scope, uncertainty and omitted qualifications. Do not turn a high overlap score or fluent style into proof that an answer is trustworthy.

## 2. Dialogue usefulness and correction

A dialogue system can manage intent/slots, retrieve a response, generate a response or combine those stages. Multi-turn state should reflect the actual user goal and evidence, not just the most recent wording. A satisfying response can agree with a false premise or avoid challenging an incorrect assumption.

Measure task completion, grounding, correction of false premises and appropriate uncertainty alongside preference judgements. LLM judges can inherit verbosity, order or agreement biases; the shared evaluation lesson covers judge auditing. Reducing unsupported content may require abstention or questions, not simply a longer reasoning trace. Keep evaluation examples independent from prompt iteration.

## Sources

[ROUGE original paper](https://aclanthology.org/W04-1013/) · [Speech and Language Processing — author-hosted draft](https://web.stanford.edu/~jurafsky/slp3/) · [Evaluation and interpretability](https://github.com/EleutherAI/lm-evaluation-harness)
