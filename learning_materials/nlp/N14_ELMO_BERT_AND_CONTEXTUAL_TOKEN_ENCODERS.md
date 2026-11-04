# N14. Contextual token encoders: ELMo and BERT

A contextual encoder gives a token a representation that depends on its occurrence, not just its word identity. ELMo combines directional recurrent language-model features; BERT uses bidirectional Transformer context and masked-token pretraining. These methods produce token-level features that still need an appropriate head or pooling rule for a downstream task.

The illustration supplies a three-position input with a masked target and hidden width four. A target probability of 0.8 gives negative log-likelihood about 0.2231. The shapes and probability are supplied examples, not an actual downloaded BERT run.

Static word vectors reuse one lexical lookup, while recurrent and attention mechanisms can incorporate context. This lesson connects those mechanisms to pretraining objectives and explains why token encodings are not automatically good sentence-similarity embeddings.

## 1. Contextual pretraining objectives

ELMo builds contextual representations from directional LSTM language models and a task-specific weighted combination of layers. The forward and backward language-model states are combined, but this is not the same computation as a jointly bidirectional masked-attention encoder.

BERT uses context on both sides and predicts selected tokens from a corrupted input during masked-language pretraining. Original BERT sometimes keeps selected targets unchanged; the displayed all-masked example is a simplified corruption case. Original BERT used MLM and next-sentence prediction; RoBERTa does not simply repeat that recipe, ELECTRA uses replaced-token detection, and DeBERTa changes attention/position handling. These names identify different design choices, not an ordered universal ranking.

A bidirectional encoder can process a complete input sentence without being a causal next-token generator. Loss masking chooses which targets are scored; the attention mask chooses what positions can read. They cannot replace one another.

## 2. Token outputs, pooling and task heads

Token features have shape B × T × H. A token head returns B × T × C; a pooled classifier returns B × C. Special-token and subword policies determine which positions receive labels. For frozen encoders, put dropout in the intended mode and record the exact model/tokenizer revision.

The [CLS] vector is not automatically optimized for cosine sentence similarity. Fine-tuning a classifier, training a sentence embedding and extracting a frozen token representation are different uses. Existing Transformer lessons explain attention blocks; here the new material is contextual pretraining, pooling choices and task alignment.

## Sources

[BERT original paper](https://aclanthology.org/N19-1423/) · [Speech and Language Processing — author-hosted draft](https://web.stanford.edu/~jurafsky/slp3/) · [ELMo](https://aclanthology.org/N18-1202/)
