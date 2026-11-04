# N05. N-gram language models and smoothing

A language model assigns probabilities to text sequences, often by predicting each token from an earlier history. An n-gram model restricts that history to the previous n−1 tokens. Smoothing adjusts count estimates so that unseen continuations can receive probability rather than making an entire held-out sequence impossible.

After 'like', the training counts are cats twice and dogs once. There are seven permitted next-token items, including an unknown token. Change the continuation and smoothing strength to see both its probability and negative log-likelihood change.

Text classification predicts labels; this lesson predicts the next token from counts. It introduces the same conditional-probability target that later recurrent and large language models learn with richer representations.

## 1. Smoothing and unseen continuations

An n-gram model conditions on a finite history. A bigram uses one previous token; it does not carry a learned hidden state. Maximum-likelihood counts assign zero probability to unseen continuations. Add-alpha smoothing adds mass to every item in a declared vocabulary and changes the denominator by alpha times vocabulary size.

The example conditions on “like”, observed three times: cats twice and dogs once. Its seven-item next-token vocabulary includes an unknown symbol. Mapping a truly new word to that symbol happens before scoring. Without smoothing, that continuation has zero probability and infinite negative log-likelihood; the interface reports this explicitly rather than replacing it with zero loss.

Backoff and interpolation combine evidence at different history lengths. Modified Kneser–Ney uses discounted counts and continuation information; it is not merely add-one smoothing with another name.

## 2. Sequence boundaries and language-model evaluation

Include beginning/end markers according to a documented policy, and do not let document endings train accidental transitions into unrelated documents. Fit counts, vocabulary and smoothing choices without looking at test answers.

A language model scores or generates word sequences; a text classifier predicts a task label. They can share counts or representations but optimize different objectives. Perplexity belongs to a matched tokenization and scored region; its complete calculation is in the shared metrics explorer. Use NLTK's language-model interfaces for an implementation exercise and compare log scores on the same held-out token sequences.

## Sources

[NLTK language-model API](https://www.nltk.org/api/nltk.lm.html) · [Speech and Language Processing — author-hosted draft](https://web.stanford.edu/~jurafsky/slp3/)
