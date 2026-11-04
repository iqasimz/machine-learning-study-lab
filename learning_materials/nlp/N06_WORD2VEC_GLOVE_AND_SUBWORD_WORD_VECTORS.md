# N06. Static word vectors: Word2vec, GloVe and fastText

A static word embedding assigns a reusable numerical vector to a word or lexical unit. Word2vec, GloVe and fastText learn these vectors from different forms of corpus evidence. Nearby vectors can capture useful regularities, but the same static lookup does not change meaning according to the sentence in which a word appears.

The worked example uses one centre word, one observed context word and one sampled negative. It calculates their compatibility scores and a negative-sampling loss from supplied vectors. No word vectors are trained when you select the explanation.

Lexical counts keep explicit vocabulary columns. Word embeddings instead learn a compact shared feature space, preparing the transition to text CNNs and recurrent models that operate on sequences of vectors.

## 1. Static-vector training objectives

Skip-gram predicts surrounding words from a centre word; CBOW predicts a centre from its neighbours. Negative sampling replaces a full vocabulary softmax with binary contrastive terms under a chosen noise distribution. The browser calculates one positive and one negative term, not the full training objective of a real corpus.

GloVe fits weighted relations between word/context vectors and global co-occurrence counts. fastText represents a word using character-subword information as well as lexical information, which can help form a vector for some unseen words. These are distinct learning recipes. A vector's axes do not come with verified human concepts.

Static vectors attach one representation to a lexical item under the chosen lookup policy. The word “bank” does not receive a different vector just because this occurrence describes a river. That limitation motivates contextual encoders, not an assumption that static vectors are useless.

## 2. Averaging, vocabulary coverage and vector bias

Average word vectors with a documented mask/weighting policy and compare to a sparse model. Averaging loses order, and normalizing before versus after aggregation changes the operation. Word frequency, missing vocabulary and subword collisions can affect downstream behaviour.

Intrinsic similarity tests do not substitute for the final classification, retrieval or transfer task. Review licensing and demographic/domain biases in pretrained vectors. Gensim provides Word2Vec/FastText tooling; use the model's actual vocabulary/subword API rather than assuming all vector files support unknown words. Sentence-encoder training is a later lesson, not a consequence of averaging any available vectors.

### Read the other training objectives

GloVe's weighted least-squares objective is
\[
J=\sum_{i,j:X_{ij}>0} f(X_{ij})
\left(w_i^\top\widetilde w_j+b_i+\widetilde b_j-\log X_{ij}\right)^2.
\]
X records context co-occurrences and f downweights selected count ranges. The word and context tables and biases are separate learned quantities. This is not the same loss as SGNS, even if both produce useful lexical vectors.

A subword model can compose a lexical representation from a set of character-n-gram vectors, optionally including a word-specific vector. That adds an inductive bias for related spellings; it does not prove semantic equivalence, correct morphological segmentation or coverage of every unknown script.

## Sources

[Word2vec negative sampling paper](https://arxiv.org/abs/1310.4546) · [Speech and Language Processing — author-hosted draft](https://web.stanford.edu/~jurafsky/slp3/) · [GloVe](https://aclanthology.org/D14-1162/) · [fastText subword vectors](https://aclanthology.org/Q17-1010/)
