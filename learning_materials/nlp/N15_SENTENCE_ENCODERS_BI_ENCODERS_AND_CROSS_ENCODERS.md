# N15. Sentence encoders and query–document interaction

A sentence encoder represents a larger text unit for comparison rather than assigning one static vector to each word. A bi-encoder computes query and document representations separately; a cross-encoder reads the pair jointly; late interaction keeps token vectors and compares them later. Where the texts first interact determines caching, computation and what information can be compared.

Compare a query with two documents using one of three architectures. Bi-encoder mode shows masked pooling and cosine scores; cross-encoder mode shows a joint pair structure without inventing a relevance score; late interaction calculates each query token's best document match.

A contextual token encoder supplies per-position vectors. Turning those into retrieval scores requires a deliberate pooling, training and interaction design, extending the earlier embedding geometry rather than treating every encoder as an interchangeable search model.

## 1. Independent, joint and token-level matching

A bi-encoder encodes the query and document separately, making document vectors cacheable. Query/document encoders may share weights or use separate towers and prompts. A cross-encoder jointly processes each pair, allowing token interactions before producing a score; it cannot reuse one final document vector to score every query.

Sentence-BERT adapts encoders for sentence-level comparison. Mean pooling must exclude padding; CLS, mean and last-token pooling are not interchangeable defaults across model families. Normalization turns a dot product of unit vectors into cosine similarity, but scalar rescaling lost under normalization can still affect another downstream head.

ColBERT-style late interaction retains multiple token vectors and aggregates token matches, often MaxSim. This is neither one pooled vector nor a full joint cross-encoder. Token storage and indexing costs are part of that choice.

## 2. Contrastive training and retrieval evaluation

The illustrative contrastive objective treats one document as positive and another as a negative: \(-\log(\exp(s_+/\tau)/\sum_j\exp(s_j/\tau))\). Temperature changes the loss and gradients without changing the already fixed score order. In-batch negatives can be false negatives when documents are actually equivalent or relevant.

Unsupervised SimCSE uses dropout views of the same sentence; supervised variants use labelled relations. Hard-negative mining must not accidentally choose true positives or use hidden evaluation labels. Compare bi-encoder candidate recall and cross-encoder reranking separately: a reranker cannot recover a relevant document absent from its candidate set. Record pooling, prompts, normalization, model revision and search configuration with every embedding index.

## Sources

[Sentence-BERT original paper](https://aclanthology.org/D19-1410/) · [Speech and Language Processing — author-hosted draft](https://web.stanford.edu/~jurafsky/slp3/) · [SimCSE](https://aclanthology.org/2021.emnlp-main.552/) · [ColBERT](https://arxiv.org/abs/2004.12832) · [Retrieve and rerank implementation](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)
