# N23. Long-document encoding and evidence coverage

A long-document system must specify how distant text can influence a prediction. Chunking encodes local windows and combines their representations; a hierarchical encoder adds a second level; sparse attention changes token-to-token connections within a sequence. These are different information paths, not interchangeable ways of increasing a context-length setting.

Eight token positions are processed in four-token windows. An answer needs positions 3 and 4 together. Overlap two makes that boundary jointly visible while increasing processed positions from eight to twelve; a separate control illustrates local versus global sparse-attention connections.

The earlier sequence encoders assumed the relevant input was accessible. Long documents expose a new failure: evidence can be lost at truncation or window boundaries before an otherwise capable model can combine it.

## 1. Chunking, hierarchy and sparse attention

Chunking fits local windows into an encoder, then aggregates their outputs. A hierarchical model can encode tokens into sentence vectors and sentences into document features. Mean pooling, attention pooling and a recurrent second level preserve different information. A bag of chunk vectors does not automatically know the order or cross-chunk relations.

Overlapping windows can preserve a span crossing one boundary but duplicate token exposure and computation. The example counts both eight unique tokens and twelve processed positions when four-token windows overlap by two. That distinction matters for fair comparisons and aggregation weights.

Sparse attention changes which token pairs can interact. A local window limits direct reach; designated global tokens add paths. It is not the same operation as independently encoding chunks and averaging them.

## 2. Evidence boundaries and fair comparison

For classification, some evidence may be local while the decision is document-wide. For extraction, a gold span crossing a hard window boundary can become unrecoverable unless windows overlap or another mechanism connects them. For retrieval, a good chunk may omit the document qualification needed to interpret it.

Longformer-style local/global attention, hierarchical encoders and retrieval-assisted reading are alternative designs, not free extensions to a model's trained context length. Match truncation, evidence access, number of windows and token budget. The diagram computes masks/coverage only; it does not execute a pretrained long-context encoder or claim end-to-end quality gains.

## Sources

[Longformer original paper](https://arxiv.org/abs/2004.05150) · [Speech and Language Processing — author-hosted draft](https://web.stanford.edu/~jurafsky/slp3/)
