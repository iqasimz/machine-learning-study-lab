# P7. Embeddings, similarity and retrieval

An embedding represents an object, such as a token, sentence or document, as a numerical vector. A retrieval system compares a query with candidate documents and returns a ranked list. Vector similarity can help find related text, but it does not establish that the text answers the question or supports a claim.

The query vector is [1,0]. Document A is [0.8,0.6] and document B is [0,1]. All have length one, so their dot products give cosine similarities of 0.8 and 0. Selecting top one or top two changes the returned list, not the scores.

Token IDs identify discrete pieces; embeddings supply numerical representations on which models and search systems can operate. This lesson introduces retrieval geometry before later lessons compare how lexical and neural encoders learn their representations.

## 1. Embeddings and evidence

**Key terms.** An **embedding** is a numerical representation; **similarity** is a comparison of representations; **evidence** directly supports a claim.

**Process.** Text becomes vectors, and vector similarity proposes related items. Inspect their actual contents before using them as evidence; numerical closeness does not establish factual support.

An embedding maps an object to a vector. Token embeddings are learned lookup vectors; contextual embeddings change with surrounding input; sentence/document embeddings are designed to represent larger units. A model good at token prediction is not automatically a good document embedding model.

Dense embeddings store many nonzero coordinates. Sparse representations activate relatively few vocabulary-like features. A learned sparse retriever can expand terms beyond literal occurrence. Neither “dense” nor “neural” implies better retrieval for every domain.

## 2. Similarity calculations

**Key terms.** A **norm** measures vector length; a **dot product** sums coordinate products; **cosine similarity** divides that product by both lengths.

**Process.** Obtain two vectors, choose a similarity convention and calculate their score. Ranking uses that score next, so normalization must match the encoder's training and index configuration.

**Calculation.** For vectors a=[3,4] and b=[0,5], the dot product is 20 and both norms are 5. Cosine similarity is 20/(5×5)=0.8. A larger norm would change the raw dot product even if the direction stayed the same.

Cosine similarity is `(q·d)/(||q||||d||)`. For `q=[1,0]`, document A `[2,0]` has cosine 1; B `[1,1]` has cosine about 0.707. Inner product ranks A higher partly because of magnitude; cosine removes that magnitude effect. For unit vectors, squared Euclidean distance is `2−2cosine`, so those rankings agree.

Negative numbers are possible in cosine. A cosine of 0.8 is not an 80% probability that a passage supports a claim. Distances and relevance labels need task-specific interpretation.

## 3. How training changes embedding geometry

**Key terms.** An **encoder** maps text to a representation; a **positive pair** should match; a **negative pair** should be separated; **contrastive loss** learns that distinction.

**Process.** Encode training pairs, compare their scores and penalize mismatched rankings. Gradients update the encoder; a later retrieval system uses the trained representations without repeating that training.

The encoder has learned parameters; a similarity calculation on fixed vectors does not itself train them. In contrastive training, the loss penalizes giving too little relative score to a designated matching pair compared with supplied alternatives. Backpropagation then changes the encoder parameters so future representations can differ. Merely retrieving top-k documents leaves those parameters unchanged.

Hard negatives look plausible but are judged wrong for this training task. False negatives are actually relevant documents labelled negative. The second case teaches an incorrect distinction, so a larger collection of negatives is not automatically better. The label policy and sampling procedure define what similarity the model is encouraged to learn.

The [sentence-encoder lesson](../nlp/N15_SENTENCE_ENCODERS_BI_ENCODERS_AND_CROSS_ENCODERS.md) owns the numerical contrastive-loss example and the detailed comparison of separate encoding, joint pair encoding and token-level late interaction. Here the important distinction is between learning the geometry and searching within a geometry that is already fixed.

## 4. Lexical retrieval still matters

**Key terms.** **Lexical retrieval** matches terms; **BM25** scores term evidence using frequency and length adjustments; **hybrid retrieval** combines lexical and vector signals.

**Process.** A query first produces candidate documents. Exact terms and semantic similarity supply different evidence, which may be combined before a more expensive reranker checks the candidates.

BM25 scores term matches with rarity weighting and document-length normalization. Rare exact terms—product IDs, drug names, error codes—can be crucial. Dense retrieval may capture paraphrases but confuse near-opposite meanings or lose exact identifiers. **Hybrid retrieval** combines lexical and dense signals.

Raw BM25 and cosine scores have different scales. One simple fusion is **reciprocal rank fusion**: add `1/(k+rank)` from each ranked list. `k` is a smoothing constant, not the number of documents returned. For illustration, with `k=60`, a document at ranks 1 and 3 scores `1/61+1/63≈0.03227`. This is a ranking heuristic, not a calibrated relevance probability.

## 5. Measuring retrieval

**Key terms.** **Recall@k** measures relevant items recovered in the first k results; **MRR** averages reciprocal first-relevant rank; **nDCG** rewards useful ordering with graded relevance.

**Process.** Freeze relevance labels, retrieve a ranked list and compare it with those labels. Evaluate candidate recall before judging the final answer generator, which cannot use evidence never retrieved.

- Recall@k: fraction of relevant items found among the first k, per query under a stated definition.
- Precision@k: fraction of retrieved items that are relevant.
- MRR: average reciprocal rank of the first relevant result.
- nDCG: compares discounted graded relevance with an ideal ranking.

For nDCG, a common gain is `2^relevance−1` and discount `log₂(rank+1)`. Specify the convention. If there are no relevant documents, define how the query enters the metric. Report retrieval quality separately from answer quality: a generator can fail even when retrieval succeeds.

## 6. Index compatibility and provenance

**Key terms.** **Index drift** means stored vectors no longer match the current encoder or data; **metadata** records provenance and permissions.

**Process.** Build vectors with a recorded encoder revision and carry source metadata into the index. When data or encoders change, refresh compatible entries and retest retrieval before serving answers.

Pooling, normalization, query/document instruction prefixes and maximum input length must match the embedding model. Re-embedding queries with a new model while leaving old document vectors in place mixes incompatible spaces. A 2D projection can create misleading visual neighbours; dimensionality reduction is a view, not the original geometry.

Negation and numbers are particularly important in scientific evidence: “did not reduce mortality” may be close in embedding space to “reduced mortality.” Retrieve the candidate, then verify the actual relation and scope. RAG will reuse this separation between finding relevant text and proving support.

### Build a document vector without silently changing the method

A Transformer returns token-level states; an embedding pipeline must define how those states become one searchable vector.

1. Apply the model's required query/document instructions and tokenizer. Keep the model revision and maximum length fixed.
2. Use the trained pooling rule. Mean pooling is not automatically interchangeable with a first-token or last-token representation.
3. For masked mean pooling, include only real positions: sum `mask[t]×hidden[t]` and divide by `sum(mask)`. Including padding makes the vector depend on how other examples were batched.
4. Apply normalization only if appropriate for the trained scoring convention. For unit vectors, dot product equals cosine; without normalization, magnitude affects the ranking.
5. Audit retrieval with relevance labels and difficult negatives. If two passages differ only by “not,” a high cosine may still be a serious evidence-selection error.

Contrastive learning compares positives with alternatives. A small temperature sharpens score differences and gradients; increasing batch size changes the available negatives, not just throughput. In-batch negatives can accidentally include other correct passages, so inspect the label policy. See [Sentence-BERT](https://arxiv.org/abs/1908.10084), [Sentence Transformers pooling](https://www.sbert.net/docs/package_reference/sentence_transformer/models.html) and [ColBERT](https://arxiv.org/abs/2004.12832) for different representation choices.

## Exercises and checks

1. Query `[1,1]`, A `[1,0]`, B `[2,2]`: cosine similarities are about 0.707 and 1.
2. Relevant items are `{A,B}` and top 3 are `[C,A,D]`: recall@3 is 0.5; precision@3 is 1/3.
3. Why add reranking? To spend richer pairwise computation on a limited candidate set; it cannot recover a relevant document never retrieved unless another retrieval step adds it.

Implementation references: [Sentence Transformers](https://www.sbert.net/) for embeddings and rerankers, and [ColBERT](https://arxiv.org/abs/2004.12832) for token-level late interaction. Work through the cosine and contrastive-loss calculations above before selecting a retrieval index.
