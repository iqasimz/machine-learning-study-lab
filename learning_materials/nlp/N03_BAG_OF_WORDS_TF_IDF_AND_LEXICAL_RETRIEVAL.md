# N03. Word counts, TF-IDF and lexical retrieval

A bag-of-words representation gives each vocabulary term a feature column and counts its occurrences in a document. TF-IDF weights those counts using term frequency and inverse document frequency, reducing the relative influence of terms that occur across many documents. This supplies sparse features for classification or lexical search without learning contextual word meaning.

Our corpus contains 'cat sat', 'cat slept' and 'dog sat'. The vocabulary has four columns. Cat occurs in two documents and dog in one, so dog receives a larger rarity weight under the displayed smoothed IDF rule.

Once text has been processed consistently, counts provide a transparent first representation. These features can feed the text classifiers in the next lesson or support lexical retrieval, offering a baseline before neural encoders.

## 1. Sparse features, TF-IDF conventions and BM25

Bag of words counts vocabulary items; word n-grams retain limited order and character n-grams capture spelling fragments. The vocabulary is fitted on training data. An unseen token is ignored or assigned an explicit unknown feature according to the implementation. Hashing uses a fixed feature space and may collide; it is not an inverse dictionary.

The worked example uses raw term frequency and smoothed IDF \(1+\log((1+N)/(1+df))\), followed by row-wise L2 normalization. Those choices match one common library configuration, not every definition of TF-IDF. Binary counts, sublinear frequency and other normalization rules change the vectors.

BM25 instead saturates term frequency and adjusts for document length. Its scores are lexical matching scores, not class probabilities or calibrated semantic truth. Exact term matching can be valuable for identifiers and rare terminology even when dense retrieval is available.

## 2. Fitting text features without test leakage

Use a scikit-learn Pipeline containing the vectorizer and estimator inside each training fold. Fitting IDF on all data before cross-validation lets held-out corpus statistics enter feature construction. Preserve the vocabulary, IDF array, normalization and label order alongside the classifier.

A sparse model can expose highly weighted phrases, but a coefficient is not a causal linguistic explanation. Check negation, spelling, domain terminology and document length. Compare word and character features under the same split before deciding a neural model is necessary. Dense embedding foundations remain in the existing retrieval lesson rather than being repeated here.

### Read a conventional BM25 term contribution

One common variant uses
\[
IDF(t)=\log\left(1+\frac{N-df(t)+0.5}{df(t)+0.5}\right),
\qquad score_t=IDF(t)\frac{tf(t,d)(k_1+1)}{tf(t,d)+k_1(1-b+b|d|/avgdl)}.
\]
N counts documents, df counts containing documents, tf counts this document's occurrences, |d| is its length and avgdl is mean training/index document length. k1 controls frequency saturation; b controls length normalization. At b=0 the explicit length normalization disappears. At b=1 it fully uses the specified length ratio. Summing term contributions gives this query score. Index implementations vary in IDF, query weighting and normalization; preserve the actual formula rather than equating BM25 and the displayed TF-IDF vectors.

## Sources

[scikit-learn text feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) · [Speech and Language Processing — author-hosted draft](https://web.stanford.edu/~jurafsky/slp3/)
