# N24. NLP evaluation and a complete study

An NLP evaluation counts matches between predicted and reference objects under a declared task rule. Token tags, complete entity spans, answer tokens and generated sequences are different objects, even when some metrics share a familiar name. A complete study also records preprocessing, model selection, source groups and failures so the result can be interpreted and reproduced.

Two gold entities and two predicted entities have one exact span-and-type match, giving one correct match, one false positive and one false negative. A separate six-token example has five correct tags. The scores differ because the evaluated objects differ.

The task lessons introduced distinct outputs and error sources. This final lesson connects them to the shared metric formulas and an executable research workflow without inventing one universal NLP accuracy.

## 1. Task-specific matching and metric units

For exact NER evaluation, both the entity span and type must match under a declared scheme/scorer. A wrong boundary can produce one false positive and one false negative even if most token labels are correct. POS token accuracy, NER entity F1 and QA answer-token F1 evaluate different objects.

The authored example has two gold entities and two predictions, only one of which exactly matches: precision=recall=F1=0.5. A separately defined six-token labelling example has five correct tags, or 0.8333 token accuracy. These are not alternative calculations of the same metric. Use the right record-level matcher before invoking a shared count formula.

For generation, BLEU/chrF/ROUGE, semantic similarity and human/learned factuality checks add different evidence. Their scores must not be mixed into one unexplained universal accuracy.

## 2. Reproducible NLP studies and practice routes

Build one source-grouped text-classification study: lexical baseline, masked sentence-vector classifier and an optional fine-tuned encoder, keeping splits and selection budgets explicit. Then build a separate token/span study so pooling and sequence decoding are not confused. Record data/model licenses, revision IDs, seeds, preprocessing, label policies and per-case outputs.

Use scikit-learn for sparse pipelines; PyTorch for sequence tensors/packing; spaCy or NLTK for linguistic processing; Transformers/Tokenizers/Datasets for pinned encoders and data alignment; Sentence Transformers for pooling and retrieval; seqeval and SacreBLEU for task-specific scoring. These links describe an implementation route, not libraries automatically installed or new web trainers. Verify dataset access/licences and dependencies before downloading anything.

Inspect important slices and valid counterfactuals, report conditional uncertainty with appropriate groups, and distinguish development exploration from final checking. Dataset/model cards document intended use and limitations; they are not safety certificates.

### Data routes for progressively harder exercises

| Task | First-party route | What to preserve or inspect |
| --- | --- | --- |
| Sparse text classification | [20 Newsgroups in scikit-learn](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset) | Original split, headers/quotes/footers as possible shortcuts, corpus provenance |
| POS and dependency parsing | [UD English EWT](https://universaldependencies.org/treebanks/en_ewt/index.html) | CoNLL-U tokenization, sentence/document metadata, scheme and original licence |
| Reading and no-answer decisions | [SQuAD official data and scoring](https://rajpurkar.github.io/SQuAD-explorer/) | Context groups, character anchors, answerability and evaluation normalization |

These links were checked during course preparation. Nothing is automatically downloaded. Review the original data/source terms, version and permitted use before obtaining or redistributing a corpus. Dataset availability does not mean identical permissions or an absence of sensitive content.

### A local first coding exercise

Use a tiny authored fixture only to check the pipeline wiring:

```python
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

train_text = [
    "the service was helpful", "a useful and clear answer",
    "excellent support today", "a pleasant experience",
    "the service was awful", "a confusing and wrong answer",
    "poor support today", "an unpleasant experience",
]
train_y = [1, 1, 1, 1, 0, 0, 0, 0]
pipeline = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    LogisticRegression(C=1, max_iter=200, random_state=0),
)
pipeline.fit(train_text, train_y)
probe = ["helpful support", "not helpful support", "unseen jargon"]
print(pipeline.predict(probe))
print(pipeline.classes_, pipeline.predict_proba(probe))
```

First inspect the vocabulary and the difference between the two negation probes. Do not call these three outputs a measured generalization result: there is no independent benchmark here, and eight training sentences are deliberately insufficient for a useful sentiment system. Next obtain one approved real corpus, preserve source groups/splits, fit preprocessing only on training and perform the capstone's controlled comparisons.

## Sources

[seqeval strict sequence evaluation](https://github.com/chakki-works/seqeval) · [Speech and Language Processing — author-hosted draft](https://web.stanford.edu/~jurafsky/slp3/)
