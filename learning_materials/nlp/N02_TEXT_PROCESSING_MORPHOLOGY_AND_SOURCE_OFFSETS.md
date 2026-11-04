# N02. Text processing, morphology and source offsets

Text processing changes how language is represented before a model uses it. Normalization, stemming, lemmatization and subword tokenization perform different jobs and may discard different information. Extraction tasks also need a reliable map back to the original text so that a predicted name or answer can be located exactly.

The original string is 'not good'. Its word spans are [0,3) and [4,8), using zero-based character positions and an exclusive end. Removing 'not' changes the sentiment; it is not harmless cleanup even though the remaining string is shorter.

An annotation policy defines which distinctions matter. Preprocessing must preserve those distinctions and the required source anchors before feature extraction, token tagging or span evaluation can be trusted.

## 1. Normalization, morphology and lost distinctions

Unicode normalization, case handling, tokenization, stemming and lemmatization are different operations. A stemmer applies surface heuristics; a lemmatizer chooses a dictionary form using linguistic analysis. A morphological analyser may return several features or analyses. Lowercasing can merge “US” and “us”; deleting stopwords can remove negation. There is no universally harmless cleaning recipe.

Keep the original string and a mapping from processed tokens back to original character spans. Character offsets are not UTF-8 byte offsets or subword indices. Unicode composition can change code-point counts, and different libraries use different conventions. Validate an offset by slicing the exact original string under the documented convention.

The example “not good” becomes “good” if negation is removed. This is a changed meaning, not a successful denoising step. Normalize only when the task allows the lost distinction.

## 2. Source offsets and subword labels

For tagging or extraction, save document ID, original text, token start/end offsets, tokenization version and gold spans. When a word splits into subwords, choose a first-piece-only loss or a documented label-propagation scheme. Do not duplicate entity evaluation counts simply because the tokenizer generated more pieces.

spaCy provides document/token/span objects and task pipelines; NLTK is useful for linguistic algorithms and corpus exercises. Hugging Face Tokenizers and SentencePiece address subword encoding, not the same job as a morphological analyser. Inspect language-specific rules and preserve offsets through truncation. Treat a pipeline change as an experiment, not an invisible preprocessing improvement.

## Sources

[spaCy linguistic features and tokenization](https://spacy.io/usage/linguistic-features) · [Speech and Language Processing — author-hosted draft](https://web.stanford.edu/~jurafsky/slp3/)
