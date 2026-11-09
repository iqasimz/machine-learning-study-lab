# NLP code and libraries

Prerequisites: strings/Unicode, arrays and the distinction between supplied values and learned parameters. NLP begins with a task and a definition of its output, not with a universal text-cleaning recipe. No dedicated NLP trainer exists in the current lab; these are checked processing, representation, inference-interface and evaluation examples.

## NC01. Tokenization

**Tokenization** divides text into units. The simplest split cuts on spaces. That is not the same as stemming, lemmatization or predicting grammatical structure.

<!-- example: NC01-split -->
```python
text = "Not good."
print(text.split())
```

**Output**
```text
['Not', 'good.']
```

`split` keeps the period attached to `good`. A tokenizer can separate the period and remember where each piece started in the original string. Keep the raw text when later predictions must point back to character spans.

### Why spaCy and NLTK

spaCy and NLTK are the two Python libraries this course uses to turn text into tokens before a neural model is involved. spaCy is widely used in applied pipelines because a document is an object: each token keeps its text and its character offset, and a blank pipeline can stop at tokenization. `spacy.blank("en")` builds that tokenizer. NLTK is the teaching library that grew up with the field. Its `RegexpTokenizer` applies a pattern you can read. Both calls stay short because the rule is the argument. A blank spaCy pipeline does not parse grammar, and a regular expression does not know that a sentence is positive or negative. Neither library decides the task.

<!-- example: NC01-main -->
```python
import spacy
from nltk.tokenize import RegexpTokenizer

text = "Not good."
nlp = spacy.blank("en")
document = nlp(text)
print([(token.text, token.idx) for token in document])
print(RegexpTokenizer(r"\w+").tokenize(text))
print(document.has_annotation("DEP"))
```

**Output**
```text
[('Not', 0), ('good', 4), ('.', 8)]
['Not', 'good']
False
```

The spaCy tokenizer preserves punctuation as a token and source offsets. The regular expression selects word characters and drops punctuation. Neither has learned that the sentence is positive or negative. The blank pipeline does not include a trained dependency parser, so its absence is explicitly checked rather than printing empty tags as if they were valid annotations.

A token ID is the vocabulary number for a token. The same word receives the same number every time you look it up.

<!-- example: NC01-ids -->
```python
tokens = ["Not", "good"]
vocabulary = {"Not": 2, "good": 3}
print([vocabulary[token] for token in tokens])
```

**Output**
```text
[2, 3]
```

`"Not"` is 2 and `"good"` is 3 because the vocabulary says so. These numbers are not a sentiment score. A later lesson adds padding and a mask. A contextual model consumes the numbers, not the original letters.

Lowercasing, deleting punctuation or removing stop words can discard useful evidence. Removing “not” reverses the obvious sentiment cue here. Lemmas reduce inflected forms; stems are often heuristic fragments; they are not interchangeable. A trained spaCy pipeline can add POS, dependency and entity predictions, while Matcher can implement explicit token rules. NLTK corpus/tagger examples may need separately licensed/downloaded resources; this regex tokenizer does not.

**Check:** If text normalization changes character length, original offsets no longer necessarily select the same span. Preserve an alignment or make predictions against the original.

Sources: [spaCy 101](https://spacy.io/usage/spacy-101), [NLTK raw text](https://www.nltk.org/book/ch03.html).

## NC02. Word counts

A bag of words counts tokens and forgets their order. Two sentences can share a word without sharing a position.

<!-- example: NC02-count -->
```python
from collections import Counter

print(dict(Counter("not good".split())))
print(dict(Counter("good".split())))
```

**Output**
```text
{'not': 1, 'good': 1}
{'good': 1}
```

The first sentence contains both words once. The second contains only `"good"`. A vectorizer puts those counts into fixed columns. TF-IDF additionally weights terms by a declared inverse-document-frequency rule. Vocabulary and IDF normally have a fitting boundary; here we supply the vocabulary explicitly to demonstrate only transformation.

<!-- example: NC02-main -->
```python
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import issparse

vectorizer = CountVectorizer(vocabulary={"good": 0, "not": 1})
matrix = vectorizer.transform(["not good", "good"])
print(vectorizer.get_feature_names_out().tolist())
print(matrix.toarray().tolist())
print(issparse(matrix), matrix.shape)
fixed_weights = [1., -2.]
print((matrix @ fixed_weights).tolist())
```

**Output**
```text
['good', 'not']
[[1, 1], [1, 0]]
True (2, 2)
[-1.0, 1.0]
```

Columns mean exactly the supplied vocabulary order. The final values are fixed weighted sums, not learned classifier outputs. Densifying is safe for this 2×2 example but can be disastrous with a huge sparse vocabulary. A real text classifier keeps its vectorizer and estimator together so the same columns are used at prediction time.

TF-IDF and Naive Bayes/linear models are strong baseline choices for many tasks, not obsolete by definition. N-gram features can retain local order, and an n-gram language model estimates next-token probabilities from histories with a declared smoothing rule. Lexical retrieval ranks documents for a query rather than assigning their class. None of those jobs is solved simply by importing CountVectorizer. The course's broader N03–N05 theory explains the objectives; the extension register tracks additional concrete recipes.

Sources: [sklearn text features](https://scikit-learn.org/1.7/modules/feature_extraction.html#text-feature-extraction), [NLTK text classification](https://www.nltk.org/book/ch06.html).

### A text CNN reads neighbouring token features

Unlike a bag of words, a one-dimensional convolution can combine nearby sequence positions. PyTorch Conv1d expects batch, channels/features and length, which differs from the common batch, length, embedding layout.

<!-- example: NC02-textcnn -->
```python
import torch
from torch import nn

embeddings = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]])
convolution = nn.Conv1d(2, 1, kernel_size=2, bias=False)
with torch.no_grad():
    convolution.weight.fill_(1.)
with torch.inference_mode():
    features = convolution(embeddings.transpose(1, 2))
print(features.tolist())
print(features.amax(dim=-1).tolist())
```

**Output**
```text
[[[2.0, 3.0]]]
[[3.0]]
```

The fixed filter sums both embedding features over two consecutive positions. The two windows sum to 2 and 3; max pooling keeps the stronger response. A learned classifier could use several such features, but the example has no trained parameters or quality claim. Padding/sequence lengths need careful handling before pooling so artificial positions do not win. Source: [PyTorch Conv1d](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Conv1d.html).

### Calculate an IDF weight from fixed counts

<!-- example: NC02-tfidf -->
```python
import numpy as np

document_count = 3
fixed_document_frequencies = np.array([2., 1.])
idf = np.log((1 + document_count) / (1 + fixed_document_frequencies)) + 1
term_counts = np.array([2., 1.])
print(np.round(idf, 3).tolist())
print(np.round(term_counts * idf, 3).tolist())
```

**Output**
```text
[1.288, 1.693]
[2.575, 1.693]
```

The rarer term receives the larger inverse-document-frequency weight. This matches the stated smoothed IDF formula, before optional vector normalization. The counts are supplied fitting-corpus statistics; they must not be recomputed from each held-out batch. Libraries differ in IDF smoothing/TF/normalization conventions, so report those choices.

### Smooth a small n-gram probability

<!-- example: NC02-ngram -->
```python
from collections import Counter

tokens = ["we", "agree", "we", "disagree"]
bigrams = Counter(zip(tokens[:-1], tokens[1:]))
following_we = sum(count for (first, second), count in bigrams.items() if first == "we")
vocabulary_size = 3
probability = (bigrams[("we", "agree")] + 1) / (following_we + vocabulary_size)
print(f"P(agree|we), add-one={probability:.1f}")
```

**Output**
```text
P(agree|we), add-one=0.4
```

Two observed continuations follow “we.” Add-one smoothing adds one pseudo-count for each of three vocabulary items, giving `(1+1)/(2+3)`. It avoids zero probabilities for this defined vocabulary, but it is not automatically a strong language model. Source: [NLTK language models](https://www.nltk.org/api/nltk.lm.html).

## NC03. Embeddings

A **word embedding** maps an item to a vector. A static word vector gives the same vector to a word regardless of context; a contextual encoder can produce different representations for different uses. A sentence embedding compresses a longer input into a fixed-size representation for a specified purpose.

<!-- example: NC03-main -->
```python
import numpy as np

fixed_vectors = {"cat": np.array([1., 0.]), "dog": np.array([0.8, 0.2])}
first, second = fixed_vectors["cat"], fixed_vectors["dog"]
cosine = first @ second / (np.linalg.norm(first) * np.linalg.norm(second))
print(f"cosine={cosine:.3f}")
print(fixed_vectors.get("unknown") is None)
tokens = np.array([[1., 0.], [0., 1.], [99., 99.]])
mask = np.array([1., 1., 0.])
print(((tokens * mask[:, None]).sum(axis=0) / mask.sum()).tolist())
```

**Output**
```text
cosine=0.970
True
[0.5, 0.5]
```

These vectors are hand-set, not measured semantic representations. Cosine compares direction; it does not prove that two sentences make the same claim. Masked pooling excludes the padding vector `[99,99]`. An unmasked mean would contaminate the representation, and an all-padding input needs an explicit policy to avoid division by zero.

### Gensim and Sentence Transformers

Gensim is the library people use for static word vectors such as Word2Vec and fastText. A word gets one vector from a trained table, including, for fastText, a vector built from character pieces. The call is a lookup. The vector is only as relevant as the text the table was trained on, and Gensim does not turn that vector into a true paraphrase.

Sentence Transformers is the library for a fixed-size vector of a whole sentence, a query, or a document, plus rerankers that score a pair. People use it because retrieval code can encode the documents once and compare vectors afterward. Encoding text uses the model’s `encode` method, its prompts, and its training objective. The `cos_sim` utility later in this lesson only compares vectors you already have. Averaging the vectors of a different encoder is a different representation. The library will compute the vector or the cosine you ask for. It does not certify that a high score means the same claim.

The Gensim table below is filled by hand. `add_vectors` stores the rows. It does not read a corpus and it does not call a training method. A missing word stays missing: this is a static table, not a fastText character model. The sentence encoder below is also filled by hand. `StaticEmbedding` averages the token rows for a string, and `encode` is the call a retrieval script would make. With one known word per string, the sentence vector is that word’s row. These numbers were written into the objects. They are not a downloaded checkpoint and they are not a measure of meaning.

<!-- example: NC03-gensim -->
```python
from gensim.models import KeyedVectors

vectors = KeyedVectors(vector_size=2)
vectors.add_vectors(["cat", "dog"], [[1.0, 0.0], [0.8, 0.2]])
print([round(value, 1) for value in vectors["cat"].tolist()])
print(f"{float(vectors.similarity('cat', 'dog')):.3f}")
print("unknown" in vectors)
```

**Output**
```text
[1.0, 0.0]
0.970
False
```

`cat` is the row `[1.0, 0.0]`. Its cosine with `dog` is 0.970, the same direction comparison as the hand-written vectors above. `"unknown"` is absent, so a lookup would fail until the program chooses an unknown-token policy. Gensim stored the table. It did not decide that the two animals are related.

<!-- example: NC03-encode -->
```python
import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding

vocabulary = {"[UNK]": 0, "hello": 1, "world": 2}
backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
weights = torch.tensor([[0., 0.], [1., 0.], [0., 1.]])
embedding = StaticEmbedding(tokenizer, embedding_weights=weights)
encoder = SentenceTransformer(modules=[embedding], device="cpu")
encoded = encoder.encode(["hello", "world"], convert_to_numpy=True)
print(np.round(encoded, 3).tolist())
```

**Output**
```text
[[1.0, 0.0], [0.0, 1.0]]
```

`"hello"` is row 1 of the weight table, `[1.0, 0.0]`. `"world"` is row 2, `[0.0, 1.0]`. `encode` tokenizes each string and asks the module for one vector. `device="cpu"` keeps the result on the same device as these checked numbers. A downloaded sentence encoder would replace this table with its own weights, prompts, and pooling rule. The call would still be `encode`. The vector would mean whatever that checkpoint was trained to mean.

Selecting an encoder requires task, language, and license checks, not just the largest embedding dimension.

Sources: [Gensim keyed vectors](https://radimrehurek.com/gensim/models/keyedvectors.html), [Sentence Transformers](https://sbert.net/docs/quickstart.html).

### A contextual encoder returns a vector at each position

The numbers passed as `input_ids` are token IDs. In this encoder, `attention_mask` uses 1 for a real token and 0 for a padding position that attention should not read as a key. That is the padding-attention convention of the call below. It does not by itself make the output vector at a padded query position zero, and it does not by itself keep those positions out of a loss or a pooled sentence vector. A pooling mask or a label mask is a separate list, used when the average or the loss should skip a position. The two tensors can have the same shape and still mean different things.

<!-- example: NC03-inputs -->
```python
import torch

input_ids = torch.tensor([[2, 3, 4]])
attention_mask = torch.tensor([[1, 1, 0]])
print(tuple(input_ids.shape), tuple(attention_mask.shape))
print(int((attention_mask == 1).sum()))
```

**Output**
```text
(1, 3) (1, 3)
2
```

Both tensors have shape `(1, 3)`: one example and three positions. Only two positions are marked real. The encoder call below uses a mask of all ones, so every position is real. The model returns one vector per position. It does not return a class label.

### Why Transformers

Transformers, the Hugging Face library, is how this course loads and runs neural language models. It is widely used because a BERT or GPT-style network, its configuration, and its tokenizer share one loading path, and the result is a PyTorch module. `BertConfig` sets the sizes. `BertModel(configuration)` builds a module with initial weights. The forward call accepts `input_ids` and `attention_mask` and returns a vector at each position. The library runs that architecture. A configuration creates the shapes. It does not supply the language ability of a trained checkpoint, and a printed shape is not a measure of meaning.

<!-- example: NC03-contextual -->
```python
import torch
from transformers import BertConfig, BertModel

configuration = BertConfig(vocab_size=8, hidden_size=8, intermediate_size=16,
    num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8)
encoder = BertModel(configuration)
encoder.eval()
with torch.inference_mode():
    result = encoder(input_ids=torch.tensor([[2, 3, 4]]),
                     attention_mask=torch.tensor([[1, 1, 1]]))
print(tuple(result.last_hidden_state.shape))
```

**Output**
```text
(1, 3, 8)
```

This locally initialized BERT demonstrates the interface, not a pretrained encoder's competence. One input produces three contextual token vectors of width eight. A sequence classifier would add/use an appropriate prediction head; a sentence encoder additionally needs its intended pooling/training contract. Never infer semantic quality from shape alone. Source: [Transformers BERT](https://huggingface.co/docs/transformers/v4.57.1/model_doc/bert).

### Use a sentence-embedding utility without pretending to have an encoder

<!-- example: NC03-similarity-api -->
```python
import torch
from sentence_transformers.util import cos_sim

fixed_query_vectors = torch.tensor([[1., 0.]])
fixed_document_vectors = torch.tensor([[1., 0.], [0., 1.]])
print(cos_sim(fixed_query_vectors, fixed_document_vectors).tolist())
```

**Output**
```text
[[1.0, 0.0]]
```

This exercises the library's cosine utility on supplied vectors. It does not encode text. A reviewed SentenceTransformer checkpoint would supply query/document embeddings through its encoding interface; that model's dimensions, prompts and training objective matter. Keeping those roles separate prevents a similarity utility from being mistaken for a semantic model.

## NC04. Token IDs and masks

A tokenizer maps text to discrete IDs. A dictionary is enough to see that mapping. `[PAD]` is a real vocabulary entry used to make a shorter row as long as a longer one.

<!-- example: NC04-ids -->
```python
vocabulary = {"[UNK]": 0, "[PAD]": 1, "hello": 2, "world": 3}
print([vocabulary[word] for word in ["hello", "world"]])
print([vocabulary["hello"], vocabulary["[PAD]"]])
```

**Output**
```text
[2, 3]
[2, 1]
```

`"hello world"` becomes `[2, 3]`. A padded second position can store 1. That ID is the pad token, not the word `"hello"`.

### Why Tokenizers

Tokenizers is the Hugging Face library that does the text-to-ID step on its own, fast, and with offsets back to the original string. Transformers wraps it when a model needs a tokenizer, which is why the next example builds a `Tokenizer` and then a `PreTrainedTokenizerFast`. People use the pair because the same vocabulary, unknown token, and padding rule should apply in training and in inference. `tokenizer(["hello world", "hello"], padding=True)` is the call: a list of strings in, ID rows and a mask out. The library applies the rules you stored in the vocabulary. It does not know which words the task cares about, and padding does not by itself hide a position from the loss.

The library example adds the mask that says which of those IDs are real. Padding makes a batch rectangular. An attention mask identifies real tokens according to the chosen model's convention. The vocabulary, special tokens and segmentation rules are part of the model's interface.

<!-- example: NC04-main -->
```python
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

backend = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "hello": 2, "world": 3}, unk_token="[UNK]"))
backend.pre_tokenizer = Whitespace()
tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")
batch = tokenizer(["hello world", "hello"], padding=True, return_tensors="np")
print(batch["input_ids"].tolist())
print(batch["attention_mask"].tolist())
print(tokenizer("hello reader")["input_ids"])
```

**Output**
```text
[[2, 3], [2, 1]]
[[1, 1], [1, 0]]
[2, 0]
```

This miniature tokenizer is constructed locally; nothing is downloaded or trained. Its word-level unknown token is deliberately simple, not representative of every byte/subword tokenizer. Padding ID 1 is a real vocabulary ID, but the second row's mask excludes that position. These are not model-generated predictions.

`AutoTokenizer` normally loads the tokenizer specified by a reviewed pretrained asset. Use its exact revision and model-specific special tokens. Truncation can remove the evidence needed for a task; a successful tensor shape does not show that all important text survived. Fast-tokenizer offset mappings and word IDs help align source spans/labels with subwords. Attention masks and training-loss masks serve different purposes and must not be conflated.

Sources: [Tokenizers components](https://huggingface.co/docs/tokenizers/components), [Transformers padding/truncation, versioned source](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/pad_truncation.md).

## NC05. Aligning labels

Sequence classification returns a label for a whole input. Token classification returns labels along a sequence. One annotated word may become several model pieces, so a policy has to say which pieces receive the word's label.

<!-- example: NC05-pieces -->
```python
word = "good"
pieces = ["go", "od"]
print(word, pieces, len(pieces))
```

**Output**
```text
good ['go', 'od'] 2
```

The word is one label. The model sees two pieces. The next loop decides that the first piece keeps the label and the later piece is ignored.

<!-- example: NC05-main -->
```python
word_ids = [None, 0, 1, 1, None]
word_labels = ["O", "B-ORG"]
label_to_id = {"O": 0, "B-ORG": 1}
aligned = []
previous_word = None
for word_id in word_ids:
    if word_id is None or word_id == previous_word:
        aligned.append(-100)
    else:
        aligned.append(label_to_id[word_labels[word_id]])
    previous_word = word_id
print(aligned)
```

**Output**
```text
[-100, 0, 1, -100, -100]
```

`None` marks special tokens; word 1 spans two pieces. This **first-piece-only** policy scores the first piece and ignores the second. Other policies label every piece, but must handle BIO continuation labels consistently and change the loss denominator. The value −100 is a common loss ignore index, not an ordinary entity label.

Structured models such as HMMs/CRFs also constrain sequences: Viterbi finds a highest-scoring path under their local scores/transitions. Parsing predicts syntactic structure; relation/event extraction predicts relationships or triggers/arguments. These are different output contracts, so a token-F1 utility is not automatically valid for entity spans, dependency edges or extracted relations. Current concrete coverage here is alignment; HMM/CRF/parsing model-specific recipes remain explicit extensions rather than hidden prerequisites.

Sources: [Transformers token classification](https://huggingface.co/docs/transformers/v4.57.1/tasks/token_classification), [NLTK information extraction](https://www.nltk.org/book/ch07.html).

### Viterbi keeps the best path, not the best independent tags

The following scores are arbitrary additive teaching scores. Their columns index two labels; transitions score a previous-label/current-label pair. HMM log probabilities or CRF potentials can use this dynamic-programming pattern with their own start/end/normalization rules.

<!-- example: NC05-viterbi -->
```python
import numpy as np

emission_scores = np.array([[2., 0.], [0., 3.]])
transition_scores = np.array([[0., 1.], [1., 0.]])
previous = emission_scores[0]
candidate_scores = previous[:, None] + transition_scores
backpointer = candidate_scores.argmax(axis=0)
current = candidate_scores.max(axis=0) + emission_scores[1]
last_label = int(current.argmax())
path = [int(backpointer[last_label]), last_label]
print(path, float(current[last_label]))
```

**Output**
```text
[0, 1] 6.0
```

The best path receives emission 2, transition 1 and emission 3, totaling 6. The backpointer remembers which earlier label produced the best route into each current label. A longer sequence repeats this operation and traces pointers backward. For a CRF, highest-path decoding differs from calculating the partition function: the latter sums over alternatives rather than taking only maxima. No HMM/CRF trainer is claimed by this calculation.

## NC06. Retrieval

Retrieval starts by comparing one query vector with several document vectors. A dot product multiplies matching positions and adds them.

<!-- example: NC06-dot -->
```python
query = [1.0, 0.0]
documents = [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]]
print([round(query[0] * doc[0] + query[1] * doc[1], 3) for doc in documents])
```

**Output**
```text
[1.0, 0.0, 0.8]
```

The first document points the same way as the query, so the score is 1. The second points elsewhere, so the score is 0. The third shares its first coordinate, so the score is 0.8. Cosine divides by the lengths, which changes a score when a vector is longer. A **bi-encoder** represents the query and each document separately, so the document vectors can be stored and reused. A **cross-encoder** reads a query and a document together, which costs more per pair. Retrieval quality and answer quality are related and still different measurements.

<!-- example: NC06-main -->
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

fixed_document_vectors = np.array([[1., 0.], [0., 1.], [0.8, 0.2]])
fixed_query_vector = np.array([[1., 0.]])
scores = cosine_similarity(fixed_query_vector, fixed_document_vectors)[0]
order = scores.argsort()[::-1]
print(order.tolist())
print(np.round(scores[order], 3).tolist())
```

**Output**
```text
[0, 2, 1]
[1.0, 0.97, 0.0]
```

The vectors are supplied numerical examples, not embeddings of unnamed real documents. Exact cosine search compares every vector and ranks results.

### Why FAISS

FAISS is a library for searching a large set of vectors quickly. People use it after the vectors already exist, because comparing a query with every stored vector becomes the slow step. An index answers which stored vectors are close. It does not decide who may read a document, and it is not a complete vector database. Approximate search trades some of the neighbors an exact scan would return for speed.

`IndexFlatIP` is the exact index: it stores every vector and scores a query with an inner product. The vectors below are the same three rows as the cosine example, normalized so the inner product matches cosine. No model is loaded. The index reports positions into the stored list.

<!-- example: NC06-faiss -->
```python
import numpy as np
import faiss

documents = np.array([[1., 0.], [0., 1.], [0.8, 0.2]], dtype="float32")
query = np.array([[1., 0.]], dtype="float32")
faiss.normalize_L2(documents)
faiss.normalize_L2(query)
index = faiss.IndexFlatIP(2)
index.add(documents)
scores, identifiers = index.search(query, 3)
print(identifiers[0].tolist())
print(" ".join(f"{float(score):.3f}" for score in scores[0]))
print(index.ntotal)
```

**Output**
```text
[0, 2, 1]
1.000 0.970 0.000
3
```

The nearest stored row is position 0, then position 2, then position 1. Those are the same ranks as the exact cosine search above, and the scores round to the same three numbers. `ntotal` is 3 because three rows were added. An approximate index could return a different order. This flat index does not. Neither index knows which documents the reader is allowed to open.

For a reviewed Sentence Transformers model, query/document encoding may require different prompts or methods; follow the model's intended use. Rerank only a bounded candidate set, preserve document IDs/source offsets and evaluate relevance under the actual candidate corpus. Long documents require chunking/aggregation: retrieving a chunk is useful only if it contains sufficient evidence. Precision@k, recall@k, MRR and nDCG answer different ranking questions.

Sources: [Sentence Transformers](https://sbert.net/docs/quickstart.html), [FAISS introduction](https://github.com/facebookresearch/faiss/wiki/Getting-started).

## NC07. Text metrics

Translation/summarization generate text; span QA selects text positions; entity extraction returns structured spans. A metric must match the unit and normalization policy. Token overlap does not establish that a summary is faithful to its source.

<!-- example: NC07-main -->
```python
from collections import Counter

reference = "the red car".split()
answer = "red car".split()
overlap = sum((Counter(reference) & Counter(answer)).values())
precision = overlap / len(answer)
recall = overlap / len(reference)
f1 = 2 * precision * recall / (precision + recall)
print(f"overlap precision={precision:.3f}, recall={recall:.3f}, F1={f1:.3f}")
context = "A red car arrived."
start, end = 2, 9
print(context[start:end])
```

**Output**
```text
overlap precision=1.000, recall=0.667, F1=0.800
red car
```

This is a transparent multiset token-overlap example, not a full SQuAD implementation: benchmark normalization can remove articles/punctuation and change the result. Character spans use a half-open end index here. QA models may instead predict token start/end positions that need mapping back to source text, plus an explicit no-answer policy.

BLEU uses clipped n-gram overlap and a length penalty under a declared tokenization. ROUGE variants measure different overlaps. The overlap printed above is a hand-written multiset count. It is not BLEU. The next example calls SacreBLEU and the ROUGE-1 scorer on the same two strings. ROUGE-1 with stemming off agrees with the hand count on this short sentence: precision 1, recall 0.667, F1 0.800. SacreBLEU’s sentence score is 60.653, because the shorter answer pays a brevity penalty and the higher-order n-grams are not a copy of that word F1. Report the library and the setting. Do not rename the hand count to either score.

<!-- example: NC07-published -->
```python
import sacrebleu
from rouge_score import rouge_scorer

reference = "the red car"
answer = "red car"
bleu = sacrebleu.sentence_bleu(answer, [reference])
rouge = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False).score(reference, answer)["rouge1"]
print(f"{bleu.score:.3f}")
print(f"{rouge.precision:.3f} {rouge.recall:.3f} {rouge.fmeasure:.3f}")
```

**Output**
```text
60.653
1.000 0.667 0.800
```

`sentence_bleu` takes the answer first and a list of references second. The score 60.653 is that library’s sentence BLEU for this pair under its default tokenizer. The second line is ROUGE-1 precision, recall, and F1. Matching the hand count is a fact about this one pair and this one ROUGE setting. A different tokenizer, stemmer, or smoothing choice changes the number. Neither score says the answer is faithful to a source document.

Encoder–decoder generation additionally has source inputs, decoder history and a decoding strategy such as beam search. A good-looking fluent sentence can still omit or contradict the source. Use examples of factual errors and human review where the task demands it.

Sources: [SacreBLEU](https://github.com/mjpost/sacrebleu), [Transformers QA](https://huggingface.co/docs/transformers/v4.57.1/tasks/question_answering), [SLP](https://web.stanford.edu/~jurafsky/slp3/).

### Beam search keeps several candidate histories

<!-- example: NC07-beam -->
```python
import math

beam = [("A", math.log(0.6)), ("B", math.log(0.4))]
fixed_next_probabilities = {"A": {"x": 0.5, "y": 0.5}, "B": {"x": 0.9, "y": 0.1}}
candidates = [(prefix + token, score + math.log(probability))
              for prefix, score in beam
              for token, probability in fixed_next_probabilities[prefix].items()]
kept = sorted(candidates, key=lambda item: (-item[1], item[0]))[:2]
print([(text, round(math.exp(score), 2)) for text, score in kept])
```

**Output**
```text
[('Bx', 0.36), ('Ax', 0.3)]
```

The lower-probability first token B leads to the strongest complete two-token path: `.4×.9=.36`. Greedy choice of A would miss it. This is one fixed beam-expansion step with width two, no length penalty and an explicit tie-break. Full generation also handles EOS, finished beams and stopping rules. A larger beam is not guaranteed to improve human usefulness or faithfulness.

## NC08. A fixed prediction report

An NLP evaluation needs examples, label definitions, predictions and grouping information. The table below is an **authored fixture**, not the output of a newly fitted sparse/encoder comparison. Its purpose is to show how an overall score can conceal a domain difference.

<!-- example: NC08-main -->
```python
from sklearn.metrics import accuracy_score, f1_score

rows = [
    {"domain": "reviews", "gold": 1, "prediction": 1},
    {"domain": "reviews", "gold": 0, "prediction": 0},
    {"domain": "dialogue", "gold": 1, "prediction": 0},
    {"domain": "dialogue", "gold": 0, "prediction": 0},
]
print(f"overall accuracy={accuracy_score([r['gold'] for r in rows], [r['prediction'] for r in rows]):.2f}")
for domain in ["reviews", "dialogue"]:
    subset = [row for row in rows if row["domain"] == domain]
    score = f1_score([r["gold"] for r in subset], [r["prediction"] for r in subset], zero_division=0)
    print(domain, f"positive F1={score:.2f}")
```

**Output**
```text
overall accuracy=0.75
reviews positive F1=1.00
dialogue positive F1=0.00
```

One missed positive produces a large difference in this deliberately tiny example. It is not evidence that a particular architecture cannot handle dialogue. Real comparisons need identical data/splits, class mappings, selection rules, meaningful sample sizes and individual error inspection.

Cross-language/domain transfer also depends on annotation definitions: the same word “argument,” “claim” or “entity” can describe different targets. A role/guideline signal might align with one corpus and fail on another without a simple vocabulary explanation. A complete reporting workflow preserves the source record, tokenization limits and declared prediction policy. The current NLP section has no trainer to launch; ML/DL lab links should not imply text support they lack.

Sources: [sklearn classification metrics](https://scikit-learn.org/1.7/modules/model_evaluation.html#classification-metrics), [SLP evaluation/task foundations](https://web.stanford.edu/~jurafsky/slp3/).
