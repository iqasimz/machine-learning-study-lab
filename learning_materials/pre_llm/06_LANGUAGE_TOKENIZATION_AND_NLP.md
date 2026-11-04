# P6. Language, tokens and model inputs

Natural language processing, or NLP, builds systems that analyze or produce human language. A model first needs a numerical input: tokenization splits text into chosen units, and a vocabulary assigns each unit an integer ID. These units are not necessarily words, and the IDs do not represent meaning by their numerical size.

We split the words 'low' and 'lower' into characters, then apply two supplied merge rules. The controls show how pieces, sequence length and IDs change. These rules are a small demonstration, not a tokenizer trained on a real corpus.

The shared foundations explained vectors and prediction targets. This lesson explains how text becomes the discrete inputs for those calculations; the following NLP lessons define what a system should predict from them.

## 1. Language, context and task definitions

**Key terms.** **Syntax** is grammatical arrangement; **semantics** is meaning; **pragmatics** includes how context affects interpretation; an **annotation guideline** defines a task's labels.

**Process.** Raw language enters a task definition, then becomes training examples. Clarify the intended label and available context before deciding what the model must predict.

Language contains ambiguity, references, implied meaning and context-dependent definitions. **Syntax** concerns structure; **semantics** meaning; **pragmatics** meaning in use. “That is just great” can be praise or sarcasm. “It reduced risk” needs a referent and comparison. Counting tokens alone does not resolve either issue.

Core NLP tasks include classification, sequence labelling, span extraction, parsing, coreference, retrieval, translation, summarization, question answering and dialogue. A classification target is not the same object as a generated explanation. An ontology defines permitted categories and relationships; a taxonomy organises categories. Neither becomes true merely because it is encoded as JSON.

## 2. Sequence-model families

**Key terms.** An **n-gram** uses a short local history; a **recurrent model** updates sequence state; **self-attention** mixes representations across permitted positions.

**Process.** Follow how each family represents preceding text and predicts what follows. This historical comparison explains design trade-offs, not an assumption that the newest family wins every task.

Bag-of-words counts ignore most ordering but work surprisingly well for some tasks. N-gram language models estimate a token from a limited preceding window; smoothing assigns probability to unseen events. An RNN repeatedly updates a hidden state; LSTMs add gates to mitigate some long-range training difficulties. Sequence-to-sequence models map an input sequence to an output; attention lets an output step access different input states. Transformers replace recurrence with token interactions plus positional information. LLMs scale and adapt such sequence modelling; they do not eliminate the earlier task definitions.

## 3. Bytes, code points and model tokens

**Key terms.** A **byte** is a storage unit; a **character** is a text symbol under an encoding; a **token** is a tokenizer-defined unit; an **ID** names a vocabulary entry.

**Process.** Text is encoded and split into tokens, then mapped to IDs. The model operates on those IDs; decoding maps them back. Count tokens rather than words when setting model context limits.

Unicode represents characters; UTF-8 encodes them as bytes. A visible symbol can span multiple code points and bytes. A tokenizer maps text to IDs from a vocabulary; its inverse maps IDs back to text under that tokenizer's rules. Tokens may be characters, subword pieces, byte sequences, whole words or special markers.

A vocabulary trades sequence length against embedding/output-layer size. Smaller vocabularies can increase tokens per text; larger vocabularies need more parameters and can waste capacity on rare pieces. Multilingual text may require different token counts for comparable content, affecting cost and context limits.

## 4. Learning BPE rules and comparing tokenizer families

**Key terms.** **Byte-pair encoding (BPE)** builds units by merging frequent adjacent symbols; a **vocabulary** is the permitted set of token units.

**Process.** Start with small units and pair counts, merge a chosen frequent pair, update counts and repeat. The learned merge rules later split new text without re-learning from evaluation data.

Start with symbol sequences and repeatedly merge a frequent adjacent pair. Toy corpus: `low`, `low`, `lower`. If the initial symbols are characters, pair `(l,o)` occurs three times, and `(o,w)` also three times. Tie-breaking must be specified. Merge `(l,o)` into `lo`, then `(lo,w)` into `low`. The word `lower` can now be represented as `low,e,r`.

This simplified example omits boundary markers, byte handling and preprocessing. Real BPE pipelines specify those. **WordPiece** uses a related subword construction with different training choices; **Unigram** tokenization starts with a candidate vocabulary and removes pieces according to a probabilistic objective. SentencePiece is a tokenizer toolkit, not a synonym for one algorithm.

Tokenizer training must not accidentally inspect held-out evaluation text if the protocol forbids that. Normalization can erase clinically or legally important distinctions. Byte fallback avoids some unknown-token problems but does not guarantee equal efficiency across languages.

## 5. Special tokens and chat templates

**Key terms.** A **special token** marks structure rather than ordinary prose; a **chat template** serializes roles and message boundaries.

**Process.** Messages enter the model's expected template, become tokens and form the training or generation prefix. Incorrect role/end markers can change behaviour before the model sees any new content.

Models may require beginning/end markers, role delimiters and turn terminators. A chat template converts messages into the exact token sequence expected by training. Two APIs accepting the same message JSON need not construct identical token inputs.

```text
messages → template → text/special tokens → token IDs → model
```

Using a wrong template can make a capable model appear poor. During SFT, labels often apply only to assistant spans; role markers and stopping behaviour must agree with inference. Do not add special tokens twice or assume every model uses the same end marker.

## 6. Context, truncation and leakage

**Key terms.** **Truncation** removes tokens beyond a length limit; **leakage** exposes information that should be unavailable; a **loss mask** selects scored targets.

**Process.** Tokenize first, check boundaries and allocate context, then apply attention and scoring rules. Verify which instruction, evidence and answer tokens survive before interpreting results.

**Calculation.** If a model permits 4,096 total tokens and 3,600 are already used by the serialized prompt, at most 496 remain for output under that shared limit. Count template/special tokens too; a tokenizer change can alter the split.

The context window limits the tokens available in one forward computation or supported serving configuration. It is not permanent memory. Truncation may remove the actual question, relevant evidence or safety instruction. Log token lengths and the truncation rule, not just a character count.

Tokenization creates other pitfalls: numbers can fragment oddly; a spelling change can alter several tokens; source code indentation matters; invisible characters can affect output. Token length is not a neutral difficulty measure. Separate token-budget constraints from the scientific question you are evaluating.

### Tokenization becomes an experimental variable

To compare two tokenizers fairly, begin with the same raw held-out text.

1. Preserve source bytes and normalization rules. Unicode normalization can change byte counts and offsets even when text looks the same.
2. Measure the token count and truncation rate by language/domain. At a fixed 1,024-token budget, different tokenizers may expose different amounts of actual content.
3. Account for vocabulary cost. With tied input/output weights, vocabulary size `V` and feature width `d` contribute approximately `Vd` parameters; an untied output head adds another such matrix.
4. Compare prediction cost using explicitly defined units. Per-token perplexity is not directly comparable when the units change. Total negative log likelihood divided by original byte count, then by `ln 2`, gives bits per byte under the chosen boundary protocol.
5. Inspect byte/span alignment for extraction tasks. Character offsets, UTF-8 byte offsets and token positions are not interchangeable; preserve the mapping rather than guessing from decoded pieces.

An end token can be both a genuine response target and the token ID used for padding elsewhere. Mask by position and role boundaries, not by deleting every occurrence of that ID. Original references: [BPE for NLP](https://aclanthology.org/P16-1162/) and [SentencePiece](https://aclanthology.org/D18-2012/); current implementations also differ in their normalization and pre-tokenization rules.

## Practice and checks

1. Why is 1,000 words not necessarily 1,000 tokens? **Answer:** subwords, punctuation, whitespace and language-specific segmentation.
2. Could labels change while the sentence stays fixed? **Yes:** the task guideline or intended context may change the target definition.
3. List what to save for reproducibility: tokenizer files/revision, normalization, vocabulary/special tokens, template, truncation and packing rules.

Visual reinforcement: [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) → inspect input tokens before generating. Try punctuation and an unfamiliar name. Deeper original teaching: [Hugging Face LLM course](https://huggingface.co/learn/llm-course/en/chapter1/1), tokenizer chapters. The selected toy model is instructional, not a statement about every current model.
