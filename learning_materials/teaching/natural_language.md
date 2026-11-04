# Natural language processing

## P6. Language, tokens and model inputs

Text is not yet a model input. A tokenizer converts it into pieces and maps those pieces to vocabulary IDs. A piece may be a word, part of a word, punctuation, a byte, or a special symbol. The integer ID is a lookup key, not a measurement of meaning: token 200 is not twice as meaningful as token 100.

Subword tokenizers balance vocabulary size against sequence length. Splitting a rare word into familiar pieces avoids needing a dedicated whole-word entry, but produces more positions. Byte-pair encoding learns a sequence of merges from training text. Those merges are fixed when encoding evaluation text; tokenizing a new sentence is not refitting the vocabulary.

Keep the tokenizer compatible with the embedding table and saved model. Normalisation and special tokens are part of that contract. If input IDs are `[a,b,c]` for next-token training, targets come from the shifted sequence `[b,c,d]` when the following token d exists. The last input position can therefore have a target. Padding occupies tensor space but should not become an ordinary scored target.

These choices affect context length, loss denominators, offsets, and cross-language efficiency. Compare token-based scores only after checking what tokens and targets each system uses.

Sources: [SLP: words and tokens](https://web.stanford.edu/~jurafsky/slp3/2.pdf), [Hugging Face: tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary).

## P7. Embeddings, similarity and retrieval

An embedding is a learned numerical representation. Retrieval compares a query representation with stored document representations to find candidates. The two representations must be compatible: equal dimensionality alone does not make vectors from unrelated models comparable.

Cosine similarity divides a dot product by both vector lengths. For $q=[1,0]$ and $d=[1,1]$, the dot product is 1 and cosine is $1/\sqrt2\approx0.707$. Multiplying d by 10 leaves cosine unchanged but changes its raw dot product. The score convention therefore matters. A zero vector needs explicit handling because its cosine denominator is zero.

An index makes search efficient; an approximate index trades exact nearest-neighbour recovery against speed and memory. Top-k means the highest-scoring eligible items under that procedure, not the k most truthful passages. Access filters must restrict what the user can retrieve, and relevance needs evaluation against a task.

Finally, a similar passage may contradict the query or omit its needed evidence. Retrieval quality and answer support are different checks. The RAG lesson builds on this distinction rather than treating an embedding score as factual verification.

Sources: [Sentence-BERT](https://arxiv.org/abs/1908.10084), [SLP: retrieval and RAG](https://web.stanford.edu/~jurafsky/slp3/11.pdf).

## N01. NLP tasks and annotation guidelines

NLP tasks make language measurable by specifying an input, an output unit, and an annotation policy. Sentiment classification assigns a document label; named-entity recognition identifies typed spans; relation extraction connects mentions. A model cannot learn the intended task from the label name alone if annotators use different definitions.

Consider “We should fund the library.” One argument-identification policy may count a standalone claim, while another requires an explicit supporting reason. The sentence has not changed; the rule defining a positive has. Disagreement can reflect ambiguity or different policies rather than a careless annotator.

Measure agreement with the unit and chance model stated. If two annotators agree on 8 of 10 cases, observed agreement is 0.8. A chance-corrected coefficient also depends on their marginal label frequencies, so 80% agreement does not determine kappa by itself. Review disputed examples before interpreting a single reliability number.

Document the guideline, context allowed, span boundaries, adjudication process, and intended population. This prepares meaningful cross-dataset experiments: a transfer failure may be a representation problem, a language shift, or a changed target definition. The distinction should be tested rather than assumed.

Sources: [Artstein and Poesio: inter-coder agreement](https://aclanthology.org/J08-4004/), [SLP: sequence labelling](https://web.stanford.edu/~jurafsky/slp3/18.pdf).

## N02. Text processing, morphology and source offsets

Preprocessing decides what distinctions survive. Lowercasing merges “Apple” and “apple”; removing punctuation can erase a question or an expression of emphasis. A stopword filter can remove “not,” reversing the useful information in “not good.” Transformations are choices about the task, not universally harmless cleaning.

Morphology concerns how word forms are built. Stemming heuristically removes pieces; lemmatisation maps an inflected form to a dictionary form using linguistic information. Neither is mandatory for subword neural models. Sentence segmentation and tokenisation also differ across writing systems and noisy conversational text.

For extraction, preserve a map from processed tokens to the original text. An entity predicted at character interval `[5,10)` must identify the intended substring under the same indexing convention. Unicode normalisation, byte offsets, Python character indices, and browser string indices can disagree, particularly for accented text or emoji.

Keep the original source and test a round trip from selected tokens back to its span. If preprocessing merges or deletes characters, a token index alone cannot reconstruct the answer. These bookkeeping details directly affect whether an otherwise correct prediction is evaluated as correct.

Sources: [SLP: words and tokens](https://web.stanford.edu/~jurafsky/slp3/2.pdf), [spaCy: linguistic features and token alignment](https://spacy.io/usage/linguistic-features).

## N03. Word counts, TF-IDF and lexical retrieval

A bag-of-words representation counts selected vocabulary items while discarding their positions. This makes a document a sparse vector. It is cheap and often useful, but unigram counts cannot distinguish all order changes: “dog bites person” and “person bites dog” have the same counts.

TF–IDF combines within-document frequency with how uncommon a term is across documents. Under the simple convention $idf(t)=\ln(N/df(t))$, a term present in one of ten documents receives $\ln10\approx2.303$; a term present in all ten receives zero. Libraries may use smoothing, an added constant, sublinear term frequency, and vector normalisation. State the convention before reproducing a number.

Fit vocabulary and document frequencies on the training/index corpus, then transform queries and evaluation text using that representation. Cosine compares direction after accounting for length. BM25 additionally incorporates term-frequency saturation and document-length normalisation under its own formula.

Lexical retrieval is strong when exact terminology matters, such as product IDs or unusual names. Dense retrieval can recover paraphrases but may blur exact strings. A hybrid system combines different signals; it should be compared on the same eligible documents and relevance judgements.

Sources: [scikit-learn: text feature extraction](https://scikit-learn.org/1.7/modules/feature_extraction.html#text-feature-extraction), [SLP: information retrieval](https://web.stanford.edu/~jurafsky/slp3/11.pdf).

## N04. Naive Bayes and linear text classifiers

A text classifier needs both a representation and a fitting rule. A unigram count vector can feed naive Bayes, logistic regression, or a linear SVM. Reusing the representation does not make their assumptions or objectives equivalent.

Naive Bayes estimates class-specific feature likelihoods and combines them with class priors. For the single presence feature “prize,” scores 0.4 for spam and 0.05 for non-spam normalise to spam probability $0.4/(0.4+0.05)\approx0.889$. With many words, conditional independence simplifies the likelihood. Logarithms turn products into sums, and smoothing prevents unseen counts from eliminating a class.

Logistic regression directly learns a weighted score used in a conditional probability model. A positive word coefficient increases the class score while other represented features remain fixed. It is not proof that the word causes the label. Bigrams can capture “not good,” but increase the feature space and sparse-data problem.

Compare against a majority baseline and inspect errors by source, topic, and label. A simple lexical classifier that fails after topic words change can reveal a shortcut. Keep vocabulary and weighting fit within the training fold.

Sources: [SLP: naive Bayes](https://web.stanford.edu/~jurafsky/slp3/B.pdf), [scikit-learn: worked text-classification comparison](https://scikit-learn.org/1.7/auto_examples/text/plot_document_classification_20newsgroups.html).

## N05. N-gram language models and smoothing

A language model assigns probabilities to token sequences. The chain rule factors sequence probability into next-token conditionals. An n-gram approximation keeps only a short recent history: a bigram uses one previous token, a trigram uses two. It is a memory assumption, not the full chain rule itself.

If “like cats” appears three times and histories ending in “like” have five observed successors, the unsmoothed estimate is $P(cats\mid like)=3/5$. An unseen permitted continuation gets probability zero, making any sequence containing it have probability zero. Add-one smoothing changes the estimate to $(3+1)/(5+V)$ for a stated vocabulary of size V. It cannot magically assign an ID to an out-of-vocabulary word; define an unknown-token policy first.

Backoff or interpolation borrows information from shorter histories. Kneser–Ney's lower-order continuation probability considers how many distinct **preceding contexts** a candidate word follows. A word frequent only in one fixed phrase should not automatically be the best fallback after an unfamiliar history.

Evaluate held-out average token log loss with the vocabulary and boundaries declared. More context gives more specific counts but fewer observations per history. This trade-off motivates distributed representations and neural models, rather than making an n-gram baseline useless.

Sources: [SLP: n-gram models](https://web.stanford.edu/~jurafsky/slp3/3.pdf), [SLP: Kneser–Ney smoothing](https://web.stanford.edu/~jurafsky/slp3/C.pdf).

## N06. Static word vectors: Word2vec, GloVe and fastText

Static embeddings give each vocabulary item a learned vector reused across occurrences. The distributional idea is that patterns of surrounding words contain useful information about meaning. A dot product or cosine then compares learned representations, not dictionary definitions.

Skip-gram learns to predict nearby words from a centre word. Negative sampling replaces an expensive full-vocabulary objective with a different objective distinguishing observed pairs from sampled noise pairs. GloVe instead fits a model of global co-occurrence statistics. These methods share a representation goal but do not optimise the same loss.

fastText composes word information from character n-grams, helping related forms share parameters and allowing representations for many unseen strings. Character pieces are not a guarantee of semantic understanding: similar spelling can belong to unrelated words.

The main limitation is that a static vector for “bank” does not change between “river bank” and “bank loan.” Contextual token encoders address this by conditioning each occurrence's representation on its sentence. Evaluate representations on the downstream task; an appealing nearest-neighbour list or analogy is not a complete quality measure.

Sources: [SLP: embeddings](https://web.stanford.edu/~jurafsky/slp3/5.pdf), [fastText subword paper](https://arxiv.org/abs/1607.04606).

## N07. Convolutional networks for text

A text CNN applies shared filters to consecutive token vectors. Unlike a unigram bag, it can detect local order; unlike a recurrent encoder, it need not process each token through one sequential state. A width-two filter can respond differently to “not good” and “good film.”

With a sequence length T, embedding width d, and a width-k filter covering all d features, each valid position uses kd weights plus a bias per filter. A nonlinearity transforms the responses. Max-pooling over positions retains the strongest response, roughly asking whether a learned local pattern appeared anywhere.

Pooling loses some location information. A negation far outside a filter's window may be missed, and increasing width changes what counts as local. Several filter widths can cover different phrase lengths, but training must learn which responses are useful for the label.

The worked example shows the products of a fixed filter and token features; it is not a trained sentiment system. Compare a real text CNN with a lexical baseline and a contextual encoder using matching inputs and splits before attributing a gain to locality alone.

Sources: [Kim: CNNs for sentence classification](https://arxiv.org/abs/1408.5882), [D2L: text CNN](https://d2l.ai/chapter_natural-language-processing-applications/sentiment-analysis-cnn.html).

## N08. Recurrent and bidirectional recurrent networks

A recurrent encoder reads one token vector at a time and updates a hidden state. The same weights are reused at every position. In a forward RNN, the state after “not good” can depend on both words, while the state after “not” cannot depend on the next word yet.

For $h_t=\tanh(0.5x_t+0.8h_{t-1})$, start with zero state and inputs 1 then 0. The first state is about 0.462; the next is $\tanh(0.8(0.462))\approx0.354$. Earlier evidence persists through the recurrence but can shrink or be transformed. These are fixed illustrative weights, not a fitted language model.

A bidirectional RNN also reads from the end toward the beginning, then combines states at each position. That can help tag an ambiguous word using both sides of a complete sentence. It is not legal to expose the true future when testing an ordinary causal next-token predictor.

Backpropagation through time differentiates through the unrolled updates. Long chains can cause vanishing or exploding gradients. Gated cells change how state is retained; bidirectionality changes which context is available. They solve different problems.

Sources: [D2L: recurrent networks](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html), [SLP: RNNs and LSTMs](https://web.stanford.edu/~jurafsky/slp3/14.pdf).

## N09. LSTMs and bidirectional LSTMs

An LSTM introduces an additive memory path called the cell state. A forget gate scales the previous cell; an input gate scales a proposed addition. Their sum forms the new cell. An output gate determines how much of its transformed content becomes the exposed hidden state. The gates are learned functions of current input and prior hidden state.

For previous cell 0.8, forget gate 0.9, input gate 0.2, and candidate −0.5, the new cell is $0.9(0.8)+0.2(-0.5)=0.62$. With output gate 0.7, hidden state is $0.7\tanh(0.62)\approx0.386$. Cell and hidden state are therefore related but not identical. Retaining information does not mean copying every input without change.

A BiLSTM runs two LSTMs in opposite directions and usually concatenates their states. Each token representation can then use left and right context. This is useful for complete-sentence tagging, while autoregressive generation requires the permitted causal direction.

Gating helps gradients and memory but does not guarantee unlimited long-range recall. Sequence length, initialisation, training data, and optimisation still matter. Compare gate definitions carefully: a compact diagram must not hide which state is retained and which is newly written.

Sources: [D2L: LSTM](https://d2l.ai/chapter_recurrent-modern/lstm.html), [SLP: recurrent encoders](https://web.stanford.edu/~jurafsky/slp3/14.pdf).

## N10. GRUs and recurrent training

A GRU controls recurrent memory using reset and update gates, without an LSTM's separate cell-state output. The reset gate changes how previous information enters the candidate. The update gate mixes the old state with that candidate. It is a learned, feature-wise mixture, not a threshold applied to the final class probability.

Using the convention $h_t=z_th_{t-1}+(1-z_t)\tilde h_t$, an old state 0.2, candidate 0.8, and update gate 0.75 give $0.75(0.2)+0.25(0.8)=0.35$. A source using the complementary gate convention may write a different-looking formula; identify which coefficient multiplies the old state before comparing implementations.

During batching, mask padding and keep each sequence's state associated with the correct example. Truncated backpropagation detaches the state between windows, limiting how far gradients travel even when numerical state is carried forward. Gradient clipping limits exploding update signals; it does not recover information lost through vanishing gradients.

GRUs can be cheaper than comparable LSTMs, but cell choice alone does not determine task performance. Compare hidden widths, parameter counts, context access, and training conditions rather than treating a smaller gate diagram as proof of superiority.

Sources: [D2L: GRU](https://d2l.ai/chapter_recurrent-modern/gru.html), [Cho et al.: recurrent encoder–decoder](https://arxiv.org/abs/1406.1078).

## N11. Encoder–decoder models, attention and beam search

An encoder represents a source sequence; a decoder produces a target sequence conditional on that representation and its generated prefix. Translation is a clear example: the source sentence is available before the target is complete. Source and target lengths need not match.

Attention lets a decoder position choose a weighted mixture of encoder states instead of relying only on one compressed source vector. In cross-attention, queries come from the decoder and keys/values from the source representation. This is separate from causal self-attention over the target prefix.

Training often supplies the gold previous target tokens, called teacher forcing. Inference supplies previously generated tokens, so early errors can change later inputs. Beam search keeps several high-scoring partial sequences rather than only one greedy choice. Sequence log-probabilities add; length penalties and stopping rules alter which completed hypothesis wins.

A larger beam is a more extensive search under the chosen model and score, not a guarantee of better translation. If the model prefers short or unfaithful outputs, search can expose that preference more strongly. Evaluate generated sequences as well as token loss, and distinguish search changes from changes to trained weights.

Sources: [D2L: sequence-to-sequence learning](https://d2l.ai/chapter_recurrent-modern/seq2seq.html), [D2L: beam search](https://d2l.ai/chapter_recurrent-modern/beam-search.html).

## N12. Sequence labelling with HMMs, CRFs and Viterbi

Sequence labelling assigns a tag to each token, such as part of speech or an entity-boundary label. Independent tag predictions may produce an invalid sequence: an inside-person tag cannot normally begin a new person span under a BIO policy. Modelling transitions makes those relationships explicit.

An HMM models a joint distribution using state transitions and observation emissions. A linear-chain CRF instead scores tag sequences conditional on the observed input and normalises over permitted sequences. Neural emission scores can supply local evidence without changing this distinction.

Viterbi finds the highest-scoring path using dynamic programming. At each position and candidate tag, keep the best prefix score ending there and its backpointer. For two tags and three positions there are eight full paths, but we do not need to enumerate all of them for a long sequence. The recurrence reuses best prefixes. Viterbi's max operation finds one path; a forward log-sum-exp calculation sums probability mass and serves a different purpose.

State whether transitions are learned, constrained, or both. Evaluate exact typed spans as well as token tags when entity boundaries matter. A correct token majority can hide missed entities.

Sources: [SLP: sequence labelling](https://web.stanford.edu/~jurafsky/slp3/18.pdf), [SLP: hidden Markov models](https://web.stanford.edu/~jurafsky/slp3/A.pdf).

## N13. Syntax: dependency and constituency parsing

Parsing represents sentence structure beyond a list of words. A dependency tree links words through grammatical relations, typically with a head for each non-root token. A constituency tree groups words into nested phrases. The representations answer different structural questions and should not be mixed into one graph without defining the conversion.

In “The student read the paper,” a dependency analysis can link “student” as subject and “paper” as object of “read.” A constituency analysis groups “the paper” into a noun phrase. These descriptions help identify who did what, but syntax alone does not establish whether a claim is true or an argument is persuasive.

Unlabelled attachment score checks whether predicted heads match the reference; labelled attachment also requires the relation type. Span-based constituency evaluation compares predicted phrase spans under a declared policy. Tokenisation differences can invalidate otherwise plausible comparisons.

Parsers may fail on conversational ellipsis, unusual punctuation, and unfamiliar domains. If a downstream system consumes a predicted parse, those errors propagate. Compare against direct text representations before assuming an explicit tree is useful merely because it looks interpretable.

Sources: [SLP: dependency parsing](https://web.stanford.edu/~jurafsky/slp3/20.pdf), [SLP: constituency parsing](https://web.stanford.edu/~jurafsky/slp3/19.pdf).

## N14. Contextual token encoders: ELMo and BERT

A contextual encoder gives a word occurrence a representation conditioned on surrounding text. “Bank” can therefore receive different vectors in a financial sentence and a river sentence. The output is usually one vector per input position, not automatically one useful vector for the whole sentence.

ELMo derives contextual representations from a bidirectional language-model architecture. BERT uses a bidirectional Transformer encoder trained partly by predicting selected masked tokens. The original BERT also used a next-sentence objective. This is not ordinary causal prediction of every next token: its attention visibility and target construction differ.

For a task such as tagging, attach a per-token classifier. For sentence classification, use an appropriate pooled representation and train the head; further fine-tuning can change the encoder. Merely averaging off-the-shelf token vectors is not guaranteed to produce strong sentence similarity embeddings.

Mask padded tokens, align subword pieces with word-level labels, and define how multiple pieces contribute to evaluation. A pretrained model can transfer useful language regularities while still carrying domain bias or missing specialised vocabulary. Contextualisation changes available information; it does not eliminate the need for task-specific evidence.

Sources: [Peters et al.: ELMo](https://arxiv.org/abs/1802.05365), [Devlin et al.: BERT](https://arxiv.org/abs/1810.04805), [SLP: masked language models](https://web.stanford.edu/~jurafsky/slp3/9.pdf).

## N15. Sentence encoders and query–document interaction

Retrieval needs to compare many candidates. A bi-encoder produces query and document vectors separately, so document vectors can be stored before the query arrives. A cross-encoder reads a query–document pair jointly and scores their interaction, usually at greater cost per candidate. The difference is where interaction occurs, not simply which model name is used.

The two towers of a bi-encoder may share weights, as in many sentence-embedding systems, or use different query and passage encoders. Their training objective must make the resulting spaces compatible. A contrastive objective rewards an intended pair relative to negatives; which negatives are sampled changes what distinctions the encoder learns.

For vectors `[1,0]` and `[1,1]`, cosine is about 0.707. That score has meaning only relative to the trained representation and candidate set; it is not a universal semantic percentage. A practical pipeline may retrieve many documents cheaply, then rerank a smaller set with a cross-encoder.

Measure candidate recall before final ranking quality. If the needed passage never reaches the reranker, better pair scoring cannot recover it. Also evaluate exact-name cases and paraphrases rather than assuming dense retrieval replaces lexical matching everywhere.

Sources: [Sentence-BERT](https://arxiv.org/abs/1908.10084), [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906).

## N16. Text classification and paired-input inference

A document classifier maps one text to a label. A paired-input classifier conditions on two texts, for example a premise and a hypothesis. In natural-language inference, the label describes their relationship: entailment, contradiction, or insufficient support under the annotation policy. It is not simply the sentiment of either sentence.

For premise “All tickets were sold” and hypothesis “No tickets remain available,” the relationship may be supported. Change the premise to “Some tickets were sold,” and that conclusion no longer follows. A hypothesis-only model cannot see this difference. If it nevertheless scores well, dataset-specific wording may be predictive of labels.

Compare full-input, premise-only, and hypothesis-only conditions when investigating such shortcuts. Keep the same evaluation examples and define what each condition is permitted to read. A score difference can then identify reliance on information, without proving every successful prediction performs valid reasoning.

Calibration and thresholding depend on the task's class structure. Use error inspection for negation, quantifiers, entity swaps, and domain changes. A high aggregate score under one corpus's policy need not transfer to another definition of inference.

Sources: [Gururangan et al.: annotation artefacts in NLI](https://aclanthology.org/N18-2017/), [Poliak et al.: hypothesis-only baselines](https://aclanthology.org/S18-2023/).

## N17. Information extraction: entities, relations and events

Information extraction turns text into structured records. Entity recognition finds typed mentions; relation extraction links mentions; event extraction identifies an event and its participants. Each extra stage introduces a question about what the source actually states.

In “Mira joined Acme in June,” an entity system may find Mira as a person and Acme as an organisation. A relation could represent employment; an event record could attach a person, organisation, and time to a joining event. The text does not specify Mira's later job title or whether she still works there. A plausible completion would be an unsupported addition.

Choose whether records use mention spans, canonical entities, or both. Two names can refer to one entity, and one name can refer to different entities. Relations may be directed; swapping participants can change the meaning. Define the schema and missing-value policy before evaluating generated JSON.

Measure exact spans, types, participant roles, and source support separately where useful. A schema-valid record can still be wrong. Pipeline errors accumulate, so compare end-to-end records as well as individual stage scores. Preserve source offsets to make every extracted claim inspectable.

Sources: [SLP: information extraction](https://web.stanford.edu/~jurafsky/slp3/21.pdf), [SLP: coreference and entity linking](https://web.stanford.edu/~jurafsky/slp3/24.pdf).

## N18. Machine translation and BLEU evaluation

Translation generates a target-language sequence preserving the source's meaning. Several target sentences may be valid, so exact string equality is often too strict. Encoder–decoder models condition generation on the source while the target decoder uses its permitted prefix.

BLEU counts modified n-gram precision against reference translations, combining different n-gram orders and a brevity penalty. Clipping prevents a repeated candidate word from earning more matched occurrences than the reference allows. The brevity penalty discourages a very short output that matches only easy words.

Corpus BLEU aggregates counts across a corpus; it is not generally the mean of sentence BLEU. Tokenisation, reference choice, smoothing, and implementation affect the number. A candidate with good overlap can still reverse a fact or omit a crucial negation, while a valid paraphrase can lose overlap.

Report the evaluation signature and inspect meaning, names, numbers, and omissions. Learned semantic metrics and human judgements offer additional evidence but also have assumptions and biases. Changing decoding or the reference set is not the same experiment as improving the model under a fixed evaluation protocol.

Sources: [SLP: machine translation](https://web.stanford.edu/~jurafsky/slp3/13.pdf), [Papineni et al.: BLEU](https://aclanthology.org/P02-1040/).

## N19. Question answering, answer spans and abstention

Extractive question answering selects a span from a supplied passage. Open-domain QA first has to find a relevant passage; generative QA may produce an answer not present as one contiguous span. These settings provide different evidence and require different failure checks.

A span model scores start and end positions, then chooses a valid span under a stated decoding rule. Independent maximum start and end scores can produce an invalid order, so constraints matter. An unanswerable question should not force an arbitrary extraction.

Exact match checks equality after a defined normalisation. SQuAD token F1 compares token overlap as a multiset: “19 August” versus “August 19” has F1 1 but exact match 0 under its official evaluator. This reveals a limitation of overlap, not partial credit for word order. Boundary and normalisation policies must match before reproducing scores.

Abstention trades coverage against errors among answered questions. At zero coverage that conditional error is undefined, not zero. Choose an operating threshold on validation data and test it unchanged. A model can remain confidently wrong under domain shift, so confidence-based abstention is not proof that every answered question is supported.

Sources: [SQuAD project](https://rajpurkar.github.io/SQuAD-explorer/), [Official SQuAD evaluator](https://github.com/rajpurkar/SQuAD-explorer/blob/master/evaluate-v2.0.py).

## N20. Summarization, dialogue and source faithfulness

A summary compresses source information for a purpose. Extractive summaries select source passages; abstractive summaries generate new wording. Fluent new wording can omit a qualification, merge incompatible claims, or introduce an unsupported statement. Readability and faithfulness are separate evaluation targets.

ROUGE measures forms of reference overlap, such as shared n-grams or a longest common subsequence. It is useful for a stated comparison but does not prove factual support. A summary can preserve many reference words while reversing “did not improve” into “improved.” Check claims against the source and consider important omissions as well as additions.

Dialogue adds an interaction history and user goal. A helpful response should advance that goal, respect the user's constraints, and remain supported. Agreement or polite tone alone is not evidence of usefulness. Evaluate complete turns or tasks, not only whether a response sounds confident in isolation.

Keep generation settings, available context, reference policy, and human evaluation instructions explicit. If a system uses a generated summary as an intermediate representation, assess both whether that representation is faithful and whether it improves the final task. A downstream gain does not automatically validate every intermediate statement.

Sources: [Lin: ROUGE](https://aclanthology.org/W04-1013/), [Maynez et al.: faithfulness in abstractive summarisation](https://aclanthology.org/2020.acl-main.173/), [SLP: conversation](https://web.stanford.edu/~jurafsky/slp3/26.pdf).

## N21. Inference, discourse and argument mining

Argument mining identifies claims, reasons, and relationships such as support or attack. Discourse analysis also studies how statements connect across a text. The positive label “argument” can mean a standalone claim in one dataset and a claim with explicit justification in another. Define the policy before comparing transfer scores.

“Close the road” states a position. “Close the road because the bridge is unsafe” additionally supplies a reason. Context may reveal the missing justification, but it may also introduce irrelevant evidence. A generated abstraction might expose a claim–reason relation or invent one. These are empirical possibilities, not automatic benefits of structure.

Separate component detection from relation prediction and end-to-end structure recovery. A rule that treats every predicted claim role as a positive label can perform well when the dataset's label is largely claim-based; that does not establish a universal argument-identification rule. Test the mapping against contrasting guidelines and inspect individual disagreements.

For cross-dataset experiments, distinguish text changes, source-domain changes, label-policy changes, and generator changes. A controlled comparison holds the relevant conditions fixed. The central question is which evidence supports the label under this policy, not which representation looks most linguistically elaborate.

Sources: [Lippi and Torroni: argument mining survey and open manuscript](https://cris.unibo.it/handle/11585/523460), [SLP: discourse coherence](https://web.stanford.edu/~jurafsky/slp3/25.pdf).

## N22. Cross-lingual and cross-domain transfer

Transfer evaluates a model beyond its fitted source distribution. Cross-lingual transfer changes language; cross-domain transfer may change topic, style, population, or annotation policy. A multilingual encoder supplies shared parameters, not a guarantee that these changes are harmless.

A source–target matrix makes the comparison visible: train on each source row, evaluate on each target column. Diagonal entries describe within-dataset performance; off-diagonal entries describe transfer. Keep label mappings and evaluation metrics explicit. A lower off-diagonal score may reflect an altered definition of the target, not merely unfamiliar words.

Leave-one-dataset-out training pools several sources and holds one dataset out. It tests whether evidence from multiple environments helps a new one, but does not make the held-out dataset independent if its results guided feature or prompt choices. Translate-train and translate-test introduce translation quality as another condition.

Inspect tokenisation efficiency, rare labels, named entities, and cultural or stylistic differences. A shared embedding space can align some meanings while leaving task conventions mismatched. Report per-target outcomes rather than allowing a large easy dataset to dominate an average and hide failed transfer.

Sources: [Conneau et al.: XLM-R](https://arxiv.org/abs/1911.02116), [D2L: distribution shift](https://d2l.ai/chapter_linear-classification/environment-and-distribution-shift.html).

## N23. Long-document encoding and evidence coverage

A model's context limit bounds what it can read at once. Truncation may discard the only sentence answering a question. Increasing the context window changes capacity to include evidence, but does not guarantee that the model will use it correctly.

Sliding windows encode overlapping pieces. A length-100 document with window 40 and stride 30 yields starts at 0, 30, and 60 under a suitable end policy; overlap preserves some boundary context but repeats content. A hierarchical model first represents chunks, then combines chunk representations. Sparse-attention architectures restrict communication patterns to reduce cost, sometimes adding global positions.

Define how predictions from multiple windows are combined and how offsets map back to the original text. Double-counting overlapping tokens can distort loss or evaluation. A document-level label attached to every chunk can also create noisy supervision when only one chunk contains the needed evidence.

Measure evidence coverage alongside downstream accuracy: was the required passage available, retained, and used? Distinguish a retrieval miss from a reader failure. The long-document worked view illustrates coverage, not a claim that a local sparse-attention model was trained by the application.

Sources: [Longformer](https://arxiv.org/abs/2004.05150), [Lost in the Middle](https://arxiv.org/abs/2307.03172).

## N24. NLP evaluation and a complete study

An NLP study should connect its annotation policy, representation, training objective, and reported output metric. Token accuracy, exact entity F1, retrieval recall, and summary faithfulness count different things. A complete report names the unit, the split, and the model-selection process before interpreting the score.

For two gold entities, predicting one exact typed span and one wrong span gives one True Positive, one False Positive, and one False Negative: precision and recall are both 0.5. A token-level metric can give a different impression because many non-entity tokens are easy. Show a concrete error, not only a leaderboard.

Behavioural testing complements a representative held-out sample. CheckList-style tests ask whether capabilities such as negation handling or name invariance behave as intended. They are test suites, but do not automatically estimate performance on a naturally sampled deployment population. A successful template test also does not prove the broader capability is solved.

Keep a baseline, meaningful ablations, uncertainty with the right independent unit, and evidence for generated intermediate records. Save prompts, preprocessing, data versions, and evaluation scripts. When a result fails to transfer, narrow the conclusion rather than hiding the target. That is a scientific outcome, not a reason to replace the metric after seeing the answer.

Sources: [Ribeiro et al.: CheckList](https://aclanthology.org/2020.acl-main.442/), [scikit-learn: evaluation metrics](https://scikit-learn.org/1.7/modules/model_evaluation.html).
