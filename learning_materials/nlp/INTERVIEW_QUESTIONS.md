# NLP interview questions

These situations connect language models to classification, search, extraction and evaluation. Explain the immediate problem first, then use an example to justify the method. Follow-ups preserve deeper distinctions without turning the opening answer into a terminology list. These are example answers, not claims of completed project experience.

## INLP01 — You receive labelled support tickets. What text-classification baseline would you build?

I would first read some tickets and their labels to understand what the classifier is supposed to distinguish. I would reserve independent evaluation data, grouping tickets by customer, conversation or template when those repeated sources would otherwise leak across splits. A majority-class predictor gives a minimum reference; word/character TF-IDF with logistic regression is a useful next baseline because it is inexpensive and its errors are easy to inspect.

TF-IDF represents which words or character sequences occur and how distinctive they are in the training collection. The vocabulary and weighting must be fitted only on training data. Choose settings with validation data, then inspect representative mistakes such as negation, rare categories and copied email signatures. An embedding classifier or fine-tuned encoder is worth comparing when it fixes a demonstrated weakness under the same input and evaluation rules.

**Follow-up.** What if the new model receives an entire conversation but the baseline sees one sentence? That is a useful operational comparison, but it changes available information as well as the model. Do not attribute the whole gain to architecture. [TF-IDF and text pipelines](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html).

## INLP02 — Would you use a bidirectional RNN to predict the next word while someone types?

Not unchanged if it can read the future word being predicted. A forward RNN summarizes the available prefix. A bidirectional RNN adds a separately learned reverse pass over the supplied input, so a representation can use both earlier and later positions. That is useful when classifying a complete sentence, but future target words are not available during live next-word prediction.

For example, a representation of “bank” can use “river” later in a completed sentence. A predictor that must complete “we sat by the bank…” cannot assume that later word was already given. Bidirectionality describes information access, not a guarantee of better prediction.

**Follow-up.** Is the backward RNN the same as backpropagation through time? No. The backward encoder computes a second forward representation in reverse input order; backpropagation computes derivatives for learning. Truncated backpropagation limits the gradient history even if a detached hidden value is carried onward. For a bidirectional sentence summary, align terminal states with valid lengths rather than blindly selecting the final padded token. [PyTorch RNN inputs and states](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html).

## INLP03 — How does an LSTM keep useful information across a sentence?

An LSTM maintains a memory state as well as an exposed hidden output. Its gates decide how much old memory to retain, how much new information to add and how much of the updated memory to expose. This creates an additive route through time that can help learning dependencies which a plain recurrent update struggles to preserve.

In “The keys beside the large wooden cabinet are missing,” the useful number information concerns “keys,” not the nearer “cabinet.” A learned LSTM may preserve information relevant to the later prediction while changing other parts of its state. This is a model capability to evaluate, not a guarantee that a gate corresponds to a named grammatical rule.

The forget gate multiplies old memory; the input gate multiplies a candidate update; their sum forms new memory. The output gate scales a transformed version of it. Gates usually depend on the current input and previous hidden output. They can reduce some gradient problems, but do not provide perfect long-term recall.

**Follow-up.** Does a bidirectional LSTM make unavailable future words usable? No. It still needs the input context supplied to its two directions. [PyTorch LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html).

## INLP04 — When would you choose a GRU instead of an LSTM?

A GRU is a gated recurrent model with one carried hidden state rather than separate LSTM hidden and cell states. At the same input/hidden widths, a standard GRU has fewer gate parameters than a standard LSTM. It is a reasonable candidate when a simpler recurrent model meets the task and resource constraints, but fewer parameters do not guarantee faster execution or better generalization.

I would compare validation quality, measured runtime and memory under the intended sequence lengths and padding rules. Equal hidden width is not equal parameter count. An LSTM's explicit cell state can be useful, while a GRU may be adequate; the decision needs evidence rather than a universal ranking.

**Follow-up.** Why might copying GRU weights between implementations fail? Check gate order, biases, matrix orientation and candidate-state equations. In PyTorch's stated convention, the update gate weights old hidden state; some textbooks define the complementary gate instead. PyTorch also applies the reset gate after the recurrent affine transformation. A scalar zero-bias example can hide a multidimensional mismatch. [PyTorch GRU conventions](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html).

## INLP05 — Your NER model predicts I-PER immediately after O. How could a CRF help?

A linear-chain conditional random field (CRF) scores complete tag sequences using both token-level scores and scores for transitions between neighboring tags. It can prefer a consistent entity sequence over independently choosing the highest score at each token. Here the issue is not merely recognizing a person word: it is deciding whether a person mention starts or continues.

The encoder supplies features, an emission head scores tags and the CRF adds transition preferences. Viterbi decoding finds the highest-scoring path. Training instead needs a sum over possible paths for normalization; replacing that sum with the maximum changes the objective.

**Follow-up.** Does a CRF guarantee valid BIO tags? Only with an explicit hard-constraint policy. Learned transition penalties can discourage O → I-PER without forbidding it. Evaluate complete entity boundaries and types after subword-to-word alignment. A single flat BIO sequence also cannot represent every nested or discontinuous entity structure. [Original CRF paper](https://www.cs.columbia.edu/~jebara/6772/papers/crf.pdf).

## INLP06 — Can you use BERT's CLS vector directly for semantic search?

You can extract the vector, but that does not establish that cosine similarity captures the relevance you need. BERT produces contextual token features; a sentence/retrieval encoder is normally trained and evaluated for a particular relation between texts. A convenient vector is not automatically a useful search representation.

I would compare a lexical baseline with a compatible sentence or retrieval encoder on labelled queries and passages. Preserve the encoder revision, any query/document prompts, pooling and normalization. If using mean pooling, exclude padding. A model that is strong for paraphrase similarity may still struggle with exact product codes, negation or whether a passage actually answers a question.

**Follow-up.** What does CLS pooling change? It chooses one representation as the sequence summary; mean pooling combines valid token features. Neither choice creates a relevance-training objective on its own. Evaluate the complete encoding/scoring pipeline. [Sentence-BERT](https://arxiv.org/abs/1908.10084).

## INLP07 — How would you search a million documents and improve the top results?

I would separate finding candidates from carefully scoring them. A bi-encoder creates independent query and document vectors, so document representations can be cached and searched efficiently. A cross-encoder reads each query–document pair jointly and can provide a more detailed relevance score for a smaller candidate set, at extra inference cost.

For example, retrieve 100 candidates, then rerank those candidates. If the relevant document is missing from the 100, reranking cannot recover it. Measure candidate recall separately from final ranking quality, under a fixed relevance definition and workload. An approximate vector index can lose neighbors even before the ranker runs.

**Follow-up.** Where does late interaction fit? It independently encodes texts but retains token vectors for a later token-level matching rule. This can retain more detail than one vector per document while costing more storage and matching work. It is a different trade-off, not automatically the best compromise. Keep query/index normalization and model versions compatible. [DPR](https://arxiv.org/abs/2004.04906) · [ColBERT](https://arxiv.org/abs/2004.12832).

## INLP08 — Your embedding training treats another valid answer as a negative. Why is that a problem?

Contrastive learning encourages chosen positive pairs to score relatively higher than selected negative pairs. If two passages both correctly answer the same query, using one as a negative teaches a misleading distinction. Lower training loss can then reflect fitting the sampling scheme rather than improving useful retrieval.

I would inspect how positives and negatives were created, including duplicates, paraphrases and mined “hard” negatives. Hard means the current model finds them confusing, not that a human has confirmed they are irrelevant. Evaluate against reviewed relevance cases and inspect failures after training; changing temperature cannot fix incorrect pair labels.

**Follow-up.** What is the idea behind unsupervised SimCSE? It uses dropout-based views of the same sentence as a positive pair. That is not permission to treat arbitrary rewrites as meaning-preserving. Keep paraphrase or translation families together when constructing independent splits. [SimCSE](https://arxiv.org/abs/2104.08821).

## INLP09 — An NER model has high token accuracy but poor entity F1. How is that possible?

Most tokens may be outside any entity, so predicting O often can produce high token accuracy. Strict entity F1 instead requires a predicted mention to have the correct complete span and type. One boundary error can count as both a missed gold entity and an incorrect predicted entity.

If the reference is “New York City” but the system extracts only “New York,” several token decisions can look reasonable while the exact entity match fails. Keep subword predictions aligned with the original annotation unit; splitting one word into several model tokens must not create several gold entities.

**Follow-up.** Is question-answering token F1 the same score? No. SQuAD-style F1 uses normalized overlap of answer tokens under a specified scoring policy. A partial phrase may receive overlap credit while failing exact NER span matching. Precision/recall formulas are shared, but the objects being counted differ. [CoNLL evaluation task](https://aclanthology.org/W03-0419/) · [SQuAD](https://aclanthology.org/D16-1264/).

## INLP10 — A summary has high ROUGE but reverses the source's conclusion. What did the metric miss?

ROUGE measures specified forms of text overlap, not whether the summary preserves the source's claims. “The treatment improved outcomes” and “The treatment worsened outcomes” share most words but convey opposite findings. A high overlap score can therefore coexist with a serious semantic error.

I would examine source support for entities, quantities, comparisons, negation and conditions, using an explicit review rubric. A faithful summary repeats what its source says; a factually correct statement agrees with reality. These can diverge if the source is wrong or the summary adds an externally true but unsupported detail. Evaluate those questions separately.

**Follow-up.** Would BLEU or a learned scorer solve this automatically? No. BLEU has its own n-gram clipping, brevity and corpus-aggregation conventions. Learned metrics can add useful signals but are version-dependent models, not truth detectors. Report scorer configuration and inspect task-relevant errors rather than replacing one unexplained number with another. [ROUGE](https://aclanthology.org/W04-1013/) · [BLEU](https://aclanthology.org/P02-1040/).

## INLP11 — Your classifier works on news but fails on social-media posts. What would you investigate?

I would compare representative mistakes and the dataset definitions before choosing a more complex model. Social posts can use different language, references and context; the target label may also mean something different. Those are different explanations: unfamiliar wording, missing evidence and a changed annotation rule do not necessarily need the same repair.

Keep source, conversation, translation and near-duplicate groups separated where appropriate. Inspect whether hashtags, publisher names or templates acted as shortcuts in training. Test whether additional context helps under controlled information access; more context can also distract or expose labels. Report each target domain rather than hiding a failure in one large average.

**Follow-up.** Can you tune on the new domain? Yes, if that is the intended adaptation setting, but disclose which target labels or examples informed the choices. A claimed unseen-domain result cannot silently use those same cases for tuning. A generated intermediate representation is another hypothesis to test, not automatic generalization. [Domain adaptation review](https://aclanthology.org/2020.coling-main.603/).

## INLP12 — A document exceeds the context window. How would you classify it or extract information?

I would first identify what evidence the task needs. Chunking splits the document into manageable windows; overlap can preserve evidence near a boundary. Each chunk needs its original position so extracted spans can be traced back and duplicate overlap predictions can be reconciled.

For classification, define how chunk information becomes one document decision: pooling scores, pooling representations or a second model that reads the ordered chunks are different choices. For extraction, merge compatible source-anchored spans instead of counting overlapping copies as independent evidence. A fact whose beginning and end fall into separate windows is a useful failure case to test.

**Follow-up.** Is overlapping chunking the same as a long-context Transformer? No. Sparse/local/global attention changes the model's information connections; independent chunk encoding does not create those connections automatically. Compare evidence coverage, actual task quality and processed-token cost. A large configured window is not proof the trained model uses distant evidence reliably. [Longformer](https://arxiv.org/abs/2004.05150) · [Hierarchical attention networks](https://aclanthology.org/N16-1174/).
