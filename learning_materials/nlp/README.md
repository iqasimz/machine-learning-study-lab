# NLP course

26 lessons: two language/tokenization and embedding foundations, followed by 24 NLP topics. In Model Lab, open **NLP → Language foundations**, then **NLP course**, or choose the NLP path in Interactive mode. Shared mathematics/metrics are studied once; the original complete Transformer walkthrough remains linked in the LLM course.

Each lesson starts with a definition, a concrete example and a connection to earlier material. Seven conceptual/comparison lessons use reading sections; nineteen use diagrams for actual computational dependencies. The selected section explains its terms, calculation and limits. Unique deeper notes remain inside the same reader rather than as a second full lecture below it. Sidebar and Interactive mode share this edition. 

## Progression

| Lesson | Topic |
| --- | --- |
| P6 | [Language, tokens and model inputs](../pre_llm/06_LANGUAGE_TOKENIZATION_AND_NLP.md) |
| P7 | [Embeddings, similarity and retrieval](../pre_llm/07_EMBEDDINGS_AND_RETRIEVAL_FOUNDATIONS.md) |
| N01 | [NLP tasks and annotation guidelines](N01_DEFINE_THE_NLP_TASK_AND_ANNOTATION_POLICY.md) |
| N02 | [Text processing, morphology and source offsets](N02_TEXT_PROCESSING_MORPHOLOGY_AND_SOURCE_OFFSETS.md) |
| N03 | [Word counts, TF-IDF and lexical retrieval](N03_BAG_OF_WORDS_TF_IDF_AND_LEXICAL_RETRIEVAL.md) |
| N04 | [Naive Bayes and linear text classifiers](N04_NAIVE_BAYES_AND_LINEAR_TEXT_CLASSIFIERS.md) |
| N05 | [N-gram language models and smoothing](N05_N_GRAM_LANGUAGE_MODELS_AND_SMOOTHING.md) |
| N06 | [Static word vectors: Word2vec, GloVe and fastText](N06_WORD2VEC_GLOVE_AND_SUBWORD_WORD_VECTORS.md) |
| N07 | [Convolutional networks for text](N07_CNNS_FOR_WORDS_CHARACTERS_AND_SENTENCES.md) |
| N08 | [Recurrent and bidirectional recurrent networks](N08_RNNS_AND_BIDIRECTIONAL_RNNS.md) |
| N09 | [LSTMs and bidirectional LSTMs](N09_LSTMS_AND_BIDIRECTIONAL_LSTMS.md) |
| N10 | [GRUs and recurrent training](N10_GRUS_BIGRUS_AND_RECURRENT_TRAINING.md) |
| N11 | [Encoder–decoder models, attention and beam search](N11_ENCODER_DECODER_MODELS_ATTENTION_AND_BEAM_SEARCH.md) |
| N12 | [Sequence labelling with HMMs, CRFs and Viterbi](N12_HMMS_CRFS_VITERBI_AND_SEQUENCE_LABELLING.md) |
| N13 | [Syntax: dependency and constituency parsing](N13_SYNTAX_DEPENDENCY_PARSING_AND_CONSTITUENCY.md) |
| N14 | [Contextual token encoders: ELMo and BERT](N14_ELMO_BERT_AND_CONTEXTUAL_TOKEN_ENCODERS.md) |
| N15 | [Sentence encoders and query–document interaction](N15_SENTENCE_ENCODERS_BI_ENCODERS_AND_CROSS_ENCODERS.md) |
| N16 | [Text classification and paired-input inference](N16_CLASSIFICATION_MULTILABEL_PREDICTION_AND_PAIR_INFERENCE.md) |
| N17 | [Information extraction: entities, relations and events](N17_ENTITIES_RELATIONS_EVENTS_AND_COREFERENCE.md) |
| N18 | [Machine translation and BLEU evaluation](N18_MACHINE_TRANSLATION_AND_GENERATION_METRICS.md) |
| N19 | [Question answering, answer spans and abstention](N19_QUESTION_ANSWERING_READING_SPANS_AND_ABSTENTION.md) |
| N20 | [Summarization, dialogue and source faithfulness](N20_SUMMARIZATION_DIALOGUE_AND_FACTUAL_SUPPORT.md) |
| N21 | [Inference, discourse and argument mining](N21_INFERENCE_DISCOURSE_AND_ARGUMENT_MINING.md) |
| N22 | [Cross-lingual and cross-domain transfer](N22_MULTILINGUAL_AND_CROSS_DOMAIN_TRANSFER.md) |
| N23 | [Long-document encoding and evidence coverage](N23_LONG_DOCUMENTS_AND_HIERARCHICAL_TEXT_ENCODERS.md) |
| N24 | [NLP evaluation and a complete study](N24_EVALUATE_AND_BUILD_AN_NLP_PIPELINE.md) |

## What you can actually do

Select components, connections and guide entries. Change small, bounded settings: feature/smoothing rules, pooling, recurrence direction/position/gates, beam width, HMM/CRF model, sentence-encoder interaction pattern, candidate translation/answer span and long-document coverage. Calculations update deterministically. The 48 checks give specific reasoning for all four answers. Twelve [interview questions](INTERVIEW_QUESTIONS.md) connect the ideas to practical decisions.

N08–N10 distinguish RNN/BiRNN, LSTM/BiLSTM and GRU/BiGRU. N14–N15 connect ELMo/BERT-style contextual representations to sentence pooling, Sentence-BERT/SimCSE training, bi-/cross-encoders and ColBERT-style late interaction. Broader architecture variants are explained where they change an objective or information path; this is a broad first edition, not an exhaustive survey of all NLP research.

## Practice and data

The final lesson contains a local lexical-pipeline exercise and checked first-party routes for 20 Newsgroups, UD English EWT and SQuAD. N10 includes an executable packed-BiLSTM shape/terminal-state exercise. No corpus or model is downloaded automatically. Review source rights, annotation policy, splits, versions and dependencies before a real experiment.

These are authored examples and offline exercises, **not** new web trainers or benchmark results. Real training controls elsewhere in Model Lab retain their existing scope. Speech recognition/synthesis, every generative image/audio method and exhaustive NLP model training are outside this course release.

## Maintenance

- `metadata/lessons.json`: allowlisted titles, source files, prerequisites, related lessons and primary references.
- `../reader_profiles.json`: canonical definitions, examples, core explanations, variants and meaningful diagram dependencies.
- Markdown lessons: matching introductions, unique advanced notes, practical routes and sources.
- `../interactive_mode/nlp_checks.json`: 48 authored questions with option-specific feedback.
- Project `web/nlp_math.js`: deterministic numerical fixtures.
- Project `web/explorer_nlp.js`: linked lesson operations and controls.
- Project `web/nlp_figures.js`: recurrent, gated, trellis, encoder and task-specific detail diagrams.
- Project `tests/test_nlp.py`: independent PyTorch/sklearn/numpy and embedded-code checks.

Do not duplicate a shared definition in another full tutorial. Link to its owning lesson; keep application-specific units, conventions and caveats adjacent to the calculation. Preserve original source anchors when processing text.

## Primary-source registry

Checked 3 November 2026. Stable documentation URLs can change; the installed runtime used for tests is recorded by the project rather than inferred from a redirected documentation version. The [author-hosted Speech and Language Processing draft](https://web.stanford.edu/~jurafsky/slp3/) is a further reading route, not copied course text. All diagrams and numerical examples here are authored locally; no external artwork was reused.

| Lesson | Primary reference |
| --- | --- |
| N01 | [CheckList behavioural testing](https://aclanthology.org/2020.acl-main.442/) |
| N02 | [spaCy linguistic features and tokenization](https://spacy.io/usage/linguistic-features) |
| N03 | [scikit-learn text feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) |
| N04 | [scikit-learn sparse text classification](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html) |
| N05 | [NLTK language-model API](https://www.nltk.org/api/nltk.lm.html) |
| N06 | [Word2vec negative sampling paper](https://arxiv.org/abs/1310.4546) |
| N07 | [TextCNN original paper](https://aclanthology.org/D14-1181/) |
| N08 | [PyTorch RNN](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html) |
| N09 | [PyTorch LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html) |
| N10 | [PyTorch GRU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html) |
| N11 | [Original neural translation attention](https://arxiv.org/abs/1409.0473) |
| N12 | [CRF author tutorial](https://arxiv.org/abs/1011.4088) |
| N13 | [Universal Dependencies guidelines](https://universaldependencies.org/guidelines.html) |
| N14 | [BERT original paper](https://aclanthology.org/N19-1423/) |
| N15 | [Sentence-BERT original paper](https://aclanthology.org/D19-1410/) |
| N16 | [Hugging Face sequence classification guide](https://huggingface.co/docs/transformers/tasks/sequence_classification) |
| N17 | [spaCy information-extraction pipeline concepts](https://spacy.io/usage/linguistic-features#named-entities) |
| N18 | [SacreBLEU official implementation](https://github.com/mjpost/sacrebleu) |
| N19 | [SQuAD 2 original paper](https://aclanthology.org/P18-2124/) |
| N20 | [ROUGE original paper](https://aclanthology.org/W04-1013/) |
| N21 | [Argument mining survey](https://aclanthology.org/J19-4006/) |
| N22 | [XLM-R original paper](https://aclanthology.org/2020.acl-main.747/) |
| N23 | [Longformer original paper](https://arxiv.org/abs/2004.05150) |
| N24 | [seqeval strict sequence evaluation](https://github.com/chakki-works/seqeval) |

Additional original sources are linked in the relevant lesson, including GloVe, fastText, ELMo, BiLSTM-CNN-CRF, SimCSE, ColBERT, BLEU and COMET. One obsolete sklearn tutorial URL was replaced after checking; an impossible annotation marginal example and an overbroad BERT corruption sentence were corrected during review.
