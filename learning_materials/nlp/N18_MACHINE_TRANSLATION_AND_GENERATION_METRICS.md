# N18. Machine translation and BLEU evaluation

Machine translation produces a target-language sequence intended to preserve a source's meaning. Automatic reference-based metrics compare a candidate with permitted reference translations, but they test different kinds of agreement. BLEU combines clipped n-gram precision with a brevity penalty; the simple example here isolates unigram BLEU-1 rather than implementing standard corpus BLEU.

The reference is 'the cat sat down'. Choose a short candidate, a repeated word or the exact reference. Clipping stops a repeated sat from gaining unlimited credit, and the brevity penalty reduces the score of an otherwise matching short fragment.

Encoder–decoder attention and search explain how a translation may be produced. This lesson asks how to evaluate the resulting text without confusing lexical overlap with preserved meaning or full translation quality.

## 1. Translation objectives and overlap metrics

Translation can use recurrent attention models, encoder–decoders or instructed generative models, but the source/target task must stay fixed. Segmentation, script, morphology, domain terminology and document context all affect what a correct translation means. Teacher forcing and beam search are explained in the sequence-to-sequence lesson rather than repeated here.

BLEU combines clipped n-gram precisions with a brevity penalty. The browser deliberately computes unigram BLEU-1 for transparency; it is not standard four-order corpus BLEU. Corpus BLEU aggregates counts before applying its formula, not by averaging sentence BLEU scores. Smoothing, effective order, reference count and tokenization affect the result.

chrF compares character n-gram overlap; COMET uses a learned evaluation model. A fluent or high-overlap output can still reverse a fact or omit a critical condition.

## 2. Scorer configuration and meaning preservation

Report the SacreBLEU signature or equivalent full configuration, corpus split, reference set, model/tokenizer revision and decoding settings. Learned metrics require their exact checkpoint and input configuration. Reference translations are not the only valid wording.

Inspect meaning preservation, omissions, additions, names, numbers and negation. Human review and targeted challenge sets can expose errors missed by aggregate overlap. Match language pairs and domain before interpreting an apparent improvement. This course does not download a multilingual translation model or claim a measured translation benchmark.

## Sources

[SacreBLEU official implementation](https://github.com/mjpost/sacrebleu) · [Speech and Language Processing — author-hosted draft](https://web.stanford.edu/~jurafsky/slp3/) · [BLEU](https://aclanthology.org/P02-1040/) · [COMET](https://aclanthology.org/2020.emnlp-main.213/)
