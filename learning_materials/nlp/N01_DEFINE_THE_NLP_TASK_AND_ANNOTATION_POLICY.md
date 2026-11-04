# N01. NLP tasks and annotation guidelines

An NLP task specifies what a system must predict from language: a document label, a token tag, an answer span, a relation or a generated sequence. Annotation guidelines explain what each label means and which context may be used. A task name such as sentiment or argument identification is not enough to define the correct answer.

For 'not good', sentence sentiment asks for one negative label; token tagging asks for labels aligned to individual words. We also compare four binary annotations from two people to see how agreement depends on their label usage.

After deciding how text is represented, decide what counts as a correct output. The same text can support different tasks or guidelines, so this choice must come before classifier architecture and score comparisons.

## 1. Annotation guidelines and independent judgements

An annotation policy states what counts as a positive example, which evidence annotators may read and how to handle uncertainty. A document, sentence, token, span and relation are different prediction units. “Argument”, “toxic” or “relevant” is not a complete operational definition. Preserve the policy version with the dataset.

Start with a few representative and difficult examples. Annotators should label independently before discussing disagreements; adjudication resolves a final label but does not erase the original disagreement. Distinguish an ambiguous instruction from a missing passage, a mistaken annotation or a genuinely subjective judgement. A model cannot repair a target that was never defined.

The example agreement calculation is deliberately small: two annotators agree on 3 of 4 binary cases, with one annotator using the classes twice each and the other using them three times and once. Observed agreement is 0.75, chance agreement from their marginals is 0.5, and Cohen's kappa is (0.75−0.5)/(1−0.5)=0.5. Kappa depends on prevalence and the annotation design; it is not a universal dataset-quality score.

## 2. Policy versions, source groups and pipeline errors

Record text identifiers, source groups, offsets, permitted context and labels separately. Split by the actual independent source before learning a vocabulary or selecting a prompt. A random sentence split can leak documents, authors or near duplicates. Inspect a simple baseline and annotation disagreements before paying for a large encoder.

An end-to-end NLP pipeline can fail at segmentation, representation, decision, decoding or evaluation. Save intermediate records so an incorrect final answer can be traced to its origin. Changing both annotation policy and model at once prevents a clean architecture comparison. The shared experimental-design lesson owns the general split/selection rules; this course applies them to language.

## Sources

[CheckList behavioural testing](https://aclanthology.org/2020.acl-main.442/) · [Speech and Language Processing — author-hosted draft](https://web.stanford.edu/~jurafsky/slp3/)
