# N22. Cross-lingual and cross-domain transfer

Transfer evaluation tests whether a system developed under one condition works under another. Cross-lingual transfer changes language; cross-domain transfer changes the source or content distribution; annotation-policy changes can alter the label meaning itself. These shifts can coexist, so a score drop alone does not explain its own cause.

A two-by-two authored table uses fine-tuning sources as rows and evaluation targets as columns. Selecting source A or B changes the highlighted row. The scores are invented for interpretation practice, not measurements of real languages or named models.

Earlier lessons fixed targets, representations and evaluation units. Transfer now asks which parts remain compatible across conditions, including whether a base multilingual model had already encountered target-language text during pretraining.

## 1. Transfer protocols and types of shift

A cross-lingual experiment may fine-tune on one language and evaluate another, while the base model has already seen both languages during pretraining. That is not zero exposure to the target language. Translate-train, translate-test, multilingual joint training and target-language adaptation are different protocols.

Domain shift changes source/content conditions; annotation-policy shift changes what the label means. Those can coexist, but a score gap alone does not identify which caused it. Report per-language/domain outcomes and relevant data characteristics, not only one grand mean.

The diagram's 2 × 2 source–target score matrix is authored for interpretation practice. Its off-diagonal drops are not empirical claims about English, German or any model. Row and column meaning must remain explicit.

## 2. Low-resource conditions and target adaptation

Consider token coverage, morphology, script, code switching, dialect, annotation availability and community-specific meaning. Character/subword features, adapters, translation and data augmentation each add assumptions and cost. Synthetic labels require review and can preserve a teacher's biases.

Avoid letting translated or near-duplicate versions of one example cross training/test groups. Use a development protocol that matches the intended transfer scenario; tuning on target test labels is adaptation, not untouched transfer. XLM-R and XTREME are useful original references, but neither establishes coverage of every language, task or social context.

## Sources

[XLM-R original paper](https://aclanthology.org/2020.acl-main.747/) · [Speech and Language Processing — author-hosted draft](https://web.stanford.edu/~jurafsky/slp3/)
