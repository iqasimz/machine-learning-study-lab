# N21. Inference, discourse and argument mining

Language understanding involves more than classifying a sentence in isolation. Natural-language inference compares a premise with a hypothesis; discourse analysis relates text units; argument mining identifies claims, reasons and support or attack relations under a specified scheme. These tasks share language evidence but do not define the same output or universal argument label.

Three authored units have roles claim, premise and other. One toy guideline counts only claims as positive; another counts claims and premises. The same premise therefore changes binary label when the policy changes, even though its text and supplied role stay fixed.

Source faithfulness asks whether a statement is supported; this lesson distinguishes semantic inference, discourse roles and argumentative structure. It also shows why a successful intermediate representation must be tested against the actual target guideline.

## 1. Inference, context and argumentative structure

Natural-language inference distinguishes entailment, contradiction and an unresolved/neutral relation under a specified premise. A contradiction can come from negation, quantification, time or incompatible entities; lexical overlap alone does not identify it. Pragmatics also considers speaker intention and context, which can make irony or indirect requests difficult.

Discourse relations connect units such as reasons, contrasts and elaborations. Argument mining may identify argumentative units, assign claim/premise roles and extract support/attack relations. Those are separate prediction targets. A role label is not automatically a universal binary definition of an argument.

The diagram compares two explicitly declared annotation policies on three authored units. It demonstrates how the same text/role can map to different labels. It does not assert that all real datasets use either of these policies.

## 2. Intermediate representations and target definitions

A generated role, claim or explanation can help a downstream predictor, introduce errors or restate an annotation shortcut. Compare direct text, context/guideline access and structured fields under controlled information and selection budgets.

A role-only rule that works on claim-oriented annotations may fail where premises, conversational implications or other definitions matter. State that scope without dismissing the diagnostic. Evaluate role accuracy, relation quality and the final task separately where gold data permit it. External discussion can motivate a hypothesis; it is not a substitute for a held-out comparison.

## Sources

[Argument mining survey](https://aclanthology.org/J19-4006/) · [Speech and Language Processing — author-hosted draft](https://web.stanford.edu/~jurafsky/slp3/)
