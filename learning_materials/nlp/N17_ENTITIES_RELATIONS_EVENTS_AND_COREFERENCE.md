# N17. Information extraction: entities, relations and events

Information extraction converts text into structured records about mentions, entities, relations and events. A mention is a source span; coreference links mentions intended to refer to the same entity. Relation and event extraction add typed connections or roles. A valid output format does not by itself make these records supported by the text.

In 'Maya founded Acme. She leads it.', the intended reading links She to Maya and it to Acme. Founder and leader are different relations. Keeping these links and original mentions visible helps locate errors instead of hiding them inside a fluent generated record.

Classification assigns a label to a chosen unit. Extraction adds boundaries and links between units, so reliable source offsets and task-specific validation become essential in addition to a good encoder.

## 1. Mentions, links and typed records

An entity mention is a text span; an entity identifier links a mention to an external record. Coreference groups mentions referring to the same entity in a document. A relation connects arguments under a schema; an event may have a trigger and several typed roles. None of these objects follows automatically from a correct sentence-class label.

A pipeline can first detect mentions and then classify mention pairs. Joint models share representations or decode structures together, potentially changing error propagation. A generated JSON record is another output format, not proof that every field refers to real source text.

The authored example contains “Maya founded Acme. She leads it.” It links She to Maya and it to Acme under the intended reading, then distinguishes founded from leads. The explicit source anchors make a mistaken link visible.

## 2. Reference validation and semantic support

Check span boundaries, type inventories, legal argument roles, identifier resolution and source support. Nested or discontinuous mentions need richer representations than one BIO tag per token. Multiple plausible readings may need adjudication or abstention rather than an invented certainty.

Relation evaluation can assume gold mentions or require end-to-end extraction; those are different protocols. Coreference metrics also vary, so report the scorer/version and mention policy. Use deterministic schema/reference checks before semantic audit. Retain intermediate candidates to identify whether an error came from mention detection, linking, relation typing or generated unsupported content.

## Sources

[spaCy information-extraction pipeline concepts](https://spacy.io/usage/linguistic-features#named-entities) · [Speech and Language Processing — author-hosted draft](https://web.stanford.edu/~jurafsky/slp3/)
