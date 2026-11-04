# N13. Syntax: dependency and constituency parsing

Syntactic parsing predicts how words form a grammatical structure. A dependency parse connects words through head–dependent links; a constituency parse groups spans into nested phrases. These are alternative representations of a sentence, not successive processing stages or the same object as a list of word-class tags.

In 'cats chase mice', the example dependency tree makes chase the root, with cats as subject and mice as object. The deeper constituency example instead groups chase mice as a verb phrase and calculates a small grammar derivation.

Sequence labelling assigns tags along input order. Parsing adds a tree structure between or around those positions. Its constraints and evaluation must therefore match the chosen grammatical representation.

## 1. Dependency trees and constituent phrases

Constituency groups spans into nested phrases; dependency parsing links words through head–dependent relations. A POS tag sequence alone gives neither tree. Treebanks define tokenization, relation inventories and annotation conventions; an attachment can be debatable under different schemes.

A dependency parser scores arcs and often relation labels. Choosing each token's best incoming arc independently can create cycles or violate root constraints. Transition-based systems build a structure through actions; graph-based systems decode from arc scores under structural constraints. Projective and non-projective parsing permit different crossing structures.

The diagram uses “cats chase mice”: chase is the root, cats its subject and mice its object under the stated simple analysis. This is an authored illustration, not the output of a downloaded parser.

## 2. Parsing evaluation and constituency charts

Unlabelled attachment score measures correct heads; labelled attachment score also requires the relation label. Tokenization, punctuation inclusion and root conventions must match before comparing percentages. A parser can be useful while making errors on the structures important to a downstream task.

Dependency paths can connect candidate entities, and constituency spans can propose extraction units, but neither automatically proves a semantic relation. Neural parsers commonly use contextual token representations and biaffine scoring; a graph neural network is an optional additional representation, not another name for dependency parsing. Use a specified Universal Dependencies treebank and official scorer for an implementation study.

### A constituency chart is another structure

Consider a deliberately small probabilistic context-free grammar: S→NP VP with probability 1; VP→V NP with probability 1; V→chase with probability 1; NP→cats and NP→mice each with probability 0.5. For “cats chase mice”, the phrase derivation has probability 0.5×1×1×0.5=0.25.

A CKY-style chart first fills one-word spans, then combines smaller spans using compatible binary rules. The span “chase mice” becomes VP, then the full span becomes S. This illustrates a grammar already in binary form; general grammars need an appropriate transformation or another parsing algorithm. Constituent trees, dependency arcs and the sequence of POS tags are three different representations, even when all describe one sentence.

## Sources

[Universal Dependencies guidelines](https://universaldependencies.org/guidelines.html) · [Speech and Language Processing — author-hosted draft](https://web.stanford.edu/~jurafsky/slp3/)
