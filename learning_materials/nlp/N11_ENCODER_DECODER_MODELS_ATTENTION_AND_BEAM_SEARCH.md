# N11. Encoder–decoder models, attention and beam search

An encoder–decoder model represents an input sequence and generates a separate output sequence. Attention lets a decoder read different source positions while producing each target token. Beam search is a decoding procedure that keeps several candidate target prefixes; it changes the search, not the learned model weights or the definition of attention.

First we mix source values [1,2,4] using supplied attention weights to obtain context value 2. Separately, a two-token search tree shows greedy choosing a x with probability 0.30 while beam width two can keep b y with probability 0.38.

Recurrent encoders supply sequence states; an encoder–decoder uses source states to support another sequence's predictions. Distinguish source conditioning from candidate search before considering translation and other generation tasks.

## 1. Source attention and teacher forcing

An encoder reads a source sequence. A decoder predicts a target sequence conditioned on that source and permitted earlier target tokens. A classical recurrent encoder–decoder may initialize its decoder from a terminal source state. Attention allows each decoder query to read source positions again, reducing reliance on one fixed vector.

Encoder source attention, causal decoder self-attention and decoder-to-source cross-attention have different axes and masks. With two target queries and three source positions, cross-attention scores have shape 2 × 3 per head. A recurrent additive-attention score and scaled dot-product attention use different scoring functions even though both normalize over source positions.

Teacher forcing provides known earlier target tokens during training. It must not expose the token currently being predicted as an allowed input. The mismatch between gold prefixes during training and generated prefixes at inference can matter.

## 2. Search rules and generation choices

Greedy decoding selects the current best next token. Beam search retains a bounded collection of candidate prefixes, expands them and ranks cumulative scores. It is an approximation, not proof of the globally best sequence or the most truthful answer. Length penalties, end-of-sequence handling and constraints change the search objective.

T5 and BART apply encoder–decoder architectures with different pretraining recipes. The shared architecture-family lesson owns the full Transformer structure; this view focuses on source/target alignment and the small prefix tree. Compare generation quality, output length and search cost rather than treating beam width as a free quality knob.

## Sources

[Original neural translation attention](https://arxiv.org/abs/1409.0473) · [Speech and Language Processing — author-hosted draft](https://web.stanford.edu/~jurafsky/slp3/)
