# N12. Sequence labelling with HMMs, CRFs and Viterbi

Sequence labelling assigns a tag to each position while allowing neighboring tags to influence one another. A hidden Markov model, or HMM, uses probabilities for hidden transitions and observed emissions; a conditional random field, or CRF, uses scores to model tag sequences given an input. Viterbi finds a best path, while summing paths supplies a different normalization calculation.

Three positions each permit tags A or B, creating eight possible sequences. The displayed emission, start and transition values determine all path scores. We compute the best sequence and, independently, the sum needed to normalize path probabilities.

An encoder can produce per-token evidence without deciding a complete output structure. This lesson adds dependencies between adjacent output tags and separates structured decoding from training and task-specific span evaluation.

## 1. Path scores, decoding and normalization

An HMM models a joint distribution over observations and hidden tags using transition and emission probabilities. A linear-chain CRF models a conditional distribution over tag sequences given an input, using emission/transition scores and one global normalizer. CRF emissions need not sum to one, and its transition matrix is not automatically a probability matrix.

Viterbi decoding keeps the best prefix score ending in each tag and stores backpointers. Its max operation differs from the forward algorithm's sum/log-sum-exp over paths. CRF training uses a log-partition function; replacing it with the Viterbi maximum is not the same objective.

The browser enumerates all eight paths of a three-position, two-tag example to independently check the dynamic program. HMM logZ is the observation log-likelihood; CRF logZ normalizes conditional path scores. The tiny tag names A/B are not a complete NER scheme.

### Reconstruct one Viterbi update

Start and transition values are shown in the live example. For the CRF, the first prefix scores are [2,1]. At the second position, ending in A gives `0+max(2+0.5, 1−0.5)=2.5`; ending in B gives `3+max(2−1, 1+0.8)=4.8`. Record the winning predecessor separately for each ending tag. At the third position the two final scores become 6.3 and 5.6. The winner is B→B→A, not the independently highest-emission path A→B→A.

For the HMM, first-position joint masses are `0.6×0.8=0.48` and `0.4×0.3=0.12`. At the second position they become `0.2×max(0.48×0.7,0.12×0.4)=0.0672` and `0.7×max(0.48×0.3,0.12×0.6)=0.1008`. The live recurrence takes logarithms to avoid tiny products; it makes the same maximum-path choice. Adding probabilities instead of maximizing would answer a different, path-sum question.

## 2. From tag sequences to exact entities

POS tagging assigns grammatical labels to tokens. NER additionally needs entity boundaries and types. BIO/IOB2/BILOU are different encoding conventions; a CRF does not guarantee legal BIO transitions unless those constraints are imposed.

A BiLSTM–CRF or encoder–CRF separates representation, emission projection and structured decoding. Subword alignment must be resolved before word/entity evaluation. Predicting the correct type for only half of a named entity can be a wrong span, even when most token labels are correct. Entity span/type F1 is demonstrated in the final evaluation lesson, not substituted with ordinary token accuracy.

### Training is not just best-path decoding

For a labelled CRF sequence, the negative conditional log-likelihood is
\[
L=\log Z(x)-s(x,y_{gold}).
\]
For the displayed CRF's gold path A→B→A, emissions sum to 7 and transitions contribute −1.5, so the gold score is 5.5. Its loss is the example's computed logZ minus 5.5, not logZ minus the Viterbi winner unless the gold sequence actually wins. Gradients compare model-expected sufficient statistics with the gold statistics; a neural emission network can receive those gradients.

For an HMM with known tags, count-based transition/emission estimation is supervised. With hidden tags, forward–backward computes posterior state/transition expectations; Baum–Welch uses them in an EM procedure. It sums uncertainty over paths rather than replacing all uncertain labels with one Viterbi path. These learning procedures are described here but are not new dashboard trainers.

## Sources

[CRF author tutorial](https://arxiv.org/abs/1011.4088) · [Speech and Language Processing — author-hosted draft](https://web.stanford.edu/~jurafsky/slp3/) · [BiLSTM-CNN-CRF](https://aclanthology.org/P16-1101/) · [HMM author chapter](https://web.stanford.edu/~jurafsky/slp3/A.pdf)
