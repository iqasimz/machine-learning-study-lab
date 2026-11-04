# N07. Convolutional networks for text

A text convolutional neural network applies shared filters to nearby positions in a sequence. A filter can detect local patterns across token embeddings or characters, and pooling summarizes responses for a larger unit such as a sentence. Local feature extraction and sentence-level aggregation are separate operations with different information losses.

Three scalar token features [1,−1,2] pass through the same width-two filter [0.5,−0.5]. Two responses are produced, then ReLU removes negative responses. Switch max and mean pooling to see how the sentence feature changes while the filter stays fixed.

Static embeddings represent individual tokens; a text CNN combines neighbouring tokens into learned local patterns. The shared DL convolution lesson explains spatial mechanics, while this lesson focuses on language sequence axes and pooling.

## 1. Text convolution, channels and local windows

A word-level TextCNN receives a sequence of embeddings, typically shaped batch × tokens × embedding width. A temporal filter reads several adjacent tokens and all input features. The same filter weights are reused at every valid position. Several widths capture different local patterns; several filters per width learn different features.

Character CNNs can build a representation of a word before a word-level recurrent or contextual encoder. A character-level document model is a different arrangement: its sequence positions are characters for the whole input. Keep the unit of each axis explicit.

The scalar example uses three token features and one width-two filter. Real embeddings have multiple dimensions, but the shared sliding calculation is the same. Max pooling keeps the strongest local response and loses its absolute location; mean pooling distributes influence differently.

## 2. Padding and receptive fields

Padding can create artificial high responses if convolution/pooling does not respect valid positions. A bias and an activation can make padded windows nonzero. Specify whether windows touching padding are allowed and mask invalid pooled positions when appropriate.

A shallow CNN has a bounded receptive field; stacked/dilated convolutions extend it. Pooling over local windows is not general syntactic reasoning. Compare parameter count, token budget and selection protocol with a recurrent or sentence-encoder baseline. The detailed spatial convolution derivation already has a shared DL lesson; this NLP view concentrates on sequence axes and pooling.

## Sources

[TextCNN original paper](https://aclanthology.org/D14-1181/) · [Speech and Language Processing — author-hosted draft](https://web.stanford.edu/~jurafsky/slp3/)
