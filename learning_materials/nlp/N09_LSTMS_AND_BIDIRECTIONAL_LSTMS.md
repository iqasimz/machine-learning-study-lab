# N09. LSTMs and bidirectional LSTMs

A long short-term memory network, or LSTM, is a recurrent model with a cell state for carried memory and a hidden state exposed to other layers. Gates determine how much old memory to retain, new information to write and memory to expose. A bidirectional LSTM uses two independent such recurrences when both input directions are available.

The scalar cell uses inputs [1,0,−1], fixed input/output gates and a selectable forget gate. Inspect one position to see its previous memory, candidate, new cell and hidden output. A second direction changes the representation width, not the sequence length.

An ordinary RNN repeatedly transforms its carried state. LSTM adds an additive gated memory path to address some training difficulties; the earlier rules for bidirectional alignment, padding and terminal summaries still apply.

## 1. Cell state, gates and hidden output

An LSTM carries cell state c and hidden state h. The forget gate scales old memory; the input gate controls a candidate addition; the output gate controls how much transformed cell state is exposed as h. Sigmoid gates lie between zero and one, but the cell state itself is not constrained to that interval.

The browser uses constant gate biases and a candidate that depends on the current input and previous hidden state. Those deliberately small parameters permit a complete numerical trace. A real LSTM normally learns input/recurrent matrices for every gate. There are no peepholes or projections in this teaching cell.

The additive memory path can help gradients travel, but forget gates can still shrink information, other paths still matter and useful long-range behaviour is not guaranteed. Increasing a forget gate retains more previous cell value in this example; it is not a universal accuracy improvement.

## 2. Stacked and bidirectional LSTMs

A BiLSTM runs two separately parameterized LSTMs, then aligns and concatenates their hidden outputs. It does not make one cell run backwards during its forward recurrence. The same future-context restrictions as a BiRNN apply.

For a padded batch, tensor layout, valid lengths and directional terminal states must be checked before classification. PyTorch's gate order and weight layout should be read from the actual API when porting parameters. A stacked BiLSTM passes a 2H-width token representation to its next layer; layers are not the same axis as directions. A CRF can follow its token features, but the LSTM gates do not enforce a valid tag sequence by themselves.

## Sources

[PyTorch LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html) · [Speech and Language Processing — author-hosted draft](https://web.stanford.edu/~jurafsky/slp3/)
