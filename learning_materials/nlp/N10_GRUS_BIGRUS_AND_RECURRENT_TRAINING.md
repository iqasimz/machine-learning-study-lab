# N10. GRUs and recurrent training

A gated recurrent unit, or GRU, controls a recurrent hidden state without a separate LSTM memory cell. The reset gate controls old-state information used in the candidate; the update gate mixes the candidate with the old state. The exact gate convention and reset placement must be read from the equation, because implementations can differ.

The inputs are [1,0,−1]. We hold the old-state mixture weight z at 0.25 and change the reset value. The example uses PyTorch's reset-after-projection convention; a separate matrix-and-bias check demonstrates why placement matters.

LSTM introduced gated memory and exposed hidden state. GRU merges these into one carried state, while the same choices about directions, padding, gradient history and sequence summaries remain separate training decisions.

## 1. GRU equations and reset placement

This course uses PyTorch's GRU convention: \(n_t=\tanh(W_xx_t+b_x+r_t\odot(W_hh_{t-1}+b_h))\), followed by \(h_t=(1-z_t)\odot n_t+z_t\odot h_{t-1}\). Thus a large update gate z retains more old state. Some descriptions reverse which quantity is called the update gate; compare the equation rather than the name.

PyTorch applies the reset gate after the recurrent affine transform in the candidate. Multiplying the old hidden state before the matrix is a different operation in multiple dimensions, especially with biases. The scalar, zero-bias example agrees under both placements, so the independent test also includes a multidimensional nonzero-bias case.

A GRU has no separate LSTM-style cell state. A BiGRU adds another independent directional pass; it is not a GRU that reads both directions in a single recurrence.

## 2. Gradient history, parameter counts and packed batches

Teacher forcing, truncated BPTT, gradient clipping, recurrent-state resets and padding address different issues. Clipping controls a gradient norm; it does not restore lost supervision or repair a misaligned target. Detaching a carried hidden state cuts the graph at that boundary.

For a fair architecture comparison, match data, preprocessing, selection budget and evaluation. Hidden width alone does not match parameter count across RNN/LSTM/GRU or across directions. Use the actual framework's parameter shapes, including its biases, to count them. Shared optimizer/system lessons own general Adam, scheduling and batching details.

### Count the actual framework parameters

For a single PyTorch layer with input width D, hidden width H, biases enabled and no projection, each direction contains \(G H(D+H+2)\) parameters. G is 1 for RNN, 3 for GRU and 4 for LSTM; the +2 accounts for two framework bias vectors per gate. With D=3 and H=4, those counts are 36, 108 and 144 per direction. Bidirectionality doubles this one-layer count. For a later bidirectional layer, input width is normally 2H, so do not simply reuse the first-layer D for the entire stack.

A shape check you can run without downloading data:

```python
import torch
x = torch.zeros(2, 5, 3)  # batch, padded positions, input features
lengths = torch.tensor([5, 3])
packed = torch.nn.utils.rnn.pack_padded_sequence(
    x, lengths.cpu(), batch_first=True, enforce_sorted=False
)
model = torch.nn.LSTM(3, 4, batch_first=True, bidirectional=True)
packed_output, (h, c) = model(packed)
output, valid_lengths = torch.nn.utils.rnn.pad_packed_sequence(
    packed_output, batch_first=True, total_length=5
)
sentence = torch.cat([h[-2], h[-1]], dim=-1)
print(output.shape, sentence.shape)  # [2, 5, 8], [2, 8]
```

This runs an untrained forward pass to verify axes and valid lengths, not language competence or a quality benchmark.

## Sources

[PyTorch GRU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html) · [Speech and Language Processing — author-hosted draft](https://web.stanford.edu/~jurafsky/slp3/)
