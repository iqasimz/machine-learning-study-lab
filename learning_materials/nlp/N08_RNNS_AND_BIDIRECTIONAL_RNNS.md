# N08. Recurrent and bidirectional recurrent networks

A recurrent neural network, or RNN, processes a sequence by updating a hidden state at each position. The state combines the current token with information carried from earlier positions. A bidirectional RNN adds a separate right-to-left pass, allowing each aligned token representation to use both sides of an already available input.

We use three scalar inputs [1,0,−1] and one hidden feature per direction. The forward and reverse passes have different supplied weights. Select a token to inspect its aligned states, or switch to forward-only mode to remove future-context access.

A text CNN reads local windows. An RNN reuses a transition along the sequence instead; bidirectionality changes which input context is available, not how backpropagation computes gradients during training.

## 1. Recurrent equations and directional context

A vanilla recurrent network applies \(h_t=\tanh(W_xx_t+W_hh_{t-1}+b)\). The hidden state has a fixed width even as the sequence grows. The recurrent parameters are shared across positions. Backpropagation through time differentiates the unrolled graph; it does not create a separate model at every token.

A BiRNN has independent forward and backward parameter sets. The forward state at token t reads its prefix; the backward state reads its suffix. After computing in opposite orders, align both states to the original token index and concatenate them. The worked example deliberately uses different directional input/recurrent weights, so it cannot be mistaken for one shared pass drawn twice.

Because a backward state reads later input tokens, this representation suits a fully observed sentence but cannot be used unchanged for causal next-token prediction or streaming where the suffix is unavailable.

## 2. Padding, terminal states and gradient paths

A bidirectional sequence output has width 2H. Its forward part at the final valid token is a terminal summary, but its backward part there has only read the final token. For a terminal sentence representation, concatenate the forward terminal state with the backward terminal state aligned to the first valid token. Alternatively use a documented masked pooling scheme.

Padding must not become a meaningful suffix. PyTorch packed sequences or correct masking/length handling avoid that failure. Truncated BPTT cuts gradient history; carrying a detached state preserves values, not a gradient path to all earlier steps. Vanishing/exploding gradients motivate gated cells, but a BiRNN itself is not a solution to gradient instability.

### Why a long gradient path can shrink

For the scalar forward example, the local derivative of h at t with respect to the earlier hidden state is \(0.8(1-h_t^2)\). Across several positions, one direct recurrent path multiplies these factors. Repeated small factors can shrink that path's gradient; large recurrent Jacobians in wider models can contribute to explosions. Other input/output paths and nonlinear states still matter. A gate or a bidirectional pass is not a proof that this problem disappears.

## Sources

[PyTorch RNN](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html) · [Speech and Language Processing — author-hosted draft](https://web.stanford.edu/~jurafsky/slp3/)
