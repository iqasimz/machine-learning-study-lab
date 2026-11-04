# DL07. Recurrent networks, GRUs and LSTMs

A recurrent neural network (RNN) processes an ordered sequence by updating a state at each position. The same cell weights are reused over time, so the current output can depend on earlier inputs. A vanilla RNN has one hidden-state update; a gated recurrent unit (GRU) controls how new and old states are mixed, and long short-term memory (LSTM) additionally maintains a separate cell-memory state.

The inputs are [1,0,1]. We calculate three state updates with fixed teaching weights. The GRU and LSTM gates are held constant to expose the arithmetic; real gates are computed from learned transformations of inputs and previous states.

This is the DL overview of shared weights and sequence history. The NLP course provides the detailed RNN/BiRNN, GRU/BiGRU and LSTM/BiLSTM gate/state walkthroughs, so their full architecture explanations are not duplicated here.

[course map](../COURSE_MAP.md). Required background: DL02, ML13.

## 1. State, gates and allowed sequence history

**Key terms.** A **recurrent state h_t** carries sequence information; **BPTT** backpropagates through time; **gates** modulate information flow. **LSTM** adds a cell state and gates; **GRU** uses a different gated update; **bidirectional** models also read later input.

**Process.** Read an input step, update hidden state and produce an output or continue to the next step. During training, propagate error backwards through a declared sequence length; at prediction, preserve only state allowed by the actual task.

A simple RNN shares its transition weights over time. Repeated derivatives can shrink or grow, complicating long-range fitting. LSTM/GRU introduce controlled paths, but do not guarantee unlimited memory. Truncated BPTT bounds derivative history and memory; carrying state and carrying gradients are separate decisions.

For a complete document classification task, bidirectional context may be available. For causal forecasting or streaming generation, future observations are not. Variable-length batches need padding/length handling, and document boundaries must not accidentally share hidden state. Masked losses and allowed information should be audited just as in the LLM course.

## 2. A worked calculation

**Symbols and assumptions.** A simple recurrence is h_t=tanh(x_t W_x+h_(t−1)W_h+b). In an LSTM cell-path illustration, c_t=f_t c_(t−1)+i_t g_t, where f/i are forget/input gates and g is proposed content; other gates determine the output.

Holding illustrative gates fixed at f=0.9,i=0.2 with previous cell c=2 and candidate g=0.5 gives new cell 1.9. The direct cell-path derivative is 0.9 under that fixed-gate calculation, not the entire network derivative. Repeated simple-path multipliers 0.5 over 10 steps give 0.0009766; multipliers 2 give 1,024.

The numbers above are illustrative calculations, not measured benchmark results.

## 3. Recurrence versus other sequence models

Lagged linear/boosted models, temporal convolutions, gated recurrence and attention represent history differently. More recurrent width increases capacity/cost; longer derivative windows increase training memory. Use a causal baseline before claiming a recurrent forecast gain.

The related settings are defined below this lesson in Model Lab, including their mathematical meaning, expected effects and interactions. The training lab exposes only controls that change an implemented operation; the broader [parameter catalogue](../metadata/parameters.json) also describes theory-only options.

## 4. Example: valid forecasting inputs

Compare a lagged demand predictor with a small GRU on the same bike forecast origins. Calendar/feature availability and forecast horizon must match; bidirectional access to the future is not a legitimate advantage.

**Check.** May you keep hidden state from the final training row when starting an unrelated shuffled test document?

**Answer.** Not under an independent-document protocol. Reset or initialize state according to the task and record any legitimate streaming carry-over explicitly.

## 5. GRU gate convention and gradient history

**Key terms.** A hidden state summarizes earlier positions. A GRU reset gate controls use of the previous state in the candidate; an update gate mixes old and candidate state. Gate conventions differ between formulations.

**Process.** At each position, combine the new input with the previous hidden state, calculate gates, form a candidate and mix it into the state. The next position receives the updated state; the task head reads a declared state or sequence.

This course uses the PyTorch-style update convention h_t=(1−z_t)h̃_t+z_t h_(t−1), so z weights the old state. With old state 0.8, candidate 0.2 and z=0.25, the result is 0.75×0.2+0.25×0.8=0.35. A larger z preserves more of the old state under this convention. Some references instead place z on the candidate; their numerical gate value has the opposite interpretation. The live DL and NLP examples use the same old-state convention.

### Backpropagation through time

Unrolling makes repeated applications of shared weights visible to autograd. Their gradient contributions add across time. A state Jacobian describes how one state changes with the previous state's entries. Multiplying many such local derivatives can shrink or amplify the gradient over a long sequence. Truncated backpropagation detaches the carried state at boundaries: it preserves a forward summary but cuts gradient paths across those boundaries.

A bidirectional encoder can read future positions, which is useful for complete-input tagging but invalid for a causal next-token task. Sequence architecture is taught here; this release does not include an executable recurrent trainer.

## Sources

- [PyTorch 2.9 GRU equations and implementation convention](https://docs.pytorch.org/docs/2.9/generated/torch.nn.GRU.html).

- [Deep Learning, author site](https://www.deeplearningbook.org/).
- [Stanford CS230, Fall 2026 syllabus](https://cs230.stanford.edu/syllabus/).
- [Dive into Deep Learning](https://d2l.ai/).
