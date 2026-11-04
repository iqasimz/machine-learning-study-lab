# P1. Computers, tensors and software

## 1. Programs, processes and hardware

**Key terms.** A **process** is a running program; a **checkpoint** stores model weights; a **framework** such as PyTorch supplies tensor operations.

**Process.** Your command starts a Python process. It loads the checkpoint, prepares input arrays and asks the hardware to calculate an output. The following sections explain the environment and arrays this requires.

A model is not an application by itself. Its checkpoint contains learned numbers. A program loads those numbers, turns an input into arrays, calculates an output and returns it. The operating system manages the running program, or **process**. A process has memory, open files, permissions and an exit status.

```text
you → terminal → shell → Python process → framework → CPU/GPU
                            │                  │
                       files/network       tensors + kernels
```

The **terminal** is the window for text input/output. The **shell**, often Bash or Zsh, interprets commands. **Linux** is an operating-system family, not another programming language. **SSH** securely connects you to a remote machine. A **server** is a machine or process providing a service. A **GPU kernel** is a function executed on the accelerator; it is not the operating-system kernel.

**SLURM** is a scheduler on some clusters: request resources, wait for an allocation, then run a process on a compute node. Logging into a GPU cluster does not mean your current process has a GPU allocation. Do not run training on a login node unless the cluster explicitly permits it.

## 2. Programs and environments

**Key terms.** An **interpreter** executes Python; a **dependency** is a package the program needs; an **environment** selects their versions.

**Process.** Before a model can run, the shell must invoke the right interpreter. That interpreter imports its packages and executes the program. Check this chain before diagnosing model behaviour.

Python is dynamically typed: a name can refer to different kinds of objects, but an operation still requires compatible types. `"3" + "4"` concatenates strings; `3 + 4` adds numbers. A function defines a reusable mapping; a class packages state and operations. Neither classes nor abstraction make an algorithm correct automatically.

An **environment** is a chosen interpreter and package set. A virtual environment isolates Python packages; it does not emulate an entire computer or install an NVIDIA driver. A container packages userspace software; GPU access still depends on the host driver/runtime. Record dependencies and hardware separately.

Illustrative local setup, only when you are ready to code:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip --version
```

Use `python -m pip` to target the interpreter you actually invoked. Do not copy environment commands into a random project or overwrite your existing environment blindly.

## 3. Tensors and shape

**Key terms.** A **shape** lists an array's axis sizes; **dtype** specifies how values are represented; **device** specifies where they are stored and computed.

**Process.** Text IDs enter as integers. Embedding lookup turns each ID into a vector, adding a feature axis. Check shape, dtype and device before feeding this tensor to the next layer.

A **scalar** is one number. A **vector** is a one-dimensional array. A **matrix** has two axes. A **tensor** generalises arrays to any number of axes. A token batch may be integer IDs of shape `[B,T]`; embeddings produce `[B,T,d]`.

- `B`: batch size, the number of examples processed together.
- `T`: sequence length, measured in tokens, not words.
- `d`: feature width, numbers representing each token.

If `B=2`, `T=3`, `d=4`, the embedding output holds 24 numbers. FP32 uses four bytes per value, so raw output storage is 96 bytes. Real training needs many other tensors too.

**Broadcasting** virtually expands compatible axes. Adding a bias `[d]` to `[B,T,d]` is intentional. Adding labels `[B,1]` to predictions `[B]` can unintentionally create `[B,B]`. Always write shapes before operations.

**Dtype** is the numeric representation: integer token IDs, floating-point weights, Boolean masks. Device placement is separate: a valid FP32 tensor can be on CPU or GPU. A GPU program may spend much of its time waiting for transfers or input data.

### From a tensor's shape to the memory it actually uses

Shape, storage and gradient history are separate. Consider an attention tensor rearranged from `[B,T,H,D]` to `[B,H,T,D]`.

1. A permutation changes the interpretation of strides: how far to move in storage along each axis. It can return a **view** without copying all values.
2. A later operation may require contiguous storage. `reshape` may then copy; do not assume that a cheap-looking reshape is always free.
3. `detach()` stops gradient tracking through that result but can still share storage. `clone()` copies values but normally preserves the differentiable connection. Use the operation that matches the intended semantics.
4. If a list retains loss tensors or full activations from every update, it can keep their computation graphs alive. Store detached summaries when you only need a log.
5. Accelerator work can run asynchronously. To measure completed computation, use the backend's timing/synchronization method, with warmup; timing only a Python call can measure enqueueing instead.

These distinctions explain otherwise mysterious memory growth and misleading throughput numbers. They also show why a tensor has more meaning than its printed shape. Official references: [PyTorch tensor views](https://docs.pytorch.org/docs/2.9/tensor_view.html) and [autograd modes](https://docs.pytorch.org/docs/2.9/notes/autograd.html).

## 4. Software structure and testing

**Key terms.** A **configuration** records chosen settings; an **artifact** is a saved output; a **test** checks an expected property.

**Process.** The loader supplies data to model code; configuration controls the run; reporting saves its evidence. Test these boundaries separately so a reporting bug is not mistaken for a model improvement.

Separate four layers: data loading; model computation; experiment configuration; reporting. Keep raw data immutable and derived datasets reproducible. Use source control for code, not secrets or enormous checkpoints. A checkpoint is not a substitute for code and configuration.

Use functions for operations such as `tokenize`, `compute_loss`, `evaluate`. Add tests for shapes, missing values, empty input, Unicode and masks. A **unit test** checks one operation; an **integration test** checks connected components. A **regression test** detects an unwanted change from previously accepted behaviour. Tests are an asset, not evidence of low-quality research.

An **exception** is a reported failure; inspect the traceback from the last relevant frame. A **log** records what happened. A **metric** aggregates a quantity. A **trace** links steps of one request. Never place access tokens or private document contents in unrestricted logs.

## 5. Time and memory complexity

**Key terms.** **Time complexity** describes how work grows with input size; **space complexity** describes how required storage grows.

**Process.** Choose the input size, estimate its growth in work and memory, then measure the actual program. These estimates help decide which operation to profile, not its exact runtime.

**Calculation.** For a full attention score array, doubling sequence length T from 128 to 256 changes T² from 16,384 to 65,536 entries per head: four times as many. This is a score-array count, not a universal fourfold wall-time prediction.

If doubling a list doubles a scan's work, it is roughly `O(n)`. Comparing every pair costs `O(n²)`. Exact full attention forms token-pair interactions, so sequence length can be a much stronger cost driver than a beginner expects. Big-O hides constants, hardware and memory traffic; it does not predict exact wall-clock time.

CPU RAM, GPU memory and disk are different budgets. A model can fit on disk but fail to load in GPU memory. A process can also be killed by host RAM exhaustion before GPU computation begins.

## 6. Reproducing a run on another machine

**Key terms.** A **revision** identifies a code or model version; a **seed** initializes random choices; the **working directory** anchors relative paths.

**Process.** When the same script behaves differently on another machine, compare its environment and inputs before changing its algorithm. Re-run a small known case after each isolated correction.

Check interpreter → dependencies → paths → permissions → model/tokenizer revisions → device/dtype → random seeds → network access. Do not immediately change the model. A relative path is resolved from the current working directory, not necessarily the script's directory. A random seed controls some randomness, not every parallel nondeterministic operation.

## Practice and answer check

1. For `[4,128,256]` in FP16, calculate raw bytes. **Answer:** `4×128×256×2 = 262,144` bytes, or 256 KiB.
2. Why might SSH succeed but training cannot see CUDA? **Answer:** you may be on a login/CPU node, have no GPU allocation, lack a compatible driver/build, or have an unavailable device. SSH only established a connection.
3. Explain virtual environment versus container versus GPU driver without using “they install everything.”

Visual reinforcement: follow the tensor and automatic-differentiation sections of [PyTorch basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html). Stop once you can predict tensor shape, dtype and device before executing an operation.
