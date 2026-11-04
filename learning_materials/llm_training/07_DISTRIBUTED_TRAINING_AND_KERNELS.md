# T7. Distributed LLM training and gradient reduction

Distributed training divides data, arithmetic or stored state across cooperating processes. Data parallelism uses model replicas on different examples and combines gradients; tensor, pipeline and state-sharding schemes divide other objects. Correct loss normalization and communication are required before more devices can be treated as a performance improvement.

Two data-parallel workers have unequal valid-token counts and supplied scalar mean gradients one and three. We calculate the intended global token-weighted gradient, then the local scaling needed when a reducer averages across the two workers. This is not an executed cluster job.

Microbatch accumulation combined evidence on one worker. This lesson adds communication and a reducer convention across workers; multiplying device counts alone does not establish either a correct gradient or a speedup.

## 1. Parallel work and communication

**Key terms.** A **worker** is a training process; **communication** moves tensors between workers; **synchronization** makes processes wait for shared progress.

**Process.** Establish a correct one-worker run, measure its bottleneck and only then divide work. Additional devices help only when useful parallel work exceeds communication and coordination overhead.

Each accelerator has finite memory and compute. Distributing work creates communication, synchronization and scheduling costs. If one worker is late, others may wait. If interconnect bandwidth is poor, sharding can save memory while losing throughput.

First distinguish **scale up**—use faster/larger resources—from **scale out**—add workers. Strong scaling fixes total work and adds workers; weak scaling increases work with resources. Their efficiency numbers answer different questions.

## 2. Parallelism families

**Key terms.** **Data parallelism** splits examples; **tensor parallelism** splits tensor operations; **pipeline parallelism** splits layers; **sharding** partitions stored state.

**Process.** Choose what each device owns, execute its local computation and exchange the required tensors. Reconstruct the intended global update before judging whether the distributed run is faster.

```text
Data parallel:       GPU 1 = full model, batch A
                     GPU 2 = full model, batch B → combine gradients

Tensor parallel:     GPUs hold slices of a layer → exchange partial results

Pipeline parallel:   layers 1–k on GPU 1 → layers k+1–L on GPU 2

Expert parallel:     experts on different GPUs → route token states between them
```

**Data parallelism** replicates the model and aggregates gradients. An all-reduce combines a value across workers and distributes the result. An all-gather assembles shards; reduce-scatter reduces values while distributing result shards.

**ZeRO/FSDP-style sharding** partitions optimizer state, gradients and/or parameters depending on the scheme. Parameters may be gathered for computation then resharded. This is not identical to tensor parallelism, which splits the layer's arithmetic itself.

**Tensor parallelism** splits matrix operations; it usually needs frequent fast communication. **Pipeline parallelism** splits layers and moves activations; microbatches reduce idle “bubbles.” **Expert parallelism** distributes MoE experts; routing can require all-to-all traffic. **Sequence/context parallelism** partitions sequence-related work or state; exact strategies differ by framework.

Combine these only after specifying the mapping. “8 GPUs with TP=2 and DP=4” describes a different execution than “DP=8.” Do not multiply every advertised parallelism dimension without checking whether groups overlap or are nested.

## 3. Effective gradients and reproducibility

**Key terms.** **DDP** is distributed data parallelism; **W** counts workers; **Sᵣ** is worker r's summed loss; **N** is the global valid-target count.

**Process.** Each worker computes its local loss contribution. Scale for the framework's gradient reduction, synchronize and apply the same global update. Unequal token counts require more care than averaging local means.

**Calculation.** For two workers with 10 and 90 valid targets and local mean losses 2 and 4, the global token mean is (10×2+90×4)/100=3.8, not (2+4)/2=3.0. With DDP gradient averaging, local losses scaled as W×local loss sum/N have numerical values 0.4 and 7.2; their average is 3.8 and their averaged gradients match the intended objective.

Verify whether gradients are summed or averaged and over which workers. The loss denominator must match the intended global valid-token objective. Each worker needs a different data shard; otherwise eight workers may process eight copies of the same examples. Save sampler and RNG states; a seed alone does not describe the full run.

Distributed failures include mismatched collective calls, a worker exiting while others wait, network timeouts, CPU data-loading stalls and inconsistent checkpoints. Diagnose the first failing worker's log, not only the final timeout message.

### Derive the loss scale under ordinary DDP averaging

Let rank `r` own token-loss sum `S_r` and valid-target count `N_r`. The intended global loss is:

\[
L=\frac{\sum_r S_r}{N},\qquad N=\sum_r N_r.
\]

For `W` workers and the ordinary DDP convention that averages parameter gradients across workers:

1. All-reduce the target counts to obtain the same global denominator `N` on every worker.
2. Each worker differentiates `W S_r/N`. After DDP's averaging, the combined gradient is `(1/W)Σ_r∇(W S_r/N)=∇L`.
3. Differentiating only `S_r/N` and then averaging shrinks the gradient by `W`. Averaging each worker's mean loss instead can give unequal-sized workers the wrong relative influence.
4. With accumulation, count the full update's intended targets. Suppress synchronization for appropriate intermediate microbatches only using a correct framework-supported pattern, then synchronize at the boundary.

Communication hooks, uneven-input handling and other frameworks can use different reduction conventions. Confirm the installed implementation rather than copying the world-size factor blindly. Start with a one-worker versus two-worker toy equivalence test, fixed data and disabled dropout. References: [PyTorch DDP](https://docs.pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html) and [ZeRO](https://arxiv.org/abs/1910.02054).

## 4. Kernels, fusion and FlashAttention

**Key terms.** A **kernel** is a device computation; **fusion** combines operations; **FlashAttention** reorganizes exact attention to reduce memory traffic.

**Process.** A framework operation selects a compatible kernel, which moves data and computes outputs. Check output/gradient agreement before attributing a speedup to better execution rather than a changed calculation.

A high-level attention expression may create enormous intermediate tensors. A kernel implements the arithmetic on hardware. **Fusion** combines operations to avoid writing/reading temporary results and reduce launch overhead.

FlashAttention reorganizes exact attention with tiling and stable online softmax, reducing transfers to/from high-bandwidth memory and avoiding materialization of the full score matrix in the usual form. It is not the same as sparse or approximate attention. Exact refers to the mathematical operation, subject to floating-point differences. Pairwise arithmetic still has quadratic sequence-length dependence for ordinary dense attention.

The crucial distinction: **algorithmic complexity**, **memory footprint**, **memory traffic** and **wall-clock time** are separate. A kernel can substantially speed up the same mathematical operation.

## 5. Cluster workflow

**Key terms.** A **login node** is the cluster entry machine; an **allocation** grants resources; a **job script** describes scheduled work; **SLURM** manages the queue.

**Process.** Connect by SSH, prepare the environment, submit a resource request and run on the assigned compute node. Collect logs and checkpoints before the allocation ends; SSH alone does not reserve a GPU.

```text
laptop → SSH → login node → submit scheduler request
→ allocated compute node(s) → distributed launcher → worker processes
→ logs, checkpoints and metrics → evaluate → release allocation
```

Read your cluster's rules first. Commands such as `squeue`, `sacct`, `sinfo` and `nvidia-smi` are read-only diagnostics where installed. `sbatch` submits a job; it is not required for every GPU system. A scheduler allocation and the Python distributed process group are separate layers.

Never kill other users' processes or run network-wide scans to debug your experiment. Keep credentials out of job scripts and logs. Use authorised data storage and obey quotas.

## 6. Profile before optimization

**Key terms.** A **profile** records where time or memory goes; **warmup** removes startup effects; **synchronization** can be needed to time completed accelerator work.

**Process.** Measure a representative baseline, isolate costly operations, change one bottleneck and profile again. Confirm numerical correctness alongside time and memory rather than optimizing a misleading timer.

Measure data-loading time, forward/backward time, optimizer time, communication, checkpoint writing and memory peaks. Accelerator operations may be asynchronous: naive wall-clock timing without synchronization can measure launch time instead of completion. Include warmup and repeated measurements; state batch/context/dtype and hardware.

## Exercises and checks

1. Microbatch 2, accumulation 8, DP 4: nominal effective example batch is 64. TP workers do not multiply the count of distinct examples in the same way.
2. A model fits only after FSDP but becomes slower. Is FSDP broken? **Not necessarily:** communication or offload may dominate; fitting and speeding up are different goals.
3. Explain why FlashAttention and PagedAttention solve different problems. **Answer:** attention IO/computation organization versus inference KV-cache memory management; see R6.

Primary sources: [FlashAttention](https://arxiv.org/abs/2205.14135), [NVIDIA parallelisms guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html), [PyTorch FSDP documentation](https://docs.pytorch.org/docs/stable/fsdp.html). APIs are version-sensitive: use docs matching your installed build.
