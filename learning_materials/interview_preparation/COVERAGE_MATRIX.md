# Interview topic coverage

Each row connects a topic with a course lesson and follow-up questions. Some older question IDs resolve to a shared topic in the website. Practical exercises are suggestions for study, not evidence of completed work.

Depth: **core implement** = actual small code plus tests; **core analyse** = derivation/design plus a feasible measurement or simulation; **specialist** = literacy and scoped optional expansion. Unknown/new topics get added here rather than silently declared mastered.

## Foundations and software

| Topic | Theory home | Interview | Depth |
| --- | --- | --- | --- |
| Python syntax, functions/classes, scope, mutability, exceptions, typing | Python basics | F01/F02 | Core implement |
| Packaging, venv, imports, dependencies, testing, debugging | Python basics | F01/F13 | Core implement |
| DSA: hash maps, lists, sets, queues/heaps, sort/search, trees/graphs, complexity | Python basics | F03/F04 | Core implement |
| Linear algebra, broadcasting, chain rule, optimization | P2/P5 | F05/F06, M04/M05 | Core implement |
| Probability, entropy, NLL/KL, Bayes, calibration | P3 | F07/F08, M11/M12 | Core analyse |
| Generalization, bias/variance, leakage, metrics, bootstrap/Holm | P3/P4/T8 | F07/F08/F14, M16/S10 | Core implement |
| Classical ML: linear/logistic models, trees/RF, neural baselines | P4/P5 | F14 | Core analyse |
| NLP tasks, TF-IDF/BM25, context, sequence models, multilingual ambiguity | P6/P7 | F15/S02 | Core implement |
| HTTP/DNS/TLS, API contracts, streaming, timeouts, retries | Python basics/R5/R8 | F10, S11 | Core implement |
| Processes/threads/async, workers, cancellation, race conditions | Python basics/R6 | F11 | Core implement |
| SQL tables/joins/indexes, transactions/isolation, migrations | Python basics + F12 | F12/S07/D02 | Core implement |
| Shell/Linux/SSH/Git, permissions, logs, SLURM, safe secrets | Python basics/T7 | F13 | Core implement, cluster conditional |
| Product framing, baselines, ownership and scientific communication | P4/T8/R8 | F16/D01 | Core analyse |

## Model lifecycle

| Topic | Theory home | Interview | Depth |
| --- | --- | --- | --- |
| Causal/masked/encoder–decoder objectives, targets/teacher forcing | T1 | M01/M03 | Core implement |
| BPE/bytes/Unicode/templates/packing/masks/dedup/rights | P6/T2 | M02/M03 | Core implement |
| Attention/QKV/heads/residuals/norm/MLP/positions | T3/T4 | M04 | Core implement |
| Training loop, AdamW/LR/precision/clipping/accumulation/resume | P5/T5 | M05/M06 | Core implement |
| Params/FLOPs/activations/optimizer/KV/resources | T6/R6 | M07/M15 | Core analyse + measure locally |
| GQA/MQA/RoPE/MoE and dense alternatives | T4 | M08 | Core analyse, optional variant code |
| DP/FSDP/ZeRO/TP/PP/EP/communication/kernels/profiling | T7 | M07/M08/D04 | Core analyse; hardware evidence separate |
| Prompting, continued pretraining, full SFT/LoRA/QLoRA | R1 | M09/M10 | Core implement, CUDA QLoRA conditional |
| Preferences/reward model/reference/critic/advantage/KL | R2 | M11/M12 | Core derive and implement small objectives |
| DPO | R2 | M12 | Core implement |
| Policy gradient/PPO | R2 | M11/M13 | Core small implementation; full LLM PPO specialist |
| GRPO/RLVR/reward hacking, equal rewards, sampling baselines | R2/R3 | M13 | Core analyse; hardware-gated run |
| Decoding/reasoning/test-time budgets/structured generation | R3 | M14/S05 | Core implement |
| Quantization/distillation/pruning/edge | R7 | M15/D03 | Quantization core; distillation/pruning scoped |
| Evaluation/transfer/judges/calibration/abstention | T8/R8 | M16/S10 | Core implement |

## LLM application and system design

| Topic | Theory home | Interview | Depth |
| --- | --- | --- | --- |
| Ingestion/chunking/spans/versions/withdrawal/citations | R4 | S01/S03 | Core implement |
| Lexical/dense/hybrid/reranking/ANN/vector DB/ACL | P7/R4 | S02/S04 | Core implement; ANN internals analyse |
| Context budgeting, retrieval memory vs weights, freshness | R3–R5 | S05/S09 | Core implement |
| Workflows/agents/multi-agent, typed tools, MCP boundaries | R5 | S06/S08 | Core implement; multi-agent experimental |
| Persistent approvals, idempotency, outbox/reconciliation | R5/R8 | S07/D02 | Core implement local, external design |
| Authentication/authorization, tenant isolation, secrets, injection, SSRF | R5/R8 | S08/D06 | Core implement controls; not security certification |
| Model gateway/routing/capabilities/provider rates/budgets | R5/R6 | S11/D05 | Core implement |
| Prefill/decode/KV, batching/caching/speculation, load/queues | R6 | M15/S11/D03 | Core analyse + local measurement |
| SLO/SLI, p95, reliability, observability, quality drift | R6/R8 | S12/D05 | Core implement locally |
| CI/Docker/rollback, artifact compatibility, backups/restore | R7/R8 | S12/D06 | Core implement local; server/k8s conditional |
| Capacity/cost, SQL/cache/queue choices, failure domains | R6/R8 + casebook | D01–D06 | Core design and local fault tests |
| Model handoff, registry/cards/licenses/lineage | T8/R8 | M16/D06 | Core implement contract later |

## Specialist ledger — explicitly not all implemented by the core

| Family | Minimum coverage | Optional deeper practical work | Completion boundary |
| --- | --- | --- | --- |
| Multimodal/VLM/audio/video | T4: modality inputs, fusion, alignment, data/compute issues | Synthetic screenshot/table retrieval | Not training a full speech/vision foundation model |
| State-space/hybrids/diffusion language | T4/R9: mechanism/trade-offs | Tiny architecture experiment | No claim to reproduce frontier architectures |
| GraphRAG/late interaction/multi-hop | P7/R4: when relationships/retrieval stages help | Compare on a labelled multi-hop subset | Not “graphs always improve RAG” |
| Custom CUDA/Triton/advanced distributed | T7: bandwidth/communication/kernel limits | Authorised GPU profile and kernel microbenchmark | Simulation cannot prove cluster throughput |
| Interpretability/patching/SAE/steering | T8/R9: probe versus intervention, causal controls | One small measured intervention | Not discovery of a reliable safety circuit |
| Editing/merging/continual learning | R9: locality/forgetting/conflicts | One scoped comparison | No universal knowledge-edit guarantee |
| Privacy/DP/unlearning/federated | R9: threat/guarantee/limits, weight versus retrieval deletion | Small formal/mechanism exercise | No privacy certification/accountant proof from a demo |
| Full RLHF/PPO and alternative preference objectives | R2: roles/objectives and assumptions | Budgeted LLM critic/reward loop, one loss expansion | DPO or toy PPO is not full RLHF implementation |
| Mobile/embedded/real cloud/Kubernetes | R7/R8: kernels, thermals, updates, HA/limits | Run on actual approved device/deployment | Mac demo/YAML is not mobile or HA evidence |
| Arbitrary code sandbox/browser automation | R5/R8: isolation and authority | Reviewed Linux sandbox or synthetic UI | No host shell or access-control bypass |
| Consensus/multi-region storage/formal security | F12/D06/R8: why local guarantees do not generalize | Dedicated systems course | Not covered to implementer depth here |

For a new technique, fill: problem → input/data → objective → changed component → compute → baseline → failure mode → test. Add a worked exercise if it becomes relevant. That method is more durable than claiming an exhaustive list of acronyms.
