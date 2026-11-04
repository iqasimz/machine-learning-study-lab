# Technical coverage — no silent gaps

Each row links a topic to an exercise and a follow-up question. P01–P16 refer to the [practical assessments](PRACTICAL_ASSESSMENTS.md), not the foundation lesson IDs. Training and application exercises are suggestions for practice, not completed work or separate projects bundled with this repository.

Depth: **core implement** = actual small code plus tests; **core analyse** = derivation/design plus a feasible measurement or simulation; **specialist** = literacy and scoped optional expansion. Unknown/new topics get added here rather than silently declared mastered.

## Foundations and software

| Topic | Theory home | Practical gate | Interview | Depth |
| --- | --- | --- | --- | --- |
| Python syntax, functions/classes, scope, mutability, exceptions, typing | programming foundations, P1 | L01/A01; P01 | F01/F02 | Core implement |
| Packaging, venv, imports, dependencies, testing, debugging | P1 | L01/A02; P01/P16 | F01/F13 | Core implement |
| DSA: hash maps, lists, sets, queues/heaps, sort/search, trees/graphs, complexity | programming foundations, P1 | P01/P02; BFS/top-k exercises | F03/F04 | Core implement |
| Linear algebra, broadcasting, chain rule, optimization | P2/P5 | L02/L04; P03/P04 | F05/F06, M04/M05 | Core implement |
| Probability, entropy, NLL/KL, Bayes, calibration | P3 | L01/L08; P03/P07 | F07/F08, M11/M12 | Core analyse |
| Generalization, bias/variance, leakage, metrics, bootstrap/Holm | P3/P4/T8 | L10/A05; P07 | F07/F08/F14, M16/S10 | Core implement |
| Classical ML: linear/logistic models, trees/RF, neural baselines | P4/P5 | Small baseline comparison, P07 | F14 | Core analyse |
| NLP tasks, TF-IDF/BM25, context, sequence models, multilingual ambiguity | P6/P7 | A03/A05; P08 | F15/S02 | Core implement |
| HTTP/DNS/TLS, API contracts, streaming, timeouts, retries | P1/R5/R8 | A02/A04; P09 | F10, S11 | Core implement |
| Processes/threads/async, workers, cancellation, race conditions | P1/R6 | A02/A08; P09/P11 | F11 | Core implement |
| SQL tables/joins/indexes, transactions/isolation, migrations | P1 + F12 | A02/A06; P10 | F12/S07/D02 | Core implement |
| Shell/Linux/SSH/Git, permissions, logs, SLURM, safe secrets | P1/T7 | P16; local tasks then authorised cluster | F13 | Core implement, cluster conditional |
| Product framing, baselines, ownership and scientific communication | P4/T8/R8 | Every milestone; P15 | F16/D01 | Core analyse |

## Model lifecycle

| Topic | Theory home | Practical gate | Interview | Depth |
| --- | --- | --- | --- | --- |
| Causal/masked/encoder–decoder objectives, targets/teacher forcing | T1 | L03/L04; P04/P05 | M01/M03 | Core implement |
| BPE/bytes/Unicode/templates/packing/masks/dedup/rights | P6/T2 | L03; P02/P05 | M02/M03 | Core implement |
| Attention/QKV/heads/residuals/norm/MLP/positions | T3/T4 | L04; P04 | M04 | Core implement |
| Training loop, AdamW/LR/precision/clipping/accumulation/resume | P5/T5 | L05; P05/P06 | M05/M06 | Core implement |
| Params/FLOPs/activations/optimizer/KV/resources | T6/R6 | L06/L09; P03/P13 | M07/M15 | Core analyse + measure locally |
| GQA/MQA/RoPE/MoE and dense alternatives | T4 | L06 one feasible controlled change | M08 | Core analyse, optional variant code |
| DP/FSDP/ZeRO/TP/PP/EP/communication/kernels/profiling | T7 | L06, small communication simulation; real jobs conditional | M07/M08/D04 | Core analyse; hardware evidence separate |
| Prompting, continued pretraining, full SFT/LoRA/QLoRA | R1 | L07; P05/P12 | M09/M10 | Core implement, CUDA QLoRA conditional |
| Preferences/reward model/reference/critic/advantage/KL | R2 | L08; P12 | M11/M12 | Core derive and implement small objectives |
| DPO | R2 | L08; P12, real small adapter run after preflight | M12 | Core implement |
| Policy gradient/PPO | R2 | L08; P12 toy policy | M11/M13 | Core small implementation; full LLM PPO specialist |
| GRPO/RLVR/reward hacking, equal rewards, sampling baselines | R2/R3 | L08 numerical + optional approved LLM run | M13 | Core analyse; hardware-gated run |
| Decoding/reasoning/test-time budgets/structured generation | R3 | L09/A04; P03/P05 | M14/S05 | Core implement |
| Quantization/distillation/pruning/edge | R7 | L09; P13 same-model comparison | M15/D03 | Quantization core; distillation/pruning scoped |
| Evaluation/transfer/judges/calibration/abstention | T8/R8 | L10/A05; P07/P08 | M16/S10 | Core implement |

## LLM application and system design

| Topic | Theory home | Practical gate | Interview | Depth |
| --- | --- | --- | --- | --- |
| Ingestion/chunking/spans/versions/withdrawal/citations | R4 | A03; P08 | S01/S03 | Core implement |
| Lexical/dense/hybrid/reranking/ANN/vector DB/ACL | P7/R4 | A05; P08 | S02/S04 | Core implement; ANN internals analyse |
| Context budgeting, retrieval memory vs weights, freshness | R3–R5 | A04/A07; P02/P08 | S05/S09 | Core implement |
| Workflows/agents/multi-agent, typed tools, MCP boundaries | R5 | A06/A07; P09/P11 | S06/S08 | Core implement; multi-agent experimental |
| Persistent approvals, idempotency, outbox/reconciliation | R5/R8 | A06; P10/P11 | S07/D02 | Core implement local, external design |
| Authentication/authorization, tenant isolation, secrets, injection, SSRF | R5/R8 | A02/A06; P11 | S08/D06 | Core implement controls; not security certification |
| Model gateway/routing/capabilities/provider rates/budgets | R5/R6 | A04/A08; P09/P14 | S11/D05 | Core implement |
| Prefill/decode/KV, batching/caching/speculation, load/queues | R6 | A08; P03/P13/P14 | M15/S11/D03 | Core analyse + local measurement |
| SLO/SLI, p95, reliability, observability, quality drift | R6/R8 | A08/A09; P14 | S12/D05 | Core implement locally |
| CI/Docker/rollback, artifact compatibility, backups/restore | R7/R8 | A09; P16 | S12/D06 | Core implement local; server/k8s conditional |
| Capacity/cost, SQL/cache/queue choices, failure domains | R6/R8 + casebook | P14/P15 | D01–D06 | Core design and local fault tests |
| Model handoff, registry/cards/licenses/lineage | T8/R8 | L12/A10; P16 | M16/D06 | Core implement contract later |

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
