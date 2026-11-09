# Python and code for machine learning

These are the authored sources for the website's **Python** course: 95 lessons across eight sections. Open Python in the sidebar, or choose Python in Interactive mode. Both display the same code, recorded outputs, explanations and references. Read the chapters in order, or use the prerequisite notes to enter at the right point.

| Chapter | Contents |
| --- | --- |
| [01 — Python basics](01_PYTHON_BASICS.md) | PY01–PY14: reading code, values, data structures, loops, functions, copies, generators, text and bytes, errors, environments, classes and decorators, tests and a complete report. Each idea is shown in a small example before a later example combines it. |
| [02 — Data and plotting](02_DATA_AND_PLOTTING.md) | DS01–DS10: NumPy arrays, indexing, axes and broadcasting; pandas tables, selection, grouping and joins; Matplotlib and one diagnostic heatmap; a checked summary. |
| [03 — Machine learning code](03_MACHINE_LEARNING.md) | SK01–SK12: estimators, pipelines, regression, thresholds, neighbours, trees, boosting, splits, clustering, calibration, time and ranking, then the existing ML lab's fitting workflow. |
| [04 — Deep learning code](04_DEEP_LEARNING.md) | PT01–PT12: why PyTorch is the shared library, then tensors, gradients, modules, batches, the training-loop order, losses, convolution, recurrence, attention, generative objectives, graphs and saved weights. |
| [05 — NLP code](05_NLP.md) | NC01–NC08: tokenization and token IDs, word counts, embeddings, masks, label alignment, retrieval and evaluation of fixed predictions. |
| [06 — LLM code](06_LLMS.md) | LC01–LC10: model, weights and tokenizer; labels; the scratch training loop; LoRA; preference signals; retrieval; validating actions; service boundaries; memory and system evaluation. |
| [07 — Other libraries](07_OTHER_LIBRARIES.md) | OL01–OL09: logging and progress, profiling, SQLite, Parquet with DuckDB, tensor rearrangement, safetensors, experiment records, property tests and graph objects. |
| [08 — Data structures and algorithms](08_DSA.md) | DA01–DA20: contracts and cost, lists, hash tables, sorting, binary search, two pointers, prefixes and windows, stacks and links, recursion, backtracking, trees and tries, heaps, graph search, topological order and Dijkstra, edit distance, zero-one knapsack, greedy choices, a bounded stream, tests, and a three-job capstone. |

Every runnable teaching example has its own input and a predetermined output. A printed label can be an explicit rule's decision or a constructed prediction fixture, not necessarily a model's learned prediction. The surrounding explanation tells you which it is. Numerical miniatures teach mechanisms; they do not establish real-world model quality.

Complete training-loop walkthroughs are the exception: read the code and explanation, then use the appropriate **existing local lab** to run a supported experiment. The course does not duplicate training controls, training logs, learning curves or lab result tables. There is no Python runner or notebook in this material.

The existing NLP course has no dedicated NLP trainer. The current DL lab does not train every architecture explained here. A checked forward calculation is not a claim that a corresponding training interface exists.

## How to study a code example

Read the problem and input first. Predict what the code will print, then compare with the displayed output. Read the explanation of how it got there. Finally, change one small condition in your head and answer the question below the example. Looking up syntax is normal; being able to explain what changes is the aim.

Python code and printed output have separate fences. Shell commands are labeled separately. Each small example is independent unless a walkthrough explicitly states its shared inputs. The output verifier executes the exact marked snippets from these chapters in isolated temporary working directories; it is a maintainer tool, not a website feature.

## Maintainer notes

- [Sources and editorial rules](SOURCES.md) record source selection and licensing boundaries.
- `maintenance/check_examples.py` verifies marked non-training snippets, lesson IDs, source presence and output matches. Its generated report records the environment and any failures.
- `maintenance/check_structure.py` checks chapter/evidence fingerprints, local links, theory IDs and direct-package records without executing the examples again.
- Reading or checking this pack must not train a model, download a model/corpus or call a paid API. The website reads these sources without executing their examples.
- Core numerical examples use CPU. There is no promise that a downloaded model or GPU produces bit-identical outputs.

**Study status:** integrated lessons, not evidence that a learner has completed or mastered the course. See the verification report for execution evidence; do not infer it from an expected-output block alone. Interactive mode adds one explanation-based check per lesson; practice history stays in the visitor's browser.

## Rechecking the source pack

Use a separate Python 3.11 example environment with the versions in `requirements-examples.txt`. That file records direct dependencies, not a complete lock. The normal check command, from this folder, is:

```bash
python maintenance/check_examples.py --record --model-dir /path/to/reviewed/local/model
python maintenance/check_structure.py
```

The pretrained example additionally requires the matching prepared-asset receipt in the lab's `assets/receipts/model.json` layout. For the original local project layout, the checker detects the existing prepared model by default. No model is downloaded if it is absent; that example fails rather than substituting invented text. The website displays the recorded output; visitors do not need to run this maintainer check. After a code/output edit, recheck the examples before serving: the reader validates their recorded hashes. Static exports include the two checked plot images, but no weights or training runs.
