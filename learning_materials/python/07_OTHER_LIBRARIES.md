# Other important libraries

Prerequisites: the Python basics through files, tests and configuration. Later lessons also use a table, a tensor, or both. This section is a set of worked tools, not a list of imports to memorize. A module in the standard library, a third-party package, a file format and a service are different things. Libraries already introduced earlier — NumPy, pandas, scikit-learn, PyTorch, spaCy, Transformers, Datasets, PEFT, HTTPX, FastAPI and Pydantic — stay in their own lessons.

## OL01. Logging

A long job needs two different records. A **log** is a line you can keep: which record was rejected, and why. A **progress bar** is a glance at how far the loop has gone. `logging` comes with Python. `tqdm` is a third-party package that draws the bar. People use the pair because printing every row buries the failure, and a bar that disappears tells you nothing the next morning.

Levels separate ordinary notes from failures. `INFO` is a note. `WARNING` and `ERROR` are for things that need attention. A log line belongs on a stream you chose. This lesson sends it to standard output so the checked text is visible. A progress bar is allowed to use a carriage return and a changing rate. Those details are not a durable record, so the checked lines below are the log line and the counters `tqdm` stored: how many steps finished, and how many there were. The text of a kept record is not logged.

<!-- example: OL01-main -->
```python
import io
import logging
import sys
from tqdm import tqdm

logger = logging.getLogger("lesson")
logger.handlers.clear()
logger.setLevel(logging.INFO)
logger.propagate = False
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
logger.addHandler(handler)

records = [
    {"id": "d1", "text": "kept"},
    {"id": "d2", "text": ""},
    {"id": "d3", "text": "kept"},
    {"id": "d4", "text": "kept"},
    {"id": "d5", "text": "kept"},
]
progress = tqdm(total=len(records), file=io.StringIO())
kept = 0
for record in records:
    if record["text"]:
        kept += 1
    else:
        logger.info("rejected id=%s reason=empty", record["id"])
    progress.update(1)
progress.close()
print(f"{progress.n}/{progress.total}")
print(kept)
```

**Output**
```text
INFO rejected id=d2 reason=empty
5/5
4
```

Five records enter the loop. `d2` has an empty text, so the log names that ID and the reason `empty`. The other four are counted as kept. `5/5` is `progress.n` and `progress.total` after the loop. The bar itself was written into a throwaway buffer, because its rate and elapsed time change from run to run. Logging the raw text would have copied private wording into the log. The ID and the reason are enough to find the row.

**Practice.** If `d4` were also empty, the log would gain a second `INFO` line and the kept count would be 3. The progress total would still be 5.

Sources: [logging](https://docs.python.org/3/library/logging.html), [tqdm](https://tqdm.github.io/).

## OL02. Profiling

A slow program is not yet a diagnosis. `timeit`, `cProfile` and `tracemalloc` come with Python, and they measure different things. `timeit` repeats a snippet and reports a duration. `cProfile` counts calls inside one run. `tracemalloc` traces memory allocated by Python itself. People use them before rewriting an algorithm, because a function that is called once on four rows is a different problem from a function that is called millions of times, and a Python allocation trace is not the GPU’s memory total.

Durations and byte totals move when the machine, the Python build or the background load changes. This lesson prints the operation result, the call count, the length of the list that was held, and whether tracing is on. It does not print a time or a byte count as if that number were part of the language.

<!-- example: OL02-main -->
```python
import cProfile
import pstats
import tracemalloc

def clean(rows):
    return [row for row in rows if row]

rows = ["a", "", "b", "c"]
profiler = cProfile.Profile()
profiler.enable()
cleaned = clean(rows)
profiler.disable()
stats = pstats.Stats(profiler)
calls = [value[1] for key, value in stats.stats.items() if key[2] == "clean"]
print(cleaned)
print(calls)
tracemalloc.start()
held = [0] * 20
print(len(held))
print(tracemalloc.is_tracing())
tracemalloc.stop()
```

**Output**
```text
['a', 'b', 'c']
[1]
20
True
```

`clean` drops the empty string, so three rows remain. The profiler table records one call to `clean`: the region between `enable` and `disable` called it once. The `20` is the length of the list created while tracing, which is the fact we chose to keep. `True` means `tracemalloc` was recording Python allocations at that moment. A later run can print `tracemalloc.get_traced_memory()` on the same machine and compare two versions of a function. That byte pair is a measurement, not a second copy of this checked output. `cProfile` also does not time work that happens in another process, in a GPU kernel, or while the program waits on a network.

**Practice.** Calling `clean` twice inside the profiled region would make the call count `2`. The cleaned rows would still be `['a', 'b', 'c']` for this input.

Sources: [The Python Profilers](https://docs.python.org/3/library/profile.html), [tracemalloc](https://docs.python.org/3/library/tracemalloc.html).

## OL03. SQL

`sqlite3` comes with Python. It talks to SQLite, a database stored in one file, or in this lesson entirely in memory. SQL is the language of the query. `sqlite3` is the Python module that sends that language and brings rows back. People use it when experiment records or document tables need a filter, a join and a grouped count, and another dictionary shape would only be readable by the script that invented it.

A **connection** is the session. A **table** is the named rows. A **cursor** carries one query’s results. `?` marks a value that is data, not SQL text. `commit` ends the transaction that holds the inserts. `ORDER BY` makes the row order part of the contract, so the checked print is not “whatever order the file happened to use.”

<!-- example: OL03-main -->
```python
import sqlite3

connection = sqlite3.connect(":memory:")
connection.execute("CREATE TABLE documents (doc_id TEXT, group_name TEXT)")
connection.execute("CREATE TABLE scores (doc_id TEXT, score INTEGER)")
connection.executemany("INSERT INTO documents VALUES (?, ?)", [("d1", "A"), ("d2", "A"), ("d3", "B")])
connection.executemany("INSERT INTO scores VALUES (?, ?)", [("d1", 2), ("d2", 4), ("d3", 6)])
connection.commit()
rows = connection.execute(
    "SELECT documents.group_name, COUNT(*) "
    "FROM documents JOIN scores ON documents.doc_id = scores.doc_id "
    "GROUP BY documents.group_name ORDER BY documents.group_name"
).fetchall()
print(rows)
connection.close()
```

**Output**
```text
[('A', 2), ('B', 1)]
```

The join keeps a document only when its ID also appears in `scores`. All three documents match. Group A has `d1` and `d2`, so its count is 2. Group B has `d3`, so its count is 1. `ORDER BY documents.group_name` puts A before B. Without that clause, a correct database could still return the groups in another order, and a lesson that required one order would be asking for an accident. The query counts rows. It does not decide that group A is the better system.

**Practice.** A score row whose `doc_id` is `d4` would not increase either count, because the join starts from documents and `d4` is not in that table.

Sources: [sqlite3](https://docs.python.org/3/library/sqlite3.html).

## OL04. Columnar tables

A CSV stores every value as text in a row. **Arrow** is an in-memory columnar representation: one typed array per column. **Parquet** is a file format that stores those columns. **DuckDB** is an analytical database that can query the file. People use the three together when a corpus table is wider than the columns a given question needs, because a reader can ask for `score` and leave the rest of the file alone.

PyArrow and DuckDB are third-party packages. The table below has three document IDs and three integer scores. Writing it, reading one column back, and querying the same file should agree. `fetchall` returns the rows of this small result. It is not a promise that every future query fits in memory, and it does not stream an answer that is larger than the process.

<!-- example: OL04-main -->
```python
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

table = pa.table({"doc_id": ["d1", "d2", "d3"], "score": pa.array([2, 4, 6], type=pa.int64())})
pq.write_table(table, "scores.parquet")
loaded = pq.read_table("scores.parquet", columns=["score"])
print(loaded.column("score").to_pylist())
answer = duckdb.sql("SELECT doc_id, score FROM 'scores.parquet' WHERE score >= 4 ORDER BY doc_id").fetchall()
print(answer)
```

**Output**
```text
[2, 4, 6]
[('d2', 4), ('d3', 6)]
```

The Arrow column `score` comes back as the integers 2, 4 and 6. The query keeps scores of at least 4 and orders the IDs, so `d2` precedes `d3`. `d1` is in the file and out of the result. Reading the column and running the query are two views of one stored table. DuckDB did not train a model on the scores. A filter of `>= 4` is the question we wrote, not a property of Parquet.

**Practice.** `WHERE score >= 6` would leave only `('d3', 6)`. The Arrow column read without that filter would still print all three scores.

Sources: [Arrow Parquet](https://arrow.apache.org/docs/python/parquet/parquet.html), [DuckDB SQL in Python](https://duckdb.org/docs/current/guides/python/execute_sql).

## OL05. Rearranging tensors

A tensor’s shape says how many items sit on each axis. It does not say which axis is the batch, the head, or the token. `reshape` reinterprets the same stored order as a new shape. `einops.rearrange` moves items by names you write in a pattern. Einops is a third-party package. People use it in model code because `batch heads tokens -> batch tokens heads` is readable beside a comment, and a silent `reshape` can look successful while pairing the wrong numbers.

The tensor below is `[[[0, 1, 2], [3, 4, 5]]]`, shape `(1, 2, 3)`. Position `[0, 1, 2]` holds 5: batch 0, head 1, token 2.

<!-- example: OL05-main -->
```python
import torch
from einops import rearrange

values = torch.arange(6).reshape(1, 2, 3)
print(values.reshape(1, 3, 2).tolist())
moved = rearrange(values, "batch heads tokens -> batch tokens heads")
print(moved.tolist())
print(int(values[0, 1, 2]), int(moved[0, 2, 1]))
```

**Output**
```text
[[[0, 1], [2, 3], [4, 5]]]
[[[0, 3], [1, 4], [2, 5]]]
5 5
```

`reshape(1, 3, 2)` walks the stored order 0, 1, 2, 3, 4, 5 and fills rows of length 2. The 5 sits beside 4. `rearrange` treats axis 1 as heads and axis 2 as tokens, then swaps those axes. Token 2 now holds the pair `[2, 5]`: the old heads 0 and 1 at that token. `moved[0, 2, 1]` is still 5, which is the check that head 1 of token 2 moved to token 2, head 1. The two prints with shape `(1, 3, 2)` are not the same arrangement. Rearranging does not learn a feature. It only places existing numbers.

**Practice.** `rearrange(values, "batch heads tokens -> heads batch tokens")` would put the two heads on the outside. The number 5 would still be the last token of the second head.

Sources: [einops basics](https://einops.rocks/1-einops-basics/), [torch.reshape](https://docs.pytorch.org/docs/2.9/generated/torch.reshape.html).

## OL06. Tensor files

`safetensors` is a file format, and a Python package, for named tensors. Hugging Face checkpoints use it, which is why a loader elsewhere in this course asks for `use_safetensors=True`. This lesson writes and reads a file directly, so the format is visible without a model. People use it because a tensor file should load as numbers rather than as arbitrary Python objects. The names, shapes and values come back. The architecture, the optimizer and the license do not. A safer file format is not a verdict on the model’s quality.

<!-- example: OL06-main -->
```python
import torch
from safetensors.torch import load_file, save_file

tensors = {"weight": torch.tensor([[1., 2.], [3., 4.]]), "bias": torch.tensor([0.5])}
save_file(tensors, "tiny.safetensors")
loaded = load_file("tiny.safetensors")
print(sorted(loaded))
print(tuple(loaded["weight"].shape), tuple(loaded["bias"].shape))
print(bool(torch.equal(loaded["weight"], tensors["weight"])))
```

**Output**
```text
['bias', 'weight']
(2, 2) (1,)
True
```

The file has two names. `weight` is a 2 by 2 matrix and `bias` is a vector of one number. `torch.equal` reports that the loaded weight matches the tensor that was saved. Sorted names keep the print stable. Nothing in the file says that `weight` belongs to a linear layer, or that these numbers were learned. Building a module still requires the code that knows which tensor fills which parameter.

**Practice.** Saving a third tensor named `scale` would add that name to the sorted list. The weight comparison would stay true.

Sources: [Safetensors](https://huggingface.co/docs/safetensors/index).

## OL07. Experiment records

MLflow is a third-party package for a **run**: a name, the parameters you set, the metrics you computed, and the files you attach. People use it so a result can be found later with the settings that produced it. This lesson writes a synthetic overlap of 0.5 and a limit of 2 into a local SQLite file. Logging those numbers does not launch training, open a dashboard, or create an account.

<!-- example: OL07-main -->
```python
import os
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
import mlflow

mlflow.set_tracking_uri("sqlite:///tracking.db")
mlflow.set_experiment("lesson")
with mlflow.start_run(run_name="fixture"):
    mlflow.log_param("limit", "2")
    mlflow.log_metric("overlap", 0.5)
client = mlflow.MlflowClient()
experiment = client.get_experiment_by_name("lesson")
runs = client.search_runs([experiment.experiment_id])
print(runs[0].data.params["limit"])
print(runs[0].data.metrics["overlap"])
```

**Output**
```text
2
0.5
```

`set_tracking_uri` chooses the local database. `set_experiment` creates the experiment named `lesson` when it is missing. Inside the run, `limit` is stored as the parameter text `"2"` and `overlap` as the metric 0.5. The search reads that run back. The printed values are the ones the lesson logged. MLflow did not check that 0.5 is a real score, and a stored metric is not evidence that a model improved. A run ID is generated and is not part of the checked output, because a second execution receives a different ID.

**Practice.** Logging `overlap` as 0.25 would make the second printed line `0.25`. The parameter line would stay `2` until the logged parameter changed.

Sources: [MLflow tracking](https://mlflow.org/docs/latest/ml/tracking/).

## OL08. Property tests

A few chosen examples can all pass while a nearby input fails. Hypothesis is a third-party package that draws inputs and looks for a counterexample. It builds on the pytest lesson: a test is still a function with an assertion. People use it when the claim is a property, such as “the first occurrence of each item stays in order,” rather than one memorized list.

`unique_in_order` keeps the first time it sees an item and drops later copies. `dict.fromkeys` does the same job with a dictionary, whose keys remember insertion order. The property says those two results agree. Hypothesis draws small integer lists. `derandomize=True` makes this checked run repeatable. A property test is not a proof. It tried the cases it drew. `database=None` keeps the example from writing a saved counterexample file.

<!-- example: OL08-main -->
```python
from hypothesis import given, settings, strategies as st

def unique_in_order(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

@settings(max_examples=15, database=None, derandomize=True)
@given(st.lists(st.integers(min_value=0, max_value=5), max_size=6))
def test_first_occurrence_order(items):
    result = unique_in_order(items)
    assert result == list(dict.fromkeys(items))

print(unique_in_order(["d1", "d2", "d1"]))
test_first_occurrence_order()
print("property checks passed")
```

**Output**
```text
['d1', 'd2']
property checks passed
```

`["d1", "d2", "d1"]` keeps `d1` at the first position and drops the later copy, so the list is `['d1', 'd2']`. The decorated function then draws 15 integer lists and checks the same property. The second line is printed only because every drawn list agreed. If `unique_in_order` sorted its result, an input such as `[2, 1]` would fail: the property wants `[2, 1]`, and sorting would return `[1, 2]`. Hypothesis would report that counterexample. This passing run does not print one, and it does not show that every possible list was tried.

**Practice.** `unique_in_order(["d1", "d1"])` is `['d1']`. The property would still require that single first occurrence, not an empty list.

Sources: [Hypothesis quick start](https://hypothesis.readthedocs.io/en/latest/quickstart.html).

## OL09. Graph objects

NetworkX is a third-party package for a graph as a Python object: nodes, edges, and questions such as “is there a path?” and “is there a cycle?”. People use it for argument maps, citation links and the dependencies between steps in a workflow. This lesson owns the object and the calls. The algorithms section, when it is added, owns why a search visits each node once and how a queue differs from a stack. A NetworkX graph is not a trained graph neural network, and it is not automatically the right store for a huge graph.

<!-- example: OL09-main -->
```python
import networkx as nx

graph = nx.DiGraph()
graph.add_edge("claim", "premise")
graph.add_edge("premise", "claim")
graph.add_edge("claim", "evidence")
print(sorted(graph.nodes))
print(sorted(graph.edges))
print(nx.has_path(graph, "claim", "evidence"))
print(nx.is_directed_acyclic_graph(graph))
```

**Output**
```text
['claim', 'evidence', 'premise']
[('claim', 'evidence'), ('claim', 'premise'), ('premise', 'claim')]
True
False
```

The graph is directed. Three edges were added, and sorting makes the print independent of storage order. `claim` can reach `evidence` by the direct edge, so `has_path` is true. `claim` and `premise` point at each other, which is a cycle, so the graph is not a directed acyclic graph. A cycle matters when the edges mean “must happen before”: the two steps cannot both go first. The path result is about this small graph. It does not say that a reader is allowed to see the evidence, and it does not train a model on the nodes.

**Practice.** Removing the edge from `premise` back to `claim` would leave a directed acyclic graph. `has_path(graph, "claim", "evidence")` would stay true.

Sources: [NetworkX tutorial](https://networkx.org/documentation/stable/tutorial.html).
