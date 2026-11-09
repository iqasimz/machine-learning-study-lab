# Data and plotting in Python

Prerequisite: lists, dictionaries, functions and imports from the basics. This chapter teaches NumPy, pandas, a small amount of SciPy, and Matplotlib. Each library has one job. An array holds a grid of numbers. A table holds named columns. A figure draws a question you have already calculated. The examples do not train models.

## DS01. NumPy arrays

### Why NumPy

NumPy is the array library the rest of this course builds on. pandas, scikit-learn, and PyTorch all speak in terms of the grids NumPy made ordinary. People use it because a table of numbers should add, multiply, and reduce by axis without a Python loop over every cell, and because that work runs in compiled code. The object is an **ndarray**. After `import numpy as np`, `np.array([1, 2]) + np.array([3, 4])` is the whole addition. NumPy will add the wrong axis if you ask it to. It does not know which axis is a batch, a time step, or a feature.

A Python list can hold numbers, but `+` on two lists joins them. It does not add corresponding numbers. Adding two arrays adds the values position by position.

<!-- example: DS01-why -->
```python
print([1, 2] + [3, 4])
import numpy as np
print((np.array([1, 2]) + np.array([3, 4])).tolist())
```

**Output**
```text
[1, 2, 3, 4]
[4, 6]
```

The list result has four items because the lists were joined. The array result has two items: `1+3` and `2+4`. `.tolist()` converts the array back to a Python list so the print is a plain list. `np` is the usual short name from `import numpy as np`. The array itself is what the rest of this chapter operates on.

`np.array` builds an array from a list. **Shape** is the length of each axis, written as a tuple. A one-dimensional array of three numbers has shape `(3,)`. The comma is there so you can see it is a tuple of one length, not just the number 3. **dtype** is the storage type of each value. Indexing starts at zero, as it does for a list. `values[0]` is the first number.

<!-- example: DS01-shape -->
```python
import numpy as np

values = np.array([1, 2, 3], dtype=np.int64)
print(values.shape)
print(int(values[0]))
print(str(values.dtype))
```

**Output**
```text
(3,)
1
int64
```

`int64` means each value uses 64 bits, eight bytes. The dtype is part of the array. It is not a Python `int` stored separately for each cell. Passing `dtype=np.int64` makes that choice explicit.

A two-dimensional array is a grid. In a table of measurements we often treat rows as examples and columns as features. NumPy does not know those meanings. Shape `(3, 2)` only says three along the first axis and two along the second. `ndim` is the number of axes. `size` is the number of values, the product of the shape. `nbytes` is the storage of those values. A `float32` value uses four bytes.

<!-- example: DS01-main -->
```python
import numpy as np

measurements = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
print(measurements.shape)
print(measurements.ndim, str(measurements.dtype))
print(measurements.size, measurements.nbytes)
print(measurements[1].tolist())
```

**Output**
```text
(3, 2)
2 float32
6 24
[3.0, 4.0]
```

There are three rows, two columns and six values. Six times four bytes is 24. That count is the number buffer, not the entire memory of the Python object. `measurements[1]` selects the second row, `[3.0, 4.0]`, because the first row is position 0.

**Practice.** `(3, 2)` and `(2, 3)` hold the same six values and describe different layouts. Reshaping only to silence an error can swap examples with features. Check which axis is which before changing the shape.

Sources: [NumPy beginner guide](https://numpy.org/doc/1.26/user/absolute_beginners.html), [NumPy array attributes](https://numpy.org/doc/1.26/reference/arrays.ndarray.html).

## DS02. NumPy indexing and copies

Indexing selects values. `values[0]` is the first row. `values[:, 1]` uses a slice on the rows and position 1 on the columns, so it is the second column. The colon means “every position along this axis.”

<!-- example: DS02-index -->
```python
import numpy as np

values = np.array([[1, 2], [3, 4], [5, 6]])
print(values[0].tolist())
print(values[:, 1].tolist())
```

**Output**
```text
[1, 2]
[2, 4, 6]
```

The first row is `[1, 2]`. The second column is `2`, `4` and `6`. A basic slice can be a **view**: another array object that shares the same stored values. Changing the view changes the original. Selecting rows with a list of positions, such as `values[[0, 2]]`, copies those rows instead.

<!-- example: DS02-main -->
```python
import numpy as np

values = np.arange(6).reshape(3, 2)
view = values[:, 0]
view[0] = 9
selected = values[[0, 2], :]
selected[0, 0] = 99
print(values.tolist())
print(values.T.shape)
print(values[values > 3].tolist())
```

**Output**
```text
[[9, 1], [2, 3], [4, 5]]
(2, 3)
[9, 4, 5]
```

`np.arange(6)` builds `0, 1, 2, 3, 4, 5`. `reshape(3, 2)` lays them out as three rows. `values[:, 0]` is the first column, a view, so `view[0] = 9` writes 9 into `values`. The list selection copies rows 0 and 2, so assigning 99 there leaves `values` unchanged. `values > 3` is an array of Booleans. Using it as an index gathers the matching values into a one-dimensional result: 9, 4 and 5. `.T` swaps the axes, so shape `(3, 2)` becomes `(2, 3)`.

`reshape` sometimes returns a view and sometimes a copy, depending on how the values are stored. Do not assume a new shape is independent data. `np.stack` adds a new axis. `np.concatenate` joins arrays along an axis that already exists. Stacking two `(3, 2)` arrays adds a group axis. Concatenating them along rows produces six rows.

**Practice.** After `view = values[:, 0]`, assigning to `view` changes `values`. After `selected = values[[0, 2], :]`, assigning to `selected` does not. The brackets around the row numbers are what made the copy.

Sources: [NumPy copies and views](https://numpy.org/doc/1.26/user/basics.copies.html), [NumPy indexing](https://numpy.org/doc/1.26/user/basics.indexing.html).

## DS03. Axes and broadcasting

A **reduction** combines values along one axis and leaves the other axes in the result. `axis=0` combines down the rows, one result per column. `axis=1` combines across the columns, one result per row.

<!-- example: DS03-axis -->
```python
import numpy as np

features = np.array([[2., 4.], [6., 8.]])
print(features.mean(axis=0).tolist())
print(features.sum(axis=1).tolist())
```

**Output**
```text
[4.0, 6.0]
[6.0, 14.0]
```

The column means are `(2+6)/2 = 4` and `(4+8)/2 = 6`. The row sums are `2+4 = 6` and `6+8 = 14`. `keepdims=True` keeps the reduced axis at length 1, so a column mean of a `(3, 2)` array has shape `(1, 2)` rather than `(2,)`. That length-1 axis is what makes the mean line up with the original grid.

**Broadcasting** stretches length-1 axes so an element-wise operation can pair values. Lengths must match, or one of them must be 1. The comparison starts from the rightmost axis. Matrix multiplication, `@`, is a different operation: it pairs a row with a column and sums the products. `*` multiplies values that have already been paired. It does not sum them.

<!-- example: DS03-main -->
```python
import numpy as np

features = np.array([[2., 4.], [6., 8.], [4., 6.]])
means = features.mean(axis=0, keepdims=True)
centered = features - means
weights = np.array([1., 2.])
print(means.shape, means.tolist())
print(centered.tolist())
print((features @ weights).tolist())
print((features * weights).tolist())
```

**Output**
```text
(1, 2) [[4.0, 6.0]]
[[-2.0, -2.0], [2.0, 2.0], [0.0, 0.0]]
[10.0, 22.0, 16.0]
[[2.0, 8.0], [6.0, 16.0], [4.0, 12.0]]
```

The means are 4 and 6, stored with shape `(1, 2)`. Subtracting them from a `(3, 2)` array uses the length-1 row three times. The first centered row is `2-4` and `4-6`. `@` on the first row is `2×1 + 4×2 = 10`. `*` leaves both products in place: `2×1` and `4×2`. A mean shaped `(3,)` does not line up with `(3, 2)` under the right-to-left rule. A mean shaped `(3, 1)` does, and it would center each row.

These weights are numbers we supplied. They are not coefficients learned from data. When a mean is used to prepare model inputs, it has to be computed from the fitting rows and then reused. DS and the later sklearn lessons return to that rule.

Sources: [NumPy broadcasting](https://numpy.org/doc/1.26/user/basics.broadcasting.html), [NumPy matmul](https://numpy.org/doc/1.26/reference/generated/numpy.matmul.html).

## DS04. Sampling with NumPy and SciPy

`np.random.default_rng(seed)` builds a pseudorandom generator. The seed fixes the sequence for this generator and this NumPy version. `integers(low, high, size=n)` draws `n` integers from `low` up to, and not including, `high`.

Sampling **with replacement** can draw the same position more than once. A bootstrap resample draws a new sample of the same size from the observed rows, with replacement, and recomputes a statistic. The item you resample has to match the study: a row, a patient, a document or a group. Resampling the wrong unit pretends that dependent rows were independent.

### Why SciPy

SciPy is the library of numerical routines that sit on NumPy arrays: statistics, optimization, sparse matrices, and special functions. People use it because those routines are shared and tested, so a standard error or a sparse count matrix does not have to be rederived in each project. `from scipy import stats` then `stats.sem(values)` is the call. SciPy computes the formula you select. It does not decide whether the rows are independent, and a standard error is not a bootstrap interval.

`scipy.stats.sem` estimates the standard error of the mean from the sample’s own spread. It is the standard error, which is different from the standard deviation of the observations. Sparse matrices from `scipy.sparse` appear in the last lesson of this chapter, for grids that are mostly zeros. That is the same library.

<!-- example: DS04-main -->
```python
import numpy as np
from scipy import stats

scores_a = np.array([1., 0., 1., 1.])
scores_b = np.array([0., 0., 1., 0.])
indices = np.array([0, 0, 2, 3])
paired_difference = scores_a[indices].mean() - scores_b[indices].mean()
print(f"paired difference={paired_difference:.2f}")
print(f"standard error={stats.sem([2., 4., 6.]):.3f}")
rng = np.random.default_rng(17)
print(rng.integers(0, 4, size=4).shape)
```

**Output**
```text
paired difference=0.75
standard error=1.155
(4,)
```

The index array `[0, 0, 2, 3]` is one chosen resample. Position 0 is used twice, and position 1 is unused. Both score arrays are indexed with those same positions, so paired rows stay paired. The printed difference is that one resample’s mean difference, not a confidence interval. `stats.sem([2., 4., 6.])` is a separate illustration on three fixed numbers. The last line only checks that a draw of four integers has shape `(4,)`. It does not print the integers, because the lesson is the draw’s shape and the paired index, not a particular random sequence.

Many resamples can describe how a statistic varies under that resampling scheme. They do not repair a biased sample. A seed that restarts training is a different experiment from a bootstrap of rows. Running several hypothesis tests and keeping the smallest p-value needs a plan that was chosen before looking at those p-values.

Sources: [SciPy bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html), [SciPy standard error](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sem.html), [Think Stats](https://allendowney.github.io/ThinkStats/).

## DS05. pandas tables

### Why pandas

pandas is the library people use for tables in Python. A column has a name, a missing value has a place, and a filter can be written as a condition on the column rather than as a loop over row indexes. Spreadsheet files, CSV files, and database results are routinely loaded into the same object, which is why the library shows up in analysis code even when the model itself is NumPy or scikit-learn. After `import pandas as pd`, `table["score"]` selects a column. pandas aligns rows by index and will happily average a column that includes values you meant to exclude. It does not know which rows belong in the study.

A **DataFrame** is the table. A **Series** is one column, with the table’s row index attached. `type(table).__name__` and `type(table["score"]).__name__` show those two objects. `.tolist()` copies values into a Python list. `.shape` is `(rows, columns)`.

<!-- example: DS05-objects -->
```python
import pandas as pd

table = pd.DataFrame({"id": ["a", "b"], "score": [2, 6]})
print(type(table).__name__)
print(type(table["score"]).__name__)
print(table["score"].tolist())
print(tuple(table.shape))
```

**Output**
```text
DataFrame
Series
[2, 6]
(2, 2)
```

The dictionary keys become column names. The values become the columns. Row labels default to 0 and 1. Those labels are an index, separate from the integer position. Two rows and two columns give shape `(2, 2)`.

`pd.read_csv` builds a DataFrame from CSV text. A blank field becomes a missing value, which pandas prints as `NaN` and tests with `.isna()`. `.count()` counts values that are present. `.mean()` on a numeric Series skips missing values in its default behavior. Filling the gap with zero would include a zero in that mean, which answers a different question.

`io` comes with Python. `StringIO` wraps a string so a function written for a file can read it. The example uses it so the CSV stays on the page. `StringIO` supplies those characters. It does not find a file on disk and it does not decide how a blank field should be filled.

<!-- example: DS05-main -->
```python
import pandas as pd
from io import StringIO

table = pd.read_csv(StringIO("id,group,score\na,A,2\nb,A,\nc,B,6\n"))
print(table.shape)
print(table["score"].isna().tolist())
print(table["score"].count())
print(table["score"].mean())
```

**Output**
```text
(3, 3)
[False, True, False]
2
4.0
```

Three rows and three columns are present. The score column is missing in the second row, so `.isna()` is false, true, false. `.count()` is 2. The mean of the observed scores is `(2+6)/2 = 4`. The missing row still exists in the table.

CSV text does not store a full schema. A column of digits may arrive as text unless you convert it. Parquet can keep column types and read selected columns. Neither format means the rows are independent or licensed for redistribution. Looking at the test split while you decide how to fill missing values uses that split for design. Inspect columns, ranges, duplicate identifiers and units before a model sees the table.

**Practice.** `table["score"].count()` answers “how many scores are present?” `len(table)` answers “how many rows are present?” On this table those numbers are 2 and 3.

Sources: [pandas data structures](https://pandas.pydata.org/pandas-docs/version/2.3/user_guide/dsintro.html), [pandas missing values](https://pandas.pydata.org/pandas-docs/version/2.3/user_guide/missing_data.html).

## DS06. Selecting and updating rows

`.loc` selects by labels and by Boolean conditions. `.iloc` selects by integer position, starting at zero, and ignores the labels. Use `.loc` when you mean “the row named `a`” or “the rows where a condition is true.” Use `.iloc` when you mean “the row at position 1,” whatever its label is.

<!-- example: DS06-select -->
```python
import pandas as pd

table = pd.DataFrame({"score": [2.0, 6.0]}, index=["a", "b"])
print(table.loc["a", "score"])
print(table.iloc[1, 0])
```

**Output**
```text
2.0
6.0
```

`loc["a", "score"]` is the score on the row labeled `a`. `iloc[1, 0]` is the second row and the first column, which is 6.0. If the index were sorted differently, those two selections could stop referring to the same cell.

Assignment through `.loc` writes into the rows you selected. A Boolean Series such as `table["score"].isna()` has one true or false per row. Missing values are not numbers, so a later numeric step needs an explicit decision. Replacing them with −1 in the next example only makes the gap visible. It is not a general rule for model inputs.

Series arithmetic aligns by index label. Two Series can have the same length and still pair `a` with `a` rather than first position with first position.

<!-- example: DS06-main -->
```python
import pandas as pd

table = pd.DataFrame({"score": [2., None, 6.]}, index=["a", "b", "c"])
table.loc[table["score"].isna(), "score"] = -1
print(table.loc[["b", "c"], "score"].tolist())
print(table.iloc[0, 0])
first = pd.Series([10, 20], index=["a", "b"])
second = pd.Series([1, 2], index=["b", "a"])
print((first + second).tolist())
```

**Output**
```text
[-1.0, 6.0]
2.0
[12, 21]
```

Row `b` was missing and becomes −1. Row `c` stays 6. `iloc[0, 0]` is still row `a`, the value 2. `first + second` adds `a` to `a` (`10+2`) and `b` to `b` (`20+1`), so the result is 12 then 21 in `first`’s index order. Adding by position would have produced 11 and 22. Resetting an index without keeping the identifier can change which rows pair.

Write `table.loc[mask, "score"] = value` when you mean to update that column on those rows. An expression such as `table[mask]["score"] = value` can update a temporary object and leave the table unchanged. Nullable integer types can represent missing integers without converting the column to float. Dates need an explicit format and timezone when the order of events matters. A string that looks like a date is still a string until you parse it.

Sources: [pandas indexing](https://pandas.pydata.org/pandas-docs/version/2.3/user_guide/indexing.html), [copy-on-write](https://pandas.pydata.org/pandas-docs/version/2.3/user_guide/copy_on_write.html).

## DS07. Grouping and joins

`groupby("group")` splits the rows by the values in that column. Selecting `["score"]` and calling `.mean()` reduces each group to one number. The result is a Series whose index is the group label.

<!-- example: DS07-group -->
```python
import pandas as pd

scores = pd.DataFrame({"group": ["A", "A", "B"], "score": [2, 4, 8]})
print(scores.groupby("group")["score"].mean().to_dict())
```

**Output**
```text
{'A': 3.0, 'B': 8.0}
```

Group A’s mean is `(2+4)/2 = 3`. Group B’s mean is 8. `.to_dict()` turns the Series into a dictionary keyed by group. The group sizes are not in that dictionary. `agg(["count", "mean"])` keeps both, which DS10 uses.

`merge` matches rows by a key. `on="group"` says which column must match. A **many-to-one** merge allows several score rows to share one name row, and it rejects the merge when a key appears twice on the name side. `validate="many_to_one"` is that check. An inner merge keeps keys found on both sides. A left merge keeps every row of the left table. Choose the kind that matches the question, then compare the row count with the count you expected.

<!-- example: DS07-main -->
```python
import pandas as pd

scores = pd.DataFrame({"group": ["A", "A", "B"], "score": [2, 4, 8]})
names = pd.DataFrame({"group": ["A", "B"], "name": ["alpha", "beta"]})
joined = scores.merge(names, on="group", validate="many_to_one")
print(joined.groupby("name")["score"].mean().to_dict())
print(len(joined))
try:
    scores.merge(pd.concat([names, names]), on="group", validate="many_to_one")
except pd.errors.MergeError:
    print("duplicate lookup keys rejected")
```

**Output**
```text
{'alpha': 3.0, 'beta': 8.0}
3
duplicate lookup keys rejected
```

Each score row finds one name. The joined table still has 3 rows, the same number as `scores`. The means match the groupby above, now labeled by name. `pd.concat([names, names])` stacks the lookup table on itself, so each group key appears twice. The validated merge refuses that table. Without `validate`, a many-to-many match would multiply rows: two A scores times two A names would create four A rows, and a later mean would not describe the original scores.

`concat` stacks tables. It does not match keys. Pivot and melt reshape a table between wide and long forms. They are separate operations from `merge`, and this lesson does not treat a mention of them as practice. Count rows and unmatched keys after a join before you trust a mean computed from it.

Sources: [pandas merge and join](https://pandas.pydata.org/pandas-docs/version/2.3/user_guide/merging.html), [pandas grouping](https://pandas.pydata.org/pandas-docs/version/2.3/user_guide/groupby.html).

## DS08. Matplotlib figures

### Why Matplotlib

Matplotlib is the plotting library the scientific Python stack draws with. pandas and seaborn can hand it an axes, and a saved figure is an ordinary image file. People use it because one script can draw the chart and write the file, with the labels in the same program as the numbers. `plt.subplots()` returns a **Figure**, the whole image, and an **Axes**, one plotting region. `axes.plot(x, y)` draws a line. `axes.set` adds the labels and the title. Matplotlib draws the pairs you give it. It does not check that the line answers the question in the title.

A line is appropriate when the horizontal value is an ordered quantity such as time. A bar chart compares separate categories. A histogram counts values that fall into bins. The chart type is part of the claim.

<!-- example: DS08-main -->
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

minutes = [0, 10, 20]
records_read = [0, 4, 9]
figure, axes = plt.subplots(figsize=(5, 3), layout="constrained")
axes.plot(minutes, records_read, marker="o")
axes.set(xlabel="Time (minutes)", ylabel="Records read", title="Reading progress")
figure.savefig("progress.png", dpi=120)
print(axes.get_xlabel())
print(axes.lines[0].get_ydata().tolist())
plt.close(figure)
```

**Output**
```text
Time (minutes)
[0, 4, 9]
```

`matplotlib.use("Agg")` selects a backend that writes a file and does not open a window. It has to run before pyplot builds a figure. `figsize=(5, 3)` sets the size in inches. `plot` pairs 0 with 0, 10 with 4, and 20 with 9. The markers show those three observations. The straight segments between them are the line’s interpolation, not a measurement of what happened at minute 5. `get_xlabel` confirms the label that was set. `get_ydata` returns the vertical values that were plotted, which is a useful check that the picture and the table still match.

`figure.savefig` writes `progress.png` in the working directory. `plt.close` releases the figure. Creating figures in a loop without closing them holds on to their memory. The checked image for this lesson is [the progress figure](outputs/DS08-main-progress.png). Several panels are `fig, axs = plt.subplots(nrows, ncols)`. Each panel still needs its own labels.

Sources: [Matplotlib quickstart](https://matplotlib.org/stable/users/explain/quick_start.html), [Matplotlib savefig](https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.savefig.html).

## DS09. Diagnostic plots

A confusion matrix is a table of counts before it is a picture. Rows are actual labels. Columns are predicted labels. You choose that orientation by passing `labels` in a fixed order and by labeling the axes the same way. Precision and recall are ratios of those counts. A heatmap only displays the table.

### Why seaborn

seaborn draws statistical charts on a Matplotlib axes. People use it because a heatmap, a distribution, or a comparison across groups is a few arguments once the table already exists. `sns.heatmap` paints a grid you already computed. Some other seaborn functions average the data and add error bars themselves. When a function does that, set the aggregation explicitly and say what the bar means: the spread of observations, a standard error, a bootstrap interval, or the spread across repeated runs. seaborn will draw the summary its function computed. It does not decide which rows were allowed into that summary.

### scikit-learn, first as a metric

`sklearn.metrics.confusion_matrix` is this course’s first call into **scikit-learn**. The library is the usual Python toolkit for classical machine learning: prepare a table, fit an estimator, predict, and score the result with a named metric. Here only the metric appears. It counts pairs of actual and predicted labels. The machine-learning chapter explains why the library is organized around `fit`, `predict`, and `transform`. The function below counts. It does not train a model, and it does not know which label order you intended until you pass `labels`.

<!-- example: DS09-main -->
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

actual = [0, 0, 1, 1]
predicted = [0, 1, 1, 1]
counts = confusion_matrix(actual, predicted, labels=[0, 1])
figure, axes = plt.subplots(figsize=(4, 3), layout="constrained")
sns.heatmap(counts, annot=True, fmt="d", cbar=False, ax=axes,
            xticklabels=["negative", "positive"], yticklabels=["negative", "positive"])
axes.set(xlabel="Predicted label", ylabel="Actual label")
figure.savefig("confusion.png", dpi=120)
print(counts.tolist())
plt.close(figure)
```

**Output**
```text
[[1, 1], [0, 2]]
```

`labels=[0, 1]` puts class 0 first and class 1 second. The printed grid is:

- actual 0, predicted 0: 1
- actual 0, predicted 1: 1
- actual 1, predicted 0: 0
- actual 1, predicted 1: 2

For class 1, the true positives are 2. The predicted positives are the column total `1+2 = 3`, so precision is `2/3`. The actual positives are the row total `0+2 = 2`, so recall is `2/2`. Swapping the axis labels would describe those same stored numbers as the opposite errors. The figure is [the confusion-matrix image](outputs/DS09-main-confusion.png). `annot=True` prints the counts in the cells. `fmt="d"` formats them as integers.

A precision–recall curve uses scores and a moving threshold, not one hard label list. A residual plot compares prediction errors with another quantity, such as the prediction itself. Training-loss curves from a real run belong on the lab’s results pages. This lesson plots a fixed matrix.

Sources: [sklearn confusion matrix](https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.confusion_matrix.html), [Seaborn error bars](https://seaborn.pydata.org/tutorial/error_bars.html).

## DS10. A small checked analysis

A small analysis should say how many values entered each number it reports. `groupby` splits the rows. `agg(["count", "mean"])` computes both reductions on the same groups, so the mean is next to the count that produced it. `scipy.sparse.csr_matrix` is SciPy’s compressed store for a grid that is mostly zeros. It records the shape and the stored entries. It does not fill in the zeros, and it does not decide what a nonzero count means.

<!-- example: DS10-main -->
```python
from io import StringIO
import pandas as pd
from scipy.sparse import csr_matrix

table = pd.read_csv(StringIO("id,group,score\na,A,2\nb,A,4\nc,B,8\n"))
if table["id"].duplicated().any():
    raise ValueError("row IDs must be unique")
summary = table.groupby("group")["score"].agg(["count", "mean"])
print(summary.to_csv().strip())
sparse = csr_matrix([[0, 0, 2], [1, 0, 0]])
print(sparse.shape, sparse.nnz)
```

**Output**
```text
group,count,mean
A,2,3.0
B,1,8.0
(2, 3) 2
```

`duplicated()` marks repeated ids. `.any()` is true if at least one row is a repeat. This table has ids `a`, `b` and `c`, so the check passes. Group A contributes two scores whose mean is 3. Group B contributes one score, 8. `to_csv` turns that summary into text. `.strip()` removes the trailing newline so the print matches the two data lines. An actual program can write that text to a path from PY08.

The last two lines are a different object, a **sparse matrix** from SciPy. `csr_matrix` stores the nonzero values and their positions. This grid has shape `(2, 3)` and `nnz` equal to 2, because only two cells are nonzero. A wide vocabulary table is often sparse. Calling `.toarray()` on it builds the full grid, including the zeros, and can exhaust memory even when the sparse object was small enough.

For a larger file, read the columns you need and check row counts after every join or group. A join can still build the whole result in memory. Keep a short validation record: row count, duplicate ids, missing counts and the path that was read. The next chapter’s model code should start from a table whose shape and missing values you can already explain.

Sources: [pandas IO](https://pandas.pydata.org/pandas-docs/version/2.3/user_guide/io.html), [SciPy sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html).
