# Python basics

You do not need to know machine learning to begin. A program takes values, performs operations in a specified order, and may return or display a result. These lessons teach you to read that order. Later chapters use the same reading habits on arrays, tables and model interfaces.

## PY01. Reading a program

A **statement** is one instruction. An **expression** produces a value, such as `20 + 15`. A **name** is a label for a value. In `total = 20 + 15`, the expression on the right runs first and produces 35. The name `total` is then attached to that 35.

Four marks are enough to start reading lines:

- `=` attaches a name. It does not compare two values.
- Parentheses call something: `print(total)` calls `print` and hands it `total`.
- Quotes mark text. `"Minutes"` is text. `Minutes` without quotes is a name Python will look up.
- `#` starts a comment. Python skips the rest of that line.

<!-- example: PY01-comment -->
```python
# This note is not executed.
minutes = 20 + 15
print(minutes)
```

**Output**
```text
35
```

The comment produces no output. The next line attaches `minutes` to 35. `print` displays that value. `print` does not store a new value, and it is not part of the arithmetic.

Save Python code in a `.py` file if you want to run it later. A terminal command such as `python3 session.py` asks your shell to start Python with that file. That command is shell syntax. It does not belong inside the `.py` file. At an interactive Python prompt, an expression can be displayed automatically. A script needs an explicit `print` for the same visible result.

The next program records two study sessions and combines them.

<!-- example: PY01-main -->
```python
morning_minutes = 20
evening_minutes = 15
total_minutes = morning_minutes + evening_minutes
print(total_minutes)
print("Minutes:", total_minutes)
```

**Output**
```text
35
Minutes: 35
```

Python runs the lines from top to bottom. `morning_minutes + evening_minutes` looks up both names, adds the values, and the result is attached to `total_minutes`. The first `print` displays the number. The second displays the text `Minutes:` and then the same number. Starting with `print(total_minutes)` before that name exists raises `NameError`, because the lookup finds nothing.

Assignment stores the value it computed at that moment. Changing an input later does not revise the stored result. Recompute the addition if you want the new total.

<!-- example: PY01-recompute -->
```python
morning_minutes = 20
evening_minutes = 15
total_minutes = morning_minutes + evening_minutes
print(total_minutes)
morning_minutes = 30
print(total_minutes)
total_minutes = morning_minutes + evening_minutes
print(total_minutes)
```

**Output**
```text
35
35
45
```

The middle 35 is the old total. The name `total_minutes` still referred to 35 after `morning_minutes` was attached to a new value. The last line runs the addition again, so the printed total becomes 45. Names are labels, not spreadsheet formulas.

**Practice.** Predict the prints before you run a change in your head: if the last line were omitted, the program would still end by printing 35. The repair is to recompute.

Sources: [Think Python: variables](https://allendowney.github.io/ThinkPython/chap02.html), [Python introduction](https://docs.python.org/3/tutorial/introduction.html).

## PY02. Values, types and operators

A **type** decides which operations a value supports. This lesson introduces the five values you will see immediately: integers, floating-point numbers, strings, Booleans and `None`. Each short example uses one of them. The last example combines conversion and a floating-point comparison.

An **integer** counts whole units. `+`, `-` and `*` do the ordinary arithmetic. `//` is floor division: it rounds the quotient down, toward negative infinity. `%` is the remainder that makes the identity `(a // b) * b + (a % b) == a` hold. `**` raises to a power. An integer in Python grows as large as memory allows. It is not a fixed 32-bit cell.

<!-- example: PY02-integers -->
```python
count = 3
print(count + 2)
print(count * 4)
print(7 // 2, 7 % 2)
print(2 ** 3)
```

**Output**
```text
5
12
3 1
8
```

`count + 2` is 5, and that result is only printed. `count` is still 3, so `count * 4` is 12. `7 // 2` is 3 because the quotient 3.5 rounds down to 3, and `7 % 2` is the leftover 1. For a positive pair, rounding down looks like dropping the fraction. `2 ** 3` is `2×2×2`.

The same rule on a negative quotient does not drop toward zero. `-7 / 2` is `-3.5`. Rounding down reaches `-4`, not `-3`. The remainder is then `1`, because `(-4) * 2 + 1 = -7`.

<!-- example: PY02-floor -->
```python
print(-7 // 2)
print(-7 % 2)
```

**Output**
```text
-4
1
```

A **string** is text. `len` counts its characters. `[0]` selects the first character, because positions start at zero. `+` between two strings joins them. It does not add numbers that happen to be written as digits.

<!-- example: PY02-strings -->
```python
label = "claim"
print(label[0])
print(len(label))
print("not " + label)
```

**Output**
```text
c
5
not claim
```

`"claim"` has characters at positions 0 through 4. Position 0 is `c`. The length is 5. Joining `"not "` and `"claim"` produces a new string. The space is a character inside the first string, which is why the printed words are separated.

A slice `word[1:4]` takes characters from position 1 up to, and not including, position 4. `split` cuts a string into a list of pieces. With no argument it cuts on whitespace. `strip` removes whitespace from both ends and returns a new string.

<!-- example: PY02-text -->
```python
word = "claim"
print(word[1:4])
print("not claim".split())
print("  claim  ".strip())
```

**Output**
```text
lai
['not', 'claim']
claim
```

`c` is position 0, so positions 1, 2 and 3 are `l`, `a` and `i`. `split` finds the space and returns two strings. `strip` returns `"claim"` without the surrounding spaces. The original text with spaces is unchanged, because these operations return new strings.

A **Boolean** is `True` or `False`. A condition such as `score >= 0.5` produces one. `bool(value)` asks whether Python treats that value as true in an `if`. Zero and an empty string are false. A non-empty string is true. `None` is a separate value meaning “not supplied.” It is a different object from `0`, `False` and `""`. The test `value is None` asks whether a name refers to that exact missing marker.

<!-- example: PY02-bool -->
```python
ready = True
print(ready)
print(bool(0), bool("claim"))
print(None is None)
missing = None
print(missing is None)
```

**Output**
```text
True
False True
True
True
```

`bool(0)` is false. `bool("claim")` is true because the string contains characters. Both `is` checks succeed because each side is the same `None` object.

`=` attaches a name. `==` compares values. Mixing text and a number with `+` is a type error, because joining and addition are different operations.

<!-- example: PY02-convert-error -->
```python
try:
    print("3" + 2)
except TypeError as error:
    print(type(error).__name__)
```

**Output**
```text
TypeError
```

The repair is a conversion when the text really is an integer: `int("3") + 2` is 5. `int("three")` and `int("3.0")` fail, because those strings are not the decimal spelling of an integer. `str(3)` goes the other way and produces the text `"3"`.

### Approximate numbers

A **float** is a fixed-size approximation of a real number, not an exact decimal. `0.1 + 0.2` is very close to `0.3` and still not the same stored value, so `==` can say false.

### The math library

`math` comes with Python. It is the usual place for a square root, a logarithm, and a closeness test, so projects share one tested function instead of each writing a private version. The call stays short: `math.isclose(left, right)` asks whether two floats are close enough for a tolerance. The module applies that test. It does not choose the tolerance that fits a measurement, and it does not turn a float into an exact decimal.

Formatting such as `.3f` changes only the displayed digits.

<!-- example: PY02-main -->
```python
import math

count = int("3")
print(count + 2)
print(type(count).__name__)
print(0.1 + 0.2 == 0.3)
print(math.isclose(0.1 + 0.2, 0.3))
print(f"fraction={2 / 3:.3f}")
```

**Output**
```text
5
int
False
True
fraction=0.667
```

`int("3")` converts text to the integer 3, then `+ 2` produces 5. `type(count).__name__` reports the type’s name, `int`. The equality of the two float expressions is false. `isclose` reports that they are close enough for this tolerance. `2 / 3` is division that produces a float. The f-string rounds the display to three digits. The stored result of the division is unchanged.

Exact stored fractions are a later detail. You can inspect them with `format(0.1, ".60f")` when a numerical surprise depends on them. You do not need that print to use `isclose` correctly.

**Practice.** `"3" + "2"` joins text and prints `32`. `int("3") + int("2")` adds and prints `5`. Choose the conversion when the characters are meant as numbers.

Sources: [Python floating point](https://docs.python.org/3/tutorial/floatingpoint.html), [Python built-in types](https://docs.python.org/3/library/stdtypes.html).

## PY03. Conditions and loops

An `if` runs its indented block only when its condition is true. The condition itself is an expression that produces a Boolean. Indentation is the syntax that says which lines belong inside the decision. The lines must line up with spaces, not with a visual guess.

<!-- example: PY03-if -->
```python
score = 0.8
if score >= 0.5:
    print("positive")
else:
    print("negative")
```

**Output**
```text
positive
```

`score >= 0.5` is true, so the first branch runs. `else` runs when the condition is false. `elif` adds another condition between them and is checked only when the earlier condition was false.

<!-- example: PY03-elif -->
```python
score = 0.4
if score >= 0.8:
    print("high")
elif score >= 0.5:
    print("middle")
else:
    print("low")
```

**Output**
```text
low
```

`0.4` fails both comparisons, so `else` prints `low`. A score of `0.6` would skip `high` and print `middle`. A score of `0.9` would print `high` and never look at `elif`.

`or` is true when at least one side is true. `not` flips a Boolean.

<!-- example: PY03-bool -->
```python
print(False or True)
print(not False)
```

**Output**
```text
True
True
```

The left side of `or` is false and the right side is true, so the result is true. `not False` is true.

A `for` loop visits each item in a collection. The name after `for` is attached to the current item, and the indented body runs once per item.

<!-- example: PY03-for -->
```python
total = 0
for number in [1, 2, 3]:
    total = total + number
print(total)
print(list(range(3)))
```

**Output**
```text
6
[0, 1, 2]
```

The body runs three times, with `number` equal to 1, then 2, then 3. `total` becomes 1, then 3, then 6. The print is outside the loop because it is not indented, so it runs once. `range(3)` produces 0, 1 and 2: it starts at zero and stops before 3. `list` collects those values so they can be printed.

`and` is true only when both comparisons are true. `or` is true when at least one is true. `not` flips a Boolean. `break` leaves the current loop immediately. `continue` skips the rest of the body and goes to the next item. Neither returns from a function.

<!-- example: PY03-break -->
```python
score = 0.8
label = "claim"
if score >= 0.5 and label == "claim":
    print("keep")

for number in [1, 2, 3, 4]:
    if number == 3:
        break
    print(number)
```

**Output**
```text
keep
1
2
```

Both parts of `and` are true, so the program prints `keep`. The loop prints 1 and 2. When `number` is 3, `break` ends the loop, so 4 is never printed.

`continue` does not end the loop. It skips the rest of this pass and starts the next item.

<!-- example: PY03-continue -->
```python
kept = []
for number in [1, 2, 3]:
    if number == 2:
        continue
    kept.append(number)
print(kept)
```

**Output**
```text
[1, 3]
```

The loop still visits 1, 2 and 3. When the number is 2, `continue` skips `append`. 1 and 3 are stored. A `while` loop repeats as long as its condition stays true, so the body has to change something that the condition looks at.

The next program counts how many scores meet a threshold, then shows `range` and a `while` that counts down.

<!-- example: PY03-main -->
```python
scores = [0.2, 0.8, 0.5]
positive_count = 0
for score in scores:
    if score >= 0.5:
        positive_count += 1
print(positive_count)
print(list(range(3)))

remaining = 2
while remaining > 0:
    remaining -= 1
print(remaining)
```

**Output**
```text
2
[0, 1, 2]
0
```

The loop sees three scores. `0.2` fails the test. `0.8` and `0.5` pass, so the count ends at 2. `+= 1` means “take the current count, add 1, and store it back.” The `while` body subtracts 1 twice, from 2 to 1 to 0, and then `remaining > 0` is false. Forgetting that subtraction would leave the condition true forever.

**Practice.** Change the test to `score > 0.5`. The value `0.5` no longer counts, so the printed count is 1. The boundary belongs in the condition on purpose. Later, NumPy will not accept a whole array as one `if` condition. An array comparison produces an array of Booleans, and you reduce it with an operation such as a sum.

Sources: [Python control flow](https://docs.python.org/3/tutorial/controlflow.html), [Think Python: iteration](https://allendowney.github.io/ThinkPython/chap07.html).

## PY04. Data structures

A **data structure** is a way of storing several values so that a later line can retrieve them. Python’s four everyday data structures are the list, the tuple, the dictionary and the set. A stack and a queue are two ways of adding and removing items. Both can be built from a list. Learn these operations before using a helper that builds one of them for you.

<!-- example: PY04-kinds -->
```python
labels = ["claim", "premise"]
pair = ("claim", "premise")
counts = {"claim": 1}
unique = {"claim", "claim", "premise"}
print(type(labels).__name__, type(pair).__name__, type(counts).__name__, type(unique).__name__)
print(labels[0], counts["claim"], sorted(unique))
```

**Output**
```text
list tuple dict set
claim 1 ['claim', 'premise']
```

Square brackets create a list. A comma inside parentheses creates a tuple: `("claim", "premise")` has a comma, so it is a tuple. Parentheses alone only group an expression, so `(3)` is the integer 3 and `(3,)` is a one-item tuple. The comma is the tuple syntax. Curly brackets with `key: value` create a dictionary, including `{}`, the empty dictionary. Curly brackets with only values create a set, and the repeated `"claim"` is stored once. An empty set is `set()`, because `{}` is already the empty dictionary. `labels[0]` selects by position. `counts["claim"]` selects by key. `sorted` returns a list because a set does not promise an order.

<!-- example: PY04-singleton -->
```python
print(type((3)).__name__)
print(type((3,)).__name__, len((3,)))
print(type(()).__name__, len(()))
print(type({}).__name__)
print(type(set()).__name__)
```

**Output**
```text
int
tuple 1
tuple 0
dict
set
```

### Lists

Square brackets select an item: `labels[0]`. A slice such as `labels[1:]` selects from position 1 through the end and produces a new list. `len` counts the items. `append` adds one item at the end of the same list.

<!-- example: PY04-list -->
```python
labels = ["claim", "premise"]
print(labels[0])
print(labels[1:])
labels.append("claim")
print(labels)
print(len(labels))
```

**Output**
```text
claim
['premise']
['claim', 'premise', 'claim']
3
```

Position 0 is the string `"claim"`. The slice starts at position 1, so it contains only `"premise"`. `append` changes `labels` itself. The length is then 3, and the last item repeats the first. A list keeps order and keeps duplicates.

### Dictionaries

A **dictionary** maps a key to a value. `counts["claim"]` looks up a key. Assigning `counts["premise"] = 1` stores a new pair. `.get(key, 0)` returns the stored value, or 0 when the key is absent, without raising an error. `in` tests whether a key is present.

<!-- example: PY04-dict -->
```python
counts = {"claim": 2}
counts["premise"] = 1
print(counts["claim"])
print(counts.get("other", 0))
print("claim" in counts)
```

**Output**
```text
2
0
True
```

Looking up a missing key with `counts["other"]` raises `KeyError`. `.get` is the version that supplies a fallback. For ordinary keys, a dictionary lookup does not scan every pair. Scanning a list for membership does grow with the list’s length. That is a reason to use a dictionary or a set when the question is “have I seen this key?”

### Sets and tuples

A **set** stores distinct items. Writing the same item twice still leaves it once. A set does not promise an order, so sort it before printing when the lesson needs a stable line. A **tuple** is an ordered sequence that does not support `append`. Use it for a fixed pair, such as a record of two fields.

<!-- example: PY04-set-tuple -->
```python
unique = {"claim", "premise", "claim"}
print(sorted(unique))
pair = ("claim", "premise")
print(pair[0], len(pair))
```

**Output**
```text
['claim', 'premise']
claim 2
```

`sorted` takes the set’s items and returns a new list in sorted order. The tuple’s position 0 is `"claim"`, and its length is 2. `pair[0] = "other"` would fail because a tuple does not allow item assignment.

### Stacks and queues

A **stack** is a list you add to and remove from at the same end. `append` puts an item on top. `pop` with no argument removes that top item and returns it. The last item added is the first one removed.

A **queue** is a line. New items go at the back, and the front item leaves first. `pop(0)` removes the item at position 0. On a long list that shift is slow. A `deque`, shown in the last example, removes the front without shifting every later item.

<!-- example: PY04-stack -->
```python
stack = []
stack.append("first")
stack.append("second")
print(stack.pop())
print(stack)
```

**Output**
```text
second
['first']
```

`"second"` was appended last, so `pop` returns it. The list that remains is `["first"]`.

<!-- example: PY04-queue -->
```python
queue = ["first", "second"]
print(queue.pop(0))
print(queue)
```

**Output**
```text
first
['second']
```

`pop(0)` returns the front item, `"first"`. `"second"` is still waiting. That is the opposite end from `pop()` with no argument.

`enumerate` pairs each position with its item. `zip` pairs items from two sequences and stops at the shorter one. `zip(..., strict=True)` raises an error when the lengths differ, which is what you want when a mismatch is a bug.

### The collections library

`collections` comes with Python. It holds containers that a list or a dictionary does not already provide. People use it because counting items and taking from the front of a line are ordinary jobs, and the module supplies an object for each. `Counter(labels)` counts. A `deque` drops the front item without sliding every later item forward. The call is one constructor. The module does not decide which items were worth counting, and it does not replace the list, dictionary, set, or tuple above.

<!-- example: PY04-main -->
```python
from collections import Counter, deque

labels = ["claim", "premise", "claim"]
counts = Counter(labels)
print(sorted(counts.items()))
print(list(enumerate(labels[:2])))
print(sorted(set(labels)))
queue = deque(["first", "second"])
print(queue.popleft())
```

**Output**
```text
[('claim', 2), ('premise', 1)]
[(0, 'claim'), (1, 'premise')]
['claim', 'premise']
first
```

`Counter(labels)` does the counting you could also write with a dictionary and `+= 1`. `.items()` produces the key/value pairs, and `sorted` makes their printed order stable. `labels[:2]` is a list of the first two labels. `enumerate` turns them into `(0, "claim")` and `(1, "premise")`. The set removes the repeated `"claim"`. `popleft` returns and removes the front of the queue, `"first"`.

**Practice.** Build the same counts with the dictionary operations from the second example: start from `{}`, and for each label add 1 to `.get(label, 0)`. The `Counter` line is that loop already written.

Sources: [Python data structures](https://docs.python.org/3/tutorial/datastructures.html), [collections](https://docs.python.org/3/library/collections.html).

## PY05. Names, mutation and copies

**Mutation** changes an existing object. **Rebinding** attaches a name to a different object. `alias = labels` does not copy the list. Both names refer to one list, so a mutation through either name is visible through the other.

<!-- example: PY05-alias -->
```python
labels = ["claim"]
alias = labels
alias.append("premise")
print(labels)
```

**Output**
```text
['claim', 'premise']
```

`append` changes the list object. `labels` and `alias` are two names for that object, so the print shows both items. `alias = ["premise"]` would be different: that line would attach `alias` to a new list and leave the original list alone.

`.copy()` makes a new outer list. It does not copy objects stored inside that list. A repeated construction such as `[[0]] * 2` also shares the inner list.

<!-- example: PY05-repeated-row -->
```python
rows = [[0]] * 2
rows[0].append(9)
print(rows)
```

**Output**
```text
[[0, 9], [0, 9]]
```

Both rows are the same inner list, so appending 9 appears twice. Build each row separately when the rows should be independent: `rows = [[0] for _ in range(2)]`.

The next example puts the alias, the outer copy and the shallow nested copy in one place.

<!-- example: PY05-main -->
```python
labels = ["claim", "premise"]
alias = labels
alias.append("other")
independent = labels.copy()
independent.append("unknown")
print(labels)
print(independent)

rows = [[0], [1]]
shallow = rows.copy()
shallow[0].append(9)
print(rows)
```

**Output**
```text
['claim', 'premise', 'other']
['claim', 'premise', 'other', 'unknown']
[[0, 9], [1]]
```

The first append changes the list also named `labels`. `independent` is a different list, so its extra item appears only in the second print. `rows.copy()` copies the outer list and still shares the inner lists. Appending 9 through `shallow[0]` changes `rows[0]` as well. A deep copy copies the nested objects too, when that is the copy you meant.

A function that receives a list can mutate the caller’s list without returning it. Returning a newly built list is a different contract. Say which one a function does. NumPy views and tensors raise the same question later: which name owns independent storage?

**Practice.** After `independent = labels.copy()`, appending to `labels` leaves `independent` unchanged. After `shallow = rows.copy()`, appending to `shallow[0]` still changes `rows`. The difference is which object `.copy()` duplicated.

Sources: [Python names and objects](https://docs.python.org/3/tutorial/classes.html), [copy](https://docs.python.org/3/library/copy.html).

## PY06. Functions and arguments

A **function** names an operation you can run again. The names in the definition are **parameters**. The values in a call are **arguments**. `return` sends a value back to the caller. `print` only displays a value. The caller cannot use a printed line as the function’s result.

<!-- example: PY06-simple -->
```python
def total_minutes(morning, evening):
    return morning + evening

print(total_minutes(20, 15))
```

**Output**
```text
35
```

The definition does not add anything yet. The call attaches 20 to `morning` and 15 to `evening`, runs the body, and `return` hands 35 back to `print`. Names assigned inside the function belong to that call. Pass a setting in as a parameter when the caller should be able to see and change it.

A name assigned inside a function does not replace a name outside it. The function’s `threshold` is a different variable from the one written before `def`.

<!-- example: PY06-scope -->
```python
threshold = 0.5

def above(score):
    threshold = 0.8
    return score >= threshold

print(above(0.6))
print(threshold)
```

**Output**
```text
False
0.5
```

Inside `above`, `threshold` is 0.8, so 0.6 is not above it and the function returns false. After the call, the outer `threshold` is still 0.5. The assignment inside the function created a local name.

`*numbers` collects every positional argument into a tuple. `sum` adds that tuple. Use this when the caller may pass any number of values. When you know the names, ordinary parameters are clearer.

<!-- example: PY06-star -->
```python
def total(*numbers):
    return sum(numbers)

print(total(1, 2, 3))
```

**Output**
```text
6
```

The call passes three arguments. `numbers` is `(1, 2, 3)`. Their sum is 6. `**kwargs` does the same job for named arguments and stores them in a dictionary. Libraries use both. Prefer `morning` and `evening` when those are the real inputs.

A parameter after `*` must be passed by name. That keeps a call readable when several numbers would otherwise be easy to swap. A default such as `threshold=0.5` applies when the caller omits that argument.

<!-- example: PY06-main -->
```python
def count_above(scores, *, threshold=0.5):
    return sum(score >= threshold for score in scores)

print(count_above([0.1, 0.5, 0.9]))
print(count_above([0.1, 0.5, 0.9], threshold=0.8))

def add_label(label, labels=None):
    if labels is None:
        labels = []
    labels.append(label)
    return labels

print(add_label("claim"))
print(add_label("premise"))
```

**Output**
```text
2
1
['claim']
['premise']
```

`score >= threshold` produces a Boolean. `sum` counts the true ones, because `True` counts as 1. The first call uses the default 0.5, so 0.5 and 0.9 qualify. The second call passes `threshold=0.8` by name, so only 0.9 qualifies. The expression inside `sum` is a compact loop. PY07 shows that form again. You could also write an ordinary `for` loop and a counter. The result would be the same.

In `add_label`, `None` means “start a new list.” Each call that omits `labels` takes the first branch and builds its own list, which is why the two prints do not share items. A default written as `labels=[]` would be created once, when the function is defined, and later calls would append to that same list.

`*args` collects extra positional arguments into a tuple. `**kwargs` collects extra named arguments into a dictionary. Libraries use both. Prefer ordinary named parameters when you know the arguments.

### Recursion needs a stopping case

A recursive function calls itself on a smaller input. It still needs a case that returns without another call.

<!-- example: PY06-recursion -->
```python
def factorial(number):
    if number < 0:
        raise ValueError("factorial needs a nonnegative integer")
    if number == 0:
        return 1
    return number * factorial(number - 1)

print(factorial(4))
```

**Output**
```text
24
```

The calls reach 0 and return 1, then multiply `1×1×2×3×4`. Without the zero case, the calls would continue until Python’s recursion limit. A long sequence of records is usually clearer as a loop than as one recursive call per record.

**Practice.** `count_above([0.1, 0.5], threshold=0.5)` returns 1. Omitting the name and writing `count_above([0.1, 0.5], 0.5)` is a type error, because `threshold` was declared keyword-only.

Sources: [Python function definitions](https://docs.python.org/3/tutorial/controlflow.html#defining-functions), [Think Python: functions](https://allendowney.github.io/ThinkPython/chap03.html).

## PY07. Iteration and generators

A `for` loop needs an **iterable**: an object that can produce values one at a time. An **iterator** remembers how far that production has gone. A list is iterable, and it also stores every item so you can index it and read `len`. A **generator** produces values on request and does not store the whole series.

`yield` pauses a function and hands one value to the caller. Calling the function does not run it to the end. It returns an iterator. `next` asks that iterator for one more value.

<!-- example: PY07-generator -->
```python
def countdown(start):
    while start > 0:
        yield start
        start = start - 1

steps = countdown(2)
print(next(steps))
print(list(steps))
```

**Output**
```text
2
[1]
```

`countdown(2)` creates the iterator and yields nothing yet. `next` runs until the first `yield` and receives 2. `list` then asks for the rest, receives 1, and the loop ends. A second `list(steps)` would be empty. Start again with `countdown(2)` if you need the series twice.

A list comprehension builds a whole list with a compact loop: `[n * n for n in range(4)]` is `[0, 1, 4, 9]`. A generator expression uses parentheses and produces values one at a time. Neither form is automatically faster. Use an ordinary loop when several conditions would make the one-line form hard to read.

<!-- example: PY07-main -->
```python
def nonempty(lines):
    for line in lines:
        if line.strip():
            yield line.strip()

cleaned = nonempty([" claim ", "", "premise"])
print(next(cleaned))
print(list(cleaned))
print(list(cleaned))
print([n * n for n in range(4)])
```

**Output**
```text
claim
['premise']
[]
[0, 1, 4, 9]
```

`strip` removes whitespace at the ends. The empty string is skipped because an empty string is false. `next` returns the first kept line. `list` consumes the remaining kept line. The following `list` finds the generator exhausted. The comprehension is a separate expression. It builds a new list of squares and does not use `cleaned`.

A streaming dataset uses the same idea: produce or read one record at a time when holding every record at once is unnecessary. An iterator supports “give me the next item.” It does not automatically support indexing, `len` or a second pass.

Sources: [Python iterators and generators](https://docs.python.org/3/tutorial/classes.html#iterators), [Python comprehensions](https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions).

## PY08. Text, bytes and files

A Python `str` is a sequence of Unicode characters. **Bytes** are the stored form of those characters under an **encoding**. UTF-8 is the encoding this course uses for text files. `text.encode("utf-8")` returns bytes. `encoded.decode("utf-8")` returns a string again. A character index, a byte index and a model-token position are three different counts.

<!-- example: PY08-encode -->
```python
text = "café"
encoded = text.encode("utf-8")
print(len(text), len(encoded))
print(encoded.decode("utf-8"))
```

**Output**
```text
4 5
café
```

`"café"` has four characters: `c`, `a`, `f` and `é`. UTF-8 stores `é` in two bytes, so the byte string has length 5. Decoding those bytes with UTF-8 returns the original four characters. Decoding them with a different encoding can fail or produce different characters. The length gap is why a byte offset into a file is not automatically a character offset into the string.

### The same letters can be different strings

Unicode sometimes has two spellings of one visible word. A precomposed `é` is one character. An `e` followed by a combining accent is two characters. They can look alike and still fail `==`.

<!-- example: PY08-normalize -->
```python
composed = "café"
decomposed = "cafe\u0301"
print(composed == decomposed)
print(len(composed), len(decomposed))
```

**Output**
```text
False
4 5
```

This is a deeper detail, not a requirement for reading the next file example. Normalize text before comparing it when the distinction matters. Tokenizers later in the course also split on their own rules, which are neither character counts nor byte counts.

### Paths, JSON and temporary directories

A **path** names a place in the filesystem. It is separate from the contents of the file.

`pathlib` comes with Python. `Path` builds a location with `/`, and the same spelling is the one this course uses on every operating system it cares about. `open` reads or writes the contents. The mode `"w"` replaces a file. A missing `encoding` on a text file uses a platform default, so pass `encoding="utf-8"` when the file is UTF-8. `with` closes the file when the block ends, including when an error occurs inside the block. A path object will not check that the file’s contents are the records you meant.

`json`, also part of Python, writes a dictionary or a list as text and reads that text back. It is widely used because programs in other languages can read the same text without learning a private format. `json.dump` writes. `json.load` reads. JSONL is one JSON value per line, so a program can handle one record at a time. The module checks that the text is valid JSON. It does not check that the values are true.

`tempfile` creates a directory for the duration of a block and removes it afterward. Lessons use it so a checked example can write a file without leaving that file in the project. The directory is real while the block runs. It is not a database and not a backup.

<!-- example: PY08-main -->
```python
import json
from pathlib import Path
from tempfile import TemporaryDirectory

text = "café"
print(len(text), len(text.encode("utf-8")))
with TemporaryDirectory() as folder:
    path = Path(folder) / "record.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"text": text, "label": "claim"}, handle, ensure_ascii=False)
    with path.open(encoding="utf-8") as handle:
        record = json.load(handle)
    print(record["text"], record["label"])
```

**Output**
```text
4 5
café claim
```

The first line repeats the character/byte count. The temporary directory exists only for this example, so the lesson leaves no file behind. `ensure_ascii=False` allows the JSON text to contain `é` directly. `record` is a dictionary, and the keys `"text"` and `"label"` select the two stored values.

A relative path is resolved from the process’s current working directory, which is not necessarily the folder that contains the script. CSV needs a CSV parser rather than `split(",")`, because a quoted field may contain a comma. Do not `pickle.load` an untrusted file. Loading a pickle can run code.

**Practice.** After `encode("utf-8")`, call `.decode("utf-8")` before any string operation such as `len` on characters or `split`. Calling string methods on the bytes object is a different type, with different operations.

Sources: [Python files and JSON](https://docs.python.org/3/tutorial/inputoutput.html), [pathlib](https://docs.python.org/3/library/pathlib.html).

## PY09. Exceptions and debugging

A **syntax error** means Python could not parse the file, so none of it ran. A **runtime exception** happens while a parsed program is running. A **logic error** produces the wrong answer and may print it with no exception at all.

<!-- example: PY09-logic -->
```python
total = 0
for value in [2, 4]:
    total = value
print(total)
```

**Output**
```text
4
```

This program is valid Python. The body replaces `total` with the current value instead of adding. The printed 4 is the last value, not the sum 6. The repair is `total = total + value`. When a result looks wrong, inspect the first value that stopped matching your expectation. Types, lengths and one small input are usually enough to find that line.

`raise` creates an exception when a function can explain what is wrong with its input. `try` runs code that might raise. `except` handles the types you name. The traceback lists the calls that led there. The last line names the immediate exception.

<!-- example: PY09-main -->
```python
def positive_length(value):
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("length must be an integer")
    if value <= 0:
        raise ValueError("length must be positive")
    return value

for candidate in [4, 0, "4"]:
    try:
        print(positive_length(candidate))
    except (TypeError, ValueError) as error:
        print(type(error).__name__, str(error))
```

**Output**
```text
4
ValueError length must be positive
TypeError length must be an integer
```

`4` is a positive integer, so it is printed. `0` fails the second check. `"4"` is text, so it fails the type check. The Boolean test is there because `bool` is a special case of an integer in Python: `True` would pass a plain `isinstance(value, int)` check. You can treat that as a library quirk to guard against, not as a reason to avoid Booleans in ordinary conditions.

Catch the exception you know how to handle. An empty `except:` also catches typos, missing files and memory errors, and the program continues as if they were bad user input. Assertions can state an assumption during development. Python can be run in a mode that skips them, so they are not the validation for untrusted input.

Sources: [Python exceptions](https://docs.python.org/3/tutorial/errors.html), [Python debugger](https://docs.python.org/3/library/pdb.html).

## PY10. Modules and environments

A **module** is an importable collection of names, often one `.py` file. `import math` runs that module and attaches it to the name `math`. `math.sqrt` is one function inside it. The dot means “look up this name in that module.” Installing a package and importing a module are different steps. Install puts files into an environment. Import loads a module that the running interpreter can already find.

The interpreter that runs a file can differ from the interpreter you used for `pip`. A successful install into one environment leaves `import` failing in another. Check the interpreter you are about to use, then install with that same interpreter.

The setup sequence, typed in the shell rather than inside a `.py` file, is:

1. `python3 -m venv .venv` creates a project environment in a `.venv` folder.
2. Activate it, or call the environment’s interpreter by path. Activation changes what the command `python` means in that shell.
   - macOS and Linux, in bash or zsh: `source .venv/bin/activate`
   - Windows Command Prompt: `.venv\Scripts\activate`
   - Without activation, macOS and Linux can use `.venv/bin/python`, and Windows can use `.venv\Scripts\python`.
3. `python -m pip install some-package` installs into that environment, because `python -m` uses the interpreter that is active. The same idea without activation is `.venv/bin/python -m pip install some-package` on macOS and Linux.
4. `python your_script.py` runs the script with the same interpreter. `python -m pip --version` reports the pip belonging to that interpreter.

`sys.executable` is the path of the interpreter running the process. That path differs by machine and by where the environment was created, so this lesson does not record it as a checked output. The major version below only says which generation of the language is running. It does not say which folder supplied the packages.

### sys and importlib

`sys` and `importlib` come with Python. `sys` describes the interpreter that is running this process. `importlib` looks up modules that interpreter can load. People use them when an install seemed to succeed and `import` still fails, because the failure is often a second interpreter. `sys.version_info.major` reports the language generation. `find_spec` reports whether a module can be imported. A missing module returns `None`, which is the situation an install into the wrong environment produces. These calls inspect the process. They do not install a package and they do not switch you into another environment.

<!-- example: PY10-interpreter -->
```python
import sys
from importlib.util import find_spec

print(sys.version_info.major)
print(find_spec("json") is not None)
print(find_spec("this_package_is_not_installed") is None)
```

**Output**
```text
3
True
True
```

This process is Python 3. `json` is in the standard library, so its spec exists. The invented package name has no spec. Importing that name would raise `ModuleNotFoundError`. The repair is to install it with the same `python` that runs the file, or to activate the environment where it is already installed.

<!-- example: PY10-main -->
```python
import math
from importlib.util import find_spec

print(math.sqrt(9))
print(find_spec("json") is not None)
print("This block is Python, not a pip command.")
```

**Output**
```text
3.0
True
This block is Python, not a pip command.
```

`math.sqrt(9)` looks up `sqrt` in the `math` module and returns the float 3.0. The printed sentence is a reminder that `pip` belongs in the shell. `pyproject.toml` can declare the packages a project needs. A constraints file can pin versions you have tested. Neither file installs anything by itself.

When a module contains `if __name__ == "__main__":`, that block runs when you execute the file as a program. It does not run when another file imports the module. Avoid naming your own file `json.py` or `torch.py`. Python may import your file instead of the library you meant.

Sources: [Python modules](https://docs.python.org/3/tutorial/modules.html), [packaging guide](https://packaging.python.org/en/latest/tutorials/installing-packages/).

## PY11. Classes

`def` creates a function. The line `def double(number):` says the function’s name and the value it expects. The indented body runs only when something calls `double`. `return` sends the result back.

<!-- example: PY11-def -->
```python
def double(number):
    return number * 2

print(double(4))
```

**Output**
```text
8
```

Nothing is doubled at the `def` line. `double(4)` attaches 4 to `number`, multiplies by 2, and `print` displays 8.

A **class** defines a kind of object. An **instance** is one object of that kind. `__init__` is a function that runs when you call the class. Its first parameter is `self`, the new instance. `self.text = text` stores the argument on that instance. After `note = Note("hello")`, `note.text` reads the stored value back.

<!-- example: PY11-init -->
```python
class Note:
    def __init__(self, text):
        self.text = text

note = Note("hello")
print(note.text)
```

**Output**
```text
hello
```

`Note("hello")` creates an instance, calls `__init__`, and stores `"hello"` on `self.text`. The print reads that attribute. `__init__` is the initialization method. This class has no further method that you would call on the note. A method is another `def` written inside the class. Calling `first.words()` passes `first` in as `self`.

<!-- example: PY11-class -->
```python
class Record:
    def __init__(self, text, label):
        self.text = text
        self.label = label

    def words(self):
        return self.text.split()

first = Record("A short claim", "claim")
print(first.words())
print(first.label)
```

**Output**
```text
['A', 'short', 'claim']
claim
```

`Record(...)` creates an instance and runs `__init__`. `first.label` reads the attribute stored on that instance. `words` calls `split` on the instance’s own text. `split` with no argument splits on whitespace, so the three words become a list of three strings. A second `Record(...)` is a different instance, with its own attributes.

Put per-instance data on `self`. A list stored on the class itself, rather than on `self`, is shared by every instance.

A subclass is a new class that starts from an existing one. `class LoudNote(Note)` says `LoudNote` is a kind of `Note`. `super().__init__(text)` runs `Note`’s `__init__`, so `self.text` is stored the same way. `shout` is an extra `def` that only the subclass has.

<!-- example: PY11-inherit -->
```python
class Note:
    def __init__(self, text):
        self.text = text

class LoudNote(Note):
    def __init__(self, text):
        super().__init__(text)

    def shout(self):
        return self.text.upper()

note = LoudNote("hello")
print(note.shout())
```

**Output**
```text
HELLO
```

`LoudNote("hello")` calls `LoudNote.__init__`, which calls `Note.__init__`. `self.text` is `"hello"`. `upper` returns a new string in capitals. Inherit when the parent already does something you want to keep. A class that only needs its own fields does not need a parent.

A **decorator** is a callable applied to the thing defined on the next line. `@name` above a function or a class means “pass that function or class through `name`, and bind the result to the same name.” The example below is the common case: a function that receives a function and returns a wrapper. `@add_exclamation` above `def greet` passes `greet` through `add_exclamation`. The wrapper calls the original function, then adds `"!"`. `@dataclass`, later in this lesson, decorates a class rather than wrapping one call.

<!-- example: PY11-decorator -->
```python
def add_exclamation(function):
    def wrapper(name):
        return function(name) + "!"
    return wrapper

@add_exclamation
def greet(name):
    return "Hello " + name

print(greet("Ada"))
```

**Output**
```text
Hello Ada!
```

`greet("Ada")` runs `wrapper`, not the original body alone. `function(name)` returns `"Hello Ada"`. The wrapper joins `"!"` and returns `"Hello Ada!"`. Without the `@` line, `greet("Ada")` would return `"Hello Ada"`.

### The dataclasses library

`dataclasses` comes with Python. A dataclass is a class whose job is to store fields. The module writes the `__init__` you would otherwise type by hand, which is why records in modern Python are often marked with `@dataclass` instead of a hand-written initializer. The mark sits on the line above the class. You still name the fields. The module stores them and builds a readable representation. It does not reject a call that passes a number where the annotation says `str`.

`@dataclass` is the same kind of mark as the decorator above. It generates `__init__` from the fields you declare, so you do not write `self.text = text` yourself. The class below is the same kind of record as the hand-written one.

<!-- example: PY11-main -->
```python
from dataclasses import dataclass

@dataclass
class Record:
    text: str
    label: str

    def words(self):
        return self.text.split()

first = Record("A short claim", "claim")
second = Record("Some evidence", "premise")
print(first.words())
print(second.label)
print(first is second)
```

**Output**
```text
['A', 'short', 'claim']
premise
False
```

`text: str` is an annotation. It documents the intended type. Ordinary Python does not reject a call that passes a number instead. `first is second` is false because the two calls created two objects. `first == second` would ask about values, and a dataclass generates that comparison from the fields.

Composition means one object holds another object. Inheritance means a new class specializes an existing class: `class Classifier(nn.Module)` in a later chapter uses PyTorch’s module as the parent. `super().__init__()` runs the parent’s initializer. Inherit when the parent supplies behavior you are actually extending.

Sources: [Python classes](https://docs.python.org/3/tutorial/classes.html), [dataclasses](https://docs.python.org/3/library/dataclasses.html).

## PY12. Tests and type hints

A **test** runs a function on a chosen input and checks the result. It does not prove the function is right for every input. Start with one ordinary case.

<!-- example: PY12-mean -->
```python
def mean(values):
    return sum(values) / len(values)

print(mean([2.0, 4.0, 6.0]))
```

**Output**
```text
4.0
```

`sum` adds the three numbers to 12. `len` is 3. The division produces 4.0. An empty list would divide by zero. The next version refuses that input, and the test checks both the ordinary case and the empty case.

A **type hint** such as `values: list[float]` and `-> float` documents the intended argument and return type. Python still runs the function if a caller passes something else. A separate checker can read the hints. The hints do not convert values.

<!-- example: PY12-main -->
```python
def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    return sum(values) / len(values)

assert mean([2.0, 4.0]) == 3.0
try:
    mean([])
except ValueError:
    print("empty-input test passed")
print(mean([5.0]))
```

**Output**
```text
empty-input test passed
5.0
```

`assert` raises `AssertionError` if its condition is false. The first assert passes, so it prints nothing. The empty call raises `ValueError`, the `except` prints the success line, and the one-item call returns 5.0. For float results that are approximations, compare with a tolerance, as PY02 did with `math.isclose`, instead of demanding every printed digit.

### pytest

pytest collects functions whose names start with `test_` and reports which assertions failed. People use it because one command can run the checks in a file and stop at the failing line with a traceback. It collects those functions. It does not treat every `assert` in a script as a test, and it does not decide which cases are enough to trust the function.

The example above runs when Python executes the file. Its asserts sit at module level, so importing or running the file runs them immediately. pytest would not collect them, because they are not inside a function named `test_...`. The empty-input branch also prints success only when `ValueError` is raised. If a broken `mean` returned a number for `[]` instead of raising, that `except` would not notice.

A collected test is a function. `pytest.raises(ValueError)` fails the test when the expected exception does not happen. The calls at the bottom of the next example run those two functions directly, so this page can show that they passed. Collection is a separate shell command, which finds `test_mean` and `test_mean_rejects_empty` without those two direct calls:

```shell
python -m pytest test_mean.py
```

<!-- example: PY12-pytest -->
```python
import pytest

def mean(values):
    if not values:
        raise ValueError("cannot average an empty list")
    return sum(values) / len(values)

def test_mean():
    assert mean([2.0, 4.0]) == 3.0

def test_mean_rejects_empty():
    with pytest.raises(ValueError):
        mean([])

test_mean()
test_mean_rejects_empty()
print("2 tests passed")
```

**Output**
```text
2 tests passed
```

`test_mean` checks the ordinary case. `test_mean_rejects_empty` checks that `[]` raises `ValueError`. A wrong return value fails the assert. A missing exception fails `pytest.raises`. A docstring on `mean` should say what the arguments mean, what the return value means, and which errors a caller can expect.

**Practice.** `mean([2.0, 4.0])` is 3.0. Passing `[]` raises `ValueError` before the division. That message is more useful than a `ZeroDivisionError` from `len` being zero.

Sources: [Python annotations](https://docs.python.org/3/tutorial/controlflow.html#function-annotations), [pytest introduction](https://docs.pytest.org/en/stable/getting-started.html).

## PY13. Configuration and randomness

A **configuration** is the set of choices that control a run: a limit, a path, a threshold, a seed. Putting those choices in one dictionary makes them visible. Scattering the same numbers through the script makes two runs hard to compare.

<!-- example: PY13-dict -->
```python
settings = {"limit": 2}
print(settings["limit"])
```

**Output**
```text
2
```

The dictionary is the configuration. The rest of a program should read `settings["limit"]` rather than repeat the number 2.

### argparse and random

`argparse` comes with Python. It is how a script receives settings from the command that starts it, with names and help text in one place. `ArgumentParser` declares the options. `add_argument("--limit", type=int, default=3)` means the option may be omitted, its text is converted with `int`, and omitting it leaves 3. The parser converts and stores the text. It does not decide whether 2 is a sensible limit for the study.

`random` also comes with Python. `random.Random(17)` builds one generator whose sequence is fixed by the seed 17. Lessons use an explicit generator so a later run can repeat the same draws. The seed repeats the sequence for this generator. It does not make a handful of draws represent every case.

<!-- example: PY13-main -->
```python
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=3)
settings = parser.parse_args(["--limit", "2"])
generator = random.Random(17)
values = [generator.random() for _ in range(settings.limit)]
print(settings.limit)
print(len(values), all(0 <= value < 1 for value in values))
```

**Output**
```text
2
2 True
```

`parse_args` normally reads the shell’s command line. Passing a list makes this example self-contained. `type=int` is the conversion from PY02. The text `"2"` becomes the integer 2. `settings.limit` reads that field. `random.Random(17)` builds a generator whose sequence starts from seed 17. The same seed on the same Python version repeats the sequence. It does not promise identical numbers on every library, device or future version. The test checks the count and the range, which is the contract, rather than printing raw decimals as if they were the lesson.

A log records progress and failures. A timer measures an interval. Threads, processes and `async` overlap different kinds of waiting. None of them replaces a NumPy or PyTorch operation that already runs in native code. Keep secrets out of printed configuration.

Sources: [argparse](https://docs.python.org/3/library/argparse.html), [random](https://docs.python.org/3/library/random.html).

## PY14. A complete label report

This program uses the pieces from the earlier lessons: a function, a dictionary-like counter, JSON text, a loop, and a check that rejects a bad record. There is no model. The input is three JSONL lines inside the example, so nothing is downloaded.

Read it in the order it runs. `label_report` receives one string. `splitlines` produces one line per record. `enumerate(..., start=1)` numbers those lines from 1 so an error can name a line. `json.loads` turns one line into a dictionary. The function then checks two fields and counts the label.

<!-- example: PY14-main -->
```python
import json
from collections import Counter

source = '\n'.join([
    '{"text":"It should change.","label":"claim"}',
    '{"text":"The evidence supports this.","label":"premise"}',
    '{"text":"We should wait.","label":"claim"}',
])

def label_report(jsonl: str) -> dict:
    counts = Counter()
    for number, line in enumerate(jsonl.splitlines(), start=1):
        record = json.loads(line)
        if not isinstance(record.get("text"), str):
            raise ValueError(f"line {number}: text must be a string")
        if record.get("label") not in {"claim", "premise"}:
            raise ValueError(f"line {number}: unknown label")
        counts[record["label"]] += 1
    return {"rows": sum(counts.values()), "labels": dict(sorted(counts.items()))}

report = label_report(source)
assert report["rows"] == 3
print(json.dumps(report, sort_keys=True))
```

**Output**
```text
{"labels": {"claim": 2, "premise": 1}, "rows": 3}
```

`Counter()` starts empty. Each accepted label increments one key, which is the dictionary update from PY04. Two lines have the label `"claim"`, so that count is 2. `sorted(counts.items())` makes the label order stable before `json.dumps` prints the dictionary. `sort_keys=True` also sorts the outer keys, so `labels` prints before `rows`. The assert checks the row count. A wrong label would raise `ValueError` and never reach the print.

The checks belong to this task. They are not a universal schema for every dataset. The report counts labels. It does not read the sentences as a classifier.

To use a file, replace `source` with the UTF-8 read from PY08. To save the report, use `json.dump` and a `with` block. The counting function can stay as it is, because reading a path and counting records are separate jobs.

**Next:** NumPy stores numbers in an array with a shape and a storage type. Lists remain the right tool for a short sequence of labels. Arrays are the tool when each operation should apply to a whole grid of numbers.

Sources: [JSON](https://docs.python.org/3/library/json.html), [Counter](https://docs.python.org/3/library/collections.html#collections.Counter).
