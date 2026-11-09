# Data structures and algorithms

This is the last Python section. A data structure lays values out so one operation stays possible and affordable. An algorithm is the procedure, the fact that remains true after every step, and the way the work grows with the input. The sidebar order is the reading order. Lists, functions, and tests from the earlier Python lessons are assumed. NumPy appears again for one selection call. NetworkX stays in the graph-objects lesson; the search lessons use their own dictionaries.

Read the input before the name of the method. The picture above the program shows that input one state at a time. Predict the next state, then use the code and its checked output to confirm it. A comparison count is a count of operations. It is not a clock reading.

Day-to-day code still calls `sorted`, `in`, and `list.append`. These lessons make the cost and the broken case visible. No example downloads a corpus or fits a model. Union-find, Knuth-Morris-Pratt matching, range trees, balanced-tree rotations, flow, and suffix structures are later study. They are named at the end, and they are not taught here.

## DA01. Contracts and cost

One list of document IDs can be asked three different questions. "Does any ID repeat?" returns a yes and the repeated value. "Remove repeats and keep the first copy" returns a shorter list. Those are different contracts. This lesson starts with the first one, shows the second beside it, and counts the work of the obvious scan.

<!-- dsa-scene: DA01 -->

The obvious scan compares each new ID with every earlier ID. On `["d1", "d2", "d1", "d3"]` it can stop at the first repeat. If all 100,000 IDs were distinct, that same shape of scan would compare every pair of positions. The number of those pairs is `n(n-1)/2`.

<!-- example: DA01-scan -->
```python
ids = ["d1", "d2", "d1", "d3"]

def scan_any(ids):
    checks = 0
    for i in range(len(ids)):
        for j in range(i):
            checks += 1
            if ids[i] == ids[j]:
                return True, checks, ids[i]
    return False, checks, None

def keep_first(ids):
    seen = set()
    kept = []
    for item in ids:
        if item not in seen:
            seen.add(item)
            kept.append(item)
    return kept

print(*scan_any(ids))
print(keep_first(ids))
print(100_000 * (100_000 - 1) // 2)
```

**Output**
```text
True 2 d1
['d1', 'd2', 'd3']
4999950000
```

The scan's first comparison is `d2` against `d1`. They differ. The second comparison is the later `d1` against the earlier `d1`, so it stops and reports `d1`. Two comparisons is the cost of this particular list, because a repeat appeared early. `keep_first` answers the other contract: `d1`, then `d2`, then `d3`, with the second `d1` dropped. The last line is the pair count for 100,000 distinct IDs, 4,999,950,000 equality checks. A hash set does about 100,000 membership tests and 100,000 insertions, plus the work of hashing each string. That hash-set count is expected amortized behavior for a well-behaved hash, not a worst-case promise, and hashing a long string still costs work proportional to the length of the string. The useful skill is noticing the nested scan when a first draft hides it.

<!-- example: DA01-growth -->
```python
def pair_checks(n):
    return n * (n - 1) // 2

def halvings(n):
    count = 0
    while n > 1:
        n //= 2
        count += 1
    return count

for n in (4, 8, 16):
    print(n, pair_checks(n), halvings(n))
```

**Output**
```text
4 6 2
8 28 3
16 120 4
```

The columns are the input size, the pair comparisons when every ID is distinct, and the number of times that size can be halved down to 1. Doubling the size from 4 to 8 takes the pair count from 6 to 28, a little more than four times. The halving count moves from 2 to 3. `O(n²)` names that pair-count shape. It leaves out the factor of one half, and it is not a number of seconds. After a comparison that does not match, that pair is known to differ.

**Practice.** If the IDs were `["d1", "d2", "d3"]`, the scan would return false, three comparisons, and no repeated value. `keep_first` would return the same three IDs.

Sources: [MIT 6.006 syllabus](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/pages/syllabus/), [Python operation costs](https://wiki.python.org/moin/TimeComplexity).


## DA02. Lists and copies

A list supports two different lookups. `row[2]` uses the position, so it does not walk the earlier items. `row.index("d2")` walks from the front until it finds a match. Building a new list or a new string can copy values that were already stored. The new idea in this lesson is that copy, not the spelling of a slice.

<!-- dsa-scene: DA02 -->

<!-- example: DA02-copies -->
```python
row = ["d1", "d2", "d3", "d2"]
print(row[2])
print(row.index("d2"))
grown = []
for item in row:
    grown.append(item)
print(grown)
snapshots = []
copied_cells = 0
for item in row:
    snapshots.append((snapshots[-1] + [item]) if snapshots else [item])
    copied_cells += len(snapshots[-1])
print(snapshots)
print(copied_cells)

def growth(n):
    capacity = occupied = copies = 0
    for _ in range(n):
        if occupied == capacity:
            capacity = 1 if capacity == 0 else capacity * 2
            copies += occupied
        occupied += 1
    print(capacity, copies)

growth(4)
growth(5)
naive_copies = 0
built = ""
for item in row:
    naive_copies += len(built) + len(item)
    built = built + item
print(naive_copies)
print(len("".join(row)))
```

**Output**
```text
d3
1
['d1', 'd2', 'd3', 'd2']
[['d1'], ['d1', 'd2'], ['d1', 'd2', 'd3'], ['d1', 'd2', 'd3', 'd2']]
10
4 3
8 7
20
8
```

`row[2]` is `d3`. `index` returns 1, the first `d2`, after looking at the first two cells. The first loop only appends. Its printed list is the one grown list. The second loop builds four new lists, and the next printed line is those lists. Their lengths are 1+2+3+4 = 10. The 10 counts cells written into lists that the second loop actually created. The following two lines are a teaching model of doubling capacity, not a reading of CPython's allocator. Four appends end at capacity 4 after copying 3 cells during growth: the fourth append finds a free slot. A fifth append doubles capacity to 8 and copies the 4 occupied cells, so the growth copies total 7. The last two lines are a model of string copies, not a measurement of every Python build. Writing `built + item` counts every character already in `built`, 20 character writes for these four IDs. `"".join(row)` writes the 8 characters of the result. The explicit addition is there so the copy is visible; some Python builds optimize a different spelling, `+=`, in some loops.

**Practice.** `row.index("d9")` would raise `ValueError` because the scan finishes without a match. A missing position and a missing search are different failures: index 2 exists, and `"d9"` does not.

Sources: [Python lists](https://docs.python.org/3/tutorial/datastructures.html), [Python operation costs](https://wiki.python.org/moin/TimeComplexity).


## DA03. Hash tables

A hash table puts a key in a bucket computed from the key. A later lookup computes the bucket again. It does not start a scan of every stored key, unless a collision has put several keys in that same bucket. The picture below uses a toy bucket: the sum of character codes modulo 4. It is an illustration. It is not CPython's layout, and CPython may cache a string's hash.

<!-- dsa-scene: DA03 -->

A hash table, a cryptographic hash, and a sentence-similarity score are three tools. The table answers "have I stored this key?" A cryptographic hash is built so a tampered message is hard to disguise. A similarity score answers a graded question about meaning. Sharing the word "hash" does not make them the same function.

<!-- example: DA03-buckets -->
```python
def toy_bucket(key):
    return sum(ord(ch) for ch in key) % 4

print([toy_bucket(key) for key in ["cat", "dog", "act"]])
order = []
seen = {}
for item in ["d1", "d2", "d1", "d3"]:
    if item not in seen:
        seen[item] = len(order)
        order.append(item)
print(order)
print(seen["d1"])
print("d9" in seen)
print(sum(len(item) for item in ["d1", "d2", "d1", "d3"]))
```

**Output**
```text
[0, 2, 0]
['d1', 'd2', 'd3']
0
False
8
```

`cat` and `act` both land in bucket 0. `dog` lands in bucket 2. The shared bucket is a collision: the bucket has to keep both keys, or a lookup cannot tell them apart. The dictionary then keeps the first copy of each ID. `d1` is stored at position 0, the later `d1` is ignored, and `d9` is absent. The last line is 8, the total number of characters in the four IDs. It is a model of text a hash might read. It is not a measurement of CPython, which can reuse a cached hash. Expected amortized membership is the cost model for a well-behaved hash. It is not a promise for every collision pattern, and it does not make a long key free.

**Practice.** `"dog"` and `"cat"` do not collide under this toy function. A cryptographic digest of the same two words would be a different calculation, and neither number says the words have similar meanings.

Sources: [dict](https://docs.python.org/3/library/stdtypes.html#mapping-types-dict), [Python operation costs](https://wiki.python.org/moin/TimeComplexity).


## DA04. Sorting

Sorting here means a comparison sort that may also be stable: two equal values keep the order they had in the input. The tagged numbers `("a", 4)` and `("c", 4)` are the test. A finished sort of these four tags is `d b a c`. Insertion sort shows the comparisons one shift at a time. Merge sort shows the split and the combine. A last-element partition then shows a cut that shrinks the problem by only one.

<!-- dsa-scene: DA04 -->

`sorted` is the function ordinary code should call. Python's sort is stable. The traces exist so the bound and the stability rule are visible.

### Insertion

<!-- example: DA04-insertion -->
```python
items = [("a", 4), ("b", 2), ("c", 4), ("d", 1)]

def insertion_sort(items):
    data = list(items)
    comparisons = 0
    for i in range(1, len(data)):
        current = data[i]
        j = i
        while j > 0:
            comparisons += 1
            if data[j - 1][1] <= current[1]:
                break
            data[j] = data[j - 1]
            j -= 1
        data[j] = current
    return [name for name, _score in data], comparisons

names, comparisons = insertion_sort(items)
print(" ".join(names))
print(comparisons)
print(" ".join(name for name, _score in sorted(items, key=lambda item: item[1])))
```

**Output**
```text
d b a c
5
d b a c
```

`b` slides in front of `a` because 2 is smaller than 4. `c` stays after `a` because the comparison uses `<=`: an equal value on the left is already in the right place, so `c` does not jump over `a`. `d` then walks to the front. Five comparisons is the work on this list. The library sort prints the same names. On a reversed list, insertion sort compares about one pair per distance an item travels, which grows like the square of the length. An already sorted list stops each walk on the first comparison.

### Merging

<!-- example: DA04-merge -->
```python
items = [("a", 4), ("b", 2), ("c", 4), ("d", 1)]
copies = 0
combines = []

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i][1] <= right[j][1]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def merge_sort(items):
    global copies
    if len(items) <= 1:
        return list(items)
    mid = len(items) // 2
    left, right = items[:mid], items[mid:]
    copies += len(left) + len(right)
    combined = merge(merge_sort(left), merge_sort(right))
    combines.append(" ".join(name for name, _score in combined))
    return combined

print(" ".join(name for name, _score in merge_sort(items)))
for line in combines:
    print(line)
print(copies)
```

**Output**
```text
d b a c
b a
d c
d b a c
8
```

The first combine is the left half, `a` then `b`, merged into `b a`. The second is `c` then `d`, merged into `d c`. The last combine uses `<=`, so `a` stays ahead of `c` when both scores are 4. Four items split into two levels. Each level copies every item into a slice, and those slice copies total 8. The merges write the items again. The shape is a logarithm number of levels times a linear amount of work per level, plus the extra copies from slicing. An index-based merge can avoid some of those slice copies. The comparison count and the copy count are still different bills.

### A pivot that fails to cut

<!-- example: DA04-partition -->
```python
def partition(values):
    data = list(values)
    pivot = data[-1]
    i = -1
    for j in range(len(data) - 1):
        if data[j] <= pivot:
            i += 1
            data[i], data[j] = data[j], data[i]
    data[i + 1], data[-1] = data[-1], data[i + 1]
    return data, i + 1

arranged, index = partition([3, 1, 4, 2])
print(arranged)
print(index)
arranged, index = partition([1, 2, 3, 4])
print(arranged)
print(index)
```

**Output**
```text
[1, 2, 4, 3]
1
[1, 2, 3, 4]
3
```

The first pivot is 2. After the swaps, everything before index 1 is no greater than 2, and everything after it is greater. That is the partition fact. The second pivot is 4, already the largest value, and it lands at index 3. The left piece still has 3 items. A quicksort that always pivots on the last item of an already sorted list therefore shrinks the problem by one, which is the slow shape, not a guaranteed halving. Expected halving needs a pivot that actually lands in the middle often enough. The worst arrangement does not.

**Practice.** Change the merge comparison to `<` instead of `<=` and the two 4s can change order: `c` may be written before `a`. The multiset of values is still sorted. The tags are not stable.

Sources: [Sorting](https://docs.python.org/3/howto/sorting.html), [Open Data Structures](https://opendatastructures.org/).


## DA05. Binary search

The contract is a lower bound. In a sorted list, return the first index whose value is at least the target. If every value is smaller, return the length of the list. The list is `[2, 4, 4, 7, 9]` and the target is 4. The answer is index 1, the first 4.

<!-- dsa-scene: DA05 -->

The unresolved interval is half-open: `lo` is included and `hi` is excluded. `hi` may equal the length, and that position is never read. Every index before `lo` holds a value below the target. Indexes at or after `hi` are no longer unresolved, and the boundary stored in `hi` can still be the answer. Sorted order is what makes one comparison speak for a whole side.

<!-- example: DA05-bound -->
```python
import bisect

def lower_bound(values, target):
    lo, hi = 0, len(values)
    while lo < hi:
        mid = (lo + hi) // 2
        if values[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo

values = [2, 4, 4, 7, 9]
index = lower_bound(values, 4)
print(index)
print(lower_bound(values, 5))
print(lower_bound(values, 10))
print(lower_bound([], 4))
print(bisect.bisect_left(values, 4))
print(index < len(values) and values[index] == 4)
absent = lower_bound(values, 5)
print(absent < len(values) and values[absent] == 5)
```

**Output**
```text
1
3
5
0
1
True
False
```

The picture reads `values[2]` first. That cell is 4, so `hi` becomes 2. The next middle cell is also 4, and `hi` becomes 1. `values[0]` is 2, which is below the target, so `lo` becomes `mid + 1`. Setting `lo = mid` there would leave `lo` at 0 while `hi` is 1, and the same middle cell would be chosen forever. When `lo` and `hi` meet, no cell is read. The printed boundary is 1.

Target 5 is absent. The boundary 3 points at 7, the first value at least 5, and the exact-membership test is false because that cell is not 5. Target 10 is larger than every cell, so the boundary is 5, the length. An empty list returns 0 without a comparison. `bisect_left` agrees with the boundary for target 4. The logarithmic count assumes the list is sorted and that reading `values[mid]` is cheap. Copying a slice on every recursive call would add a different bill, the one the merge trace already counted.

**Practice.** The inclusive-hi version, where both ends are legal indexes, is a different program. Its stop rule and its picture of `hi` are different. Mixing that picture with this half-open trace produces an off-by-one even when the final index sometimes matches.

Sources: [bisect](https://docs.python.org/3/library/bisect.html).


## DA06. Two pointers

On a sorted list, a pair of indexes can search for a target sum without trying every pair. The left index starts at the small end and the right index starts at the large end. If the sum is too small, the left value is the only one that can raise it. If the sum is too large, the right value is the only one that can lower it. Each step moves one index inward, so the number of moves is at most the length of the list. Two nested loops are a different program; they are not automatically this one, and they are not automatically quadratic if the inner loop does not run to the end every time.

<!-- dsa-scene: DA06 -->

<!-- example: DA06-pair -->
```python
def pair_sum(values, target):
    lo, hi = 0, len(values) - 1
    steps = []
    while lo < hi:
        total = values[lo] + values[hi]
        steps.append((lo, hi, total))
        if total == target:
            return (lo, hi), steps
        if total < target:
            lo += 1
        else:
            hi -= 1
    return None, steps

found, steps = pair_sum([1, 2, 4, 7], 6)
for step in steps:
    print(*step)
print(found)
missed, _steps = pair_sum([5, 1, 4, 2], 3)
print(missed)
values = [5, 1, 4, 2]
print(any(values[i] + values[j] == 3 for i in range(len(values)) for j in range(i + 1, len(values))))
```

**Output**
```text
0 3 8
0 2 5
1 2 6
(1, 2)
None
True
```

`1 + 7` is 8, so the right index steps inward. `1 + 4` is 5, so the left index steps inward. `2 + 4` is 6, at indexes 1 and 2. The unsorted list `[5, 1, 4, 2]` still contains `1 + 2`. The same movement rule returns `None`, because "too large, so discard the right end" is justified only by sorted order. The last line is the brute-force proof that a pair exists. The fast rule was not allowed to use it.

**Practice.** An empty list and a one-element list both return `None` for any target sum of two positions. There are not two indexes to hold.

Sources: [Python operation costs](https://wiki.python.org/moin/TimeComplexity).


## DA07. Prefix sums and windows

A prefix array stores the sum of every leading piece. The sum from index 1 up to but not including index 3 is `prefix[3] - prefix[1]`: two reads and one subtraction, however long the piece is. A fixed window of width 2 can keep a running sum: add the cell that enters and subtract the cell that leaves. Both of those updates stay valid when a value is negative, because they do not discard a cell forever.

<!-- dsa-scene: DA07 -->

A variable window that only moves its left edge forward is a different claim. It is valid when every value is nonnegative, because growing the right edge is the only way the sum increases, and shrinking the left edge can be done without ever needing that cell again. A negative value can repair a sum that looked too large. The left edge has already thrown the earlier cell away.

<!-- example: DA07-spans -->
```python
costs = [2, 1, 3, 1]
prefix = [0]
for cost in costs:
    prefix.append(prefix[-1] + cost)
print(prefix)
print(prefix[3] - prefix[1])

def fixed_window(costs, width):
    if width <= 0 or width > len(costs):
        return []
    total = sum(costs[:width])
    totals = [total]
    for i in range(width, len(costs)):
        total += costs[i] - costs[i - width]
        totals.append(total)
    return totals

def one_way(costs, budget):
    total = left = best = 0
    for right, cost in enumerate(costs):
        total += cost
        while total > budget and left <= right:
            total -= costs[left]
            left += 1
        if left <= right:
            best = max(best, right - left + 1)
    return best

def brute(costs, budget):
    best = 0
    for i in range(len(costs)):
        total = 0
        for j in range(i, len(costs)):
            total += costs[j]
            if total <= budget:
                best = max(best, j - i + 1)
    return best

print(fixed_window(costs, 2))
print(fixed_window([5, -2, 3], 2))
print(fixed_window([1, 2], 3))
print(fixed_window([], 2))
print(fixed_window([1, 2], 0))
print(one_way(costs, 4), brute(costs, 4))
print(one_way([5, -3, 1], 4), brute([5, -3, 1], 4))
```

**Output**
```text
[0, 2, 3, 6, 7]
4
[3, 4, 4]
[3, 1]
[]
[]
[]
2 2
2 3
```

The prefix is 0, then 2, 3, 6, and 7. The span of the middle two costs, 1 and 3, is 4. The width-2 windows on the nonnegative costs are 3, 4, and 4. The same update on `[5, -2, 3]` produces 3 and 1. Negative numbers did not break that update. A width of 3 on a list of length 2, a width on an empty list, and a width of 0 each return `[]`. The function does not invent a window for an illegal width.

The budget 4 on `[2, 1, 3, 1]` has longest feasible length 2. The one-way window and the brute force agree. On `[5, -3, 1]` they do not. The one-way window sees 5, which is already over 4, and drops it. It later keeps `[-3, 1]`, length 2. The whole list sums to 3, which is within the budget, so the brute force reports 3. The dropped 5 was useful once `-3` arrived. A token budget of nonnegative costs may use the one-way window. A sum that may contain negatives needs the prefix, or the brute force, or another stated rule. A word count is a different unit from a token cost. This list is a list of numbers.

**Practice.** The prefix sum of an empty leading piece is the stored 0. A window whose width is longer than the list returns `[]`, the same result as a width of 0.

Sources: [Python lists](https://docs.python.org/3/tutorial/datastructures.html).


## DA08. Stacks, queues and links

A stack returns the most recently stored item. A queue returns the earliest one. A linked node stores one value and a reference to another node. Python's `list` can act as a stack with `append` and `pop`. Using `pop(0)` as a queue moves every later item one slot left. `collections.deque` keeps an end that can give up its first item without that slide. The linked-list fact depends on the kind of list: in a singly linked list, deleting a node requires the predecessor, because that is the reference which must change.

<!-- dsa-scene: DA08 -->

<!-- example: DA08-links -->
```python
from collections import deque

def delimiters(text):
    pairs = {")": "(", "]": "["}
    stack = []
    for ch in text:
        if ch in "([":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack

print(delimiters("(a[b])"))
print(delimiters("(]"))
print(delimiters(""))
frontier = deque(["claim"])
frontier.append("evidence")
print(frontier.popleft())
print(list(frontier))
row = ["d1", "d2", "d3"]
print(row.pop(0), len(row))

class Node:
    def __init__(self, value, nxt=None):
        self.value = value
        self.nxt = nxt

head = Node("a", Node("b", Node("c")))
previous, target = head, head.nxt
previous.nxt = target.nxt
values = []
node = head
while node:
    values.append(node.value)
    node = node.nxt
print(values)
```

**Output**
```text
True
False
True
claim
['evidence']
d1 2
['a', 'c']
```

`(a[b])` pushes `(`, pushes `[`, pops `[` for `]`, and pops `(` for `)`. The stack ends empty, so the delimiters match. `(]` pops `(` for `]`, and those marks do not match. The empty string has nothing left open. The deque starts as `claim`, gains `evidence` at the right, and `popleft` returns `claim`. The list `pop(0)` also returns the first item, `d1`, and leaves two items. Those two items each moved one slot. That movement is the cost `pop(0)` adds on a long list.

The link picture starts as `a → b → c`. The code holds `a`, the predecessor, and points it at `c`. The printed list is `a`, `c`. Holding only `b` would not tell the program which node points at `b`. A doubly linked list, with a reference back to the predecessor already stored in the node, has a different contract. This example is the singly linked one.

**Practice.** `([)]` fails even though each mark has a partner. The stack's most recent opener is `[`, and the next closer is `)`.

Sources: [collections.deque](https://docs.python.org/3/library/collections.html#collections.deque), [Python lists](https://docs.python.org/3/tutorial/datastructures.html).


## DA09. Recursion

A recursive call is a smaller copy of the same problem, plus a base case that stops. The call stack is the path of calls that have started and have not returned. It is not the whole tree of calls at once. Returned values travel back to the caller. The second count in this lesson is the warning: a function that branches twice, with no shared subproblem saved, doubles the work at every level.

<!-- dsa-scene: DA09 -->

<!-- example: DA09-calls -->
```python
def trace(n, stack, lines):
    stack.append(n)
    lines.append("call " + " ".join(str(x) for x in stack))
    result = 1 if n == 1 else n * trace(n - 1, stack, lines)
    lines.append("return " + " ".join(str(x) for x in stack) + " " + str(result))
    stack.pop()
    return result

def count_calls(n, counter):
    counter[0] += 1
    if n == 0:
        return 1
    return count_calls(n - 1, counter) + count_calls(n - 1, counter)

lines = []
print(trace(3, [], lines))
for line in lines:
    print(line)
counter = [0]
print(count_calls(3, counter), counter[0])
```

**Output**
```text
6
call 3
call 3 2
call 3 2 1
return 3 2 1 1
return 3 2 2
return 3 6
8 15
```

The stack grows `3`, then `3 2`, then `3 2 1`. The base case returns 1. The caller of 2 multiplies by 2 and returns 2. The caller of 3 multiplies by 3 and returns 6. Each return line still shows the stack of that call, and the `pop` happens after the line is recorded. The active path never contains a finished call.

`count_calls(3)` returns 8 leaf results and performs 15 calls. Each level launches two copies of the smaller problem, and those copies do not reuse each other's work. Three levels are enough to see the doubling. Generating the permutations of a corpus would be the same explosion with a larger branching factor. The next lesson saves the choice so it can be undone. It still does not save a repeated subproblem. That saving is what the edit-distance lesson is for.

**Practice.** `trace(1)` prints one call and one return of 1. There is no smaller call. A missing base case keeps pushing until the interpreter rejects the depth.

Sources: [Functions and calls](https://docs.python.org/3/tutorial/controlflow.html#defining-functions), [Open Data Structures](https://opendatastructures.org/).


## DA10. Backtracking

Backtracking builds a choice, explores what that choice allows, and then undoes it. The undo is the difference between this search and a recursive function that only returns a number. Here the choices are whether to take the next integer, and the limit is a sum of at most 3. A branch whose sum is already over the limit is pruned only because every later item is nonnegative. A later negative item could bring the sum back under the limit.

<!-- dsa-scene: DA10 -->

<!-- example: DA10-choices -->
```python
def subsets(items, limit):
    if any(item < 0 for item in items):
        raise ValueError("nonnegative")
    chosen = []
    lines = []

    def walk(i):
        if sum(chosen) > limit:
            lines.append("prune " + " ".join(map(str, chosen)))
            return
        if i == len(items):
            lines.append("answer" + ("" if not chosen else " " + " ".join(map(str, chosen))))
            return
        chosen.append(items[i])
        lines.append("choose " + str(items[i]))
        walk(i + 1)
        chosen.pop()
        lines.append("undo " + str(items[i]))
        walk(i + 1)

    walk(0)
    return lines

for line in subsets([1, 2, 3], 3):
    print(line)
try:
    subsets([4, -2], 3)
    print("ran")
except ValueError:
    print("rejected")
print(4 + -2)
```

**Output**
```text
choose 1
choose 2
choose 3
prune 1 2 3
undo 3
answer 1 2
undo 2
choose 3
prune 1 3
undo 3
answer 1
undo 1
choose 2
choose 3
prune 2 3
undo 3
answer 2
undo 2
choose 3
answer 3
undo 3
answer
rejected
2
```

`1 2 3` sums to 6, so that branch is pruned and 3 is undone. `1 2` sums to 3, which is allowed, and it is recorded before 2 is undone. `1 3` sums to 4 and is pruned. The empty line `answer` is the subset that took nothing. Every `choose` has a later `undo` of the same value. That pairing is the invariant. A search that appended and forgot to pop would carry the old choice into the next branch and invent subsets that were never selected. `[4, -2]` sums to 2, which is within the limit, but a prune after seeing only 4 would have dropped it. The function rejects that input instead of printing a search that missed a valid subset.

**Practice.** Raising the limit to 6 removes every prune. The same tree then records `1 2 3` as an answer. Lowering it to 0 leaves only the empty answer, after pruning each singleton.

Sources: [MIT 6.006 syllabus](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/pages/syllabus/).


## DA11. Trees and tries

A binary search tree keeps a subtree rule: every key in the left subtree is smaller than the node, and every key in the right subtree is larger. Comparing a node only with its immediate children is not enough. In-order traversal then reads the keys from small to large. The rule does not keep the tree short. Inserting 1, then 2, then 3 builds a line, and a line makes the lookup walk every node. A scikit-learn decision tree is a different object. It splits features to predict a label. It does not store these keys, and it does not promise this order.

<!-- dsa-scene: DA11 -->

A trie stores words along character edges. Lookup of a prefix follows those characters and then lists the words that continue. The trie spends a node per character it stores. That is the memory side of the prefix lookup.

<!-- example: DA11-tree -->
```python
class Tree:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

def insert(node, key):
    if node is None:
        return Tree(key)
    if key < node.key:
        node.left = insert(node.left, key)
    elif key > node.key:
        node.right = insert(node.right, key)
    return node

def inorder(node, keys):
    if node:
        inorder(node.left, keys)
        keys.append(node.key)
        inorder(node.right, keys)

def height(node):
    if node is None:
        return 0
    return 1 + max(height(node.left), height(node.right))

def ordered(node, low=None, high=None):
    if node is None:
        return True
    if low is not None and node.key <= low:
        return False
    if high is not None and node.key >= high:
        return False
    return ordered(node.left, low, node.key) and ordered(node.right, node.key, high)

root = None
for key in [2, 1, 3]:
    root = insert(root, key)
keys = []
inorder(root, keys)
print(keys)
print(height(root))
line = None
for key in [1, 2, 3]:
    line = insert(line, key)
print(height(line))
bad = Tree(5)
bad.left = Tree(1)
bad.left.right = Tree(6)
print(ordered(root))
print(ordered(bad))
```

**Output**
```text
[1, 2, 3]
2
3
True
False
```

The balanced-looking insert order 2, 1, 3 has height 2: the root plus one child. The sorted insert order 1, 2, 3 has height 3, one node per key. Both trees print the same in-order keys if the line is walked the same way. Equal lookup cost is not part of the binary-search-tree property. Balance is a further promise, and the rotations that maintain it are left for later. The tree with root 5, left child 1, and 6 to the right of 1 passes every immediate-child comparison: 1 is smaller than 5, and 6 is larger than 1. It fails the subtree bound because 6 is in the left subtree of 5. `ordered` returns false for that tree and true for the tree built from 2, 1, 3.

<!-- example: DA11-trie -->
```python
def add(node, word):
    for ch in word:
        node = node.setdefault(ch, {})
    node["$"] = True

def words_with(node, prefix):
    for ch in prefix:
        if ch not in node:
            return []
        node = node[ch]
    found = []

    def walk(current, built):
        if "$" in current:
            found.append(built)
        for ch in sorted(k for k in current if k != "$"):
            walk(current[ch], built + ch)

    walk(node, prefix)
    return found

trie = {}
add(trie, "cat")
add(trie, "car")
print(words_with(trie, "ca"))
print(words_with(trie, "do"))
```

**Output**
```text
['car', 'cat']
[]
```

`cat` and `car` share the edges `c` and `a`, then split into `t` and `r`. The prefix `ca` finds both words. The walk sorts the edges, so `car` comes before `cat`. The prefix `do` has no edge and returns an empty list. A hash table of whole words can answer "is `car` stored?" and cannot, by itself, list every word that starts with `ca` without looking at the other keys.

**Practice.** Inserting the key 2 a second time does not add a node. The comparisons are strict, so an equal key is left where it already is. A multiset would need a different rule.

Sources: [Open Data Structures](https://opendatastructures.org/).


## DA12. Heaps

A heap is a tree packed into an array. In this min-heap the parent is no greater than its children. Index 0 is the root. The children of index `i` are `2i+1` and `2i+2`, and the parent of index `i` is `(i-1)//2`. The root is the smallest score, not the largest, and the array read from left to right is not a sorted list.

<!-- dsa-scene: DA12 -->

<!-- example: DA12-shape -->
```python
import heapq

rows = [("a", 1), ("b", 5), ("c", 3), ("d", 4), ("e", 2)]
heap = []
for doc_id, score in rows:
    heapq.heappush(heap, (score, doc_id))
print(" ".join(f"{doc_id}:{score}" for score, doc_id in heap))
last = len(heap) - 1
parent = (last - 1) // 2
print(parent, heap[parent][0])
```

**Output**
```text
a:1 e:2 c:3 b:5 d:4
1 2
```

The array is `a:1`, `e:2`, `c:3`, `b:5`, `d:4`. The root is the smallest score. `b:5` sits before `d:4`, so a left-to-right read is not sorted order. The parent of the last index is index 1, and that entry's score is 2.

The job is different when only the best few scores must be kept. A bounded min-heap stores those few. Its root is the worst of the ones kept, so a new score replaces the root only when it is better. If the scores are already a NumPy array and the whole array fits, `argpartition` selects a cut. The two tools are required to return the same IDs only when the contract says so. Distinct scores make the top three a unique set. Tied scores at the cutoff do not.

The tie policy is: higher score first, and for an equal score the smaller ID wins. `argpartition` may keep either side of a tie. Sorting the IDs it already selected cannot recover an ID it dropped.

<!-- example: DA12-heap -->
```python
import heapq
import numpy as np

class Rank:
    def __init__(self, score, doc_id):
        self.score = score
        self.doc_id = doc_id

    def __lt__(self, other):
        if self.score != other.score:
            return self.score < other.score
        return self.doc_id > other.doc_id

def top_k(rows, k):
    if k <= 0:
        return []
    kept = []
    for doc_id, score in rows:
        heapq.heappush(kept, Rank(score, doc_id))
        if len(kept) > k:
            heapq.heappop(kept)
    return [item.doc_id for item in sorted(kept, reverse=True)]

rows = [("a", 1), ("b", 5), ("c", 3), ("d", 4), ("e", 2)]
print(" ".join(top_k(rows, 3)))
scores = np.array([score for _doc_id, score in rows])
chosen = np.argpartition(scores, -3)[-3:]
order = chosen[np.argsort(scores[chosen])[::-1]]
print(" ".join(rows[i][0] for i in order))
print(" ".join(top_k(rows, 0)) or "none")
print(" ".join(top_k(rows, 9)))
tied = [("a", 4), ("b", 4), ("c", 5)]
print(" ".join(top_k(tied, 2)))
part = np.argpartition(np.array([score for _doc_id, score in tied]), -2)[-2:]
print(" ".join(sorted(tied[i][0] for i in part)))
```

**Output**
```text
b d c
b d c
none
b d c e a
c a
b c
```

The bounded heap and the partition-then-sort both return `b d c` because those three scores are unique. `k` of 0 returns no IDs. `k` larger than the list returns every ID in policy order. On the tie, the policy keeps `c` and then `a`: `a` and `b` share the score 4, and `a` is the smaller ID. This `argpartition` returns the set `b` and `c`. `a` was excluded. Sorting `b` and `c` cannot produce `a`.

**Practice.** A heap of the three winners is still not the fully sorted list of all five. Reading index 0 of the bounded heap shows the worst of the winners, the one that would be replaced next.

Sources: [heapq](https://docs.python.org/3.11/library/heapq.html), [numpy.argpartition](https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html).


## DA13. Graph search

The graph is a dictionary from a node to its neighbors, in a fixed order. Breadth-first search keeps a queue. A node is marked when it is put on the queue, so a later edge back to it does not enqueue it again. Depth-first search follows one neighbor all the way before the next neighbor. Its recursion stack is the current path. The visited set is every node the search has started. Those are different collections.

<!-- dsa-scene: DA13 -->

Breadth-first search gives a shortest path when every edge costs the same, because the queue explores in rings of equal length. The moment edges have different costs, a queue of nodes is not a queue of distances. The NetworkX object for this kind of picture is the [graph-objects lesson](07_OTHER_LIBRARIES.md). This lesson builds the traversal itself.

<!-- example: DA13-search -->
```python
from collections import deque

graph = {
    "claim": ["evidence", "warrant"],
    "evidence": ["detail"],
    "warrant": [],
    "detail": [],
    "note": [],
}

def bfs(start):
    seen = {start}
    queue = deque([start])
    order, snapshots = [], []
    while queue:
        snapshots.append(list(queue))
        node = queue.popleft()
        order.append(node)
        for nxt in graph[node]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return order, snapshots, seen

def dfs(start):
    seen, order = set(), []

    def walk(node):
        seen.add(node)
        order.append(node)
        for nxt in graph[node]:
            if nxt not in seen:
                walk(nxt)

    walk(start)
    return order

order, snapshots, seen = bfs("claim")
print("bfs", *order)
for snap in snapshots:
    print("queue", *snap)
print("missed", *sorted(set(graph) - seen))
print("dfs", *dfs("claim"))
```

**Output**
```text
bfs claim evidence warrant detail
queue claim
queue evidence warrant
queue warrant detail
queue detail
missed note
dfs claim evidence detail warrant
```

The queue starts as `claim`. After `claim` is taken, the neighbors `evidence` and `warrant` are marked and enqueued, in that listed order. `evidence` then contributes `detail`. `warrant` contributes nothing. `note` is never reached, because no edge from the started component points at it. A search that reported only the nodes it saw would quietly drop that component.

Depth-first order is `claim`, `evidence`, `detail`, `warrant`. `detail` is finished before `warrant` starts, because `evidence` is the first neighbor. Marking a node when it is enqueued, in the breadth-first version, is what keeps a cycle from filling the queue with repeat visits. Marking it only when it is dequeued would allow the same node to sit in the queue many times.

**Practice.** Add an edge from `detail` back to `claim`. The breadth-first order stays the same, because `claim` was marked at the start. The edge is still real. The visited set is what stops the repeat, not the absence of the edge.

Sources: [collections.deque](https://docs.python.org/3/library/collections.html#collections.deque), [NetworkX tutorial](https://networkx.org/documentation/stable/tutorial.html).


## DA14. Topological order and Dijkstra

A topological order lists every node after the nodes that must precede it. Several orders can be legal. This lesson forces one snapshot: whenever more than one node is ready, the alphabetically smallest goes next. If a node is left over, the graph contains a cycle. The leftover node may sit on the cycle or downstream of it.

<!-- dsa-scene: DA14 -->

Dijkstra's algorithm is a different question. Each edge has a nonnegative length. The search stores a tentative distance and a predecessor. It settles a node when that node's tentative distance is the smallest unsettled one. Relaxation replaces a stored distance when a path through the current node is shorter. An old heap entry for a worse distance is stale: its number no longer matches the stored distance, so it must not settle the node. A queue with the edges renamed as weights does not become this algorithm. Equal weights were the breadth-first case.

### One legal order

<!-- example: DA14-order -->
```python
def topo(nodes, edges):
    incoming = {node: 0 for node in nodes}
    outs = {node: [] for node in nodes}
    for src, dst in edges:
        outs[src].append(dst)
        incoming[dst] += 1
    ready = sorted(node for node, count in incoming.items() if count == 0)
    order = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for dst in outs[node]:
            incoming[dst] -= 1
            if incoming[dst] == 0:
                ready.append(dst)
                ready.sort()
    if len(order) != len(nodes):
        return "cycle"
    return " ".join(order)

print(topo(["A", "B", "C", "D"], [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]))
print(topo(["A", "B"], [("A", "B"), ("B", "A")]))
```

**Output**
```text
A B C D
cycle
```

`A` starts with no incoming edge. Removing it makes `B` and `C` ready, and the alphabetical rule takes `B` first. `D` waits until `C` has also been taken. `A C B D` is another legal order under a different tie rule. This program's tie rule is why the printed line is `A B C D` rather than that other order. The ready list is sorted again and then uses `pop(0)`, so this snapshot is not the linear-time textbook implementation. The two-node cycle never reduces both incoming counts to zero, so the result is `cycle` instead of a forced order.

### Several routes, and one negative edge

<!-- example: DA14-distance -->
```python
import heapq

def shortest(graph, start, goal):
    dist = {start: 0}
    prev = {start: None}
    heap = [(0, start)]
    settled, skipped = [], []
    done = set()
    while heap:
        distance, node = heapq.heappop(heap)
        if node in done or distance != dist.get(node):
            if distance != dist.get(node):
                skipped.append(f"{node}:{distance}")
            continue
        done.add(node)
        settled.append(f"{node}:{distance}")
        for nxt, weight in graph[node]:
            if nxt in done:
                continue
            candidate = distance + weight
            if nxt not in dist or candidate < dist[nxt]:
                dist[nxt] = candidate
                prev[nxt] = node
                heapq.heappush(heap, (candidate, nxt))
    if goal not in dist:
        return settled, skipped, None, dist
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = prev[node]
    return settled, skipped, list(reversed(path)), dist

graph = {
    "A": [("B", 1), ("C", 4)],
    "B": [("C", 1), ("D", 10)],
    "C": [("D", 1)],
    "D": [],
}
settled, skipped, path, _dist = shortest(graph, "A", "D")
for line in settled:
    print(line)
print(" ".join(skipped))
print(" ".join(path))
bad = {"A": [("B", 1), ("C", 4)], "B": [], "C": [("B", -10)]}
_settled, _skipped, _path, dist = shortest(bad, "A", "B")
print("reported", dist["B"])
print("true", min(1, 4 - 10))
large, _skipped, large_path, _dist = shortest({"A": [("B", 10**9)], "B": []}, "A", "B")
print(" ".join(large_path))
_settled, _skipped, missing, _dist = shortest({"A": [("B", 1)], "B": []}, "A", "Z")
print("none" if missing is None else " ".join(missing))
```

**Output**
```text
A:0
B:1
C:2
D:3
C:4 D:11
A B C D
reported 1
true -6
A B
none
```

`A` reaches `B` at 1 and `C` at 4. `B` then reaches `C` at 2, which replaces 4, and reaches `D` at 11. `C` reaches `D` at 3, which replaces 11. The settled lines are those final distances: `D` is 3, and the path walked from the predecessors is `A B C D`. The skipped entries `C:4` and `D:11` are the old candidates. Settling either of them would have frozen a distance the algorithm had already improved.

The next graph has a negative edge. The direct path to `B` costs 1. The path `A C B` costs `4 + (-10) = -6`. The program settles `B` at 1 before it walks the edge out of `C`, and it refuses to reopen a settled node. It reports 1. The true shorter path is -6. Nonnegative weights are the assumption that makes "settled" mean finished. With a negative weight, that sentence is false for this program. An edge of weight `10**9` is still nonnegative, so the path `A B` is accepted. A missing goal returns `none` instead of walking a made-up terminator.

**Practice.** Delete the edge from `B` to `C`. The settled distance of `C` stays 4, from `A`, and `D` is then reached from `C` at 5. The stale improvement to 2 never exists.

Sources: [heapq](https://docs.python.org/3.11/library/heapq.html), [Dijkstra lecture](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/resources/lecture-13-dijkstra/), [MIT 6.006 lecture notes](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/pages/lecture-notes/).


## DA15. Edit distance

`cat` can become `cut` by replacing `a` with `u`. The picture labels the prefixes of both words and fills one cell, `ca` to `cu`, before the general rule.

<!-- dsa-scene: DA15 -->

`rows[i][j]` is the fewest edits that turn the first `i` characters of the source into the first `j` characters of the target. For `ca` to `cu`, the cell above gives a deletion, the cell to the left gives an insertion, and the diagonal gives a substitution. Those three cells are filled earlier. The corner of the finished table is the cost of the whole change. Recovering the edits is a second pass. A rolling row can keep the corner and can no longer walk backward. The same dependency shape appears in the Viterbi walk in NC05. This table does not rebuild that labeling theory.

<!-- example: DA15-edits -->
```python
def table(source, target):
    rows = [[0] * (len(target) + 1) for _ in range(len(source) + 1)]
    for i in range(1, len(source) + 1):
        rows[i][0] = i
    for j in range(1, len(target) + 1):
        rows[0][j] = j
    for i, left in enumerate(source, 1):
        for j, right in enumerate(target, 1):
            rows[i][j] = min(
                rows[i - 1][j - 1] + (left != right),
                rows[i - 1][j] + 1,
                rows[i][j - 1] + 1,
            )
    return rows

def reconstruct(source, target):
    rows = table(source, target)
    i, j = len(source), len(target)
    steps = []
    while i or j:
        if i and j and rows[i][j] == rows[i - 1][j - 1] + (source[i - 1] != target[j - 1]):
            if source[i - 1] != target[j - 1]:
                steps.append("replace " + source[i - 1] + " " + target[j - 1])
            i, j = i - 1, j - 1
        elif i and rows[i][j] == rows[i - 1][j] + 1:
            steps.append("delete " + source[i - 1])
            i -= 1
        else:
            steps.append("insert " + target[j - 1])
            j -= 1
    return rows[-1][-1], list(reversed(steps))

for row in table("cat", "cut"):
    print(" ".join(map(str, row)))
cost, steps = reconstruct("cat", "cut")
print(cost)
print(" ".join(steps) if steps else "none")
cost, steps = reconstruct("cat", "at")
print(cost)
print(" ".join(steps))
print(reconstruct("", "")[0])
```

**Output**
```text
0 1 2 3
1 0 1 2
2 1 1 2
3 2 2 1
1
replace a u
1
delete c
0
```

The first row and the first column are the base cases: turning an empty prefix into `c u t` takes 1, 2, and 3 insertions, and deleting `c a t` takes 1, 2, and 3 deletions. The corner is 1. Walking backward prefers the diagonal when it explains the cell. `t` matches `t`, `a` becomes `u`, and `c` matches `c`. The recorded edit is `replace a u`. Turning `cat` into `at` costs one deletion, `delete c`. Two empty strings cost 0. If several edits explain a cell, this walk prefers the diagonal, then a deletion, then an insertion. Another preference would write a different sequence with the same cost.

**Practice.** `rows[i][j]` for `cat` and `cut` is a cost, not the spelling of the edited word. The spelling is recovered only by the walk. Throwing away all but the latest row leaves the cost and loses that walk.

Sources: [MIT 6.006 lecture notes](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/pages/lecture-notes/), [Open Data Structures](https://opendatastructures.org/).


## DA16. Zero-one knapsack

Edit distance is one table. This lesson uses another. Each item may be taken once or skipped. A cell `rows[i][cap]` is the best value using only the first `i` items and a capacity of `cap`. Taking item `i` is legal when its weight fits, and the value added to that choice is the best value of the earlier items in the remaining capacity. Skipping copies the cell from the previous item at the same capacity. The capacity axis and the item axis are both dependencies: a cell reads an earlier item, and it reads a smaller capacity.

<!-- dsa-scene: DA16 -->

The items have positive integer weights. `a` weighs 2 and is worth 3, `b` weighs 2 and is worth 4, and `c` weighs 1 and is worth 2. The capacity is the nonnegative integer 3. The objective is total value.

<!-- example: DA16-choose -->
```python
items = [("a", 2, 3), ("b", 2, 4), ("c", 1, 2)]

def choose(items, capacity):
    rows = [[0] * (capacity + 1) for _ in range(len(items) + 1)]
    for i, (_name, weight, value) in enumerate(items, 1):
        for cap in range(capacity + 1):
            skip = rows[i - 1][cap]
            take = rows[i - 1][cap - weight] + value if weight <= cap else -1
            rows[i][cap] = max(skip, take)
    chosen = []
    cap = capacity
    for i in range(len(items), 0, -1):
        if rows[i][cap] != rows[i - 1][cap]:
            name, weight, _value = items[i - 1]
            chosen.append(name)
            cap -= weight
    return rows[-1][-1], list(reversed(chosen)), rows

value, chosen, rows = choose(items, 3)
for row in rows:
    print(" ".join(map(str, row)))
print(value)
print(" ".join(chosen))
print(choose([], 3)[0])
print(choose(items, 0)[0])
```

**Output**
```text
0 0 0 0
0 0 3 3
0 0 4 4
0 2 4 6
6
b c
0
0
```

The row for `a` can spend 2 or 3 and gain 3. It cannot spend a capacity below 2. The row for `b` improves the capacity-2 cell from 3 to 4, because `b` alone is worth more than `a`. The row for `c` can add 2 on top of the best capacity-2 result, and 4+2 is 6. The reconstruction takes an item when its row differs from the skip. That selects `c`, leaves capacity 2, and then selects `b`. `a` is not taken. The value 6 and the names `b c` are the two outputs. No items, or a capacity of 0, produce 0. A cell that stored only the value would still know the 6. The names require the comparison with the skipped row, or a choice stored beside the value.

**Practice.** Capacity 2 selects `b` alone, value 4. `a` and `c` weigh 3, so they do not fit in 2. The edit-distance table cannot represent this capacity. Its columns were characters, not remaining weight.

Sources: [MIT 6.006 lecture notes](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/pages/lecture-notes/), [Open Data Structures](https://opendatastructures.org/).


## DA17. Greedy choices

A greedy rule keeps one local choice and never takes it back. That is safe only when the problem's structure says the local choice can be part of some best answer. The objective here is the maximum number of non-overlapping half-open intervals, not their total duration. Among the intervals that start at or after the current time, take the one that ends first. The unsafe rule looks similar. It takes the one that starts first. The counterexample is small enough to list.

<!-- dsa-scene: DA17 -->

Beam search, in a decoder, keeps several partial answers because a single early choice can block the better finished answer. It still does not promise that the best finished answer was kept. This lesson does not train a decoder. It shows the commitment with two sort keys.

<!-- example: DA17-intervals -->
```python
intervals = [("a", 0, 3), ("b", 1, 2), ("c", 2, 4)]

def pick(intervals, key):
    chosen = []
    now = None
    for name, start, end in sorted(intervals, key=key):
        if now is None or start >= now:
            chosen.append(name)
            now = end
    return chosen

print(pick(intervals, lambda item: (item[2], item[0])))
print(pick(intervals, lambda item: (item[1], item[0])))
print(pick([("z", -3, -2)], lambda item: (item[2], item[0])))
```

**Output**
```text
['b', 'c']
['a']
['z']
```

The intervals are `a` on `[0, 3)`, `b` on `[1, 2)`, and `c` on `[2, 4)`. Earliest finish takes `b`, sets the clock to 2, and then takes `c`, which starts exactly at 2. Two intervals fit. Earliest start takes `a` and sets the clock to 3. `c` starts at 2, which is already behind that clock, so the result is only `a`. The earliest-start rule looked reasonable and lost an interval. The earliest-finish rule is the one whose exchange argument holds for this count: any answer that picked something else first can swap in this earlier finish without losing a later interval. That argument is about the number of intervals. It does not transfer to a knapsack just because both problems involve choosing items. The clock starts unset, so `z` on `[-3, -2)` is accepted. A clock that started at `-1` would have rejected that valid interval.

**Practice.** If `c` started at 1, it would overlap `b`. Earliest finish would take `b` and then be unable to take `c`. The best size would be 1, and the rule would still be right. The failure above was a rule that left a compatible interval behind.

Sources: [Sorting](https://docs.python.org/3/howto/sorting.html), [MIT 6.006 syllabus](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/pages/syllabus/).


## DA18. A bounded stream

A stream that does not fit in memory can still answer some questions if the state has a bound. Top-two scores need a heap of size two, updated once per arriving pair. The caller below yields one chunk at a time. Four scores are seen. Two stay stored after each update, and the heap can briefly hold three. The profiler in the [profiling lesson](07_OTHER_LIBRARIES.md) is the tool for asking which function ran, and this page does not print a duration. A tighter loop around a nested scan can be faster on four rows and still grow like the square of a large file.

<!-- dsa-scene: DA18 -->

<!-- example: DA18-stream -->
```python
import heapq

def top_from_stream(chunks, k):
    heap = []
    seen = 0
    peak = 0
    for chunk in chunks:
        for doc_id, score in chunk:
            seen += 1
            heapq.heappush(heap, (score, doc_id))
            peak = max(peak, len(heap))
            if len(heap) > k:
                heapq.heappop(heap)
    return seen, len(heap), peak, [doc_id for score, doc_id in sorted(heap, reverse=True)]

def chunks():
    yield [("a", 1), ("b", 5)]
    yield [("c", 3), ("d", 4)]

seen, stored, peak, kept = top_from_stream(chunks(), 2)
print(seen)
print(stored)
print(peak)
print(" ".join(kept))
```

**Output**
```text
4
2
3
b d
```

The generator yields the first chunk and then the second. The caller does not hold both chunks in one list. The loop variable keeps the current chunk until the next yield replaces it. After a push the heap can hold one more than the bound: the peak is 3, then the pop leaves `b` at 5 and `d` at 4. The printed stored size 2 is the size after the update, not the peak. Scores here are distinct, so the min-heap of pairs agrees with "higher score wins." A tie would need the policy from the heap lesson. Keeping every chunk and sorting at the end would answer the same question and would store the whole stream. The bound `O(chunk size + k)` belongs to this generator-and-heap arrangement.

**Practice.** A bound of 0 stores nothing and still counts the arrivals. A bound of 4 on this stream stores every ID. The bound is part of the contract, not a property of `heapq` by itself.

Sources: [heapq](https://docs.python.org/3/library/heapq.html), [Profiling](07_OTHER_LIBRARIES.md).


## DA19. Tests for a change

The method for a changed requirement is the same method as the first lesson, practiced on one function. Write the contract. Keep a slow version that is obviously about that contract. Name the fact that stays true. Change the implementation. Test the empty input, a duplicate, a malformed record, and a result that the caller then mutates. This page calls those checks directly. A pytest file, as in the testing lesson, is how the same checks would be collected later. Collection is not what prints these four lines.

<!-- dsa-scene: DA19 -->

The contract is `keep_first`. Every item must be a non-empty string. The first copy of each accepted ID is kept, in the order it appeared. The input list is not modified.

<!-- example: DA19-checks -->
```python
def keep_first(ids):
    seen, kept = set(), []
    for item in ids:
        if not isinstance(item, str) or not item:
            raise ValueError("id")
        if item not in seen:
            seen.add(item)
            kept.append(item)
    return kept

print(keep_first([]))
print(keep_first(["d1", "d1"]))
try:
    keep_first(["d1", ""])
    print("accepted")
except ValueError:
    print("rejected")
original = ["d1", "d2"]
copied = keep_first(original)
copied.append("d9")
print(original)

def sorted_unique(ids):
    return sorted(set(ids))

sample = ["d2", "d1", "d2"]
print(keep_first(sample))
print(sorted_unique(sample))
```

**Output**
```text
[]
['d1']
rejected
['d1', 'd2']
['d2', 'd1']
['d1', 'd2']
```

The empty input has no IDs to keep. The duplicated `d1` is kept once. The empty string is a malformed ID, so the function raises and the example prints `rejected` rather than a list that quietly contains it. Appending `d9` to the returned list leaves the original `["d1", "d2"]` alone. If `keep_first` had returned the same list object it received, that append would have changed the caller's data. `sorted(set(...))` on `d2`, `d1`, `d2` prints `d1`, `d2`. `keep_first` prints `d2`, `d1`. A check that only compared the set of names would accept the broken order. The scan from the first lesson is the slow version of the duplicate question. This function is the first-copy contract, with the malformed case added. The cost claim is still the one from that lesson: the set replaces the nested equality scan, under the hash assumptions stated there.

**Practice.** `keep_first(["d2", "d1", "d2"])` returns `["d2", "d1"]`. Sorting that result would invent a different contract. A test that only checks the set of names would accept the sorted list and miss the order bug.

Sources: [assert](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement), [pytest](https://docs.pytest.org/en/stable/how-to/parametrize.html).


## DA20. Dedup, top scores and a walk

Three jobs are fixed before any solution appears. Keep the first copy of `d1`, `d2`, `d1`. Keep the top two of `d2:5`, `d5:5`, `d3:5`, `d4:1`, using the heap lesson's tie policy. Walk `claim` to `evidence` to `detail` in breadth-first order. Predict each line from the picture, then read the code that prints it. None of the three is hidden inside a library call whose rule is different from the lesson that introduced it. Finishing the three prints is practice at choosing a representation. It is not an estimate of an interview outcome, and it is not a quota of memorized problems.

<!-- dsa-scene: DA20 -->

Timed practice can come later, on new inputs, without the lesson open. Union-find, Knuth-Morris-Pratt or Z-style string matching, range queries, and balanced-tree updates are the next structures if you continue. Each one needs its own trace. Flow algorithms and suffix structures are further out. A recruiter still has to be asked what a coding round contains.

<!-- example: DA20-jobs -->
```python
import heapq
from collections import deque

class Rank:
    def __init__(self, score, doc_id):
        self.score = score
        self.doc_id = doc_id

    def __lt__(self, other):
        if self.score != other.score:
            return self.score < other.score
        return self.doc_id > other.doc_id

def keep_first(ids):
    seen, kept = set(), []
    for item in ids:
        if item not in seen:
            seen.add(item)
            kept.append(item)
    return kept

def top_k(rows, k):
    kept = []
    for doc_id, score in rows:
        heapq.heappush(kept, Rank(score, doc_id))
        if len(kept) > k:
            heapq.heappop(kept)
    return [item.doc_id for item in sorted(kept, reverse=True)]

def bfs(graph, start):
    seen = {start}
    queue = deque([start])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for nxt in graph[node]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return order

print(" ".join(keep_first(["d1", "d2", "d1"])))
print(" ".join(top_k([("d2", 5), ("d5", 5), ("d3", 5), ("d4", 1)], 2)))
print(" ".join(bfs({"claim": ["evidence"], "evidence": ["detail"], "detail": []}, "claim")))
```

**Output**
```text
d1 d2
d2 d3
claim evidence detail
```

The IDs keep `d1` and `d2`. Three documents share the score 5, and the tie policy keeps the two smaller IDs, `d2` and `d3`, rather than `d5`. The graph has one path, so the queue order is `claim`, then `evidence`, then `detail`. Changing any one contract changes one line. Asking for every ID instead of the top two would print `d2 d3 d5 d4`. Marking the graph nodes in a different neighbor order would print a different walk. The three lines are checkable because each contract was fixed before the code ran.

**Practice.** Add a second neighbor, `note`, on `claim`, listed after `evidence`. The walk becomes `claim evidence note detail` if `note` has no children and `evidence` still leads to `detail`. The ID line and the score line do not move.


Sources: [MIT 6.006 syllabus](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/pages/syllabus/), [Python lists](https://docs.python.org/3/tutorial/datastructures.html).
