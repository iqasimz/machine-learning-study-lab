"""Authored DSA explanations and small, tested recipes for the served reader.

Pictures show example states; recipes explain control flow. Clicking a recipe
node inspects code, not an execution state. The longer instrumented programs
remain in learning_materials/python/08_DSA.md.
"""
from copy import deepcopy
from textwrap import dedent


def recipe(title, code, nodes, edges, examples):
    return {"title": title, "code": dedent(code).strip(),
            "nodes": [{"id": str(i), "label": label, "anchor": anchor,
                       "explanation": explanation, "kind": kind}
                      for i, (label, anchor, explanation, kind) in enumerate(nodes)],
            "edges": [{"from": str(a), "to": str(b), "label": label} for a, b, label in edges],
            "examples": [{"expression": expression, "output": output} for expression, output in examples]}


def lesson(definition, problem, intuition, terms, worked, cost, trap, practice, answer, main):
    return dict(definition=definition, problem=problem, intuition=intuition, terms=terms,
                worked=worked, cost=cost, trap=trap, practice=practice, answer=answer, recipes=[main])


LESSONS = {
    "DA01": lesson(
        "An algorithm is a precise procedure for turning an input into an output. A data structure is the way you arrange the information that procedure needs.",
        "A corpus contains document IDs ['d1', 'd2', 'd1', 'd3']. Keep the first occurrence of each ID, in its original order: ['d1', 'd2', 'd3'].",
        "Imagine checking names at a door. Keep a membership list of people already admitted; admit a name only the first time you see it. You do not need to compare every name with every other name.",
        [["Contract", "What the input is allowed to contain and what the output must mean. Here, IDs are strings; order must be preserved; the input must not be changed."],
         ["n and Big O", "n is the number of input items. O(n) means the amount of work grows at most proportionally to n for sufficiently large inputs, ignoring constant factors. It is a growth bound, not a time in seconds."],
         ["Time versus space", "Time counts operations. Extra space counts information retained in addition to the input. A faster method can use more memory."]],
        "Read d1: new → save it. Read d2: new → save it. Read d1 again: already seen → skip it. Read d3: new → save it. Four membership checks replace six pair comparisons for this four-item example. At n = 100,000, comparing every pair would mean 4,999,950,000 comparisons.",
        "Expected O(n) time with a Python set and ordinary bounded-length string IDs; O(u) extra space for u unique IDs. Hashing a newly encountered long string also reads its characters. Big O describes growth, not a guarantee that a set is faster on every tiny input.",
        "set(ids) removes duplicates but does not satisfy the original-order contract. sorted(set(ids)) imposes a different order. An optimisation is wrong if it solves a different problem.",
        "What should happen to ['b', 'a', 'b', 'a']? Which two containers grow, and why do we need both?",
        "Return ['b', 'a']. The set answers membership questions; the list records the required order. They store different kinds of information about the same unique IDs.",
        recipe("Keep first occurrences", '''
            def keep_first(ids):
                seen = set()
                result = []
                for item in ids:
                    if item not in seen:
                        seen.add(item)
                        result.append(item)
                return result
        ''', [
            ("Create seen and result", "seen = set()", "Start with no remembered IDs and no output.", "start"),
            ("Another ID?", "for item in ids:", "The loop reads each input ID exactly once. When it runs out, return the result.", "decision"),
            ("Is this ID new?", "if item not in seen:", "Membership, not position, is the question the set answers.", "decision"),
            ("Remember and append", "seen.add(item)", "Add the ID to seen and to the end of result. Keeping those actions together preserves first-occurrence order.", "process"),
            ("Return ordered IDs", "return result", "The list contains each distinct ID once, in order of first appearance.", "end")
        ], [(0,1,""),(1,2,"Yes"),(1,4,"No"),(2,3,"Yes"),(2,1,"No: skip"),(3,1,"Next ID")],
        [("keep_first(['d1', 'd2', 'd1', 'd3'])", "['d1', 'd2', 'd3']"), ("keep_first([])", "[]")])
    ),
    "DA02": lesson(
        "A Python list is an ordered, resizable array of references. An index tells you where an item is; a search asks where a value might be.",
        "Keep the last two document IDs from ['d1', 'd2', 'd3', 'd4'] without changing the original list.",
        "Think of numbered lockers. If you know the locker number, you can go straight to it. If you know only the name inside, you may need to open many lockers. Copying a row of lockers still means visiting every copied position.",
        [["Reference", "A list slot points to a Python object. Slicing copies the slots into a new list, not the objects inside them."],
         ["Amortised cost", "An occasional expensive operation is averaged across a sequence. A list append is usually cheap, but sometimes the internal array must grow and existing references must be copied."]],
        "len(ids) is 4, so the last two begin at index 2. ids[2:] copies references to d3 and d4 into a separate list. The original remains unchanged. Building snapshots of lengths 1, 2, 3, 4 copies 1 + 2 + 3 + 4 = 10 slots, even though only four IDs arrived.",
        "Indexing is O(1); searching with .index is O(n); a slice of k entries takes O(k) time and O(k) new list space. Repeatedly copying the entire growing prefix takes O(n²) total copied slots. Append is amortised O(1), not worst-case O(1) for every individual call.",
        "A shallow copy does not isolate nested mutable objects. Also, ids[-0:] means ids[0:]—the whole list—not an empty suffix.",
        "If k is 0 or larger than the list, what should a 'last k' function return? What if k is negative?",
        "Here k = 0 returns [], k greater than length returns a copy of the whole list, and negative k raises ValueError. These are explicit contract choices.",
        recipe("Copy the last k items", '''
            def last_k(items, k):
                if k < 0:
                    raise ValueError("k must be nonnegative")
                if k == 0:
                    return []
                start = max(0, len(items) - k)
                return items[start:]
        ''', [
            ("Read items and k", "def last_k", "The output must be a new list; the input must stay unchanged.", "start"),
            ("Is k negative?", "if k < 0:", "Reject a request that has no meaning under this contract.", "decision"),
            ("Raise an error", "raise ValueError", "Invalid input stops the function here.", "end"),
            ("Is k zero?", "if k == 0:", "Handle zero before slicing: -0 is just 0 in Python.", "decision"),
            ("Return empty list", "return []", "No items were requested.", "end"),
            ("Find start and slice", "start = max", "Clamp the start to zero, then copy the suffix. The slice stops at the end of the input.", "end")
        ], [(0,1,""),(1,2,"Yes"),(1,3,"No"),(3,4,"Yes"),(3,5,"No")],
        [("last_k(['d1', 'd2', 'd3', 'd4'], 2)", "['d3', 'd4']"), ("last_k([1, 2], 0)", "[]")])
    ),
    "DA03": lesson(
        "A hash table turns a key into a location where its value can be found. Python dictionaries use this idea for key–value lookup; sets use it for membership.",
        "Count repeated document IDs: ['d1', 'd2', 'd1'] should produce {'d1': 2, 'd2': 1}.",
        "Use the ID as a label on a drawer. Instead of scanning all previous IDs, go to that ID's drawer and increase its counter. Two labels can lead to the same initial drawer; the table must still distinguish the actual keys.",
        [["Hash", "A number calculated from a key and used to locate candidate storage slots."],
         ["Collision", "Different keys produce the same candidate slot. A correct hash table resolves this; it must not treat those keys as equal."],
         ["Hashable", "A key has a stable hash and equality behaviour while stored. Strings and tuples of hashable values work; lists do not."]],
        "The first d1 reads a missing counter as 0, then stores 1. d2 gets its own counter of 1. The final d1 reads 1 and replaces it with 2. A collision changes how the dictionary searches internally, not the counts it should return.",
        "Expected O(n) time and O(u) space for n bounded-length string IDs and u distinct keys. A lookup is expected O(1), not unconditional worst-case O(1). Reading and hashing a long new key is additional character work.",
        "Do not use a homemade hash number as a unique document ID. Different strings can collide. Keep the actual key or verify equality.",
        "Would counts[item] += 1 work the first time a new ID appears?",
        "No: it raises KeyError because the key has no value yet. get(item, 0) supplies the starting count. collections.Counter is the standard-library alternative.",
        recipe("Count IDs with a dictionary", '''
            def count_ids(ids):
                counts = {}
                for item in ids:
                    previous = counts.get(item, 0)
                    counts[item] = previous + 1
                return counts
        ''', [
            ("Create an empty dictionary", "counts = {}", "Each key will be an ID; each value will be its count.", "start"),
            ("Another ID?", "for item in ids:", "Process one occurrence, not one distinct ID, per iteration.", "decision"),
            ("Read count or zero", "previous =", "A missing key starts at zero. This does not insert the key yet.", "process"),
            ("Store count plus one", "counts[item] =", "Replace the old count with the new one.", "process"),
            ("Return counts", "return counts", "Only distinct keys occupy dictionary entries.", "end")
        ], [(0,1,""),(1,2,"Yes"),(1,4,"No"),(2,3,""),(3,1,"Next ID")],
        [("count_ids(['d1', 'd2', 'd1'])", "{'d1': 2, 'd2': 1}"), ("count_ids([])", "{}")])
    ),
    "DA04": lesson(
        "Sorting puts items into a chosen order. The interesting question is how much work it takes and what happens to items whose keys are equal.",
        "Sort scores [5, 2, 4] into [2, 4, 5]. Then consider document records where equal scores should retain their original order.",
        "Insertion sort works like arranging playing cards. The cards already in your hand are sorted; take the next card and shift larger cards right until there is a gap for it. Merge sort instead sorts smaller halves and combines them. Quicksort partitions around a pivot.",
        [["Sorted prefix", "The part before the current index is already in order."],
         ["Stable sort", "Items with equal sorting keys keep their relative input order."],
         ["In place", "The same list is changed rather than returning a separately sorted list. It does not imply literally zero temporary storage."]],
        "Start with [5 | 2, 4]. Hold 2, shift 5 right, insert 2 → [2, 5 | 4]. Hold 4, shift 5 right, stop at 2, insert 4 → [2, 4, 5]. The hole moves; the held value must not be lost.",
        "Insertion sort uses O(n²) comparisons/moves in the worst case and O(n) on an already sorted list. Its shifting needs O(1) working state, but this recipe first copies the input, so its total extra list storage is O(n). Merge sort is O(n log n) with O(n) auxiliary storage in a usual array implementation. Quicksort is O(n log n) on average but O(n²) with persistently poor partitions. Python's sorted() and list.sort() are stable, adaptive sorts; use them in real applications.",
        "Changing > to >= in insertion sort can reverse equal-key records. Also, a sort key and a tie-breaking rule are separate choices: document ID order is not the same as arrival order.",
        "Which input makes insertion sort do the least shifting: ascending or descending values?",
        "Ascending values. Each next value already belongs at the right edge of the sorted prefix, so the while condition fails immediately. Descending values must move across the entire prefix.",
        recipe("Insertion sort: grow a sorted prefix", '''
            def insertion_sort(values):
                result = list(values)
                for i in range(1, len(result)):
                    held = result[i]
                    j = i - 1
                    while j >= 0 and result[j] > held:
                        result[j + 1] = result[j]
                        j -= 1
                    result[j + 1] = held
                return result
        ''', [
            ("Copy the input", "result = list", "This version leaves the original unchanged. The sorting itself rearranges the copy.", "start"),
            ("Another unsorted item?", "for i in range", "Before i, the prefix is already sorted.", "decision"),
            ("Hold item; look left", "held =", "Store the item before shifting over its old slot.", "process"),
            ("Left neighbour larger?", "while j >=", "Only a strictly larger neighbour must move. Equal values stay in order.", "decision"),
            ("Shift right; move left", "result[j + 1] = result[j]", "The gap moves left by one position; held remains safely stored.", "process"),
            ("Insert held item", "result[j + 1] = held", "Either the left neighbour is small enough, or the start was reached.", "process"),
            ("Return sorted copy", "return result", "The whole prefix now covers the input.", "end")
        ], [(0,1,""),(1,2,"Yes"),(1,6,"No"),(2,3,""),(3,4,"Yes"),(4,3,"Repeat"),(3,5,"No"),(5,1,"Next item")],
        [("insertion_sort([5, 2, 4])", "[2, 4, 5]"), ("insertion_sort([2, 2, 1])", "[1, 2, 2]")])
    ),
    "DA05": lesson(
        "Binary search finds a boundary in ordered data by repeatedly ruling out about half the remaining possibilities. It is useful only when the comparison lets you safely discard a whole side.",
        "In [2, 4, 4, 7, 9], find the first index whose value is at least 4. The answer is index 1, not just any occurrence of 4.",
        "Think of finding the first eligible applicant in a list sorted by score. If the middle score is too low, every earlier score is too low too. Otherwise the first eligible score could be the middle one or an earlier one.",
        [["Lower bound", "The first index with value ≥ target, or len(values) if no such value exists."],
         ["Half-open interval [lo, hi)", "lo is included; hi is excluded from the next read. At the end lo = hi is a boundary that may be one past the list."],
         ["Invariant", "A fact that remains true after every update. Here everything before lo is too small, and every valid index at or after hi has value ≥ target."]],
        "lo = 0, hi = 5 → mid = 2, value 4: move hi to 2. Now mid = 1, value 4: move hi to 1. Now mid = 0, value 2: move lo to 1. The bounds meet at 1. Notice that seeing 4 does not stop the search: an earlier 4 may exist.",
        "O(log n) comparisons and O(1) auxiliary space for random-access sorted data. Sorting first costs O(n log n); binary search is attractive when many queries reuse the sorted data. Finding the boundary does not prove that an exact target is present.",
        "lo = mid can leave a one-item interval unchanged forever. Use lo = mid + 1 after proving mid is too small. Do not apply this rule to unsorted data.",
        "What boundary should targets 5 and 10 return? How would you test whether the exact value exists?",
        "5 returns index 3; 10 returns index 5. Check i < len(values) and values[i] == target before claiming an exact match.",
        recipe("Find the first value at least target", '''
            def lower_bound(values, target):
                lo, hi = 0, len(values)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if values[mid] < target:
                        lo = mid + 1
                    else:
                        hi = mid
                return lo
        ''', [
            ("Set lo = 0, hi = n", "lo, hi =", "Initially no positions have been ruled out.", "start"),
            ("Is lo < hi?", "while lo < hi:", "Continue only while an unresolved position remains.", "decision"),
            ("Read the middle value", "mid =", "Integer division selects an index inside the half-open interval.", "process"),
            ("Is middle < target?", "if values[mid] < target:", "This comparison decides which whole side can be ruled out.", "decision"),
            ("Move lo to mid + 1", "lo = mid + 1", "mid and every earlier position are too small.", "process"),
            ("Move hi to mid", "hi = mid", "mid may be the answer; keep it as the right boundary while searching earlier positions.", "process"),
            ("Return lo", "return lo", "The first eligible position is the boundary where the two bounds meet.", "end")
        ], [(0,1,""),(1,2,"Yes"),(1,6,"No"),(2,3,""),(3,4,"Yes"),(3,5,"No"),(4,1,"Repeat"),(5,1,"Repeat")],
        [("lower_bound([2, 4, 4, 7, 9], 4)", "1"), ("lower_bound([2, 4, 4, 7, 9], 10)", "5"), ("lower_bound([], 4)", "0")])
    ),
    "DA06": lesson(
        "Two pointers use two positions to narrow a search without testing every pair. The reason a pointer may move is more important than the pointer syntax.",
        "Find two different positions in sorted [1, 2, 4, 7] whose values sum to 6. Return indices (1, 2), because 2 + 4 = 6.",
        "Start with the smallest and largest remaining values. If even the largest partner makes the left value too small, discard that left value. If even the smallest partner makes the right value too large, discard that right value.",
        [["Pointer", "Here just an integer index, not a raw memory address."],
         ["Monotonic order", "Moving right never decreases the value. This property makes the discard argument valid."]],
        "1 + 7 = 8 is too large → move right leftwards. 1 + 4 = 5 is too small → move left rightwards. 2 + 4 = 6 → return (1, 2). Each movement permanently discards one impossible endpoint.",
        "O(n) time and O(1) auxiliary space once the array is sorted. Returning original positions after sorting requires carrying the original indices. A hash-table two-sum method is another expected O(n) approach when the input is not sorted.",
        "The same moves on [1, 3, 2] can miss 1 + 2 = 3: order is part of the contract. left < right also prevents using the same position twice.",
        "If the current sum is too large, why not move the left pointer right?",
        "That cannot help: it would keep or increase the sum. Moving the right pointer left is the move that can reduce it.",
        recipe("Find a pair in sorted values", '''
            def pair_sum(values, target):
                left, right = 0, len(values) - 1
                while left < right:
                    total = values[left] + values[right]
                    if total == target:
                        return left, right
                    if total < target:
                        left += 1
                    else:
                        right -= 1
                return None
        ''', [
            ("Start at both ends", "left, right =", "The two positions must remain distinct.", "start"),
            ("Is left < right?", "while left < right:", "If not, no untested pair remains.", "decision"),
            ("Add the two values", "total =", "Only the current endpoint pair is inspected.", "process"),
            ("Does sum equal target?", "if total == target:", "A match can be returned immediately because any pair is acceptable here.", "decision"),
            ("Return the positions", "return left, right", "These are positions in the sorted input.", "end"),
            ("Too small? Move left; otherwise right", "if total < target:", "Increase the smaller endpoint if the sum is too small; decrease the larger endpoint otherwise.", "process"),
            ("Return no pair", "return None", "Failure is an explicit output, not an invalid index.", "end")
        ], [(0,1,""),(1,2,"Yes"),(1,6,"No"),(2,3,""),(3,4,"Yes"),(3,5,"No"),(5,1,"Repeat")],
        [("pair_sum([1, 2, 4, 7], 6)", "(1, 2)"), ("pair_sum([], 6)", "None")])
    ),
    "DA07": lesson(
        "A prefix sum stores the total before each boundary. A sliding window reuses the total of one span when moving to the next. Both avoid adding the same values repeatedly.",
        "For token counts [2, 5, 1, 3], compute every two-document total: [7, 6, 4]. Separately, answer the range query from index 1 up to index 3: 5 + 1 = 6.",
        "If you already know a two-document total, moving one document right removes the departing count and adds the arriving count. A prefix sum instead acts like an odometer: subtract readings at two boundaries to measure the distance between them.",
        [["Prefix array", "prefix[0] = 0 and prefix[i] is the total of items before i. The extra initial zero makes ranges starting at zero work normally."],
         ["Window", "A contiguous span of the input. Fixed width means its number of items never changes."]],
        "First window: 2 + 5 = 7. Slide right: 7 − 2 + 1 = 6. Slide again: 6 − 5 + 3 = 4. The prefix array is [0, 2, 7, 8, 11], so the separate range [1, 3) totals prefix[3] − prefix[1] = 8 − 2 = 6.",
        "Prefix sums cost O(n) time and O(n) storage, then each range-total query costs O(1). A fixed-window sweep costs O(n) time and O(1) working state beyond its output. Returning all windows still stores O(n) output values.",
        "Fixed-window subtraction works with negative values. A different method—shrinking a variable window whenever its sum exceeds a budget—needs a suitable monotonicity argument and can fail with negatives. Do not transfer that assumption between the two algorithms.",
        "For [5, -2, 3] and width 2, what are the totals? What should width 4 return?",
        "The totals are [3, 1]. This recipe returns [] when the width is nonpositive or larger than the input; it does not invent a partial window.",
        recipe("Slide a fixed-width total", '''
            def window_sums(values, width):
                if width <= 0 or width > len(values):
                    return []
                total = sum(values[:width])
                totals = [total]
                for right in range(width, len(values)):
                    total += values[right] - values[right - width]
                    totals.append(total)
                return totals
        ''', [
            ("Read values and width", "def window_sums", "Only complete windows are part of the output.", "start"),
            ("Is width invalid?", "if width <=", "Reject zero, negative and oversized windows by returning an empty result.", "decision"),
            ("Return empty result", "return []", "There are no complete windows of the requested width.", "end"),
            ("Sum the first window", "total = sum", "This is the only window summed from scratch.", "process"),
            ("Another arriving value?", "for right in range", "right names the new rightmost position.", "decision"),
            ("Add arriving; remove departing", "total +=", "The old left endpoint is right − width. Save the updated total.", "process"),
            ("Return all totals", "return totals", "The number of outputs is len(values) − width + 1 for a valid width.", "end")
        ], [(0,1,""),(1,2,"Yes"),(1,3,"No"),(3,4,""),(4,5,"Yes"),(5,4,"Slide"),(4,6,"No")],
        [("window_sums([2, 5, 1, 3], 2)", "[7, 6, 4]"), ("window_sums([5, -2, 3], 2)", "[3, 1]"), ("window_sums([1], 2)", "[]")])
    ),
    "DA08": lesson(
        "A stack removes the most recently added item; a queue removes the oldest. These access rules express different kinds of unfinished work. A linked list stores the next link with each item instead of keeping all slots in one array.",
        "Check whether brackets are properly nested: '([])' is valid, but '(]' is not.",
        "Opening brackets are promises to close something later. The most recent promise must be fulfilled first: the ] must close [ before ) can close (. That is exactly a stack's last-in, first-out rule.",
        [["LIFO / FIFO", "Last in, first out describes a stack. First in, first out describes a queue."],
         ["Linked node", "A value plus a reference to the next node. Finding a position still requires following links; insertion is cheap only when the relevant link is already known."]],
        "Read ( → stack ['(']. Read [ → ['(', '[']. Read ] → pop [, which matches. Read ) → pop (, which matches. The input is valid only if the stack is empty at the end; '(()' leaves an unmatched opening bracket.",
        "Bracket checking takes O(n) time and O(n) worst-case stack space. Python list append/pop at the end support a stack. collections.deque append/popleft supports a queue with O(1) endpoint operations; list.pop(0) shifts the remaining references.",
        "Counting opening and closing brackets is not enough: '([)]' has equal counts but incorrect nesting. An empty stack before a closing bracket is also an immediate failure.",
        "Why is a queue suitable for breadth-first search but not for matching nested brackets?",
        "Breadth-first search must expand the oldest discovered work first. Nested brackets must close the newest unmatched opener first. The required order determines the container.",
        recipe("Match brackets with a stack", '''
            def balanced(text):
                stack = []
                closing = {')': '(', ']': '[', '}': '{'}
                for char in text:
                    if char in '([{':
                        stack.append(char)
                    elif char in closing:
                        if not stack or stack.pop() != closing[char]:
                            return False
                return not stack
        ''', [
            ("Create empty stack", "stack = []", "Non-bracket characters are ignored by this contract.", "start"),
            ("Another character?", "for char in text:", "Read characters left to right.", "decision"),
            ("Opening bracket?", "if char in '([{':", "An opener creates a new most-recent obligation.", "decision"),
            ("Push opener", "stack.append", "The newest opener is now on top.", "process"),
            ("Closing bracket?", "elif char in closing:", "Other characters need no action.", "decision"),
            ("Missing or wrong opener?", "if not stack or", "Python short-circuits or: pop is not called if the stack is empty.", "decision"),
            ("Return False", "return False", "A mismatch cannot be repaired by later characters.", "end"),
            ("Return whether stack is empty", "return not stack", "Leftover opening brackets mean the input was incomplete.", "end")
        ], [(0,1,""),(1,2,"Yes"),(1,7,"No"),(2,3,"Yes"),(3,1,"Next"),(2,4,"No"),(4,5,"Yes"),(4,1,"No"),(5,6,"Yes"),(5,1,"No: matched")],
        [("balanced('([])')", "True"), ("balanced('([)]')", "False"), ("balanced('(()')", "False")])
    ),
    "DA09": lesson(
        "Recursion solves a problem by calling the same function on a smaller problem, then using the returned answer. Every unfinished call keeps its own local state on the call stack.",
        "Calculate 3! = 3 × 2 × 1. Understand not just the answer 6, but which call is waiting for which result.",
        "A manager asks an assistant to solve the smaller task. The manager pauses, remembering what to do when the assistant returns. Eventually someone receives a task small enough to answer immediately; the waiting work then finishes in reverse order.",
        [["Base case", "The case answered without another recursive call."],
         ["Recursive case", "The case that asks for a smaller answer and combines it with local work."],
         ["Call stack", "The active chain of unfinished calls. It is a stack, not the same thing as the entire tree of calls made over time."]],
        "factorial(3) waits for factorial(2), which waits for factorial(1), which waits for factorial(0). The base returns 1. The waiting calls return 1 × 1 = 1, then 2 × 1 = 2, then 3 × 2 = 6. Going down creates unfinished work; returning resolves it.",
        "This recipe makes n + 1 calls and retains O(n) stack frames. Under a unit-cost arithmetic model it takes O(n) operations; Python's arbitrarily large integers make multiplication cost grow for large n. Python does not optimise tail calls and has a recursion-depth limit.",
        "A recursive function must move toward a reachable base case. Negative n is rejected here. Two recursive calls per level can create exponentially more total calls than this one-branch example.",
        "When factorial(0) returns, which call resumes first? Is factorial usually best written recursively in Python?",
        "factorial(1) resumes first. An iterative version avoids recursion depth and stack overhead; the recursive version is useful here for understanding calls and returns.",
        recipe("Recursive factorial", '''
            def factorial(n):
                if n < 0:
                    raise ValueError("n must be nonnegative")
                if n == 0:
                    return 1
                smaller = factorial(n - 1)
                return n * smaller
        ''', [
            ("Call factorial(n)", "def factorial", "Each call has its own n and later its own smaller value.", "start"),
            ("Is n negative?", "if n < 0:", "Keep invalid inputs from recursing away from zero.", "decision"),
            ("Raise an error", "raise ValueError", "The function's contract accepts nonnegative integers.", "end"),
            ("Is n zero?", "if n == 0:", "This is the stopping condition.", "decision"),
            ("Return 1", "return 1", "The base case starts the chain of returns.", "end"),
            ("Call factorial(n − 1); wait", "smaller = factorial", "The current call pauses until the complete smaller call returns. This is a new invocation, not a jump within the same frame.", "process"),
            ("Multiply and return", "return n * smaller", "The paused caller resumes with the returned value and finishes its own work.", "end")
        ], [(0,1,""),(1,2,"Yes"),(1,3,"No"),(3,4,"Yes"),(3,5,"No"),(5,6,"On return")],
        [("factorial(3)", "6"), ("factorial(0)", "1")])
    ),
    "DA10": lesson(
        "Backtracking explores choices, abandons branches that cannot work, and restores the previous state before trying a different choice. It is systematic trial, not guessing.",
        "List all subsets of nonnegative costs [1, 2, 4] whose sum is at most 3. Choosing nothing is allowed.",
        "For each item, ask 'take it or leave it?' A path through those decisions describes one subset. If the current sum already exceeds 3, every extension is impossible because the remaining numbers cannot reduce the sum.",
        [["Search tree", "A tree of partial choices, not a tree stored in your dataset."],
         ["Pruning", "Skipping an entire branch after proving that no valid answer lies below it."],
         ["Undo", "Restore the partial choice after a recursive call. In this recipe each branch gets a new list instead, so it does not mutate the caller's list."]],
        "Start with []. Leaving 1 leads to choices without 1; taking 1 leads to [1]. Taking 2 from there gives [1, 2], total 3. Adding 4 would give 7, so that branch stops. The complete output is [[], [2], [1], [1, 2]] in this recipe's exclude-before-include order.",
        "There are up to 2ⁿ subsets. Copying and returning all subsets can require O(n·2ⁿ) time and output space. Pruning helps some inputs but does not remove the exponential worst case. This simple copying version can retain O(n²) live path entries across recursive calls; an append/pop version uses O(n) path state, excluding the output.",
        "Pruning sum > limit is wrong if later items can be negative: [4, -2] can come back below 3. The recipe therefore rejects negative item costs.",
        "Could the same subset values appear twice if two input positions contain the same value?",
        "Yes. These are subsets of positions. [1, 1] has two distinct one-position choices that both display as [1]. Deduplicating by value would be a different contract.",
        recipe("Include or exclude each cost", '''
            def subsets_within(items, limit):
                if any(value < 0 for value in items):
                    raise ValueError("costs must be nonnegative")
                answers = []
                def visit(i, chosen, total):
                    if total > limit:
                        return
                    if i == len(items):
                        answers.append(chosen)
                        return
                    visit(i + 1, chosen, total)
                    visit(i + 1, chosen + [items[i]], total + items[i])
                visit(0, [], 0)
                return answers
        ''', [
            ("Validate costs; start empty", "if any(value < 0", "The nonnegative condition is what makes the pruning proof work.", "start"),
            ("Current sum over limit?", "if total > limit:", "Every descendant can only keep or increase this sum.", "decision"),
            ("Stop this branch", "return", "This return leaves the current recursive branch, not the entire search.", "end"),
            ("All positions decided?", "if i == len(items):", "A full sequence of choices now describes one subset.", "decision"),
            ("Save subset and return", "answers.append", "Only a complete, feasible subset is an answer.", "end"),
            ("Explore without this item", "visit(i + 1, chosen, total)", "The recursive call considers all later choices before returning here.", "process"),
            ("Explore with this item", "visit(i + 1, chosen +", "Use a new chosen list. After both calls return, this invocation is finished.", "end")
        ], [(0,1,"Call visit"),(1,2,"Yes"),(1,3,"No"),(3,4,"Yes"),(3,5,"No"),(5,6,"After return")],
        [("subsets_within([1, 2, 4], 3)", "[[], [2], [1], [1, 2]]"), ("subsets_within([], 0)", "[[]]")])
    ),
    "DA11": lesson(
        "A tree connects nodes through parent–child relationships without cycles. A binary search tree orders whole subtrees by key. A trie instead follows one character or token per edge to represent prefixes.",
        "Find key 6 in a binary search tree rooted at 4: the right child is 8, whose left child is 6. Separately, understand why 'cat' and 'car' share a path in a trie.",
        "At each search-tree node, ask whether the target is smaller or larger. The ordering promise lets you ignore the entire wrong subtree. A trie asks a different question: which next character extends this prefix?",
        [["Subtree", "A node and every descendant below it."],
         ["Height", "The length of the longest root-to-leaf path under a stated counting convention. This lesson's reference program counts nodes; complexity is unaffected by counting edges instead."],
         ["Terminal marker", "A flag recording that a trie path forms a complete stored word, not just a prefix."]],
        "Search 6: 6 > 4, go right to 8. Then 6 < 8, go left to 6. Match. In a trie, cat and car share root → c → a, then diverge to t and r. A terminal flag distinguishes the word 'car' from a longer word such as 'cart'.",
        "Search-tree lookup is O(h) for height h: O(log n) if balanced, O(n) for a chain. Trie lookup follows O(L) character steps for a query of length L, assuming expected constant-time child lookup. Listing all matching words also costs the size of the returned output.",
        "Comparing only each node with its immediate children does not validate a search tree. A node in the root's left subtree must be less than the root even if its own parent comparison passes.",
        "Where would 5 belong in the example tree? Why would inserting sorted keys into an ordinary unbalanced BST be a problem?",
        "5 belongs left of 6. Repeatedly inserting increasing keys can build a chain, so searching may visit every node. Balanced search trees prevent that degeneration.",
        recipe("Search an ordered tree", '''
            def contains(node, target):
                while node is not None:
                    key, left, right = node
                    if target == key:
                        return True
                    if target < key:
                        node = left
                    else:
                        node = right
                return False
        ''', [
            ("Start at the root", "def contains", "Here a node is a tuple (key, left, right); an absent child is None.", "start"),
            ("Does this node exist?", "while node is not None:", "Reaching an absent child proves the key is not stored on this search path.", "decision"),
            ("Is target this key?", "if target == key:", "Equality completes the lookup.", "decision"),
            ("Return True", "return True", "The target is present.", "end"),
            ("Target smaller?", "if target < key:", "The whole left subtree is smaller; the whole right subtree is larger.", "decision"),
            ("Go left", "node = left", "Discard the right subtree.", "process"),
            ("Go right", "node = right", "Discard the left subtree.", "process"),
            ("Return False", "return False", "The search ran out of nodes without a match.", "end")
        ], [(0,1,""),(1,2,"Yes"),(1,7,"No"),(2,3,"Yes"),(2,4,"No"),(4,5,"Yes"),(4,6,"No"),(5,1,"Repeat"),(6,1,"Repeat")],
        [("contains((4, None, (8, (6, None, None), None)), 6)", "True"), ("contains(None, 6)", "False")])
    ),
    "DA12": lesson(
        "A min-heap is a partially ordered tree stored in an array. Every parent is no greater than its children, so the smallest item is always at the root. The other positions are not globally sorted.",
        "Keep the three highest scores from [1, 5, 3, 4, 2] without sorting and retaining every score.",
        "Keep the current winners in a small heap with the weakest winner at the top. A new score only needs to beat that weakest winner. If it does, replace the root; if it does not, leave the winners alone.",
        [["Heap property", "parent ≤ child in a min-heap; it is a relationship along edges, not between all array neighbours."],
         ["Array positions", "For index i, the children are 2i + 1 and 2i + 2; for i > 0 the parent is (i − 1) // 2."],
         ["Top-k", "The k highest-ranked items. This numeric recipe has no document-ID tie rule; the reference program makes its record-ranking rule explicit."]],
        "Fill the heap with 1, 5, 3 → [1, 5, 3]. The weakest winner is 1. Score 4 replaces it, and heap repair gives [3, 5, 4]. Score 2 cannot beat root 3, so it is discarded. Sorting only the final winners gives [5, 4, 3].",
        "For 1 ≤ k ≤ n, O(n log(k + 1)) processing time and O(k) retained heap space; sorting the final winners adds O(k log k). A heap push/pop travels at most the tree height, O(log k). Building a heap from all n items with heapify is O(n), a different operation.",
        "Iterating the heap array does not return sorted order. Also, heapq is a min-heap by default: keeping the largest k values means the root must be the smallest retained winner.",
        "If the current winners are [3, 5, 4], what should happen when 6 arrives? Why not replace the largest winner?",
        "Replace root 3, giving winners 4, 5, 6. Replacing the largest would throw away the best item and violate the top-k contract.",
        recipe("Keep the largest k scores", '''
            import heapq

            def top_k(scores, k):
                if k <= 0:
                    return []
                heap = []
                for score in scores:
                    if len(heap) < k:
                        heapq.heappush(heap, score)
                    elif score > heap[0]:
                        heapq.heapreplace(heap, score)
                return sorted(heap, reverse=True)
        ''', [
            ("Create heap; handle k ≤ 0", "if k <= 0:", "This recipe returns [] if no winners are requested.", "start"),
            ("Another score?", "for score in scores:", "Read one candidate at a time; no full sorted input is needed.", "decision"),
            ("Fewer than k winners?", "if len(heap) < k:", "The heap has room, so the candidate can be admitted directly.", "decision"),
            ("Push candidate", "heapq.heappush", "heapq restores the parent–child ordering after insertion.", "process"),
            ("Better than root?", "elif score > heap[0]:", "The root is the weakest retained winner.", "decision"),
            ("Replace root; repair heap", "heapq.heapreplace", "Discard exactly one weakest winner, then restore the heap property.", "process"),
            ("Sort winners; return", "return sorted", "Sorting is needed only if the returned winners must be ranked.", "end")
        ], [(0,1,"k > 0"),(1,2,"Yes"),(1,6,"No"),(2,3,"Yes"),(3,1,"Next"),(2,4,"No"),(4,5,"Yes"),(4,1,"No: discard"),(5,1,"Next")],
        [("top_k([1, 5, 3, 4, 2], 3)", "[5, 4, 3]"), ("top_k([2, 2], 5)", "[2, 2]"), ("top_k([1], 0)", "[]")])
    ),
    "DA13": lesson(
        "A graph represents objects as nodes and relationships as edges. Breadth-first search (BFS) explores by distance in edges; depth-first search (DFS) follows one branch before returning to alternatives.",
        "Starting at 'claim', visit reachable argument components: claim → evidence and warrant; evidence → detail. A separate 'note' is not reachable from claim.",
        "BFS is a queue of people waiting their turn: newly discovered neighbours join the back. DFS is exploring corridors: follow one corridor until it ends, then return to the last junction. Both need a seen set so cycles do not cause endless revisits.",
        [["Adjacency list", "A mapping from each node to its outgoing neighbours."],
         ["Discovered / processed", "Discovered means already put on the work list. Processed means removed from that list and expanded. They are not the same moment."],
         ["Directed edge", "A → B permits travel from A to B; it does not automatically permit travel back."]],
        "The BFS queue begins [claim]. Remove claim; add evidence and warrant → [evidence, warrant]. Remove evidence; add detail → [warrant, detail]. Remove warrant → [detail]. Remove detail → []. The visit order is claim, evidence, warrant, detail. Neighbour order breaks ties within a distance level.",
        "With adjacency lists and expected constant-time set membership, BFS/DFS take O(Vᵣ + Eᵣ) time over the reachable nodes and edges. The work list and seen set use O(Vᵣ) space. Saving a full queue snapshot at every step can add quadratic tracing overhead; it is not part of ordinary BFS.",
        "BFS gives shortest paths by number of edges, not by arbitrary edge weights. The seen set contains all discovered nodes; a DFS active path contains only unfinished calls, so nodes can leave the path while remaining seen.",
        "Why mark a neighbour seen when adding it to the queue rather than waiting until it is removed?",
        "Two parents could otherwise enqueue the same neighbour before either copy is processed. Marking at discovery prevents repeated work and makes the queue easier to reason about.",
        recipe("Breadth-first search", '''
            from collections import deque

            def bfs(graph, start):
                queue = deque([start])
                seen = {start}
                order = []
                while queue:
                    node = queue.popleft()
                    order.append(node)
                    for neighbour in graph.get(node, []):
                        if neighbour not in seen:
                            seen.add(neighbour)
                            queue.append(neighbour)
                return order
        ''', [
            ("Queue start; mark it seen", "queue = deque", "A queued node is discovered but has not necessarily been processed.", "start"),
            ("Any queued node?", "while queue:", "An empty queue means all reachable work has been processed.", "decision"),
            ("Remove oldest; visit it", "node = queue.popleft()", "The FIFO rule is what makes exploration breadth-first.", "process"),
            ("Another outgoing neighbour?", "for neighbour in graph.get", "Only edges leaving this node are considered.", "decision"),
            ("Is neighbour new?", "if neighbour not in seen:", "Skip a node already discovered, even if it is still waiting in the queue.", "decision"),
            ("Mark seen; add to back", "seen.add(neighbour)", "These two operations belong together: discovery creates exactly one future queue entry.", "process"),
            ("Return visit order", "return order", "Unreachable nodes do not appear in this result.", "end")
        ], [(0,1,""),(1,2,"Yes"),(1,6,"No"),(2,3,""),(3,4,"Yes"),(3,1,"No"),(4,5,"Yes"),(4,3,"No"),(5,3,"Next neighbour")],
        [("bfs({'claim': ['evidence', 'warrant'], 'evidence': ['detail']}, 'claim')", "['claim', 'evidence', 'warrant', 'detail']"), ("bfs({'a': ['a']}, 'a')", "['a']")])
    ),
    "DA14": lesson(
        "Topological sorting puts prerequisites before their dependants. Dijkstra's algorithm solves a different problem: finding minimum-cost routes when all edge weights are nonnegative. Both use a frontier, but choose the next node for different reasons.",
        "If A must precede B and C, and both B and C must precede D, find a legal processing order. Separately, compare a direct route costing 10 with a two-edge route costing 2 + 3.",
        "For dependencies, repeatedly do a task with no unfinished prerequisite. For distances, repeatedly commit the smallest tentative route cost. 'Ready to run' and 'closest so far' are different selection rules; the correct container follows that rule.",
        [["Indegree", "How many incoming dependency edges a node still has."],
         ["Cycle", "A directed route that returns to its start. A dependency cycle prevents a complete topological order."],
         ["Relaxation", "If reaching u costs d, test whether d + weight(u, v) improves the best known cost of reaching v."]],
        "Initially A has indegree 0; B and C each have 1; D has 2. Process A → B and C become ready. Process B → D still waits for C. Process C → D becomes ready. A, B, C, D is valid; A, C, B, D is valid too. In the route example, Dijkstra must prefer total cost 5 over direct cost 10.",
        "Kahn's topological sort is O(V + E) time and O(V + E) auxiliary storage in this implementation, which builds adjacency lists. Heap-based Dijkstra is O((V + E) log V) for a simple graph, with nonnegative weights; stale heap entries must be skipped.",
        "A topological order is not unique. A result shorter than the node list signals a cycle. Dijkstra cannot safely finalise a node with negative edges: a later route could make its supposedly final distance smaller.",
        "If B also requires D in this graph, can the queue finish every node?",
        "No. B and D form a dependency cycle. Processing A and C does not release B or D, so the returned order would be incomplete; this recipe returns None instead.",
        recipe("Process tasks with no remaining prerequisites", '''
            from collections import deque

            def topological_order(nodes, edges):
                outgoing = {node: [] for node in nodes}
                indegree = {node: 0 for node in nodes}
                for source, target in edges:
                    outgoing[source].append(target)
                    indegree[target] += 1
                ready = deque(node for node in nodes if indegree[node] == 0)
                order = []
                while ready:
                    node = ready.popleft()
                    order.append(node)
                    for neighbour in outgoing[node]:
                        indegree[neighbour] -= 1
                        if indegree[neighbour] == 0:
                            ready.append(neighbour)
                return order if len(order) == len(nodes) else None
        ''', [
            ("Count prerequisites; queue zeros", "outgoing =", "Inputs list every node; each edge points from prerequisite to dependant.", "start"),
            ("Any ready task?", "while ready:", "Ready means all of this task's incoming prerequisites have been removed.", "decision"),
            ("Process oldest ready task", "node = ready.popleft()", "Any ready task is legal; the queue chooses one deterministic order.", "process"),
            ("Another dependant?", "for neighbour in outgoing", "Completing this task removes one incoming prerequisite from each dependant.", "decision"),
            ("Subtract one prerequisite", "indegree[neighbour] -= 1", "The remaining count must reach zero before the dependant becomes ready.", "process"),
            ("Now zero? Enqueue it", "if indegree[neighbour] == 0:", "A task is added precisely when its last prerequisite is removed.", "process"),
            ("All processed? Return order or None", "return order if", "Unprocessed nodes after the ready queue empties imply a cycle.", "end")
        ], [(0,1,""),(1,2,"Yes"),(1,6,"No"),(2,3,""),(3,4,"Yes"),(3,1,"No"),(4,5,""),(5,3,"Next edge")],
        [("topological_order(['A', 'B', 'C', 'D'], [('A','B'), ('A','C'), ('B','D'), ('C','D')])", "['A', 'B', 'C', 'D']"), ("topological_order(['A', 'B'], [('A','B'), ('B','A')])", "None")])
    ),
    "DA15": lesson(
        "Edit distance is the smallest number of insertions, deletions and substitutions that turn one string into another. Dynamic programming finds it by saving answers for smaller prefix pairs.",
        "Turn 'cat' into 'cut' with unit-cost edits. Replacing a with u costs 1; deleting and inserting would cost 2.",
        "Ask an easier question first: how many edits turn 'ca' into 'cu'? Its last operation must delete, insert, or match/replace the final character. If those smaller answers are already stored, choose the cheapest route into the current cell.",
        [["State dp[i][j]", "Minimum edits from the first i source characters to the first j target characters. i and j count characters; they are not the last character indices."],
         ["Recurrence", "A rule that computes one state from smaller states. Here: min(delete, insert, match/replace)."],
         ["Base row and column", "Turning an empty string into j characters takes j insertions; turning i characters into empty takes i deletions."]],
        "For ca → cu, the cell above costs 1, so deleting a costs 1 + 1 = 2. The cell left costs 1, so inserting u costs 1 + 1 = 2. The diagonal c → c costs 0; substituting a → u costs 0 + 1 = 1. Store min(2, 2, 1) = 1. Matching the final t adds 0, so cat → cut also costs 1.",
        "For lengths m and n, this table takes O(mn) time and O(mn) space. If only the distance is needed, two rows reduce storage to O(min(m, n)); recovering an actual edit sequence needs more information or a reconstruction method.",
        "A table cell is a complete prefix-pair question, not a guessed label for a character. Filling cells in the wrong order can read answers that have not been computed. The distance alone does not specify the edits.",
        "Why does the diagonal add zero when the two final characters match? What should empty → 'at' cost?",
        "The final matching character requires no edit, so reuse the shorter-prefix answer unchanged. Empty → 'at' costs 2 insertions.",
        recipe("Build an edit-distance table", '''
            def edit_distance(source, target):
                m, n = len(source), len(target)
                dp = [[0] * (n + 1) for _ in range(m + 1)]
                for i in range(m + 1):
                    dp[i][0] = i
                for j in range(n + 1):
                    dp[0][j] = j
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        replacement = int(source[i - 1] != target[j - 1])
                        dp[i][j] = min(dp[i - 1][j] + 1,
                                       dp[i][j - 1] + 1,
                                       dp[i - 1][j - 1] + replacement)
                return dp[m][n]
        ''', [
            ("Create table; fill empty prefixes", "dp = [[0]", "Each row is independent. Initialise column zero and row zero before the interior.", "start"),
            ("Another source prefix?", "for i in range(1", "Rows are processed from shorter to longer source prefixes.", "decision"),
            ("Another target prefix?", "for j in range(1", "Inside a row, process target prefixes from left to right.", "decision"),
            ("Match cost 0; replace cost 1", "replacement =", "Compare the new final characters, at indices i − 1 and j − 1.", "process"),
            ("Store min of three routes", "dp[i][j] = min", "Above + 1 deletes; left + 1 inserts; diagonal + replacement matches or substitutes.", "process"),
            ("Return bottom-right cell", "return dp[m][n]", "This state describes the complete source and target strings.", "end")
        ], [(0,1,""),(1,2,"Yes"),(1,5,"No"),(2,3,"Yes"),(2,1,"Next row"),(3,4,""),(4,2,"Next column")],
        [("edit_distance('cat', 'cut')", "1"), ("edit_distance('', 'at')", "2"), ("edit_distance('cat', 'at')", "1")])
    ),
    "DA16": lesson(
        "Zero-one knapsack selects items with the highest total value without exceeding a capacity. Each item can be used once or not at all. Dynamic programming stores the best answer for each item-prefix and capacity.",
        "You have a capacity of 3 and items (weight, value): a = (2, 3), b = (2, 4), c = (1, 2). Choose b + c for value 6, rather than a + c for 5.",
        "For the next item, compare two complete alternatives: leave it out, or take it and use the remaining capacity on earlier items. Both alternatives must use the previous row so that this item is not accidentally taken twice.",
        [["State dp[i][c]", "Best value using only the first i items with capacity at most c."],
         ["Zero-one", "An item is either absent or present once. Unlimited reuse is a different problem."],
         ["Pseudo-polynomial", "Runtime depends on the numerical capacity C, not just the number of bits needed to write C."]],
        "Before c, the best value at capacity 3 is 4 from b. Now compare: skip c → keep 4. Take c → its value 2 + previous-row value at remaining capacity 2, which is 4 from b. Store max(4, 2 + 4) = 6. That combines b and c without using either item twice.",
        "O(nC) time and O(nC) table space for n items and nonnegative integer capacity C. A one-row version uses O(C) space, but must scan capacities downward to avoid reusing the current item. This recipe assumes positive integer weights and nonnegative values.",
        "Sorting by value or value/weight does not generally solve zero-one knapsack. The table returns the best value; recovering the selected items requires tracing the choices or storing them.",
        "With only one item of weight 2 and value 4, what should capacity 4 return?",
        "4, not 8. The item is available once. A one-row algorithm that updates upward may mistakenly reuse it.",
        recipe("Compare skip versus take", '''
            def knapsack(items, capacity):
                dp = [[0] * (capacity + 1) for _ in range(len(items) + 1)]
                for i, (weight, value) in enumerate(items, start=1):
                    for room in range(capacity + 1):
                        best = dp[i - 1][room]
                        if weight <= room:
                            best = max(best, value + dp[i - 1][room - weight])
                        dp[i][room] = best
                return dp[-1][capacity]
        ''', [
            ("Create zero-item row", "dp = [[0]", "No items gives value zero for every capacity.", "start"),
            ("Another item?", "for i, (weight, value)", "Row i may use this item at most once.", "decision"),
            ("Another capacity?", "for room in range", "Each column asks a separate allowed-capacity question.", "decision"),
            ("Start with skip value", "best = dp[i - 1][room]", "Not taking the item preserves the best value from the previous row.", "process"),
            ("Does the item fit?", "if weight <= room:", "If not, the skip value is the only valid alternative.", "decision"),
            ("Compare with take value", "best = max", "Take this item's value plus the previous row at the remaining capacity.", "process"),
            ("Store the better value", "dp[i][room] = best", "This cell now answers the current prefix/capacity question.", "process"),
            ("Return final value", "return dp[-1][capacity]", "All items have been considered at the requested capacity.", "end")
        ], [(0,1,""),(1,2,"Yes"),(1,7,"No"),(2,3,"Yes"),(2,1,"Next item"),(3,4,""),(4,5,"Yes"),(4,6,"No"),(5,6,""),(6,2,"Next capacity")],
        [("knapsack([(2, 3), (2, 4), (1, 2)], 3)", "6"), ("knapsack([(2, 4)], 4)", "4"), ("knapsack([], 3)", "0")])
    ),
    "DA17": lesson(
        "A greedy algorithm commits to the best-looking next choice without exploring every combination. It is correct only when that local choice can be justified for the exact objective.",
        "Choose as many non-overlapping time intervals as possible. Each interval has equal value; an interval ending at t can be followed by one starting at t.",
        "Choose the interval that finishes earliest. It leaves the largest possible amount of future time. Choosing the earliest start can trap you in one long interval and block many short ones.",
        [["Greedy choice", "A permanent local decision made without reconsidering all earlier decisions."],
         ["Exchange argument", "Show that an optimal solution can swap in the greedy choice without getting worse. This explains why the choice is safe, rather than just showing examples where it works."]],
        "For long = [0, 6), a = [1, 2), b = [2, 3), c = [3, 4): earliest start picks only long. Earliest finish picks a, then b, then c: three intervals. An optimal first interval can be replaced by the earliest-finishing one without making later choices less feasible.",
        "Sort by finish time in O(n log n), then scan in O(n); this implementation stores O(n) output and uses sorted()'s copy. If values differ and the goal is total reward rather than interval count, use weighted interval scheduling instead.",
        "Do not initialise the last finish time to −1: valid intervals can start earlier. Use None to mean that no interval has been chosen yet. This scheduling contract assumes positive-duration intervals, start < end, and comparable boundary times.",
        "If long has reward 100 and each short interval reward 1, is earliest finish still optimal for reward?",
        "No. It maximises count under equal value, not arbitrary reward. Changing the objective invalidates the proof.",
        recipe("Choose the maximum number of intervals", '''
            def schedule(intervals):
                chosen = []
                last_end = None
                for name, start, end in sorted(intervals, key=lambda item: item[2]):
                    if last_end is None or start >= last_end:
                        chosen.append(name)
                        last_end = end
                return chosen
        ''', [
            ("Sort by finish time", "for name, start, end in sorted", "The loop consumes intervals in increasing finish time, not start time.", "start"),
            ("Another interval?", "for name, start, end in sorted", "The next candidate is the earliest-finishing interval not yet examined.", "decision"),
            ("Compatible with last choice?", "if last_end is None", "The first choice is always allowed; later choices must begin after or at the last end.", "decision"),
            ("Keep it; update finish", "chosen.append(name)", "The choice is permanent because the exchange argument makes it safe for this objective.", "process"),
            ("Return chosen names", "return chosen", "The output maximises interval count, not weighted reward.", "end")
        ], [(0,1,""),(1,2,"Yes"),(1,4,"No"),(2,3,"Yes"),(2,1,"No: skip"),(3,1,"Next interval")],
        [("schedule([('long', 0, 6), ('a', 1, 2), ('b', 2, 3), ('c', 3, 4)])", "['a', 'b', 'c']"), ("schedule([('z', -3, -2)])", "['z']")])
    ),
    "DA18": lesson(
        "A streaming algorithm consumes items as they arrive and retains only the state needed for its answer. It does not mean that every algorithm can produce an exact answer with constant memory.",
        "A file arrives in batches of scores. Retain only the highest two scores, not every earlier batch.",
        "The bounded heap from DA12 becomes useful when scores come from an iterator rather than a ready-made list. The algorithm can forget a weak score permanently because later arrivals cannot make it a top-two winner again.",
        [["Iterator", "An object that supplies the next item on request; it need not hold the entire sequence in memory."],
         ["Working set", "The live data the algorithm currently retains, distinct from the input source's own buffering."],
         ["Exact versus approximate", "An exact answer satisfies the full contract. Some streaming tasks need approximate summaries because exact state would grow with the data."]],
        "Read batch [1, 5]: retain [1, 5]. Read [3, 4]: 3 replaces 1, then 4 replaces 3 → winners [4, 5]. Read [2]: discard 2. Five scores were seen, but only two winners are retained. A generator can release each consumed batch; a prebuilt list of batches cannot demonstrate that memory benefit.",
        "For positive k, O(n log(k + 1)) processing and O(k) heap storage, plus the input provider's current batch and any output sorting. Exact deduplication is different: remembering every distinct ID needs O(u) state unless the problem permits external storage or approximation.",
        "Do not call list(generator) before processing if the aim is streaming. Memory claims must include input buffering and retained traces; keeping every intermediate heap snapshot defeats a bounded-state claim.",
        "Can the same bounded heap give the exact median of an arbitrary-length stream?",
        "No. A top-k heap with a fixed small k does not preserve enough information for an arbitrary exact median. Different questions require different retained state.",
        recipe("Keep top scores from batches", '''
            import heapq

            def streamed_top_k(batches, k):
                if k <= 0:
                    return []
                heap = []
                for batch in batches:
                    for score in batch:
                        if len(heap) < k:
                            heapq.heappush(heap, score)
                        elif score > heap[0]:
                            heapq.heapreplace(heap, score)
                return sorted(heap, reverse=True)
        ''', [
            ("Create heap; handle k ≤ 0", "if k <= 0:", "The recipe returns immediately if no winners are requested.", "start"),
            ("Another batch?", "for batch in batches:", "The input can be a generator that loads one batch at a time.", "decision"),
            ("Another score in batch?", "for score in batch:", "All retained ranking state lives in heap, not in old batches.", "decision"),
            ("Heap not full?", "if len(heap) < k:", "The first k scores fill the available winner slots.", "decision"),
            ("Push score", "heapq.heappush", "The heap grows by one, up to k entries.", "process"),
            ("Beats root? Replace it", "elif score > heap[0]:", "Otherwise discard this candidate. No extra history is needed for exact top-k scores.", "process"),
            ("Sort and return winners", "return sorted", "No more batches remain; only the winners are sorted.", "end")
        ], [(0,1,"k > 0"),(1,2,"Yes"),(1,6,"No"),(2,3,"Yes"),(2,1,"No"),(3,4,"Yes"),(4,2,"Next score"),(3,5,"No"),(5,2,"Next score")],
        [("streamed_top_k(iter([[1, 5], [3, 4], [2]]), 2)", "[5, 4]"), ("streamed_top_k(iter([]), 2)", "[]")])
    ),
    "DA19": lesson(
        "Testing an algorithm means checking its promised behaviour, not merely seeing whether the code runs. A faster replacement must preserve the same answers, ordering and mutation rules.",
        "Someone replaces order-preserving deduplication with sorted(set(ids)). Design a tiny test that exposes the changed contract.",
        "First write down an answer you can establish independently. Then choose an input that forces the disputed behaviour to matter. ['b', 'a', 'b'] is more informative for order preservation than ['a', 'b', 'a'].",
        [["Oracle", "An independent source of the expected answer: a hand-worked example, a trusted reference, or a simple exhaustive implementation on small inputs."],
         ["Differential test", "Run two implementations on the same inputs and compare their outputs."],
         ["Property", "A rule that should hold for many inputs, such as removing duplicates twice having the same result as doing it once."]],
        "The expected answer for ['b', 'a', 'b'] is ['b', 'a']. sorted(set(...)) returns ['a', 'b'], so the order test fails even though both outputs contain the same set of values. Also compare the original list after the call to catch unwanted mutation.",
        "Small exhaustive tests give strong evidence for their tested domain but are not a proof for every possible input. Benchmark only after correctness checks, use representative sizes, and separate algorithm work from input generation and logging.",
        "If both implementations share the same mistake, differential agreement is weak evidence. Include independently worked cases and invariants, not only a second implementation.",
        "What other cases should a deduplication test include?",
        "Empty input, one item, all identical, all distinct, repeated items separated by other items, nonalphabetical order, and verification that the input is unchanged.",
        recipe("Check an order-preserving deduplicator", '''
            def check_keep_first(candidate):
                cases = [([], []), (['x', 'x'], ['x']),
                         (['b', 'a', 'b'], ['b', 'a'])]
                for original, expected in cases:
                    supplied = list(original)
                    actual = candidate(supplied)
                    if actual != expected or supplied != original:
                        return False
                return True
        ''', [
            ("Choose independent expected answers", "cases =", "Include a case where sorting would visibly violate order preservation.", "start"),
            ("Another test case?", "for original, expected", "Each case checks output and input preservation.", "decision"),
            ("Copy input; run candidate", "supplied = list", "Keep the original untouched so mutation can be detected.", "process"),
            ("Wrong output or mutated input?", "if actual != expected", "Either discrepancy violates the agreed contract.", "decision"),
            ("Return failure", "return False", "One counterexample is enough to disprove correctness for this contract.", "end"),
            ("Return all tests passed", "return True", "This means these tests passed, not that every possible input has been proved correct.", "end")
        ], [(0,1,""),(1,2,"Yes"),(1,5,"No"),(2,3,""),(3,4,"Yes"),(3,1,"No: next test")],
        [("check_keep_first(lambda ids: list(dict.fromkeys(ids)))", "True"), ("check_keep_first(lambda ids: sorted(set(ids)))", "False")])
    ),
    "DA20": lesson(
        "A real data-processing pipeline combines several small algorithms with explicit boundaries. Each stage should answer one question and pass a clearly defined result to the next.",
        "Rank scored documents, but do not return the same document twice. For [('d1', 2), ('d2', 5), ('d1', 9), ('d3', 4)], keep each ID's first score, then return the best two IDs: ['d2', 'd3'].",
        "First settle identity and duplicate policy, then settle ranking. Otherwise a later high-scoring duplicate silently changes what 'first occurrence' means. Graph traversal is another independent stage: following evidence links does not alter a document's ranking score by itself.",
        [["Stage contract", "What a stage consumes, produces, and guarantees. Deduplication and ranking need different contracts."],
         ["Tie rule", "How equally scored records are ordered. This recipe uses earlier first occurrence first, because Python's sort is stable."]],
        "Deduplication keeps d1:2, d2:5, d3:4 and discards the later d1:9. Ranking gives d2:5, d3:4, d1:2. Taking two returns d2 and d3. Reversing the order of ranking and deduplication could instead preserve d1:9, a different task.",
        "For n bounded-length IDs and u distinct IDs: expected O(n) deduplication, O(u log u) full sorting and O(u) retained records. For a much larger u and small k, replace the sort with the explicitly tie-broken bounded heap from DA12. Do not introduce that complexity until the contracts are correct.",
        "This is a clarity-first sort-based recipe, not a claim that every stage is memory-bounded. The full reference below also contains top-k record ranking and a graph walk; those are separate reusable functions.",
        "If you wanted the highest score ever observed per document instead of the first score, which stage changes?",
        "The deduplication stage becomes aggregation: store max(previous, score) per ID. The ranking stage can stay the same. Naming the changed contract prevents an accidental behaviour change.",
        recipe("Deduplicate first, then rank", '''
            def best_unique(rows, k):
                if k <= 0:
                    return []
                seen = set()
                unique = []
                for doc_id, score in rows:
                    if doc_id not in seen:
                        seen.add(doc_id)
                        unique.append((doc_id, score))
                ranked = sorted(unique, key=lambda row: row[1], reverse=True)
                return [doc_id for doc_id, score in ranked[:k]]
        ''', [
            ("Handle k; create seen and unique", "if k <= 0:", "Return [] when no documents are requested; otherwise prepare the identity stage.", "start"),
            ("Another scored record?", "for doc_id, score in rows:", "Read records in their original arrival order.", "decision"),
            ("Is document ID new?", "if doc_id not in seen:", "This policy deliberately ignores later scores for an already seen ID.", "decision"),
            ("Keep first scored record", "seen.add(doc_id)", "Remember identity and preserve that record's score together.", "process"),
            ("Sort unique records by score", "ranked = sorted", "Stable sorting preserves first-occurrence order for ties.", "process"),
            ("Return first k IDs", "return [doc_id", "Return identifiers, not the intermediate record tuples.", "end")
        ], [(0,1,"k > 0"),(1,2,"Yes"),(1,4,"No"),(2,3,"Yes"),(2,1,"No: skip"),(3,1,"Next record"),(4,5,"")],
        [("best_unique([('d1', 2), ('d2', 5), ('d1', 9), ('d3', 4)], 2)", "['d2', 'd3']"), ("best_unique([('b', 5), ('a', 5)], 2)", "['b', 'a']")])
    ),
}

LESSONS["DA13"]["recipes"].append(recipe("Depth-first search: return before taking the next branch", '''
    def dfs(graph, start):
        seen = set()
        order = []
        def visit(node):
            seen.add(node)
            order.append(node)
            for neighbour in graph.get(node, []):
                if neighbour not in seen:
                    visit(neighbour)
            return None
        visit(start)
        return order
''', [
    ("Create seen and order; call start", "seen = set()", "Unlike BFS, the call stack holds the unfinished traversal work.", "start"),
    ("Enter node; mark and record it", "seen.add(node)", "Record discovery before following edges, so a cycle cannot re-enter this node.", "process"),
    ("Another neighbour in this call?", "for neighbour in graph.get", "This loop belongs to the current call. A recursive child must finish before it resumes.", "decision"),
    ("Is neighbour new?", "if neighbour not in seen:", "A previously discovered node needs no further recursive visit.", "decision"),
    ("Visit neighbour; wait for return", "visit(neighbour)", "The child runs the same entry/loop procedure in a new call. On return, continue the current call's next neighbour.", "process"),
    ("Return to caller", "return None", "Finish this visit and resume its caller's neighbour loop. After the initial visit(start) returns, the outer dfs function returns the recorded order.", "end")
], [(0,1,""),(1,2,""),(2,3,"Yes"),(2,5,"No"),(3,4,"Yes"),(3,2,"No: next"),(4,2,"After child returns")],
[("dfs({'claim': ['evidence', 'warrant'], 'evidence': ['detail']}, 'claim')", "['claim', 'evidence', 'detail', 'warrant']"), ("dfs({'a': ['b'], 'b': ['a']}, 'a')", "['a', 'b']")]))

LESSONS["DA14"]["recipes"].append(recipe("Dijkstra: choose the cheapest tentative route", '''
    import heapq

    def shortest_cost(graph, start, goal):
        if any(weight < 0 for edges in graph.values() for _, weight in edges):
            raise ValueError("Dijkstra needs nonnegative weights")
        distances = {start: 0}
        frontier = [(0, start)]
        while frontier:
            cost, node = heapq.heappop(frontier)
            if cost != distances[node]:
                continue
            if node == goal:
                return cost
            for neighbour, weight in graph.get(node, []):
                candidate = cost + weight
                if candidate < distances.get(neighbour, float('inf')):
                    distances[neighbour] = candidate
                    heapq.heappush(frontier, (candidate, neighbour))
        return None
''', [
    ("Reject negative weights; queue start", "if any(weight < 0", "Edges are (neighbour, weight) pairs; node IDs here are comparable strings for heap ties. An unseen distance is infinity, not a large guessed number.", "start"),
    ("Any tentative route left?", "while frontier:", "The frontier is a min-heap of tentative cost/node pairs.", "decision"),
    ("Pop cheapest cost; stale entry?", "if cost != distances[node]:", "A node can have several heap entries after improvements. Skip a cost that is no longer its best known distance.", "decision"),
    ("Is this the goal?", "if node == goal:", "With nonnegative weights, this current smallest cost cannot be improved by a later route.", "decision"),
    ("Return minimum cost", "return cost", "The output is a cost only; recovering a path would also require predecessor links.", "end"),
    ("Another outgoing edge?", "for neighbour, weight", "Examine one possible extension of the current route.", "decision"),
    ("Does new route improve cost?", "if candidate < distances.get", "Relaxation compares cost + edge weight with the previous best for this neighbour.", "decision"),
    ("Save improvement; push route", "distances[neighbour] = candidate", "The older heap entry may remain; the stale-entry test deals with it later.", "process"),
    ("Return no route", "return None", "An empty frontier without reaching the goal means it is unreachable.", "end")
], [(0,1,"Valid input"),(1,2,"Yes"),(1,8,"No"),(2,1,"Yes: skip"),(2,3,"No"),(3,4,"Yes"),(3,5,"No"),(5,6,"Yes"),(5,1,"No"),(6,7,"Yes"),(6,5,"No"),(7,5,"Next edge")],
[("shortest_cost({'A': [('B', 2), ('C', 10)], 'B': [('C', 3)]}, 'A', 'C')", "5"), ("shortest_cost({'A': [('B', 10**9)]}, 'A', 'B')", "1000000000"), ("shortest_cost({'A': []}, 'A', 'Z')", "None")]))


def teaching_for(identifier):
    """Return data only. No lesson code executes while serving the website."""
    return deepcopy(LESSONS[identifier])


def check_recipes():
    """Execute only these small standard-library examples during explicit QA."""
    count = 0
    for identifier, teaching in LESSONS.items():
        for item in teaching["recipes"]:
            ids = {node["id"] for node in item["nodes"]}
            for node in item["nodes"]:
                if node["anchor"] not in item["code"]:
                    raise AssertionError((identifier, "missing code anchor", node["anchor"]))
            for edge in item["edges"]:
                if edge["from"] not in ids or edge["to"] not in ids:
                    raise AssertionError((identifier, "unknown flow node", edge))
            scope = {}
            exec(compile(item["code"], identifier + " recipe", "exec"), scope)
            for example in item["examples"]:
                actual = repr(eval(example["expression"], scope))
                if actual != example["output"]:
                    raise AssertionError((identifier, example["expression"], actual, example["output"]))
                count += 1
    return count
