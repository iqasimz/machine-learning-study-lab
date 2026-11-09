"""Finite picture steps for the DSA lessons.

The website renders this payload. It does not keep a second copy of an
algorithm in JavaScript. Every step is a state the reference procedure
actually reaches, or a labelled counterexample that the procedure rejects.
"""
import heapq
from collections import deque

KINDS = {"bounds", "heap", "graph", "grid", "tree", "intervals", "rows"}


def step(action, why, alt, picture, code="", flow=None, flow_at=0, predict=""):
    if picture.get("kind") not in KINDS:
        raise ValueError("Unknown picture.")
    row = {"action": action, "why": why, "alt": alt, "picture": picture, "code": code}
    if flow:
        row["flow"] = list(flow)
        row["flowAt"] = flow_at
    if predict:
        row["predict"] = predict
    return row


def case(identifier, label, prompt, steps):
    if len(steps) < 2 or steps[0]["alt"] == steps[1]["alt"]:
        raise ValueError("A case needs two different states: " + identifier)
    return {"id": identifier, "label": label, "prompt": prompt, "steps": steps}


def pack(identifier, title, legend, cases):
    return {"id": identifier, "title": title, "legend": list(legend), "cases": list(cases)}


def cells(*pairs):
    return [{"text": text, "mark": mark} for text, mark in pairs]


def rows(*items):
    return {"kind": "rows", "rows": [{"label": label, "cells": data} for label, data in items]}


def bounds(values, lo, hi, mid=None, move=""):
    picture = {"kind": "bounds", "values": list(values), "lo": lo, "hi": hi}
    if mid is not None:
        picture["mid"] = mid
    if move:
        picture["move"] = move
    return picture


def heap_picture(entries, focus=""):
    return {"kind": "heap", "entries": [{"id": doc_id, "score": score} for doc_id, score in entries], "focus": focus}


def graph_picture(places, edges, frontier, visited, path, active=""):
    return {
        "kind": "graph",
        "nodes": [{"id": name, "x": x, "y": y} for name, (x, y) in places.items()],
        "edges": [{"src": src, "dst": dst} for src, dst in edges],
        "frontier": list(frontier),
        "visited": sorted(visited),
        "path": list(path),
        "active": active,
    }


def grid(row_labels, col_labels, table, focus=None, deps=(), row_axis="source prefix", col_axis="target prefix", meaning="Cell = minimum edit cost. ∅ means the empty prefix."):
    return {
        "kind": "grid",
        "rowLabels": list(row_labels),
        "colLabels": list(col_labels),
        "cells": table,
        "focus": list(focus) if focus else [],
        "deps": [list(pair) for pair in deps],
        "rowAxis": row_axis, "colAxis": col_axis, "meaning": meaning,
    }


def tree(nodes, edges, focus=""):
    return {
        "kind": "tree",
        "nodes": [{"id": name, "label": label, "x": x, "y": y} for name, label, x, y in nodes],
        "edges": [{"src": src, "dst": dst} for src, dst in edges],
        "focus": focus,
    }


SEARCH_FLOW = [
    "Set the unresolved interval",
    "Read the middle cell",
    "Move exactly one boundary",
    "Stop when lo and hi meet",
]


def binary_case(values, target, identifier, label, prompt):
    lo, hi = 0, len(values)
    shown = ", ".join(map(str, values)) if values else "no cells"
    steps = [step(
        "Mark the unresolved interval.",
        f"The target is {target}. lo is 0 and hi is {len(values)}. hi may sit one past the last cell, and that position is never read.",
        f"Values {shown}. Unresolved from index {lo} up to {hi}. No cell has been read.",
        bounds(values, lo, hi),
        "lo, hi = 0, len(values)",
        SEARCH_FLOW, 0,
        "Which index is read first?" if values else "",
    )]
    while lo < hi:
        mid = (lo + hi) // 2
        value = values[mid]
        below = value < target
        if below:
            action = f"Read index {mid}. The value is below the target, so move lo."
            why = f"{value} is below {target}. Indexes through {mid} are too small. lo becomes {mid + 1}. Setting lo to mid would leave the interval unchanged when mid is already lo."
            code = "lo = mid + 1"
            move = "lo"
        else:
            action = f"Read index {mid}. The value is not below the target, so move hi."
            why = f"{value} is not below {target}. The first valid index is no later than {mid}, so hi becomes {mid}. The boundary hi can still be the answer."
            code = "hi = mid"
            move = "hi"
        steps.append(step(
            action, why,
            f"Index {mid} holds {value}. Before the move, lo is {lo} and hi is {hi}. The move is {move}.",
            bounds(values, lo, hi, mid, move),
            code, SEARCH_FLOW, 2,
        ))
        if below:
            lo = mid + 1
        else:
            hi = mid
        steps.append(step(
            f"The new unresolved interval is [{lo}, {hi}).",
            "The boundary has now moved. Grey positions have been ruled out of the next read; green positions still need checking. "
            "The answer may equal hi, even though hi itself is excluded from the next read.",
            f"After the update, lo is {lo} and hi is {hi}. The next iteration reads only inside this smaller interval.",
            bounds(values, lo, hi), "while lo < hi",
        ))
    steps.append(step(
        "Stop. The bounds meet.",
        f"lo and hi are both {lo}. The test lo < hi fails, so no further cell is read. The boundary is {lo}.",
        f"Empty unresolved interval at index {lo}. Return {lo}.",
        bounds(values, lo, hi),
        "return lo", SEARCH_FLOW, 3,
    ))
    return case(identifier, label, prompt, steps)


def stall_case():
    values = [2, 4, 4, 7, 9]
    return case("stall", "An update that stalls", "lo is 0, hi is 1, and the inspected value 2 is below the target.", [
        step(
            "The interval is one cell wide.",
            "mid is (0 + 1) // 2, which is 0. The value 2 is below 4, so this cell cannot be the boundary.",
            "lo is 0, hi is 1, mid is 0, and the value is 2.",
            bounds(values, 0, 1, 0, "lo"),
            "mid = (lo + hi) // 2",
            SEARCH_FLOW, 1,
            "What happens if the code sets lo = mid?",
        ),
        step(
            "The wrong update does not shrink the interval.",
            "lo = mid writes 0 again. hi is still 1. The next mid is 0 again, so the loop never finishes.",
            "After lo = mid, lo is still 0 and hi is still 1. The same cell would be read forever.",
            bounds(values, 0, 1, 0, "stall"),
            "lo = mid",
            SEARCH_FLOW, 2,
        ),
        step(
            "The working update moves lo past the cell.",
            "lo = mid + 1 writes 1. lo and hi now meet, and the boundary is 1.",
            "After lo = mid + 1, lo is 1 and hi is 1. The search stops at boundary 1.",
            bounds(values, 1, 1),
            "lo = mid + 1",
            SEARCH_FLOW, 3,
        ),
    ])


def scene_da05():
    values = [2, 4, 4, 7, 9]
    return pack("DA05", "Binary search picture", [
        "below: index is before lo",
        "unresolved: lo up to but not including hi",
        "later: index is at or after hi; hi itself can still be returned",
        "inspected: the middle cell just read",
    ], [
        binary_case(values, 4, "present", "First 4", "Find the first value at least 4 in [2, 4, 4, 7, 9]."),
        binary_case(values, 5, "absent", "Absent 5", "5 is missing. The boundary should land on the first larger value."),
        binary_case([], 4, "empty", "Empty list", "An empty list has no cell to read."),
        stall_case(),
    ])


def heap_frames(pairs):
    heap = []
    frames = [[]]
    for doc_id, score in pairs:
        heapq.heappush(heap, (score, doc_id))
        frames.append([(item_id, item_score) for item_score, item_id in heap])
    return frames


def scene_da12():
    pairs = [("a", 1), ("b", 5), ("c", 3), ("d", 4), ("e", 2)]
    frames = heap_frames(pairs)
    steps = [step(
        "Start with an empty heap.",
        "Index 0 will be the root. Children of index i are 2i+1 and 2i+2. The parent of i is (i-1)//2.",
        "The heap has no entries.",
        heap_picture([]),
        "heap = []",
        ["Add the next score at the end", "Swap it upward while its parent is larger", "Read the root as the smallest score"],
        0,
        "Where does the first score sit?",
    )]
    building = []
    for doc_id, score in pairs:
        building.append((doc_id, score))
        index = len(building) - 1
        steps.append(step(
            f"Append {doc_id}:{score} at index {index}.",
            "Appending preserves the complete tree shape. The heap order may temporarily be broken; compare the new item with its parent next. "
            "These are internal insertion steps, not separately exposed heapq calls.",
            "After append, before repair: " + " ".join(f"{name}:{value}" for name, value in building) + ".",
            heap_picture(building, doc_id), "heappush: append internally",
        ))
        while index > 0:
            parent = (index - 1) // 2
            parent_id, parent_score = building[parent]
            if (parent_score, parent_id) <= (score, doc_id):
                steps.append(step(
                    f"Stop: parent {parent_id}:{parent_score} is no greater.",
                    f"The new item {doc_id}:{score} can stay at index {index}. The heap property is restored along its ancestor path.",
                    f"Parent at {parent} is {parent_id}:{parent_score}; child at {index} is {doc_id}:{score}. No swap.",
                    heap_picture(building, doc_id), "heappush: stop repair",
                ))
                break
            building[parent], building[index] = building[index], building[parent]
            steps.append(step(
                f"Swap {doc_id}:{score} with parent {parent_id}:{parent_score}.",
                f"{score} is smaller than {parent_score}. Move it from index {index} to index {parent}; the larger parent moves down. Only the next ancestor can still violate the order.",
                "After swap: " + " ".join(f"{name}:{value}" for name, value in building) + ".",
                heap_picture(building, doc_id), "heappush: swap upward internally",
            ))
            index = parent
    if building != frames[-1]:
        raise AssertionError("The expanded insertion trace differs from heapq.")
    final_text = " ".join(f"{item_id}:{item_score}" for item_id, item_score in frames[-1])
    steps.append(step(
        "Read the finished array from left to right.",
        f"The array is {final_text}. The root is the smallest score. A parent is no greater than its children, and left-to-right order is not sorted order.",
        f"Final array {final_text}. The parent of index 4 is index 1.",
        heap_picture(frames[-1], "b"),
        "(i - 1) // 2",
        ["Add the next score at the end", "Swap it upward while its parent is larger", "Read the root as the smallest score"],
        2,
    ))
    kept = []
    top_steps = [step(
        "Keep at most three winners.",
        "This second heap wants high scores. Its root is the worst score still kept, because that is the one a better arrival should replace.",
        "No winner has been kept yet.",
        heap_picture([]),
        "if len(kept) > k: heappop(kept)",
        ["Push the arrival", "If the heap holds more than k, pop the worst", "The root is the worst winner"],
        0,
    )]
    for doc_id, score in pairs:
        heapq.heappush(kept, (score, doc_id))
        popped = ""
        if len(kept) > 3:
            popped = heapq.heappop(kept)[1]
        entries = [(item_id, item_score) for item_score, item_id in kept]
        why = f"After {doc_id}:{score}, the kept root is {entries[0][0]}:{entries[0][1]}."
        if popped:
            why += f" {popped} left because it was the worst of four."
        top_steps.append(step(
            f"Consider {doc_id}:{score}.",
            why,
            "Kept: " + " ".join(f"{item_id}:{item_score}" for item_id, item_score in entries) + ".",
            heap_picture(entries, entries[0][0]),
            "heappush / heappop",
            ["Push the arrival", "If the heap holds more than k, pop the worst", "The root is the worst winner"],
            1 if not popped else 2,
        ))
    tie = rows(
        ("policy", cells(("c:5", "keep"), ("a:4", "keep"), ("b:4", "drop"))),
        ("partition", cells(("b:4", "keep"), ("c:5", "keep"), ("a:4", "drop"))),
    )
    return pack("DA12", "Heap picture", [
        "root: index 0, the smallest score in this min-heap",
        "child indexes: 2i+1 and 2i+2",
        "left-to-right array order is not sorted order",
    ], [
        case("shape", "Five scores", "Push a:1, b:5, c:3, d:4, and e:2 into a min-heap.", steps),
        case("topk", "Best three", "Keep the three highest scores. The root is the worst of those three.", top_steps),
        case("tie", "A tie at the cutoff", "a and b both score 4. The policy keeps the smaller ID. One partition does not.", [
            step(
                "Two tools can answer different tie questions.",
                "The policy ranks a higher score first and, for equal scores, the smaller ID first. a and b are tied at 4, so a stays and b is the one that can be dropped.",
                "Scores c:5, a:4, b:4. The policy's cutoff is between a and b.",
                rows(("scores", cells(("c:5", "keep"), ("a:4", ""), ("b:4", "")))),
                "smaller ID wins a tie",
                predict="Which ID does the stated policy drop?",
            ),
            step(
                "The policy result and this partition result are different sets.",
                "The bounded heap returns c a. This NumPy argpartition returns the set b c. Sorting b and c cannot restore a, because a was never selected.",
                "Policy keeps c and a. This partition keeps b and c. a is absent from the partition result.",
                tie,
                "argpartition does not read the ID policy",
            ),
        ]),
    ])


ARGUMENT_PLACES = {
    "claim": (80, 28), "evidence": (36, 100), "warrant": (150, 100),
    "detail": (36, 176), "note": (214, 100),
}
ARGUMENT_EDGES = [("claim", "evidence"), ("claim", "warrant"), ("evidence", "detail")]
SECOND_PLACES = {"claim": (90, 28), "evidence": (40, 104), "note": (170, 104), "detail": (40, 180)}
SECOND_EDGES = [("claim", "evidence"), ("claim", "note"), ("evidence", "detail")]


def bfs_steps(graph, places, edges, start, identifier, label, prompt, isolated):
    seen = {start}
    queue = deque([start])
    steps = []
    produced = []
    processed = []
    while queue:
        snapshot = list(queue)
        visited_now = sorted(seen)
        joined = " ".join(snapshot)
        if not steps:
            why = f"{start} is queued and already marked. No neighbor has been enqueued."
            if isolated:
                why += f" {isolated} is drawn because the graph contains it."
            predict = "Which nodes are enqueued after the front node is taken?"
        else:
            why = "Marked when the previous front was taken: " + (", ".join(produced) if produced else "no new node") + "."
            predict = ""
        steps.append(step(
            f"Queue: {joined}.",
            why,
            f"Queue {joined}. Visited {', '.join(visited_now)}.",
            graph_picture(places, edges, snapshot, visited_now, [], snapshot[0]),
            "seen.add(nxt) before queue.append(nxt)",
            ["Mark a node when it is enqueued", "Take the front of the queue", "Enqueue each unmarked neighbor"],
            0 if not steps else 1,
            predict,
        ))
        node = queue.popleft()
        processed.append(node)
        produced = []
        for nxt in graph[node]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
                produced.append(nxt)
        steps.append(step(
            f"Process {node}; add its new neighbours.",
            ("Add " + ", ".join(produced) + " to the back of the queue. " if produced else "No new neighbour is added. ")
            + "Processed so far: " + ", ".join(processed) + ". The next node comes from the front, not from the most recently added position.",
            f"After processing {node}, queue is {list(queue)}. Discovered {', '.join(sorted(seen))}. Processed {', '.join(processed)}.",
            graph_picture(places, edges, queue, sorted(seen), [], node),
            "queue.popleft(); mark and append new neighbours",
        ))
    steps.append(step(
        "The queue is empty: stop.",
        "Every discovered node has now been processed. A node with no directed route from the start is absent from the visit order.",
        f"Queue empty. Visit order {processed}. Unreachable nodes: {sorted(set(graph) - seen)}.",
        graph_picture(places, edges, [], sorted(seen), [], ""), "return order",
    ))
    return case(identifier, label, prompt, steps)


def dfs_steps():
    graph = {"claim": ["evidence", "warrant"], "evidence": ["detail"], "warrant": [], "detail": [], "note": []}
    seen, path, steps = set(), [], []
    def walk(node):
        seen.add(node)
        path.append(node)
        steps.append(step(
            f"Enter {node}.",
            f"The active path is {' '.join(path)}. The visited set is {' '.join(sorted(seen))}. Entering a node adds it to both.",
            f"Path {' '.join(path)}. Visited {' '.join(sorted(seen))}.",
            graph_picture(ARGUMENT_PLACES, ARGUMENT_EDGES, [], seen, path, node),
            "walk(node)",
            ["Push the node on the active path", "Visit the first unmarked neighbor", "Pop the path when the node returns"],
            0,
        ))
        for nxt in graph[node]:
            if nxt not in seen:
                walk(nxt)
        path.pop()
        steps.append(step(
            f"Leave {node}.",
            f"The path shrinks to {' '.join(path) if path else 'empty'}. The visited set still contains {node}. Path and visited are different collections.",
            f"Path {' '.join(path) if path else 'empty'}. Visited still {' '.join(sorted(seen))}.",
            graph_picture(ARGUMENT_PLACES, ARGUMENT_EDGES, [], seen, path, ""),
            "return",
            ["Push the node on the active path", "Visit the first unmarked neighbor", "Pop the path when the node returns"],
            2,
        ))
    walk("claim")
    return case("dfs", "Depth-first path", "Follow the first neighbor all the way. The path is not the visited set.", steps)


def scene_da13():
    graph = {"claim": ["evidence", "warrant"], "evidence": ["detail"], "warrant": [], "detail": [], "note": []}
    second = {"claim": ["evidence", "note"], "evidence": ["detail"], "note": [], "detail": []}
    return pack("DA13", "Graph search picture", [
        "frontier: the queue, drawn as the nodes waiting",
        "seen: marked on enqueue, so a node is not queued twice",
        "path: the depth-first call stack, which shrinks on return",
        "an isolated node stays on the drawing when no edge reaches it",
    ], [
        bfs_steps(graph, ARGUMENT_PLACES, ARGUMENT_EDGES, "claim", "bfs", "Breadth-first",
                  "Neighbors of claim are evidence, then warrant. evidence leads to detail. note has no incoming edge.", "note"),
        dfs_steps(),
        bfs_steps(second, SECOND_PLACES, SECOND_EDGES, "claim", "second", "A second neighbor",
                  "claim now lists evidence and then note. evidence still leads to detail.", ""),
    ])


def edit_table(source, target):
    rows = [list(range(len(target) + 1))]
    for i, left in enumerate(source, 1):
        row = [i]
        for j, right in enumerate(target, 1):
            delete = rows[i - 1][j] + 1
            insert = row[j - 1] + 1
            replace = rows[i - 1][j - 1] + (0 if left == right else 1)
            row.append(min(delete, insert, replace))
        rows.append(row)
    return rows


def masked(table, visible):
    return [[value if (i, j) in visible else None for j, value in enumerate(row)] for i, row in enumerate(table)]


def reconstruct(source, target, table):
    i, j = len(source), len(target)
    edits, path = [], [(i, j)]
    while i or j:
        if i and j and table[i][j] == table[i - 1][j - 1] + (0 if source[i - 1] == target[j - 1] else 1):
            if source[i - 1] != target[j - 1]:
                edits.append(f"replace {source[i - 1]} {target[j - 1]}")
            i, j = i - 1, j - 1
        elif i and table[i][j] == table[i - 1][j] + 1:
            edits.append(f"delete {source[i - 1]}")
            i -= 1
        else:
            edits.append(f"insert {target[j - 1]}")
            j -= 1
        path.append((i, j))
    return list(reversed(edits)), path


def edit_case(source, target, identifier, label, prompt, focus=None):
    table = edit_table(source, target)
    row_labels = ["∅"] + [source[:i] for i in range(1, len(source) + 1)]
    col_labels = ["∅"] + [target[:j] for j in range(1, len(target) + 1)]
    base = {(i, 0) for i in range(len(source) + 1)} | {(0, j) for j in range(len(target) + 1)}
    steps = [step(
        "Label the prefixes before filling numbers.",
        f"Rows are prefixes of {source or 'the empty source'}. Columns are prefixes of {target or 'the empty target'}. The corner will be the cost of the whole change.",
        f"Source {source or 'empty'}, target {target or 'empty'}. The axes are labelled and the costs are not filled yet.",
        grid(row_labels, col_labels, masked(table, set())),
        "rows[i][j]",
        ["Fill the empty prefixes", "Read above, left, and diagonal", "Keep the cheapest candidate"],
        0,
        "What is the cost of the empty prefix against the empty prefix?",
    ), step(
        "Fill the base row and base column.",
        "The empty-to-empty cell is 0. Each extra target character costs one insertion. Each extra source character costs one deletion.",
        "First row " + " ".join(map(str, table[0])) + ". First column " + " ".join(str(table[i][0]) for i in range(len(table))) + ".",
        grid(row_labels, col_labels, masked(table, base), [0, 0]),
        "rows[0][j] = j; rows[i][0] = i",
        ["Fill the empty prefixes", "Read above, left, and diagonal", "Keep the cheapest candidate"],
        0,
    )]
    visible = set(base)
    for i in range(1, len(source) + 1):
        for j in range(1, len(target) + 1):
            delete = table[i - 1][j] + 1
            insert = table[i][j - 1] + 1
            same = source[i - 1] == target[j - 1]
            replace = table[i - 1][j - 1] + (0 if same else 1)
            visible.add((i, j))
            steps.append(step(
                f"Fill {source[:i]} → {target[:j]}: cost {table[i][j]}.",
                f"Delete: {table[i - 1][j]} + 1 = {delete}. Insert: {table[i][j - 1]} + 1 = {insert}. "
                f"Diagonal: {table[i - 1][j - 1]} + {0 if same else 1} = {replace}. "
                f"Keep min({delete}, {insert}, {replace}) = {table[i][j]}. All three source cells were filled before this one.",
                f"Cell ({i}, {j}) chooses {table[i][j]} from delete {delete}, insert {insert}, and diagonal {replace}.",
                grid(row_labels, col_labels, masked(table, visible), [i, j], [(i - 1, j), (i, j - 1), (i - 1, j - 1)]),
                "dp[i][j] = min(delete, insert, match_or_replace)",
            ))
    steps.append(step(
        "Read the finished corner.",
        f"The corner is {table[-1][-1]}. A rolling row could keep that number and would lose the route back through the cells.",
        "Completed table corner " + str(table[-1][-1]) + ".",
        grid(row_labels, col_labels, [list(row) for row in table], [len(source), len(target)]),
        "rows[-1][-1]",
        ["Fill the empty prefixes", "Read above, left, and diagonal", "Keep the cheapest candidate"],
        2,
    ))
    edits, path = reconstruct(source, target, table)
    if source or target:
        steps.append(step(
            "Walk backward for the edits.",
            "This walk prefers the diagonal when that cell explains the cost, then a deletion, then an insertion. "
            + ("Recorded edit: " + ", ".join(edits) + "." if edits else "The strings already match, so no edit is recorded."),
            "Walk cells " + " ".join(f"{a},{b}" for a, b in reversed(path)) + ". Edits: " + (", ".join(edits) if edits else "none") + ".",
            grid(row_labels, col_labels, [list(row) for row in table], path[0], path[1:]),
            "prefer diagonal, then delete, then insert",
        ))
    return case(identifier, label, prompt, steps)


def scene_da15():
    return pack("DA15", "Edit-distance picture", [
        "row: a prefix of the source",
        "column: a prefix of the target",
        "above: delete one source character",
        "left: insert one target character",
        "diagonal: keep or replace",
    ], [
        edit_case("cat", "cut", "cut", "cat to cut", "Change cat into cut. Look at the prefixes ca and cu before the general formula.", (2, 2)),
        edit_case("cat", "at", "at", "cat to at", "The first character is the only one that has to go.", None),
        edit_case("", "", "empty", "Empty to empty", "Two empty strings are already the same.", None),
    ])


def scene_da01():
    ids = ["d1", "d2", "d1", "d3"]
    steps = [step(
        "Read the IDs before choosing a method.",
        "The question on this pass is whether any ID repeats. Keeping the first copy is a different contract and a different result.",
        "IDs d1, d2, d1, d3. No comparison has happened.",
        rows(("ids", cells(*[(item, "") for item in ids]))),
        "for i in range(len(ids))",
        predict="Which pair is compared first?",
    )]
    checks = 0
    for i in range(len(ids)):
        for j in range(i):
            checks += 1
            match = ids[i] == ids[j]
            marked = []
            for index, item in enumerate(ids):
                mark = "current" if index == i else "match" if index == j and match else "drop" if index == j else ""
                marked.append((item, mark))
            steps.append(step(
                f"Compare position {i} with position {j}.",
                f"{ids[i]} against {ids[j]} is {'a match, so the scan stops' if match else 'not a match, so that pair is known to differ'}.",
                f"Comparison {checks}: {ids[i]} and {ids[j]}.",
                rows(("ids", cells(*marked))),
                "if ids[i] == ids[j]",
            ))
            if match:
                break
        else:
            continue
        break
    growth = rows(
        ("n = 4", cells(("pairs 6", ""), ("halvings 2", ""))),
        ("n = 8", cells(("pairs 28", "current"), ("halvings 3", ""))),
        ("n = 16", cells(("pairs 120", "current"), ("halvings 4", ""))),
    )
    return pack("DA01", "Scan and growth", [
        "one comparison establishes one pair",
        "pairs: n(n-1)/2 when every ID is distinct",
        "halvings: how many times n can be cut in half down to 1",
    ], [
        case("scan", "Find a repeat", "Does any ID in d1, d2, d1, d3 repeat?", steps),
        case("growth", "How the pair count grows", "All IDs are distinct. The middle number is the pair count, not a time.", [
            step(
                "Compare the three sizes.",
                "From 4 to 8, the pair count goes from 6 to 28, a bit more than four times. The halving count goes from 2 to 3. O(n²) names that pair-count shape and leaves out the 1/2. It is not a number of seconds.",
                "4 has 6 pair checks and 2 halvings. 8 has 28 and 3. 16 has 120 and 4.",
                growth,
                "n * (n - 1) // 2",
                predict="Does doubling n double the pair count?",
            ),
            step(
                "Name the invariant, not a clock.",
                "After a comparison that does not match, that pair is known to differ. The scan may stop at the first match. A hash set replaces the pairs only under the hash assumptions in the next lessons.",
                "The pair counts are 6, 28, and 120. A linear pass would have printed 4, 8, and 16.",
                growth,
                "while n > 1: n //= 2",
            ),
        ]),
    ])


def scene_da02():
    row = ["d1", "d2", "d3", "d2"]
    snapshots, copied = [], 0
    frames = []
    for item in row:
        snapshots.append((snapshots[-1] + [item]) if snapshots else [item])
        copied += len(snapshots[-1])
        frames.append((list(snapshots), copied))
    copy_steps = [step(
        "Append without taking a snapshot.",
        "The first loop only stores the four IDs in one list. It does not copy that list on each append. The 10 does not come from this loop.",
        "One list: d1 d2 d3 d2. No snapshot list has been built.",
        rows(("grown", cells(*[(item, "") for item in row]))),
        "grown.append(item)",
    )]
    for built, total in frames:
        shown = rows(*[(f"copy {index + 1}", cells(*[(item, "current") for item in snap])) for index, snap in enumerate(built)])
        copy_steps.append(step(
            "Build the next snapshot by copying the previous cells.",
            f"This list holds {len(built[-1])} cells. The running total of cells written into snapshots is {total}.",
            "Snapshots " + " | ".join(" ".join(snap) for snap in built) + f". Copied cells {total}.",
            shown,
            "snapshots.append(previous + [item])",
        ))
    return pack("DA02", "Copies and capacity", [
        "append grows one list",
        "a snapshot is a new list containing the cells so far",
        "doubling is a teaching model, not CPython's allocator",
    ], [
        case("snapshots", "Real snapshot copies", "Four appends, then a second loop that actually copies each prefix.", copy_steps),
        case("capacity", "Doubling model", "Capacity doubles when it is full. This is a model of growth, not a reading of the allocator.", [
            step("Four arrivals fit in capacity 4.", "The model copies 3 occupied cells while growing to 4. The fourth arrival finds a free slot.", "n=4 ends at capacity 4 with 3 copied cells.",
                 rows(("slots", cells(("d1", "keep"), ("d2", "keep"), ("d3", "keep"), ("d4", "keep")))), "capacity, copies = 4, 3"),
            step("The fifth arrival doubles the model to 8.", "The 4 occupied cells are copied, so the growth copies total 7. Three slots stay empty.", "n=5 ends at capacity 8 with 7 copied cells.",
                 rows(("slots", cells(("1", "keep"), ("2", "keep"), ("3", "keep"), ("4", "keep"), ("5", "current"), ("", ""), ("", ""), ("", "")))), "capacity, copies = 8, 7"),
        ]),
    ])


def scene_da03():
    return pack("DA03", "Toy buckets", ["bucket = sum of character codes modulo 4", "this is an illustration, not CPython's layout"], [
        case("buckets", "cat, dog, act", "The toy function sends cat and act to the same bucket.", [
            step("Compute three buckets.", "cat and act both land in bucket 0. dog lands in bucket 2. The shared bucket has to keep both keys.",
                 "Buckets: 0 holds cat and act, 2 holds dog, 1 and 3 are empty.",
                 rows(("0", cells(("cat", "match"), ("act", "match"))), ("1", cells(("empty", ""))), ("2", cells(("dog", "current"))), ("3", cells(("empty", "")))),
                 "sum(ord(ch) for ch in key) % 4", predict="Do cat and act share a bucket?"),
            step("Count characters without claiming a CPython measurement.", "The four IDs contain 8 characters. That is a model of text a hash might read. A cached hash is not re-read, and this number does not prove what CPython did.",
                 "Character total 8. It is a model, not a profiler reading.",
                 rows(("ids", cells(("d1", ""), ("d2", ""), ("d1", "drop"), ("d3", ""))), ("model", cells(("8 characters", "current")))),
                 "sum(len(item) for item in ids)"),
        ]),
    ])


def scene_da04():
    return pack("DA04", "Sorting pictures", ["stable: equal scores keep their earlier order", "a bad last pivot still shrinks the list by one"], [
        case("insertion", "Insertion", "Sort a:4, b:2, c:4, d:1. Equal scores must keep a before c.", [
            step("Start in input order.", "b has to move in front of a because 2 is smaller than 4.", "Order a4 b2 c4 d1.",
                 rows(("order", cells(("a:4", ""), ("b:2", "current"), ("c:4", ""), ("d:1", "")))), "insertion", predict="Does c jump over a?"),
            step("c stays after a.", "The comparison uses <=, so an equal score on the left is already in place. c does not jump over a.", "Order b2 a4 c4 d1.",
                 rows(("order", cells(("b:2", ""), ("a:4", "keep"), ("c:4", "keep"), ("d:1", "")))), "if data[j - 1][1] <= current[1]"),
            step("d walks to the front.", "The finished names are d b a c. Five comparisons is the work on this list, not a duration.", "Order d1 b2 a4 c4.",
                 rows(("order", cells(("d:1", "current"), ("b:2", ""), ("a:4", "keep"), ("c:4", "keep")))), "return names"),
        ]),
        case("merge", "Merge", "The two halves combine with the same stability rule.", [
            step("Merge the left half.", "a then b becomes b a.", "Combine b a.", rows(("left", cells(("b:2", "keep"), ("a:4", "keep")))), "merge"),
            step("Merge the right half, then the two results.", "d c combines with b a. a stays ahead of c because both scores are 4 and the comparison is <=. The slice copies on this trace total 8.",
                 "Combines b a, d c, and d b a c. Slice copies 8.",
                 rows(("halves", cells(("b a", ""), ("d c", ""))), ("result", cells(("d", "keep"), ("b", ""), ("a", "keep"), ("c", "keep")))), "left[i][1] <= right[j][1]"),
        ]),
        case("partition", "Last-element pivot", "A pivot at the end of a sorted list still moves, but only by one slot.", [
            step("Partition 3, 1, 4, 2.", "The pivot 2 lands at index 1. Values before it are no greater, and values after it are greater.",
                 "Result 1, 2, 4, 3. Pivot index 1.", rows(("values", cells(("1", "keep"), ("2", "current"), ("4", ""), ("3", "")))), "pivot = data[-1]"),
            step("Partition 1, 2, 3, 4.", "The pivot 4 lands at index 3. The left piece still has 3 items. The problem shrinks by one. It does not halve.",
                 "Result 1, 2, 3, 4. Pivot index 3. Left piece length 3.",
                 rows(("values", cells(("1", ""), ("2", ""), ("3", ""), ("4", "current"))), ("left", cells(("3 items", "drop")))), "left piece length 3"),
        ]),
    ])


def scene_da06():
    return pack("DA06", "Two pointers", ["the list must be sorted before a large sum can discard the right end"], [
        case("sorted", "Sorted pair", "Find two indexes in [1, 2, 4, 7] that sum to 6.", [
            step("The ends sum to 8.", "8 is too large, so the right index steps inward. That move is justified because the list is sorted.",
                 "Indexes 0 and 3 hold 1 and 7. Sum 8.", rows(("values", cells(("1", "current"), ("2", ""), ("4", ""), ("7", "current")))), "right -= 1",
                 predict="Which end moves when the sum is too small?"),
            step("The new ends sum to 5.", "5 is too small, so the left index steps inward.", "Indexes 0 and 2 hold 1 and 4. Sum 5.",
                 rows(("values", cells(("1", "current"), ("2", ""), ("4", "current"), ("7", "drop")))), "left += 1"),
            step("The ends sum to 6.", "Indexes 1 and 2 hold 2 and 4. That is the pair.", "Pair (1, 2).",
                 rows(("values", cells(("1", "drop"), ("2", "keep"), ("4", "keep"), ("7", "drop")))), "return left, right"),
        ]),
        case("unsorted", "The rule without sorted order", "The same movement on [5, 1, 4, 2] is not allowed to claim there is no pair.", [
            step("The unsorted ends do not support the rule.", "The procedure returns no pair. The brute-force check is true because 1 + 2 exists. Sorted order was the missing fact.",
                 "Unsorted values 5, 1, 4, 2. The fast result is none. A pair still exists.",
                 rows(("values", cells(("5", ""), ("1", "keep"), ("4", ""), ("2", "keep"))), ("result", cells(("none", "drop"), ("brute True", "keep")))),
                 "return None"),
            step("Name the failed assumption.", "Too large, so discard the right end is true only when every value to the left is no larger. This list does not have that property.",
                 "1 and 2 are a pair. The pointer rule missed them because the list was not sorted.",
                 rows(("missed", cells(("1", "keep"), ("2", "keep")))), "sorted order"),
        ]),
    ])


def scene_da07():
    return pack("DA07", "Prefixes and windows", ["a fixed window adds the cell that enters and subtracts the cell that leaves", "an illegal width returns no window"], [
        case("prefix", "Prefix span", "The sum from index 1 up to but not including 3 is one subtraction of two stored sums.", [
            step("Store every leading sum.", "The prefix is 0, 2, 3, 6, 7. prefix[3] - prefix[1] reads two entries and subtracts once. It is not two subtractions.",
                 "Prefix 0 2 3 6 7. The span of 1 and 3 is 4.",
                 rows(("prefix", cells(("0", ""), ("2", "current"), ("3", ""), ("6", "current"), ("7", "")))), "prefix[3] - prefix[1]"),
            step("Slide a width of 2.", "The windows are 3, 4, and 4. The same update on a negative value still has a defined window because it does not drop a cell forever.",
                 "Windows 3, 4, 4.", rows(("windows", cells(("3", ""), ("4", "current"), ("4", "")))), "total += costs[i] - costs[i - width]"),
        ]),
        case("illegal", "A width that does not fit", "Width 3 cannot cover a list of two costs.", [
            step("Refuse the width.", "The old version summed an empty slice and reported a made-up window. The repaired function returns an empty list when width is 0, negative, or longer than the list.",
                 "fixed_window([1, 2], 3) is []. fixed_window([], 2) is []. fixed_window([1, 2], 0) is [].",
                 rows(("request", cells(("width 3", "drop"), ("length 2", ""))), ("result", cells(("[]", "current")))),
                 "if width <= 0 or width > len(costs): return []", predict="Does the function invent a total?"),
            step("Keep the negative-budget counterexample separate.", "On [5, -3, 1] with budget 4, the one-way window reports 2 and the brute force reports 3. Dropping 5 was unsafe once -3 arrived.",
                 "One-way length 2. Brute-force length 3.",
                 rows(("one-way", cells(("2", "drop"))), ("brute", cells(("3", "keep")))), "nonnegative values"),
        ]),
    ])


def scene_da08():
    return pack("DA08", "Stack, queue, and links", ["a stack returns the latest item", "pop(0) slides the later items", "a singly linked delete needs the predecessor"], [
        case("stack", "Delimiters", "Check (a[b]) and then (].", [
            step("The matching pairs empty the stack.", "( opens, [ opens, ] closes [, ) closes (. The stack ends empty, so the line is balanced.",
                 "Stack empty after (a[b]).", rows(("stack", cells(("empty", "keep")))), "stack.pop()"),
            step("A closer without its opener fails.", "( ] closes a parenthesis with a bracket. The kinds differ, so the line is not balanced.",
                 "( ] is false.", rows(("stack", cells(("(", "drop"), ("]", "drop")))), "return False"),
        ]),
        case("queue", "Two queue ends", "A deque can drop its first item. A list pop(0) moves the survivors.", [
            step("popleft returns the earliest item.", "claim leaves. evidence remains. The later item did not slide through a Python list.",
                 "Returned claim. Remaining evidence.", rows(("deque", cells(("evidence", "keep")))), "popleft"),
            step("pop(0) prints the front and a slide count.", "d1 leaves, and the two later items each move one slot. The printed line is d1 2.",
                 "pop(0) returns d1 and slides 2 items.", rows(("list", cells(("d2", "current"), ("d3", "current"))), ("slide", cells(("2", "drop")))), "pop(0)"),
        ]),
        case("links", "Delete b", "Nodes a, b, c. The only handle in one version is b.", [
            step("The predecessor is what changes.", "a.next must point at c. Holding only b does not give a way to change a. The resulting list is a, c.",
                 "Before: a to b to c. After: a to c. b is the node that required a.",
                 rows(("before", cells(("a", ""), ("b", "drop"), ("c", ""))), ("after", cells(("a", "keep"), ("c", "keep")))), "a.next = b.next",
                 predict="Is a reference to b enough to unlink b?"),
            step("Answer the prediction.", "No. The reference that must change lives in a, the predecessor.",
                 "The changed reference is a.next. The result is a, c.",
                 rows(("changed", cells(("a.next", "current"), ("c", "keep")))), "predecessor"),
        ]),
    ])


def scene_da09():
    frames, factorial_steps = [], []
    def stack_picture():
        return rows(*[(f"frame {i + 1}", cells((f"factorial({n})", "current" if i == len(frames) - 1 else "waiting")))
                      for i, n in enumerate(frames)])
    def walk(n):
        frames.append(n)
        factorial_steps.append(step(
            f"Call factorial({n}).",
            "Create a new frame with its own n. " + ("This is the base case; it returns 1 without another call." if n == 0 else f"This frame waits for factorial({n - 1}), then multiplies the returned value by {n}."),
            f"Unfinished calls, oldest to newest: {frames}. The newest frame is active.",
            stack_picture(), "if n == 0: return 1" if n == 0 else "smaller = factorial(n - 1)",
        ))
        result = 1 if n == 0 else n * walk(n - 1)
        factorial_steps.append(step(
            f"factorial({n}) returns {result}.",
            ("The base case supplies 1." if n == 0 else f"The smaller call returned {result // n}; multiply {n} × {result // n} = {result}.")
            + " This frame now finishes. Its caller resumes; the older frames have not been restarted.",
            f"Frame factorial({n}) returns {result}; active stack before removing it: {frames}.",
            stack_picture(), "return 1" if n == 0 else "return n * smaller",
        ))
        frames.pop()
        return result
    walk(3)
    factorial_steps.append(step(
        "All calls have returned: the answer is 6.",
        "There are no unfinished factorial calls left. The call stack has unwound completely; 6 is the returned answer, not a value still waiting to be multiplied.",
        "Active stack empty. Final answer 6.",
        rows(("active calls", cells(("none", ""))), ("answer", cells(("6", "keep")))), "return to caller",
    ))
    call_nodes, call_edges = [], []
    def branch(n, identifier, depth, left, right):
        x = (left + right) / 2
        call_nodes.append((identifier, str(n), x, 22 + depth * 44))
        if n:
            for suffix, a, b in (("L", left, x), ("R", x, right)):
                child = identifier + suffix
                call_edges.append((identifier, child))
                branch(n - 1, child, depth + 1, a, b)
    branch(3, "root", 0, 0, 320)
    return pack("DA09", "Call stack and call tree", ["the stack is the unfinished path", "the call tree includes every call over time", "this worked factorial uses base n == 0, like the short recipe; the older trace program below stops at 1"], [
        case("factorial", "Factorial stack", "factorial(3) uses the short recipe's base case n == 0. Follow calls down, then values back up.", factorial_steps),
        case("branch", "Two calls, no sharing", "Each non-base call starts two smaller calls. The diagram shows all 15 calls for n = 3.", [
            step("Each call starts two smaller calls.", "count_calls(3) returns 8 leaf results and performs 15 calls. The repeated smaller problems are solved again. Nothing is saved for the next copy.",
                 "Return 8. Calls 15. Base case n == 0.",
                 tree(call_nodes, call_edges, "root"),
                 "counter[0] += 1"),
            step("Keep the two pictures apart.", "The factorial picture is one path of frames. This picture is a tree of repeated calls. Edit distance later saves a repeated subproblem. This function does not.",
                 "8 is the returned leaf total. 15 is the number of calls, including the base cases.",
                 rows(("result", cells(("8 leaves", "keep"))), ("calls", cells(("15", "current")))), "n == 0"),
        ]),
    ])


def scene_da10():
    return pack("DA10", "Choose, explore, undo", ["a prune is safe only when later items cannot reduce the sum"], [
        case("positive", "Nonnegative items", "Items 1, 2, 3 and a limit of 3.", [
            step("Take 1, then 2, then 3.", "The sum is 6, which is over 3, so the branch is pruned. The undo of 3 is what lets 1 2 be recorded next.",
                 "Chosen 1 2 3 is pruned. The next recorded answer is 1 2.",
                 rows(("chosen", cells(("1", "keep"), ("2", "keep"), ("3", "drop")))), "if sum(chosen) > limit",
                 predict="Which line pairs with every choose?"),
            step("Every choose has a later undo.", "The empty answer is the subset that took nothing. A missing pop would carry an old choice into the next branch.",
                 "The trace ends with a bare answer line for the empty subset.",
                 rows(("pair", cells(("choose 3", ""), ("undo 3", "keep"))), ("empty", cells(("answer", "current")))), "chosen.pop()"),
        ]),
        case("negative", "A negative item", "4 and -2 can sum to 2, which is within a limit of 3.", [
            step("Pruning after 4 would be wrong.", "4 is already over 3, but -2 brings the sum to 2. A prune that refuses every partial sum over the limit drops the valid subset 4, -2.",
                 "Partial 4 is over the limit. Extended 4, -2 sums to 2 and is valid.",
                 rows(("partial", cells(("4", "drop"))), ("extended", cells(("4", "keep"), ("-2", "keep"), ("sum 2", "current")))),
                 "later item can reduce the sum", predict="Should the function keep the old prune?"),
            step("The function rejects the input.", "The prune stays in the nonnegative lesson. Negative items raise ValueError instead of printing a search that missed a valid subset.",
                 "subsets([4, -2], 3) is rejected. The missed subset sums to 2.",
                 rows(("result", cells(("rejected", "current"), ("sum 2", "keep")))), "raise ValueError"),
        ]),
    ])


def scene_da11():
    return pack("DA11", "Trees and a trie", ["every key in the left subtree is smaller", "a child check is not the subtree check"], [
        case("shapes", "Two insert orders", "The same keys can make a short tree or a line.", [
            step("Insert 2, then 1, then 3.", "In-order reads 1, 2, 3. The height is 2.",
                 "Balanced-looking tree, height 2.",
                 tree([("2", "2", 160, 28), ("1", "1", 80, 100), ("3", "3", 240, 100)], [("2", "1"), ("2", "3")]),
                 "key < node.key"),
            step("Insert 1, then 2, then 3.", "Each key hangs to the right of the previous one. The height is 3. The search rule did not promise a short tree.",
                 "Line tree, height 3.",
                 tree([("1", "1", 80, 24), ("2", "2", 160, 84), ("3", "3", 240, 144)], [("1", "2"), ("2", "3")]),
                 "height 3"),
        ]),
        case("bad", "A child rule that is too small", "Root 5, left child 1, and that child's right child 6.", [
            step("The parent comparisons pass.", "1 is smaller than 5, and 6 is larger than 1. Checking only the child against its parent accepts this tree.",
                 "5 has left child 1. 1 has right child 6.",
                 tree([("5", "5", 160, 24), ("1", "1", 80, 90), ("6", "6", 150, 150)], [("5", "1"), ("1", "6")], "6"),
                 "child comparison only", predict="Is 6 allowed anywhere in the left subtree of 5?"),
            step("The subtree bound fails.", "6 is in the left subtree of 5 and is larger than 5. Every key in that subtree must be smaller. The bound check returns false.",
                 "ordered(bad) is False. The good tree of 2, 1, 3 returns True.",
                 tree([("5", "5", 160, 24), ("1", "1", 80, 90), ("6", "6", 150, 150)], [("5", "1"), ("1", "6")], "6"),
                 "low < key < high"),
        ]),
        case("trie", "Shared prefix", "cat and car share c and a.", [
            step("Follow ca.", "The walk lists car and then cat because the edges are sorted. do has no edge.",
                 "Prefix ca finds car and cat. Prefix do finds nothing.",
                 tree([("c", "c", 160, 24), ("a", "a", 160, 84), ("r", "r", 90, 144), ("t", "t", 230, 144)], [("c", "a"), ("a", "r"), ("a", "t")]),
                 "words_with(trie, prefix)"),
            step("A whole-word table cannot list the prefix by itself.", "It can answer whether car is stored. Listing every word that starts with ca needs the character edges or another stated index.",
                 "ca: car, cat. do: empty.",
                 rows(("ca", cells(("car", "keep"), ("cat", "keep"))), ("do", cells(("[]", "drop")))), "prefix"),
        ]),
    ])


def scene_da14():
    places = {"A": (40, 40), "B": (140, 40), "C": (140, 130), "D": (250, 80)}
    dependency_edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
    remaining = {node: 0 for node in places}
    outgoing = {node: [] for node in places}
    for source, target in dependency_edges:
        remaining[target] += 1
        outgoing[source].append(target)
    ready = deque(["A"])
    processed = []
    def dependency_picture(active=""):
        pic = graph_picture(places, dependency_edges, ready, processed, [], active)
        pic["visitedLabel"] = "processed"
        pic["nodeNotes"] = {node: f"needs {remaining[node]}" for node in places if node not in processed}
        return pic
    topo_steps = [step(
        "Only A has no prerequisite.",
        "A needs 0 incoming prerequisites; B and C each need 1; D needs 2. Queue A. The arrows point from prerequisite to dependant.",
        "Remaining counts A:0, B:1, C:1, D:2. Ready queue [A].", dependency_picture(), "ready = nodes with indegree zero",
    )]
    while ready:
        node = ready.popleft()
        processed.append(node)
        for neighbour in outgoing[node]:
            remaining[neighbour] -= 1
            if remaining[neighbour] == 0:
                ready.append(neighbour)
        topo_steps.append(step(
            f"Process {node}; release its outgoing prerequisites.",
            f"Processed order: {', '.join(processed)}. Ready next: {', '.join(ready) or 'none'}. "
            "A dependant is queued only when its remaining count reaches zero.",
            f"Order {processed}. Ready {list(ready)}. Remaining counts {remaining}.",
            dependency_picture(node), "indegree[neighbour] -= 1; enqueue if zero",
        ))
    topo_steps.append(step(
        "All four tasks were processed.",
        "A B C D is one legal order. A C B D would also respect these four edges. If the queue emptied before every node was processed, a directed cycle would block completion.",
        "Order A B C D. All remaining counts zero. A C B D is also legal.",
        dependency_picture(), "len(order) == len(nodes)",
    ))
    weighted = {"A": [("B", 1), ("C", 4)], "B": [("C", 1), ("D", 10)], "C": [("D", 1)], "D": []}
    weighted_edges = [(source, target) for source, neighbours in weighted.items() for target, _ in neighbours]
    distances = {"A": 0}
    frontier = [(0, "A")]
    settled = set()
    def distance_picture(active=""):
        candidates = sorted(frontier)
        pic = graph_picture(places, weighted_edges, [node for _, node in candidates], settled, [], active)
        pic["visitedLabel"] = "settled"
        pic["nodeNotes"] = {node: "cost " + str(distances[node]) if node in distances else "cost ∞" for node in places}
        pic["frontierLabel"] = "Tentative routes, cheapest first"
        pic["frontierEntries"] = [f"{node}:{cost}" for cost, node in candidates]
        for edge in pic["edges"]:
            edge["weight"] = next(weight for target, weight in weighted[edge["src"]] if target == edge["dst"])
        return pic
    distance_steps = [step(
        "Start at A with cost zero.",
        "Only A has a known route. ∞ means no route is known yet, not a large guessed cutoff. Numbers on arrows are edge costs; the heap chooses the smallest tentative total.",
        "A:0, B:infinity, C:infinity, D:infinity. Heap [A:0].", distance_picture(), "distances = {start: 0}",
    )]
    while frontier:
        cost, node = heapq.heappop(frontier)
        if cost != distances[node]:
            distance_steps.append(step(
                f"Skip stale route {node}:{cost}.",
                f"The best known cost for {node} is already {distances[node]}. This older heap entry is no longer useful.",
                f"Skip {node}:{cost}; current best {distances[node]}. Remaining heap {sorted(frontier)}.",
                distance_picture(), "if cost != distances[node]: continue",
            ))
            continue
        settled.add(node)
        distance_steps.append(step(
            f"Settle {node} at cost {cost}.",
            "This is the cheapest current heap entry. With nonnegative edges, no later route can undercut it. Now inspect its outgoing edges.",
            f"Settled {node}:{cost}. Settled nodes {sorted(settled)}. Remaining heap {sorted(frontier)}.",
            distance_picture(node), "cost, node = heapq.heappop(frontier)",
        ))
        for neighbour, weight in weighted[node]:
            candidate = cost + weight
            old = distances.get(neighbour, float("inf"))
            if candidate < old:
                distances[neighbour] = candidate
                heapq.heappush(frontier, (candidate, neighbour))
                distance_steps.append(step(
                    f"Improve {neighbour}: {cost} + {weight} = {candidate}.",
                    f"The route through {node} costs {candidate}, better than {old if old != float('inf') else 'no known route'}. Save it and push a new heap entry. An older entry, if present, stays until it is skipped.",
                    f"Set {neighbour}:{candidate}. Heap {sorted(frontier)}.",
                    distance_picture(node), "candidate = cost + weight; save if smaller",
                ))
    distance_steps.append(step(
        "The cheapest route to D costs 3.",
        "A → B → C → D costs 1 + 1 + 1 = 3. Old candidates C:4 and D:11 were skipped. This trace continues to an empty heap to show those stale entries; the goal-only recipe below may stop as soon as D is settled.",
        "Final distances A:0, B:1, C:2, D:3. Heap empty. Route A B C D costs 3.",
        distance_picture(), "return distances[goal]",
    ))
    return pack("DA14", "Order and distance", [
        "ready: no remaining incoming edge",
        "settled: smallest tentative distance, only while weights are nonnegative",
        "stale: an old heap entry whose distance was already improved",
    ], [
        case("topo", "Dependency order", "Edges A to B, A to C, B to D, and C to D. This graph has no B to C edge.", topo_steps),
        case("distance", "Nonnegative routes", "A to B costs 1, A to C costs 4, B to C costs 1, B to D costs 10, and C to D costs 1.", distance_steps),
        case("negative", "A negative edge", "A to B costs 1, A to C costs 4, and C to B costs -10. This case is outside the assumption.", [
            step("The procedure freezes B too early.", "B is settled at 1. The path A C B costs 4 + (-10) = -6, but a settled node is not reopened.",
                 "Reported distance of B is 1. The true shorter path is -6.",
                 rows(("reported", cells(("B:1", "drop"))), ("true", cells(("B:-6", "keep")))),
                 "do not reopen a settled node", predict="Is the reported 1 the shortest path?"),
            step("The assumption is the thing that broke.", "Nonnegative weights are what make settled mean finished. The huge weight 10**9 is still a legal nonnegative edge. A missing goal returns none instead of walking a fake terminator.",
                 "Weight 10**9 from A to B still produces the path A B. An unreachable goal returns none.",
                 rows(("large", cells(("A B", "keep"))), ("missing", cells(("none", "current")))),
                 "if goal not in dist: return none"),
        ]),
    ])


def knapsack(items, capacity):
    rows = [[0] * (capacity + 1) for _ in range(len(items) + 1)]
    for i, (_name, weight, value) in enumerate(items, 1):
        for cap in range(capacity + 1):
            skip = rows[i - 1][cap]
            take = rows[i - 1][cap - weight] + value if weight <= cap else -1
            rows[i][cap] = max(skip, take)
    return rows


def scene_da16():
    items = [("a", 2, 3), ("b", 2, 4), ("c", 1, 2)]
    table = knapsack(items, 3)
    labels = ["none", "a", "a, b", "a, b, c"]
    columns = ["0", "1", "2", "3"]
    def picture(visible, focus=None, deps=()):
        return grid(labels, columns, masked(table, visible), focus, deps,
                    "items allowed", "capacity", "Cell = best total value using the row's items within the column's capacity.")
    visible = {(0, cap) for cap in range(4)} | {(i, 0) for i in range(4)}
    steps = [step(
        "No items, or no capacity: value zero.",
        "Each row allows one more item. Each column sets the capacity. The blank interior will compare skipping and taking that row's newly allowed item.",
        "Base row and column are zero. Other values have not yet been computed.",
        picture(visible), "dp = zero table",
    )]
    for i, (name, weight, value) in enumerate(items, 1):
        for capacity in range(1, 4):
            skip = table[i - 1][capacity]
            deps = [(i - 1, capacity)]
            why = f"Skip {name}: keep previous-row value {skip}. "
            if weight <= capacity:
                take = value + table[i - 1][capacity - weight]
                deps.append((i - 1, capacity - weight))
                why += f"Take {name}: its value {value} + previous-row value {table[i - 1][capacity - weight]} at remaining capacity {capacity - weight} = {take}. Keep the larger value, {table[i][capacity]}."
            else:
                why += f"It weighs {weight}, more than capacity {capacity}, so taking it is impossible. Store {skip}."
            visible.add((i, capacity))
            steps.append(step(
                f"Allow {name}, capacity {capacity}: best value {table[i][capacity]}.", why,
                f"Cell ({i}, {capacity}) = {table[i][capacity]}. " + why,
                picture(visible, [i, capacity], deps), "max(skip, value + previous_row[remaining])",
            ))
    steps.append(step(
        "Read the final value: 6.",
        "Items b and c weigh 2 + 1 = 3 and are worth 4 + 2 = 6. Each take uses the previous row, so an item cannot be selected again within its own row.",
        "Final value 6; chosen b and c. The cell depends on previous-row capacity 2 plus c's value 2.",
        picture(visible, [3, 3], [(2, 3), (2, 2)]), "return dp[-1][capacity]",
    ))
    return pack("DA16", "Zero-one knapsack", ["each item is taken once or skipped", "weights and capacity are nonnegative integers", "the objective is total value"], [
        case("table", "Capacity 3", "a weighs 2 and is worth 3. b weighs 2 and is worth 4. c weighs 1 and is worth 2.", steps),
        case("small", "Capacity 2", "The same items with only 2 units of capacity.", [
            step("b is the whole selection.", "a and c together weigh 3, so they do not fit in 2. The value is 4.",
                 "Capacity 2 selects b and value 4.", rows(("chosen", cells(("b", "keep"), ("value 4", "current")))), "capacity 2"),
            step("Empty inputs stay defined.", "No items, or a capacity of 0, produce 0.", "Both empty cases return 0.",
                 rows(("empty", cells(("0", "keep"), ("0", "keep")))), "return 0"),
        ]),
    ])


def scene_da17():
    return pack("DA17", "Earliest finish", ["the objective is the maximum number of non-overlapping intervals", "intervals are half-open"], [
        case("count", "Two rules", "a is [0, 3), b is [1, 2), and c is [2, 4).", [
            step("Earliest finish takes b and then c.", "The clock starts unset, takes b, and moves to 2. c starts at 2, so it fits. The result has two intervals.",
                 "Chosen b and c.",
                 {"kind": "intervals", "min": 0, "max": 4, "rows": [
                     {"name": "a", "start": 0, "end": 3, "state": "rejected"},
                     {"name": "b", "start": 1, "end": 2, "state": "chosen"},
                     {"name": "c", "start": 2, "end": 4, "state": "chosen"},
                 ]},
                 "sort by end", predict="What does earliest start keep?"),
            step("Earliest start keeps only a.", "a ends at 3, and c starts at 2, so c is already behind the clock. The count is 1. The exchange argument belongs to the count objective, not to duration or value.",
                 "Earliest start returns a. The best count on this input is 2.",
                 {"kind": "intervals", "min": 0, "max": 4, "rows": [
                     {"name": "a", "start": 0, "end": 3, "state": "chosen"},
                     {"name": "b", "start": 1, "end": 2, "state": "rejected"},
                     {"name": "c", "start": 2, "end": 4, "state": "rejected"},
                 ]},
                 "sort by start"),
        ]),
        case("negative", "A negative start", "z occupies [-3, -2).", [
            step("The first interval is accepted.", "The clock starts as None, so a negative start is still a valid first choice. A clock that started at -1 would have rejected z.",
                 "Chosen z.",
                 {"kind": "intervals", "min": -3, "max": 0, "rows": [{"name": "z", "start": -3, "end": -2, "state": "chosen"}]},
                 "now is None or start >= now"),
            step("State the objective beside the result.", "The result counts intervals. It does not add their lengths, and it does not rank them by a value.",
                 "The printed result is z. The objective is the count.",
                 rows(("objective", cells(("count", "keep"), ("duration", "drop")))), "maximum count"),
        ]),
    ])


def scene_da18():
    return pack("DA18", "A bounded stream", ["the caller yields one chunk at a time", "after a push the heap can briefly hold k+1"], [
        case("stream", "Top two", "Chunks (a:1, b:5) and (c:3, d:4). Keep two.", [
            step("The first chunk arrives alone.", "The generator has not produced the second chunk. The heap holds a and b.",
                 "Seen 2. Heap a:1 b:5. The second chunk is not stored by the caller.",
                 rows(("chunk", cells(("a:1", ""), ("b:5", "keep"))), ("heap", cells(("a:1", ""), ("b:5", "current")))),
                 "for chunk in chunks", predict="How large can the heap get before the pop?"),
            step("A push can make the heap one larger than the bound.", "d is pushed while b and c are still there, so the peak is 3. The pop then leaves b and d. The printed stored size 2 is the size after the update, not the peak.",
                 "Seen 4. Stored 2. Peak 3. Kept b d.",
                 rows(("peak", cells(("b", "keep"), ("c", "drop"), ("d", "current"))), ("after", cells(("b", "keep"), ("d", "keep")))),
                 "peak = max(peak, len(heap))"),
        ]),
    ])


def scene_da19():
    return pack("DA19", "Two implementations of a contract", ["keep the first copy in arrival order", "a sorted set answers a different contract"], [
        case("checks", "The contract checks", "Empty input, a duplicate, a bad ID, and a caller that mutates the result.", [
            step("Four checks, four expected results.", "The empty list stays empty. d1 is kept once. An empty string is rejected. Appending d9 to the result leaves the original list unchanged.",
                 "Results: empty, d1, rejected, original d1 d2.",
                 rows(("checks", cells(("[]", "keep"), ("d1", "keep"), ("rejected", "current"), ("d1 d2", "keep")))),
                 "raise ValueError"),
            step("Show an order bug that a set check would miss.", "keep_first of d2, d1, d2 returns d2 d1. sorted(set(...)) returns d1 d2. The names match as a set and the order does not.",
                 "keep_first returns d2 d1. The sorted set returns d1 d2.",
                 rows(("keep_first", cells(("d2", "keep"), ("d1", "keep"))), ("sorted set", cells(("d1", "drop"), ("d2", "drop")))),
                 "sorted(set(ids))"),
        ]),
    ])


def scene_da20():
    return pack("DA20", "Predict the three results", ["each line has its own contract", "the next step shows the result"], [
        case("ids", "First copies", "Input d1, d2, d1. Keep the first copy of each ID.", [
            step("Predict the ID line.", "The contract is arrival order, not sorted order. The second d1 is a duplicate.",
                 "Input d1 d2 d1. The result is hidden on this step.",
                 rows(("input", cells(("d1", ""), ("d2", ""), ("d1", "drop")))), "keep_first",
                 predict="Which IDs stay, and in which order?"),
            step("The ID line is d1 d2.", "d1 stays because it was first. d2 stays. The later d1 is dropped.",
                 "Result d1 d2.", rows(("result", cells(("d1", "keep"), ("d2", "keep")))), "seen.add(item)"),
        ]),
        case("scores", "Top two with the tie policy", "d2, d5, and d3 all score 5. d4 scores 1. Keep two.", [
            step("Predict which two IDs survive.", "Higher score wins, and an equal score keeps the smaller ID. d4 is the low score.",
                 "Scores d2:5, d5:5, d3:5, d4:1.",
                 rows(("scores", cells(("d2:5", ""), ("d5:5", ""), ("d3:5", ""), ("d4:1", "drop")))), "Rank",
                 predict="Which ID at score 5 is dropped?"),
            step("d5 is the one that drops.", "The two smaller IDs are d2 and d3. Asking for every ID would also print d5 and d4.",
                 "Result d2 d3.", rows(("result", cells(("d2", "keep"), ("d3", "keep"), ("d5", "drop")))), "top_k"),
        ]),
        case("walk", "One graph, then a second neighbor", "claim leads to evidence, and evidence leads to detail.", [
            step("Predict the first walk.", "There is one route, so the queue order follows that route.",
                 "Edges claim to evidence to detail.",
                 graph_picture({"claim": (50, 30), "evidence": (150, 30), "detail": (250, 30)},
                               [("claim", "evidence"), ("evidence", "detail")], ["claim"], ["claim"], [], "claim"),
                 "bfs", predict="What is the order?"),
            step("The walk is claim evidence detail.", "Add note after evidence on claim. The next queue after claim is evidence note, and detail follows evidence. The order becomes claim evidence note detail.",
                 "First walk claim evidence detail. With note: claim evidence note detail.",
                 graph_picture({"claim": (70, 30), "evidence": (40, 110), "note": (160, 110), "detail": (40, 190)},
                               [("claim", "evidence"), ("claim", "note"), ("evidence", "detail")],
                               ["evidence", "note"], ["claim", "evidence", "note"], [], "evidence"),
                 "queue evidence note"),
        ]),
    ])


BUILDERS = {
    "DA01": scene_da01, "DA02": scene_da02, "DA03": scene_da03, "DA04": scene_da04,
    "DA05": scene_da05, "DA06": scene_da06, "DA07": scene_da07, "DA08": scene_da08,
    "DA09": scene_da09, "DA10": scene_da10, "DA11": scene_da11, "DA12": scene_da12,
    "DA13": scene_da13, "DA14": scene_da14, "DA15": scene_da15, "DA16": scene_da16,
    "DA17": scene_da17, "DA18": scene_da18, "DA19": scene_da19, "DA20": scene_da20,
}


def scene_for(identifier):
    try:
        return BUILDERS[identifier]()
    except KeyError:
        raise ValueError("No DSA picture for " + identifier) from None


def check():
    scenes = {identifier: scene_for(identifier) for identifier in BUILDERS}
    if set(scenes) != {f"DA{i:02d}" for i in range(1, 21)}:
        raise AssertionError("Expected one picture for every DSA lesson.")
    search = scenes["DA05"]["cases"][0]
    moves = [item["picture"].get("move") for item in search["steps"] if item["picture"].get("move")]
    if moves != ["hi", "hi", "lo"] or search["steps"][-1]["picture"]["lo"] != 1:
        raise AssertionError(moves)
    if "never finishes" not in scenes["DA05"]["cases"][3]["steps"][1]["why"]:
        raise AssertionError("The stalled update is missing.")
    final = scenes["DA12"]["cases"][0]["steps"][-1]["alt"]
    if "a:1 e:2 c:3 b:5 d:4" not in final:
        raise AssertionError(final)
    queues = [item["picture"]["frontier"] for item in scenes["DA13"]["cases"][0]["steps"] if item["action"].startswith("Queue:")]
    if queues != [["claim"], ["evidence", "warrant"], ["warrant", "detail"], ["detail"]]:
        raise AssertionError(queues)
    second = scenes["DA13"]["cases"][2]["steps"][0]
    if second["picture"]["frontier"] != ["claim"]:
        raise AssertionError(second["picture"]["frontier"])
    left = next(item for item in scenes["DA13"]["cases"][1]["steps"] if item["action"] == "Leave detail.")
    if "detail" not in left["picture"]["visited"] or "detail" in left["picture"]["path"]:
        raise AssertionError("The depth-first path and visited set did not diverge.")
    focus = next(item for item in scenes["DA15"]["cases"][0]["steps"] if item["action"].startswith("Fill ca → cu:"))
    if "delete 2" not in focus["alt"] or "insert 2" not in focus["alt"] or "diagonal 1" not in focus["alt"]:
        raise AssertionError(focus["alt"])
    finished = next(item for item in scenes["DA15"]["cases"][0]["steps"] if item["action"] == "Read the finished corner.")
    if finished["picture"]["cells"][-1][-1] != 1:
        raise AssertionError("Edit corner changed.")
    if scenes["DA16"]["cases"][0]["steps"][-1]["picture"]["cells"][-1][-1] != 6:
        raise AssertionError("Knapsack corner changed.")
    table = edit_table("cat", "cut")
    edits, _path = reconstruct("cat", "cut", table)
    if edits != ["replace a u"] or reconstruct("cat", "at", edit_table("cat", "at"))[0] != ["delete c"]:
        raise AssertionError(edits)
    return scenes
