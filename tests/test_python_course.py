"""Integration is static/read-only; code snippets are not executed by the reader."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from training_lab import curriculum, guided_course, python_course


class PythonCourseTests(unittest.TestCase):
    def test_all_lessons_have_the_authored_code_reader_in_both_modes(self):
        self.assertEqual(len(python_course.rows()), 95)
        examples = []
        handoffs = []
        for identifier, *_ in python_course.rows():
            doc = curriculum.document(identifier)
            self.assertEqual(doc["reader"]["type"], "code")
            self.assertEqual(guided_course.lesson(identifier)["steps"][0]["document"], doc)
            self.assertEqual(doc["sha256"], hashlib.sha256(doc["markdown"].encode()).hexdigest())
            self.assertTrue(doc["reader"]["sources"])
            examples.extend(b["id"] for b in doc["reader"]["blocks"] if b["kind"] == "example")
            if doc["reader"]["lab_handoff"]: handoffs.append(identifier)
        self.assertEqual(len(examples), 208)
        self.assertEqual(len(set(examples)), 208)
        self.assertEqual(set(handoffs), {"SK12", "PT05", "LC03", "LC04"})

    def test_checked_outputs_are_not_replaced_by_generated_training_results(self):
        for identifier in ["SK12", "PT05", "LC03"]:
            doc = curriculum.document(identifier)
            self.assertFalse(any(b["kind"] == "example" for b in doc["reader"]["blocks"]))
            self.assertIn("python walkthrough", doc["markdown"])
        doc = curriculum.document("LC01")
        sample = next(b for b in doc["reader"]["blocks"] if b.get("id") == "LC01-pretrained")
        self.assertEqual(sample["output_kind"], "captured_pretrained_inference")
        self.assertEqual(sample["output"], "2 + 2 = 4")
        self.assertNotIn("lab_handoff", {k:v for k,v in curriculum.document("PT08")["reader"].items() if v})

    def test_plot_assets_are_explicit_and_hash_verified(self):
        self.assertEqual(len(python_course.assets()), 2)
        for row in python_course.assets():
            self.assertTrue(row["alt"])
            self.assertEqual(hashlib.sha256(python_course.asset(Path(row["source"]).name).read_bytes()).hexdigest(),row["sha256"])
        for value in ["../../metadata.json", "metadata.json", "DS08-main-progress.png/../x", ""]:
            with self.assertRaises(ValueError): python_course.asset(value)

    def test_example_edit_invalidates_serving_until_rechecked(self):
        original = python_course.read
        def changed(name):
            text = original(name)
            return text.replace("morning_minutes = 20", "morning_minutes = 99") if name == "01_PYTHON_BASICS.md" else text
        with patch.object(python_course, "read", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "Recheck"): python_course.document("PY01")

    def test_meaningful_checks_and_exact_handoffs(self):
        checks = python_course.checks()
        self.assertEqual(len(checks), 95)
        for identifier, rows in checks.items():
            question = rows[0]
            self.assertEqual(len(question["options"]), 4)
            self.assertEqual(len({x["feedback"] for x in question["options"]}), 4)
            self.assertEqual(question["options"][question["answer"]]["feedback"], question["explanation"])
        for identifier, mode in [("LC03", "scratch"), ("LC04", "lora")]:
            step = next(s for s in guided_course.lesson(identifier)["steps"] if s["kind"] == "lab")
            self.assertEqual(step["lab"]["mode"], mode)
            self.assertNotIn("settings", step["lab"])

    def test_dsa_pictures_use_the_reference_steps_and_repaired_edges(self):
        from training_lab.dsa_scenes import check
        check()
        for identifier, *_ in python_course.rows():
            if not identifier.startswith("DA"):
                continue
            kinds = [block["kind"] for block in python_course.document(identifier)["reader"]["blocks"]]
            self.assertLess(kinds.index("scene"), kinds.index("example"), identifier)

        def run(lesson, example):
            code = next(block["code"] for block in python_course.document(lesson)["reader"]["blocks"] if block.get("id") == example)
            scope = {}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(code, scope)
            return scope

        windows = run("DA07", "DA07-spans")["fixed_window"]
        self.assertEqual(windows([1, 2], 3), [])
        self.assertEqual(windows([], 2), [])
        self.assertEqual(windows([1, 2], 0), [])
        shortest = run("DA14", "DA14-distance")["shortest"]
        _settled, _skipped, path, dist = shortest({"A": [("B", 10**9)], "B": []}, "A", "B")
        self.assertEqual(path, ["A", "B"])
        self.assertEqual(dist["B"], 10**9)
        self.assertIsNone(shortest({"A": [("B", 1)], "B": []}, "A", "Z")[2])
        with self.assertRaises(ValueError):
            run("DA10", "DA10-choices")["subsets"]([4, -2], 3)
        self.assertEqual(run("DA17", "DA17-intervals")["pick"]([("z", -3, -2)], lambda item: (item[2], item[0])), ["z"])
        tree = run("DA11", "DA11-tree")
        self.assertTrue(tree["ordered"](tree["root"]))
        self.assertFalse(tree["ordered"](tree["bad"]))
        self.assertEqual(run("DA05", "DA05-bound")["lower_bound"]([2, 4, 4, 7, 9], 4), 1)
        self.assertEqual(run("DA05", "DA05-bound")["lower_bound"]([], 4), 0)

    def test_dsa_teaching_recipes_are_executable_and_link_to_real_code(self):
        from training_lab.dsa_teaching import LESSONS, check_recipes
        self.assertEqual(set(LESSONS), {f"DA{i:02}" for i in range(1, 21)})
        self.assertEqual(check_recipes(), 51)
        for identifier, teaching in LESSONS.items():
            self.assertEqual(python_course.document(identifier)["reader"]["teaching"], teaching)
            for field in ("definition", "problem", "intuition", "worked", "cost", "trap", "practice", "answer"):
                self.assertGreater(len(teaching[field]), 40, (identifier, field))
            for recipe in teaching["recipes"]:
                for node in recipe["nodes"]:
                    outgoing = [e for e in recipe["edges"] if e["from"] == node["id"]]
                    if node["kind"] == "decision":
                        self.assertEqual(len(outgoing), 2, (identifier, node))
                        self.assertTrue(all(edge["label"] for edge in outgoing))
                self.assertTrue(recipe["examples"])

    def test_short_recipes_against_small_independent_oracles(self):
        import bisect
        import itertools
        from training_lab.dsa_teaching import LESSONS
        def load(identifier, index=0):
            scope = {}
            exec(LESSONS[identifier]["recipes"][index]["code"], scope)
            return scope
        bound = load("DA05")["lower_bound"]
        top_k = load("DA12")["top_k"]
        windows = load("DA07")["window_sums"]
        insertion = load("DA04")["insertion_sort"]
        for size in range(5):
            for values in itertools.product(range(3), repeat=size):
                supplied = list(values)
                self.assertEqual(insertion(supplied), sorted(values))
                self.assertEqual(supplied, list(values))
                ordered = sorted(values)
                for target in range(4):
                    self.assertEqual(bound(ordered, target), bisect.bisect_left(ordered, target))
                for k in range(6):
                    self.assertEqual(top_k(values, k), sorted(values, reverse=True)[:k])
                for width in range(1, 6):
                    expected = [sum(values[i:i + width]) for i in range(len(values) - width + 1)]
                    self.assertEqual(windows(values, width), expected)
        shortest = load("DA14", 1)["shortest_cost"]
        with self.assertRaises(ValueError):
            shortest({"a": [("b", -1)]}, "a", "b")
        self.assertEqual(shortest({"a": [("b", 10), ("c", 2)], "c": [("b", 1)]}, "a", "b"), 3)
        self.assertIsNone(shortest({"a": []}, "a", "b"))

    def test_dsa_tables_are_progressive_and_have_their_own_axes(self):
        for identifier in ("DA15", "DA16"):
            scene = next(block["scene"] for block in python_course.document(identifier)["reader"]["blocks"] if block["kind"] == "scene")
            counts = [sum(value is not None for row in s["picture"]["cells"] for value in row)
                      for s in scene["cases"][0]["steps"]]
            self.assertEqual(counts, sorted(counts))
            self.assertGreater(len(set(counts)), 8)
            first = scene["cases"][0]["steps"][0]["picture"]
            self.assertEqual(first["rowAxis"], "source prefix" if identifier == "DA15" else "items allowed")
            self.assertEqual(first["colAxis"], "target prefix" if identifier == "DA15" else "capacity")

    def test_dependency_and_distance_pictures_do_not_share_different_edges(self):
        from training_lab.dsa_scenes import scene_for
        scene = scene_for("DA14")
        topo = scene["cases"][0]["steps"][0]["picture"]
        self.assertEqual({(e["src"], e["dst"]) for e in topo["edges"]},
                         {("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")})
        distance = scene["cases"][1]["steps"]
        self.assertTrue(all("weight" in edge for edge in distance[0]["picture"]["edges"]))
        self.assertIn("cost 3", distance[-1]["picture"]["nodeNotes"]["D"])
        self.assertTrue(any("Skip stale route C:4" in s["action"] for s in distance))
        self.assertEqual(scene_for("DA14"), scene, "Picture payloads are deterministic")

    def test_static_export_contains_code_checks_and_verified_plot_assets(self):
        script = Path(__file__).resolve().parents[1] / "scripts/export_site.py"
        spec = importlib.util.spec_from_file_location("python_course_static_export", script)
        exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exporter)
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder)
            report = exporter.export(destination)
            self.assertEqual(report["lessons"], 174)
            self.assertEqual(report["interactive_paths"], 6)
            self.assertFalse(report["training_enabled"])
            self.assertEqual(report["user_runs_exported"], 0)
            snapshots = json.loads((destination / "api-data.json").read_text())
            self.assertEqual(snapshots["/api/runs"], [])
            self.assertEqual(snapshots["/api/models/runs"], [])
            for identifier, *_ in python_course.rows():
                doc = snapshots["/api/learning/document/" + identifier]
                unit = snapshots["/api/course/lesson/" + identifier]
                self.assertEqual(doc["reader"]["type"], "code")
                self.assertEqual(unit["steps"][0]["document"], doc)
            for row in python_course.assets():
                data = (destination / row["url"].removeprefix("./")).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), row["sha256"])
            self.assertNotIn("/Users/", (destination / "api-data.json").read_text())


if __name__ == "__main__": unittest.main()
