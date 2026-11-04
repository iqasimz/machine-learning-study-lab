"""Export the canonical courses to a backend-free GitHub Pages preview."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def export(destination):
    from training_lab import curriculum, experiments, guided_course, model_lab, adaptation
    web = ROOT / "src/training_lab/web"
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Export destination must be new or empty; existing files are not overwritten.")
    destination.mkdir(parents=True, exist_ok=True)
    index = curriculum.catalog()
    courses = guided_course.catalog()
    hardware = {"cpu": False, "cuda": False, "mps": False, "selected": None, "probe_errors": {},
                "summary": "Online course preview - training is available in the local installation.",
                "note": "This static website cannot detect or use your GPU. Your local installation performs hardware detection."}
    models = model_lab.catalog()
    models["hardware"] = hardware
    for dataset in models["datasets"]:
        if "ready" in dataset: dataset["ready"] = False
    adapt = adaptation.catalog()
    adapt["assets"].update(ready=False, model_ready=False, data_ready=False)
    # Do not leak the build machine's model configuration or prepared-data status.
    adapt["model_config"] = None
    snapshots = {"/api/learning": index, "/api/course/catalog": courses,
                 "/api/catalog": experiments.catalog(), "/api/models/catalog": models,
                 "/api/adaptation": adapt, "/api/hardware": hardware,
                 "/api/runs": [], "/api/models/runs": [], "/api/studies": [], "/api/job": {"running": False}}
    for row in index["lessons"] + index["resources"]:
        snapshots["/api/learning/document/"+row["id"]] = curriculum.document(row["id"])
    for row in index["questions"]:
        snapshots["/api/learning/interview/"+row["id"]] = curriculum.interview(row["id"])
    for identifier in courses["units"]:
        snapshots["/api/course/lesson/"+identifier] = guided_course.lesson(identifier)
    for identifier in experiments.LESSONS:
        snapshots["/api/lesson/"+identifier] = experiments.lesson(identifier)
    for file in web.rglob("*"):
        if file.is_file() and not file.is_symlink():
            target = destination / file.relative_to(web)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
    html = (destination / "index.html").read_text()
    html = html.replace('src="/', 'src="./').replace('href="/', 'href="./')
    html = html.replace('<script src="./vendor/', '<script src="./preview.js" defer></script><script src="./vendor/', 1)
    html = html.replace('<body>', '<body><div class="preview-notice" role="note">Online course preview. Lessons, diagrams and quizzes work here. Install the repository locally to train models and save experiments.</div>', 1)
    (destination / "index.html").write_text(html)
    (destination / "api-data.json").write_text(json.dumps(snapshots, ensure_ascii=False, allow_nan=False))
    shutil.copy2(ROOT / "scripts/preview.js", destination / "preview.js")
    (destination / ".nojekyll").touch()
    with (destination / "style.css").open("a") as stream:
        stream.write("\n.preview-notice{position:relative;z-index:10;margin-left:270px;padding:10px 24px;background:#edf2f7;color:#22354a;font-size:14px;border-bottom:1px solid #d4dce5}@media(max-width:800px){.preview-notice{margin-left:0}}\n")
    report = {"lessons": len(index["lessons"]), "questions": len(index["questions"]),
              "interactive_paths": len(courses["paths"]), "snapshot_endpoints": len(snapshots),
              "training_enabled": False, "user_runs_exported": 0}
    (destination / "export-info.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "site")
    print(json.dumps(export(parser.parse_args().output), indent=2))
