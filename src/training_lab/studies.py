"""Serial, budget-bounded local experiments. A study records predictions and all outcomes."""
from datetime import datetime, timezone
import uuid

from .artifacts import ROOT, RUNS, read_json, run_path, source_hashes, verify, write_json, sha256
from .experiments import VERSION, estimate, validate

STUDIES = RUNS / "_studies"


def create_study(plan, root=STUDIES):
    if not isinstance(plan,dict) or not 1 <= len(plan.get("runs",[])) <= 3:
        raise ValueError("A study contains one to three serial local runs.")
    for row in plan["runs"]:
        validate(row["settings"])
    identifier = datetime.now(timezone.utc).strftime("study-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    record = {**plan,"id":identifier,"status":"QUEUED","created_at":datetime.now(timezone.utc).isoformat(),
              "source_hashes":source_hashes(),"completed_runs":0,
              "runs":[{**row,"id":f"{identifier}-{i+1}","status":"QUEUED"} for i,row in enumerate(plan["runs"])]}
    write_json(run_path(identifier,root).with_suffix(".json"),record)
    return record


def custom_plan(settings, prediction=""):
    if not isinstance(prediction,str) or len(prediction)>2000:
        raise ValueError("Prediction must be text of at most 2000 characters.")
    settings=validate(settings)
    return {"registry_version":VERSION,"guide":"custom","question":"Custom local configuration",
            "varied_parameter":None,"prediction":prediction,
            "runs":[{"label":"Workbench configuration","settings":settings,"estimate":estimate(settings)}],
            "limits":["A custom run alone is not a controlled comparison; inspect all changed settings."]}


def execute_study(identifier, root=STUDIES, runs_root=RUNS):
    from .training import train
    path=run_path(identifier,root).with_suffix(".json")
    record=read_json(path)
    if record["status"] != "QUEUED":
        raise ValueError("Study has already started; create a new study instead of overwriting it.")
    record["status"]="RUNNING"
    write_json(path,record)
    current=None
    try:
        for current in record["runs"]:
            if record["source_hashes"] != source_hashes():
                raise ValueError("Source changed after study creation; recreate the study with the current version.")
            settings=validate(current["settings"])
            current["status"]="RUNNING"
            write_json(path,record)
            manifest=train(current["id"],steps=settings["steps"],overrides=settings,runs_root=runs_root,
                           learning={"study":identifier,"guide":record["guide"],"label":current["label"],
                                     "question":record["question"],"prediction":record["prediction"],
                                     "varied_parameter":record["varied_parameter"],"registry_version":VERSION})
            current["status"]=manifest["status"]
            record["completed_runs"]+=1
            write_json(path,record)
        record["status"]="MEASURED"
    except BaseException as exc:
        record["status"]="CANCELLED" if isinstance(exc,KeyboardInterrupt) else "FAILED"
        record["error"]=f"{type(exc).__name__}: {exc}"
        if current is not None:
            current["status"]=record["status"]
        for row in record["runs"]:
            if row["status"]=="QUEUED":row["status"]="NOT_RUN"
        raise
    finally:
        record["updated_at"]=datetime.now(timezone.utc).isoformat()
        write_json(path,record)
    return record


def all_studies():
    if not STUDIES.exists():return []
    result=[]
    for path in STUDIES.glob("*.json"):
        if path.is_symlink():continue
        try:result.append(read_json(path))
        except (ValueError,OSError):continue
    return sorted(result,key=lambda r:r["created_at"],reverse=True)


def resume_settings(manifest):
    if manifest.get("schema_version")!=2 or "settings" not in manifest:
        raise ValueError("This older run has no complete editable settings snapshot. Start a new v0.2 run.")
    if manifest["source_hashes"] != source_hashes():
        raise ValueError("Code changed since this run. Start a new run; do not rewrite its source provenance.")
    if manifest["config"]["corpus_sha256"] != sha256(ROOT/"data"/"corpus.jsonl"):
        raise ValueError("The dashboard resumes only the current authored corpus. Use the CLI for custom local data.")
    stored=dict(manifest["settings"])
    stored["steps"]=next((s for s in [20,40,80,120,240,400,800] if s>=stored["steps"]),800)
    return validate(stored)


def resume_run(identifier, additional_steps):
    from .training import train
    if type(additional_steps) is not int or additional_steps not in {40,80,120,240}:
        raise ValueError("Choose 40, 80, 120 or 240 additional updates.")
    directory=run_path(identifier)
    verify(directory)
    manifest=read_json(directory/"manifest.json")
    settings=resume_settings(manifest)
    total=manifest["completed_steps"]+additional_steps
    if total>800:raise ValueError("Local dashboard resume limit is 800 total updates.")
    # Validate a conservative rounded-up budget without changing the saved schedule.
    ceiling=next(s for s in [20,40,80,120,240,400,800] if s>=total)
    validate({**settings,"steps":ceiling})
    # Preserve the previous report before intentionally extending this run in place.
    history=RUNS/"_history"/identifier
    snapshot={"manifest":manifest,"metrics":read_json(directory/"metrics.json"),
              "evaluation":read_json(directory/"evaluation.json") if (directory/"evaluation.json").exists() else None}
    write_json(history/f"before-resume-{manifest['completed_steps']}-{uuid.uuid4().hex[:8]}.json",snapshot)
    return train(identifier,preset=manifest["config"]["preset"],steps=total,overrides=settings,resume=True,learning=manifest.get("learning",{}))
