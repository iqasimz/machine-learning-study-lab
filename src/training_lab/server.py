"""Loopback-only dashboard. Run controls accept presets, never shell commands or paths."""
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import subprocess
import sys
import threading
from urllib.parse import urlsplit
import uuid

from .artifacts import ROOT, RUNS, read_json, run_path, verify, write_json, source_hashes
from .tokenization import ByteBPE
from .experiments import catalog, lesson, make_plan, validate
from .studies import STUDIES, all_studies, create_study, custom_plan, resume_settings
from . import curriculum

WEB = ROOT / "src" / "training_lab" / "web"


class Jobs:
    def __init__(self):
        self.lock, self.process, self.current, self.log = threading.Lock(), None, None, None
        self.kind, self.cancelled = "run", False

    def state(self):
        if self.process is not None and self.process.poll() is not None:
            if self.log:
                self.log.close()
                self.log = None
            path = run_path(self.current,STUDIES).with_suffix(".json") if self.kind=="study" else run_path(self.current)/"manifest.json"
            if self.kind=="model":
                from .model_lab import RUNS as MODEL_RUNS
                path=run_path(self.current,MODEL_RUNS)/"manifest.json"
            if self.kind=="dataset":path=RUNS/"_logs"/"no-dataset-manifest.json"
            if path.exists():
                manifest = read_json(path)
                if manifest["status"] in {"RUNNING","QUEUED"}:
                    manifest.update(status="CANCELLED" if self.cancelled else "FAILED", error="Worker exited before recording completion.")
                    write_json(path, manifest)
            result = {"id": self.current, "kind":self.kind, "running": False, "exit_code": self.process.returncode}
            self.process, self.current = None, None
            return result
        return {"id": self.current, "kind":self.kind, "running": self.process is not None}

    def launch(self,name,arguments,kind="run"):
        log_dir=RUNS/"_logs"
        log_dir.mkdir(parents=True,exist_ok=True)
        self.log=(log_dir/f"{name}.log").open("w",encoding="utf-8")
        try:
            self.process=subprocess.Popen([sys.executable,str(ROOT/"lab.py"),*arguments],cwd=ROOT,
                                          stdout=self.log,stderr=subprocess.STDOUT)
        except BaseException:
            self.log.close();self.log=None
            raise
        self.current,self.kind,self.cancelled=name,kind,False
        return self.state()

    def start(self, preset):
        if preset not in {"smoke", "tiny"}:
            raise ValueError("Only smoke and tiny are allowed.")
        with self.lock:
            if self.state()["running"]:
                raise ValueError("One local training worker at a time; cancel or wait.")
            name = datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
            return self.launch(name,["train","--name",name,"--preset",preset,"--device","auto"])

    def start_study(self,plan):
        from .hardware import choose_device
        for run in plan["runs"]: choose_device(run["settings"]["device"])
        with self.lock:
            if self.state()["running"]:raise ValueError("A worker is active; wait or cancel it first.")
            record=create_study(plan)
            return {**self.launch(record["id"],["study",record["id"]],"study"),"study":record}

    def start_adaptation(self,settings):
        from .adaptation import validate
        from .hardware import choose_device
        settings=validate(settings)
        choose_device(settings["device"])
        with self.lock:
            if self.state()["running"]:raise ValueError("Training is already active. Wait or stop it first.")
            name=datetime.now(timezone.utc).strftime("language-%Y%m%d-%H%M%S-")+uuid.uuid4().hex[:5]
            return self.launch(name,["adapt","--name",name,"--settings",json.dumps(settings)])

    def start_model(self,settings):
        from .model_lab import validate
        from .hardware import choose_device
        settings=validate(settings)
        if settings["area"] == "dl": choose_device(settings["controls"]["device"])
        with self.lock:
            if self.state()["running"]:raise ValueError("One local worker at a time across ML, DL and LLMs. Wait or stop it first.")
            name=datetime.now(timezone.utc).strftime(settings["area"]+"-%Y%m%d-%H%M%S-")+uuid.uuid4().hex[:5]
            return self.launch(name,["model-fit","--name",name,"--settings",json.dumps(settings)],"model")

    def prepare_dataset(self,dataset):
        if not isinstance(dataset,str) or dataset not in {"wine_quality","fashion_mnist"}:raise ValueError("Unknown downloadable dataset.")
        with self.lock:
            if self.state()["running"]:raise ValueError("Wait for the active local worker before preparing data.")
            name="prepare-"+dataset+"-"+uuid.uuid4().hex[:5]
            return self.launch(name,["model-prepare",dataset],"dataset")

    def resume(self,identifier,additional):
        if type(additional) is not int or additional not in {40,80,120,240}:
            raise ValueError("Choose 40, 80, 120 or 240 additional updates.")
        with self.lock:
            if self.state()["running"]:raise ValueError("A worker is active; wait or cancel it first.")
            directory=run_path(identifier)
            verify(directory)
            manifest=read_json(directory/"manifest.json")
            settings=resume_settings(manifest)
            if manifest["completed_steps"]+additional>800:raise ValueError("Resume exceeds the 800-update local ceiling.")
            total=manifest["completed_steps"]+additional
            ceiling=next(s for s in [20,40,80,120,240,400,800] if s>=total)
            validate({**settings,"steps":ceiling})
            return self.launch(identifier,["resume",identifier,"--additional-steps",str(additional)])

    def cancel(self):
        with self.lock:
            state = self.state()
            if state["running"]:
                self.cancelled=True
                self.process.terminate()
            return {**state, "cancellation_requested": state["running"]}

    def close(self):
        self.cancel()
        if self.process:
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.log:
            self.log.close()


def listing():
    result = []
    if RUNS.exists():
        for path in RUNS.iterdir():
            if path.is_dir() and not path.is_symlink() and (path / "manifest.json").is_file():
                try:
                    m = read_json(path / "manifest.json")
                    result.append({k: m.get(k) for k in ("id", "engine", "status", "started_at", "completed_steps", "requested_steps", "model", "learning")})
                except (OSError, ValueError):
                    continue
    return sorted(result, key=lambda r: r["started_at"] or "", reverse=True)


def detail(name):
    directory = run_path(name)
    manifest = read_json(directory / "manifest.json")
    result = {"manifest": manifest}
    try:
        resume_settings(manifest)
        result["can_resume"] = manifest["status"] in {"MEASURED","CANCELLED"} and (directory/"last.pt").is_file()
    except (ValueError,OSError):
        result["can_resume"]=False
    for artifact in ("metrics", "data", "tokenizer", "evaluation", "inspection", "samples", "baseline"):
        if artifact in {"evaluation", "inspection", "samples"} and manifest["status"] != "MEASURED":
            continue  # Do not show a previous checkpoint's results as the current resumed run.
        path = directory / f"{artifact}.json"
        if path.is_file() and not path.is_symlink():
            result[artifact] = read_json(path)
    if manifest["status"] == "MEASURED":
        try:
            verify(directory)
            result["integrity"] = "PASS"
        except (OSError, ValueError) as exc:
            result["integrity"] = "FAIL: " + str(exc)
    else:
        result["integrity"] = "Not sealed while active / incomplete run"
    return result


def make_handler(jobs, port):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, payload, kind="application/json", policy=None):
            body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", policy or "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(body)

        def valid_host(self):
            return self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"}

        def do_GET(self):
            if not self.valid_host():
                return self.reply(403, {"error": "Loopback host only."})
            path = urlsplit(self.path).path
            try:
                if path.startswith("/vendor/katex/"):
                    relative=path.removeprefix("/vendor/katex/")
                    from .assets import VENDOR
                    file=(VENDOR/relative).resolve()
                    if not file.is_relative_to(VENDOR.resolve()) or not file.is_file():return self.reply(404,{"error":"Asset not found."})
                    kinds={".js":"text/javascript",".css":"text/css",".woff2":"font/woff2",".woff":"font/woff",".ttf":"font/ttf"}
                    if file.suffix not in kinds:return self.reply(404,{"error":"Asset not served."})
                    return self.reply(200,file.read_bytes(),kinds[file.suffix])
                if path in {"/", "/app.js", "/readouts.js", "/learning.js", "/visuals.js", "/ml_visuals.js", "/architecture_math.js", "/architecture_lessons.js", "/explorer_lessons.js", "/explorer_ml.js", "/explorer_dl.js", "/explorer_llm.js", "/explorer_figures.js", "/nlp_math.js", "/explorer_nlp.js", "/nlp_figures.js", "/metrics.js", "/architecture_explorer.js", "/models.js", "/course.js", "/style.css"}:
                    file, kind = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
                                  "/learning.js": ("learning.js", "text/javascript"),
                                  "/readouts.js": ("readouts.js", "text/javascript"),
                                  "/visuals.js": ("visuals.js", "text/javascript"),
                                  "/ml_visuals.js": ("ml_visuals.js", "text/javascript"),
                                  "/architecture_math.js": ("architecture_math.js", "text/javascript"),
                                  "/architecture_lessons.js": ("architecture_lessons.js", "text/javascript"),
                                  "/architecture_explorer.js": ("architecture_explorer.js", "text/javascript"),
                                  "/explorer_lessons.js": ("explorer_lessons.js", "text/javascript"),
                                  "/explorer_ml.js": ("explorer_ml.js", "text/javascript"),
                                  "/explorer_dl.js": ("explorer_dl.js", "text/javascript"),
                                  "/explorer_llm.js": ("explorer_llm.js", "text/javascript"),
                                  "/explorer_figures.js": ("explorer_figures.js", "text/javascript"),
                                  "/metrics.js": ("metrics.js", "text/javascript"),
                                  "/nlp_math.js": ("nlp_math.js", "text/javascript"),
                                  "/explorer_nlp.js": ("explorer_nlp.js", "text/javascript"),
                                  "/nlp_figures.js": ("nlp_figures.js", "text/javascript"),
                                  "/models.js": ("models.js", "text/javascript"),
                                  "/course.js": ("course.js", "text/javascript"),
                                  "/style.css": ("style.css", "text/css")}[path]
                    return self.reply(200, (WEB / file).read_bytes(), kind)
                if path == "/api/learning":
                    return self.reply(200, curriculum.catalog())
                if path == "/api/hardware":
                    from .hardware import hardware_info
                    return self.reply(200, hardware_info())
                if path=="/api/course/catalog":
                    from .guided_course import catalog as course_catalog
                    return self.reply(200,course_catalog())
                if path.startswith("/api/course/lesson/"):
                    from .guided_course import lesson as course_lesson
                    return self.reply(200,course_lesson(path.removeprefix("/api/course/lesson/")))
                if path=="/api/models/catalog":
                    from .model_lab import catalog as models_catalog
                    return self.reply(200,models_catalog())
                if path=="/api/models/runs":
                    from .model_lab import listing as models_listing
                    return self.reply(200,models_listing())
                if path.startswith("/api/models/run/"):
                    from .model_lab import detail as model_detail
                    return self.reply(200,model_detail(path.removeprefix("/api/models/run/")))
                if path == "/api/adaptation":
                    from .adaptation import catalog as adaptation_catalog
                    return self.reply(200,adaptation_catalog())
                if path.startswith("/api/learning/document/"):
                    return self.reply(200, curriculum.document(path.removeprefix("/api/learning/document/")))
                if path.startswith("/api/learning/interview/"):
                    return self.reply(200, curriculum.interview(path.removeprefix("/api/learning/interview/")))
                if path == "/api/runs":
                    return self.reply(200, listing())
                if path=="/api/catalog":
                    return self.reply(200,catalog())
                if path=="/api/studies":
                    return self.reply(200,all_studies())
                if path.startswith("/api/history/"):
                    directory=run_path(path.removeprefix("/api/history/"),RUNS/"_history")
                    snapshots=[read_json(p) for p in directory.glob("*.json") if not p.is_symlink()]
                    return self.reply(200,sorted(snapshots,key=lambda s:s["manifest"]["completed_steps"]))
                if path.startswith("/api/lesson/"):
                    return self.reply(200,lesson(path.removeprefix("/api/lesson/")))
                if path == "/api/job":
                    with jobs.lock:
                        return self.reply(200, jobs.state())
                if path.startswith("/api/run/"):
                    return self.reply(200, detail(path.removeprefix("/api/run/")))
                return self.reply(404, {"error": "Not found."})
            except (OSError, ValueError) as exc:
                return self.reply(400, {"error": str(exc)})

        def do_POST(self):
            origins = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}
            if not self.valid_host() or self.headers.get("Origin") not in origins:
                return self.reply(403, {"error": "Same-origin local dashboard only."})
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.reply(415, {"error": "JSON required."})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16384:
                    raise ValueError("Request must be 1–16384 bytes.")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError("JSON object required.")
                path = urlsplit(self.path).path
                if path=="/api/models/train" and set(data)=={"settings"}:
                    return self.reply(202,jobs.start_model(data["settings"]))
                if path=="/api/models/prepare" and set(data)=={"dataset"}:
                    return self.reply(202,jobs.prepare_dataset(data["dataset"]))
                if path=="/api/models/compare" and set(data)=={"runs"}:
                    from .model_lab import compare as models_compare
                    return self.reply(200,models_compare(data["runs"]))
                if path in {"/api/preview","/api/guided"} and set(data)=={"guide","variation","overrides","prediction"}:
                    plan=make_plan(data["guide"],data["variation"],data["overrides"],data["prediction"])
                    return self.reply(200,plan) if path=="/api/preview" else self.reply(202,jobs.start_study(plan))
                if path=="/api/workbench" and set(data)=={"settings","prediction"}:
                    return self.reply(202,jobs.start_study(custom_plan(data["settings"],data["prediction"])))
                if path=="/api/adapt" and set(data)=={"settings"}:
                    return self.reply(202,jobs.start_adaptation(data["settings"]))
                if path in {"/api/trace","/api/trace-update"} and set(data)=={"run","text"}:
                    from .traces import capture,capture_update
                    with jobs.lock:
                        if jobs.state()["running"]:raise ValueError("Inspect the model when training finishes.")
                        return self.reply(200,(capture_update if path=="/api/trace-update" else capture)(data["run"],data["text"]))
                if path=="/api/compare" and set(data)=={"runs"}:
                    from .comparison import compare_runs
                    return self.reply(200,compare_runs(data["runs"]))
                if path=="/api/generate":
                    from .interactive import generate_saved
                    with jobs.lock:
                        if jobs.state()["running"]:raise ValueError("Wait for training to finish before generating.")
                        return self.reply(200,generate_saved(data))
                if path=="/api/export" and set(data)=={"run"}:
                    from .interactive import export_run
                    with jobs.lock:
                        if jobs.state()["running"]:raise ValueError("Wait for the worker before exporting.")
                        return self.reply(200,export_run(data["run"]))
                if path=="/api/resume" and set(data)=={"run","additional_steps"}:
                    return self.reply(202,jobs.resume(data["run"],data["additional_steps"]))
                if path=="/api/reflection" and set(data)=={"study","reflection"}:
                    if not isinstance(data["reflection"],str) or len(data["reflection"])>4000:
                        raise ValueError("Reflection must be text of at most 4000 characters.")
                    file=run_path(data["study"],STUDIES).with_suffix(".json")
                    record=read_json(file)
                    if record["status"] in {"QUEUED","RUNNING"}:raise ValueError("Wait for the study to finish before saving a reflection.")
                    record["reflection"]=data["reflection"]
                    write_json(file,record)
                    return self.reply(200,{"saved":True})
                if path == "/api/train" and set(data) == {"preset"}:
                    return self.reply(202, jobs.start(data["preset"]))
                if path == "/api/cancel" and not data:
                    return self.reply(200, jobs.cancel())
                if path == "/api/tokenize" and set(data) == {"run", "text"}:
                    if not isinstance(data["text"], str) or len(data["text"]) > 1000:
                        raise ValueError("Text must be at most 1000 characters.")
                    directory = run_path(data["run"])
                    if read_json(directory/"manifest.json").get("engine")=="lora":
                        from transformers import AutoTokenizer
                        from .assets import MODEL_DIR
                        tokenizer=AutoTokenizer.from_pretrained(MODEL_DIR,local_files_only=True,trust_remote_code=False)
                        ids=tokenizer.encode(data["text"],add_special_tokens=False)
                        return self.reply(200,{"ids":ids,"pieces":[tokenizer.decode([i]) for i in ids],"round_trip":tokenizer.decode(ids)})
                    tokenizer = ByteBPE.from_dict(read_json(directory / "tokenizer.json"))
                    ids = tokenizer.encode(data["text"])
                    return self.reply(200, {"ids": ids, "pieces": [tokenizer.label(i) for i in ids],
                                            "round_trip": tokenizer.decode(ids)})
                return self.reply(400, {"error": "Unknown operation or unsupported fields."})
            except (OSError, ValueError, TypeError, RuntimeError) as exc:
                return self.reply(400, {"error": str(exc)})

        def log_message(self, format, *args):
            pass
    return Handler


def serve(port=8765):
    if not 1024 <= port <= 65535:
        raise ValueError("Choose a non-privileged port in [1024, 65535].")
    jobs = Jobs()
    # Do not mutate a RUNNING manifest from an independently launched CLI worker.
    # Only this server's owned child can be cancelled; after restart ownership is unknown.
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(jobs, port))
    print(f"Machine Learning Study Lab: http://127.0.0.1:{port} (local only; Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        jobs.close()
        server.server_close()
