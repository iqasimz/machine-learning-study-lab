"""CLI is the primary execution interface; the dashboard is an optional client."""
import argparse
import json
import shutil
import signal
from pathlib import Path

from .artifacts import ROOT, RUNS, read_json, run_path, sha256, verify, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="training-lab")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train offline on the authored corpus")
    train.add_argument("--name", required=True)
    train.add_argument("--preset", choices=["smoke", "tiny"], default="tiny")
    train.add_argument("--steps", type=int)
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--corpus", type=Path)
    study = commands.add_parser("study",help="Execute an already validated local study snapshot")
    study.add_argument("id")
    resume = commands.add_parser("resume",help="Resume a current-version guided/workbench run")
    resume.add_argument("run")
    resume.add_argument("--additional-steps",type=int,choices=[40,80,120,240],default=40)
    commands.add_parser("doctor", help="Inspect environment without downloads")
    commands.add_parser("catalog", help="Print the executable learning/parameter registry")
    prepare=commands.add_parser("prepare",help="Explicit pinned model/data and local math asset downloads")
    prepare.add_argument("--model",action="store_true")
    prepare.add_argument("--math",action="store_true")
    adapt=commands.add_parser("adapt",help="Train local LoRA adapters on the prepared 360M language model")
    adapt.add_argument("--name",required=True)
    adapt.add_argument("--settings",default="{}",help="Validated JSON settings, no arbitrary paths")
    model_fit=commands.add_parser("model-fit",help="Run one validated local ML/DL experiment")
    model_fit.add_argument("--name",required=True)
    model_fit.add_argument("--settings",required=True)
    data_prepare=commands.add_parser("model-prepare",help="Explicitly download an allowlisted public dataset")
    data_prepare.add_argument("dataset",choices=["fashion_mnist","mnist","cifar10","wine_quality","sms_spam","uci_har","bike_sharing"])
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("run")
    generate = commands.add_parser("generate")
    generate.add_argument("run")
    generate.add_argument("--prompt", default="The model")
    generate.add_argument("--tokens", type=int, default=80)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--seed", type=int, default=17)
    export = commands.add_parser("export")
    export.add_argument("run")
    serve = commands.add_parser("serve")
    serve.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if args.command == "model-prepare":
        from .model_data import prepare
        print(json.dumps(prepare(args.dataset)))
    elif args.command == "model-fit":
        def stop_model(signum,frame): raise KeyboardInterrupt("Stopped by user.")
        signal.signal(signal.SIGTERM,stop_model)
        from .model_lab import fit
        result=fit(args.name,json.loads(args.settings))
        print(json.dumps({"id":result["id"],"status":result["status"]}))
    elif args.command == "prepare":
        from .assets import prepare_math,prepare_model,status
        if args.math: print(json.dumps(prepare_math()))
        if args.model: print(json.dumps(prepare_model()))
        if not args.math and not args.model: print(json.dumps(status()))
    elif args.command == "catalog":
        from .experiments import catalog
        print(json.dumps(catalog(),ensure_ascii=False))
    elif args.command == "doctor":
        import platform
        import torch
        from .hardware import hardware_info
        print(json.dumps({"python": platform.python_version(), "torch": str(torch.__version__),
                          "machine": platform.machine(), "mps": torch.backends.mps.is_available(),
                          "cuda": torch.cuda.is_available(), "network_required": False,
                          "root": str(ROOT), "hardware": hardware_info()}, indent=2))
    elif args.command in {"train","study","resume","adapt"}:
        def stop(signum, frame):
            raise KeyboardInterrupt("Cancelled at process signal.")
        signal.signal(signal.SIGTERM, stop)
        if args.command=="adapt":
            from .adaptation import train as run_adapt
            result=run_adapt(args.name,json.loads(args.settings))
            print(json.dumps({"id":result["id"],"status":result["status"],"updates":result["completed_steps"]}))
        elif args.command=="study":
            from .studies import execute_study
            result=execute_study(args.id)
            print(json.dumps({"id":result["id"],"status":result["status"],"completed_runs":result["completed_runs"]},indent=2))
        elif args.command=="resume":
            from .studies import resume_run
            result=resume_run(args.run,args.additional_steps)
            print(json.dumps({"id":result["id"],"status":result["status"]},indent=2))
        else:
            from .training import train as run_train
            result = run_train(args.name, args.preset, args.steps, args.seed, args.device, args.resume, args.corpus)
            print(json.dumps({k: result[k] for k in ("id", "status", "completed_steps", "tokens_processed", "training_seconds")}, indent=2))
    elif args.command == "verify":
        print(json.dumps(verify(run_path(args.run)), indent=2))
    elif args.command == "generate":
        import torch
        from .model import generate as sample
        from .training import load_model
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        model, tokenizer, _ = load_model(run_path(args.run))
        print(sample(model, tokenizer, args.prompt, args.tokens, args.temperature))
    elif args.command == "export":
        from .interactive import export_run
        print(json.dumps(export_run(args.run),indent=2))
    elif args.command == "serve":
        from .server import serve
        serve(args.port)


if __name__ == "__main__":
    main()
