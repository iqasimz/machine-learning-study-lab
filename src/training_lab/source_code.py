"""Read-only excerpts from a fixed set of trainer symbols; no requested paths."""
import ast
import hashlib
from .artifacts import ROOT


def excerpt(module, symbol):
    if module not in {"model_data", "model_lab", "sequence_data", "sequence_models", "adaptation", "training", "data", "model"}:
        raise ValueError("Unknown teaching source module.")
    file = ROOT / "src/training_lab" / (module + ".py")
    text = file.read_text(encoding="utf-8")
    node = next((n for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == symbol), None)
    if node is None: raise ValueError("Unknown teaching source symbol.")
    first = min([node.lineno] + [n.lineno for n in getattr(node, "decorator_list", [])])
    return {"file":"src/training_lab/" + module + ".py", "symbol":symbol,
            "first_line":first, "last_line":node.end_lineno,
            "sha256":hashlib.sha256(text.encode()).hexdigest(),
            "code":"\n".join(text.splitlines()[first-1:node.end_lineno])}


def group(label, *symbols):
    return {"label":label, "parts":[excerpt(*symbol) for symbol in symbols]}


def models():
    return {
        "data":group("Data and splits", ("model_data", "load")),
        "sequence_data":group("Data, splits and tokenization", ("sequence_data", "load"), ("sequence_data", "encode_text"), ("sequence_data", "sms_rows"), ("sequence_data", "har_rows"), ("sequence_data", "bike_rows")),
        "sequence_model":group("Sequence encoder", ("sequence_models", "SequencePredictor"), ("sequence_models", "TemporalBlock"), ("sequence_models", "EncoderBlock")),
        "vision_transformer":group("Patch encoder and attention", ("sequence_models", "TinyViT"), ("sequence_models", "EncoderBlock")),
        "classical_model":group("Model", ("model_lab", "estimator")),
        "neural_model":group("Model", ("model_lab", "network")),
        "classical_train":group("Train and evaluate", ("model_lab", "_classical"), ("model_lab", "fit")),
        "neural_train":group("Train and evaluate", ("model_lab", "_neural"), ("sequence_data", "neural_inputs"), ("model_lab", "fit")),
        "metrics":group("Metrics", ("model_lab", "scores"), ("model_lab", "_prediction_artifact"), ("model_lab", "interval")),
    }


def language(engine):
    if engine == "lora":
        return {"data":group("Data and labels", ("adaptation", "conversations"), ("adaptation", "encode_example"), ("adaptation", "prepared_data")),
                "model":group("Starting model", ("adaptation", "load_base")),
                "train":group("Train and evaluate", ("adaptation", "train"), ("adaptation", "evaluate"))}
    return {"data":group("Data and targets", ("data", "load_corpus"), ("data", "examples")),
            "model":group("Decoder model", ("model", "ModelConfig"), ("model", "TinyDecoder")),
            "train":group("Train and evaluate", ("training", "train"), ("training", "evaluate"), ("training", "make_optimizer"))}
