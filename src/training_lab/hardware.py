"""Local device discovery shared by the neural training engines and dashboard."""
from functools import lru_cache
import os
import platform

import torch


def available_devices():
    return {"cpu": True, "cuda": torch.cuda.is_available(),
            "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())}


@lru_cache(maxsize=3)
def probe(device):
    """Check a tiny forward/backward operation, without consuming random state."""
    try:
        x = torch.ones((2, 2), device=device, requires_grad=True)
        (x @ x).sum().backward()
        if device == "cuda": torch.cuda.synchronize()
        if device == "mps": torch.mps.synchronize()
        return None
    except (RuntimeError, NotImplementedError, AssertionError) as exc:
        return str(exc).splitlines()[0][:250]


def choose_device(requested="auto"):
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("Unknown device; choose auto, cpu, cuda or mps.")
    available = available_devices()
    if requested != "auto":
        if not available[requested]:
            raise ValueError(f"{requested.upper()} is unavailable in this Python installation. Choose auto or cpu.")
        error = probe(requested)
        if error:
            raise ValueError(f"{requested.upper()} failed its basic training check: {error}")
        return requested
    for device in ("cuda", "mps", "cpu"):
        if available[device] and probe(device) is None:
            return device
    raise ValueError("No working local training device was found.")


def hardware_info():
    available = available_devices()
    selected = choose_device("auto")
    memory = None
    try:
        memory = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (OSError, ValueError, AttributeError):
        pass
    gpus = []
    if available["cuda"]:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({"index": i, "name": props.name, "memory_gib": round(props.total_memory / 2**30, 1)})
    failed = {d: probe(d) for d in ("cuda", "mps") if available[d] and probe(d)}
    return {**available, "selected": selected, "system": platform.system(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "ram_gib": memory, "gpus": gpus, "probe_errors": failed,
            "summary": f"Neural training: {selected.upper()} selected automatically. Classical ML: CPU only.",
            "note": "Detection checks installed PyTorch and a tiny training operation, not whether every model fits in memory. No cloud jobs or downloads. Unsupported model operations can still fail; choose CPU if needed. CUDA uses the current/default GPU, not multiple GPUs."}
