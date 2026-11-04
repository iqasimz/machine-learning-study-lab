"""Opt-in real adapter smoke test against a running lab. Creates one saved run.

Run: python tests/check_adapter_live.py http://127.0.0.1:8766
Requires prepared local model/data and an idle server; no downloads or API calls.
"""
import json
import sys
import time
from urllib.request import Request, urlopen


def main(base):
    def request(path, body=None):
        headers = {"Origin": base, "Content-Type": "application/json"}
        req = Request(base + path, headers=headers,
                      data=None if body is None else json.dumps(body).encode())
        with urlopen(req, timeout=90) as response:
            return json.load(response)

    assert not request("/api/job")["running"], "Wait for the active job first."
    job = request("/api/adapt", {"settings": {"steps": 10, "block_size": 128,
                                             "accumulation": 1, "examples": 128}})
    identifier = job["id"]
    print("Started", identifier, flush=True)
    deadline = time.monotonic() + 180
    observations = []
    while time.monotonic() < deadline:
        listing = request("/api/runs")
        if any(r["id"] == identifier for r in listing):
            run = request("/api/run/" + identifier)
            m = run["manifest"]
            observation = (m["status"], m.get("phase"), m["completed_steps"], len(run.get("metrics", [])))
            if not observations or observation != observations[-1]:
                observations.append(observation)
                print(observation, flush=True)
        if not request("/api/job")["running"]:
            break
        time.sleep(.5)
    else:
        request("/api/cancel", {})
        raise TimeoutError("Stopped owned adapter smoke test after 180 seconds.")
    run = request("/api/run/" + identifier)
    assert run["manifest"]["status"] == "MEASURED", run["manifest"]
    assert run["integrity"] == "PASS", run["integrity"]
    assert [r["step"] for r in run["metrics"]] == list(range(11)), "Baseline plus ten measured updates."
    assert any(o[0] == "RUNNING" for o in observations), "No live state observed."
    assert len({o[2] for o in observations if o[0] == "RUNNING"}) > 1, observations
    result = request("/api/generate", {"run": identifier, "prompt": "What is a language model?",
                                        "temperature": 0, "top_k": 20, "tokens": 40, "seed": 17})
    assert len(result["completion"]) > 20, result
    trace = request("/api/trace", {"run": identifier, "text": "The model learns from text."})
    assert trace["kind"] == "measured"
    assert len(trace["attention"]) == len(trace["tokens"]) > 1
    print(json.dumps({"run": identifier, "status": "PASS", "device": run["manifest"]["hardware"],
                      "observed_live_states": len(observations), "metrics": len(run["metrics"]),
                      "completion": result["completion"], "trace_positions": len(trace["tokens"])}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765")
