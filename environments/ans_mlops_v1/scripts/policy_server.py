"""OpenAI-compatible scripted policies for ans-mlops-v1, served to `vf-run`.

  oracle - verifies the legitimate retrainer, retrains on the oracle's best window (or keeps the
           current model), gates the promotion on a shared held-out evaluation, files the true root
           cause. Every rollout should score 1.0; this smoke-tests vf-run -> harness -> MCP tools -> rewards.
  naive  - what the repo's demo pipeline does: the drift alarm fires, so take the first retrainer ANS
           lists (no verification), retrain on the last three weeks, deploy immediately, and call it
           concept drift.

The policy is a fixed action script indexed by how many assistant turns the conversation already has.

Run: python scripts/policy_server.py <port> [oracle|naive]
"""

import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from ans_mlops_v1.world import DATA_FAULTS, build_world, solve

POLICY = sys.argv[2] if len(sys.argv) > 2 else "oracle"
RETRAIN_CAP = "model-retraining"


def script(seed: int, difficulty: str, policy: str) -> list[tuple[str, dict] | str]:
    world, exp = build_world(seed, difficulty), solve(seed, difficulty)
    n = world.labeled_weeks
    if policy == "naive":
        first = next(a.ans_name for a in world.agents if a.capability == RETRAIN_CAP)
        last3 = list(range(n - 3, n))
        return [
            ("ans_discover", {"capability": RETRAIN_CAP}),
            ("get_drift_report", {}),
            ("retrain", {"retrainer": first, "train_weeks": last3}),
            ("deploy", {"candidate_id": "cand-1"}),
            ("submit_incident_report", {"root_cause": "concept_drift", "summary": "Drift alarm; retrained on recent data."}),
            "Retrained on the last three weeks and deployed.",
        ]
    legit = exp["legit_retrainer"]
    steps: list = [
        ("ans_discover", {"capability": RETRAIN_CAP}),
        ("ans_verify", {"ans_name": legit, "capability": RETRAIN_CAP}),
        ("get_drift_report", {}),
    ]
    best = exp["best_train_weeks"]
    if best:
        # Compare on the most recent clean weeks the candidate did not train on (else its own weeks' holdout).
        fault_from = min([wk for c, wk in world.onsets.items() if c in DATA_FAULTS], default=n)
        clean = list(range(min(n, fault_from)))
        eval_weeks = [w for w in clean if w not in best][-2:] or best[-2:]
        steps += [
            ("retrain", {"retrainer": legit, "train_weeks": best}),
            ("evaluate", {"candidate_id": "cand-1", "weeks": eval_weeks}),
            ("evaluate", {"candidate_id": "current", "weeks": eval_weeks}),
            ("deploy", {"candidate_id": "cand-1"}),
        ]
    else:
        steps.append(("deploy", {"candidate_id": "current"}))
    steps += [
        ("submit_incident_report", {"root_cause": exp["root_cause"], "summary": "Oracle reference policy."}),
        "Done.",
    ]
    return steps


def resolve_tool(name: str, tools: list[dict]) -> str:
    """Match a bare tool name to the advertised one (the harness may prefix server names)."""
    names = [t["function"]["name"] for t in tools or []]
    return next((n for n in names if n == name or n.endswith("_" + name) or n.endswith("__" + name)), name)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msgs = body["messages"]
        user = next(m for m in msgs if m["role"] == "user")
        text = user["content"] if isinstance(user["content"], str) else user["content"][0]["text"]
        m = re.search(r"INC-(H-)?(\d+)", text)
        seed, difficulty = int(m.group(2)), "hard" if m.group(1) else "standard"
        steps = script(seed, difficulty, POLICY)
        turn = sum(1 for m in msgs if m["role"] == "assistant")
        step = steps[min(turn, len(steps) - 1)]
        if isinstance(step, str):
            delta, finish = {"role": "assistant", "content": step}, "stop"
        else:
            name, args = step
            delta = {"role": "assistant", "content": None, "tool_calls": [{
                "index": 0, "id": f"call_{turn}", "type": "function",
                "function": {"name": resolve_tool(name, body.get("tools")), "arguments": json.dumps(args)}}]}
            finish = "tool_calls"
        created, model = int(time.time()), body.get("model")
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if not body.get("stream"):
            msg = {k: v for k, v in delta.items()}
            for tc in msg.get("tool_calls", []):
                tc.pop("index", None)
            out = {"id": "policy", "object": "chat.completion", "created": created, "model": model,
                   "choices": [{"index": 0, "finish_reason": finish, "message": msg}], "usage": usage}
            return self._send("application/json", json.dumps(out).encode())
        chunk = {"id": "policy", "object": "chat.completion.chunk", "created": created, "model": model}
        events = [
            {**chunk, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
            {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": usage},
        ]
        sse = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
        self._send("text/event-stream", sse.encode())

    def _send(self, content_type: str, data: bytes):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 8099), Handler).serve_forever()
