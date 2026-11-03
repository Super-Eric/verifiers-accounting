"""OpenAI-compatible stub policies for reference baselines, served to `vf-eval`.

  oracle - answers with the programmatic ground truth; every rollout should score 1.0
           (smoke test of the wiring vf-eval -> harness -> rewards).
  naive  - files the ledger as recorded: sums every sale line and the recorded tax, accrues
           no use tax, flags no faults (an un-audited bookkeeper's return). On gpu-rightsizing-v1
           it keeps every pod at its full request (today's over-reserved fleet): nothing reclaimed. On weight-space-v1
           it guesses "majority" and lists every unit (read), makes no edit (fix) and writes an all-zero net (write).

The environment is detected from the prompt, so one server covers both environments.

Run: python scripts/oracle_server.py <port> [oracle|naive]
"""

import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from decimal import Decimal

from gpu_rightsizing_v1.fleet import fleet_from_prompt, solve
from weight_space_v1.nets import FUNCTIONS, Net, functional_units, reference_net
from weight_space_v1.taskset import WeightSpaceConfig, WeightSpaceTaskset
from us_sut_reconciliation_v1.ledger import EXEMPT_CATEGORIES, Ledger, Row, reconcile

POLICY = sys.argv[2] if len(sys.argv) > 2 else "oracle"


def naive(ledger: Ledger) -> dict:
    sales = [r for r in ledger.rows if r.type == "sale"]
    gross = sum((Decimal(r.gross_amount) for r in sales), Decimal("0"))
    exempt = sum((Decimal(r.gross_amount) for r in sales
                  if r.category in EXEMPT_CATEGORIES or r.tax_exempt_claimed), Decimal("0"))
    tax = sum((Decimal(r.tax_collected_or_paid) for r in sales), Decimal("0"))
    return {"gross_sales": f"{gross:.2f}", "exempt_sales": f"{exempt:.2f}", "taxable_sales": f"{gross - exempt:.2f}",
            "sales_tax_due": f"{tax:.2f}", "use_tax_due": "0.00", "net_remittance": f"{tax:.2f}", "faults": {}}


def ledger_from_prompt(text: str) -> Ledger:
    year, quarter = map(int, re.search(r"Filing period: (\d{4}) Q(\d)", text).groups())
    business_zip = re.search(r"business location .*?ZIP (\d{5})", text).group(1)
    csv = text.split("Ledger (CSV):\n", 1)[1].strip().splitlines()
    cols = csv[0].split(",")
    rows = []
    for line in csv[1:]:
        values = dict(zip(cols, line.split(",")))
        values["tax_exempt_claimed"] = values["tax_exempt_claimed"] == "true"
        rows.append(Row(**values))
    return Ledger(seed=0, year=year, quarter=quarter, business_zip=business_zip, rows=rows)


def gpu_naive(text: str) -> dict:
    fleet = fleet_from_prompt(text)
    reserved = sum((Decimal(p.gpu_request) * p.hours for p in fleet.pods), Decimal("0"))
    return {"allocations": {p.pod_id: p.gpu_request for p in fleet.pods}, "vetoed": [],
            "reserved_gpu_hours": f"{reserved:.2f}", "reclaimed_gpu_hours": "0.00", "reclaim_pct": "0.00",
            "annual_savings_usd": "0.00"}


WSP_TASKS = {}


def wsp(text: str) -> dict:
    """weight-space-v1: read answers come from the shown weights; fix restores the corrupted parameter."""
    if not WSP_TASKS:
        WSP_TASKS.update((t.data.prompt, t.data) for t in WeightSpaceTaskset(WeightSpaceConfig(num_tasks=300)).load())
    data = WSP_TASKS[text]
    if POLICY == "naive":
        if data.mode == "read":
            return {"function": "majority", "functional_units": list(range(len(data.net["b1"])))}
        if data.mode == "fix":
            return {"edits": []}
        return {"W1": [[0] * 5], "b1": [0], "v": [0], "c": -1}
    if data.mode == "read":
        net = Net.from_json(data.net)
        table = net.truth_table()
        name = next(n for n in FUNCTIONS if reference_net(n).truth_table() == table)
        return {"function": name, "functional_units": functional_units(net)}
    if data.mode == "fix":
        loc = data.expected["corruption"]
        return {"edits": [{**{k: loc[k] for k in ("param", "row", "col") if k in loc}, "value": loc["original"]}]}
    net = reference_net(data.function)
    return {"W1": [[float(w) for w in r] for r in net.W1], "b1": [float(b) for b in net.b1],
            "v": [float(v) for v in net.v], "c": float(net.c)}


def respond(text: str) -> dict:
    if text.startswith(("READ.", "FIX.", "WRITE.")):
        return wsp(text)
    if "Pods (CSV):" in text:
        return gpu_naive(text) if POLICY == "naive" else solve(fleet_from_prompt(text))
    return (naive if POLICY == "naive" else reconcile)(ledger_from_prompt(text))


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        user = next(m for m in body["messages"] if m["role"] == "user")
        content = user["content"] if isinstance(user["content"], str) else user["content"][0]["text"]
        answer = f"<answer>\n{json.dumps(respond(content))}\n</answer>"
        created, model = int(time.time()), body.get("model")
        if not body.get("stream"):
            out = {
                "id": "oracle", "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": answer}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
            return self._send("application/json", json.dumps(out).encode())
        chunk = {"id": "oracle", "object": "chat.completion.chunk", "created": created, "model": model}
        events = [
            {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": answer}, "finish_reason": None}]},
            {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
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
