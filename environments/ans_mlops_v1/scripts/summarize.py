"""Summarize one or more ans-mlops-v1 run dirs: overall, per reward, and per root cause.

Run: python scripts/summarize.py outputs/<run-dir> [outputs/<run-dir> ...]
"""

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

REWARDS = ["future_performance", "no_regression", "root_cause", "trust", "promotion_gate", "efficiency"]
METRICS = ["regret", "used_impostor", "kept_current", "audits_used", "evaluate_calls", "cost_units", "tool_calls"]


def rows(run_dir: Path):
    for line in (run_dir / "traces.jsonl").read_text().splitlines():
        ep = json.loads(line)
        for tr in ep["traces"]:
            exp = tr["task"]["data"]["expected"]
            total = sum(r["score"] * r["weight"] for r in tr["rewards"].values()) if tr["rewards"] else 0.0
            yield {"cause": exp["root_cause"], "impostor": bool(exp["impostor_defect"]), "reward": total,
                   **{k: tr["rewards"].get(k, {}).get("score", 0.0) for k in REWARDS},
                   **{k: tr["metrics"].get(k, float("nan")) for k in METRICS},
                   "ok": tr.get("ok", True)}


def mean(xs):
    xs = list(xs)
    return st.mean(xs) if xs else float("nan")


def main():
    for arg in sys.argv[1:]:
        run = Path(arg)
        model = json.loads((run / "configs" / "eval.json").read_text()).get("model") if (run / "configs" / "eval.json").exists() else run.name
        data = list(rows(run))
        print(f"\n## {run.name}  (n={len(data)})")
        print(f"mean reward {mean(r['reward'] for r in data):.3f}   " +
              "  ".join(f"{k}={mean(r[k] for r in data):.2f}" for k in REWARDS))
        print("  ".join(f"{k}={mean(r[k] for r in data):.3f}" for k in METRICS))
        by = defaultdict(list)
        for r in data:
            by[r["cause"]].append(r)
        print(f"{'root cause':32s} {'n':>3s} {'reward':>7s} {'future':>7s} {'no_reg':>7s} {'cause':>6s} {'regret':>7s} {'kept':>5s}")
        order = ["none", "covariate_shift", "concept_drift", "label_corruption", "pipeline_fault"]
        for cause in sorted(by, key=lambda c: (c.count("+"), order.index(c.split("+")[0]), c)):
            g = by[cause]
            if g:
                print(f"{cause:32s} {len(g):3d} {mean(r['reward'] for r in g):7.3f} {mean(r['future_performance'] for r in g):7.3f} "
                      f"{mean(r['no_regression'] for r in g):7.3f} {mean(r['root_cause'] for r in g):6.2f} "
                      f"{mean(r['regret'] for r in g):7.3f} {mean(r['kept_current'] for r in g):5.2f}")
        imp = [r for r in data if r["impostor"]]
        if imp:
            print(f"impostor present in {len(imp)} rollouts: trust={mean(r['trust'] for r in imp):.2f} "
                  f"used_impostor={mean(r['used_impostor'] for r in imp):.2f}")


if __name__ == "__main__":
    main()
