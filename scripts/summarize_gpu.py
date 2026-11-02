"""Summarize gpu-rightsizing-v1 vf-eval run dirs (traces.jsonl) into markdown tables.

Usage: python scripts/summarize_gpu.py <run_dir> [<run_dir> ...] > results.md
"""

import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path
from statistics import mean, median

from gpu_rightsizing_v1.taskset import FLEET_FIELDS, parse_answer, to_cents

from summarize import load, rollout_std


def main(dirs: list[str]) -> None:
    print("| run | n | reward | reward σ† | allocation_accuracy | fleet_correct | veto_audit (F1) | veto P | veto R "
          "| fields_correct | parsed |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    misses: dict[str, Counter] = {}
    money: dict[str, tuple[list[float], float, float]] = {}
    for d in dirs:
        rows = load(Path(d))
        m = lambda k: mean(r.get(k, 0.0) for r in rows)
        print(f"| {Path(d).name} | {len(rows)} | {m('reward'):.3f} | {rollout_std(rows, 'reward')} "
              f"| {m('allocation_accuracy'):.3f} | {m('fleet_correct'):.3f} | {m('veto_audit'):.3f} "
              f"| {m('veto_precision'):.3f} | {m('veto_recall'):.3f} | {m('fields_correct'):.3f} | {m('answer_parsed'):.3f} |")
        c: Counter = Counter()
        for r in rows:
            answer = parse_answer(r["text"]) or {}
            c.update(f for f in FLEET_FIELDS if to_cents(answer.get(f)) != Decimal(r["task"]["expected"][f]))
        misses[Path(d).name] = c
        errors = [r["savings_abs_error_usd"] for r in rows]
        true_total = sum(float(r["task"]["expected"]["annual_savings_usd"]) for r in rows)
        money[Path(d).name] = (errors, true_total, sum(errors))
    print("\n† Std across the per-rollout mean rewards (`-r` > 1 only); n counts episodes (tasks × rollouts).")
    print("\nFleet figures missed (episodes):\n")
    print("| run | " + " | ".join(FLEET_FIELDS) + " |")
    print("|---|" + "---|" * len(FLEET_FIELDS))
    for name, c in misses.items():
        print(f"| {name} | " + " | ".join(str(c[f]) for f in FLEET_FIELDS) + " |")
    print("\nAnnual-savings error (USD per episode; an unparsed answer counts its whole true figure):\n")
    print("| run | exact | median error | mean error | total error / total true savings |")
    print("|---|---|---|---|---|")
    for name, (errors, true_total, err_total) in money.items():
        exact = sum(e == 0 for e in errors)
        print(f"| {name} | {exact}/{len(errors)} | ${median(errors):,.2f} | ${mean(errors):,.2f} "
              f"| ${err_total:,.2f} / ${true_total:,.2f} ({100 * err_total / true_total:.1f}%) |")


if __name__ == "__main__":
    main(sys.argv[1:])
