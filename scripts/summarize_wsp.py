"""Summarize weight-space-v1 vf-eval run dirs (traces.jsonl) into markdown tables, overall, per mode and per function.

Usage: python scripts/summarize_wsp.py <run_dir> [<run_dir> ...] > results.md
"""

import sys
from pathlib import Path
from statistics import mean

from weight_space_v1.nets import FUNCTIONS

from summarize import load, rollout_std

MODES = ("read", "fix", "write")


def main(dirs: list[str]) -> None:
    runs = {Path(d).name: load(Path(d)) for d in dirs}
    print("| run | n | score | score σ† | read | fix | write | solved | parsed | output tokens / episode |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, rows in runs.items():
        by = lambda m: mean(r["reward"] for r in rows if r["task"]["mode"] == m)
        print(f"| {name} | {len(rows)} | {mean(r['reward'] for r in rows):.3f} | {rollout_std(rows, 'reward')} "
              f"| {by('read'):.3f} | {by('fix'):.3f} | {by('write'):.3f} | {mean(r['solved'] for r in rows):.3f} "
              f"| {mean(r['answer_parsed'] for r in rows):.3f} | {mean(r['out_tokens'] for r in rows):,.0f} |")
    print("\n† Std across the per-rollout mean scores; n counts episodes (tasks × rollouts).")

    print("\nPer mode (solved = read: right function and exact unit set; fix / write: right on all 32 inputs):\n")
    print("| run | read: function right | read: unit F1 | read solved | fix: found the bad weight | fix: accuracy "
          "| fix solved | write: accuracy | write solved |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, rows in runs.items():
        m = lambda mode, k: mean(r[k] for r in rows if r["task"]["mode"] == mode)
        print(f"| {name} | {m('read', 'function_correct'):.3f} | {m('read', 'unit_f1'):.3f} | {m('read', 'solved'):.3f} "
              f"| {m('fix', 'found_corruption'):.3f} | {m('fix', 'accuracy_32'):.3f} | {m('fix', 'solved'):.3f} "
              f"| {m('write', 'accuracy_32'):.3f} | {m('write', 'solved'):.3f} |")

    print("\nSolved, by function and mode (episodes solved / episodes):\n")
    print("| function | " + " | ".join(f"{n} {mode}" for n in runs for mode in MODES) + " |")
    print("|---|" + "---|" * (len(runs) * len(MODES)))
    for fn in FUNCTIONS:
        cells = []
        for rows in runs.values():
            for mode in MODES:
                sel = [r for r in rows if r["task"]["function"] == fn and r["task"]["mode"] == mode]
                cells.append(f"{sum(r['solved'] for r in sel):.0f}/{len(sel)}" if sel else "-")
        print(f"| {fn} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
