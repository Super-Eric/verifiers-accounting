"""Summarize arch-kernels-v1: vf-eval run dirs, then the matched amd64 vs arm64 comparison from scripts/arch_compare.py.

Usage: python scripts/summarize_arch.py compare.json <run_dir> [<run_dir> ...] > results.md
"""

import json
import math
import sys
from pathlib import Path
from statistics import mean, median

from arch_kernels_v1.kernels import KERNELS

from summarize import load

A, R = "amd64", "arm64"


def gmean(xs: list[float]) -> float:
    return math.exp(mean(math.log(x) for x in xs)) if xs else float("nan")


def runs_table(dirs: list[str]) -> None:
    print("| run | arch | n | score | passed | correct | compiled | within memory | median speedup (passed) "
          "| output tokens / episode |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    runs = {}
    for d in dirs:
        rows = load(Path(d))
        arch = rows[0]["task"]["arch"]
        model = json.loads(next(iter((Path(d) / "traces.jsonl").open())))["traces"][0]["agent"]["config"]["model"]
        runs[f"{model} · {arch}"] = rows
        m = lambda k: mean(r[k] for r in rows)
        sp = [r["speedup"] for r in rows if r["passed"]]
        print(f"| {model} | {arch} | {len(rows)} | {m('reward'):.3f} | {m('passed'):.3f} | {m('correct'):.3f} "
              f"| {m('compiled'):.3f} | {m('within_memory'):.3f} | {median(sp) if sp else 0:.2f}× "
              f"| {mean(r['out_tokens'] for r in rows):,.0f} |")
    print("\nMean score by kernel:\n")
    print("| kernel | " + " | ".join(runs) + " |")
    print("|---|" + "---|" * len(runs))
    for k in KERNELS:
        cells = []
        for rows in runs.values():
            sel = [r for r in rows if r["task"]["kernel"] == k]
            cells.append(f"{mean(r['reward'] for r in sel):.2f} ({sum(r['passed'] for r in sel):.0f}/{len(sel)})")
        print(f"| {k} | " + " | ".join(cells) + " |")


def compare_tables(path: str) -> None:
    data = json.loads(Path(path).read_text())
    rows = [r for r in data["rows"] if "error" not in r["result"]]
    print("\nMachines (2 vCPUs, 4 GB each):\n")
    for arch, info in data["machines"].items():
        lines = dict(line.split(": ", 1) for line in (info or "").splitlines() if ": " in line)
        print(f"- **{arch}**: {lines.get('Vendor ID', '?')} {lines.get('Model name', '?')}")

    def ref_times(kernel: str, arch: str) -> list[float]:
        """Every benchmark that got that far also timed the reference."""
        return [r["result"]["ref_s"] * 1e3 for r in rows
                if r["kernel"] == kernel and r["arch"] == arch and "ref_s" in r["result"]]

    print("\nSame code on both machines. Time in ms (median over every run); ratio > 1 means arm64 is faster.\n")
    print("| kernel | unit | reference amd64 | reference arm64 | ratio | tuned amd64 | tuned arm64 | ratio "
          "| best model code amd64 | best model code arm64 | ratio |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    agg = {"reference": [], "tuned": [], "best": []}
    for k, kern in KERNELS.items():
        ref = {a: median(ref_times(k, a)) for a in (A, R) if ref_times(k, a)}
        tuned = {a: median(ok) for a in (A, R)
                 if (ok := [r["result"]["sol_s"] * 1e3 for r in rows if r["kernel"] == k and r["arch"] == a
                            and r["source"] == "tuned" and r["result"].get("passed")])}
        best = {}
        for a in (A, R):
            ok = [r["result"]["sol_s"] * 1e3 for r in rows if r["kernel"] == k and r["arch"] == a
                  and r["source"] not in ("naive", "tuned") and r["result"].get("passed")]
            if ok:
                best[a] = min(ok)
        cells = []
        for name, d in (("reference", ref), ("tuned", tuned), ("best", best)):
            if A in d and R in d:
                agg[name].append(d[A] / d[R])
                cells += [f"{d[A]:.1f}", f"{d[R]:.1f}", f"**{d[A] / d[R]:.2f}**"]
            else:
                cells += [f"{d.get(A, math.nan):.1f}", f"{d.get(R, math.nan):.1f}", "-"]
        print(f"| {k} | {kern.unit} | " + " | ".join(cells) + " |")
    print("\nGeometric mean of the amd64 / arm64 time ratio over kernels: "
          + ", ".join(f"{n} {gmean(v):.2f}× ({sum(x > 1 for x in v)}/{len(v)} kernels faster on arm64)"
                      for n, v in agg.items()))

    print("\nMemory, same code (KiB): peak RSS of the whole process at the benchmark size, median over all sources; "
          "extra = growth during one call, for the tuned code.\n")
    print("| kernel | peak amd64 | peak arm64 | arm64 − amd64 | tuned extra amd64 | tuned extra arm64 |")
    print("|---|---|---|---|---|---|")
    diffs = []
    for k in KERNELS:
        pk = {a: median([r["result"]["peak_kb"] for r in rows if r["kernel"] == k and r["arch"] == a
                         and "peak_kb" in r["result"]] or [math.nan]) for a in (A, R)}
        ex = {r["arch"]: r["result"].get("extra_kb") for r in rows if r["kernel"] == k and r["source"] == "tuned"}  # same in every batch
        diffs.append(pk[R] - pk[A])
        print(f"| {k} | {pk[A]:,.0f} | {pk[R]:,.0f} | {pk[R] - pk[A]:+,.0f} | {ex.get(A)} | {ex.get(R)} |")
    print(f"\nMedian peak-RSS difference (arm64 − amd64): {median(diffs):+,.0f} KiB")

    print("\nPortability: model answers built on the architecture they were written for, and on the other one.\n")
    print("| source | written for | n | passed on amd64 | passed on arm64 | compiled on amd64 | compiled on arm64 |")
    print("|---|---|---|---|---|---|---|")
    groups = sorted({(r["source"], r["written_for"]) for r in rows if r["source"] not in ("naive", "tuned")})
    for src, wf in groups:
        sel = {a: [r["result"] for r in rows if r["source"] == src and r["written_for"] == wf and r["arch"] == a]
               for a in (A, R)}
        f = lambda a, key: f"{sum(x.get(key, 0) for x in sel[a])}/{len(sel[a])}"
        print(f"| {src} | {wf} | {len(sel[A])} | {f(A, 'passed')} | {f(R, 'passed')} | {f(A, 'compiled')} "
              f"| {f(R, 'compiled')} |")


if __name__ == "__main__":
    dirs = sys.argv[2:]
    if dirs:
        runs_table(dirs)
    compare_tables(sys.argv[1])
