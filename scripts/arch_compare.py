"""Run the same C sources on an amd64 and an arm64 Prime sandbox and save every measurement.

Sources: the NAIVE and TUNED baselines, plus every rollout's answer in the given arch-kernels-v1 run dirs (so code
written for one architecture is also built and timed on the other). Each benchmark also times the reference
implementation, so every row carries a same-code arch comparison for free.

Usage: python scripts/arch_compare.py out.json [<run_dir> ...]
"""

import asyncio
import json
import sys
from pathlib import Path

from arch_kernels_v1.bench import Machine
from arch_kernels_v1.kernels import KERNELS
from arch_kernels_v1.solutions import NAIVE, TUNED
from arch_kernels_v1.taskset import parse_answer

from summarize import final_text

ARCHES = ("amd64", "arm64")


def sources(dirs: list[str]) -> list[dict]:
    out = [{"source": name, "written_for": "portable", "kernel": k, "rollout": 0, "code": code}
           for name, table in (("naive", NAIVE), ("tuned", TUNED)) for k, code in table.items()]
    for d in dirs:
        seen: dict[str, int] = {}
        for line in (Path(d) / "traces.jsonl").read_text().splitlines():
            episode = json.loads(line)
            trace, task = episode["traces"][0], episode["task"]["data"]
            code = parse_answer(final_text(trace))
            k = task["kernel"]
            seen[k] = seen.get(k, 0) + 1
            if code:
                out.append({"source": trace["agent"]["config"]["model"], "written_for": task["arch"], "kernel": k,
                            "rollout": seen[k] - 1, "code": code})
    return out


async def main(out_path: str, dirs: list[str]) -> None:
    jobs = sources(dirs)
    machines = {a: Machine(a, labels=["arch-kernels-v1", "arch-compare"]) for a in ARCHES}
    await asyncio.gather(*(m.start() for m in machines.values()))
    info = {a: m.info for a, m in machines.items()}
    print(json.dumps(info, indent=1), flush=True)
    rows: list[dict] = []

    async def one(job: dict, arch: str) -> None:
        try:
            result = await machines[arch].run(job["kernel"], job["code"])
        except Exception as e:  # keep going; the row records the failure
            result = {"error": repr(e)}
        rows.append({**{k: v for k, v in job.items() if k != "code"}, "arch": arch, "result": result})
        r = result
        print(f"{len(rows):4d}/{2 * len(jobs)} {arch} {job['source']:>18s} {job['written_for']:>8s} "
              f"{job['kernel']:20s} passed={r.get('passed')} speedup={r.get('speedup', 0):.2f}", flush=True)
        Path(out_path).write_text(json.dumps({"machines": info, "rows": rows}, indent=1))

    try:
        await asyncio.gather(*(one(j, a) for j in jobs for a in ARCHES))
    finally:
        await asyncio.gather(*(m.close() for m in machines.values()))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2:]))
