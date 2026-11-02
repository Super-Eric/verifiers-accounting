"""Synthetic online-inference GPU fleet snapshots and the right-sizing policy that scores them.

Modelled on "Predictive GPU Right-Sizing for Production Online Inference" (Adhikari & Bhandari, 2026),
which studies the Alibaba ASI GPU trace (OSDI '26). The policy is that paper's A1 history rule with a
per-segment safety margin, rounded up to a discrete slice grid, plus a Model-B-style readiness veto,
then priced at the paper's illustrative on-demand GPU-hour rates. Everything is closed-book: all
tables go in the prompt. Numbers are stylized and are not ASI measurements.
"""

import math
import random
import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

D = Decimal
CENT = D("0.01")

SLICE_GRID = tuple(D(x) for x in ("0.1", "0.2", "0.25", "0.3", "0.4", "0.5", "0.7", "1.0"))
PRICE_PER_GPU_HOUR = {"A10": D("1.00"), "A100": D("3.43"), "H100": D("3.90")}
RISK_SENSITIVITY = {"A10": D("8"), "A100": D("12"), "H100": D("20")}
"""Readiness-risk points added by shrinking a pod to zero (linear in the share of its request removed)."""
COLD_FRACTION = {
    ("rec", "A10"): D("0.60"), ("rec", "A100"): D("0.50"), ("rec", "H100"): D("0.45"),
    ("genai", "A10"): D("0.95"), ("genai", "A100"): D("0.95"), ("genai", "H100"): D("0.95"),
}
"""Cold-start need as a fraction of the request, by model type x GPU type."""
MARGIN = {("rec", "warm"): D("0.10"), ("rec", "cold"): D("0.25"), ("genai", "warm"): D("0.05"), ("genai", "cold"): D("0.15")}
RISK_CEILING = D("50")
HISTORY_DAYS = 7
MIN_HISTORY = 3
WINDOW_HOURS = 730
MONTHS_PER_YEAR = 12


@dataclass
class Workload:
    workload_id: str
    model_type: str
    gpu_type: str
    gpu_request: str


@dataclass
class Pod:
    pod_id: str
    workload_id: str
    model_type: str
    gpu_type: str
    gpu_request: str
    hours: int
    ready_risk_pct: str


@dataclass
class Observation:
    workload_id: str
    day: int
    need_gpu: str


@dataclass
class Fleet:
    seed: int
    snapshot_day: int
    pods: list[Pod] = field(default_factory=list)
    history: list[Observation] = field(default_factory=list)


def _fmt(x: float, places: int) -> str:
    return f"{x:.{places}f}"


def generate(seed: int, num_pods: int = 24) -> Fleet:
    rng = random.Random(seed)
    day = rng.randint(60, 170)
    fleet = Fleet(seed=seed, snapshot_day=day)
    workloads = []
    for w in range(max(3, num_pods // 2)):
        model_type = "rec" if rng.random() < 0.65 else "genai"
        gpu_type = rng.choices(["A10", "A100", "H100"], weights=[5, 3, 2])[0]
        if model_type == "rec":
            request = rng.choices(["0.5", "1", "2"], weights=[15, 70, 15])[0]
            typical = rng.uniform(0.05, 0.75)
        else:
            request = rng.choices(["1", "2", "4"], weights=[50, 30, 20])[0]
            typical = rng.uniform(0.80, 0.99)
        wl = Workload(f"wl_{seed % 1000:03d}{w:02d}", model_type, gpu_type, request)
        workloads.append(wl)
        n_eligible = rng.choices([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], weights=[1, 1, 2, 4, 5, 5, 4, 3, 2, 2])[0]
        days = [day - rng.randint(1, HISTORY_DAYS) for _ in range(n_eligible)]
        # Distractors: too old for the window, or admitted on/after the snapshot day (not finished before admission).
        days += [day - rng.randint(HISTORY_DAYS + 1, HISTORY_DAYS + 10) for _ in range(rng.randint(0, 3))]
        days += [day + rng.randint(0, 2) for _ in range(rng.randint(0, 2))]
        for d in days:
            frac = min(1.0, max(0.01, typical * rng.lognormvariate(0, 0.25)))
            fleet.history.append(Observation(wl.workload_id, d, _fmt(frac * float(request), 3)))
    rng.shuffle(fleet.history)
    for p in range(num_pods):
        if rng.random() < 0.2:
            model_type = "rec" if rng.random() < 0.65 else "genai"
            gpu_type = rng.choice(list(PRICE_PER_GPU_HOUR))
            request = rng.choice(["0.5", "1", "2"] if model_type == "rec" else ["1", "2", "4"])
            wid = ""
        else:
            wl = rng.choice(workloads)
            model_type, gpu_type, request, wid = wl.model_type, wl.gpu_type, wl.gpu_request, wl.workload_id
        fleet.pods.append(Pod(
            pod_id=f"pod_{p + 1:02d}", workload_id=wid, model_type=model_type, gpu_type=gpu_type,
            gpu_request=request, hours=rng.randint(24, WINDOW_HOURS), ready_risk_pct=_fmt(rng.uniform(15, 49.9), 1),
        ))
    return fleet


def eligible_history(fleet: Fleet, workload_id: str) -> list[Decimal]:
    lo, hi = fleet.snapshot_day - HISTORY_DAYS, fleet.snapshot_day - 1
    return [D(o.need_gpu) for o in fleet.history if o.workload_id == workload_id and lo <= o.day <= hi]


def p90(values: list[Decimal]) -> Decimal:
    """Nearest-rank 90th percentile: the ceil(0.9 * n)-th smallest value."""
    ordered = sorted(values)
    return ordered[math.ceil(D("0.9") * len(ordered)) - 1]


def round_to_slice(need: Decimal, request: Decimal) -> Decimal:
    if need >= request:
        return request
    if need <= 1:
        size = next(s for s in SLICE_GRID if s >= need)
    else:
        size = D(math.ceil(need))
    return min(size, request)


def allocate(fleet: Fleet, pod: Pod) -> tuple[Decimal, bool]:
    """(allocated GPUs, vetoed) for one pod."""
    request = D(pod.gpu_request)
    history = eligible_history(fleet, pod.workload_id) if pod.workload_id else []
    if len(history) >= MIN_HISTORY:
        base, state = p90(history), "warm"
    else:
        base, state = COLD_FRACTION[(pod.model_type, pod.gpu_type)] * request, "cold"
    alloc = round_to_slice(base * (1 + MARGIN[(pod.model_type, state)]), request)
    if alloc < request:
        added_risk = RISK_SENSITIVITY[pod.gpu_type] * (request - alloc) / request
        if D(pod.ready_risk_pct) + added_risk > RISK_CEILING:
            return request, True
    return alloc, False


def _num(x: Decimal) -> str:
    return format(x.normalize(), "f")


def solve(fleet: Fleet) -> dict:
    allocations, vetoed = {}, []
    reserved = allocated = savings = D(0)
    for pod in fleet.pods:
        alloc, veto = allocate(fleet, pod)
        allocations[pod.pod_id] = _num(alloc)
        if veto:
            vetoed.append(pod.pod_id)
        request = D(pod.gpu_request)
        reserved += request * pod.hours
        allocated += alloc * pod.hours
        savings += (request - alloc) * pod.hours * PRICE_PER_GPU_HOUR[pod.gpu_type]
    reclaimed = reserved - allocated
    return {
        "allocations": allocations,
        "vetoed": vetoed,
        "reserved_gpu_hours": str(reserved.quantize(CENT)),
        "reclaimed_gpu_hours": str(reclaimed.quantize(CENT)),
        "reclaim_pct": str((reclaimed / reserved * 100).quantize(CENT, ROUND_HALF_UP)),
        "annual_savings_usd": str((savings * MONTHS_PER_YEAR).quantize(CENT, ROUND_HALF_UP)),
    }


def render_rulebook(fleet: Fleet) -> str:
    grid = ", ".join(_num(s) for s in SLICE_GRID)
    cold = "\n".join(f"| {m} | {g} | {_num(f)} |" for (m, g), f in COLD_FRACTION.items())
    margin = "\n".join(f"| {m} | {s} | {_num(v)} |" for (m, s), v in MARGIN.items())
    gpus = "\n".join(f"| {g} | {_num(RISK_SENSITIVITY[g])} | {PRICE_PER_GPU_HOUR[g]} |" for g in PRICE_PER_GPU_HOUR)
    return f"""\
Fleet: online-inference pods on a shared GPU cluster. Snapshot (admission) day: {fleet.snapshot_day}.
Billing window: one month = {WINDOW_HOURS} hours; annualise by multiplying the month by {MONTHS_PER_YEAR}.

Right-sizing policy (apply to every pod, in this order):
1. History. A workload's eligible history is its rows in the history table with
   day in [{fleet.snapshot_day - HISTORY_DAYS}, {fleet.snapshot_day - 1}] (the {HISTORY_DAYS} days before the snapshot day; rows on or
   after the snapshot day were not finished before admission and rows older than {HISTORY_DAYS} days are stale).
   A pod is WARM if it has a workload_id with at least {MIN_HISTORY} eligible rows; otherwise it is COLD
   (including pods with an empty workload_id).
2. Base need (GPUs). WARM: the nearest-rank P90 of the eligible need_gpu values (sort ascending and take
   the ceil(0.9*n)-th value, 1-indexed). COLD: cold_fraction(model_type, gpu_type) x gpu_request.
3. Safety margin. need = base x (1 + margin(model_type, WARM/COLD)). Use exact arithmetic.
4. Slice. If need >= gpu_request, allocate gpu_request. Otherwise, if need <= 1, allocate the smallest
   slice in the grid [{grid}] that is >= need; if need > 1, allocate ceil(need) whole GPUs.
   Never allocate more than gpu_request.
5. Readiness veto. If the allocation is smaller than gpu_request, the shrink adds
   sensitivity(gpu_type) x (gpu_request - allocation) / gpu_request percentage points of
   failure-to-ready risk. If ready_risk_pct + added risk > {_num(RISK_CEILING)}, veto the shrink: allocate gpu_request
   and list the pod as vetoed. A pod whose step-4 allocation already equals gpu_request is not vetoed.
6. Fleet totals. reserved_gpu_hours = sum(gpu_request x hours). reclaimed_gpu_hours =
   sum((gpu_request - allocation) x hours). reclaim_pct = 100 x reclaimed / reserved, rounded half-up
   to 2 decimals. annual_savings_usd = {MONTHS_PER_YEAR} x sum((gpu_request - allocation) x hours x price(gpu_type)),
   rounded half-up to the cent.

Cold-start fraction:
| model_type | gpu_type | cold_fraction |
|---|---|---|
{cold}

Safety margin:
| model_type | state | margin |
|---|---|---|
{margin}

GPU types:
| gpu_type | risk sensitivity (points) | price per GPU-hour (USD) |
|---|---|---|
{gpus}"""


POD_COLS = ("pod_id", "workload_id", "model_type", "gpu_type", "gpu_request", "hours", "ready_risk_pct")
HIST_COLS = ("workload_id", "day", "need_gpu")


def render_tables(fleet: Fleet) -> str:
    pods = [",".join(POD_COLS)] + [",".join(str(getattr(p, c)) for c in POD_COLS) for p in fleet.pods]
    hist = [",".join(HIST_COLS)] + [",".join(str(getattr(o, c)) for c in HIST_COLS) for o in fleet.history]
    return "Pods (CSV):\n" + "\n".join(pods) + "\n\nWorkload history (CSV):\n" + "\n".join(hist)


def render_prompt(fleet: Fleet) -> str:
    return f"{render_rulebook(fleet)}\n\n{render_tables(fleet)}"


def fleet_from_prompt(text: str) -> Fleet:
    """Rebuild a fleet from its rendered prompt (used by the reference-policy stub server)."""
    day = int(re.search(r"Snapshot \(admission\) day: (\d+)", text).group(1))
    pods_csv = text.split("Pods (CSV):\n", 1)[1].split("\n\n", 1)[0].splitlines()[1:]
    hist_csv = text.split("Workload history (CSV):\n", 1)[1].strip().splitlines()[1:]
    fleet = Fleet(seed=0, snapshot_day=day)
    for line in pods_csv:
        v = dict(zip(POD_COLS, line.split(",")))
        v["hours"] = int(v["hours"])
        fleet.pods.append(Pod(**v))
    for line in hist_csv:
        w, d, n = line.split(",")
        fleet.history.append(Observation(w, int(d), n))
    return fleet
