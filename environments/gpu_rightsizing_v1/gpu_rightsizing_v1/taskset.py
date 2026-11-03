import json
import re
from decimal import Decimal, InvalidOperation

import verifiers.v1 as vf

from gpu_rightsizing_v1.fleet import generate, render_prompt, solve

FLEET_FIELDS = ("reserved_gpu_hours", "reclaimed_gpu_hours", "reclaim_pct", "annual_savings_usd")

SYSTEM_PROMPT = """\
You are a GPU capacity engineer. Right-size every online-inference pod in the fleet snapshot by applying
the policy exactly, then report the fleet totals. Think as needed, then finish with exactly one block:
<answer>
{"allocations": {"pod_01": 0.4, "pod_02": 2}, "vetoed": ["pod_07"],
 "reserved_gpu_hours": 0.00, "reclaimed_gpu_hours": 0.00, "reclaim_pct": 0.00, "annual_savings_usd": 0.00}
</answer>
Give an allocation (in GPUs) for every pod. "vetoed" lists only pods whose shrink the readiness veto blocked."""


def parse_answer(text: str) -> dict | None:
    # The last non-empty <answer> block, found from the end so that an opener mentioned
    # in the reasoning ("Now the <answer> block:") cannot swallow the real block.
    text = text or ""
    start = len(text)
    while (start := text.rfind("<answer>", 0, start)) >= 0:
        close = text.find("</answer>", start + len("<answer>"))
        if close < 0:
            continue
        body = text[start + len("<answer>"):close].strip()
        body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body)
        if not body:
            continue
        try:
            parsed = json.loads(body)
        except (ValueError, RecursionError):  # JSONDecodeError, the int digit limit, deep nesting
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def to_cents(value) -> Decimal | None:
    try:
        return Decimal(str(value).replace(",", "").replace("$", "").replace("%", "")).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return None


def id_set(values) -> set[str]:
    return {str(v) for v in values} if isinstance(values, list) else set()


class GpuRightsizingData(vf.TaskData):
    seed: int
    expected: dict
    """Ground-truth allocations, vetoed pods and fleet totals (decimal strings)."""


class GpuRightsizingTask(vf.Task[GpuRightsizingData]):
    def _answer(self, trace: vf.Trace) -> dict | None:
        return parse_answer(trace.last_reply)

    def _field_ok(self, answer: dict | None, field: str) -> bool:
        return answer is not None and to_cents(answer.get(field)) == Decimal(self.data.expected[field])

    def _prf(self, trace: vf.Trace) -> tuple[float, float, float]:
        answer = self._answer(trace)
        if answer is None:
            return 0.0, 0.0, 0.0
        predicted, gold = id_set(answer.get("vetoed")), set(self.data.expected["vetoed"])
        if not gold and not predicted:
            return 1.0, 1.0, 1.0
        hits = len(predicted & gold)
        precision = hits / len(predicted) if predicted else 0.0
        recall = hits / len(gold) if gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    @vf.reward(weight=0.4)
    async def allocation_accuracy(self, trace: vf.Trace) -> float:
        answer = self._answer(trace)
        allocations = answer.get("allocations") if answer else None
        if not isinstance(allocations, dict):
            return 0.0
        gold = self.data.expected["allocations"]
        return sum(to_cents(allocations.get(pid)) == Decimal(v) for pid, v in gold.items()) / len(gold)

    @vf.reward(weight=0.4)
    async def fleet_correct(self, trace: vf.Trace) -> float:
        answer = self._answer(trace)
        return float(all(self._field_ok(answer, f) for f in FLEET_FIELDS))

    @vf.reward(weight=0.2)
    async def veto_audit(self, trace: vf.Trace) -> float:
        return self._prf(trace)[2]

    @vf.metric
    async def answer_parsed(self, trace: vf.Trace) -> float:
        return float(self._answer(trace) is not None)

    @vf.metric
    async def fields_correct(self, trace: vf.Trace) -> float:
        answer = self._answer(trace)
        return sum(self._field_ok(answer, f) for f in FLEET_FIELDS) / len(FLEET_FIELDS)

    @vf.metric
    async def savings_abs_error_usd(self, trace: vf.Trace) -> float:
        """|reported - true| annual savings; the true figure itself when the answer is missing or unparseable."""
        answer = self._answer(trace)
        gold = Decimal(self.data.expected["annual_savings_usd"])
        got = to_cents(answer.get("annual_savings_usd")) if answer else None
        return float(abs(got - gold)) if got is not None else float(gold)

    @vf.metric
    async def veto_precision(self, trace: vf.Trace) -> float:
        return self._prf(trace)[0]

    @vf.metric
    async def veto_recall(self, trace: vf.Trace) -> float:
        return self._prf(trace)[1]


class GpuRightsizingConfig(vf.TasksetConfig):
    num_tasks: int = 50
    """How many fleet snapshots (one task each) to build."""
    num_pods: int = 24
    """Pods per snapshot."""
    seed: int = 20261102
    """Base seed; task i uses seed + i."""


class GpuRightsizingTaskset(vf.Taskset[GpuRightsizingTask, GpuRightsizingConfig]):
    def load(self) -> list[GpuRightsizingTask]:
        tasks = []
        for i in range(self.config.num_tasks):
            seed = self.config.seed + i
            fleet = generate(seed, self.config.num_pods)
            data = GpuRightsizingData(
                id=f"gpu-{seed}",
                name=f"fleet-{seed}",
                system_prompt=SYSTEM_PROMPT,
                prompt=render_prompt(fleet),
                seed=seed,
                expected=solve(fleet),
            )
            tasks.append(GpuRightsizingTask(data, self.config.task))
        return tasks
