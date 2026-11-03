import json
import re
from decimal import Decimal, InvalidOperation

import verifiers.v1 as vf

from us_sut_reconciliation_v1.ledger import generate, reconcile, render_ledger, render_rulebook

RETURN_FIELDS = ("gross_sales", "exempt_sales", "taxable_sales", "sales_tax_due", "use_tax_due", "net_remittance")

SYSTEM_PROMPT = """\
You are a US state sales & use tax auditor. Reconcile the quarterly ledger against the rulebook, prepare the
return, and flag every bookkeeping fault. Think as needed, then finish with exactly one block:
<answer>
{"gross_sales": 0.00, "exempt_sales": 0.00, "taxable_sales": 0.00, "sales_tax_due": 0.00,
 "use_tax_due": 0.00, "net_remittance": 0.00, "faults": {"txn_001": ["fault_code"]}}
</answer>
Amounts are dollars to the cent. Omit rows with no faults from "faults"."""


def parse_answer(text: str) -> dict | None:
    blocks = re.findall(r"<answer>(.*?)</answer>", text or "", re.DOTALL)
    if not blocks:
        return None
    body = blocks[-1].strip()
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body)
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def to_cents(value) -> Decimal | None:
    try:
        return Decimal(str(value).replace(",", "").replace("$", "")).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return None


def fault_pairs(faults) -> set[tuple[str, str]]:
    if not isinstance(faults, dict):
        return set()
    pairs = set()
    for tid, codes in faults.items():
        for code in codes if isinstance(codes, list) else [codes]:
            pairs.add((str(tid), str(code)))
    return pairs


class UsSutReconciliationData(vf.TaskData):
    seed: int
    expected: dict
    """Ground-truth return figures (as cent strings) and the fault map."""


class UsSutReconciliationTask(vf.Task[UsSutReconciliationData]):
    def _answer(self, trace: vf.Trace) -> dict | None:
        return parse_answer(trace.last_reply)

    def _field_ok(self, answer: dict | None, field: str) -> bool:
        return answer is not None and to_cents(answer.get(field)) == Decimal(self.data.expected[field])

    def _prf(self, trace: vf.Trace) -> tuple[float, float, float]:
        answer = self._answer(trace)
        if answer is None:
            return 0.0, 0.0, 0.0
        predicted = fault_pairs(answer.get("faults"))
        gold = fault_pairs(self.data.expected["faults"])
        if not gold and not predicted:
            return 1.0, 1.0, 1.0
        hits = len(predicted & gold)
        precision = hits / len(predicted) if predicted else 0.0
        recall = hits / len(gold) if gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    @vf.reward(weight=0.6)
    async def return_correct(self, trace: vf.Trace) -> float:
        answer = self._answer(trace)
        return float(all(self._field_ok(answer, f) for f in RETURN_FIELDS))

    @vf.reward(weight=0.4)
    async def fault_audit(self, trace: vf.Trace) -> float:
        return self._prf(trace)[2]

    @vf.metric
    async def answer_parsed(self, trace: vf.Trace) -> float:
        return float(self._answer(trace) is not None)

    @vf.metric
    async def fields_correct(self, trace: vf.Trace) -> float:
        answer = self._answer(trace)
        return sum(self._field_ok(answer, f) for f in RETURN_FIELDS) / len(RETURN_FIELDS)

    @vf.metric
    async def fault_precision(self, trace: vf.Trace) -> float:
        return self._prf(trace)[0]

    @vf.metric
    async def fault_recall(self, trace: vf.Trace) -> float:
        return self._prf(trace)[1]


class UsSutReconciliationConfig(vf.TasksetConfig):
    num_tasks: int = 50
    """How many ledgers (one task each) to build."""
    num_rows: int = 30
    """Transactions per ledger, before injected duplicates."""
    fault_rate: float = 0.25
    """Probability a generated row carries an injected fault."""
    seed: int = 20260701
    """Base seed; task i uses seed + i."""


class UsSutReconciliationTaskset(vf.Taskset[UsSutReconciliationTask, UsSutReconciliationConfig]):
    def load(self) -> list[UsSutReconciliationTask]:
        tasks = []
        for i in range(self.config.num_tasks):
            seed = self.config.seed + i
            ledger = generate(seed, self.config.num_rows, self.config.fault_rate)
            prompt = f"{render_rulebook(ledger)}\n\nLedger (CSV):\n{render_ledger(ledger)}"
            data = UsSutReconciliationData(
                id=f"sut-{seed}",
                name=f"ledger-{seed}",
                system_prompt=SYSTEM_PROMPT,
                prompt=prompt,
                seed=seed,
                expected=reconcile(ledger),
            )
            tasks.append(UsSutReconciliationTask(data, self.config.task))
        return tasks
