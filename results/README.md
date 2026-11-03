# us-sut-reconciliation-v1 — eval results

**Eval set:** the first 30 tasks of the default taskset (seeds 20260701–20260730). Each ledger has 30 rows plus injected
duplicates (31.2 rows on average) and the default fault rate of 0.25. In total that is 199 gold `(txn, fault)` pairs
across all six codes.
**Harness:** `null` (plain chat, no tools), `subprocess` runtime, `verifiers` 0.4.0. One rollout per task for the
reference policies and Qwen, three rollouts per task for the Claude models.
**Run date:** 2026-10-28.
**Environments Hub:** [`tarive/us-sut-reconciliation-v1`](https://app.primeintellect.ai/dashboard/environments/tarive/us-sut-reconciliation-v1).
The Claude runs are also uploaded to Prime Evals:
[Haiku 5.5](https://app.primeintellect.ai/dashboard/evaluations/hravp9mu0i6iypzbct2a4mmm),
[Sonnet 5.5](https://app.primeintellect.ai/dashboard/evaluations/ckrjqdglwjnnbtscqk81xadh).

## Summary

| run | n | reward | reward σ† | return_correct | fault_audit (F1) | fault P | fault R | fields_correct | parsed | lenient reward* | lenient F1* | lenient fields* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 30 | 1.000 | - | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| naive-bookkeeper | 30 | 0.013 | - | 0.000 | 0.033 | 0.033 | 0.033 | 0.113 | 1.000 | 0.013 | 0.033 | 0.113 |
| qwen2.5-3b-instruct-q4km | 30 | 0.000 | - | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.029 | 0.071 | 0.007 |
| claude-haiku-5-5 | 90 | 0.795 | 0.033 | 0.667 | 0.986 | 0.976 | 0.999 | 0.862 | 1.000 | 0.795 | 0.986 | 0.862 |
| claude-sonnet-5-5 | 90 | 0.978 | 0.017 | 0.967 | 0.995 | 0.995 | 0.996 | 0.987 | 1.000 | 0.978 | 0.995 | 0.987 |

† Std across the per-rollout mean rewards (`-r` > 1 only); n counts episodes (tasks × rollouts).

\* Diagnostic only: also accepts a bare JSON object when the `<answer>` tags are missing.

Per-fault recall, lenient parse (hits / gold; the Claude runs count each gold pair once per rollout):

| run | duplicate_invoice | missing_exemption_cert | nexus_misallocation | outside_period | tax_miscalculated | tax_on_exempt_item |
|---|---|---|---|---|---|---|
| oracle | 31/31 | 45/45 | 29/29 | 40/40 | 22/22 | 32/32 |
| naive-bookkeeper | 0/31 | 0/45 | 0/29 | 0/40 | 0/22 | 0/32 |
| qwen2.5-3b-instruct-q4km | 9/31 | 3/45 | 0/29 | 0/40 | 0/22 | 0/32 |
| claude-haiku-5-5 | 93/93 | 135/135 | 86/87 | 120/120 | 66/66 | 96/96 |
| claude-sonnet-5-5 | 93/93 | 135/135 | 85/87 | 119/120 | 66/66 | 96/96 |

Return figures missed (rollouts out of 90; a wrong `sales_tax_due` also makes `net_remittance` wrong):

| run | gross_sales | taxable_sales | sales_tax_due | use_tax_due | net_remittance | tasks with ≥1 miss |
|---|---|---|---|---|---|---|
| claude-haiku-5-5 | 0 | 1 | 25 | 6 | 30 | 20/30 |
| claude-sonnet-5-5 | 0 | 0 | 3 | 0 | 3 | 3/30 |

`reward = 0.6·return_correct + 0.4·fault_audit`. `fields_correct` is the fraction of the five return figures that
match to the cent. `parsed` is the share of rollouts with a valid `<answer>{json}</answer>` block.

## Policies

| Run | What it is |
|---|---|
| `oracle` | Stub endpoint (`scripts/oracle_server.py`) that answers with `reconcile()` ground truth. It is the upper bound and an end-to-end wiring check. |
| `naive-bookkeeper` | Stub that files the ledger as recorded: it sums every sale line and its recorded tax, accrues no use tax and flags nothing. It is the lower bound. |
| `qwen2.5-3b-instruct-q4km` | Qwen2.5-3B-Instruct (Q4_K_M GGUF) served by llama.cpp on 4 CPU cores. Temperature 0, max 2048 tokens. |
| `claude-haiku-5-5` | Claude Haiku 5.5 through Anthropic's OpenAI-compatible endpoint (`https://api.anthropic.com/v1`). Default sampling and effort, max 32000 tokens, 3 rollouts per task. 246K input / 460K output tokens, about $0.25. |
| `claude-sonnet-5-5` | Claude Sonnet 5.5, same setup. 246K input / 294K output tokens, about $3.43. |

## Findings

- **The env is well-posed.** The oracle scores 1.000 on every task through the full `vf-eval` → harness → reward path.
  The naive bookkeeper scores 0.013, and its only points come from a fault-free ledger where an empty fault map is
  correct.
- **Finding the faults is easy for frontier models, as predicted.** Both Claude models reach ≥ 0.986 fault F1 and
  overall recall ≥ 0.996, and at least 85/87 hits on every fault code, including the rate-based ones the 3B model
  never finds. Haiku's lower
  precision (0.976) comes from a few extra flags, not missed faults.
- **The exact roll-up separates the models, but less than the design predicted.** Haiku files the whole return
  exactly right in 67% of rollouts. Sonnet does so in 97%, so the predicted "high F1, low exact-return" gap shows up
  clearly only for the smaller model. Almost every miss is `sales_tax_due` (and therefore `net_remittance`) off by
  1–4 cents. That is the signature of summing unrounded line taxes, or rounding at the wrong step, instead of rounding
  each line half-up and then summing. Haiku has 24 such cent-level misses plus one answer off by $131.37. Sonnet's
  three misses are −$0.20, −$0.01 and −$0.01. Haiku also misses `use_tax_due` in 6 rollouts.
- **Both models follow the output contract.** All 180 rollouts emit a parseable `<answer>` block, so the strict and
  lenient scores are identical.
- **Run-to-run variance is small.** The std of the per-rollout mean reward is 0.033 (Haiku) and 0.017 (Sonnet).
- **The 3B model fails the output contract.** In 0 of 30 rollouts did Qwen2.5-3B emit the required `<answer>` tags,
  so it scores 0 under the strict reward. In 28 rollouts it answered with bare JSON straight away (no worked
  reasoning). In the other 2 it wrote arithmetic expressions inside the JSON, which made the JSON invalid. Even with a
  lenient parse it reaches only 0.071 fault F1 and 0.7% field accuracy. It finds some `duplicate_invoice` (9/31) and
  `missing_exemption_cert` (3/45) faults, finds no rate-based fault (`tax_miscalculated`, `nexus_misallocation`,
  `tax_on_exempt_item`), and finds no `outside_period` fault. Those categories need per-row arithmetic that a 3B
  model answering in one shot does not do.
- **A reward loophole was fixed during the first runs.** An unparseable answer used to count as an empty fault map,
  which earns F1 = 1.0 on a fault-free ledger (task 3 scored 0.4 that way). Unparsed answers now score 0 on
  `fault_audit`. The oracle, naive and Qwen runs were re-scored with `vf-replay` against the fixed rubric, and the
  published `traces.jsonl` files are those re-scored traces. The Claude runs were scored with the fixed rubric directly.

## Caveats

- Sonnet 5.5 nearly saturates the benchmark at 30 rows per ledger (0.978). Before it can rank stronger models it
  needs a harder setting, such as more rows (`--env.taskset.num-rows`) or a higher fault rate.
- The Claude runs use default effort (Sonnet 5.5: `high`, Haiku 5.5: `medium`) with adaptive thinking. The thinking
  is not returned through the OpenAI-compatible endpoint, so the traces hold only the final answer.
- The jurisdiction is an illustrative rulebook with stylized rates. It is not current statutory law.
- **Grading changes since these runs (v0.1.1):**
  - `exempt_sales` is now one of the graded return figures. It was asked for in the prompt but never compared.
  - `parse_answer` now reads the last non-empty `<answer>` block, searching from the end. This fixes an `<answer>`
    mentioned in the reasoning, or an empty pair after the answer, zeroing a correct reply.
  - Neither change alters any reward above. No rollout missed only `exempt_sales`, and no rollout has a second or
    empty `<answer>` tag.
  - `fields_correct` re-scored on 6 figures: Haiku 0.883, Sonnet 0.989, naive 0.117 (was 0.862 / 0.987 / 0.113 on
    5 figures).

## Reproduce

```bash
uv pip install -e environments/us_sut_reconciliation_v1
uv run vf-eval us-sut-reconciliation-v1 -m <model> -n 30 -c 8 --env.agent.harness.id null \
  --env.agent.runtime.type subprocess --client.base-url <url> --client.api-key-var <KEY_VAR> --no-push
python scripts/summarize.py outputs/<run-dir>
```

For the Claude runs: `-m claude-sonnet-5-5` (or `claude-haiku-5-5`) `-r 3 --sampling.max-tokens 32000
--client.base-url https://api.anthropic.com/v1 --client.api-key-var ANTHROPIC_API_KEY`.
`python scripts/push_run.py <run-dir>` uploads a finished `--no-push` run to Prime Evals.

Each `results/<run>/` folder holds the resolved `eval-config.json` and the per-rollout `traces.jsonl` (full transcripts,
rewards and metrics).
