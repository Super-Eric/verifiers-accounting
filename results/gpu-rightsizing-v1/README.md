# gpu-rightsizing-v1: eval results

**Eval set:** the first 30 tasks of the default taskset (seeds 20261102–20261131). Each task has 24 pods and about
12 workloads, 720 pods in total. Under the policy, 28% of pods shrink and 5% have their shrink vetoed.
**Harness:** `null` (plain chat, no tools), `subprocess` runtime, `verifiers` 0.4.0. One rollout per task for the
reference policies, three rollouts per task for the Claude models.
**Run date:** 2026-11-02. Infographic: [`../infographic.png`](../infographic.png) (`scripts/make_infographic.py`).
**Environments Hub:** [`tarive/gpu-rightsizing-v1`](https://app.primeintellect.ai/dashboard/environments/tarive/gpu-rightsizing-v1).
**Prime Evals:**
[Haiku 5.5](https://app.primeintellect.ai/dashboard/evaluations/d1ks7r47w9yxbnrw73itb1po),
[Sonnet 5.5](https://app.primeintellect.ai/dashboard/evaluations/vtdhx5jk9sk5wito6sl3j4qt).

## Summary

| run | n | reward | reward σ† | allocation_accuracy | fleet_correct | veto_audit (F1) | veto P | veto R | fields_correct | parsed |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 30 | 1.000 | - | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| naive-overreserve | 30 | 0.346 | - | 0.715 | 0.000 | 0.300 | 0.300 | 0.300 | 0.250 | 1.000 |
| claude-haiku-5-5 | 90 | 0.965 | 0.017 | 0.998 | 0.922 | 0.985 | 0.983 | 0.989 | 0.950 | 1.000 |
| claude-sonnet-5-5 | 90 | 0.972 | 0.011 | 0.996 | 0.933 | 1.000 | 1.000 | 1.000 | 0.956 | 1.000 |

† Std across the per-rollout mean rewards; n counts episodes (tasks × rollouts).

`reward = 0.4·allocation_accuracy + 0.4·fleet_correct + 0.2·veto_audit`.

Fleet figures missed (episodes):

| run | reserved_gpu_hours | reclaimed_gpu_hours | reclaim_pct | annual_savings_usd |
|---|---|---|---|---|
| naive-overreserve | 0 | 30 | 30 | 30 |
| claude-haiku-5-5 | 0 | 6 | 5 | 7 |
| claude-sonnet-5-5 | 0 | 5 | 5 | 6 |

## The money

Annual savings the policy finds, and how far off each model's dollar figure is (USD; summed over episodes):

| run | true annual savings | exact to the cent | median error | mean error | total error |
|---|---|---|---|---|---|
| oracle | $1,130,851.76 (30 fleets) | 30/30 | $0.00 | $0.00 | $0.00 (0.0%) |
| naive-overreserve | $1,130,851.76 | 0/30 | $33,179.81 | $37,695.06 | $1,130,851.76 (100%) |
| claude-haiku-5-5 | $3,392,555.28 (30 fleets × 3) | 83/90 | $0.00 | $496.16 | $44,654.82 (1.3%) |
| claude-sonnet-5-5 | $3,392,555.28 | 84/90 | $0.00 | $150.37 | $13,533.60 (0.4%) |

The average fleet of 24 pods leaves about **$37.7K a year** on the table when every pod keeps its full request.

What the benchmark cost to run (Anthropic list prices: Haiku 5.5 $0.10 / $0.50, Sonnet 5.5 $2 / $10 per MTok in / out):

| model | env | input tokens | output tokens | cost |
|---|---|---|---|---|
| claude-haiku-5-5 | gpu-rightsizing-v1 | 273K | 610K | $0.33 |
| claude-sonnet-5-5 | gpu-rightsizing-v1 | 273K | 376K | $4.30 |
| claude-haiku-5-5 | us-sut-reconciliation-v1 | 246K | 460K | $0.25 |
| claude-sonnet-5-5 | us-sut-reconciliation-v1 | 246K | 294K | $3.43 |
| **total** | | | | **$8.31** |

For scale, the source paper (table 16) prices the same policy family at **$1.84M–$7.17M per year per 1,000 reserved
inference GPUs** (21% reclaimed at $1.00–$3.90 per GPU-hour).

## Findings

- **The env is well-posed.** The oracle scores 1.000 through the full `vf-eval` → harness → reward path. The
  over-reserving baseline (every pod keeps its request) scores 0.346. Its points come from pods that the policy also
  leaves at full size, and from fleets with no vetoes.
- **Both models apply the policy almost perfectly.** Allocation accuracy is ≥ 0.996, so at most about 1 pod in 250
  gets the wrong slice. Haiku and Sonnet are within 0.007 reward of each other. This task does not separate them the
  way the tax env did (0.795 vs 0.978).
- **The errors are in the roll-up, not in finding the history.** No model ever got `reserved_gpu_hours` wrong.
  Misses come from two sources:
  - **Slice rounding.** A cold `rec`/A10 pod needs 0.6 × 1.25 = 0.75 GPU, which the rule rounds *up* to 1.0. Sonnet
    rounds it down to 0.7 on the same task in all three rollouts.
  - **Summation slips.** Totals were wrong even when every allocation was right, for example $39,773.22 reported vs
    $39,774.42 true.
- **Small slips become large dollar errors.** On task 3, one Haiku rollout got one slice wrong, added one veto and
  mis-summed the totals. It reported $67,088 of savings against a true $32,926. Over 90 episodes Haiku's dollar error totals $44.7K (1.3%) and Sonnet's $13.5K (0.4%).
- **Both models follow the output contract.** All 180 rollouts parse.

## Caveats

- At 24 pods per snapshot the env is close to saturated for frontier models. Raise `--env.taskset.num-pods` to make it
  harder.
- The fleet is synthetic. Rates, margins and risk sensitivities are stylized versions of the paper's setting, not
  ASI measurements. The env tests whether a model can *apply* a right-sizing policy, not whether it can learn one.
- Default effort (Sonnet 5.5 `high`, Haiku 5.5 `medium`). The thinking is not returned through the OpenAI-compatible
  endpoint, so the traces hold only the final answer.

## Reproduce

```bash
uv pip install -e environments/gpu_rightsizing_v1
python scripts/oracle_server.py 8101 oracle &   # or: naive
uv run vf-eval gpu-rightsizing-v1 -m <model> -n 30 -r 3 -c 8 --sampling.max-tokens 32000 \
  --env.agent.harness.id null --env.agent.runtime.type subprocess \
  --client.base-url https://api.anthropic.com/v1 --client.api-key-var ANTHROPIC_API_KEY --no-push
python scripts/summarize_gpu.py outputs/<run-dir> ...
python scripts/push_run.py outputs/<run-dir>
```
