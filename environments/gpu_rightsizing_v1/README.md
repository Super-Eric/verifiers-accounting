# gpu-rightsizing-v1

GPU right-sizing for production online inference, built on the v1 `verifiers` stack
(`import verifiers.v1 as vf`). Each task is a synthetic fleet snapshot of online-inference pods
(recommendation and GenAI) with per-workload usage history. The model applies a right-sizing
policy to every pod, applies a readiness veto, and prices the reclaimed GPU-hours.

The policy follows *Predictive GPU Right-Sizing for Production Online Inference* (Adhikari &
Bhandari, Texas State University, 2026). That paper studies the Alibaba Serverless Infrastructure
GPU trace (`cluster-trace-gpu-v2026`, OSDI '26) and finds that a per-workload history rule reclaims
about 21% of reserved online-inference GPU-hours at a 5% under-provisioning budget.

## What comes from the paper

| Paper | Environment |
|---|---|
| A1 history rule: per-workload P90 of past need over the last 7 days, using only pods finished before admission | Step 1–2: the history window is `[day−7, day−1]`. Rows on or after the snapshot day, and rows older than 7 days, are distractors. |
| Cold start (61–68% of pods have no workload history) | Fewer than 3 eligible rows (or no `workload_id`) → cold-start fraction × request |
| Split-conformal margin per (model type × cold/warm) | Step 3: a fixed multiplicative margin per (model type × warm/cold) |
| Discrete slice grid with jumps 0.5 → 0.7 → 1.0 | Step 4: grid `0.1 … 0.5, 0.7, 1.0`, then whole GPUs; never above the request |
| GenAI need ≈ 0.98 of a GPU, so almost nothing is reclaimable | GenAI history sits at 0.80–0.99 × request, and its cold fraction is 0.95 |
| Model B veto: accept a shrink only if the added failure-to-ready risk stays under a cap | Step 5: added risk = sensitivity(GPU type) × share of request removed. Veto if base + added > 50 |
| Business value at $1.00 (A10), $3.43 (A100) and $3.90 (H100) per GPU-hour | Step 6: annual savings = 12 × monthly reclaimed GPU-hours × price |

All tables (cold fractions, margins, sensitivities, prices) are in every prompt, so the task is
closed-book. The numbers are stylized. They are not ASI measurements.

## Schemas

Task input: the policy plus two CSV tables.

- `Pods`: `pod_id, workload_id, model_type (rec|genai), gpu_type (A10|A100|H100), gpu_request, hours, ready_risk_pct`
- `Workload history`: `workload_id, day, need_gpu` (24 pods and about 12 workloads per snapshot by default)

Model output: a final `<answer>` block holding JSON:

```json
{"allocations": {"pod_01": 0.4, "pod_02": 2}, "vetoed": ["pod_07"],
 "reserved_gpu_hours": 13945.00, "reclaimed_gpu_hours": 2909.35, "reclaim_pct": 20.86,
 "annual_savings_usd": 64191.47}
```

Ground truth comes from `fleet.solve()`, which uses exact `Decimal` arithmetic.

## Rewards

| Reward | Weight | Definition |
|---|---|---|
| `allocation_accuracy` | 0.4 | Fraction of pods whose allocation matches exactly |
| `fleet_correct` | 0.4 | 1.0 iff `reserved_gpu_hours`, `reclaimed_gpu_hours`, `reclaim_pct` and `annual_savings_usd` all match to 0.01 |
| `veto_audit` | 0.2 | F1 over vetoed pod IDs (1.0 when both sets are empty) |

Metrics (unweighted): `answer_parsed`, `fields_correct` (fraction of the 4 fleet figures),
`savings_abs_error_usd`, `veto_precision`, `veto_recall`.

## Config

`--env.taskset.num-tasks` (default 50), `--env.taskset.num-pods` (24),
`--env.taskset.seed` (20261102; task *i* uses seed + *i*).

## Run

```bash
uv pip install -e environments/gpu_rightsizing_v1
uv run vf-eval gpu-rightsizing-v1 -m <model> -n 30 --env.agent.harness.id null \
  --env.agent.runtime.type subprocess --client.base-url <openai-compatible-url> --client.api-key-var <VAR>
python scripts/summarize_gpu.py outputs/<run-dir> ...
```

Results are in [`results/gpu-rightsizing-v1/`](../../results/gpu-rightsizing-v1/README.md).
