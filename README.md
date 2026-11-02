# verifiers-accounting

Accounting, tax-compliance and GPU-capacity RL environments for Prime Intellect's
[`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) (v1 stack).

| Environment | What it tests |
|---|---|
| [`us-sut-reconciliation-v1`](environments/us_sut_reconciliation_v1/README.md) | US state sales & use tax return reconciliation plus a fault audit over a synthetic quarterly ledger (US analogue of `au-bas-reconciliation-v1`) |
| [`gpu-rightsizing-v1`](environments/gpu_rightsizing_v1/README.md) | Right-size online-inference GPU pods (history rule, safety margin, slice grid, readiness veto) and price the reclaimed GPU-hours |

- `scripts/oracle_server.py` serves reference policies (ground-truth oracle, naive baseline) for both envs through an OpenAI-compatible endpoint.
- `scripts/summarize.py` and `scripts/summarize_gpu.py` turn `vf-eval` run dirs into results tables.
- `scripts/push_run.py` uploads a finished `vf-eval --no-push` run to Prime Evals.
- Published on the Environments Hub as [`tarive/us-sut-reconciliation-v1`](https://app.primeintellect.ai/dashboard/environments/tarive/us-sut-reconciliation-v1) and [`tarive/gpu-rightsizing-v1`](https://app.primeintellect.ai/dashboard/environments/tarive/gpu-rightsizing-v1).
- [`results/`](results/README.md) and [`results/gpu-rightsizing-v1/`](results/gpu-rightsizing-v1/README.md) hold the published eval runs.
