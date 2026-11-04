# verifiers-environments

Accounting, tax-compliance, GPU-capacity, weight-space-programming and MLOps-incident RL environments for Prime Intellect's
[`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) (v1 stack).

| Environment | What it tests |
|---|---|
| [`us-sut-reconciliation-v1`](environments/us_sut_reconciliation_v1/README.md) | US state sales & use tax return reconciliation plus a fault audit over a synthetic quarterly ledger (US analogue of `au-bas-reconciliation-v1`) |
| [`gpu-rightsizing-v1`](environments/gpu_rightsizing_v1/README.md) | Right-size online-inference GPU pods (history rule, safety margin, slice grid, readiness veto) and price the reclaimed GPU-hours |
| [`weight-space-v1`](environments/weight_space_v1/README.md) | Read, repair and write the raw weights of small ReLU networks as text, checked exactly on all 32 inputs (after *Sparks of Weight-Space Programming*) |
| [`ans-mlops-v1`](environments/ans_mlops_v1/README.md) | Work a model-drift incident through Agent Name Service agents (verify identities, diagnose drift vs data faults, retrain, gate, deploy), scored on the deployed model's accuracy on hidden future weeks; `standard` and `hard` modes |

- `scripts/oracle_server.py` serves reference policies (ground-truth oracle, naive baseline) for all three envs through an OpenAI-compatible endpoint.
- `scripts/summarize.py`, `scripts/summarize_gpu.py` and `scripts/summarize_wsp.py` turn `vf-eval` run dirs into results tables.
- `scripts/push_run.py` uploads a finished `vf-eval --no-push` run to Prime Evals.
- Published on the Environments Hub as [`tarive/us-sut-reconciliation-v1`](https://app.primeintellect.ai/dashboard/environments/tarive/us-sut-reconciliation-v1), [`tarive/gpu-rightsizing-v1`](https://app.primeintellect.ai/dashboard/environments/tarive/gpu-rightsizing-v1) [`tarive/weight-space-v1`](https://app.primeintellect.ai/dashboard/environments/tarive/weight-space-v1) and [`tarive/ans-mlops-v1`](https://app.primeintellect.ai/dashboard/environments/tarive/ans-mlops-v1).
- [`results/`](results/README.md), [`results/gpu-rightsizing-v1/`](results/gpu-rightsizing-v1/README.md) and [`results/weight-space-v1/`](results/weight-space-v1/README.md) hold the published eval runs. `scripts/make_wsp_infographics.py` renders the weight-space cards.
