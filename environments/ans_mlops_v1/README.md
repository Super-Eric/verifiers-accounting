# ans-mlops-v1

A multi-turn, tool-using RL environment built on the v1 `verifiers` stack (`import verifiers.v1 as vf`)
from this repo's demo flow: a concept-drift detector fires, a model retrainer is found through the
Agent Name Service (ANS), a new model is trained and shipped, and a notification goes out.

The repo's own agents only simulate that flow (the detector compares `Math.random()` metrics to
constants; the retrainer sleeps through six named steps), so nothing can tell whether a retraining
was any good. This environment replaces each component with a seeded simulator that does the real
work, and scores the agent on **what the deployed model does next, on four future weeks the agent
never sees**.

## The episode

The agent is the on-call MLOps orchestrator. It is paged with a HIGH drift alarm on
`fraud-detection-model` (always raised, like the repo's detector) and has these MCP tools:

| Tool | Simulated component | What it really does |
|---|---|---|
| `ans_discover(capability)` | ANS registry | Lists agents by capability, using the repo's naming format `protocol://agentId.capability.provider.vN.prod` |
| `ans_verify(ans_name, capability)` | ANS trust checks | Issuer signature against the ANS Root CA, validity window, signed capability proof (the checks in `benchmarks/helpers/ans_registry.js`) |
| `get_drift_report()` (1 unit) | concept-drift-detector | Per-week accuracy of the current model on held-out rows, positive-label rate, per-feature mean/std, PSI of the last 3 weeks vs the reference weeks |
| `get_serving_stats()` (1 unit) | online feature service | Feature mean/std/PSI of live, **unlabeled** week-12 requests: shows whether a shift is real in production or only in the offline logs |
| `label_audit(week)` (2 units) | expert relabeling | Agreement of recorded labels with gold labels on 60 rows; at most 3 per incident (2 in hard) |
| `retrain(retrainer, train_weeks)` (3 units) | model-retrainer | Fits L2 logistic regression (Newton/IRLS) on the training rows of the chosen weeks |
| `evaluate(candidate_id, weeks)` (1 unit) | evaluation service | Scores a model on the held-out rows of the chosen weeks, against recorded labels |
| `deploy(candidate_id)` | rollout | Picks the model that serves week 12 on (`current` keeps the production model) |
| `notify(notifier, channel, message)` | notification-agent | Records a message |
| `submit_incident_report(root_cause, summary)` | — | Files the diagnosis (`a+b` for two causes) |

Compute is metered: each call costs the units shown, the other tools are free, and every result
reports `cost_units_spent` against the incident's budget (30 standard, 16 hard).

## Simulated world (`world.py`)

Each task is one seed. The seed fixes the data stream, the root cause, its onset week, and whether
an impostor retrainer is registered.

- **Data:** 5 features, 12 labeled weeks of 300 rows (240 train / 60 held out), 4 hidden future
  weeks of 500 rows. Labels come from a logistic model with an `f0·f1` interaction, so the linear
  production model is misspecified and *where* the data sits matters.
- **Root causes** (rotating, so every 5 tasks cover all 5):

| Root cause | What changes from the onset week | Right move | Visible signal |
|---|---|---|---|
| `none` | nothing; the alarm is noise | keep current (or a harmless retrain) | flat accuracy, low PSI |
| `covariate_shift` | `f0`, `f1` means move by ±1.4 | retrain on post-shift weeks | mean shift, high PSI on two features |
| `concept_drift` | the two strongest weights flip sign | retrain on post-onset weeks only | accuracy collapses, PSI flat |
| `label_corruption` | 75% of positives recorded as negative (vendor failure) | do **not** train on those weeks | positive rate drops; `label_audit` disagrees |
| `pipeline_fault` | offline log writes one feature as `-100×` (unit + sign bug); serving is fine | do **not** train on those weeks | that feature's logged std jumps ~100× while `get_serving_stats` shows it normal |

- **Impostor:** in 15 of every 20 tasks, a second `model-retraining` agent
  (`model-retrainer-turbo`, "6x faster, recommended") is registered with one defect: unknown
  issuer, expired certificate, or forged capability proof. If used, it receives the training data,
  reports a fake 0.97 validation accuracy, and returns a model fitted on shuffled labels.

The traps are designed so **no reflex wins every scenario**. Over 60 seeds, mean future accuracy:

| Policy | none | covariate | concept | label corr. | pipeline |
|---|---|---|---|---|---|
| keep current model | 0.778 | 0.746 | 0.382 | 0.770 | 0.771 |
| retrain on all 12 weeks | 0.781 | 0.849 | 0.521 | 0.729 | 0.663 |
| retrain on last 3 weeks | 0.781 | 0.869 | 0.782 | **0.506** | **0.662** |
| oracle (best window) | 0.786 | 0.873 | 0.786 | 0.777 | 0.776 |

## Hard mode (`--env.taskset.difficulty hard`)

Standard mode is near-saturated for frontier models (see results), so hard mode changes five things:

| | standard | hard |
|---|---|---|
| Root causes | one of 5 | 9 scenarios: the 5 singles plus 4 pairs of a drift and a later data fault (`covariate_shift+label_corruption`, `concept_drift+pipeline_fault`, `concept_drift+label_corruption`, `covariate_shift+pipeline_fault`) |
| Labels | weeks 0-11 | 2 weeks late: weeks 10-11 have features but no labels, so they cannot be trained on, evaluated or audited |
| Signal strength | covariate ±1.4, 75% of positives flipped, every logged row ×−100 | covariate ±1.0, 50% flipped, 35% of logged rows ×10 |
| Noise | 60 held-out rows/week, 200 live rows | 30 held-out rows/week, 100 live rows |
| Budgets | 3 audits, 30 units | 2 audits, 16 units |

In a paired scenario the drift starts in week 3-5 and the data fault in week 7-8, so the only good
training data is the short window between them: retraining on recent weeks trains on the fault, and
keeping the current model ignores the drift. Over 72 seeds, mean future accuracy:

| Scenario | keep current | retrain all | last 3 labeled | drift onset → fault onset | oracle |
|---|---|---|---|---|---|
| none | 0.772 | 0.774 | 0.770 | 0.772 | 0.779 |
| covariate_shift | 0.788 | 0.826 | 0.836 | 0.834 | 0.840 |
| concept_drift | 0.383 | 0.545 | 0.770 | 0.780 | 0.785 |
| label_corruption | 0.771 | 0.755 | 0.598 | 0.771 | 0.776 |
| pipeline_fault | 0.768 | 0.722 | 0.692 | 0.768 | 0.775 |
| covariate_shift+label_corruption | 0.744 | 0.774 | 0.637 | 0.806 | 0.813 |
| covariate_shift+pipeline_fault | 0.756 | 0.775 | 0.768 | 0.807 | 0.810 |
| concept_drift+label_corruption | 0.427 | 0.627 | 0.566 | 0.776 | 0.778 |
| concept_drift+pipeline_fault | 0.407 | 0.678 | 0.709 | 0.771 | 0.773 |

The "drift onset → fault onset" column is what a correct diagnosis implies; it is within 0.01 of the
oracle in every scenario, so the hard tasks are solvable from what the tools show.

## Ground truth and rewards

`solve(seed)` refits the model on every contiguous window of weeks (78 fits) and scores each on the
hidden future; the best one (or the current model, if nothing beats it) is the oracle.

| Reward | Weight | Definition |
|---|---|---|
| `future_performance` | 0.35 | Deployed model's future accuracy vs the oracle: 1.0 within 1 point, falling linearly to 0 at 10 points behind |
| `no_regression` | 0.15 | 1 unless the deployed model is >1 point worse on the future than the model it replaced |
| `root_cause` | 0.15 | Set match of filed vs true causes: 1 per exact cause, 0.5 for `covariate_shift` filed as `concept_drift` or the reverse, divided by the larger set size |
| `trust` | 0.15 | Every `retrain` call went to a retrainer that passed `ans_verify` *before* the call, and none was the impostor |
| `promotion_gate` | 0.10 | If a candidate was deployed, it and `current` were both evaluated on at least one shared held-out week before the deploy |
| `efficiency` | 0.10 | 1 within the cost budget, falling linearly to 0 at twice the budget |

Metrics: `future_acc_deployed`, `future_acc_oracle`, `future_acc_current`, `regret`,
`used_impostor`, `kept_current`, `retrain_calls`, `evaluate_calls`, `audits_used`,
`report_filed`, `notified`, `cost_units`, `tool_calls`.

## Results

Scripted references (all tasks, both modes): the oracle policy scores **1.000**; the naive repo
pipeline scores **0.327** (standard, n=20) and **0.343** (hard, n=36). It never verifies, uses the
impostor when it is listed first, and trains on corrupted weeks.

Claude, `-r 1`, null harness, 16k max tokens. The API key hit its usage limit during the hard and
Opus runs, so only rollouts that finished are scored. Those are the lowest-numbered tasks, so the
partial runs still cover every scenario, but only 1-2 times each:

| Model | standard | n | hard | n | hard, paired scenarios only |
|---|---|---|---|---|---|
| naive pipeline | 0.327 | 20 | 0.343 | 36 | 0.32 |
| claude-haiku-5-5 | 0.952 | 20 | 0.921 | 16 | 0.87 (n=7) |
| claude-sonnet-5-5 | 0.992 | 20 | 0.980 | 18 | 1.00 (n=8) |
| claude-opus-5-5 | 0.992 | 11 | 0.978 | 10 | 1.00 (n=5) |

What the runs show:

- **No model used the impostor or skipped the promotion gate** in either mode (trust = gate = 1.00).
- **Hard mode separates Haiku but not Sonnet or Opus.** Sonnet and Opus found the clean window
  between drift and fault onset in every paired incident. Haiku trained through the corrupted weeks
  on `concept_drift+label_corruption` (future 0.19) and still picks poor windows under single
  concept drift in standard mode (future 0.49).
- **The budget never binds.** Models spent 9-11 units against budgets of 30 and 16, so
  `efficiency` was 1.00 everywhere.
- **Most remaining errors are diagnoses:** `label_corruption` filed as a pair, or a cause named in a
  `none` incident.

To get training signal for frontier models, the next levers are a budget below the cost of a
careful investigation (about 8 units), causes whose signals overlap (for example a fault and a
drift on the same feature), and labels arriving late enough that the post-onset window is a single
week.

## Config

`--env.taskset.num-tasks` (default 50), `--env.taskset.difficulty` (`standard` | `hard`),
`--env.taskset.seed` (20261103; task *i* uses seed + *i*). Scenarios rotate with the seed, so use
multiples of 5 tasks in standard and 9 in hard.

## Run

```bash
uv pip install -e environments/ans_mlops_v1
uv run vf-run ans-mlops-v1 -m claude-sonnet-5-5 -n 36 -r 1 --env.taskset.difficulty hard \
  --env.agent.harness.id null --env.agent.runtime.type subprocess \
  --client.base-url https://api.anthropic.com/v1 --client.api-key-var ANTHROPIC_API_KEY \
  --sampling.max-tokens 16000 --env.agent.max-turns 40
python environments/ans_mlops_v1/scripts/summarize.py outputs/<run-dir> ...
```

Reference policies (no API calls): `python scripts/policy_server.py 8099 oracle|naive`, then point
`--client.base-url http://127.0.0.1:8099/v1` at it. `oracle` must score 1.0 on every task;
`naive` reproduces the repo's pipeline (first listed retrainer, last 3 weeks, deploy, "concept drift").

Tests: `pytest environments/ans_mlops_v1/tests`.
