# weight-space-v1: eval results

**Eval set:** the default 48 tasks (seeds 20261103–20261150). Modes cycle read, fix and write, so every one of the 16
functions appears once per mode. Read and fix nets have 16 hidden units; written nets may use up to 8.
**Harness:** `null` (plain chat, no tools), `subprocess` runtime, `verifiers` 0.4.0. One rollout per task for the
reference policies, three for the Claude models (default effort, `--sampling.max-tokens 32000`).
**Run date:** 2026-11-03. Cards for X: [`cards/`](cards/) (`scripts/make_wsp_infographics.py`).
**Environments Hub:** [`tarive/weight-space-v1`](https://app.primeintellect.ai/dashboard/environments/tarive/weight-space-v1).
**Prime Evals:**
[Haiku 5.5](https://app.primeintellect.ai/dashboard/evaluations/mlg22m5jxtbcq9exxuzneog8),
[Sonnet 5.5](https://app.primeintellect.ai/dashboard/evaluations/zbzvqxr0lfdjsyn9ywea6mzk).

## Summary

| run | n | score | score σ† | read | fix | write | solved | parsed | output tokens / episode |
|---|---|---|---|---|---|---|---|---|---|
| oracle | 48 | 1.000 | - | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| naive | 48 | 0.069 | - | 0.206 | 0.000 | 0.000 | 0.000 | 1.000 | 0 |
| claude-haiku-5-5 | 144 | 0.971 | 0.006 | 0.964 | 0.949 | 1.000 | 0.861 | 0.986 | 7,483 |
| claude-sonnet-5-5 | 144 | 0.998 | 0.002 | 0.997 | 0.996 | 1.000 | 0.972 | 1.000 | 3,496 |

† Std across the per-rollout mean scores; n counts episodes (tasks × rollouts).

Per mode (solved = read: right function and exact unit set; fix / write: right on all 32 inputs):

| run | read: function right | read: unit F1 | read solved | fix: found the bad weight | fix: accuracy | fix solved | write: accuracy | write solved |
|---|---|---|---|---|---|---|---|---|
| oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| naive | 0.062 | 0.421 | 0.000 | 0.000 | 0.676 | 0.000 | 0.625 | 0.000 |
| claude-haiku-5-5 | 0.979 | 0.941 | 0.646 | 0.500 | 0.956 | 0.938 | 1.000 | 1.000 |
| claude-sonnet-5-5 | 1.000 | 0.992 | 0.938 | 0.667 | 0.999 | 0.979 | 1.000 | 1.000 |

Solved, by function and mode (episodes solved / episodes):

| function | oracle read | oracle fix | oracle write | naive read | naive fix | naive write | claude-haiku-5-5 read | claude-haiku-5-5 fix | claude-haiku-5-5 write | claude-sonnet-5-5 read | claude-sonnet-5-5 fix | claude-sonnet-5-5 write |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all_ones | 1/1 | 1/1 | 1/1 | 0/1 | 0/1 | 0/1 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| any_one | 1/1 | 1/1 | 1/1 | 0/1 | 0/1 | 0/1 | 3/3 | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 |
| majority | 1/1 | 1/1 | 1/1 | 0/1 | 0/1 | 0/1 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| exactly_two | 1/1 | 1/1 | 1/1 | 0/1 | 0/1 | 0/1 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| one_hot | 1/1 | 1/1 | 1/1 | 0/1 | 0/1 | 0/1 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| parity_odd | 1/1 | 1/1 | 1/1 | 0/1 | 0/1 | 0/1 | 2/3 | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 |
| popcount_prime | 1/1 | 1/1 | 1/1 | 0/1 | 0/1 | 0/1 | 1/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| x0_xor_x4 | 1/1 | 1/1 | 1/1 | 0/1 | 0/1 | 0/1 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| x2_and_not_x0 | 1/1 | 1/1 | 1/1 | 0/1 | 0/1 | 0/1 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| high_gt_low | 1/1 | 1/1 | 1/1 | 0/1 | 0/1 | 0/1 | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| high_eq_low | 1/1 | 1/1 | 1/1 | 0/1 | 0/1 | 0/1 | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| palindrome | 1/1 | 1/1 | 1/1 | 0/1 | 0/1 | 0/1 | 1/3 | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 |
| n_ge_20 | 1/1 | 1/1 | 1/1 | 0/1 | 0/1 | 0/1 | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| n_in_10_20 | 1/1 | 1/1 | 1/1 | 0/1 | 0/1 | 0/1 | 1/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| div_by_3 | 1/1 | 1/1 | 1/1 | 0/1 | 0/1 | 0/1 | 1/3 | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 |
| perfect_square | 1/1 | 1/1 | 1/1 | 0/1 | 0/1 | 0/1 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |

## Cost

Anthropic list prices: Haiku 5.5 $0.10 / $0.50 and Sonnet 5.5 $2 / $10 per MTok, input / output.

| model | input tokens | output tokens | cost |
|---|---|---|---|
| claude-haiku-5-5 | 160K | 1,078K | $0.55 |
| claude-sonnet-5-5 | 160K | 503K | $5.35 |

## Findings

- **The env is well-posed.**
  - The oracle scores 1.000 through the full `vf-eval` → harness → reward path.
  - The naive policy guesses "majority", makes no edits and writes an all-zero net. It scores 0.069.
- **Writing is saturated.**
  - Both models wrote all 48 networks correctly, on all 32 inputs, in all 3 attempts.
  - For `div_by_3`, Sonnet 5.5 used the identity N mod 3 = (x0 − x1 + x2 − x3 + x4) mod 3. It built two ReLU spikes
    from **4 units in every attempt**. Our hand-built reference uses 5.
  - Haiku 5.5 found the same 4-unit network once. Its other two attempts used input weights `1 2 1 2 1`
    (2^i mod 3) and a 7–8-unit ladder.
- **Reading is the biggest gap.**
  - Both models name the function almost every time: Haiku 98%, Sonnet 100%.
  - The gap is in the exact set of units the network depends on: Sonnet 94%, Haiku 65%.
  - Haiku mostly over-includes units, typically the redundant half of a split unit or a near-silent unit.
- **The biggest single fail** was Haiku on `read popcount_prime`.
  - It listed all 7 functional units exactly and even noted that they "act as threshold units on the popcount".
  - It then answered `n_ge_20`. The five inputs it checked (N = 0, 16, 20, 21, 31) fit both functions.
  - 17 of the 32 inputs separate the two, and none of them appears in its written reasoning.
- **Fixing often succeeds without touching the broken weight.**
  - Sonnet repaired 98% of nets exactly, but only 67% of its edits touched the overwritten parameter.
  - In the rest it compensated elsewhere. For example, on `fix popcount_prime` it set the matching weight of the
    corrupted unit's *cancelling twin* (W1[4][4] = −3.413), so the pair cancels again.
  - Haiku fixed 94% of nets. Two of its fix runs hit the 32K-token cap with no answer.
- **Haiku spends 2.1× more output tokens than Sonnet** (7.5K vs 3.5K per task) for a lower score.

## Relation to the post

The env follows the protocol of *Sparks of Weight-Space Programming* (logs:
[RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs)):
- one tool-free reply per task;
- weights shown as rounded text;
- edits written in a small edit language and applied by the harness to the weights exactly as shown;
- scoring done outside the model.

It differs in four ways:
- The networks are tiny: 5 inputs and 16 hidden units, against the post's 17K to 1.5M weights.
- Every answer is checked exactly on all 32 inputs.
- The post's read prompts forbid mental forward passes. Here they are allowed, because no harness can enforce that.
- The read tasks score the unit set by ablation, as the analogue of the post's copying-head finding.

## Caveats

- The first full run used 12-unit nets and an ablation ground truth that could hinge on the ±0.01 jitter. For
  example, one unit's removal left a logit margin of 0.00006. That run was discarded.
- The env now requires every single-unit ablation to keep the logit at least 0.05 from zero. The results above are
  from that version.
- Default effort. The thinking is not returned through the OpenAI-compatible endpoint, so the traces hold only the
  final answers.

## Reproduce

```bash
uv pip install -e environments/weight_space_v1
python scripts/oracle_server.py 8111 oracle &   # or: naive
uv run vf-eval weight-space-v1 -m <model> -n 48 -r 3 -c 8 --sampling.max-tokens 32000 \
  --env.agent.harness.id null --env.agent.runtime.type subprocess \
  --client.base-url https://api.anthropic.com/v1 --client.api-key-var ANTHROPIC_API_KEY --no-push
python scripts/summarize_wsp.py outputs/<run-dir> ...
python scripts/make_wsp_infographics.py <haiku-run-dir> <sonnet-run-dir>
```
