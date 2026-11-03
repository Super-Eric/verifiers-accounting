# weight-space-v1

Weight-space programming on the v1 `verifiers` stack (`import verifiers.v1 as vf`). The model gets the raw weights of
a small ReLU network as text and must **read** them, **fix** them or **write** them from scratch. It has no code
execution and cannot run the network, so it has to reason about what each unit computes.

The setup follows *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*
(Takehi, Teoh, Zhang & Khattab, MIT CSAIL, 2026), scaled down so that every answer is checked exactly. The post
gives a frontier model the weights of game policies, diffusion models and language models and asks which game or
language they were trained on, to repair a stuck policy, and to write game-playing networks with no training.
Here the networks are small enough (5 binary inputs, 32 possible inputs) that every claim is verified on every input.

## Networks

```
inputs  x0..x4 in {0,1}         N = x0 + 2*x1 + 4*x2 + 8*x3 + 16*x4
hidden  h_j = max(0, W1[j] . x + b1[j])
output  y = sum_j v[j] * h_j + c          output 1 if y > 0
```

16 target functions span counting (`majority`, `exactly_two`, `parity_odd`, `popcount_prime`), bit logic
(`x0_xor_x4`, `palindrome`, `high_eq_low`) and arithmetic on N (`n_in_10_20`, `div_by_3`, `perfect_square`). Each
has a hand-built reference construction of at most 6 units (for example, `div_by_3` uses N mod 3 = (x0 − x1 + x2 −
x3 + x4) mod 3 and two ReLU "hats").

The networks shown for `read` and `fix` tasks hide that construction with function-preserving rewrites:

- units split into two copies with different scales;
- per-unit positive rescaling (ReLU is positively homogeneous);
- padding with distractors up to 16 units: dead units that never fire, live units with negligible output weight, and
  pairs that cancel exactly;
- a random permutation of units, ±0.01 jitter, and rounding to 3 decimals.

Every network is re-checked on all 32 inputs. The logit stays more than 0.05 from zero on every input, both for the
full network and with any single unit removed. This means "does unit j matter" never depends on the jitter.

## Modes

Modes cycle `read`, `fix`, `write` across tasks. With the default 48 tasks, each function appears once in each mode.

| Mode | What the model sees | What it answers | `weight_score` |
|---|---|---|---|
| read | 16-unit network, menu of the 16 functions | `{"function": name, "functional_units": [j, ...]}` | 0.6 × right function + 0.4 × F1 over functional units |
| fix | 16-unit network for a stated function, with one parameter overwritten (≥3 of 32 outputs wrong) | `{"edits": [{"param": "W1", "row": j, "col": i, "value": x}, ...]}`, at most 2 | accuracy lift over the broken net: (acc − acc₀) / (1 − acc₀) |
| write | a stated function | `{"W1": [[...]], "b1": [...], "v": [...], "c": x}`, at most 8 units | accuracy lift over the best constant output |

A unit is *functional* if zeroing its output weight changes the output on at least one input. This is the analogue
of the post's "find the copying heads", with ground truth from ablation.

Metrics (unweighted): `answer_parsed`, `solved` (read: right function and exact unit set; fix / write: right on all
32 inputs), `function_correct`, `unit_f1`, `accuracy_32`, `found_corruption` (fix: an edit touched the overwritten
parameter).

## Config

`--env.taskset.num-tasks` (default 48), `--env.taskset.hidden` (16), `--env.taskset.seed` (20261103; task *i* uses
seed + *i*).

## Run

```bash
uv pip install -e environments/weight_space_v1
uv run vf-eval weight-space-v1 -m <model> -n 48 --env.agent.harness.id null \
  --env.agent.runtime.type subprocess --client.base-url <openai-compatible-url> --client.api-key-var <VAR>
python scripts/summarize_wsp.py outputs/<run-dir> ...
```

Results are in [`results/weight-space-v1/`](../../results/weight-space-v1/README.md).
