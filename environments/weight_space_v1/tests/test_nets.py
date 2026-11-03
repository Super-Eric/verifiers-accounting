import json

from weight_space_v1.nets import (FUNCTIONS, MAX_WRITE_HIDDEN, Net, accuracy, apply_edits, functional_units, generate,
                                  net_from_answer, reference_net, render_prompt, spec_from_prompt, target_table)
from weight_space_v1.taskset import lift, parse_answer, unit_prf


def test_reference_constructions_are_exact_and_fit_the_write_budget():
    for name in FUNCTIONS:
        net = reference_net(name)
        assert net.truth_table() == target_table(name), name
        assert net.hidden <= MAX_WRITE_HIDDEN, name


def test_functions_are_distinct():
    assert len({tuple(target_table(n)) for n in FUNCTIONS}) == len(FUNCTIONS)


def test_obfuscation_preserves_function():
    for seed in range(10):
        for name in FUNCTIONS:
            spec = generate(seed, "read", name)
            assert spec.net.truth_table() == target_table(name)
            assert spec.expected["functional_units"] == functional_units(spec.net)


def test_fix_restoring_the_corrupted_parameter_repairs_the_net():
    for seed in range(10):
        for name in FUNCTIONS:
            spec = generate(seed, "fix", name)
            loc = spec.expected["corruption"]
            assert accuracy(spec.net, name) < 1
            edit = {k: loc[k] for k in ("param", "row", "col") if k in loc} | {"value": loc["original"]}
            assert accuracy(apply_edits(spec.net, [edit]), name) == 1


def test_apply_edits_rejects_malformed_or_too_many():
    net = reference_net("parity_odd")
    assert apply_edits(net, [{"param": "W1", "row": 99, "col": 0, "value": 1}]) is None
    assert apply_edits(net, [{"param": "v", "row": 0, "value": 1}] * 3) is None
    assert apply_edits(net, "nope") is None


def test_written_net_shapes():
    good = reference_net("div_by_3").to_json()
    assert accuracy(net_from_answer(good), "div_by_3") == 1
    assert net_from_answer({**good, "b1": good["b1"][:-1]}) is None
    assert net_from_answer({"W1": [[0] * 5] * 9, "b1": [0] * 9, "v": [0] * 9, "c": 0}) is None


def test_prompt_roundtrip():
    spec = generate(3, "fix", "palindrome")
    mode, net, desc = spec_from_prompt(render_prompt(spec))
    assert mode == "fix" and net.to_json() == spec.net.to_json() and desc == FUNCTIONS["palindrome"][0]


def test_scoring_helpers():
    assert unit_prf([1, 2], [1, 2]) == (1.0, 1.0, 1.0)
    assert unit_prf([], []) == (1.0, 1.0, 1.0)
    assert unit_prf(["x"], [1])[2] == 0.0
    assert lift(0.75, 0.75) == 0.0 and lift(1.0, 0.75) == 1.0
    assert parse_answer(f"...<answer>\n```json\n{json.dumps({'edits': []})}\n```\n</answer>") == {"edits": []}
    assert isinstance(Net.from_json(reference_net("one_hot").to_json()), Net)
