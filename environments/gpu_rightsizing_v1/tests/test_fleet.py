import json
from decimal import Decimal

from gpu_rightsizing_v1.fleet import fleet_from_prompt, generate, p90, render_prompt, round_to_slice, solve
from gpu_rightsizing_v1.taskset import parse_answer, to_cents


def test_deterministic():
    assert solve(generate(7)) == solve(generate(7))


def test_p90_nearest_rank():
    assert p90([Decimal(x) for x in ("0.1", "0.5", "0.3")]) == Decimal("0.5")
    assert p90([Decimal(str(i)) for i in range(1, 11)]) == Decimal("9")


def test_round_to_slice():
    assert round_to_slice(Decimal("0.21"), Decimal("1")) == Decimal("0.25")
    assert round_to_slice(Decimal("0.71"), Decimal("1")) == Decimal("1")
    assert round_to_slice(Decimal("1.2"), Decimal("4")) == Decimal("2")
    assert round_to_slice(Decimal("0.45"), Decimal("0.5")) == Decimal("0.5")


def test_prompt_roundtrip():
    for s in range(20):
        fleet = generate(s)
        assert solve(fleet_from_prompt(render_prompt(fleet))) == solve(fleet)


def test_cases_occur():
    sols = [solve(generate(s)) for s in range(30)]
    assert any(s["vetoed"] for s in sols)
    assert all(Decimal(s["reclaimed_gpu_hours"]) >= 0 for s in sols)


def test_parse_oracle_roundtrip():
    e = solve(generate(3))
    parsed = parse_answer(f"work...\n<answer>\n```json\n{json.dumps(e)}\n```\n</answer>")
    assert to_cents(parsed["annual_savings_usd"]) == Decimal(e["annual_savings_usd"])
    assert parse_answer("no answer") is None


def test_parse_takes_the_real_block():
    e = solve(generate(3))
    block = f"<answer>{json.dumps(e)}</answer>"
    assert parse_answer(f"Now the <answer> block:\n{block}") == e  # a tag named in the reasoning
    assert parse_answer(f"{block}\nI wrapped it in <answer></answer> as asked.") == e  # an empty pair after
    assert parse_answer(f"<answer>{{}}</answer>\n{block}") == e  # the last block is still the one read
    assert parse_answer(f"{block}\n<answer>{{}}</answer>") == {}
    assert parse_answer("<answer>{\"allocations\": " + "1" * 5000 + "}</answer>") is None  # not a crash
    assert parse_answer("<answer>" + "[" * 100000 + "]" * 100000 + "</answer>") is None
