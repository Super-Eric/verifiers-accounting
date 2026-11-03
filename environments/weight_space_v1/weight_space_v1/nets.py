"""Tiny ReLU networks on 5-bit inputs, read and written as text.

Every network has the same shape: inputs x0..x4 in {0, 1}, one hidden ReLU layer, one output logit.

    h_j = max(0, sum_i W1[j][i] * x_i + b1[j])
    y   = sum_j v[j] * h_j + c          output = 1 if y > 0 else 0

There are only 32 inputs, so every claim about a network is checked exactly: forward passes use `Decimal`
arithmetic on the 3-decimal weights the model sees.

Three task modes, after *Sparks of Weight-Space Programming* (Takehi et al., 2026):
  read  - name the function a network computes and list the hidden units it depends on
  fix   - one parameter of a network was corrupted; repair it with at most two edits
  write - write the weights of a network for a described function, from scratch
"""

import random
from dataclasses import dataclass
from decimal import Decimal

N_IN = 5
INPUTS = [tuple((n >> i) & 1 for i in range(N_IN)) for n in range(2**N_IN)]
Q = Decimal("0.001")
MAX_WRITE_HIDDEN = 8
MAX_FIX_EDITS = 2


@dataclass
class Net:
    W1: list[list[Decimal]]
    b1: list[Decimal]
    v: list[Decimal]
    c: Decimal

    @property
    def hidden(self) -> int:
        return len(self.b1)

    def hidden_acts(self, x: tuple[int, ...]) -> list[Decimal]:
        return [max(Decimal(0), sum((w * xi for w, xi in zip(row, x)), b)) for row, b in zip(self.W1, self.b1)]

    def logit(self, x: tuple[int, ...], drop: int | None = None) -> Decimal:
        hs = self.hidden_acts(x)
        return sum((vj * hj for j, (vj, hj) in enumerate(zip(self.v, hs)) if j != drop), self.c)

    def truth_table(self, drop: int | None = None) -> list[int]:
        return [int(self.logit(x, drop) > 0) for x in INPUTS]

    def copy(self) -> "Net":
        return Net([list(r) for r in self.W1], list(self.b1), list(self.v), self.c)

    def to_json(self) -> dict:
        return {"W1": [[str(w) for w in r] for r in self.W1], "b1": [str(b) for b in self.b1],
                "v": [str(v) for v in self.v], "c": str(self.c)}

    @classmethod
    def from_json(cls, d: dict) -> "Net":
        return cls([[Decimal(str(w)) for w in r] for r in d["W1"]], [Decimal(str(b)) for b in d["b1"]],
                   [Decimal(str(v)) for v in d["v"]], Decimal(str(d["c"])))


def N(x: tuple[int, ...]) -> int:
    return sum(b << i for i, b in enumerate(x))


# --- the function library: definition + a reference construction (units are (weights, bias, out_weight)) ---

ONES = [1] * N_IN
BIN = [1, 2, 4, 8, 16]


def _u(w, b, v):
    return ([Decimal(a) for a in w], Decimal(b), Decimal(v))


def _ladder(w, coeffs):
    """sum_k coeffs[k] * relu(w.x - k): a piecewise-linear function of the integer w.x."""
    return [_u(w, -k, a) for k, a in enumerate(coeffs) if a]


def _abs(w):
    return [_u(w, 0, 1), _u([-a for a in w], 0, 1)]


def _minterm(n):
    p = [(n >> i) & 1 for i in range(N_IN)]
    return _u([1 if b else -1 for b in p], 1 - sum(p), 1)


D_HL = [-1, -2, 0, 1, 2]  # (2*x4 + x3) - (2*x1 + x0)
D_MOD3 = [1, -1, 1, -1, 1]  # N mod 3 == (x0 - x1 + x2 - x3 + x4) mod 3

FUNCTIONS: dict[str, tuple[str, callable, list, str]] = {
    # name: (description, definition, reference units, reference c)
    "all_ones": ("all five bits are 1", lambda x: sum(x) == 5, _ladder(ONES, [0, 0, 0, 0, 1]), "-0.5"),
    "any_one": ("at least one bit is 1", lambda x: sum(x) >= 1, _ladder(ONES, [1]), "-0.5"),
    "majority": ("at least three of the five bits are 1", lambda x: sum(x) >= 3, _ladder(ONES, [0, 0, 1]), "-0.5"),
    "exactly_two": ("exactly two bits are 1", lambda x: sum(x) == 2, _ladder(ONES, [0, 1, -2, 1]), "-0.5"),
    "one_hot": ("exactly one bit is 1", lambda x: sum(x) == 1, _ladder(ONES, [1, -2, 1]), "-0.5"),
    "parity_odd": ("an odd number of bits are 1", lambda x: sum(x) % 2 == 1, _ladder(ONES, [1, -2, 2, -2, 2]), "-0.5"),
    "popcount_prime": ("the number of 1 bits is prime (2, 3 or 5)", lambda x: sum(x) in (2, 3, 5),
                       _ladder(ONES, [0, 1, -1, -1, 2]), "-0.5"),
    "x0_xor_x4": ("x0 XOR x4 (the other bits are ignored)", lambda x: x[0] != x[4], _abs([1, 0, 0, 0, -1]), "-0.5"),
    "x2_and_not_x0": ("x2 AND NOT x0 (the other bits are ignored)", lambda x: x[2] == 1 and x[0] == 0,
                      [_u([-1, 0, 1, 0, 0], 0, 1)], "-0.5"),
    "high_gt_low": ("the 2-bit number (x4 x3) is greater than the 2-bit number (x1 x0), i.e. 2*x4+x3 > 2*x1+x0",
                    lambda x: 2 * x[4] + x[3] > 2 * x[1] + x[0], [_u(D_HL, 0, 1)], "-0.5"),
    "high_eq_low": ("the 2-bit number (x4 x3) equals the 2-bit number (x1 x0)",
                    lambda x: 2 * x[4] + x[3] == 2 * x[1] + x[0], [_u(w, b, -1) for w, b, _ in _abs(D_HL)], "0.5"),
    "palindrome": ("the bit string is a palindrome: x0 = x4 and x1 = x3",
                   lambda x: x[0] == x[4] and x[1] == x[3],
                   [_u(w, b, -1) for w, b, _ in _abs([1, 0, 0, 0, -1]) + _abs([0, 1, 0, -1, 0])], "0.5"),
    "n_ge_20": ("N >= 20", lambda x: N(x) >= 20, [_u(BIN, -19, 1)], "-0.5"),
    "n_in_10_20": ("10 <= N <= 20", lambda x: 10 <= N(x) <= 20,
                   [_u(BIN, -9, 1), _u(BIN, -10, -1), _u(BIN, -20, -1), _u(BIN, -21, 1)], "-0.5"),
    "div_by_3": ("N is divisible by 3 (0 counts)", lambda x: N(x) % 3 == 0,
                 [_u(D_MOD3, 1, 1), _u(D_MOD3, 0, -2), _u(D_MOD3, -1, 1), _u(D_MOD3, -2, 1), _u(D_MOD3, -3, -2)],
                 "-0.5"),
    "perfect_square": ("N is a perfect square (0, 1, 4, 9, 16 or 25)", lambda x: N(x) in (0, 1, 4, 9, 16, 25),
                       [_minterm(n) for n in (0, 1, 4, 9, 16, 25)], "-0.5"),
}


def target_table(name: str) -> list[int]:
    return [int(FUNCTIONS[name][1](x)) for x in INPUTS]


def reference_net(name: str) -> Net:
    _, _, units, c = FUNCTIONS[name]
    return Net([list(w) for w, _, _ in units], [b for _, b, _ in units], [v for _, _, v in units], Decimal(c))


# --- obfuscation: function-preserving rewrites that hide the reference construction ---

def _r(rng: random.Random, lo: float, hi: float) -> Decimal:
    return Decimal(str(rng.uniform(lo, hi))).quantize(Q)


def _distractor(rng: random.Random) -> list[tuple[list[Decimal], Decimal, Decimal]]:
    kind = rng.choice(["dead", "dead", "tiny", "pair"])
    w = [_r(rng, -2, 2) for _ in range(N_IN)]
    if kind == "dead":  # never fires on a binary input
        b = -sum((a for a in w if a > 0), Decimal(0)) - _r(rng, 0.2, 1.5)
        return [(w, b, _r(rng, 0.5, 3) * rng.choice([1, -1]))]
    b = _r(rng, -1, 1)
    if kind == "tiny":  # fires, but its output weight is too small to matter
        return [(w, b, _r(rng, -0.02, 0.02))]
    v = _r(rng, 0.3, 1.5) * rng.choice([1, -1])  # two copies that cancel exactly
    return [(w, b, v), (list(w), b, -v)]


def obfuscate(net: Net, hidden: int, rng: random.Random) -> Net:
    """Split, rescale, pad with distractors, permute, jitter and round, keeping the truth table."""
    want = net.truth_table()
    for _ in range(2000):
        units = []
        for w, b, v in zip(net.W1, net.b1, net.v):
            if rng.random() < 0.35 and len(units) + 2 <= hidden:
                a = Decimal(str(rng.uniform(0.25, 0.75)))
                units += [(w, b, v * a), (w, b, v * (1 - a))]
            else:
                units.append((w, b, v))
        while len(units) < hidden:
            extra = _distractor(rng)
            if len(units) + len(extra) <= hidden:
                units += extra
        rng.shuffle(units)
        out_scale = Decimal(str(rng.uniform(0.5, 2.0)))
        W1, b1, vs = [], [], []
        for w, b, v in units:
            s = Decimal(str(rng.uniform(0.4, 2.5)))  # relu(s*z) = s*relu(z) for s > 0
            W1.append([(a * s + _r(rng, -0.01, 0.01)).quantize(Q) for a in w])
            b1.append((b * s + _r(rng, -0.01, 0.01)).quantize(Q))
            vs.append((v / s * out_scale).quantize(Q))
        c = (net.c * out_scale + _r(rng, -0.01, 0.01)).quantize(Q)
        out = Net(W1, b1, vs, c)
        # Every logit, with or without any single unit, stays clear of 0, so "does unit j matter" is never
        # decided by the jitter.
        margin = min(abs(out.logit(x, drop)) for x in INPUTS for drop in [None, *range(hidden)])
        if out.truth_table() == want and margin > Decimal("0.05"):
            return out
    raise RuntimeError("could not obfuscate network")


def functional_units(net: Net) -> list[int]:
    """Units whose removal (zeroing their output weight) changes the output on at least one input."""
    base = net.truth_table()
    return [j for j in range(net.hidden) if net.truth_table(drop=j) != base]


def accuracy(net: Net, name: str) -> Decimal:
    want = target_table(name)
    return Decimal(sum(a == b for a, b in zip(net.truth_table(), want))) / len(INPUTS)


# --- corruption for fix tasks ---

def corrupt(net: Net, name: str, rng: random.Random) -> tuple[Net, dict]:
    """Change one parameter of a functional unit so that at least 3 of the 32 outputs go wrong."""
    units = functional_units(net) or [j for j in range(net.hidden) if abs(net.v[j]) > Decimal("0.05")
                                      and any(net.hidden_acts(x)[j] > 0 for x in INPUTS)]
    for _ in range(500):
        j = rng.choice(units)
        kind = rng.choice(["W1", "W1", "b1", "v"])
        bad = net.copy()
        if kind == "W1":
            i = rng.choice([i for i in range(N_IN) if abs(net.W1[j][i]) > Decimal("0.3")] or [0])
            old = net.W1[j][i]
            new = -old if rng.random() < 0.5 else (old + _r(rng, 1, 3) * rng.choice([1, -1])).quantize(Q)
            bad.W1[j][i] = new
            loc = {"param": "W1", "row": j, "col": i}
        elif kind == "b1":
            old = net.b1[j]
            new = (old + _r(rng, 1, 3) * rng.choice([1, -1])).quantize(Q)
            bad.b1[j] = new
            loc = {"param": "b1", "row": j}
        else:
            old = net.v[j]
            new = -old
            bad.v[j] = new
            loc = {"param": "v", "row": j}
        wrong = sum(a != b for a, b in zip(bad.truth_table(), target_table(name)))
        if wrong >= 3 and new != old:
            return bad, {**loc, "original": str(old), "corrupted": str(new), "wrong_outputs": wrong}
    raise RuntimeError("could not corrupt network")


def apply_edits(net: Net, edits) -> Net | None:
    """Apply [{"param": "W1", "row": j, "col": i, "value": x}, ...]; None if any edit is malformed."""
    out = net.copy()
    if not isinstance(edits, list) or len(edits) > MAX_FIX_EDITS:
        return None
    try:
        for e in edits:
            p, value = e["param"], Decimal(str(e["value"]))
            if p == "c":
                out.c = value
                continue
            j = int(e["row"])
            if not 0 <= j < out.hidden:
                return None
            if p == "W1":
                i = int(e["col"])
                if not 0 <= i < N_IN:
                    return None
                out.W1[j][i] = value
            elif p == "b1":
                out.b1[j] = value
            elif p == "v":
                out.v[j] = value
            else:
                return None
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return None
    return out


def net_from_answer(d) -> Net | None:
    """A written network from the model's JSON; None if shapes or values are invalid."""
    try:
        W1, b1, v = d["W1"], d["b1"], d["v"]
        if not (1 <= len(W1) <= MAX_WRITE_HIDDEN and len(b1) == len(W1) == len(v)):
            return None
        if any(len(r) != N_IN for r in W1):
            return None
        net = Net.from_json({"W1": W1, "b1": b1, "v": v, "c": d["c"]})
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return None
    if any(abs(a) > 1000 for a in [*net.b1, *net.v, net.c, *(w for r in net.W1 for w in r)]):
        return None
    return net


# --- tasks ---

@dataclass
class Spec:
    seed: int
    mode: str  # read | fix | write
    function: str
    net: Net | None  # what the model sees (read, fix)
    expected: dict


def generate(seed: int, mode: str, function: str, hidden: int = 16) -> Spec:
    rng = random.Random(seed)
    if mode == "write":
        return Spec(seed, mode, function, None, {"function": function, "truth_table": target_table(function)})
    net = obfuscate(reference_net(function), hidden, rng)
    if mode == "read":
        return Spec(seed, mode, function, net, {"function": function, "functional_units": functional_units(net)})
    bad, loc = corrupt(net, function, rng)
    return Spec(seed, mode, function, bad, {"function": function, "corruption": loc, "clean": net.to_json()})


CONVENTIONS = """\
Network convention (identical for every task):
  inputs x0, x1, x2, x3, x4, each 0 or 1. N = x0 + 2*x1 + 4*x2 + 8*x3 + 16*x4 (x4 is the most significant bit).
  hidden unit j:  h_j = max(0, W1[j][0]*x0 + W1[j][1]*x1 + W1[j][2]*x2 + W1[j][3]*x3 + W1[j][4]*x4 + b1[j])
  output logit:   y = sum_j v[j]*h_j + c
  the network outputs 1 if y > 0, otherwise 0."""


def render_net(net: Net) -> str:
    lines = ["unit |   W1[j][0]   W1[j][1]   W1[j][2]   W1[j][3]   W1[j][4] |      b1[j] |       v[j]"]
    for j in range(net.hidden):
        ws = " ".join(f"{w:>10}" for w in net.W1[j])
        lines.append(f"{j:>4} | {ws} | {net.b1[j]:>10} | {net.v[j]:>10}")
    lines.append(f"c = {net.c}")
    return "\n".join(lines)


def function_menu() -> str:
    return "\n".join(f"  {name}: {desc}" for name, (desc, *_rest) in FUNCTIONS.items())


def render_prompt(spec: Spec) -> str:
    if spec.mode == "read":
        return f"""READ. Below are the raw weights of a trained-looking {spec.net.hidden}-unit network. It computes exactly one of the
functions listed. Work out which one by reasoning about the weights, and find which hidden units it actually depends on.

{CONVENTIONS}

Candidate functions:
{function_menu()}

A hidden unit is *functional* if setting its output weight v[j] to 0 changes the network's output on at least one of the
32 inputs. Some units may never fire, may barely matter, or may only matter in combination.

Weights:
{render_net(spec.net)}

Answer with JSON: {{"function": "<name from the list>", "functional_units": [<unit indices>]}}"""
    desc = FUNCTIONS[spec.function][0]
    if spec.mode == "fix":
        return f"""FIX. This {spec.net.hidden}-unit network was built to output 1 exactly when: {desc}.
Exactly one of its parameters (one entry of W1, b1 or v) was overwritten, and it now gets some inputs wrong.
Repair it with at most {MAX_FIX_EDITS} edits so it computes the intended function on all 32 inputs.

{CONVENTIONS}

Weights:
{render_net(spec.net)}

Each edit sets one parameter: {{"param": "W1", "row": j, "col": i, "value": x}}, {{"param": "b1", "row": j, "value": x}},
{{"param": "v", "row": j, "value": x}} or {{"param": "c", "value": x}}.
Answer with JSON: {{"edits": [<edit>, ...]}}"""
    return f"""WRITE. Write, from scratch, the weights of a network that outputs 1 exactly when: {desc}.
Use at most {MAX_WRITE_HIDDEN} hidden units. The network must be exactly right on all 32 inputs. You cannot run it.

{CONVENTIONS}

Answer with JSON: {{"W1": [[w0, w1, w2, w3, w4], ...], "b1": [...], "v": [...], "c": <number>}}
(one W1 row, one b1 entry and one v entry per hidden unit)."""


def spec_from_prompt(text: str) -> tuple[str, Net | None, str | None]:
    """(mode, network shown, function description) recovered from a prompt; used by the reference policies."""
    mode = text.split(".", 1)[0].strip().lower()
    net = None
    if "Weights:\n" in text:
        rows = text.split("Weights:\n", 1)[1].splitlines()[1:]
        W1, b1, v = [], [], []
        for row in rows:
            if row.startswith("c = "):
                c = Decimal(row[4:])
                break
            _, ws, b, vv = row.split("|")
            W1.append([Decimal(w) for w in ws.split()])
            b1.append(Decimal(b))
            v.append(Decimal(vv))
        net = Net(W1, b1, v, c)
    desc = None
    if mode != "read":
        desc = text.split("output 1 exactly when: ", 1)[1].split(".\n", 1)[0]
    return mode, net, desc
