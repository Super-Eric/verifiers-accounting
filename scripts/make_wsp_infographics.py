"""Render the weight-space-v1 cards for X in the three-panel style of the Weight-Space Programming post's teaser.

Writes HTML to results/weight-space-v1/cards/ and screenshots it with headless Chrome (1600x920 @2x):
  1_difference.png  READ / FIX / WRITE: where Haiku 5.5 and Sonnet 5.5 differ
  2_win.png         WRITE / CHECK / COMPARE: divisible-by-3 in fewer units than the answer key
  3_fail.png        READ / CHECK / ANSWER: Haiku reads the right units, then names the wrong function

Usage: python scripts/make_wsp_infographics.py <haiku_run_dir> <sonnet_run_dir>
"""

import subprocess
import sys
from pathlib import Path
from statistics import mean

from weight_space_v1.nets import Net, reference_net, target_table
from weight_space_v1.taskset import parse_answer

from summarize import load

OUT = Path(__file__).resolve().parent.parent / "results" / "weight-space-v1" / "cards"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

GREEN, GREEN_TXT, HL, GRAY, RED = "#3b8a5a", "#2e6b45", "#cfe8d6", "#8b949e", "#d0453b"

CSS = f"""
* {{ box-sizing:border-box; margin:0; padding:0; }}
html, body {{ width:1600px; height:920px; background:#fff; color:#1f1f1f; font-family:'Source Sans 3', sans-serif; }}
.row {{ display:flex; gap:34px; padding:30px 26px; height:920px; }}
.panel {{ flex:1 1 0; min-width:0; border:1.5px solid #d0d0d0; padding:34px 34px 30px; display:flex; flex-direction:column; }}
.lab {{ font-size:23px; font-weight:700; letter-spacing:2.6px; margin-bottom:22px; }}
.ask {{ position:relative; border:1.5px solid #cfcfcf; border-radius:30px; padding:20px 70px 22px 26px; font-size:26px;
  line-height:1.5; min-height:162px; box-shadow:0 1px 2px rgba(0,0,0,.06); }}
.send {{ position:absolute; right:18px; bottom:18px; width:46px; height:46px; border-radius:50%; background:#111;
  display:flex; align-items:center; justify-content:center; }}
.vis {{ margin-top:24px; height:262px; border:1.5px solid #e2e2e2; position:relative; overflow:hidden; }}
.cap {{ color:{GREEN_TXT}; font-size:26px; line-height:1.55; margin-top:30px; min-height:82px; }}
.res {{ margin-top:auto; display:flex; align-items:baseline; gap:14px; }}
.big {{ font-size:50px; font-weight:700; letter-spacing:-0.5px; white-space:nowrap; }}
.small {{ font-size:24px; color:#2b2b2b; }}
.mono {{ font-family:'JetBrains Mono', monospace; }}
.w {{ font-family:'JetBrains Mono', monospace; font-size:21px; line-height:35px; white-space:nowrap; padding:8px 14px;
  background:#fafafa; height:100%; }}
.w span.c {{ display:inline-block; padding:0 4px; margin-right:6px; }}
.w span.h {{ background:{HL}; }}
.w .fade {{ opacity:.35; }}
.w .u {{ color:#8a8a8a; display:inline-block; width:92px; }}
"""

ARROW = ('<div class="send"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.6" '
         'stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg></div>')


def panel(label: str, ask: str, vis: str, cap: str, big: str, small: str) -> str:
    return f"""<div class="panel"><div class="lab">{label}</div>
<div class="ask">{ask}{ARROW}</div><div class="vis">{vis}</div>
<div class="cap">{cap}</div><div class="res"><span class="big">{big}</span><span class="small">{small}</span></div></div>"""


def page(*panels: str) -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Source+Sans+3:wght@400;600;700&display=block" rel="stylesheet">
<style>{CSS}</style></head><body><div class="row">{"".join(panels)}</div></body></html>"""


def render(name: str, doc: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    src = OUT / f"{name}.html"
    src.write_text(doc)
    subprocess.run([CHROME, "--headless=new", "--hide-scrollbars", "--force-device-scale-factor=2",
                    "--window-size=1600,920", "--virtual-time-budget=8000", f"--screenshot={OUT / f'{name}.png'}",
                    src.as_uri()], check=True, capture_output=True)
    print("wrote", OUT / f"{name}.png")


# --- visuals ---

def weights(net: Net, units: list[int], hl: set[int], fade: set[int] = frozenset()) -> str:
    """Raw weight rows, highlighted like the post's teaser; first/last rows faded as if the list continues."""
    lines = []
    for j in units:
        cells = "".join(f'<span class="c{" h" if j in hl else ""}">{net.W1[j][i]:.3f}</span>' for i in range(5))
        lines.append(f'<div class="{"fade" if j in fade else ""}"><span class="u">unit {j}</span>{cells}</div>')
    return f'<div class="w">{"".join(lines)}</div>'


def int_weights(W1: list[list[int]], b1: list[int]) -> str:
    lines = []
    for j, row in enumerate(W1):
        cells = "".join(f'<span class="c h">{w:+d}</span>' for w in row)
        lines.append(f'<div><span class="u">unit {j}</span>{cells}'
                     f'<span class="c" style="color:#8a8a8a">b {b1[j]:+d}</span></div>')
    return f'<div class="w" style="font-size:22px;line-height:44px;padding-top:28px">{"".join(lines)}</div>'


def bars(items: list[tuple[str, float, str, str]], vmax: float) -> str:
    """Bars like the teaser's 'base vs Astra's edit': (label, value, value text, color)."""
    W, H, base = 460, 262, 214
    slot = W / len(items)
    out = []
    for k, (label, v, txt, color) in enumerate(items):
        bw = min(118, slot * 0.62)
        x = k * slot + (slot - bw) / 2
        h = max(4, (base - 46) * v / vmax)
        out.append(f'<rect x="{x:.0f}" y="{base - h:.0f}" width="{bw:.0f}" height="{h:.0f}" fill="{color}"/>'
                   f'<text x="{x + bw / 2:.0f}" y="{base - h - 12:.0f}" text-anchor="middle" font-size="24" '
                   f'font-weight="700">{txt}</text>'
                   f'<text x="{x + bw / 2:.0f}" y="{base + 32}" text-anchor="middle" font-size="23">{label}</text>')
    return f'<svg width="100%" height="100%" viewBox="0 0 {W} {H}" style="font-family:Source Sans 3">{"".join(out)}</svg>'


def grid(fill: dict[int, str], hatch: set[int] = frozenset(), outline: set[int] = frozenset()) -> str:
    """The 32 inputs N = 0..31 as a 4x8 grid of cells."""
    W, H = 460, 262
    cw, ch, gx, gy = 50, 50, 6, 10
    x0, y0 = (W - (8 * cw + 7 * gx)) / 2, (H - (4 * ch + 3 * gy)) / 2
    out = ['<defs><pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
           f'<line x1="0" y1="0" x2="0" y2="8" stroke="{RED}" stroke-width="3" opacity=".55"/></pattern></defs>']
    for n in range(32):
        x, y = x0 + (n % 8) * (cw + gx), y0 + (n // 8) * (ch + gy)
        out.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="4" fill="{fill.get(n, "#f1f1f1")}"/>')
        if n in hatch:
            out.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="4" fill="url(#hatch)"/>')
        if n in outline:
            out.append(f'<rect x="{x - 2}" y="{y - 2}" width="{cw + 4}" height="{ch + 4}" rx="6" fill="none" '
                       f'stroke="#111" stroke-width="3.5"/>')
        color = "#fff" if fill.get(n) in (GREEN, "#111") else "#333"
        out.append(f'<text x="{x + cw / 2}" y="{y + ch / 2 + 8}" text-anchor="middle" font-size="22" '
                   f'font-weight="{700 if n in fill else 400}" fill="{color}">{n}</text>')
    return f'<svg width="100%" height="100%" viewBox="0 0 {W} {H}" style="font-family:Source Sans 3">{"".join(out)}</svg>'


def network(broken: tuple[int, int], twin: tuple[int, int]) -> str:
    """5 inputs -> 6 hidden -> 1 output, with the overwritten edge (red) and the repaired edge on its twin (green)."""
    W, H = 460, 262
    ins = [(60, 40 + k * 46) for k in range(5)]
    hid = [(230, 30 + k * 41) for k in range(6)]
    out_n = (400, 131)
    e = [f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="#dcdcdc" stroke-width="1.4"/>'
         for a in ins for b in hid]
    e += [f'<line x1="{b[0]}" y1="{b[1]}" x2="{out_n[0]}" y2="{out_n[1]}" stroke="#dcdcdc" stroke-width="1.4"/>' for b in hid]
    (bi, bh), (ti, th) = broken, twin
    a, b = ins[bi], hid[bh]
    e.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="{RED}" stroke-width="4" stroke-dasharray="9 6"/>')
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    e.append(f'<g stroke="{RED}" stroke-width="4" stroke-linecap="round"><line x1="{mx - 9}" y1="{my - 9}" x2="{mx + 9}" '
             f'y2="{my + 9}"/><line x1="{mx + 9}" y1="{my - 9}" x2="{mx - 9}" y2="{my + 9}"/></g>')
    a, b = ins[ti], hid[th]
    e.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="{GREEN}" stroke-width="5"/>')
    nodes = [f'<circle cx="{x}" cy="{y}" r="13" fill="#fff" stroke="#9a9a9a" stroke-width="2"/>' for x, y in ins + hid]
    nodes += [f'<text x="{x - 24}" y="{y + 7}" text-anchor="end" font-size="19" fill="#666">x{k}</text>'
              for k, (x, y) in enumerate(ins)]
    nodes.append(f'<circle cx="{hid[bh][0]}" cy="{hid[bh][1]}" r="13" fill="#fbe3e1" stroke="{RED}" stroke-width="2.5"/>')
    nodes.append(f'<circle cx="{hid[th][0]}" cy="{hid[th][1]}" r="13" fill="{HL}" stroke="{GREEN}" stroke-width="2.5"/>')
    nodes.append(f'<circle cx="{out_n[0]}" cy="{out_n[1]}" r="16" fill="#111"/>')
    nodes.append(f'<text x="{hid[bh][0] + 22}" y="{hid[bh][1] + 7}" font-size="19" fill="{RED}">broken</text>')
    nodes.append(f'<text x="{hid[th][0] + 22}" y="{hid[th][1] + 7}" font-size="19" fill="{GREEN_TXT}">its twin</text>')
    return f'<svg width="100%" height="100%" viewBox="0 0 {W} {H}" style="font-family:Source Sans 3">{"".join(e + nodes)}</svg>'


def chips(wrong: str, right: str, gloss: str) -> str:
    def chip(who, name, style):
        return (f'<div style="display:flex;align-items:center;gap:16px"><span style="font-size:21px;color:#666;width:80px">'
                f'{who}</span><span class="mono" style="font-size:25px;padding:8px 16px;border-radius:6px;{style}">{name}</span></div>')
    return (f'<div style="display:flex;flex-direction:column;justify-content:center;gap:24px;height:100%;padding:0 28px">'
            + chip("Haiku", wrong, f"border:2px solid {RED};color:{RED};text-decoration:line-through")
            + chip("truth", right, f"background:{HL};color:{GREEN_TXT}")
            + f'<div style="font-size:21px;color:#555;padding-left:96px">{gloss}</div></div>')


# --- data helpers ---

def rate(rows, mode, key="solved"):
    return mean(r[key] for r in rows if r["task"]["mode"] == mode)


def first(rows, mode, fn):
    return next(r for r in rows if r["task"]["mode"] == mode and r["task"]["function"] == fn)


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


# --- cards ---

def card_difference(h, s) -> str:
    read_net = Net.from_json(first(s, "read", "popcount_prime")["task"]["net"])
    return page(
        panel("READ", "Which function does this network compute, and which units does it use?",
              weights(read_net, [1, 2, 3, 5, 8, 13, 14], hl={2, 3, 5, 8, 13}, fade={1, 14}),
              "Both name the function. Sonnet also finds exactly the units that matter",
              pct(rate(s, "read")), f"exact, vs {pct(rate(h, 'read'))} Haiku"),
        panel("FIX", "One weight was overwritten. Repair the network in two edits or fewer.",
              network(broken=(4, 1), twin=(4, 4)),
              f"A third of Sonnet's fixes leave the broken weight alone and repair around it",
              pct(rate(s, "fix")), f"Sonnet, vs {pct(rate(h, 'fix'))} Haiku"),
        panel("WRITE", "Write a network that outputs 1 when N is divisible by 3.",
              bars([("baseline", 0.0, "0%", GRAY), ("Haiku", rate(h, "write"), pct(rate(h, "write")), GREEN),
                    ("Sonnet", rate(s, "write"), pct(rate(s, "write")), GREEN)], 1.0),
              "Both write every network from scratch, with no training",
              "48 / 48", "functions, both models"))


def card_win(h, s) -> str:
    ans = parse_answer(first(s, "write", "div_by_3")["text"])
    net = Net.from_json({k: ans[k] for k in ("W1", "b1", "v", "c")})
    table = net.truth_table()
    runs = [parse_answer(r["text"]) for r in s if r["task"]["mode"] == "write" and r["task"]["function"] == "div_by_3"]
    same = sum(len(a["b1"]) == len(ans["b1"]) for a in runs)
    key, units = reference_net("div_by_3").hidden, len(ans["b1"])
    right = sum(a == b for a, b in zip(table, target_table("div_by_3")))
    return page(
        panel("WRITE", "Write a network that outputs 1 when N is divisible by 3.",
              int_weights(ans["W1"], ans["b1"]),
              "Sonnet spots that N mod 3 = x0 − x1 + x2 − x3 + x4",
              f"{units} units", "in one reply"),
        panel("CHECK", "Run it on all 32 inputs, N = 0 to 31.",
              grid({n: GREEN for n in range(32) if table[n]}),
              "It fires on exactly the multiples of 3",
              f"{right} / 32", "inputs right"),
        panel("COMPARE", "How does that compare with our hand-built answer key?",
              bars([("our key", key, f"{key} units", GRAY), ("Sonnet", units, f"{units} units", GREEN)], key),
              f"The same {units} units in {same} of {len(runs)} attempts",
              f"{units} vs {key}", "ReLU units"))


def card_fail(h) -> str:
    r = next(r for r in h if r["task"]["mode"] == "read" and r["task"]["function"] == "popcount_prime"
             and (parse_answer(r["text"]) or {}).get("function") == "n_ge_20")
    net = Net.from_json(r["task"]["net"])
    ans = parse_answer(r["text"])
    gold = r["task"]["expected"]["functional_units"]
    truth, guess = target_table("popcount_prime"), target_table("n_ge_20")
    checked = {0, 16, 20, 21, 31}  # the inputs Haiku's written reasoning checks
    tell = {n for n in range(32) if truth[n] != guess[n]}
    return page(
        panel("READ", "Which function does this network compute, and which units does it use?",
              weights(net, [1, 2, 3, 5, 8, 13, 14], hl={2, 3, 5, 8, 13}, fade={1, 14}),
              "Haiku spots the units that count 1-bits",
              f"{len(set(ans['functional_units']) & set(gold))} / {len(gold)}", "units, exactly right"),
        panel("CHECK", "Before answering, Haiku checks five inputs (black).",
              grid({n: "#111" for n in checked}, hatch=tell),
              f"All five fit both answers. The {len(tell)} red inputs would have told them apart",
              f"0 / {len(tell)}", "telling inputs checked"),
        panel("ANSWER", "So which function is it?",
              chips(ans["function"], "popcount_prime", "“N ≥ 20” vs “number of 1-bits is prime”"),
              "It names the wrong function and scores 0.40 of 1",
              f'<span style="color:{RED}">✗</span>', "wrong function"))


if __name__ == "__main__":
    H_ROWS, S_ROWS = load(Path(sys.argv[1])), load(Path(sys.argv[2]))
    render("1_difference", card_difference(H_ROWS, S_ROWS))
    render("2_win", card_win(H_ROWS, S_ROWS))
    render("3_fail", card_fail(H_ROWS))
