"""Render the arch-kernels-v1 cards for X, in the same three-panel style as the weight-space cards.

Writes HTML and PNG to results/arch-kernels-v1/cards/:
  1_x86_vs_arm.png  THROUGHPUT / BIGGEST GAP / MEMORY: the same C on AMD EPYC (x86) and NVIDIA Arm
  2_native.png      WRITE / PORT / SCORE: Claude writes for the chip it is told about, and that code does not port

Usage: python scripts/make_arch_infographics.py results/arch-kernels-v1/compare.json <run_dir> ...
"""

import json
import math
import re
import sys
from pathlib import Path
from statistics import mean, median

import make_wsp_infographics as W
from arch_kernels_v1.kernels import KERNELS
from make_wsp_infographics import GRAY, GREEN, GREEN_TXT, HL, RED, bars, page, panel

from summarize import load

W.OUT = Path(__file__).resolve().parent.parent / "results" / "arch-kernels-v1" / "cards"
X86, ARM = "#6b6f76", GREEN  # x86 in gray, Arm in green throughout
NATIVE = re.compile(r"immintrin|x86intrin|nmmintrin|wmmintrin|arm_neon|arm_acle|arm_sve|__builtin_ia32|__crc32")
SHORT = {"dot_f32": "dot", "prefix_sum_i64": "prefix sum", "byte_histogram": "histogram", "count_primes": "primes",
         "matmul_f32": "matmul", "sort_u32": "sort", "count_substr": "substring", "popcount_u64": "popcount",
         "count_distinct_u32": "distinct", "transpose_f32": "transpose", "crc32_ieee": "CRC-32",
         "levenshtein": "edit distance", "mandel_count": "mandelbrot"}


def best_times(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Fastest passing model answer per kernel and machine, in ms."""
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        if r["source"] in ("naive", "tuned") or not r["result"].get("passed"):
            continue
        t = r["result"]["sol_s"] * 1e3
        out.setdefault(r["kernel"], {})
        out[r["kernel"]][r["arch"]] = min(t, out[r["kernel"]].get(r["arch"], math.inf))
    return out


def diverging(ratios: list[tuple[str, float]]) -> str:
    """One row per kernel: bar right (green) = faster on Arm, left (gray) = faster on x86; log scale, ±3x."""
    Wd, H = 460, 262
    mid, half, top, rh = 268, 170, 10, 18.6
    out = [f'<line x1="{mid}" y1="4" x2="{mid}" y2="{H - 4}" stroke="#bbb" stroke-width="1.5"/>']
    for k, (name, ratio) in enumerate(sorted(ratios, key=lambda x: x[1])):
        y = top + k * rh
        L = max(-1.0, min(1.0, math.log(ratio) / math.log(3))) * half
        x, w = (mid, L) if L >= 0 else (mid + L, -L)
        color = ARM if L >= 0 else X86
        out.append(f'<rect x="{x:.1f}" y="{y + 2:.1f}" width="{max(w, 2):.1f}" height="{rh - 5:.1f}" fill="{color}"/>')
        out.append(f'<text x="{(mid - 8) if L >= 0 else (mid + 8)}" y="{y + 14:.1f}" font-size="14.5" fill="#444" '
                   f'text-anchor="{"end" if L >= 0 else "start"}">{name}</text>')
    out.append(f'<text x="{mid - half}" y="{H - 2}" font-size="13" fill="{X86}">◀ x86 faster</text>')
    out.append(f'<text x="{mid + half}" y="{H - 2}" font-size="13" fill="{GREEN_TXT}" text-anchor="end">Arm faster ▶</text>')
    return f'<svg width="100%" height="100%" viewBox="0 0 {Wd} {H}" style="font-family:Source Sans 3">{"".join(out)}</svg>'


def code_rows(lines: list[tuple[str, str, str, str]]) -> str:
    """(chip, code, time, background) rows: the instruction each machine's best answer is built on."""
    rows = "".join(
        f'<div style="display:flex;align-items:center;gap:12px;margin:18px 0">'
        + (f'<span style="width:44px;font-size:21px;color:#666">{chip}</span>' if chip else "")
        + f'<span class="mono" style="font-size:17px;padding:8px 10px;border-radius:6px;background:{bg};flex:1;'
        f'white-space:nowrap">{code}</span>'
        + (f'<span style="font-size:23px;font-weight:700;white-space:nowrap">{t}</span>' if t else "")
        + "</div>"
        for chip, code, t, bg in lines)
    return f'<div style="padding:34px 20px">{rows}</div>'


def dots(passed: int, total: int) -> str:
    """One cell per answer: green = builds on the other chip, hatched red = does not."""
    Wd, H = 460, 262
    cols, cw, gap = 13, 28, 6
    x0 = (Wd - (cols * cw + (cols - 1) * gap)) / 2
    y0 = (H - (3 * cw + 2 * gap)) / 2
    out = ['<defs><pattern id="h2" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
           f'<line x1="0" y1="0" x2="0" y2="8" stroke="{RED}" stroke-width="3" opacity=".55"/></pattern></defs>']
    for n in range(total):
        x, y = x0 + (n % cols) * (cw + gap), y0 + (n // cols) * (cw + gap)
        fill = GREEN if n < passed else "url(#h2)"
        out.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{cw}" rx="4" fill="{fill}"'
                   f'{"" if n < passed else f" stroke=\"{RED}\" stroke-opacity=\".5\""}/>')
    return f'<svg width="100%" height="100%" viewBox="0 0 {Wd} {H}">{"".join(out)}</svg>'


def card_arch(data: dict) -> str:
    rows = data["rows"]
    best = best_times(rows)
    ratios = [(SHORT[k], best[k]["amd64"] / best[k]["arm64"]) for k in KERNELS if len(best.get(k, {})) == 2]
    arm_wins = sum(r > 1 for _, r in ratios)
    crc = best["crc32_ieee"]
    peaks = {a: {k: median([r["result"]["peak_kb"] for r in rows if r["kernel"] == k and r["arch"] == a
                            and "peak_kb" in r["result"]]) for k in KERNELS} for a in ("amd64", "arm64")}
    diff = median(peaks["arm64"][k] - peaks["amd64"][k] for k in KERNELS)
    rel = mean((peaks["arm64"][k] - peaks["amd64"][k]) / peaks["amd64"][k] for k in KERNELS
               if peaks["amd64"][k] > 50_000)
    return page(
        panel("THROUGHPUT", "Same C, 2 vCPUs each: AMD EPYC (x86) vs NVIDIA Arm. Which runs it faster?",
              diverging(ratios),
              "Claude's fastest answer per kernel, timed on both. Neither chip wins everywhere",
              f"{arm_wins} / {len(ratios)}", "kernels faster on Arm"),
        panel("BIGGEST GAP", "Write the fastest CRC-32 you can for this machine.",
              code_rows([("Arm", "__crc32d(c, word)", f"{crc['arm64']:.2f} ms", HL),
                         ("x86", "_mm_clmulepi64_si128", f"{crc['amd64']:.2f} ms", "#efefef")]),
              "On Arm, Claude uses the chip's own CRC instruction",
              f"{crc['amd64'] / crc['arm64']:.1f}×", "faster on Arm"),
        panel("MEMORY", "How much memory does the same program use on each chip?",
              bars([("x86", 1.0, "100%", X86), ("Arm", 1 + rel, f"{(1 + rel) * 100:.1f}%", ARM)], 1.0),
              "Same data layout, same bytes. Arm processes just start a little lighter",
              f"{diff / 1024:+.2f} MB".replace("-", "−"), "per process, on every kernel"))


def card_native(data: dict, runs: dict[str, list[dict]]) -> str:
    rows = data["rows"]
    son_x86 = [r for r in rows if r["source"] == "claude-sonnet-5-5" and r["written_for"] == "amd64"
               and r["arch"] == "arm64"]
    built = sum(r["result"].get("compiled", 0) for r in son_x86)
    native = [r for k, rs in runs.items() if k.startswith("claude-sonnet") for r in rs]
    share = mean(bool(NATIVE.search(r["text"])) for r in native)
    scores = {k: mean(r["reward"] for r in rs) for k, rs in runs.items()}
    sx, sa = scores["claude-sonnet-5-5 amd64"], scores["claude-sonnet-5-5 arm64"]
    hx, ha = scores["claude-haiku-5-5 amd64"], scores["claude-haiku-5-5 arm64"]
    return page(
        panel("WRITE", "Implement crc32_ieee as fast as you can for AArch64 Linux.",
              code_rows([("", "#include &lt;arm_acle.h&gt;", "", HL),
                         ("", "c = __crc32d(c, w);", "", HL),
                         ("", "c = __crc32b(c, *p++);", "", "#fafafa")]),
              "Told the target chip, Sonnet writes intrinsics for that chip",
              f"{share * 100:.0f}%", "of Sonnet's answers are chip-specific"),
        panel("PORT", "Now build Sonnet's x86 answers on the Arm machine.",
              dots(built, len(son_x86)),
              "The rest name x86 headers and instructions that Arm does not have",
              f"{built} / {len(son_x86)}", "even compile"),
        panel("SCORE", "Score on 13 kernels, 3 tries each (1 = correct and 8× faster).",
              bars([("Haiku", hx, f"{hx:.2f}", X86), ("Haiku", ha, f"{ha:.2f}", ARM),
                    ("Sonnet", sx, f"{sx:.2f}", X86), ("Sonnet", sa, f"{sa:.2f}", ARM)], 1.0),
              "Gray x86, green Arm. Sonnet's x86 answers: 39 / 39 correct",
              f"{max(sx, sa):.2f}", "best run, Sonnet 5.5"))


if __name__ == "__main__":
    DATA = json.loads(Path(sys.argv[1]).read_text())
    RUNS = {}
    for d in sys.argv[2:]:
        rows = load(Path(d))
        model = json.loads(next(iter((Path(d) / "traces.jsonl").open())))["traces"][0]["agent"]["config"]["model"]
        RUNS[f"{model} {rows[0]['task']['arch']}"] = rows
    W.render("1_x86_vs_arm", card_arch(DATA))
    W.render("2_native", card_native(DATA, RUNS))
