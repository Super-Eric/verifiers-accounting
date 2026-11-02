"""Render results/infographic.png (1600x900, for social posts) from the published numbers.

Usage: python scripts/make_infographic.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
HAIKU, SONNET = "#2a78d6", "#eb6834"

ENVS = ["GPU\nright-sizing", "Sales &\nuse tax"]
REWARD = {"Haiku 5.5": [0.965, 0.795], "Sonnet 5.5": [0.972, 0.978]}
NAIVE = [0.346, 0.013]

MONEY = [
    ("", "Haiku 5.5", "Sonnet 5.5"),
    ("Savings figure exact to the cent", "83 / 90", "84 / 90"),
    ("Mean $ error per fleet (annual)", "$496", "$150"),
    ("Total $ error over 90 fleets", "$44,655 (1.3%)", "$13,534 (0.4%)"),
    ("Tax return exact to the cent", "60 / 90", "87 / 90"),
    ("Cost to run both benchmarks", "$0.58", "$7.73"),
]

plt.rcParams["text.parse_math"] = False
fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=SURFACE)
plt.rcParams["font.family"] = "DejaVu Sans"

fig.text(0.04, 0.925, "Can Claude right-size a GPU fleet to the cent?", fontsize=32, weight="bold", color=INK)
fig.text(0.04, 0.875, "New open RL environment built from our Alibaba GPU-trace research (ISAN 3392, Texas State), "
         "plus our tax benchmark. 30 tasks × 3 attempts per model.", fontsize=14, color=INK2)

# Left: reward by environment.
ax = fig.add_axes([0.13, 0.30, 0.34, 0.45], facecolor=SURFACE)
ys = [1.0, 0.0]
h = 0.3
for k, (name, color) in enumerate([("Haiku 5.5", HAIKU), ("Sonnet 5.5", SONNET)]):
    for y, v in zip(ys, REWARD[name]):
        yy = y + (h / 2 + 0.02) * (1 if k == 0 else -1)
        ax.barh(yy, v, height=h, color=color, edgecolor=SURFACE, linewidth=2)
        ax.text(v + 0.012, yy, f"{v:.3f}", va="center", fontsize=14, color=INK, weight="bold")
for y, v in zip(ys, NAIVE):
    ax.plot([v, v], [y - 0.42, y + 0.42], color=INK2, linewidth=2, linestyle=(0, (3, 2)))
    ax.text(v + 0.01, y - 0.47, f"naive baseline {v:.3f}", fontsize=11, color=INK2, va="top")
ax.set_yticks(ys, ENVS, fontsize=13, color=INK)
ax.set_xlim(0, 1.1)
ax.set_ylim(-0.75, 1.55)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.tick_params(axis="x", colors=MUTED, labelsize=11)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color=GRID, linewidth=1)
ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Reward (oracle = 1.000)", loc="left", fontsize=16, color=INK, weight="bold", pad=12)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=HAIKU), plt.Rectangle((0, 0), 1, 1, color=SONNET)],
          labels=["Claude Haiku 5.5", "Claude Sonnet 5.5"], loc="lower left", frameon=False, fontsize=12,
          bbox_to_anchor=(-0.02, -0.22), ncol=2)

# Right: the money table.
fig.text(0.53, 0.80, "The money", fontsize=16, weight="bold", color=INK)
y0, row_h = 0.74, 0.072
cols = [0.53, 0.75, 0.87]
for i, row in enumerate(MONEY):
    y = y0 - i * row_h
    if i > 0:
        fig.add_artist(plt.Line2D([0.53, 0.97], [y + row_h / 2, y + row_h / 2], color=GRID, linewidth=1))
    for j, cell in enumerate(row):
        weight = "bold" if i == 0 or j > 0 else "normal"
        color = INK if j > 0 or i == 0 else INK2
        fig.text(cols[j], y, cell, fontsize=13, va="center", weight=weight, color=color)
fig.add_artist(plt.Rectangle((0.739, y0 - 0.011), 0.006, 0.022, color=HAIKU, transform=fig.transFigure))
fig.add_artist(plt.Rectangle((0.859, y0 - 0.011), 0.006, 0.022, color=SONNET, transform=fig.transFigure))

# Bottom: headline tiles.
tiles = [
    ("$37.7K / yr", "idle GPU spend per 24-pod fleet\nwhen every pod keeps its full request"),
    ("99.6%+", "of pods given the exact right GPU slice\nby both models"),
    ("$1.8M–$7.2M / yr", "paper's estimate per 1,000 reserved\ninference GPUs (21% reclaimed)"),
    ("$8.31", "total API cost for all 360 Claude\nrollouts across both benchmarks"),
]
w, gap, x = 0.215, 0.012, 0.04
for big, small in tiles:
    fig.add_artist(FancyBboxPatch((x, 0.06), w, 0.15, boxstyle="round,pad=0,rounding_size=0.012",
                                  facecolor="#f2f1ed", edgecolor="none", transform=fig.transFigure))
    fig.text(x + 0.015, 0.165, big, fontsize=21, weight="bold", color=INK, va="center")
    fig.text(x + 0.015, 0.098, small, fontsize=11.5, color=INK2, va="center", linespacing=1.4)
    x += w + gap

fig.text(0.96, 0.025, "Prime Intellect Environments Hub: tarive/gpu-rightsizing-v1 · tarive/us-sut-reconciliation-v1",
         fontsize=11, color=MUTED, ha="right")
fig.savefig("results/infographic.png", facecolor=SURFACE)
print("wrote results/infographic.png")
