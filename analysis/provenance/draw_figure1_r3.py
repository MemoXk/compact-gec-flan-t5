"""Figure 1 for the round-3 paper: one row of five stages, large type (smallest text about 10 pt when the figure is
printed at the full text width of 17.2 cm). Content follows the paper's primary system exactly.
Output: figures/figure1_r3.{png,pdf}"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(r"<PROJECT>/figures"); OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "Segoe UI"})
WIN, HIN = 6.77, 2.55                       # inches: printed at the 17.2 cm text width, so 1 pt here = 1 pt in print
fig = plt.figure(figsize=(WIN, HIN), dpi=400); ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100); ax.set_ylim(0, 37.7); ax.axis("off")

COL = [("#2f9e44", "#eef8ef"), ("#3b5bdb", "#eef2ff"), ("#6741d9", "#f3efff"), ("#e8590c", "#fff3ea"), ("#343a40", "#f1f3f5")]
PANELS = [
    ("1  Synthetic data",
     ["C4-200M sample:", "1.15M training pairs", "Composite", "filtering policy", "+15% identity pairs"]),
    ("2  Stage 1",
     ["FLAN-T5-small", "(77M parameters)", "Instruction prefix", "1 epoch, 10,333 steps", "Selection: val. loss"]),
    ("3  Stage 2",
     ["W&I+LOCNESS", "learner pairs", "Essay-level split:", "32,314 train / 1,623", "holdout, 3 seeds"]),
    ("4  Decoding",
     ["Beam 5, length", "penalty 0.7", "Guarded refinement", "Revert edits with", "margin below \u03c4"]),
    ("5  Evaluation",
     ["Primary: BEA-2019", "blind test", "BEA dev, CoNLL,", "JFLEG, in-domain", "Blinded human ratings"]),
]
BW, GAP, X0, Y0, BH = 18.2, 1.9, 0.8, 1.2, 35.2
for k, (title, lines) in enumerate(PANELS):
    x = X0 + k * (BW + GAP); ec, fc = COL[k]
    ax.add_patch(FancyBboxPatch((x, Y0), BW, BH, boxstyle="round,pad=0,rounding_size=1.6", ec=ec, fc=fc, lw=1.3, zorder=1))
    ax.add_patch(FancyBboxPatch((x, Y0 + BH - 7.0), BW, 7.0, boxstyle="round,pad=0,rounding_size=1.6", ec=ec, fc=ec, lw=1.3, zorder=2))
    ax.text(x + BW / 2, Y0 + BH - 3.5, title, ha="center", va="center", fontsize=10, color="white", fontweight="bold", zorder=3)
    for i, s in enumerate(lines):
        ax.text(x + BW / 2, Y0 + BH - 11.6 - i * 5.0, s, ha="center", va="center", fontsize=8.0, color="#212529", zorder=3)
    if k < len(PANELS) - 1:
        xa, xb = x + BW + 0.1, x + BW + GAP - 0.1
        ax.add_patch(FancyArrowPatch((xa, Y0 + BH / 2), (xb, Y0 + BH / 2), arrowstyle="-|>", mutation_scale=15,
                                     color="#343a40", lw=1.8, zorder=4))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"figure1_r3.{ext}", dpi=400, facecolor="white")
print("wrote", OUT / "figure1_r3.png")
