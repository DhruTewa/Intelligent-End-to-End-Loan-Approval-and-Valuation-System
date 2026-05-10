"""
Generate the system architecture diagram and save it to assets/architecture.png.

Run from the project root:
    python scripts/generate_architecture.py
"""

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ASSETS_DIR = ROOT / "assets"
ASSETS_DIR.mkdir(exist_ok=True)
OUTPUT = ASSETS_DIR / "architecture.png"

# ── colour palette ────────────────────────────────────────────────────────────
C = {
    "data":    "#DBEAFE",   # light blue
    "prep":    "#D1FAE5",   # light green
    "model":   "#EDE9FE",   # light purple
    "app":     "#FEF3C7",   # light amber
    "output":  "#FFE4E6",   # light red
    "border":  "#1E3A5F",   # dark navy
    "arrow":   "#374151",
    "text":    "#111827",
    "sub":     "#4B5563",
}


def box(ax, x, y, w, h, label, sublabel="", color="white",
        fontsize=11, subfontsize=9, radius=0.02):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        linewidth=1.5, edgecolor=C["border"], facecolor=color, zorder=3,
    )
    ax.add_patch(patch)
    y_text = y + 0.012 if sublabel else y
    ax.text(x, y_text, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=C["text"], zorder=4)
    if sublabel:
        ax.text(x, y - 0.025, sublabel, ha="center", va="center",
                fontsize=subfontsize, color=C["sub"], zorder=4)


def arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=C["arrow"],
                        lw=1.8, mutation_scale=14),
        zorder=2,
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.015, my, label, fontsize=8, color=C["sub"],
                va="center", zorder=5)


def layer_bg(ax, y_center, height, label, color):
    rect = mpatches.FancyBboxPatch(
        (0.02, y_center - height / 2), 0.96, height,
        boxstyle="round,pad=0.005,rounding_size=0.01",
        linewidth=1, edgecolor="#CBD5E1", facecolor=color,
        alpha=0.35, zorder=1,
    )
    ax.add_patch(rect)
    ax.text(0.035, y_center + height / 2 - 0.022, label,
            fontsize=8, color="#6B7280", style="italic", zorder=2)


# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(13, 9))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor("white")

# title
ax.text(0.5, 0.965, "Intelligent Loan Approval & Valuation System",
        ha="center", va="center", fontsize=15, fontweight="bold",
        color=C["border"])
ax.text(0.5, 0.94, "System Architecture",
        ha="center", va="center", fontsize=10, color=C["sub"])

# ── layer backgrounds ─────────────────────────────────────────────────────────
layer_bg(ax, 0.855, 0.09, "Data Layer",         "#EFF6FF")
layer_bg(ax, 0.690, 0.14, "Preprocessing Layer","#F0FDF4")
layer_bg(ax, 0.490, 0.19, "Model Layer",         "#F5F3FF")
layer_bg(ax, 0.245, 0.19, "Application Layer",   "#FFFBEB")
layer_bg(ax, 0.075, 0.09, "Output Layer",        "#FFF1F2")

# ── Data Layer ────────────────────────────────────────────────────────────────
box(ax, 0.5, 0.855, 0.36, 0.065,
    "loan_approval_dataset.csv",
    "4,269 rows · 13 columns", C["data"], fontsize=10)

# ── Preprocessing Layer ───────────────────────────────────────────────────────
steps = [
    ("1. Encode\nCategoricals", 0.12),
    ("2. Feature\nEngineering",  0.31),
    ("3. Outlier\nCapping (IQR)",0.50),
    ("4. Log\nTransform",        0.69),
    ("5. Standard\nScaler",      0.88),
]
py = 0.690
for label, xfrac in steps:
    box(ax, xfrac, py, 0.155, 0.095, label,
        color=C["prep"], fontsize=8.5)

# thin arrows between preprocessing steps
for i in range(len(steps) - 1):
    x1 = steps[i][1] + 0.078
    x2 = steps[i + 1][1] - 0.078
    arrow(ax, x1, py, x2, py)

# arrow: data → preprocessing
arrow(ax, 0.5, 0.812, 0.5, 0.738)
ax.text(0.515, 0.775, "raw CSV", fontsize=8, color=C["sub"])

# ── Model Layer ───────────────────────────────────────────────────────────────
box(ax, 0.28, 0.50, 0.38, 0.10,
    "RF Classifier",
    "Predicts Approval / Rejection  +  Probability",
    C["model"], fontsize=10)

box(ax, 0.75, 0.50, 0.38, 0.10,
    "RF Regressor",
    "Predicts Approved Loan Amount  (₹)",
    C["model"], fontsize=10)

# arrow: preprocessing → classifier
arrow(ax, 0.5, 0.643, 0.28, 0.552)
ax.text(0.34, 0.603, "14 features", fontsize=8, color=C["sub"])

# arrow: classifier → regressor (conditional)
arrow(ax, 0.47, 0.50, 0.56, 0.50)
ax.text(0.485, 0.513, "if Approved\n12 features", fontsize=7.5,
        color=C["sub"], ha="center")

# ── Application Layer ─────────────────────────────────────────────────────────
pages = [
    ("Home",       "Overview\n& navigation", 0.15),
    ("Predict",    "Input form\n→ result",    0.38),
    ("Dashboard",  "Metrics &\ncharts",       0.62),
    ("About",      "Docs &\narchitecture",    0.85),
]
ay = 0.255
for title, sub, xf in pages:
    box(ax, xf, ay, 0.19, 0.105, title, sub, C["app"], fontsize=10)

# brace label: Streamlit Web App
ax.text(0.5, 0.355, "Streamlit Web App",
        ha="center", fontsize=9, color=C["sub"], style="italic")

# arrows: models → predict page
arrow(ax, 0.28, 0.448, 0.38, 0.310)
arrow(ax, 0.75, 0.448, 0.38, 0.310)

# arrows: models → dashboard page
arrow(ax, 0.28, 0.448, 0.62, 0.310)

# ── Output Layer ──────────────────────────────────────────────────────────────
box(ax, 0.35, 0.075, 0.30, 0.065,
    "Approve / Reject  +  Probability",
    color=C["output"], fontsize=9.5)
box(ax, 0.72, 0.075, 0.30, 0.065,
    "Predicted Loan Amount (₹)",
    color=C["output"], fontsize=9.5)

# arrows: predict page → outputs
arrow(ax, 0.38, 0.203, 0.35, 0.108)
arrow(ax, 0.38, 0.203, 0.72, 0.108)

# ── module labels ─────────────────────────────────────────────────────────────
ax.text(0.5, 0.643,  "src/preprocessing.py",
        ha="center", fontsize=7.5, color="#6B7280",
        bbox=dict(fc="white", ec="#E5E7EB", pad=2, boxstyle="round"))
ax.text(0.28, 0.418, "src/predict.py",
        ha="center", fontsize=7.5, color="#6B7280",
        bbox=dict(fc="white", ec="#E5E7EB", pad=2, boxstyle="round"))

plt.tight_layout(pad=0.3)
plt.savefig(OUTPUT, dpi=160, bbox_inches="tight", facecolor="white")
plt.close()

print(f"Architecture diagram saved → {OUTPUT}")
