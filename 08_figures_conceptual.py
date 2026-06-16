"""
08_figures_conceptual.py  —  Fig 5 (LURE mechanism) + Fig 6 (RHOA flowchart)
"""

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config as C

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42})
JPG_DPI = 300   # 300 DPI minimum for publication-quality output

def savejpg(fig, path):
    fig.savefig(str(path), dpi=JPG_DPI, bbox_inches="tight",
                format="jpg", pil_kwargs={"quality": 95})
    plt.close(fig)
    print(f"  Saved: {path.name}")

def box(ax, cx, cy, w, h, text, fc="#D6EAF8", ec="#1A5276",
        fs=9, bold=False, va="center"):
    from matplotlib.patches import FancyBboxPatch
    p = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                       boxstyle="round,pad=0.03",
                       facecolor=fc, edgecolor=ec, linewidth=1.3, zorder=2)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va=va, fontsize=fs,
            fontweight="bold" if bold else "normal",
            multialignment="center", zorder=3, wrap=True)

def arrow(ax, x1, y1, x2, y2, col="#1A5276"):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="-|>", color=col,
                                lw=1.5, mutation_scale=16), zorder=1)

def diamond(ax, cx, cy, w, h, text, fc="#FEF9E7", ec="#7D6608", fs=9):
    import numpy as np
    poly = plt.Polygon([[cx,cy+h/2],[cx+w/2,cy],[cx,cy-h/2],[cx-w/2,cy]],
                       closed=True, facecolor=fc, edgecolor=ec,
                       linewidth=1.3, zorder=2)
    ax.add_patch(poly)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            multialignment="center", zorder=3)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 – LURE Causal Pathway
# ─────────────────────────────────────────────────────────────────────────────
def fig5_mechanism():
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.set_xlim(0, 10); ax.set_ylim(0, 14); ax.axis("off")

    # Title
    ax.text(5, 13.55, "LURE Causal Pathway to ORB Verdict Reversal",
            ha="center", va="center", fontsize=13, fontweight="bold")

    # Trigger box
    box(ax, 5, 12.8, 8.5, 0.9,
        "LURE Trigger:  Uncontrolled Global RNG State ω\n"
        "(OS entropy at time of ORB assessment — varies each run)",
        fc="#FADBD8", ec="#C0392B", fs=10, bold=True)

    stages = [
        (11.4, "FM-1  Hidden System State",
         "Global RNG state ω varies run-to-run\n"
         "OS entropy, library init order, prior operations\n"
         "NOT captured by fold-level seeds",
         "#EBF5FB", "#1A5276"),
        (9.8,  "FM-2  Mask Geometry Coupling",
         "ω propagates to MCAR mask via NumPy global state\n"
         "Different ω  →  different missing-position geometry\n"
         "per fold, even with identical fold-level seeds",
         "#EBF5FB", "#1A5276"),
        (8.2,  "FM-3  Feature Availability Perturbation",
         "Different mask geometry  →  different feature availability\n"
         "per fold  →  different imputation residuals\n"
         "→  different effective training distribution",
         "#EBF5FB", "#1A5276"),
        (6.6,  "FM-4  Model Margin Erosion",
         "XGBoost native missing-value handler creates differential\n"
         "advantage under specific mask geometries\n"
         "When margin < ε (≈ 0.003 AUROC), mask determines winner",
         "#FEF9E7", "#7D6608"),
        (5.0,  "FM-5  ORB Verdict Reversal",
         "Different mask geometry  →  different flip_pct\n"
         "→  different ORB verdict\n"
         "→  different reliability certification",
         "#FDEDEC", "#C0392B"),
    ]

    prev_y = 12.35
    for y, title, desc, fc, ec in stages:
        box(ax, 5, y, 8.5, 1.25, f"{title}\n{desc}",
            fc=fc, ec=ec, fs=9)
        arrow(ax, 5, prev_y - 0.45, 5, y + 0.63)
        prev_y = y

    # Outcome
    box(ax, 5, 3.0, 8.5, 1.1,
        "OUTCOME:  Same pipeline + same data + same environment\n"
        "→  50% ORB verdict reversal  |  4× failure onset divergence\n"
        "[Run A: ORB = 0.70   vs   Run B: ORB = 0.05  (Split S1)]",
        fc="#FADBD8", ec="#C0392B", fs=10, bold=True)
    arrow(ax, 5, prev_y - 0.63, 5, 3.55, col="#C0392B")

    # Zero-stress callout
    box(ax, 8.8, 6.0, 2.2, 1.9,
        "Zero-Stress\nControl:\nδ = 0 agreement\n= 100%\n(LURE is\ninteraction\neffect)",
        fc="#E8F8F5", ec="#1E8449", fs=8.5)
    arrow(ax, 7.74, 6.0, 8.69, 6.0, col="#1E8449")

    savejpg(fig, C.OUT_FIGURES / "Fig5_LURE_causal_pathway.jpg")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 – RHOA Protocol Flowchart
# ─────────────────────────────────────────────────────────────────────────────
def fig6_rhoa():
    fig, ax = plt.subplots(figsize=(10, 15))
    ax.set_xlim(0, 10); ax.set_ylim(0, 15); ax.axis("off")

    ax.text(5, 14.55, "RHOA Protocol: RNG-Hardened ORB Assessment",
            ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(5, 14.1, "(5-step Procedural Standard for Certifiable ORB Estimation)",
            ha="center", va="center", fontsize=10, style="italic")

    steps = [
        (13.1, "RHOA-1  Global State Declaration",
         "Set  random.seed(N),  np.random.seed(N),  torch.manual_seed(N)\n"
         "before ANY data operation.\n"
         "N must appear in paper and system documentation.",
         "#D5F5E3", "#1E8449"),
        (11.3, "RHOA-2  Perturbation RNG Isolation",
         "Use  np.random.default_rng(seed)  instances for degradation\n"
         "injection.  NEVER share state with the global RNG.\n"
         "Fold-level:  rng = default_rng(seed + fold_offset)",
         "#D5F5E3", "#1E8449"),
        (9.5,  "RHOA-3  Multi-State Sensitivity Audit",
         "Run ORB assessment with ≥ 3 distinct global seeds.\n"
         "If verdicts differ:  report ORB as interval\n"
         "[min_ORB, max_ORB],  not a point estimate.",
         "#FEF9E7", "#7D6608"),
        (7.7,  "RHOA-4  Certification Reporting Standard",
         "ORB certification MUST state:\n"
         "global seed value, NumPy version, OS platform,\n"
         "Python version, RHOA compliance flag (True / False).",
         "#EBF5FB", "#1A5276"),
        (5.9,  "RHOA-5  Deployment Uncertainty Flag",
         "If RHOA-3 interval crosses expected operational stress range:\n"
         "→  classify as 'ORB-uncertain'\n"
         "→  require additional testing before deployment authorization.",
         "#FDEDEC", "#C0392B"),
    ]

    prev_y = 13.75
    for y, title, desc, fc, ec in steps:
        box(ax, 5, y, 9.2, 1.55, f"{title}\n{desc}", fc=fc, ec=ec, fs=9)
        if prev_y < 14:
            arrow(ax, 5, prev_y - 0.78, 5, y + 0.78)
        prev_y = y

    # Decision diamond
    diamond(ax, 5, 4.1, 5.0, 1.35,
            "RHOA-3 interval\nspans operational\nstress range?",
            fs=9)
    arrow(ax, 5, 5.12, 5, 4.78)

    # No → Certify
    box(ax, 2.0, 2.4, 3.2, 0.9,
        "CERTIFY\nORB = point estimate\n(RHOA-compliant)",
        fc="#D5F5E3", ec="#1E8449", fs=9)
    ax.annotate("No", xy=(2.0, 2.85), xytext=(3.3, 4.1),
                fontsize=9, color="#1E8449", ha="center",
                arrowprops=dict(arrowstyle="-|>", color="#1E8449", lw=1.2))

    # Yes → Uncertain
    box(ax, 8.0, 2.4, 3.2, 0.9,
        "ORB-UNCERTAIN\nAdditional testing\nrequired",
        fc="#FDEDEC", ec="#C0392B", fs=9)
    ax.annotate("Yes", xy=(8.0, 2.85), xytext=(6.7, 4.1),
                fontsize=9, color="#C0392B", ha="center",
                arrowprops=dict(arrowstyle="-|>", color="#C0392B", lw=1.2))

    # Footer
    ax.text(5, 1.3,
            "RHOA compliance eliminates LURE as a source of ORB certification error.\n"
            "Protocol ensures ORB estimates are reproducible, auditable, and deployable.",
            ha="center", va="center", fontsize=9, style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F2F3F4",
                      edgecolor="grey", linewidth=0.8))

    savejpg(fig, C.OUT_FIGURES / "Fig6_RHOA_protocol_flowchart.jpg")


def main():
    C.OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    figs = [("Fig5 LURE Mechanism", fig5_mechanism),
            ("Fig6 RHOA Flowchart",   fig6_rhoa)]
    for name, fn in tqdm(figs, desc="Conceptual figures"):
        fn()
    print(f"\n[08] Done.")

if __name__ == "__main__":
    main()
