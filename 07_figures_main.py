"""
07_figures_main.py  —  Publication figures (clean layout, .jpg output)
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config as C

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "font.size"        : 11,
    "axes.titlesize"   : 11,
    "axes.labelsize"   : 11,
    "xtick.labelsize"  : 10,
    "ytick.labelsize"  : 10,
    "legend.fontsize"  : 9,
    "axes.linewidth"   : 0.9,
    "grid.alpha"       : 0.35,
    "pdf.fonttype"     : 42,
    "ps.fonttype"      : 42,
})

JPG_DPI    = 300   # 300 DPI minimum for publication-quality output
COLOR_A    = "#1A5276"
COLOR_B    = "#C0392B"
COLOR_REL  = "#1E8449"
COLOR_UNREL= "#C0392B"
LS_A, LS_B = "--", "-"
MK_A, MK_B = "o", "s"
MS          = 6

def savejpg(fig, path):
    fig.savefig(str(path), dpi=JPG_DPI, bbox_inches="tight",
                format="jpg", pil_kwargs={"quality": 95})
    plt.close(fig)
    print(f"  Saved: {path.name}")

def load_env(run, phase):
    p = C.P4A[run]["envelope"] if phase == "4A" else C.P4B[run]["envelope"]
    df = pd.read_csv(p)
    df["stable"] = df["stable_under_thresholds"].map(
        {True:1, False:0, "True":1, "False":0})
    return df

def load_win(run, phase):
    p = C.P4A[run]["winners"] if phase == "4A" else C.P4B[run]["winners"]
    return pd.read_csv(p)

def load_sum(run, phase):
    p = C.P4A[run]["summary"] if phase == "4A" else C.P4B[run]["summary"]
    return pd.read_csv(p)

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 – ORB Verdict Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def fig1_heatmap(phase):
    sc = "miss_rate" if phase == "4A" else "target_prev"
    xl = "Stress Level δ (MCAR Rate)" if phase == "4A" else "Target Prevalence π"

    eA = load_env("A", phase)
    eB = load_env("B", phase)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                             constrained_layout=True)

    cmap = matplotlib.colors.ListedColormap([COLOR_UNREL, COLOR_REL])
    norm = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    for ax, env, run in zip(axes, [eA, eB], ["A", "B"]):
        pivot = env.pivot_table(index="split", columns=sc, values="stable", aggfunc="first")
        ax.pcolormesh(pivot.values, cmap=cmap, norm=norm,
                      edgecolors="white", linewidth=1.5)
        ax.set_xticks(np.arange(len(pivot.columns)) + 0.5)
        ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)) + 0.5)
        ax.set_yticklabels(pivot.index.tolist(), fontsize=11)
        ax.set_xlabel(xl, fontsize=11, labelpad=6)
        ax.set_ylabel("Split Policy", fontsize=11)

        n_stable = int(pivot.values.sum())
        n_total  = pivot.values.size
        label = C.RUN_LABELS[run]
        ax.set_title(f"{label}\nRELIABLE: {n_stable}/{n_total}",
                     fontsize=12, fontweight="bold", pad=10)

    legend_elements = [
        mpatches.Patch(facecolor=COLOR_REL,   label="RELIABLE (ORB Certified)"),
        mpatches.Patch(facecolor=COLOR_UNREL, label="UNRELIABLE (ORB Failed)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, -0.08), frameon=True)

    ab = "a" if phase == "4A" else "b"
    savejpg(fig, C.OUT_FIGURES / f"Fig1{ab}_ORB_verdict_heatmap_Phase{phase}.jpg")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 – Failure Onset Curve
# ─────────────────────────────────────────────────────────────────────────────
def fig2_failure_onset(phase):
    sc = "miss_rate" if phase == "4A" else "target_prev"
    xl = "Stress Level δ (MCAR Rate)" if phase == "4A" else "Target Prevalence"

    eA = load_env("A", phase)
    eB = load_env("B", phase)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             constrained_layout=True, sharey=True)
    fig.suptitle("Fig. 2: ORB Failure Onset — LURE Divergence", fontsize=13, fontweight="bold")

    for ax, split in zip(axes, C.SPLITS):
        sA = eA[eA["split"] == split].sort_values(sc)
        sB = eB[eB["split"] == split].sort_values(sc)

        ax.plot(sA[sc], sA["winner_flip_pct"],
                color=COLOR_A, ls=LS_A, marker=MK_A, ms=MS, lw=1.8,
                label=C.RUN_LABELS["A"])
        ax.plot(sB[sc], sB["winner_flip_pct"],
                color=COLOR_B, ls=LS_B, marker=MK_B, ms=MS, lw=1.8,
                label=C.RUN_LABELS["B"])

        for env, run, col, y_off in [(sA,"A",COLOR_A,42),(sB,"B",COLOR_B,54)]:
            ff = env[env["winner_flip_pct"] > 0]
            if not ff.empty:
                d = float(ff.iloc[0][sc])
                ax.axvline(d, color=col, lw=1.0, ls=":", alpha=0.8)
                ax.text(d + 0.01, y_off, f"δ*={d:.2f}",
                        color=col, fontsize=9, va="center")

        ax.axhline(C.DEFAULT_FLIP_THR, color="grey", lw=1.0, ls="-.",
                   label=f"Threshold {C.DEFAULT_FLIP_THR}%")
        ax.set_xlabel(xl, fontsize=11)
        ax.set_ylabel("Winner Flip Rate (%)", fontsize=11)
        ax.set_title(f"Split Policy: {split}", fontsize=12, pad=8)
        ax.set_ylim(-3, 70)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
        ax.grid(True, alpha=0.35)

    ab = "a" if phase == "4A" else "b"
    savejpg(fig, C.OUT_FIGURES / f"Fig2{ab}_failure_onset_divergence_Phase{phase}.jpg")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 – Kendall τ Trajectory
# ─────────────────────────────────────────────────────────────────────────────
def fig3_kendall(phase):
    sc = "miss_rate" if phase == "4A" else "target_prev"
    xl = "Stress Level δ" if phase == "4A" else "Target Prevalence"

    eA = load_env("A", phase)
    eB = load_env("B", phase)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             constrained_layout=True, sharey=True)
    fig.suptitle("Fig. 3: Rank-Order Stability (Kendall τ) vs Stress Level",
                 fontsize=13, fontweight="bold")

    for ax, split in zip(axes, C.SPLITS):
        sA = eA[eA["split"] == split].sort_values(sc)
        sB = eB[eB["split"] == split].sort_values(sc)

        ax.plot(sA[sc], sA["kendall_tau_mean"],
                color=COLOR_A, ls=LS_A, marker=MK_A, ms=MS, lw=1.8,
                label=C.RUN_LABELS["A"])
        ax.plot(sB[sc], sB["kendall_tau_mean"],
                color=COLOR_B, ls=LS_B, marker=MK_B, ms=MS, lw=1.8,
                label=C.RUN_LABELS["B"])

        ax.axhline(C.DEFAULT_TAU_THR, color="grey", lw=1.0, ls="-.",
                   label=f"τ_K = {C.DEFAULT_TAU_THR} threshold")
        ax.fill_between(sB[sc], 0, C.DEFAULT_TAU_THR,
                        alpha=0.07, color=COLOR_UNREL)

        # Annotate min tau in Run B
        idx_min = sB["kendall_tau_mean"].idxmin()
        min_tau = sB.loc[idx_min, "kendall_tau_mean"]
        min_x   = float(sB.loc[idx_min, sc])
        ax.annotate(f"τ={min_tau:.3f}",
                    xy=(min_x, min_tau),
                    xytext=(min_x - 0.12, min_tau - 0.09),
                    fontsize=9, color=COLOR_B,
                    arrowprops=dict(arrowstyle="->", color=COLOR_B, lw=0.9))

        ax.set_xlabel(xl, fontsize=11)
        ax.set_ylabel("Kendall τ", fontsize=11)
        ax.set_title(f"Split Policy: {split}", fontsize=12, pad=8)
        ax.set_ylim(0.3, 1.08)
        ax.legend(loc="lower left", fontsize=9, framealpha=0.85)
        ax.grid(True, alpha=0.35)

    ab = "a" if phase == "4A" else "b"
    savejpg(fig, C.OUT_FIGURES / f"Fig3{ab}_kendall_tau_reliability_Phase{phase}.jpg")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 – Certified System Distribution
# ─────────────────────────────────────────────────────────────────────────────
def fig4_cert_dist(phase):
    sc = "miss_rate" if phase == "4A" else "target_prev"
    xl = "δ (MCAR Rate)" if phase == "4A" else "Target Prevalence"

    wA = load_win("A", phase)
    wB = load_win("B", phase)

    model_colors = {
        "xgb"           : "#1A5276",
        "rf"            : "#1E8449",
        "extratrees"    : "#7D3C98",
        "lr_l2"         : "#D35400",
        "svm_linear_cal": "#A93226",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle("Fig. 4: Certified System Distribution — Fragmentation Under LURE",
                 fontsize=13, fontweight="bold")

    pairs = [
        (axes[0,0], wA, "A", "S1"),
        (axes[0,1], wB, "B", "S1"),
        (axes[1,0], wA, "A", "S2"),
        (axes[1,1], wB, "B", "S2"),
    ]

    for ax, wdf, run, split in pairs:
        sub    = wdf[wdf["split"] == split]
        rates  = sorted(sub[sc].unique())
        x      = np.arange(len(rates))
        bottom = np.zeros(len(rates))

        for model in C.MODELS:
            pct = [
                (sub[sub[sc] == r]["winner_model"] == model).sum()
                / max(len(sub[sub[sc] == r]), 1) * 100
                for r in rates
            ]
            ax.bar(x, pct, bottom=bottom, label=model,
                   color=model_colors.get(model, "#7F8C8D"), alpha=0.88)
            bottom += np.array(pct)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{r:.2f}" for r in rates],
                           fontsize=9, rotation=45, ha="right")
        ax.set_xlabel(xl, fontsize=10, labelpad=4)
        ax.set_ylabel("Winner Share (%)", fontsize=10)
        ax.set_ylim(0, 108)
        ax.set_title(f"{split}  |  {C.RUN_LABELS[run]}", fontsize=11, pad=6)
        ax.grid(axis="y", alpha=0.3)
        if run == "B" and split == "S1":
            ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

    ab = "a" if phase == "4A" else "b"
    savejpg(fig, C.OUT_FIGURES / f"Fig4{ab}_ORB_certification_distribution_Phase{phase}.jpg")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 – Baseline Comparison at δ=0
# ─────────────────────────────────────────────────────────────────────────────
def fig7_baseline(phase):
    sc = "miss_rate" if phase == "4A" else "target_prev"

    wA = load_win("A", phase)
    wB = load_win("B", phase)
    base = wA[sc].min()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             constrained_layout=True)
    fig.suptitle("Fig. 7: Baseline Agreement at δ=0 — Causal Proof of LURE",
                 fontsize=13, fontweight="bold")

    for ax, split in zip(axes, C.SPLITS):
        aA = wA[(wA["split"]==split)&(wA[sc]==base)]["winner_auc"].values
        aB = wB[(wB["split"]==split)&(wB[sc]==base)]["winner_auc"].values
        n  = min(len(aA), len(aB))
        idx = np.arange(n)

        ax.scatter(idx, aA[:n], color=COLOR_A, marker=MK_A, s=40, alpha=0.8,
                   label=f"{C.RUN_LABELS['A']}  μ={aA.mean():.4f}")
        ax.scatter(idx, aB[:n], color=COLOR_B, marker=MK_B, s=40, alpha=0.8,
                   label=f"{C.RUN_LABELS['B']}  μ={aB.mean():.4f}")

        diff = abs(aA.mean() - aB.mean())
        ax.set_title(f"Split: {split}   |Δμ| = {diff:.4f} (non-significant)",
                     fontsize=11, pad=8)
        ax.set_xlabel("Seed Index", fontsize=11)
        ax.set_ylabel("Winner AUROC", fontsize=11)
        ax.legend(fontsize=9, framealpha=0.85)
        ax.grid(True, alpha=0.35)
        all_v = np.concatenate([aA, aB])
        mg = (all_v.max() - all_v.min()) * 0.4
        ax.set_ylim(all_v.min() - mg, all_v.max() + mg)

    ab = "a" if phase == "4A" else "b"
    savejpg(fig, C.OUT_FIGURES / f"Fig7{ab}_baseline_performance_comparison_Phase{phase}.jpg")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8 – Winner Margin Trajectory
# ─────────────────────────────────────────────────────────────────────────────
def fig8_margin(phase):
    sc = "miss_rate" if phase == "4A" else "target_prev"
    xl = "Stress Level δ" if phase == "4A" else "Target Prevalence"

    mfile = C.OUT_STATS / "conjecture1_winner_margins.csv"
    if not mfile.exists():
        print("  [Fig8] Run 04 first."); return

    df = pd.read_csv(mfile)
    df = df[df["phase"] == phase]
    agg = (df.groupby(["run","split",sc])["margin_auc"]
             .agg(["mean","std"]).reset_index())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             constrained_layout=True, sharey=True)
    fig.suptitle("Fig. 8: Winner Margin Trajectory — Margin Erosion Activates LURE",
                 fontsize=13, fontweight="bold")

    for ax, split in zip(axes, C.SPLITS):
        for run, col, ls, mk in [("A",COLOR_A,LS_A,MK_A),("B",COLOR_B,LS_B,MK_B)]:
            sub = agg[(agg["run"]==run)&(agg["split"]==split)].sort_values(sc)
            ax.plot(sub[sc], sub["mean"], color=col, ls=ls,
                    marker=mk, ms=MS, lw=1.8, label=C.RUN_LABELS[run])
            ax.fill_between(sub[sc], sub["mean"]-sub["std"],
                            sub["mean"]+sub["std"], color=col, alpha=0.12)

        ax.axhline(0, color="grey", lw=0.7, ls=":")
        ax.set_xlabel(xl, fontsize=11)
        ax.set_ylabel("Winner Margin (AUROC Rank1 − Rank2)", fontsize=11)
        ax.set_title(f"Split Policy: {split}", fontsize=12, pad=8)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
        ax.grid(True, alpha=0.35)

    ab = "a" if phase == "4A" else "b"
    savejpg(fig, C.OUT_FIGURES / f"Fig8{ab}_winner_margin_trajectory_Phase{phase}.jpg")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 9 – Bootstrap CI Ribbon
# ─────────────────────────────────────────────────────────────────────────────
def fig9_bootstrap(phase):
    sc = "miss_rate" if phase == "4A" else "target_prev"
    xl = "Stress Level δ" if phase == "4A" else "Target Prevalence"

    bfile = C.OUT_STATS / "bootstrap_ci_flip_rate.csv"
    if not bfile.exists():
        print("  [Fig9] Run 02 first."); return

    df = pd.read_csv(bfile)
    df = df[df["phase"] == phase]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             constrained_layout=True, sharey=True)
    fig.suptitle("Fig. 9: Bootstrap 95% CI — Non-Overlapping Intervals Confirm LURE",
                 fontsize=13, fontweight="bold")

    for ax, split in zip(axes, C.SPLITS):
        for run, col, ls in [("A",COLOR_A,LS_A),("B",COLOR_B,LS_B)]:
            sub = df[(df["run"]==run)&(df["split"]==split)].sort_values(sc)
            ax.plot(sub[sc], sub["flip_pct"], color=col, ls=ls, lw=1.8,
                    label=C.RUN_LABELS[run])
            ax.fill_between(sub[sc], sub["boot_ci_lo"], sub["boot_ci_hi"],
                            color=col, alpha=0.15)

        ax.axhline(C.DEFAULT_FLIP_THR, color="grey", lw=1.0, ls="-.",
                   label=f"Threshold {C.DEFAULT_FLIP_THR}%")
        ax.set_xlabel(xl, fontsize=11)
        ax.set_ylabel("Flip Rate (%) with 95% CI", fontsize=11)
        ax.set_title(f"Split Policy: {split}", fontsize=12, pad=8)
        ax.set_ylim(-3, 75)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
        ax.grid(True, alpha=0.35)

    ab = "a" if phase == "4A" else "b"
    savejpg(fig, C.OUT_FIGURES / f"Fig9{ab}_bootstrap_CI_flip_rate_Phase{phase}.jpg")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    C.check_data()
    C.OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    C.OUT_STATS.mkdir(parents=True, exist_ok=True)

    funcs = [fig1_heatmap, fig2_failure_onset, fig3_kendall,
             fig4_cert_dist, fig7_baseline, fig8_margin, fig9_bootstrap]

    for phase in tqdm(["4A","4B"], desc="Phases"):
        print(f"\nPhase {phase}")
        for fn in tqdm(funcs, desc=f"  Figures", leave=False):
            fn(phase)

    print(f"\n[07] Done — {C.OUT_FIGURES}")

if __name__ == "__main__":
    main()