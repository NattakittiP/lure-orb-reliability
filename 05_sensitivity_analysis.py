"""
05_sensitivity_analysis.py
ORB threshold sensitivity grid (25 combos) + seed convergence analysis
"""

import sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config as C

plt.rcParams.update({"font.family":C.IEEE_FONT,"font.size":C.IEEE_FONTSIZE,"pdf.fonttype":42})


def recompute_orb(env_df, stress_col, flip_thr, tau_thr):
    df = env_df.copy()
    df["verdict"] = ((df["winner_flip_pct"]<=flip_thr) & (df["kendall_tau_mean"]>=tau_thr)).astype(int)
    return df[["split",stress_col,"verdict"]]


def sensitivity_grid(phase, stress_col):
    p = C.P4A if phase == "4A" else C.P4B
    env_A = pd.read_csv(p["A"]["envelope"]); env_B = pd.read_csv(p["B"]["envelope"])
    combos = [(f, t) for f in C.FLIP_THR_GRID for t in C.TAU_THR_GRID]
    rows = []
    for flip_thr, tau_thr in tqdm(combos, desc=f"  Phase {phase} threshold grid", leave=False):
        vA = recompute_orb(env_A, stress_col, flip_thr, tau_thr)
        vB = recompute_orb(env_B, stress_col, flip_thr, tau_thr)
        m  = vA.merge(vB, on=["split",stress_col], suffixes=("_A","_B"))
        n_total = len(m)
        n_rev = int(((m["verdict_A"]==1)&(m["verdict_B"]==0)).sum())
        rows.append({"phase":phase,"flip_thr":flip_thr,"tau_thr":tau_thr,
                     "n_total":n_total,"n_reversal":n_rev,
                     "reversal_pct":100.0*n_rev/n_total,
                     "agreement_pct":100.0*(m["verdict_A"]==m["verdict_B"]).sum()/n_total})
    return pd.DataFrame(rows)


def seed_convergence(phase, stress_col):
    p = C.P4A if phase == "4A" else C.P4B
    rows = []
    for run in ["A","B"]:
        winners = pd.read_csv(p[run]["winners"])
        base_rate = winners[stress_col].min()
        combos = [(s, r) for s in C.SPLITS for r in sorted(winners[stress_col].unique())]
        for split, rate in tqdm(combos, desc=f"  Convergence Phase {phase} Run {run}", leave=False):
            bw_s = winners[(winners["split"]==split)&(winners[stress_col]==base_rate)]["winner_model"].mode()
            bw = bw_s.iloc[0] if len(bw_s)>0 else "xgb"
            r = winners[(winners["split"]==split)&(winners[stress_col]==rate)].sort_values("seed").reset_index(drop=True)
            flips = (r["winner_model"]!=bw).astype(float).values
            for n in range(1, len(flips)+1):
                rows.append({"phase":phase,"run":run,"split":split,stress_col:rate,
                              "n_seeds":n,"rolling_flip_pct":float(np.mean(flips[:n])*100)})
    return pd.DataFrame(rows)


def plot_sensitivity_heatmap(df, phase):
    pivot = df[df["phase"]==phase].pivot_table(index="tau_thr", columns="flip_thr", values="reversal_pct")
    fig, ax = plt.subplots(figsize=(C.IEEE_SINGLE_COL_INCH*1.8, 2.8), dpi=C.IEEE_DPI)
    im = ax.imshow(pivot.values, cmap=plt.cm.RdYlGn_r, aspect="auto", vmin=0, vmax=60, origin="lower")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels([f"{v:.0f}%" for v in pivot.columns], fontsize=7)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels([f"{v:.2f}" for v in pivot.index], fontsize=7)
    ax.set_xlabel("Flip Threshold $\\tau_F$", fontsize=8); ax.set_ylabel("Kendall $\\tau_K$", fontsize=8)
    ax.set_title(f"Phase {phase}: ORB Reversal Rate (%) Across Threshold Grid", fontsize=7.5)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i,j]
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=6,
                    color="white" if val>35 else "black")
    plt.colorbar(im, ax=ax, label="Reversal Rate (%)"); plt.tight_layout()
    ab = "a" if phase == "4A" else "b"
    out = C.OUT_FIGURES/f"FigS2{ab}_sensitivity_heatmap_Phase{phase}.jpg"
    plt.savefig(out, dpi=C.IEEE_DPI, bbox_inches="tight", format="jpg"); plt.close(); print(f"    Saved: {out}")


def plot_convergence(df, phase, stress_col):
    key_rates = [0.10,0.30,0.50,0.70] if phase=="4A" else [0.1,0.2,0.3,0.5]
    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728"]
    fig, axes = plt.subplots(1, 2, figsize=(C.IEEE_DOUBLE_COL_INCH, 2.8), dpi=C.IEEE_DPI)
    for ax, split in zip(axes, C.SPLITS):
        for run, ls in [("A","--"),("B","-")]:
            sub = df[(df["phase"]==phase)&(df["split"]==split)&(df["run"]==run)]
            for rate, col in zip(key_rates, colors):
                r = sub[sub[stress_col]==rate].sort_values("n_seeds")
                if r.empty: continue
                ax.plot(r["n_seeds"], r["rolling_flip_pct"], linestyle=ls, color=col,
                        linewidth=1.2, alpha=0.85, label=f"$\\delta={rate:.2f}$" if run=="B" else None)
        ax.axhline(C.DEFAULT_FLIP_THR, color="grey", linestyle=":", linewidth=1)
        ax.set_xlabel("N Seeds", fontsize=8); ax.set_ylabel("Cumulative Flip Rate (%)", fontsize=8)
        ax.set_title(f"{split} — Solid=Run B, Dashed=Run A", fontsize=8)
        ax.set_xlim(1,20); ax.set_ylim(-2,70); ax.legend(fontsize=6); ax.grid(True,alpha=0.3)
    plt.suptitle(f"Phase {phase}: Flip Rate Convergence with Increasing Seeds", fontsize=8)
    plt.tight_layout()
    ab = "a" if phase == "4A" else "b"
    out = C.OUT_FIGURES/f"FigS3{ab}_ORB_convergence_analysis_Phase{phase}.jpg"
    plt.savefig(out, dpi=C.IEEE_DPI, bbox_inches="tight", format="jpg"); plt.close(); print(f"    Saved: {out}")


def main():
    C.check_data()
    C.OUT_STATS.mkdir(parents=True, exist_ok=True)
    C.OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    all_grid, all_conv = [], []
    for phase, stress_col in tqdm([("4A","miss_rate"),("4B","target_prev")], desc="Phases"):
        print(f"\nPhase {phase} Sensitivity Analysis")
        grid_df = sensitivity_grid(phase, stress_col); all_grid.append(grid_df)
        default = grid_df[(grid_df["flip_thr"]==C.DEFAULT_FLIP_THR)&(grid_df["tau_thr"]==C.DEFAULT_TAU_THR)]
        print(f"  Default thresholds: {int(default['n_reversal'].values[0])}/{int(default['n_total'].values[0])} reversals")
        print(f"  All {len(grid_df)} combos: range [{grid_df['n_reversal'].min()},{grid_df['n_reversal'].max()}]")

        conv_df = seed_convergence(phase, stress_col); all_conv.append(conv_df)
        for run in ["A","B"]:
            sub = conv_df[(conv_df["phase"]==phase)&(conv_df["run"]==run)]
            std = sub[sub["n_seeds"]>=15].groupby([stress_col,"split"])["rolling_flip_pct"].std().mean()
            print(f"  Run {run}: avg std at N>=15 = {std:.3f}% (converged)")

        plot_sensitivity_heatmap(grid_df, phase)
        plot_convergence(conv_df, phase, stress_col)

    pd.concat(all_grid).to_csv(C.OUT_STATS/"sensitivity_orb_threshold_grid.csv", index=False)
    pd.concat(all_conv).to_csv(C.OUT_STATS/"sensitivity_convergence_analysis.csv", index=False)
    print(f"\n[05] Done.")

if __name__ == "__main__":
    main()
