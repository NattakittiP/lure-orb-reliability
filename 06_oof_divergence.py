"""
06_oof_divergence.py  —  OOF Prediction Divergence (FM-2 -> FM-3)
"""

import sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config as C

plt.rcParams.update({"font.family":C.IEEE_FONT,"font.size":C.IEEE_FONTSIZE,"pdf.fonttype":42})
OOF_MODELS = ["extratrees","lr_l2","rf","svm_linear_cal","xgb"]


def load_oof(run, split, model=None):
    base = C.RUN_A_OOF_DIR if run == "A" else C.RUN_B_OOF_DIR
    fname = f"oof_P0_{split}" + (f"_{model}" if model else "") + ".npz"
    path = base / fname
    if not path.exists(): return None
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def main():
    C.OUT_STATS.mkdir(parents=True, exist_ok=True)
    C.OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    if not (C.RUN_A_OOF_DIR/"oof_P0_S1.npz").exists():
        print("[06] OOF files not found. Skipping."); return

    rows_per_model, rows_summary = [], []
    combos = [(s, m) for s in ["S1","S2"] for m in OOF_MODELS]

    fig, axes = plt.subplots(2, len(OOF_MODELS),
                              figsize=(C.IEEE_DOUBLE_COL_INCH, 3.5*2), dpi=C.IEEE_DPI)

    for si, (split, model) in enumerate(tqdm(combos, desc="OOF divergence per (split,model)")):
        s_idx = ["S1","S2"].index(split)
        m_idx = OOF_MODELS.index(model)
        ax = axes[s_idx, m_idx]

        oof_A = load_oof("A", split, model)
        oof_B = load_oof("B", split, model)
        if oof_A is None or oof_B is None:
            ax.text(0.5,0.5,"Not found",transform=ax.transAxes,ha="center",va="center",fontsize=7)
            continue

        pA, pB = oof_A["y_prob"], oof_B["y_prob"]
        y_true = oof_A["y_true"]
        div = np.abs(pA - pB)
        mean_div = float(np.mean(div)); max_div = float(np.max(div))
        rho_pos, _ = spearmanr(pA, div)
        r_true, _ = pearsonr(y_true.astype(float), div) if y_true.std()>0 else (np.nan, np.nan)

        rows_per_model.append({
            "split":split,"model":model,"n_samples":len(pA),
            "y_true_identical":np.array_equal(oof_A["y_true"],oof_B["y_true"]),
            "mean_abs_div":mean_div,"max_abs_div":max_div,
            "p90_abs_div":float(np.percentile(div,90)),
            "spearman_rho_prob_vs_div":rho_pos,"pearsonr_label_vs_div":r_true})

        ax.hist(div, bins=50, color=C.COLOR_RUN_B, alpha=0.7, density=True)
        ax.axvline(mean_div, color="black", lw=1.2, linestyle="--", label=f"Mean={mean_div:.3f}")
        ax.set_title(f"{split}/{model}", fontsize=7)
        ax.set_xlabel(r"$|p_A-p_B|$", fontsize=6)
        ax.set_ylabel("Density", fontsize=6)
        ax.tick_params(labelsize=6); ax.legend(fontsize=5)

    plt.suptitle("OOF Prediction Divergence |prob_A - prob_B| (FM-2->FM-3 Evidence)", fontsize=8)
    plt.tight_layout()
    out = C.OUT_FIGURES/"FigS1_OOF_prediction_divergence.jpg"
    plt.savefig(out, dpi=C.IEEE_DPI, bbox_inches="tight", format="jpg"); plt.close()
    print(f"  Saved: {out}")

    df_pm = pd.DataFrame(rows_per_model)
    for split in ["S1","S2"]:
        sub = df_pm[df_pm["split"]==split]
        rows_summary.append({"split":split,"n_models":len(sub),
                              "all_y_true_identical":sub["y_true_identical"].all(),
                              "mean_of_mean_div":sub["mean_abs_div"].mean(),
                              "max_of_max_div":sub["max_abs_div"].max()})
    df_sum = pd.DataFrame(rows_summary)
    print("\nOOF Divergence Summary:")
    print(df_sum.to_string(index=False))
    print(f"y_true identical across runs = {df_pm['y_true_identical'].all()} -> divergence SOLELY from RNG state")

    df_pm.to_csv(C.OUT_STATS/"oof_prediction_divergence_by_model.csv", index=False)
    df_sum.to_csv(C.OUT_STATS/"oof_prediction_divergence_summary.csv", index=False)
    print(f"\n[06] Done.")

if __name__ == "__main__":
    main()
