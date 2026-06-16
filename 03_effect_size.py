"""
03_effect_size.py
=================
Cohen's dz, Hedges' g, paired t-test, Wilcoxon signed-rank, and rank-biserial r
per (split, delta) — Phase 4A + 4B.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_rel, wilcoxon
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config as C


def cohens_dz(x, y):
    diff = x - y; sd = np.std(diff, ddof=1)
    return float(np.mean(diff)/sd) if sd > 0 else 0.0

def hedges_g(x, y):
    dz = cohens_dz(x, y); n = len(x)
    J = 1.0 - 3.0/(4.0*(n-1)-1)
    return float(dz*J)

def rank_biserial(w_stat, n):
    mu = n*(n+1)/4.0; sig = np.sqrt(n*(n+1)*(2*n+1)/24.0)
    Z = (w_stat-mu)/sig if sig > 0 else 0.0
    return float(Z/np.sqrt(n)) if n > 0 else 0.0


def effect_size_analysis(phase, stress_col):
    p_dict = C.P4A if phase == "4A" else C.P4B
    win_A = pd.read_csv(p_dict["A"]["winners"])
    win_B = pd.read_csv(p_dict["B"]["winners"])
    merged = win_A.merge(win_B, on=["split",stress_col,"seed"], suffixes=("_A","_B"))

    rows = []
    combos = [(s, r) for s in C.SPLITS for r in sorted(merged[stress_col].unique())]
    for split, rate in tqdm(combos, desc=f"  Effect sizes Phase {phase}", leave=False):
        r = merged[(merged["split"]==split)&(merged[stress_col]==rate)]
        aA, aB = r["winner_auc_A"].values, r["winner_auc_B"].values
        n = len(aA); diffs = aA - aB
        dz = cohens_dz(aA, aB); g = hedges_g(aA, aB)
        t_s, t_p = ttest_rel(aA, aB) if n>=2 and np.std(diffs,ddof=1)>0 else (np.nan, np.nan)
        try:
            w_s, w_p = wilcoxon(aA, aB, alternative="two-sided") if n>=4 and not (diffs==0).all() else (np.nan, np.nan)
            r_rb = rank_biserial(w_s, n)
        except Exception: w_s, w_p, r_rb = np.nan, np.nan, np.nan

        def sig(p):
            return "***" if not np.isnan(p) and p<0.001 else "**" if not np.isnan(p) and p<0.01 else "*" if not np.isnan(p) and p<0.05 else "ns"

        rows.append({"phase":phase,"split":split,stress_col:rate,"n_seeds":n,
                     "mean_auc_A":float(aA.mean()),"mean_auc_B":float(aB.mean()),
                     "mean_diff":float(diffs.mean()),"sd_diff":float(np.std(diffs,ddof=1)),
                     "cohens_dz":dz,"hedges_g":g,
                     "interp_g":"|trivial|" if abs(g)<0.2 else "|small|" if abs(g)<0.5 else "|medium|" if abs(g)<0.8 else "|large|",
                     "t_stat":t_s,"t_p":t_p,"t_sig":sig(t_p),
                     "wilcoxon_stat":w_s,"wilcoxon_p":w_p,"rank_biserial_r":r_rb,"w_sig":sig(w_p)})
    return pd.DataFrame(rows)


def main():
    C.check_data()
    C.OUT_STATS.mkdir(parents=True, exist_ok=True)
    C.OUT_TABLES.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for phase, stress_col in tqdm([("4A","miss_rate"),("4B","target_prev")], desc="Phases"):
        df = effect_size_analysis(phase, stress_col)
        all_dfs.append(df)
        df.to_csv(C.OUT_STATS/f"effect_size_phase{phase}.csv", index=False)
        print(f"\nPhase {phase} — effect sizes:")
        print(df[["split",stress_col,"mean_diff","cohens_dz","hedges_g","interp_g","t_sig","w_sig"]].to_string(index=False))

    # LaTeX (Phase 4A)
    df_4A = all_dfs[0]
    tex = [r"\begin{table}[!t]",r"\caption{Effect Size: Winner AUC Run A vs Run B (Phase 4A)}",
           r"\label{tab:effect_size}",r"\centering\small",
           r"\begin{tabular}{llrrrrr}",r"\toprule",
           r"Split & $\delta$ & $\Delta\mu$ & Hedges'~$g$ & Size & $p$ ($t$) & $p$ (W) \\",r"\midrule"]
    for split in C.SPLITS:
        sub = df_4A[df_4A["split"]==split]; first = True
        for _, row in sub.iterrows():
            sp = split if first else ""; first = False
            tex.append(f"{sp} & {row['miss_rate']:.2f} & {row['mean_diff']:+.4f} "
                       f"& {row['hedges_g']:+.3f} & {row['interp_g']} "
                       f"& {row['t_p']:.4f}{row['t_sig']} & {row['wilcoxon_p']:.4f}{row['w_sig']} \\\\")
        tex.append(r"\midrule")
    tex += [r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    (C.OUT_TABLES/"TABLE_effect_size.tex").write_text("\n".join(tex))
    print(f"\n[03] Done.")

if __name__ == "__main__":
    main()
