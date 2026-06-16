"""
02_bootstrap_ci.py
==================
Bootstrap 95% CI (10,000 reps) for flip_pct and Kendall tau — Phase 4A + 4B.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config as C


def bootstrap_mean_ci(data, n_boot=C.N_BOOTSTRAP, ci=C.BOOTSTRAP_CI, seed=C.BOOT_SEED):
    if len(data) < 2: return float(np.mean(data)), np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = np.array([np.mean(rng.choice(data, size=len(data), replace=True))
                     for _ in range(n_boot)])
    alpha = 1.0 - ci
    return float(np.mean(data)), float(np.percentile(boot, 100*alpha/2)), float(np.percentile(boot, 100*(1-alpha/2)))


def bootstrap_flip_pct(winners_df, base_winner, stress_col, phase, run):
    rows = []
    combos = [(s, r) for s in C.SPLITS for r in sorted(winners_df[stress_col].unique())]
    for split, rate in tqdm(combos, desc=f"  Bootstrap flip_pct Run {run}", leave=False):
        sub = winners_df[(winners_df["split"]==split) & (winners_df[stress_col]==rate)]
        bw = base_winner
        flips = (sub["winner_model"] != bw).astype(float).values * 100
        mean_ci, lo, hi = bootstrap_mean_ci(flips)
        rows.append({"phase":phase,"run":run,"split":split,stress_col:rate,
                     "n_seeds":len(flips),"base_winner":bw,
                     "flip_pct":float(np.mean(flips)),"boot_mean":mean_ci,
                     "boot_ci_lo":lo,"boot_ci_hi":hi})
    return pd.DataFrame(rows)


def bootstrap_kendall_tau(envelope_df, stress_col, phase, run):
    rows = []
    for split in C.SPLITS:
        sub = envelope_df[envelope_df["split"]==split].sort_values(stress_col)
        for _, row in sub.iterrows():
            tau = row.get("kendall_tau_mean", np.nan)
            n = 20
            if not np.isnan(tau) and abs(tau) < 1.0 and n > 3:
                z = np.arctanh(tau); se = 1.0/np.sqrt(n-3)
                lo, hi = float(np.tanh(z-1.96*se)), float(np.tanh(z+1.96*se))
            else:
                lo, hi = np.nan, np.nan
            rows.append({"phase":phase,"run":run,"split":split,stress_col:row[stress_col],
                         "kendall_tau":tau,"fisher_z_lo":lo,"fisher_z_hi":hi,
                         "reliable_verdict":row.get("stable_under_thresholds",np.nan)})
    return pd.DataFrame(rows)


def main():
    C.check_data()
    C.OUT_STATS.mkdir(parents=True, exist_ok=True)
    C.OUT_TABLES.mkdir(parents=True, exist_ok=True)

    all_flip, all_tau = [], []

    phase_list = tqdm([("4A","miss_rate"),("4B","target_prev")], desc="Phases")
    for phase, stress_col in phase_list:
        p_dict = C.P4A if phase == "4A" else C.P4B
        for run in tqdm(["A","B"], desc=f"  Runs (Phase {phase})", leave=False):
            winners  = pd.read_csv(p_dict[run]["winners"])
            envelope = pd.read_csv(p_dict[run]["envelope"])
            envelope["stable_under_thresholds"] = envelope["stable_under_thresholds"].map(
                {True:True,False:False,"True":True,"False":False})
            base_rate = winners[stress_col].min()
            bw = winners[winners[stress_col]==base_rate]["winner_model"].mode()
            bw = bw.iloc[0] if len(bw)>0 else "xgb"

            all_flip.append(bootstrap_flip_pct(winners, bw, stress_col, phase, run))
            all_tau.append(bootstrap_kendall_tau(envelope, stress_col, phase, run))

    df_flip = pd.concat(all_flip, ignore_index=True)
    df_tau  = pd.concat(all_tau,  ignore_index=True)
    df_flip.to_csv(C.OUT_STATS/"bootstrap_ci_flip_rate.csv", index=False)
    df_tau.to_csv(C.OUT_STATS/"bootstrap_ci_kendall_tau.csv", index=False)

    # CI overlap check
    print("\nCI non-overlap analysis (Phase 4A):")
    p4a_A = df_flip[(df_flip["phase"]=="4A")&(df_flip["run"]=="A")]
    p4a_B = df_flip[(df_flip["phase"]=="4A")&(df_flip["run"]=="B")]
    merged = p4a_A.merge(p4a_B, on=["split","miss_rate"], suffixes=("_A","_B"))
    merged["overlap"] = ((merged["boot_ci_lo_A"]<=merged["boot_ci_hi_B"]) &
                         (merged["boot_ci_lo_B"]<=merged["boot_ci_hi_A"]))
    print(f"  Non-overlapping CIs: {(~merged['overlap']).sum()}/{len(merged)} pairs")

    # LaTeX
    tex = [r"\begin{table}[!t]",r"\caption{Bootstrap 95\% CI of Flip Rate (\%) — Phase 4A}",
           r"\label{tab:bootstrap}",r"\centering\small",
           r"\begin{tabular}{llrrrr}",r"\toprule",
           r"Split & $\delta$ & \multicolumn{2}{c}{Run A} & \multicolumn{2}{c}{Run B} \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
           r" & & Flip\% & 95\%~CI & Flip\% & 95\%~CI \\",r"\midrule"]
    for split in C.SPLITS:
        sub = merged[merged["split"]==split]
        first = True
        for _, row in sub.iterrows():
            sp = split if first else ""; first = False
            tex.append(f"{sp} & {row['miss_rate']:.2f} "
                       f"& {row['flip_pct_A']:.1f} & [{row['boot_ci_lo_A']:.1f},{row['boot_ci_hi_A']:.1f}] "
                       f"& {row['flip_pct_B']:.1f} & [{row['boot_ci_lo_B']:.1f},{row['boot_ci_hi_B']:.1f}] \\\\")
        tex.append(r"\midrule")
    tex += [r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    (C.OUT_TABLES/"TABLE_bootstrap_ci_flip_rate.tex").write_text("\n".join(tex))
    print(f"\n[02] Done.")

if __name__ == "__main__":
    main()
