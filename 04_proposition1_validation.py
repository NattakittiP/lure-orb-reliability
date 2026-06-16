"""
04_proposition1_validation.py
==============================
Winner margin trajectory, Spearman rho, point-biserial correlation,
and logistic regression — validates the causal mechanism underlying LURE.
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pointbiserialr
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config as C
warnings.filterwarnings("ignore")


def compute_winner_margin(summary_df, stress_col):
    rows = []
    groups = list(summary_df.groupby(["split","seed",stress_col]))
    for (split, seed, rate), grp in tqdm(groups, desc="  Computing margins", leave=False):
        ranked = grp.sort_values(["auroc_mean","ap_mean","brier_mean"],
                                  ascending=[False,False,True]).reset_index(drop=True)
        if len(ranked) >= 2:
            rows.append({"split":split,"seed":seed,stress_col:rate,
                         "rank1_model":ranked.iloc[0]["model"],
                         "rank2_model":ranked.iloc[1]["model"],
                         "rank1_auroc":float(ranked.iloc[0]["auroc_mean"]),
                         "rank2_auroc":float(ranked.iloc[1]["auroc_mean"]),
                         "margin_auc":float(ranked.iloc[0]["auroc_mean"]-ranked.iloc[1]["auroc_mean"])})
    return pd.DataFrame(rows)


def run_logistic_regression(df, stress_col):
    try:
        import statsmodels.api as sm
        X = pd.DataFrame({"intercept":1.0,"delta":df[stress_col].astype(float),
                           "is_run_B":(df["run"]=="B").astype(float),
                           "split_S2":(df["split"]=="S2").astype(float),
                           "delta_x_runB":df[stress_col].astype(float)*(df["run"]=="B").astype(float)})
        y = df["verdict"].astype(int)
        model = sm.Logit(y, X).fit(disp=0)
        summary = pd.DataFrame({"variable":X.columns,"coef":model.params.values,
                                  "OR":np.exp(model.params.values),"p_value":model.pvalues.values,
                                  "lower_CI":np.exp(model.conf_int().values[:,0]),
                                  "upper_CI":np.exp(model.conf_int().values[:,1])})
        print(f"\nP4 Logistic Regression  McFadden R2={model.prsquared:.4f}")
        print(summary.to_string(index=False))
        return summary, float(model.prsquared)
    except Exception as e:
        print(f"[04] Logistic: {e}")
        return pd.DataFrame(), np.nan


def main():
    C.check_data()
    C.OUT_STATS.mkdir(parents=True, exist_ok=True)
    C.OUT_TABLES.mkdir(parents=True, exist_ok=True)

    all_margins, all_spearman, all_logistic = [], [], []

    for phase, stress_col in tqdm([("4A","miss_rate"),("4B","target_prev")], desc="Phases"):
        p_dict = C.P4A if phase == "4A" else C.P4B
        print(f"\n{'='*60}\nPhase {phase} — Proposition 1\n{'='*60}")

        run_margins = []
        for run in tqdm(["A","B"], desc="  Runs", leave=False):
            summary = pd.read_csv(p_dict[run]["summary"])
            margins = compute_winner_margin(summary, stress_col)
            margins["run"] = run; margins["phase"] = phase
            run_margins.append(margins)
            all_margins.append(margins)

        margins_both = pd.concat(run_margins, ignore_index=True)
        margin_agg = (margins_both.groupby(["run","split",stress_col])
                      .agg(margin_mean=("margin_auc","mean"), margin_std=("margin_auc","std"),
                           n_seeds=("margin_auc","count")).reset_index())

        # Spearman rho(margin, flip_pct)
        env_A = pd.read_csv(p_dict["A"]["envelope"]); env_A["run"] = "A"
        env_B = pd.read_csv(p_dict["B"]["envelope"]); env_B["run"] = "B"
        env_both = pd.concat([env_A, env_B], ignore_index=True)
        merged_sp = margin_agg.merge(env_both[["run","split",stress_col,"winner_flip_pct"]],
                                      on=["run","split",stress_col], how="inner")
        print(f"\nP2 Spearman rho(margin, flip_pct):")
        for run in tqdm(["A","B"], desc="  Spearman", leave=False):
            for split in C.SPLITS:
                sub = merged_sp[(merged_sp["run"]==run)&(merged_sp["split"]==split)]
                if len(sub) >= 4:
                    rho, p = spearmanr(sub["margin_mean"], sub["winner_flip_pct"])
                    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
                    print(f"  Run {run} {split}: rho={rho:+.4f}  p={p:.4f} {sig}")
                    all_spearman.append({"phase":phase,"run":run,"split":split,"spearman_rho":rho,"spearman_p":p,"n":len(sub)})

        print(f"\nP6 Critical epsilon per split (Run B):")
        flip_B = env_B[["split",stress_col,"winner_flip_pct"]].copy()
        m_B = margin_agg[margin_agg["run"]=="B"][["split",stress_col,"margin_mean"]]
        ep_df = flip_B.merge(m_B, on=["split",stress_col])
        for split in C.SPLITS:
            sub = ep_df[ep_df["split"]==split].sort_values(stress_col)
            first_flip = sub[sub["winner_flip_pct"]>0]
            if not first_flip.empty:
                print(f"  {split}: onset delta={float(first_flip.iloc[0][stress_col]):.2f}  "
                      f"margin={float(first_flip.iloc[0]['margin_mean']):.5f}")

        # Logistic regression
        env_both["stable_int"] = env_both["stable_under_thresholds"].map(
            {True:1,False:0,"True":1,"False":0})
        env_both["verdict"] = env_both["stable_int"]
        lr_df, _ = run_logistic_regression(env_both, stress_col)
        if not lr_df.empty:
            lr_df["phase"] = phase; all_logistic.append(lr_df)

    df_margins  = pd.concat(all_margins, ignore_index=True)
    df_spearman = pd.DataFrame(all_spearman)
    df_margins.to_csv(C.OUT_STATS/"conjecture1_winner_margins.csv", index=False)
    df_spearman.to_csv(C.OUT_STATS/"conjecture1_spearman_correlation.csv", index=False)
    if all_logistic:
        pd.concat(all_logistic).to_csv(C.OUT_STATS/"conjecture1_logistic_regression.csv", index=False)

    tex = [r"\begin{table}[!t]",
           r"\caption{Proposition~1 Validation: Spearman $\rho$ (Margin vs Flip Rate)}",
           r"\label{tab:proposition1}",r"\centering\small",
           r"\begin{tabular}{lllrrl}",r"\toprule",
           r"Phase & Run & Split & Spearman $\rho$ & $p$ & Note \\",r"\midrule"]
    for _, row in df_spearman.iterrows():
        rho  = row["spearman_rho"]
        pval = row["spearman_p"]
        is_nan = pd.isna(rho) or pd.isna(pval)
        if is_nan:
            # NaN arises when flip_pct = 0 for all delta (Run A S2) - no variance
            rho_str = "---"
            p_str   = "---"
            sig     = ""
            note    = "zero variance (no flips)"
        else:
            rho_val  = float(rho)
            pval_val = float(pval)
            rho_str  = f"{rho_val:+.4f}"
            p_str  = f"{pval_val:.4f}"
            sig     = ("***" if pval_val < 0.001 else "**" if pval_val < 0.01
                       else "*" if pval_val < 0.05 else "ns")
            if pval_val < 0.05 and rho_val < 0:
                note = "sig. negative (Prop.~1 supported)"
            elif pval_val < 0.05 and rho_val > 0:
                note = "sig. positive (unexpected)"
            else:
                note = "ns"
        tex.append(
            f"Phase {row['phase']} & Run {row['run']} & {row['split']} "
            f"& ${rho_str}$ & ${p_str}$ {sig} & {note} \\\\"
        )
    foot1 = (r"\multicolumn{6}{l}{\footnotesize $\rho$ = Spearman (margin vs.\ flip rate)."
             r" $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.}\\")
    foot2 = (r"\multicolumn{6}{l}{\footnotesize $^{\dagger}$ Run A flip\_pct$\equiv 0$ at all"
             r" $\delta$ --- Spearman undefined; confirms LURE absent under RHOA control.}\\")
    tex += [r"\midrule", foot1, foot2, r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (C.OUT_TABLES / "TABLE_conjecture1_spearman.tex").write_text(
        "\n".join(tex), encoding="utf-8"
    )
    print(f"\n[04] Outputs saved to {C.OUT_STATS}")
    print("[04] Done.")


if __name__ == "__main__":
    main()
