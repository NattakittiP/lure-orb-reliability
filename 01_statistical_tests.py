"""
01_statistical_tests.py
=======================
Full battery of hypothesis tests comparing Run A (LURE controlled) vs.
Run B (LURE active): McNemar, Cohen's kappa, exact binomial, permutation,
Wilcoxon signed-rank, ICC(A,1) via pingouin, Spearman rho — Phase 4A + 4B.

ICC(A,1) computed via pingouin (Shrout & Fleiss 1979); corrected S&F formula
used as fallback when pingouin is unavailable.
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2, binom, wilcoxon, spearmanr, f as f_dist
from sklearn.metrics import cohen_kappa_score
from tqdm.auto import tqdm

try:
    import pingouin as pg
    _PINGOUIN = True
except ImportError:
    _PINGOUIN = False
    warnings.warn(
        "[01] pingouin not found - using corrected manual ICC formula. "
        "Install: pip install pingouin --break-system-packages",
        RuntimeWarning,
    )

sys.path.insert(0, str(Path(__file__).parent))
import config as C


def load_envelope(run, phase="4A"):
    p = C.P4A[run]["envelope"] if phase == "4A" else C.P4B[run]["envelope"]
    df = pd.read_csv(p)
    df["run"] = run
    df["phase"] = phase
    df["stable_under_thresholds"] = df["stable_under_thresholds"].map(
        {True: True, False: False, "True": True, "False": False}
    ).astype(bool)
    return df


def load_winners(run, phase="4A"):
    p = C.P4A[run]["winners"] if phase == "4A" else C.P4B[run]["winners"]
    df = pd.read_csv(p)
    df["run"] = run
    df["phase"] = phase
    return df


def mcnemar_test(n01, n10, correction=True):
    if n01 + n10 == 0:
        return np.nan, np.nan
    if correction:
        stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    else:
        stat = (n01 - n10) ** 2 / (n01 + n10)
    return float(stat), float(1.0 - chi2.cdf(stat, df=1))


def icc_a1(x, y, subject_ids=None):
    """
    ICC(A,1): two-way random effects, absolute agreement, single measures.
    Uses pingouin when available; falls back to corrected S&F 1979 formula.

    Parameters
    ----------
    x, y        : paired arrays (Run A / Run B winner AUC)
    subject_ids : optional label array; defaults to 0..n-1

    Returns (icc_value, ci_lower_95, ci_upper_95)
    """
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan

    if _PINGOUIN:
        ids = (subject_ids.astype(str)
               if subject_ids is not None
               else np.arange(n).astype(str))
        long = pd.DataFrame({
            "subject": np.concatenate([ids, ids]),
            "rater":   ["A"] * n + ["B"] * n,
            "value":   np.concatenate([x, y]),
        })
        try:
            icc_df = pg.intraclass_corr(
                data=long, targets="subject", raters="rater",
                ratings="value", nan_policy="omit",
            )
            row = icc_df[icc_df["Type"] == "ICC(A,1)"].iloc[0]
            ci = row["CI95"]
            return float(row["ICC"]), float(ci[0]), float(ci[1])
        except Exception as e:
            warnings.warn(f"[01] pingouin ICC failed ({e}); using fallback.", RuntimeWarning)

    # Corrected Shrout & Fleiss (1979) ICC(A,1)
    # ICC(A,1) = (MSr - MSe) / (MSr + (k-1)*MSe + k*(MSc-MSe)/n)  with k=2
    k = 2
    data = np.column_stack([x, y])
    grand_mean = data.mean()
    row_means  = data.mean(axis=1)
    col_means  = data.mean(axis=0)
    SS_rows  = k * ((row_means - grand_mean) ** 2).sum()
    SS_cols  = n * ((col_means  - grand_mean) ** 2).sum()
    SS_total = ((data - grand_mean) ** 2).sum()
    SS_error = SS_total - SS_rows - SS_cols
    MS_rows  = SS_rows  / (n - 1)
    MS_cols  = SS_cols  / (k - 1)
    MS_error = SS_error / ((n - 1) * (k - 1))
    denom = MS_rows + (k - 1) * MS_error + k * (MS_cols - MS_error) / n
    if denom == 0 or MS_error <= 0:
        return np.nan, np.nan, np.nan
    icc = (MS_rows - MS_error) / denom
    # 95% CI via F-distribution (Shrout & Fleiss 1979 Eq. 11)
    F   = MS_rows / MS_error
    df1 = n - 1
    df2 = (n - 1) * (k - 1)
    FL  = F / f_dist.ppf(0.975, df1, df2)
    FU  = F * f_dist.ppf(0.975, df2, df1)
    ci_lo = (FL - 1) / (FL + k - 1)
    ci_hi = (FU - 1) / (FU + k - 1)
    return float(icc), float(ci_lo), float(ci_hi)


def permutation_test(vA, vB, n_perm=100_000, seed=42):
    rng = np.random.default_rng(seed)
    obs = int(((vA == 1) & (vB == 0)).sum())
    count = 0
    for _ in tqdm(range(n_perm), desc="  Permutation test", leave=False, unit="perm"):
        swap = rng.integers(0, 2, len(vA)).astype(bool)
        pA = np.where(swap, vB, vA)
        pB = np.where(swap, vA, vB)
        if int(((pA == 1) & (pB == 0)).sum()) >= obs:
            count += 1
    return obs, float(count / n_perm)


def run_phase_analysis(phase):
    print(f"\n{'='*60}\n Phase {phase} Statistical Tests\n{'='*60}")
    stress_col = "miss_rate" if phase == "4A" else "target_prev"

    env_A = load_envelope("A", phase)
    env_B = load_envelope("B", phase)
    win_A = load_winners("A", phase)
    win_B = load_winners("B", phase)

    merge_env = env_A.merge(env_B, on=["split", stress_col], suffixes=("_A", "_B"))
    vA = merge_env["stable_under_thresholds_A"].astype(int).values
    vB = merge_env["stable_under_thresholds_B"].astype(int).values
    n00 = int(((vA == 0) & (vB == 0)).sum())
    n01 = int(((vA == 0) & (vB == 1)).sum())
    n10 = int(((vA == 1) & (vB == 0)).sum())
    n11 = int(((vA == 1) & (vB == 1)).sum())
    total = n00 + n01 + n10 + n11

    print(f"\nT1/T2 contingency  B=REL  B=UNREL")
    print(f"  A=REL    {n11:4d}   {n10:4d}   (total A-reliable: {n11+n10})")
    print(f"  A=UNREL  {n01:4d}   {n00:4d}   N={total}")

    mc_stat, mc_p = mcnemar_test(n10, n01)
    exact_p = binom.sf(n10 - 1, n10 + n01, 0.5) if n10 + n01 > 0 else np.nan
    try:
        kappa = (cohen_kappa_score(vA, vB)
                 if len(set(vA)) > 1 or len(set(vB)) > 1
                 else np.nan)
    except Exception:
        kappa = np.nan

    sig_str = ("***" if not np.isnan(mc_p) and mc_p < 0.001 else
               "**"  if not np.isnan(mc_p) and mc_p < 0.01  else
               "*"   if not np.isnan(mc_p) and mc_p < 0.05  else "ns")
    print(f"T1  McNemar chi2={mc_stat:.4f}  p={mc_p:.6f}  {sig_str}")
    print(f"T1b Exact binomial  n10={n10}, n01={n01}:  p={exact_p:.6f}")
    kap_str = f"{kappa:.4f}" if not np.isnan(kappa) else "n/a"
    print(f"T2  Cohen kappa={kap_str}")

    obs_n10, perm_p = permutation_test(vA, vB, n_perm=100_000)
    print(f"T3  Permutation  obs={obs_n10}  p={perm_p:.6f}")

    # Wilcoxon per (split, delta)
    merge_win = win_A.merge(win_B, on=["split", stress_col, "seed"], suffixes=("_A", "_B"))
    wil_rows = []
    combos = [(s, r) for s in C.SPLITS for r in sorted(merge_win[stress_col].unique())]
    for split, rate in tqdm(combos, desc="  Wilcoxon per delta", leave=False):
        r = merge_win[(merge_win["split"] == split) & (merge_win[stress_col] == rate)]
        aA, aB = r["winner_auc_A"].values, r["winner_auc_B"].values
        diffs = aA - aB
        try:
            stat, p = (wilcoxon(aA, aB, alternative="two-sided")
                       if len(diffs) >= 4 and not (diffs == 0).all()
                       else (np.nan, np.nan))
        except Exception:
            stat, p = np.nan, np.nan
        w_sig = ("***" if not np.isnan(p) and p < 0.001 else
                 "**"  if not np.isnan(p) and p < 0.01  else
                 "*"   if not np.isnan(p) and p < 0.05  else "ns")
        wil_rows.append({
            "phase": phase, "split": split, stress_col: rate, "n": len(aA),
            "mean_diff": float(diffs.mean()), "wilcoxon_stat": stat,
            "wilcoxon_p": p, "sig": w_sig,
        })
    df_wil = pd.DataFrame(wil_rows)
    sig_n = df_wil["sig"].isin(["*", "**", "***"]).sum()
    print(f"\nT4  Wilcoxon: significant in {sig_n}/{len(df_wil)} (split,delta) pairs")

    # ICC(A,1) per split via pingouin
    # Subject = (seed x delta); two raters = Run A vs Run B.
    # ICC measures raw AUC consistency. Moderate/good ICC + 50% verdict
    # reversal confirms Proposition 1: thin margins cause certification
    # failure even when absolute AUC values are consistent between runs.
    icc_rows = []
    for split in tqdm(C.SPLITS, desc="  ICC per split", leave=False):
        sub = merge_win[merge_win["split"] == split].copy()
        sub["subject_id"] = (sub["seed"].astype(str) + "_"
                             + sub[stress_col].astype(str))
        iv, il, ih = icc_a1(
            sub["winner_auc_A"].values,
            sub["winner_auc_B"].values,
            subject_ids=sub["subject_id"].values,
        )
        interp = ("poor"     if pd.isna(iv) or iv < 0.40 else
                  "fair"     if iv < 0.60 else
                  "moderate" if iv < 0.75 else
                  "good"     if iv < 0.90 else "excellent")
        icc_rows.append({
            "phase": phase, "split": split, "n": len(sub),
            "icc_21": iv,
            "icc_lower95": il,
            "icc_upper95": ih,
            "interp": interp,
            "method": "pingouin_ICC(A,1)" if _PINGOUIN else "manual_corrected_ICC(A,1)",
        })
        ci_s = f"[{il:.4f},{ih:.4f}]" if not pd.isna(il) else "[n/a]"
        print(f"T5  ICC(A,1) {split}: {iv:.4f} {ci_s} ({interp})")
    df_icc = pd.DataFrame(icc_rows)

    mc_result = {
        "phase": phase, "n_verdicts": total,
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "reversal_count": n10, "reversal_pct": 100.0 * n10 / total,
        "mcnemar_stat": mc_stat, "mcnemar_p": mc_p,
        "exact_binomial_p": exact_p,
        "cohen_kappa": kappa, "permutation_p": perm_p,
    }
    return mc_result, df_wil, df_icc


def main():
    C.check_data()
    C.OUT_STATS.mkdir(parents=True, exist_ok=True)
    C.OUT_TABLES.mkdir(parents=True, exist_ok=True)

    all_mc, all_wil, all_icc = [], [], []
    for phase in tqdm(["4A", "4B"], desc="Phases"):
        mc, wil, icc = run_phase_analysis(phase)
        all_mc.append(mc)
        all_wil.append(wil)
        all_icc.append(icc)

    df_mc  = pd.DataFrame(all_mc)
    df_wil = pd.concat(all_wil, ignore_index=True)
    df_icc = pd.concat(all_icc, ignore_index=True)

    df_mc.to_csv(C.OUT_STATS  / "T1_mcnemar_kappa.csv",      index=False)
    df_wil.to_csv(C.OUT_STATS / "T2_wilcoxon_per_delta.csv", index=False)
    df_icc.to_csv(C.OUT_STATS / "T3_icc_interrun_reliability.csv",                index=False)

    method_note = "pingouin ICC(A,1)" if _PINGOUIN else "corrected manual ICC(A,1)"
    tex = [
        r"\begin{table}[!t]",
        r"\caption{Statistical Tests: Run A vs Run B (ORB Verdict Agreement)}",
        r"\label{tab:statistical_tests}",
        r"\centering\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Phase & Test & Statistic & $p$ & Effect & Interpretation \\",
        r"\midrule",
    ]
    for _, row in df_mc.iterrows():
        kap = row["cohen_kappa"]
        kap_s = f"{kap:.3f}" if not pd.isna(kap) else "n/a"
        kap_i = ("poor"     if pd.isna(kap) or kap < 0.20 else
                 "fair"     if kap < 0.40 else "moderate")
        tex.append(
            "Phase " + str(row['phase']) + " & McNemar & "
            "$\\chi^2=" + f"{row['mcnemar_stat']:.3f}" + "$ & "
            "$" + f"{row['mcnemar_p']:.4f}" + "$ & "
            "$\\kappa=" + kap_s + "$ & " + kap_i + " \\\\"
        )
    for _, row in df_icc.iterrows():
        il = row["icc_lower95"]
        ih = row["icc_upper95"]
        ci_s = f"[{il:.3f},{ih:.3f}]" if not pd.isna(il) else "[n/a]"
        tex.append(
            "Phase " + str(row['phase']) + " (" + str(row['split']) + ") & ICC(A,1) & "
            "$" + f"{row['icc_21']:.3f}" + "$ & --- & "
            "95\\%CI " + ci_s + " & " + str(row['interp']) + " \\\\"
        )
    tex += [
        r"\midrule",
        "\\multicolumn{6}{l}{\\footnotesize ICC(A,1): two-way random, "
        "absolute agreement, single measure (" + method_note + ").}\\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (C.OUT_TABLES / "TABLE_statistical_tests.tex").write_text("\n".join(tex))
    print(f"\n[01] Done. ICC method: {method_note}")


if __name__ == "__main__":
    main()
