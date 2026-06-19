# ============================================================
# Script 19: LURE-Impl Criterion Reproduction Test
# ============================================================
# PURPOSE: Direct criterion-reversion test for LURE-Impl causal attribution.
#          Reverts the winner-selection criterion from AUROC+AP+Brier back to
#          AUROC-only (the Run A criterion) while keeping all other PATCH code
#          unchanged, to isolate the criterion sub-component of LURE-Impl.
#
# DESIGN:
#   - Imports phase4a as a module and monkey-patches the winner-selection
#     sort criterion back to AUROC-only (same as Run A's original code).
#   - All other logic (preprocessing, calibration, nested CV, etc.) is
#     unchanged from phase4a -- zero code duplication risk.
#
# CRITERION REVERSION (the key change):
#   Current phase4a (Run B PATCH):
#       sort by ["auroc_mean","ap_mean","brier_mean"], ascending=[False,False,True]
#   This test (AUROC-only revert):
#       sort by ["auroc_mean"], ascending=[False]
#
# SEEDS: {1001, 1002, 1003} -- 3 seeds sufficient (Script 13: sigma=0.000,
#        implementation-stable within a fixed code version).
#
# ACTUAL RESULTS (Jun 17, 2026):
#   ORB S1: 0.50  (partial recovery from Run B's 0.05; criterion explains 69% of gap)
#   ORB S2: 0.70  (full recovery to Run A level; criterion is the sole identified driver)
#
# COMPARISON TABLE:
#   | Criterion           | Code    | ORB S1 | ORB S2 |
#   |---------------------|---------|--------|--------|
#   | AUROC+AP+Brier      | PATCH   | 0.05   | 0.30   | (Run B)
#   | AUROC-only (REVERT) | PATCH   | 0.50   | 0.70   | (THIS SCRIPT -- Jun 17, 2026)
#   | AUROC-only          | original| 0.70   | 0.70   | (Run A)
#   S2: criterion reversion fully recovers ORB (confirmed).
#   S1: criterion reversion partially recovers ORB (dominant identified driver, 69% of gap).
#       Residual S1 gap (0.50 -> 0.70) is an accepted limitation (L7); see Script 20.
#
# HOW MONKEY-PATCHING WORKS:
#   1. Import phase4a as a module
#   2. Replace phase4a.eval_one_setting with a wrapper that intercepts
#      the agg_sorted step and re-sorts by AUROC-only
#   3. Call phase4a.main() normally -- it uses the patched function
#
# OUTPUT:
#   Outputs/lure_impl_criterion_test/  (robustness_envelope.csv, etc.)
#   Outputs/stats/lure_impl_criterion_summary.txt
# ============================================================

import os
import sys
import importlib
import types
import numpy as np
import pandas as pd
from pathlib import Path

# ---- Path setup --------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent
DATA_CODE   = REPO_ROOT / "Data" / "Code"
DATASET_DIR = REPO_ROOT / "Data" / "Dataset"
OUT_DIR     = REPO_ROOT / "Outputs" / "lure_impl_criterion_test"
STAT_DIR    = REPO_ROOT / "Outputs" / "stats"

OUT_DIR.mkdir(parents=True, exist_ok=True)
STAT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(DATA_CODE))

# ---- Patch runner dataset paths (same pattern as Script 13) ------------
import phase1_3_main_audit_runner as _runner
_runner.DATASET_A_PATH = str(DATASET_DIR / "full_analytic_dataset_mortality_all_admissions.csv").replace("\\", "/")
_runner.DATASET_B_PATH = str(DATASET_DIR / "Synthetic_Dataset_1500_Patients_precise.csv").replace("\\", "/")

# ---- Import phase4a as a module ----------------------------------------
import phase4a_missingness_severity_sweep as phase4a

# ---- Override phase4a config -------------------------------------------
phase4a.SEEDS  = [1001, 1002, 1003]   # 3 seeds sufficient (Script 13: sigma=0.000)
phase4a.OUT_DIR = str(OUT_DIR)

# ============================================================
# MONKEY-PATCH: AUROC-only sort (reverts the LURE-Impl PATCH)
# ============================================================
_original_eval_one_setting = phase4a.eval_one_setting

def _eval_one_setting_auroc_only(X, y, groups, split_key, seed, miss_rate,
                                  num_cols, cat_cols, **kwargs):
    """
    Wrapper around phase4a.eval_one_setting that replaces the
    (AUROC, AP, -Brier) sort with AUROC-only -- reverts the PATCH.
    """
    result = _original_eval_one_setting(
        X=X, y=y, groups=groups,
        split_key=split_key, seed=seed, miss_rate=miss_rate,
        num_cols=num_cols, cat_cols=cat_cols,
        **kwargs
    )

    # Re-sort model_summary and re-derive winner using AUROC-only
    agg = result["model_summary"].copy()
    agg_sorted = agg.sort_values(
        ["auroc_mean"],
        ascending=[False],
        kind="mergesort",
    ).reset_index(drop=True)

    winner_model  = agg_sorted.iloc[0]["model"]
    winner_auc    = float(agg_sorted.iloc[0]["auroc_mean"])
    winner_ap     = float(agg_sorted.iloc[0]["ap_mean"])
    winner_brier  = float(agg_sorted.iloc[0]["brier_mean"])
    ranking       = agg_sorted["model"].tolist()

    result["winner_model"]  = winner_model
    result["winner_auc"]    = winner_auc
    result["winner_ap"]     = winner_ap
    result["winner_brier"]  = winner_brier
    result["ranking"]       = ranking

    return result

# Install the patched function into the module
phase4a.eval_one_setting = _eval_one_setting_auroc_only

# ============================================================
# MAIN: Run phase4a with patched criterion, then report
# ============================================================
def main():
    print("=" * 60)
    print("Script 19: LURE-Impl Criterion Reproduction Test")
    print("Criterion: AUROC-only (Run A original, reverts PATCH)")
    print(f"Seeds: {phase4a.SEEDS}")
    print(f"Output: {OUT_DIR}")
    print("Expected ORB if LURE-Impl confirmed: ~0.70 (matching Run A)")
    print("=" * 60)

    # Run phase4a with the monkey-patched criterion
    phase4a.main()

    # ---- Read results and compute ORB per split --------------------------
    env_path = OUT_DIR / "robustness_envelope.csv"
    if not env_path.exists():
        print(f"\nERROR: {env_path} not found -- phase4a.main() may have failed.")
        return

    env_df = pd.read_csv(env_path)

    RUN_A_ORB = {"S1": 0.70, "S2": 0.70}
    RUN_B_ORB = {"S1": 0.05, "S2": 0.30}

    orb_results = {}
    for split_key in ["S1", "S2"]:
        sub = env_df[env_df["split"] == split_key]
        stable = sub[sub["stable_under_thresholds"] == True]
        orb = float(stable["miss_rate"].max()) if len(stable) > 0 else 0.0
        orb_results[split_key] = orb

    orb_s1 = orb_results.get("S1", float("nan"))
    orb_s2 = orb_results.get("S2", float("nan"))

    match_run_a_s1 = abs(orb_s1 - RUN_A_ORB["S1"]) <= 0.10
    match_run_b_s1 = abs(orb_s1 - RUN_B_ORB["S1"]) <= 0.10

    if match_run_a_s1:
        verdict = "LURE-Impl CONFIRMED: AUROC-only criterion reproduces Run A ORB (~0.70)"
    elif match_run_b_s1:
        verdict = "UNEXPECTED: ORB still matches Run B -- investigate model config change independently"
    else:
        verdict = f"INCONCLUSIVE: ORB S1={orb_s1:.2f} -- does not cleanly match Run A ({RUN_A_ORB['S1']}) or Run B ({RUN_B_ORB['S1']})"

    summary_lines = [
        "=" * 60,
        "Script 19: LURE-Impl Criterion Reproduction Test",
        "=" * 60,
        "",
        "CRITERION USED: AUROC-only (Run A original -- reverts PATCH)",
        f"SEEDS: {phase4a.SEEDS}  (3 sufficient: Script 13 sigma=0.000)",
        "",
        "RESULTS:",
        f"  ORB S1 (AUROC-only criterion): {orb_s1:.2f}",
        f"  ORB S2 (AUROC-only criterion): {orb_s2:.2f}",
        "",
        "COMPARISON:",
        f"  Run A (original code, AUROC-only):         ORB S1={RUN_A_ORB['S1']}  ORB S2={RUN_A_ORB['S2']}",
        f"  Script 19 (PATCH code, AUROC-only revert): ORB S1={orb_s1:.2f}  ORB S2={orb_s2:.2f}",
        f"  Run B (PATCH code, AUROC+AP+Brier):        ORB S1={RUN_B_ORB['S1']}  ORB S2={RUN_B_ORB['S2']}",
        "",
        "VERDICT:",
        f"  {verdict}",
        "",
        "PAPER USE (section 3.3 causal attribution):",
        "  If ORB returns to ~0.70: upgrades language from",
        "  'consistent with LURE-Impl attribution' to 'confirmed by",
        "  direct criterion-reversion experiment (Script 19)'.",
        "=" * 60,
    ]

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    summary_path = STAT_DIR / "lure_impl_criterion_summary.txt"
    summary_path.write_text(summary_text + "\n", encoding="utf-8")
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
