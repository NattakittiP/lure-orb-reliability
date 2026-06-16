"""
config.py  —  LURE Analysis
==============================
Central configuration: all paths, constants, and thresholds.

Folder semantics
----------------
  Run A  (LURE CONTROLLED)   ->  global seed set before main()
         = Data/Result/Old/PHASE4A_SEVERITY_SWEEP (Old)/
           Data/Result/Old/PHASE4B_PREVALENCE_SHIFT (Old)/

  Run B  (LURE UNCONTROLLED) ->  global seed NOT set (OS entropy)
         = Data/Result/New/PHASE4A_SEVERITY_SWEEP (New)/
           Data/Result/New/PHASE4B_PREVALENCE_SHIFT (New)/

Language reframe (Blueprint §0)
-------------------------------
  robustness envelope        ->  Operational Reliability Boundary (ORB)
  winner flip / selection    ->  ORB verdict reversal / reliability boundary failure
  missing data rate delta    ->  input degradation severity / stress level delta
  random seed not fixed      ->  uncontrolled stochastic failure mode (LURE)
  stable under thresholds    ->  operationally reliable / within certified limits
"""

from pathlib import Path

# -----------------------------------------------------------------
# Base directories  (edit BASE_ROOT if you move the project)
# -----------------------------------------------------------------
BASE_ROOT = Path(__file__).resolve().parent.parent   # .../IEEE Realibility_Submission/
DATA_DIR  = BASE_ROOT / "Data"

# Run A  — LURE controlled (seeded run, Feb 2026)
RUN_A_4A  = DATA_DIR / "Result" / "Old" / "PHASE4A_SEVERITY_SWEEP (Old)"
RUN_A_4B  = DATA_DIR / "Result" / "Old" / "PHASE4B_PREVALENCE_SHIFT (Old)"
RUN_A_1_3 = DATA_DIR / "Result" / "Old" / "PHASE1-3"

# Run B  — LURE uncontrolled (unseeded run, Jun 2026)
CODE_DIR  = DATA_DIR / "Code"
RUN_B_4A  = DATA_DIR / "Result" / "New" / "PHASE4A_SEVERITY_SWEEP (New)"
RUN_B_4B  = DATA_DIR / "Result" / "New" / "PHASE4B_PREVALENCE_SHIFT (New)"
RUN_B_1_3 = DATA_DIR / "Result" / "New" / "PHASE1_3_Results (New)"

# OOF files  (Phase 1-3 out-of-fold predictions — optional, 06_oof_divergence.py skips if absent)
RUN_A_OOF_DIR = RUN_A_1_3 / "oof_P0"
RUN_B_OOF_DIR = RUN_B_1_3

# Output directory  (matches existing Outputs/ folder at project root)
OUT_DIR      = BASE_ROOT / "Outputs"
OUT_FIGURES  = OUT_DIR / "figures"
OUT_STATS    = OUT_DIR / "stats"
OUT_TABLES   = OUT_DIR / "tables"
OUT_N25      = OUT_DIR / "N5_unseeded"   # N=5 unseeded runs (all identical → sufficient)

# -----------------------------------------------------------------
# Phase 4A  — Missingness (MCAR stress) file names
# -----------------------------------------------------------------
P4A = {
    "A": {
        "envelope"  : RUN_A_4A / "PHASE_4a_robustness_envelope.csv",
        "summary"   : RUN_A_4A / "PHASE_4a_severity_summary_by_model.csv",
        "winners"   : RUN_A_4A / "PHASE_4a_severity_winner_by_seed.csv",
        "flip_onset": RUN_A_4A / "PHASE_4a_flip_onset.csv",
        "fold_met"  : RUN_A_4A / "PHASE_4a_severity_fold_metrics.csv",
    },
    "B": {
        "envelope"  : RUN_B_4A / "robustness_envelope.csv",
        "summary"   : RUN_B_4A / "severity_summary_by_model.csv",
        "winners"   : RUN_B_4A / "severity_winner_by_seed.csv",
        "flip_onset": RUN_B_4A / "flip_onset.csv",
        "fold_met"  : RUN_B_4A / "severity_fold_metrics.csv",
    },
}

# -----------------------------------------------------------------
# Phase 4B  — Prevalence shift file names
# -----------------------------------------------------------------
P4B = {
    "A": {
        "envelope"  : RUN_A_4B / "PHASE_4b_robustness_envelope_prevalence_shift.csv",
        "summary"   : RUN_A_4B / "PHASE_4b_prevalence_shift_summary_by_model.csv",
        "winners"   : RUN_A_4B / "PHASE_4b_prevalence_shift_winner_by_seed.csv",
        "flip_onset": RUN_A_4B / "PHASE_4b_flip_onset_prevalence_shift.csv",
    },
    "B": {
        "envelope"  : RUN_B_4B / "robustness_envelope_prevalence_shift.csv",
        "summary"   : RUN_B_4B / "prevalence_shift_summary_by_model.csv",
        "winners"   : RUN_B_4B / "prevalence_shift_winner_by_seed.csv",
        "flip_onset": RUN_B_4B / "flip_onset_prevalence_shift.csv",
    },
}

# -----------------------------------------------------------------
# ORB stability thresholds  (default from Blueprint §3.4)
# -----------------------------------------------------------------
DEFAULT_FLIP_THR = 5.0    # %  — max winner_flip_pct to call "RELIABLE"
DEFAULT_TAU_THR  = 0.80   #    — min Kendall tau to call "RELIABLE"

# Sensitivity grid (used in 05_sensitivity_analysis.py)
FLIP_THR_GRID = [0.0, 5.0, 10.0, 15.0, 20.0]
TAU_THR_GRID  = [0.70, 0.75, 0.80, 0.85, 0.90]

# -----------------------------------------------------------------
# Experiment constants
# -----------------------------------------------------------------
SEEDS_PHASE4 = list(range(1001, 1021))   # 20 seeds
MISS_RATES   = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
PREV_TARGETS = [-1.0, 0.1, 0.2, 0.3, 0.4, 0.5]   # -1 = no shift (baseline)
SPLITS       = ["S1", "S2"]
MODELS       = ["lr_l2", "svm_linear_cal", "rf", "xgb", "extratrees"]

# -----------------------------------------------------------------
# Figure style (IEEE TR)
# -----------------------------------------------------------------
IEEE_SINGLE_COL_INCH = 3.50
IEEE_DOUBLE_COL_INCH = 7.16
IEEE_DPI             = 300
IEEE_FONT            = "Arial"
IEEE_FONTSIZE        = 9

COLOR_RUN_A      = "#1A5276"   # dark blue  — LURE controlled
COLOR_RUN_B      = "#C0392B"   # dark red   — LURE uncontrolled
COLOR_RELIABLE   = "#1E8449"   # dark green — RELIABLE verdict
COLOR_UNRELIABLE = "#C0392B"   # dark red   — UNRELIABLE verdict
COLOR_NEUTRAL    = "#7F8C8D"   # grey
RUN_LABELS = {"A": "Run A (LURE Controlled)", "B": "Run B (LURE Active)"}

# -----------------------------------------------------------------
# Bootstrap
# -----------------------------------------------------------------
N_BOOTSTRAP  = 10_000
BOOTSTRAP_CI = 0.95
BOOT_SEED    = 42

# -----------------------------------------------------------------
# N=5 unseeded batch  (10_unseeded_batch.py)
# All 5 runs produced identical ORB boundary — no value in running more.
# -----------------------------------------------------------------
N_UNSEEDED_RUNS = 5
PHASE4A_SCRIPT  = CODE_DIR / "phase4a_missingness_severity_sweep.py"
PHASE4B_SCRIPT  = CODE_DIR / "phase4b_prevalence_shift_sweep.py"

# -----------------------------------------------------------------
# RHOA-3 Direct Global-Seed Validation  (13_rhoa3_direct_validation.py)
# -----------------------------------------------------------------
RHOA3_GLOBAL_SEEDS = [0, 1, 2]          # ≥3 seeds — RHOA-3 requirement
OUT_RHOA3          = OUT_DIR / "rhoa3_direct"

# -----------------------------------------------------------------
# Run C — RHOA-2 Isolation Experiment  (14_rhoa2_isolation.py)
# RHOA-1 ✓ (global seed=0) + RHOA-2 ✗ (perturbation via global RNG)
# Empirically isolates RHOA-2's independent contribution.
# -----------------------------------------------------------------
RUN_C_GLOBAL_SEED  = 0                  # same as Run A to isolate RHOA-2 effect
OUT_RUN_C          = OUT_DIR / "run_c_rhoa2_isolation"

# -----------------------------------------------------------------
# Recovery Experiment  (15_recovery_experiment.py)
# Analytical: uses existing Run A / Run B data
# -----------------------------------------------------------------
OUT_RECOVERY       = OUT_DIR / "recovery_experiment"

# -----------------------------------------------------------------
# Validation helper
# -----------------------------------------------------------------
def check_data():
    """Verify all required input files exist."""
    required = []
    for run in ("A", "B"):
        for path in P4A[run].values():
            required.append(path)
        for path in P4B[run].values():
            required.append(path)
    missing = [p for p in required if not p.exists()]
    if missing:
        msgs = "\n  ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Missing {len(missing)} required data file(s):\n  {msgs}\n"
            "Check that BASE_ROOT in config.py points to the correct directory."
        )
    print(f"[config] All {len(required)} data files found. OK")

