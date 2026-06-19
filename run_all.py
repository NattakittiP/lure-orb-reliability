"""
run_all.py  —  Master Runner
==============================
Runs all LURE analysis scripts in the correct order.
Run this ONCE to reproduce all statistics and figures.
Last updated: Jun 19, 2026

Order
-----
  1  config check          — verify all data files exist
  2  statistical tests      — McNemar, kappa, Wilcoxon, ICC
  3  bootstrap CI           — 10,000-rep CI for flip_pct
  4  effect sizes           — Cohen's d, Hedges' g per delta
  5  Conjecture 1           — margin analysis, Spearman, logistic
  6  sensitivity analysis   — threshold grid + convergence
  7  OOF divergence         — prediction divergence (FM-2→FM-3)
  8  figures (data)         — Figs 1-4, 7-9
  9  figures (conceptual)   — Figs 5-6
  10 N=5 unseeded batch     — OPTIONAL (long: skip with --skip-n5)
  11 supplementary tests    — Tests 1-7: ICC, trend, TOST, logistic, power, margin
  12 RHOA seed validation   — LURE determinism + RHOA baseline seed sensitivity
  13 RHOA-3 direct          — Phase 4A x 3 global seeds [~44h/seed, skip with --skip-heavy]
  14 RHOA-2 isolation       — Run C: RHOA-1 RHOA-2 violated [~44h, skip with --skip-heavy]
  15 recovery experiment    — Formal LURE recovery framing (analytical, fast)
  16 RHOA report            — LaTeX tables, claims, checklist (runs LAST before LURE-Impl)
  19 LURE-Impl criterion    — AUROC-only revert: S2 CONFIRMED, S1 partially supported
                              [~44h, skip with --skip-lure-impl; COMPLETED Jun 17, 2026]
  20 LURE-Impl model config — 2x2 factorial criterion x model_config
                              [~44h, skip with --skip-lure-impl; DONE WITH LIMITATIONS Jun 19, 2026]

ARCHIVED SCRIPTS (do not run — hypothesis invalidated)
-------------------------------------------------------
  17 lure_env_disentanglement.py  — Library version hypothesis (INVALIDATED: conda-meta
                                    confirmed sklearn 1.8.0 identical in Run A and Run B)
  18 lure_env_pinned_test.py      — Library pinning test (INVALIDATED: same reason as 17)
  Archive location: Code/archive/

Usage
-----
  python run_all.py                         # run everything (scripts 1-16, 19-20)
  python run_all.py --skip-n5               # skip N=5 unseeded batch (script 10)
  python run_all.py --skip-heavy            # skip scripts 13+14 (long pipeline re-runs)
  python run_all.py --skip-lure-impl        # skip scripts 19+20 (LURE-Impl experiments)
  python run_all.py --only 2 3              # run only scripts 2 and 3
  python run_all.py --only 15 16            # run only recovery + report (fast mode)
  python run_all.py --only 19               # run LURE-Impl criterion test only
  python run_all.py --only 20               # run LURE-Impl model config test only

Recommended first run
---------------------
  python run_all.py --skip-heavy --skip-n5 --skip-lure-impl
  (completes in <2h, produces all main figures and statistics)

  Then run heavy scripts separately when compute time is available:
  python run_all.py --only 13        # RHOA-3 direct (~132h, checkpoint/resume)
  python run_all.py --only 14        # Run C (~44h, checkpoint/resume)
  python run_all.py --only 19        # LURE-Impl criterion (~44h) [ALREADY DONE Jun 17]
  python run_all.py --only 20        # LURE-Impl model config (~44h) [ALREADY DONE Jun 19]
  python run_all.py --only 16        # regenerate report after 13+14
"""

import sys
import time
import traceback
import subprocess
import argparse
from pathlib import Path

HERE = Path(__file__).parent

SCRIPTS = [
    (1,  "config_check",        None),                                    # inline
    (2,  "statistical",         "01_statistical_tests.py"),
    (3,  "bootstrap_ci",        "02_bootstrap_ci.py"),
    (4,  "effect_size",         "03_effect_size.py"),
    (5,  "conjecture1",         "04_proposition1_validation.py"),
    (6,  "sensitivity",         "05_sensitivity_analysis.py"),
    (7,  "oof_divergence",      "06_oof_divergence.py"),
    (8,  "figures_main",        "07_figures_main.py"),
    (9,  "figures_concept",     "08_figures_conceptual.py"),
    (10, "n5_unseeded",         "10_unseeded_batch.py"),                  # optional heavy
    (11, "supplementary",       "11_supplementary_tests.py"),
    (12, "rhoa_validation",     "12_rhoa_seed_validation.py"),
    (13, "rhoa3_direct",        "13_rhoa3_direct_validation.py"),         # compute-heavy (~132h)
    (14, "rhoa2_isolation",     "14_rhoa2_isolation.py"),                 # compute-heavy (~44h)
    (15, "recovery",            "15_recovery_experiment.py"),             # analytical (fast)
    (16, "rhoa_report",         "09_rhoa_report.py"),                     # report (runs before LURE-Impl)
    (19, "lure_impl_criterion", "19_lure_impl_criterion_test.py"),        # LURE-Impl (~44h; COMPLETED Jun 17)
    (20, "lure_impl_model_cfg", "20_lure_impl_model_config_test.py"),     # LURE-Impl (~44h; DONE Jun 19)
]

# Scripts that involve full Phase 4A re-runs (compute-heavy)
HEAVY_SCRIPTS      = {13, 14}
LURE_IMPL_SCRIPTS  = {19, 20}
N5_SCRIPT          = 10


def run_script(script_path: Path, label: str) -> bool:
    print(f"\n{'─'*60}")
    print(f"  Running: {label} ({script_path.name})")
    print(f"{'─'*60}")
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(HERE),
            timeout=None       # no timeout — heavy scripts run until completion
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            print(f"  [OK]  {label}  ({elapsed:.1f}s)")
            return True
        else:
            print(f"  [FAIL]  {label}  (returncode={result.returncode}, {elapsed:.1f}s)")
            return False
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT]  {label}")
        return False
    except Exception as e:
        print(f"  [ERROR]  {label}:  {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="LURE IEEE TR Master Runner"
    )
    parser.add_argument(
        "--skip-n5", action="store_true",
        help="Skip N=5 unseeded batch (script 10)",
    )
    parser.add_argument(
        "--skip-heavy", action="store_true",
        help=(
            f"Skip heavy pipeline re-run scripts "
            f"({', '.join(str(s) for s in sorted(HEAVY_SCRIPTS))}): "
            "RHOA-3 direct (~132h) and RHOA-2 isolation (~44h)"
        ),
    )
    parser.add_argument(
        "--skip-lure-impl", action="store_true",
        help=(
            f"Skip LURE-Impl experiments "
            f"({', '.join(str(s) for s in sorted(LURE_IMPL_SCRIPTS))}): "
            "criterion test and model config test (~44h each; already completed Jun 2026)"
        ),
    )
    parser.add_argument(
        "--only", nargs="+", type=int,
        help="Run only specified script numbers",
    )
    args = parser.parse_args()

    # ── Step 1: Config check ────────────────────────────────────────
    print("\n" + "="*60)
    print("  LURE IEEE TR Analysis — Master Runner")
    print("="*60)
    sys.path.insert(0, str(HERE))
    import config as C
    try:
        C.check_data()
    except FileNotFoundError as e:
        print(f"\n[ABORT] Data check failed:\n{e}")
        sys.exit(1)

    results = {}
    t_total = time.time()
    skipped = []

    for num, label, script_name in SCRIPTS:
        if num == 1:
            continue   # Already done above

        if args.only and num not in args.only:
            print(f"  [SKIP]  {num}: {label}  (--only filter)")
            results[label] = "SKIP"
            continue

        if args.skip_n5 and num == N5_SCRIPT:
            print(f"  [SKIP]  {num}: {label}  (--skip-n5)")
            results[label] = "SKIP"
            skipped.append(label)
            continue

        if args.skip_heavy and num in HEAVY_SCRIPTS:
            print(f"  [SKIP]  {num}: {label}  (--skip-heavy — run separately when compute is available)")
            results[label] = "SKIP"
            skipped.append(label)
            continue

        if args.skip_lure_impl and num in LURE_IMPL_SCRIPTS:
            print(f"  [SKIP]  {num}: {label}  (--skip-lure-impl — already completed Jun 2026)")
            results[label] = "SKIP"
            skipped.append(label)
            continue

        script_path = HERE / script_name
        if not script_path.exists():
            print(f"  [MISSING]  {script_path}")
            results[label] = "MISSING"
            continue

        ok = run_script(script_path, f"{num}: {label}")
        results[label] = "OK" if ok else "FAILED"

    # ── Summary ─────────────────────────────────────────────────────
    elapsed_total = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  Run All Complete  ({elapsed_total:.0f}s total)")
    print(f"{'='*60}")
    for label, status in results.items():
        icon = "✓" if status == "OK" else ("⊘" if status == "SKIP" else "✗")
        print(f"  {icon}  {label}: {status}")

    n_ok   = sum(1 for s in results.values() if s == "OK")
    n_fail = sum(1 for s in results.values() if s == "FAILED")
    n_skip = sum(1 for s in results.values() if s == "SKIP")
    print(f"\n  {n_ok} succeeded, {n_fail} failed, {n_skip} skipped")

    if skipped:
        print(f"\n  Skipped (run separately when ready): {', '.join(skipped)}")
        if any(s in skipped for s in ["rhoa3_direct", "rhoa2_isolation"]):
            print("    python run_all.py --only 13   # RHOA-3 direct (~132h, checkpoint/resume)")
            print("    python run_all.py --only 14   # RHOA-2 isolation (~44h, checkpoint/resume)")
            print("    python run_all.py --only 16   # regenerate report after 13+14")
        if any(s in skipped for s in ["lure_impl_criterion", "lure_impl_model_cfg"]):
            print("    python run_all.py --only 19   # LURE-Impl criterion (~44h) [COMPLETED Jun 17]")
            print("    python run_all.py --only 20   # LURE-Impl model config (~44h) [DONE Jun 19]")

    if n_fail == 0:
        print("\n  All outputs saved to:")
        print(f"    Figures          : {C.OUT_FIGURES}")
        print(f"    Statistics       : {C.OUT_STATS}")
        print(f"    Tables           : {C.OUT_TABLES}")
        print(f"    N5 Batch         : {C.OUT_N25}")
        print(f"    RHOA-3 Direct    : {C.OUT_RHOA3}")
        print(f"    Run C (RHOA-2)   : {C.OUT_RUN_C}")
        print(f"    Recovery Exp.    : {C.OUT_RECOVERY}")
        print(f"    LURE-Impl Crit.  : {C.OUT_LURE_IMPL_CRITERION}")
        print(f"    LURE-Impl Config : {C.OUT_LURE_IMPL_MODEL_CONFIG}")
        print("\n  Next steps:")
        print("    1. Review outputs/stats/pre_submission_checklist.txt")
        print("    2. Copy outputs/tables/ALL_TABLES.tex into your LaTeX paper")
        print("    3. Insert abstract numbers from outputs/stats/abstract_numbers.txt")
        if skipped:
            print("    4. Run skipped scripts when compute is available, then re-run 16")
    else:
        print("\n  Check error messages above for failed scripts.")
        sys.exit(1)


if __name__ == "__main__":
    main()
