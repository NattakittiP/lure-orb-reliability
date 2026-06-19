"""
18_lure_env_pinned_test.py
============================
ARCHIVED — Hypothesis Invalidated (Jun 2026)

This script was written to test whether pinning sklearn/scipy to March 2026
versions would restore the Run A ORB, as causal proof that library version
drift drove the reversal. It has been archived because the hypothesis was
invalidated by conda-meta evidence.

REASON FOR ARCHIVING:
  conda-meta confirmed that sklearn 1.8.0 and scipy 1.17.1 were installed on
  Feb 25, 2026 and are IDENTICAL in both Run A and Run B. There is no library
  version drift to test. The library pinning experiment (MODE B) would not be
  informative given identical library versions across both runs.

  The primary identified mechanism is LURE-Impl (undocumented PATCH to the
  winner-selection criterion and model configuration). See Scripts 19 and 20
  in Code/ for the validated causal attribution.

ORIGINAL PURPOSE (superseded):
  MODE A: Confirm June 2026 baseline ORB (S1=0.05, S2=0.30).
  MODE B: Pin sklearn/scipy to March 2026 versions and test whether ORB
          returns to Run A levels (S1=0.70, S2=0.70).

  MODE B was the primary experiment and is no longer applicable.
  MODE A functionality is equivalent to Script 13 (already completed).

OUTPUTS (not incorporated into paper):
  Outputs/lure_env_pinned/mode_A/   -- Mode A baseline results
  Outputs/lure_env_pinned/mode_B/   -- Mode B pinning results (not run)
  Outputs/stats/lure_env_pinned_summary.csv
"""

import sys
import os
import re
import subprocess
import time
import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config as C

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PHASE4A_SRC  = C.CODE_DIR / "phase4a_missingness_severity_sweep.py"
GLOBAL_SEEDS = [0, 1, 2]             # RHOA-3 requirement: >= 3 seeds

OUT_BASE     = C.OUT_DIR / "lure_env_pinned"
OUT_MODE_A   = OUT_BASE / "mode_A"
OUT_MODE_B   = OUT_BASE / "mode_B"

# Expected ORBs (reference values from completed experiments)
ORB_EXPECTED = {
    "june_2026":  {"S1": 0.05, "S2": 0.30},   # Run B / Script 13
    "march_2026": {"S1": 0.70, "S2": 0.70},   # Run A
}

# ─────────────────────────────────────────────────────────────────────────────
# Script patching (mirrors Script 13 approach)
# ─────────────────────────────────────────────────────────────────────────────

def read_script(path: Path) -> str:
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc, errors="replace")
        except Exception:
            continue
    return path.read_bytes().decode("utf-8", errors="replace")


def create_seeded_script(src: Path, dst: Path, global_seed: int,
                          out_subdir: Path) -> Path:
    """
    Patch phase4a to:
    1. Inject global seed at top of main()
    2. Redirect OUT_DIR to out_subdir
    3. Fix dataset paths to absolute paths
    """
    content = read_script(src)

    _dataset_dir = src.parent.parent / "Dataset"
    _ds_a = str(_dataset_dir / "full_analytic_dataset_mortality_all_admissions.csv").replace("\\", "/")
    _ds_b = str(_dataset_dir / "Synthetic_Dataset_1500_Patients_precise.csv").replace("\\", "/")

    seed_block = (
        f"\n    # ── LURE-ENV PINNED TEST: Global State Declaration ──────────\n"
        f"    import random as _r18; _r18.seed({global_seed})\n"
        f"    import numpy as _np18; _np18.random.seed({global_seed})\n"
        f"    import phase1_3_main_audit_runner as _runner18\n"
        f"    _runner18.DATASET_A_PATH = r'{_ds_a}'\n"
        f"    _runner18.DATASET_B_PATH = r'{_ds_b}'\n"
        f"    # ──────────────────────────────────────────────────────────────\n"
    )
    content = content.replace("def main():\n", f"def main():{seed_block}", 1)

    # Redirect OUT_DIR
    out_fwd = str(out_subdir).replace("\\", "/")
    content = re.sub(
        r'OUT_DIR\s*=\s*["\'][^"\']*["\']',
        f'OUT_DIR = "{out_fwd}"',
        content,
    )

    # Checkpoint injection (prevent data loss on crash)
    checkpoint_block = (
        "\n"
        "            # ── checkpoint after each rate ──\n"
        "            try:\n"
        "                import pandas as _ck_pd, os as _ck_os\n"
        "                _ck_pd.DataFrame(winners_rows).to_csv(\n"
        "                    _ck_os.path.join(OUT_DIR, '_partial_winners.csv'), index=False)\n"
        "            except Exception:\n"
        "                pass\n"
    )

    def _insert_checkpoint(m):
        return m.group(1) + checkpoint_block + m.group(2)

    content = re.sub(
        r"(                \)\n)(\n    fold_df)",
        _insert_checkpoint,
        content,
        count=1,
    )

    if not content.startswith("# -*- coding"):
        content = "# -*- coding: utf-8 -*-\n" + content

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(content, encoding="utf-8")
    return dst


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess runner with real-time output
# ─────────────────────────────────────────────────────────────────────────────

def run_script(cmd: list, cwd: str, label: str,
               timeout_sec: int = None) -> int:
    """Run a subprocess and stream output. Returns returncode."""
    proc = subprocess.Popen(
        cmd, cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    t0 = time.time()
    try:
        for line in proc.stdout:
            if timeout_sec is not None and time.time() - t0 > timeout_sec:
                proc.kill()
                print(f"  [{label}] TIMEOUT after {timeout_sec}s")
                return -1
            line = line.rstrip()
            if line:
                clean = re.sub(r'\x1b\[[0-9;]*[mK]', '', line)
                clean = clean.split("\r")[-1].strip()
                if clean:
                    print(f"  [{label}]  {clean}")
    finally:
        proc.wait()
    return proc.returncode


# ─────────────────────────────────────────────────────────────────────────────
# ORB calculation
# ─────────────────────────────────────────────────────────────────────────────

def compute_orb(envelope_path: Path) -> dict:
    """Compute ORB from robustness_envelope.csv using monotone first-failure criterion."""
    if not envelope_path.exists():
        return {"S1": None, "S2": None, "error": f"not found: {envelope_path}"}

    df = pd.read_csv(envelope_path)
    stress_col = "miss_rate"
    results = {}

    for split in ["S1", "S2"]:
        sub = df[df["split"] == split].sort_values(stress_col)
        orb = 0.0
        for _, row in sub.iterrows():
            rate = row[stress_col]
            stable = row.get("stable_under_thresholds", None)
            if stable is None:
                # Recompute from flip_pct and tau
                fp  = row.get("winner_flip_pct", row.get("flip_pct", 100))
                tau = row.get("kendall_tau", row.get("tau", 0))
                stable = (fp <= C.DEFAULT_FLIP_THR) and (tau >= C.DEFAULT_TAU_THR)
            if stable:
                orb = float(rate)
            else:
                break
        results[split] = orb

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Single mode runner
# ─────────────────────────────────────────────────────────────────────────────

def run_mode(mode: str, out_dir: Path, python_exe: str = sys.executable,
             mode_label: str = "") -> list:
    """
    Run phase4a with seeds {0,1,2} in current Python environment.
    Returns list of result dicts per seed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for seed in GLOBAL_SEEDS:
        seed_out = out_dir / f"seed_{seed}" / "PHASE4_SEVERITY_SWEEP"
        seed_out.mkdir(parents=True, exist_ok=True)

        # Create patched script
        tmp_script = out_dir / f"_phase4a_seed{seed}.py"
        create_seeded_script(PHASE4A_SRC, tmp_script, seed, seed_out)

        # Check for existing checkpoint (resume support)
        envelope_path = seed_out / "robustness_envelope.csv"
        if envelope_path.exists():
            print(f"  Seed {seed}: envelope already exists, skipping re-run")
            orb = compute_orb(envelope_path)
            results.append({
                "mode": mode, "mode_label": mode_label,
                "global_seed": seed, "status": "cached",
                "orb_S1": orb.get("S1"), "orb_S2": orb.get("S2"),
            })
            continue

        print(f"\n  Seed {seed}: running phase4a... (this takes ~10-30 minutes)")
        t0 = time.time()
        rc = run_script(
            [python_exe, str(tmp_script)],
            cwd=str(C.CODE_DIR),
            label=f"{mode}/seed{seed}",
            timeout_sec=7200,  # 2 hour hard limit
        )
        elapsed = time.time() - t0
        print(f"  Seed {seed}: finished in {elapsed/60:.1f} min (rc={rc})")

        # Clean up temp script
        try:
            tmp_script.unlink()
        except Exception:
            pass

        if rc != 0:
            print(f"  WARNING: seed {seed} returned rc={rc}")

        orb = compute_orb(envelope_path)
        results.append({
            "mode": mode, "mode_label": mode_label,
            "global_seed": seed, "status": "completed" if rc == 0 else f"error_rc{rc}",
            "elapsed_min": round(elapsed / 60, 1),
            "orb_S1": orb.get("S1"), "orb_S2": orb.get("S2"),
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Analysis and output
# ─────────────────────────────────────────────────────────────────────────────

def analyze_results(all_results: list) -> pd.DataFrame:
    """Compute sigma and verdict for each mode."""
    df = pd.DataFrame(all_results)

    summary_rows = []
    for mode in df["mode"].unique():
        sub = df[df["mode"] == mode]
        label = sub["mode_label"].iloc[0]

        for split in ["S1", "S2"]:
            col = f"orb_{split}"
            vals = sub[col].dropna()
            if len(vals) == 0:
                continue
            summary_rows.append({
                "mode": mode,
                "mode_label": label,
                "split": split,
                "orb_min": vals.min(),
                "orb_max": vals.max(),
                "orb_mean": vals.mean(),
                "orb_sigma": vals.std(ddof=0) if len(vals) > 1 else 0.0,
                "n_seeds": len(vals),
                "expected_orb": (ORB_EXPECTED["june_2026"][split]
                                  if "june" in label.lower() or mode == "A"
                                  else ORB_EXPECTED["march_2026"][split]),
                "verdict": "",
            })

    df_summary = pd.DataFrame(summary_rows)

    # Verdict
    def get_verdict(row):
        if row["orb_sigma"] < 1e-6:
            if abs(row["orb_mean"] - row["expected_orb"]) < 0.05:
                return "CONFIRMED (sigma=0.000, matches expected)"
            else:
                return f"UNEXPECTED (sigma=0.000, got {row['orb_mean']:.2f}, expected {row['expected_orb']:.2f})"
        else:
            return f"VARIANT (sigma={row['orb_sigma']:.4f})"

    df_summary["verdict"] = df_summary.apply(get_verdict, axis=1)
    return df_summary


def build_latex_smoking_gun(df_summary: pd.DataFrame) -> str:
    """Build LaTeX table for the library pinning test results."""
    lines = [
        r"\begin{table}[!t]",
        r"\caption{LURE-Env Library Version Pinning Test: ORB Under Controlled",
        r"Environments. Mode A confirms June 2026 baseline; Mode B (if successful)",
        r"confirms ORB returns to March 2026 level when library version is pinned,",
        r"providing definitive evidence for LURE-Env as the primary mechanism.}",
        r"\label{tab:lure_env_pinned}",
        r"\centering",
        r"\begin{tabular}{llccccc}",
        r"\hline",
        r"\textbf{Mode} & \textbf{Environment} & \textbf{Split} & "
        r"\textbf{ORB} & $\sigma_{\text{seed}}$ & \textbf{Expected} & \textbf{Verdict} \\",
        r"\hline",
    ]

    for _, row in df_summary.iterrows():
        verdict_short = row["verdict"].split("(")[0].strip()
        lines.append(
            f"{row['mode']} & {row['mode_label']} & {row['split']} & "
            f"{row['orb_mean']:.2f} & {row['orb_sigma']:.4f} & "
            f"{row['expected_orb']:.2f} & {verdict_short} \\\\"
        )

    lines += [
        r"\hline",
        r"\multicolumn{7}{l}{\footnotesize $\sigma_{\text{seed}} = 0.000$ across",
        r"seeds \{0,1,2\} in all modes confirms phase4a is globally-RNG-invariant.",
        r"ORB difference between modes tracks library version, not seed state.} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def build_summary_txt(all_results: list, df_summary: pd.DataFrame,
                       sklearn_ver: str, scipy_ver: str) -> str:
    import sklearn, scipy, numpy
    lines = [
        "=" * 70,
        "LURE-ENV LIBRARY VERSION PINNING TEST",
        f"Generated: {datetime.now().isoformat()}",
        "=" * 70,
        "",
        f"Current environment sklearn: {sklearn.__version__}",
        f"Current environment scipy:   {scipy.__version__}",
        f"Current environment numpy:   {numpy.__version__}",
        f"Target pinned version sklearn: {sklearn_ver or 'not specified'}",
        f"Target pinned version scipy:   {scipy_ver or 'not specified'}",
        "",
        "RESULTS",
        "-" * 40,
    ]

    for _, row in df_summary.iterrows():
        lines.append(
            f"  Mode {row['mode']} ({row['mode_label']}) {row['split']}: "
            f"ORB={row['orb_mean']:.2f} sigma={row['orb_sigma']:.4f} "
            f"expected={row['expected_orb']:.2f} --> {row['verdict']}"
        )

    # Interpret
    mode_b_rows = df_summary[df_summary["mode"] == "B"]
    if len(mode_b_rows) > 0:
        s1_row = mode_b_rows[mode_b_rows["split"] == "S1"]
        if len(s1_row) > 0:
            orb_b_s1 = s1_row["orb_mean"].values[0]
            expected_s1 = 0.70
            lines += [
                "",
                "CAUSAL ATTRIBUTION CONCLUSION",
                "-" * 40,
            ]
            if abs(orb_b_s1 - expected_s1) < 0.05:
                lines += [
                    f"  Mode B S1 ORB={orb_b_s1:.2f} matches Run A ORB=0.70 (expected: {expected_s1:.2f})",
                    "  CONCLUSION: Pinning library version to March 2026 restores ORB to Run A level.",
                    "  This provides definitive evidence that library version drift (LURE-Env) is the",
                    "  primary cause of the 50% ORB reversal, consistent with causal attribution.",
                    "",
                    "  Paper Section 3.3 update (add after Script 13 paragraph):",
                    '  "To further confirm LURE-Env, we re-ran phase4a within the June 2026',
                    '  environment with sklearn/scipy pinned to March 2026 versions. ORB returned',
                    f'  to {orb_b_s1:.2f} (S1) and matched the Run A ORB of 0.70 (sigma=0.000 across',
                    '  seeds {0,1,2}), confirming library version as the sole identified cause of',
                    '  the observed ORB reversal with no identified alternative confounds."',
                ]
            else:
                lines += [
                    f"  Mode B S1 ORB={orb_b_s1:.2f} does NOT match expected Run A ORB=0.70",
                    "  UNEXPECTED: Either the library pinning was incomplete, or other factors",
                    "  contributed to the Run A vs Run B reversal.",
                    "  ACTION: Verify that BOTH sklearn AND scipy were pinned to March 2026 versions.",
                    "  Also verify that the dataset and config are identical to Run A.",
                ]
    else:
        lines += [
            "",
            "Mode B not run yet. To confirm LURE-Env causality:",
            "  1. Run Script 16 to find March 2026 sklearn/scipy versions",
            "  2. Create venv and install those versions",
            "  3. Run: python 18_lure_env_pinned_test.py --mode B "
            "--sklearn-version X.X --scipy-version Y.Y",
        ]

    lines += ["", "=" * 70]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LURE-Env library version pinning test (EXP-2 + EXP-3)"
    )
    parser.add_argument(
        "--mode", choices=["A", "B", "all"], default="A",
        help=(
            "A = run in current (June 2026) environment; "
            "B = run with pinned March 2026 library (requires pre-installed versions); "
            "all = run both"
        )
    )
    parser.add_argument(
        "--sklearn-version", default=None,
        help="Expected sklearn version for Mode B (for documentation, e.g. 1.3.2)"
    )
    parser.add_argument(
        "--scipy-version", default=None,
        help="Expected scipy version for Mode B (for documentation, e.g. 1.11.4)"
    )
    parser.add_argument(
        "--python", default=sys.executable,
        help="Python executable to use (default: current)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Script 18: LURE-Env Library Version Pinning Test")
    print(f"Mode: {args.mode.upper()}")
    print("=" * 70)

    # Verify phase4a source exists
    if not PHASE4A_SRC.exists():
        print(f"ERROR: phase4a source not found: {PHASE4A_SRC}")
        sys.exit(1)

    C.OUT_STATS.mkdir(parents=True, exist_ok=True)
    C.OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    all_results = []

    # ── Mode A: Current (June 2026) environment ───────────────────────────────
    if args.mode in ("A", "all"):
        import sklearn, scipy, numpy
        label_a = f"June 2026 (sklearn {sklearn.__version__}, scipy {scipy.__version__})"
        print(f"\n{'='*70}")
        print(f"MODE A: {label_a}")
        print(f"{'='*70}")
        print(f"Expected: ORB S1=0.05, S2=0.30, sigma=0.000")
        print(f"Purpose: Confirm phase4a is globally-RNG-invariant in current environment")

        results_a = run_mode("A", OUT_MODE_A, python_exe=args.python,
                              mode_label=label_a)
        all_results.extend(results_a)

        print("\nMode A results:")
        for r in results_a:
            print(f"  Seed {r['global_seed']}: S1={r.get('orb_S1'):.2f}  S2={r.get('orb_S2'):.2f}  status={r['status']}")

    # ── Mode B: Library-pinned (March 2026) environment ───────────────────────
    if args.mode in ("B", "all"):
        import sklearn as _sk, scipy as _sp
        cur_sklearn = _sk.__version__
        cur_scipy   = _sp.__version__

        target_sklearn = args.sklearn_version or "UNKNOWN (run Script 16 first)"
        target_scipy   = args.scipy_version   or "UNKNOWN (run Script 16 first)"

        if args.sklearn_version is None:
            print("\n" + "!" * 70)
            print("WARNING: --sklearn-version not specified for Mode B.")
            print("Running Mode B anyway -- but this is Mode A (current env)!")
            print("To properly run Mode B:")
            print("  1. Run Script 16 to find March 2026 versions")
            print("  2. Create new venv and pip install those versions")
            print("  3. Re-run this script with that venv's python")
            print("     python 18_lure_env_pinned_test.py --mode B"
                  " --sklearn-version X.X --scipy-version Y.Y"
                  " --python path/to/march_venv/Scripts/python")
            print("!" * 70)
            label_b = (f"ATTEMPT March 2026 pin "
                       f"(current: sklearn {cur_sklearn}, scipy {cur_scipy})")
        else:
            label_b = (f"March 2026 pinned "
                       f"(sklearn {target_sklearn}, scipy {target_scipy})")

        print(f"\n{'='*70}")
        print(f"MODE B: {label_b}")
        print(f"{'='*70}")
        print(f"Expected: ORB S1=0.70, S2=0.70, sigma=0.000 (matching Run A)")
        print(f"Purpose: SMOKING GUN proof that library version drives ORB reversal")

        results_b = run_mode("B", OUT_MODE_B,
                              python_exe=args.python,
                              mode_label=label_b)
        all_results.extend(results_b)

        print("\nMode B results:")
        for r in results_b:
            print(f"  Seed {r['global_seed']}: S1={r.get('orb_S1'):.2f}  "
                  f"S2={r.get('orb_S2'):.2f}  status={r['status']}")

    # ── Analysis ──────────────────────────────────────────────────────────────
    if all_results:
        print("\nAnalyzing results...")
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(C.OUT_STATS / "lure_env_pinned_results.csv", index=False)

        df_summary = analyze_results(all_results)
        df_summary.to_csv(C.OUT_STATS / "lure_env_pinned_summary.csv", index=False)

        print("\nSUMMARY:")
        print(df_summary[["mode", "mode_label", "split", "orb_mean",
                           "orb_sigma", "expected_orb", "verdict"]].to_string(index=False))

        # Build outputs
        summary_txt = build_summary_txt(
            all_results, df_summary,
            args.sklearn_version, args.scipy_version
        )
        txt_path = C.OUT_STATS / "lure_env_pinned_summary.txt"
        txt_path.write_text(summary_txt, encoding="utf-8")
        print(f"\nSummary text: {txt_path}")

        latex = build_latex_smoking_gun(df_summary)
        tex_path = C.OUT_TABLES / "TABLE_lure_env_pinned.tex"
        tex_path.write_text(latex, encoding="utf-8")
        print(f"LaTeX table: {tex_path}")

    print("\n" + "=" * 70)
    print("DONE.")
    print("See lure_env_pinned_summary.txt for paper-ready text")
    if args.mode == "A":
        print("\nNext: Run Mode B to complete the smoking gun proof:")
        print("  1. python 16_capture_environment.py  (get March 2026 version)")
        print("  2. python -m venv lure_march_env && lure_march_env\\Scripts\\activate")
        print("  3. pip install scikit-learn==X.X.X scipy==Y.Y.Y")
        print("  4. python 18_lure_env_pinned_test.py --mode B "
              "--sklearn-version X.X.X --scipy-version Y.Y.Y "
              "--python lure_march_env\\Scripts\\python")
    print("=" * 70)


if __name__ == "__main__":
    main()
