"""
16_capture_environment.py
==========================
EXP-1 + EXP-5: Environment Documentation (RHOA-4 Compliance)

PURPOSE
-------
Capture the exact computational environment specification for BOTH:
  - June 2026 environment (current -- Run B, Script 13, Run C)
  - March 2026 environment evidence (search logs, pip history, git tags)

This script is the FIRST thing to run before any new experiments because
library versions can change at any time.  Run this NOW to freeze the record.

RHOA-4 REQUIREMENT
------------------
  "ORB certification must state: sklearn version, scipy version, NumPy version,
   Python version, OS platform, global seed value, RHOA compliance flag."

WHY THIS IS CRITICAL (Option C)
---------------------------------
The central claim of the paper is that library version drift between March 2026
and June 2026 caused 50% ORB reversal.  Without documented version strings for
BOTH environments, a reviewer can dismiss the causal claim as "unverified."

OUTPUTS
-------
  Outputs/stats/environment_june2026.txt        -- current environment spec
  Outputs/stats/environment_run_a_evidence.txt  -- March 2026 evidence (if found)
  Outputs/stats/environment_comparison.csv      -- side-by-side comparison table
  Outputs/tables/TABLE_environment_comparison.tex
"""

import sys
import os
import platform
import subprocess
import json
import re
import importlib
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
import config as C

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_version(package_name: str) -> str:
    """Get installed version of a package."""
    try:
        import importlib.metadata
        return importlib.metadata.version(package_name)
    except Exception:
        try:
            mod = importlib.import_module(package_name)
            return getattr(mod, "__version__", "unknown")
        except Exception:
            return "NOT INSTALLED"


def get_full_env() -> dict:
    """Capture complete environment specification."""
    env = {
        "capture_timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "os_name": os.name,
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    # Core ML packages
    packages = [
        "scikit-learn", "scipy", "numpy", "pandas",
        "xgboost", "matplotlib", "tqdm", "pingouin",
    ]
    for pkg in packages:
        key = pkg.replace("-", "_")
        env[f"version_{key}"] = get_version(pkg)

    # sklearn sub-details
    try:
        import sklearn
        env["sklearn_full"] = sklearn.__version__
        # Check lbfgs solver behavior changed in 1.1+ (key for LR-L2)
        from sklearn.linear_model import LogisticRegression
        env["sklearn_lr_default_solver"] = LogisticRegression().solver
        env["sklearn_lr_default_max_iter"] = LogisticRegression().max_iter
    except Exception as e:
        env["sklearn_detail_error"] = str(e)

    # scipy sub-details
    try:
        import scipy
        env["scipy_full"] = scipy.__version__
        # liblinear version via sklearn
        try:
            from sklearn.svm import LinearSVC
            env["liblinear_via_sklearn"] = "sklearn " + sklearn.__version__
        except Exception:
            pass
    except Exception as e:
        env["scipy_detail_error"] = str(e)

    # pip freeze (full)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=30
        )
        env["pip_freeze"] = result.stdout
    except Exception as e:
        env["pip_freeze_error"] = str(e)

    return env


def search_run_a_evidence() -> dict:
    """
    Search for March 2026 (Run A) environment evidence.
    Checks: git log, conda history, pip log, any saved spec files.
    """
    evidence = {
        "search_timestamp": datetime.now().isoformat(),
        "sources_checked": [],
        "found": {},
        "notes": [],
    }

    project_root = C.BASE_ROOT

    # 1. Check for any saved environment files in the project
    env_file_patterns = [
        "requirements*.txt", "environment*.yml", "environment*.yaml",
        "*.freeze.txt", "pip_*.txt", "*_requirements.txt",
        "environment_spec*.txt", "env_march*.txt",
    ]
    for pattern in env_file_patterns:
        for f in project_root.rglob(pattern):
            evidence["sources_checked"].append(str(f))
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                # Look for sklearn/scipy version strings
                for line in content.splitlines():
                    for pkg in ["scikit-learn", "sklearn", "scipy", "numpy"]:
                        if pkg.lower() in line.lower() and any(c.isdigit() for c in line):
                            key = f"{pkg}_from_{f.name}"
                            evidence["found"][key] = line.strip()
            except Exception:
                pass

    # 2. Check git log for environment-related commits around March 2026
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "--after=2026-01-01", "--before=2026-04-01",
             "--", "requirements*.txt", "environment*.yml", "*.freeze.txt"],
            capture_output=True, text=True, timeout=10, cwd=str(project_root)
        )
        if result.stdout.strip():
            evidence["git_env_commits_mar2026"] = result.stdout.strip()
            evidence["sources_checked"].append("git log (Jan-Apr 2026)")
        else:
            evidence["notes"].append("No git commits touching env files in Jan-Apr 2026")
    except Exception as e:
        evidence["notes"].append(f"git not available: {e}")

    # 3. Check git log for any files modified around Run A date (March 2026)
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "--after=2026-02-01", "--before=2026-04-30"],
            capture_output=True, text=True, timeout=10, cwd=str(project_root)
        )
        if result.stdout.strip():
            evidence["git_log_mar2026"] = result.stdout.strip()[:2000]
            evidence["sources_checked"].append("git log (Feb-Apr 2026)")
    except Exception:
        pass

    # 4. Check conda history
    conda_history_paths = [
        Path.home() / ".conda" / "environments.txt",
        Path.home() / "anaconda3" / "conda-meta" / "history",
        Path.home() / "miniconda3" / "conda-meta" / "history",
    ]
    for cp in conda_history_paths:
        if cp.exists():
            evidence["sources_checked"].append(str(cp))
            try:
                content = cp.read_text(encoding="utf-8", errors="replace")
                # Find March 2026 entries
                lines = content.splitlines()
                for i, line in enumerate(lines):
                    if "2026-03" in line or "2026-02" in line:
                        context = "\n".join(lines[max(0, i-2):i+5])
                        evidence["found"]["conda_history_mar2026"] = context
                        break
            except Exception:
                pass

    # 5. Check pip log
    pip_log_paths = [
        Path.home() / ".local" / "share" / "pip" / "pip.log",
        Path(sys.prefix) / "pip.log",
    ]
    for pl in pip_log_paths:
        if pl.exists():
            evidence["sources_checked"].append(str(pl))
            try:
                content = pl.read_text(encoding="utf-8", errors="replace")
                # Find March 2026 install entries for sklearn/scipy
                for line in content.splitlines():
                    if ("2026-03" in line or "2026-02" in line) and \
                       any(pkg in line.lower() for pkg in ["scikit", "scipy"]):
                        evidence["found"].setdefault("pip_log_entries", []).append(line.strip())
            except Exception:
                pass

    # 6. Check Data/Result/Old/ for any version metadata
    run_a_dir = C.RUN_A_4A
    if run_a_dir.exists():
        for f in run_a_dir.glob("*.txt"):
            evidence["sources_checked"].append(str(f))
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                if any(pkg in content.lower() for pkg in ["sklearn", "scipy", "numpy", "version"]):
                    evidence["found"][f"run_a_{f.name}"] = content[:500]
            except Exception:
                pass

    return evidence


def format_env_txt(env: dict) -> str:
    """Format environment as human-readable text."""
    lines = [
        "=" * 70,
        "COMPUTATIONAL ENVIRONMENT SPECIFICATION (RHOA-4)",
        f"Captured: {env.get('capture_timestamp', 'unknown')}",
        "=" * 70,
        "",
        "PYTHON",
        f"  Version   : {env.get('python_version', 'unknown').split()[0]}",
        f"  Executable: {env.get('python_executable', 'unknown')}",
        "",
        "OPERATING SYSTEM",
        f"  Platform  : {env.get('platform', 'unknown')}",
        f"  OS        : {env.get('os_name', 'unknown')}",
        f"  Machine   : {env.get('machine', 'unknown')}",
        "",
        "ML PACKAGES (LURE-ENV CRITICAL)",
        f"  scikit-learn : {env.get('version_scikit_learn', 'unknown')}",
        f"  scipy        : {env.get('version_scipy', 'unknown')}",
        f"  numpy        : {env.get('version_numpy', 'unknown')}",
        f"  pandas       : {env.get('version_pandas', 'unknown')}",
        f"  xgboost      : {env.get('version_xgboost', 'unknown')}",
        "",
        "SKLEARN DEFAULTS (solver behavior)",
        f"  LogisticRegression default solver   : {env.get('sklearn_lr_default_solver', 'unknown')}",
        f"  LogisticRegression default max_iter : {env.get('sklearn_lr_default_max_iter', 'unknown')}",
        "",
        "PAPER CITATION STRING",
        f"  sklearn=={env.get('version_scikit_learn', '?')}, "
        f"scipy=={env.get('version_scipy', '?')}, "
        f"numpy=={env.get('version_numpy', '?')}, "
        f"Python {env.get('python_version', '?').split()[0]}",
        "",
        "=" * 70,
        "FULL pip freeze (for complete reproducibility)",
        "=" * 70,
        env.get("pip_freeze", env.get("pip_freeze_error", "unavailable")),
    ]
    return "\n".join(lines)


def format_evidence_txt(evidence: dict) -> str:
    """Format Run A evidence as human-readable text."""
    lines = [
        "=" * 70,
        "RUN A (MARCH 2026) ENVIRONMENT EVIDENCE SEARCH",
        f"Searched: {evidence.get('search_timestamp', 'unknown')}",
        "=" * 70,
        "",
        "SOURCES CHECKED:",
    ]
    for s in evidence.get("sources_checked", []):
        lines.append(f"  - {s}")

    lines += ["", "EVIDENCE FOUND:"]
    found = evidence.get("found", {})
    if found:
        for key, val in found.items():
            lines += [f"\n  [{key}]", f"  {val}"]
    else:
        lines.append("  NONE FOUND -- Manual inspection required")
        lines.append("")
        lines.append("  ACTION REQUIRED: Check the following manually:")
        lines.append("  1. Email/Slack from March 2026 mentioning 'conda install' or 'pip install'")
        lines.append("  2. Any Jupyter notebook cells showing import statements + versions")
        lines.append("  3. The conda environment YAML used when Run A was first executed")
        lines.append("  4. pip show scikit-learn output from that period")

    lines += ["", "NOTES:"]
    for note in evidence.get("notes", []):
        lines.append(f"  - {note}")

    return "\n".join(lines)


def build_comparison_csv(env_june: dict) -> "pd.DataFrame":
    """Build side-by-side comparison table (fills Run A column as TBD)."""
    import pandas as pd

    packages = {
        "scikit-learn": "version_scikit_learn",
        "scipy":        "version_scipy",
        "numpy":        "version_numpy",
        "pandas":       "version_pandas",
        "xgboost":      "version_xgboost",
        "python":       "python_version",
        "platform":     "platform",
    }

    rows = []
    for pkg, key in packages.items():
        val_june = env_june.get(key, "unknown")
        if "\n" in str(val_june):
            val_june = str(val_june).split("\n")[0]
        rows.append({
            "package":        pkg,
            "run_a_march2026": "TBD -- see environment_run_a_evidence.txt",
            "run_b_june2026":  val_june,
            "script13_june2026": val_june,
            "run_c_june2026":  val_june,
            "lure_env_driver": "YES" if pkg in ("scikit-learn", "scipy") else "indirect",
        })

    df = pd.DataFrame(rows)
    return df


def build_latex_table(df: "pd.DataFrame") -> str:
    """Build LaTeX environment comparison table for paper."""
    lines = [
        r"\begin{table}[!t]",
        r"\caption{Computational Environment Specification (RHOA-4 Documentation)}",
        r"\label{tab:environment}",
        r"\centering",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"\textbf{Package} & \textbf{Run A} & \textbf{Run B} & "
        r"\textbf{Script 13} & \textbf{Run C} \\",
        r" & \textbf{(Mar 2026)} & \textbf{(Jun 2026)} & "
        r"\textbf{(Jun 2026)} & \textbf{(Jun 2026)} \\",
        r"\hline",
    ]

    for _, row in df.iterrows():
        pkg = row["package"]
        run_a = "TBD" if "TBD" in str(row["run_a_march2026"]) else str(row["run_a_march2026"])
        run_b = str(row["run_b_june2026"])[:20]
        s13   = str(row["script13_june2026"])[:20]
        rc    = str(row["run_c_june2026"])[:20]
        # Escape special chars
        for s in [pkg, run_a, run_b, s13, rc]:
            s = s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
        lines.append(
            f"{pkg.replace('_', r'_')} & {run_a} & {run_b} & {s13} & {rc} \\\\"
        )

    lines += [
        r"\hline",
        r"\multicolumn{5}{l}{\footnotesize LURE-Env mechanism: sklearn/scipy version",
        r"drift changes lbfgs/liblinear solver convergence, altering LR-L2 and SVM AUC} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import pandas as pd

    print("=" * 70)
    print("Script 16: Environment Documentation (RHOA-4)")
    print("EXP-1 (March 2026 evidence) + EXP-5 (June 2026 spec capture)")
    print("=" * 70)

    C.OUT_STATS.mkdir(parents=True, exist_ok=True)
    C.OUT_TABLES.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Capture current (June 2026) environment ──────────────────────
    print("\n[1/4] Capturing current environment (June 2026)...")
    env_june = get_full_env()

    # Save pip freeze separately
    pip_path = C.OUT_STATS / "environment_june2026_pip_freeze.txt"
    pip_path.write_text(env_june.get("pip_freeze", "unavailable"), encoding="utf-8")
    print(f"  pip freeze saved: {pip_path}")

    # Save full env spec
    env_txt = format_env_txt(env_june)
    spec_path = C.OUT_STATS / "environment_june2026.txt"
    spec_path.write_text(env_txt, encoding="utf-8")
    print(f"  Environment spec saved: {spec_path}")
    print(f"\n  KEY VERSIONS (June 2026):")
    print(f"    scikit-learn : {env_june.get('version_scikit_learn', '?')}")
    print(f"    scipy        : {env_june.get('version_scipy', '?')}")
    print(f"    numpy        : {env_june.get('version_numpy', '?')}")
    print(f"    python       : {env_june.get('python_version', '?').split()[0]}")

    # ── Step 2: Search for Run A (March 2026) evidence ───────────────────────
    print("\n[2/4] Searching for Run A (March 2026) environment evidence...")
    evidence = search_run_a_evidence()

    evidence_txt = format_evidence_txt(evidence)
    ev_path = C.OUT_STATS / "environment_run_a_evidence.txt"
    ev_path.write_text(evidence_txt, encoding="utf-8")
    print(f"  Evidence report saved: {ev_path}")

    found_count = len(evidence.get("found", {}))
    if found_count > 0:
        print(f"  FOUND {found_count} pieces of evidence -- see report")
    else:
        print("  No automatic evidence found -- manual investigation required")
        print("  See report for action items")

    # ── Step 3: Build comparison table ───────────────────────────────────────
    print("\n[3/4] Building environment comparison table...")
    df = build_comparison_csv(env_june)

    csv_path = C.OUT_STATS / "environment_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV saved: {csv_path}")

    latex = build_latex_table(df)
    tex_path = C.OUT_TABLES / "TABLE_environment_comparison.tex"
    tex_path.write_text(latex, encoding="utf-8")
    print(f"  LaTeX table saved: {tex_path}")

    # ── Step 4: Print citation string ─────────────────────────────────────────
    print("\n[4/4] Paper citation string (RHOA-4):")
    print(f"""
  June 2026 environment (Run B, Script 13, Run C):
    Python {env_june.get("python_version", "?").split()[0]},
    scikit-learn=={env_june.get("version_scikit_learn", "?")},
    scipy=={env_june.get("version_scipy", "?")},
    numpy=={env_june.get("version_numpy", "?")}

  March 2026 environment (Run A):
    PENDING -- update environment_comparison.csv once found

  For paper Section 3.1 table, use: TABLE_environment_comparison.tex
    (update run_a_march2026 column once March 2026 versions are found)
""")

    print("=" * 70)
    print("DONE. Next steps:")
    print("  1. Open environment_run_a_evidence.txt -- check if versions were found")
    print("  2. If not found, manually record March 2026 sklearn/scipy versions")
    print("     and update environment_comparison.csv run_a_march2026 column")
    print("  3. Run Script 17 (lure_env_disentanglement.py) next")
    print("=" * 70)


if __name__ == "__main__":
    main()
