"""
14_rhoa2_isolation.py
======================
Run C: RHOA-2 Isolation Experiment
IEEE TR LURE Gap Closure — RHOA-2 Independent Empirical Contribution

PURPOSE
-------
Empirically isolates the contribution of RHOA-2 (Perturbation RNG Isolation)
independently of RHOA-1 (Global State Declaration) by constructing:

  Run A: RHOA-1 ✓  +  RHOA-2 ✓  →  ORB = 0.70 (full compliance)
  Run C: RHOA-1 ✓  +  RHOA-2 ✗  →  ORB = ?    [THIS SCRIPT]
  Run B: RHOA-1 ✗  +  RHOA-2 ✗  →  ORB = 0.05 / 0.30 (LURE fully active)

RHOA-2 VIOLATION MECHANISM IN RUN C
-------------------------------------
RHOA-2 requires that perturbation injection uses isolated RNG instances
(np.random.default_rng(seed)) that NEVER modify global RNG state ω.

In Run C, we violate RHOA-2 by replacing the isolated RNG calls in
apply_mcar_missingness_split() with global np.random calls:

  ORIGINAL (RHOA-2 compliant):
    rng_tr = np.random.default_rng(seed + 12345 + int(rate * 1000))
    rng_te = np.random.default_rng(seed + 54321 + int(rate * 1000))
    m_tr = rng_tr.random(len(X_tr)) < rate
    m_te = rng_te.random(len(X_te)) < rate

  PATCHED (RHOA-2 violated):
    m_tr = np.random.random(len(X_tr)) < rate   # consumes from global ω
    m_te = np.random.random(len(X_te)) < rate   # consumes from global ω

Effect: each perturbation call consumes from the global Mersenne Twister,
coupling mask geometry to the global state ω — activating FM-2 even though
RHOA-1 has declared ω = 0.

EXPECTED OUTCOMES
-----------------
  Case A — RHOA-2 matters independently:
    ORB_C < ORB_A (e.g., ORB_C ≈ 0.30–0.50)
    → RHOA-2 directly suppresses ORB even when global seed is declared.
    → FM-2 (mask geometry coupling) is activated by RHOA-2 violation alone.

  Case B — RHOA-2's contribution is primarily cross-run stability:
    ORB_C ≈ ORB_A (e.g., ORB_C = 0.70)
    → With global seed fixed, isolated vs. global perturbation RNG
      produces identical masks at each (seed, rate) combination
      because the global state is reproducible from ω=0.
    → RHOA-2's value: prevents ω from drifting across iterative
      calls, not changing the point estimate when ω is fixed.

  Either outcome is scientifically valuable:
    Case A → Strengthens RHOA-2 as an independent protocol requirement.
    Case B → Clarifies RHOA-2 as a guard against cross-call ω drift,
             complementary to RHOA-1 rather than independent.

COMPARISON TABLE (paper-ready)
-------------------------------
  | Configuration | RHOA-1 | RHOA-2 | ORB S1 | ORB S2 | FM-2 |
  |---------------|--------|--------|--------|--------|------|
  | Run A         |   ✓    |   ✓    |  0.70  |  0.70  |  No  |
  | Run C (this)  |   ✓    |   ✗    |   ?    |   ?    |  ?   |
  | Run B         |   ✗    |   ✗    |  0.05  |  0.30  |  Yes |

OUTPUTS
-------
  Outputs/run_c_rhoa2_isolation/PHASE4_SEVERITY_SWEEP/   ← Run C Phase 4A
  Outputs/stats/rhoa2_isolation_orb_table.csv
  Outputs/stats/rhoa2_isolation_summary.txt
  Outputs/tables/TABLE_rhoa2_isolation.tex
  Outputs/stats/rhoa2_isolation_paper_text.txt
"""

import sys, os, re, subprocess, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config as C

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PHASE4A_SRC   = C.CODE_DIR / "phase4a_missingness_severity_sweep.py"
ENVELOPE_FILE = "robustness_envelope.csv"
STRESS_COL    = "miss_rate"
GLOBAL_SEED   = C.RUN_C_GLOBAL_SEED           # 0 (same as Run A — isolate RHOA-2)
OUT_DIR       = C.OUT_RUN_C

# Reference values from existing runs (for comparison table)
RUN_A_ORB = {"S1": 0.70, "S2": 0.70}
RUN_B_ORB = {"S1": 0.05, "S2": 0.30}


# ─────────────────────────────────────────────────────────────────────────────
# Real-time streaming subprocess
# ─────────────────────────────────────────────────────────────────────────────

def _stream_subprocess(cmd: list, cwd: str, label: str,
                       timeout: int = None) -> int:
    """Stream subprocess stdout line-by-line via tqdm.write. Returns returncode."""
    proc = subprocess.Popen(
        cmd, cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    t0 = time.time()
    prefix = f"  [{label}]"
    try:
        for line in proc.stdout:
            if timeout is not None and time.time() - t0 > timeout:
                proc.kill()
                tqdm.write(f"{prefix} [TIMEOUT]")
                return -1
            line = line.rstrip()
            if not line:
                continue
            clean = re.sub(r'\x1b\[[0-9;]*[mK]', '', line)
            clean = clean.strip().split("\r")[-1].strip()
            if clean:
                tqdm.write(f"{prefix}  {clean}")
    finally:
        proc.wait()
    return proc.returncode


# ─────────────────────────────────────────────────────────────────────────────
# Script patching: construct Run C from phase4a source
# ─────────────────────────────────────────────────────────────────────────────

def read_script(path: Path) -> str:
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc, errors="replace")
        except Exception:
            continue
    return path.read_bytes().decode("utf-8", errors="replace")


def create_run_c_script(src: Path, dst: Path) -> tuple:
    """
    Patch phase4a to produce Run C:
      - RHOA-1 ✓: inject global seed declaration at start of main()
      - RHOA-2 ✗: replace isolated default_rng calls in
                   apply_mcar_missingness_split() with global np.random calls

    Returns (dst_path, n_rhoa1_injections, n_rhoa2_patches)
    """
    content = read_script(src)
    original_content = content

    # ── RHOA-1: Inject global seed at start of main() ─────────────────────────
    # Also patch dataset paths to absolute (phase1_3_main_audit_runner uses
    # relative paths that only resolve when cwd == Data/Dataset/).
    _dataset_dir = PHASE4A_SRC.parent.parent / "Dataset"
    _ds_a = str(_dataset_dir / "full_analytic_dataset_mortality_all_admissions.csv").replace("\\", "/")
    _ds_b = str(_dataset_dir / "Synthetic_Dataset_1500_Patients_precise.csv").replace("\\", "/")

    rhoa1_block = (
        f"\n    # ── RHOA-1 ✓: Global State Declaration ──────────────────────────\n"
        f"    # Run C: global seed declared (ω = {GLOBAL_SEED}) — RHOA-1 compliant\n"
        f"    import random as _r14; _r14.seed({GLOBAL_SEED})\n"
        f"    import numpy as _np14; _np14.random.seed({GLOBAL_SEED})\n"
        f"    # Fix dataset paths to absolute so cwd doesn't matter\n"
        f"    import phase1_3_main_audit_runner as _runner14\n"
        f"    _runner14.DATASET_A_PATH = r'{_ds_a}'\n"
        f"    _runner14.DATASET_B_PATH = r'{_ds_b}'\n"
        f"    # ── RHOA-2 ✗: Perturbation uses global RNG (see patched function) ─\n"
        f"    # ─────────────────────────────────────────────────────────────────\n"
    )
    n_rhoa1 = content.count("def main():\n")
    content = content.replace("def main():\n", f"def main():{rhoa1_block}", 1)

    # ── RHOA-2: Patch apply_mcar_missingness_split to violate RNG isolation ───
    #
    # TARGET pattern (verbatim from phase4a lines 82–93):
    #   rng_tr = np.random.default_rng(seed + 12345 + int(rate * 1000))
    #   rng_te = np.random.default_rng(seed + 54321 + int(rate * 1000))
    #   for c in num_cols:
    #       ...
    #       m_tr = rng_tr.random(len(X_tr2)) < rate
    #       m_te = rng_te.random(len(X_te2)) < rate
    #
    # REPLACEMENT (RHOA-2 violated — global RNG consumes ω):
    #   # rng_tr / rng_te REMOVED — global np.random used instead
    #   for c in num_cols:
    #       ...
    #       m_tr = np.random.random(len(X_tr2)) < rate  # RHOA-2 ✗
    #       m_te = np.random.random(len(X_te2)) < rate  # RHOA-2 ✗

    # Step A: Remove isolated RNG instantiation lines
    rng_init_pattern = re.compile(
        r"[ \t]*rng_tr\s*=\s*np\.random\.default_rng\([^)]+\)\s*\n"
        r"[ \t]*rng_te\s*=\s*np\.random\.default_rng\([^)]+\)\s*\n",
        re.MULTILINE,
    )
    rhoa2_removed_header = (
        "    # ── RHOA-2 ✗: Isolated RNG removed — global np.random used ──────\n"
        "    # rng_tr = np.random.default_rng(...)  ← DISABLED (RHOA-2 violated)\n"
        "    # rng_te = np.random.default_rng(...)  ← DISABLED (RHOA-2 violated)\n"
    )
    content_after_a, n_a = rng_init_pattern.subn(rhoa2_removed_header, content)
    content = content_after_a

    # Step B: Replace rng_tr.random(...) < rate with np.random.random(...) < rate
    # Pattern handles one level of nested parens, e.g. len(X_tr2).
    # IMPORTANT: also capture the trailing comparison (< rate) so the comment
    # is placed AFTER it, not before — otherwise "# comment < rate" makes
    # Python treat < rate as dead comment text and m_tr gets raw floats.
    pattern_tr = re.compile(
        r'\brng_tr\.random\(((?:[^()]*|\([^()]*\))*)\)(\s*<\s*\w+)'
    )
    content_after_b, n_b = pattern_tr.subn(
        r'np.random.random(\1)\2  # RHOA-2 ✗ global RNG', content
    )
    content = content_after_b

    # Step C: Replace rng_te.random(...) < rate with np.random.random(...) < rate
    pattern_te = re.compile(
        r'\brng_te\.random\(((?:[^()]*|\([^()]*\))*)\)(\s*<\s*\w+)'
    )
    content_after_c, n_c = pattern_te.subn(
        r'np.random.random(\1)\2  # RHOA-2 ✗ global RNG', content
    )
    content = content_after_c

    # ── Redirect OUT_DIR ──────────────────────────────────────────────────────
    run_c_out = str(OUT_DIR / "PHASE4_SEVERITY_SWEEP").replace("\\", "/")
    content = re.sub(
        r'OUT_DIR\s*=\s*["\'][^"\']*["\']',
        f'OUT_DIR = "{run_c_out}"',
        content,
    )

    # ── Step D: Inject checkpoint / resume logic ───────────────────────────────
    #
    # D0: Prevent rankings_map reset from wiping loaded checkpoint data.
    #     Original: rankings_map[(split_key, miss_rate)] = {}
    #     Patched:  rankings_map[(split_key, miss_rate)] = rankings_map.get(...)
    content = content.replace(
        "            rankings_map[(split_key, miss_rate)] = {}\n",
        "            rankings_map[(split_key, miss_rate)] = rankings_map.get((split_key, miss_rate), {})\n",
    )

    # D1: After `rankings_map: Dict[...] = {}` — inject checkpoint-load block.
    ckpt_load_block = (
        "\n"
        "    # ── CHECKPOINT / RESUME (injected by 14_rhoa2_isolation.py) ──────\n"
        "    import json as _json\n"
        "    _CKPT_FOLD    = os.path.join(OUT_DIR, '_ckpt_fold_metrics.csv')\n"
        "    _CKPT_SUMM    = os.path.join(OUT_DIR, '_ckpt_model_summaries.csv')\n"
        "    _CKPT_WINNERS = os.path.join(OUT_DIR, '_ckpt_winners.csv')\n"
        "    _CKPT_RANKS   = os.path.join(OUT_DIR, '_ckpt_rankings.json')\n"
        "    _completed = set()\n"
        "    if os.path.exists(_CKPT_WINNERS):\n"
        "        _cw = pd.read_csv(_CKPT_WINNERS)\n"
        "        _completed = set(zip(_cw['split'], _cw['miss_rate'].astype(float), _cw['seed'].astype(int)))\n"
        "        winners_rows = _cw.to_dict('records')\n"
        "        tqdm.write(f'[RESUME] {len(_completed)} (split,rate,seed) combos already done — skipping.')\n"
        "        if os.path.exists(_CKPT_FOLD):\n"
        "            _cf = pd.read_csv(_CKPT_FOLD)\n"
        "            # Filter out any orphaned partial rows (crash between fold/summary write and winners write)\n"
        "            _mask_f = _cf.apply(lambda r: (r['split'], float(r['miss_rate']), int(r['seed'])) in _completed, axis=1)\n"
        "            all_fold_metrics = [_cf[_mask_f]]\n"
        "        if os.path.exists(_CKPT_SUMM):\n"
        "            _cs = pd.read_csv(_CKPT_SUMM)\n"
        "            _mask_s = _cs.apply(lambda r: (r['split'], float(r['miss_rate']), int(r['seed'])) in _completed, axis=1)\n"
        "            all_model_summaries = [_cs[_mask_s]]\n"
        "        if os.path.exists(_CKPT_RANKS):\n"
        "            _rmap = _json.load(open(_CKPT_RANKS))\n"
        "            for _k, _v in _rmap.items():\n"
        "                _sp, _rt = _k.split('|')\n"
        "                rankings_map[(str(_sp), float(_rt))] = {int(s): lst for s, lst in _v.items()}\n"
        "    # ────────────────────────────────────────────────────────────────────\n"
    )
    content = content.replace(
        "    rankings_map: Dict[Tuple[str, float], Dict[int, List[str]]] = {}\n",
        "    rankings_map: Dict[Tuple[str, float], Dict[int, List[str]]] = {}\n" + ckpt_load_block,
    )

    # D2: Inside `for seed in seed_iter:` — skip already-completed combos.
    ckpt_skip_block = (
        "                if (split_key, miss_rate, seed) in _completed:\n"
        "                    tqdm.write(f'[SKIP] split={split_key} r={miss_rate:.2f} s={seed} (checkpoint)')\n"
        "                    continue\n"
    )
    content = content.replace(
        "            for seed in seed_iter:\n"
        "                out = eval_one_setting(",
        "            for seed in seed_iter:\n"
        + ckpt_skip_block
        + "                out = eval_one_setting(",
    )

    # D3: After `rankings_map[(split_key, miss_rate)][seed] = out["ranking"]`
    #     — save checkpoint (write winners LAST so it acts as the commit signal).
    ckpt_save_block = (
        "\n\n"
        "                # ── CHECKPOINT SAVE (per seed) — injected by 14_rhoa2_isolation.py ─\n"
        "                _fold_part = out['fold_metrics']\n"
        "                _summ_part = out['model_summary']\n"
        "                _fold_part.to_csv(_CKPT_FOLD, mode='a', header=not os.path.exists(_CKPT_FOLD), index=False)\n"
        "                _summ_part.to_csv(_CKPT_SUMM, mode='a', header=not os.path.exists(_CKPT_SUMM), index=False)\n"
        "                _rmap_serial = {f'{k[0]}|{k[1]}': {str(s): lst for s, lst in v.items()} for k, v in rankings_map.items()}\n"
        "                with open(_CKPT_RANKS, 'w') as _f: _json.dump(_rmap_serial, _f)\n"
        "                # winners written last — crash before this = seed re-runs cleanly on resume\n"
        "                _wr = pd.DataFrame([winners_rows[-1]])\n"
        "                _wr.to_csv(_CKPT_WINNERS, mode='a', header=not os.path.exists(_CKPT_WINNERS), index=False)\n"
        "                _completed.add((split_key, miss_rate, seed))\n"
        "                # ──────────────────────────────────────────────────────────────────\n"
    )
    content = content.replace(
        "                rankings_map[(split_key, miss_rate)][seed] = out[\"ranking\"]\n",
        "                rankings_map[(split_key, miss_rate)][seed] = out[\"ranking\"]\n" + ckpt_save_block,
    )

    # D4: After final CSV writes — delete checkpoint files (clean run succeeded).
    ckpt_cleanup_block = (
        "\n"
        "    # ── CLEANUP CHECKPOINTS (run completed successfully) ───────────────\n"
        "    for _cf in [_CKPT_FOLD, _CKPT_SUMM, _CKPT_WINNERS, _CKPT_RANKS]:\n"
        "        if os.path.exists(_cf):\n"
        "            os.remove(_cf)\n"
        "            tqdm.write(f'[CKPT] Removed {_cf}')\n"
        "    # ────────────────────────────────────────────────────────────────────\n"
    )
    content = content.replace(
        '    env_df.to_csv(os.path.join(OUT_DIR, "robustness_envelope.csv"), index=False)\n'
        '    onset_df.to_csv(os.path.join(OUT_DIR, "flip_onset.csv"), index=False)\n',
        '    env_df.to_csv(os.path.join(OUT_DIR, "robustness_envelope.csv"), index=False)\n'
        '    onset_df.to_csv(os.path.join(OUT_DIR, "flip_onset.csv"), index=False)\n'
        + ckpt_cleanup_block,
    )

    n_rhoa2_patches = n_b + n_c

    if not content.startswith("# -*- coding"):
        content = "# -*- coding: utf-8 -*-\n" + content

    dst.write_text(content, encoding="utf-8")
    return dst, n_rhoa1, n_rhoa2_patches


# ─────────────────────────────────────────────────────────────────────────────
# ORB computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_orb_from_envelope(envelope_path: Path) -> dict:
    """Compute ORB and flip_onset per split from robustness_envelope.csv."""
    if not envelope_path.exists():
        return {s: {"orb": float("nan"), "flip_onset": float("nan")}
                for s in C.SPLITS}

    df = pd.read_csv(envelope_path)
    result = {}
    for split in C.SPLITS:
        sub = df[df["split"] == split].sort_values(STRESS_COL).reset_index(drop=True)
        orb        = 0.0
        flip_onset = float("nan")

        for _, row in sub.iterrows():
            flip_ok = float(row.get("winner_flip_pct", 100.0)) <= C.DEFAULT_FLIP_THR
            tau_ok  = float(row.get("kendall_tau_mean",  0.0))  >= C.DEFAULT_TAU_THR
            if flip_ok and tau_ok:
                orb = float(row[STRESS_COL])
            else:
                break

        for _, row in sub.iterrows():
            if float(row.get("winner_flip_pct", 0.0)) > 0 and float(row[STRESS_COL]) > 0:
                flip_onset = float(row[STRESS_COL])
                break

        result[split] = {"orb": orb, "flip_onset": flip_onset}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess runner
# ─────────────────────────────────────────────────────────────────────────────

def run_c_phase4a() -> dict:
    """Patch phase4a to create Run C, execute as subprocess, return ORB results."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tmp_script = PHASE4A_SRC.parent / "_run_c_rhoa2_violation_temp.py"
    script_path, n_rhoa1, n_rhoa2 = create_run_c_script(PHASE4A_SRC, tmp_script)

    print(f"\n{'─'*65}")
    print(f"  Run C: RHOA-2 Isolation  [{datetime.now().strftime('%H:%M:%S')}]")
    print(f"  Global seed = {GLOBAL_SEED} (RHOA-1 ✓)")
    print(f"  RHOA-2 violation patches: {n_rhoa2} (isolated→global RNG)")
    print(f"  Output: {OUT_DIR}")
    print(f"{'─'*65}")

    if n_rhoa2 == 0:
        print("  [WARNING] No RHOA-2 patches applied — pattern not found.")
        print("  Check that phase4a source matches expected structure.")

    t0 = time.time()
    rc = _stream_subprocess(
        [sys.executable, str(tmp_script)],
        cwd=str(PHASE4A_SRC.parent),
        label="Run C | RHOA-2✗",
        timeout=None,            # no timeout — run until completion
    )
    elapsed = time.time() - t0
    if tmp_script.exists():
        try: tmp_script.unlink()
        except: pass

    if rc != 0:
        print(f"  ✗ Run C FAILED  (rc={rc}, {elapsed:.0f}s)")
        return {"returncode": rc, "elapsed": elapsed,
                "S1": {"orb": float("nan"), "flip_onset": float("nan")},
                "S2": {"orb": float("nan"), "flip_onset": float("nan")},
                "n_rhoa2_patches": n_rhoa2}

    env_path = OUT_DIR / "PHASE4_SEVERITY_SWEEP" / ENVELOPE_FILE
    orb_data = compute_orb_from_envelope(env_path)

    orb_str = "  ".join(f"{s}={orb_data[s]['orb']:.3f}" for s in C.SPLITS)
    print(f"  ✓ Run C OK  ({elapsed:.0f}s)  ORB: {orb_str}")

    return {"returncode": rc, "elapsed": elapsed,
            "n_rhoa2_patches": n_rhoa2, **orb_data}


# ─────────────────────────────────────────────────────────────────────────────
# Analysis and reporting
# ─────────────────────────────────────────────────────────────────────────────

def classify_rhoa2_effect(orb_c: float, orb_a: float, orb_b: float,
                           threshold: float = 0.05) -> str:
    """
    Classify the RHOA-2 effect:
      - 'INDEPENDENT': ORB_C significantly below ORB_A → RHOA-2 has own contribution
      - 'COMPLEMENTARY': ORB_C ≈ ORB_A → RHOA-2 guards cross-run drift, not point estimate
      - 'PARTIAL': Intermediate case
    """
    if np.isnan(orb_c) or np.isnan(orb_a):
        return "UNKNOWN"
    delta = orb_a - orb_c
    if delta > threshold:
        return "INDEPENDENT"
    elif delta < -threshold:
        return "ANOMALY"
    else:
        return "COMPLEMENTARY"


def build_comparison_table(run_c_result: dict) -> pd.DataFrame:
    """Three-row comparison: Run A / Run C / Run B."""
    rows = []
    for split in C.SPLITS:
        orb_c = run_c_result[split]["orb"]
        orb_a = RUN_A_ORB[split]
        orb_b = RUN_B_ORB[split]

        effect    = classify_rhoa2_effect(orb_c, orb_a, orb_b)
        delta_ac  = orb_a - orb_c
        delta_bc  = orb_b - orb_c   # negative = C better than B

        rows.append({
            "split":          split,
            "run":            "A",
            "rhoa1":          "✓",
            "rhoa2":          "✓",
            "orb":            orb_a,
            "flip_onset":     float("nan"),
            "delta_vs_run_a": 0.0,
            "rhoa2_effect":   "reference",
        })
        rows.append({
            "split":          split,
            "run":            "C (this)",
            "rhoa1":          "✓",
            "rhoa2":          "✗",
            "orb":            orb_c,
            "flip_onset":     run_c_result[split]["flip_onset"],
            "delta_vs_run_a": -delta_ac,
            "rhoa2_effect":   effect,
        })
        rows.append({
            "split":          split,
            "run":            "B",
            "rhoa1":          "✗",
            "rhoa2":          "✗",
            "orb":            orb_b,
            "flip_onset":     float("nan"),
            "delta_vs_run_a": -(orb_a - orb_b),
            "rhoa2_effect":   "LURE fully active",
        })
    return pd.DataFrame(rows)


def make_latex_table(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[!t]",
        r"\caption{RHOA-2 Isolation Experiment (Run C): Three-Condition",
        r"  ORB Comparison Isolating Perturbation RNG Contribution}",
        r"\label{tab:rhoa2_isolation}",
        r"\centering\small",
        r"\begin{tabular}{llllrrl}",
        r"\toprule",
        r"Split & Run & RHOA-1 & RHOA-2 & ORB & $\Delta$ORB$_{\text{vs A}}$ & Interpretation \\",
        r"\midrule",
    ]

    for split in C.SPLITS:
        sub = df[df["split"] == split]
        first = True
        for _, row in sub.iterrows():
            orb_str   = f"${row['orb']:.2f}$" if not np.isnan(row["orb"]) else "---"
            delta_str = f"${row['delta_vs_run_a']:+.2f}$" if row["run"] != "A" else "---"
            effect_tex = row["rhoa2_effect"].replace("✓", "\\checkmark")

            if first:
                lines.append(
                    f"\\multirow{{3}}{{*}}{{{split}}} & "
                    f"{row['run']} & {row['rhoa1']} & {row['rhoa2']} & "
                    f"{orb_str} & {delta_str} & {effect_tex} \\\\"
                )
                first = False
            else:
                lines.append(
                    f" & {row['run']} & {row['rhoa1']} & {row['rhoa2']} & "
                    f"{orb_str} & {delta_str} & {effect_tex} \\\\"
                )
        lines.append(r"\midrule")

    lines += [
        r"\multicolumn{7}{l}{\footnotesize RHOA-2 violation: perturbation uses global \texttt{np.random}"
        r" instead of isolated \texttt{np.random.default\_rng(seed)}.}\\",
        r"\multicolumn{7}{l}{\footnotesize INDEPENDENT: $\Delta\text{ORB}_{\text{A-C}} > 0.05$"
        r" confirms RHOA-2 has independent empirical contribution.}\\",
        r"\multicolumn{7}{l}{\footnotesize COMPLEMENTARY: $\Delta\text{ORB}_{\text{A-C}} \approx 0$"
        r" confirms RHOA-2 guards cross-run RNG drift, not point estimate.}\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def make_figure(df: pd.DataFrame):
    """Three-bar comparison chart for each split: Run A / Run C / Run B."""
    runs   = ["A", "C (this)", "B"]
    colors = [C.COLOR_RUN_A, "#8E44AD", C.COLOR_RUN_B]   # blue / purple / red
    hatches = ["", "//", ""]
    labels  = ["Run A\n(RHOA-1✓ RHOA-2✓)", "Run C\n(RHOA-1✓ RHOA-2✗)", "Run B\n(LURE active)"]

    fig, axes = plt.subplots(1, len(C.SPLITS), figsize=(C.IEEE_DOUBLE_COL_INCH, 3.8),
                             constrained_layout=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

    if len(C.SPLITS) == 1:
        axes = [axes]

    for si, split in enumerate(C.SPLITS):
        ax  = axes[si]
        sub = df[df["split"] == split]

        orbs_ordered = []
        for run in runs:
            match = sub[sub["run"] == run]["orb"]
            orbs_ordered.append(
                float(match.iloc[0]) if len(match) > 0 and not np.isnan(match.iloc[0]) else 0.0
            )

        x = np.arange(len(runs))
        bars = ax.bar(
            x, orbs_ordered, width=0.55,
            color=colors, alpha=0.85, hatch=hatches,
            edgecolor="white", linewidth=0.8,
        )

        for bar, orb, label in zip(bars, orbs_ordered, labels):
            ax.text(bar.get_x() + bar.get_width() / 2, orb + 0.01,
                    f"{orb:.2f}", ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylim(0, 0.85)
        ax.set_ylabel("ORB Estimate", fontsize=9)
        ax.set_title(f"Split {split}", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

        # Annotate RHOA-2 effect
        sub_c = sub[sub["run"] == "C (this)"]
        if not sub_c.empty:
            effect = sub_c.iloc[0]["rhoa2_effect"]
            color  = "#1E8449" if effect == "COMPLEMENTARY" else "#C0392B"
            ax.text(0.5, 0.73, f"RHOA-2 effect:\n{effect}",
                    ha="center", va="top", fontsize=7.5,
                    transform=ax.transAxes, color=color,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec=color))

    fig.suptitle(
        "Run C: RHOA-2 Isolation Experiment\n"
        "(A=full compliance, C=RHOA-1✓/RHOA-2✗, B=LURE active)",
        fontsize=9, fontweight="bold",
    )

    out = C.OUT_FIGURES / "FigS6_RunC_RHOA2_isolation.jpg"
    fig.savefig(str(out), dpi=C.IEEE_DPI, bbox_inches="tight",
                format="jpg", pil_kwargs={"quality": 95})
    plt.close(fig)
    print(f"\n  Figure saved → {out.name}")


def generate_paper_text(df: pd.DataFrame, run_c_result: dict) -> str:
    lines = [
        "=" * 70,
        "  RHOA-2 ISOLATION EXPERIMENT (RUN C) — PAPER-READY TEXT",
        "=" * 70,
        "",
        "─" * 70,
        "  §5.x — RHOA-2 Empirical Isolation (NEW SECTION)",
        "─" * 70,
        "",
    ]

    for split in C.SPLITS:
        sub   = df[df["split"] == split]
        row_c = sub[sub["run"] == "C (this)"]
        if row_c.empty:
            continue
        orb_c  = float(row_c.iloc[0]["orb"])
        orb_a  = RUN_A_ORB[split]
        orb_b  = RUN_B_ORB[split]
        effect = row_c.iloc[0]["rhoa2_effect"]
        delta  = orb_a - orb_c

        if effect == "INDEPENDENT":
            text = (
                f"  '{split}: Run C (RHOA-1 compliant, RHOA-2 violated) yields\n"
                f"   ORB={orb_c:.2f}, compared with Run A ORB={orb_a:.2f} (full\n"
                f"   compliance) and Run B ORB={orb_b:.2f} (LURE fully active).\n"
                f"   The ΔORB = {delta:.2f} between Run C and Run A confirms that\n"
                f"   RHOA-2 (Perturbation RNG Isolation) has an independent empirical\n"
                f"   contribution beyond RHOA-1: global seed declaration alone is\n"
                f"   insufficient to prevent ORB degradation when perturbation\n"
                f"   injection couples mask geometry to the global state ω.\n"
                f"   FM-2 (mask geometry coupling) is activated by RHOA-2 violation\n"
                f"   even when ω is explicitly declared.'"
            )
        elif effect == "COMPLEMENTARY":
            text = (
                f"  '{split}: Run C (RHOA-1 compliant, RHOA-2 violated) yields\n"
                f"   ORB={orb_c:.2f} ≈ Run A ORB={orb_a:.2f} (ΔORB={delta:.2f}).\n"
                f"   When the global seed is declared (RHOA-1 ✓), the isolated vs.\n"
                f"   global perturbation RNG produces numerically equivalent masks\n"
                f"   at each (CV-seed, stress-level) combination, because the global\n"
                f"   state ω is reproducible from ω₀=0. RHOA-2's primary role is\n"
                f"   therefore to guard against ω drift across iterative calls in\n"
                f"   deployments where RHOA-1 may not be applied — a complementary\n"
                f"   safety layer rather than an independent ORB determinant.\n"
                f"   Contrast Run B (ORB={orb_b:.2f}): without RHOA-1, global RNG\n"
                f"   propagates ω uncertainty into mask geometry, activating FM-2.'"
            )
        else:
            text = f"  '{split}: Run C ORB={orb_c:.2f} (effect classification: {effect}).'"

        lines += [f"  {split}:", text, ""]

    lines += [
        "─" * 70,
        "  PLACEMENT: §5.3 or NEW §5.4 — insert after RHOA-3 direct results",
        "─" * 70,
        "",
        "=" * 70,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    C.OUT_STATS.mkdir(parents=True, exist_ok=True)
    C.OUT_TABLES.mkdir(parents=True, exist_ok=True)
    C.OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not PHASE4A_SRC.exists():
        print(f"[14] ERROR: Phase 4A script not found: {PHASE4A_SRC}")
        print("     Check C.CODE_DIR in config.py")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("  14_rhoa2_isolation.py")
    print("  Run C: RHOA-2 Isolation Experiment")
    print("=" * 65)
    print(f"  Global seed (RHOA-1 ✓) : {GLOBAL_SEED}")
    print(f"  RHOA-2 status          : ✗ (perturbation → global np.random)")
    print(f"  Phase 4A source        : {PHASE4A_SRC}")
    print(f"  Output                 : {OUT_DIR}")
    print(f"  Estimated time         : ~12h")
    print("=" * 65)

    # ── Run C ─────────────────────────────────────────────────────────────────
    run_c_result = run_c_phase4a()

    if run_c_result["returncode"] != 0:
        print(f"\n[14] Run C failed (returncode={run_c_result['returncode']})")
        sys.exit(1)

    # ── Post-processing with tqdm ─────────────────────────────────────────────
    post_steps = [
        ("Building comparison table", lambda: build_comparison_table(run_c_result)),
        ("Saving CSV",                None),
        ("Building LaTeX table",      None),
        ("Generating figure",         None),
        ("Writing paper text",        None),
    ]
    pbar = tqdm(total=len(post_steps), desc="Post-processing", unit="step")

    pbar.set_description("Building comparison table")
    df_compare = build_comparison_table(run_c_result)
    pbar.update(1)

    pbar.set_description("Saving CSV")
    df_compare.to_csv(
        C.OUT_STATS / "rhoa2_isolation_comparison.csv", index=False, encoding="utf-8"
    )
    pbar.update(1)

    pbar.set_description("Building LaTeX table")
    tex = make_latex_table(df_compare)
    (C.OUT_TABLES / "TABLE_rhoa2_isolation.tex").write_text(tex, encoding="utf-8")
    tqdm.write(f"  LaTeX table saved → TABLE_rhoa2_isolation.tex")
    pbar.update(1)

    pbar.set_description("Generating figure")
    make_figure(df_compare)
    pbar.update(1)

    pbar.set_description("Writing paper text")
    paper_text = generate_paper_text(df_compare, run_c_result)
    (C.OUT_STATS / "rhoa2_isolation_paper_text.txt").write_text(
        paper_text, encoding="utf-8"
    )
    pbar.update(1)
    pbar.close()

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  RUN C: RHOA-2 ISOLATION SUMMARY")
    print(f"{'='*65}")
    print()
    print(f"  {'Split':<8} {'Run A':<10} {'Run C':<10} {'Run B':<10}  Effect")
    print(f"  {'─'*55}")
    for split in C.SPLITS:
        sub = df_compare[df_compare["split"] == split]
        orb_a = float(sub[sub["run"] == "A"]["orb"].iloc[0])
        orb_c_row = sub[sub["run"] == "C (this)"]
        orb_c = float(orb_c_row["orb"].iloc[0]) if not orb_c_row.empty else float("nan")
        orb_b = float(sub[sub["run"] == "B"]["orb"].iloc[0])
        effect = orb_c_row["rhoa2_effect"].iloc[0] if not orb_c_row.empty else "UNKNOWN"
        print(f"  {split:<8} {orb_a:<10.3f} {orb_c:<10.3f} {orb_b:<10.3f}  {effect}")

    print(f"\n  RHOA-2 patches applied: {run_c_result['n_rhoa2_patches']}")
    print(f"\n[14] All outputs saved to {C.OUT_STATS}")
    print("[14] Done.")


if __name__ == "__main__":
    main()
