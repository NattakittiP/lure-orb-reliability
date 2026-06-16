"""
10_unseeded_batch.py  —  N=5 Unseeded Runs (Phase 4A + 4B)
============================================================
Runs phase4a and phase4b each 5 times with global seed DISABLED.
All 5 runs produced identical ORB boundaries — no value in running more.
Streams inner tqdm progress in real-time so you can see what's
happening inside each long-running subprocess.
"""

import sys, os, re, subprocess, threading, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config as C

N_RUNS     = C.N_UNSEEDED_RUNS   # 5  (all runs identical → sufficient)
SCRIPTS    = {
    "4A": C.CODE_DIR / "phase4a_missingness_severity_sweep.py",
    "4B": C.CODE_DIR / "phase4b_prevalence_shift_sweep.py",
}
OUT_DIR_VAR = {"4A": "PHASE4_SEVERITY_SWEEP", "4B": "PHASE4B_PREVALENCE_SHIFT"}
ENVELOPE_FILE = {
    "4A": "robustness_envelope.csv",
    "4B": "robustness_envelope_prevalence_shift.csv",
}
STRESS_COL = {"4A": "miss_rate", "4B": "target_prev"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_script(path: Path) -> str:
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc, errors="replace")
        except Exception:
            continue
    return path.read_bytes().decode("utf-8", errors="replace")


def create_unseeded_script(src: Path, dst: Path) -> Path:
    content = read_script(src)
    import re as _re
    def _comment_seed(m):
        indent = m.group(1); call = m.group(2)
        return f"{indent}# {call}  # LURE UNCONTROLLED"
    content = _re.sub(
        r'^([ \t]*)(?<!#\s)((?:random|np\.random)\.seed\(0\))',
        _comment_seed, content, flags=_re.MULTILINE
    )
    if not content.startswith("# -*- coding"):
        content = "# -*- coding: utf-8 -*-\n" + content
    dst.write_text(content, encoding="utf-8")
    return dst


def patch_out_dir(content: str, out_dir_str: str) -> str:
    fwd = out_dir_str.replace("\\", "/")
    return re.sub(r'OUT_DIR\s*=\s*["\'][^"\']+["\']',
                  lambda m: f'OUT_DIR = "{fwd}"', content)


def compute_orb(envelope_path: Path, stress_col: str) -> dict:
    if not envelope_path.exists():
        return {}
    df = pd.read_csv(envelope_path)
    result = {}
    for split in C.SPLITS:
        sub = df[df["split"] == split].sort_values(stress_col)
        orb = 0.0; first_flip = None
        for _, row in sub.iterrows():
            if (float(row["winner_flip_pct"]) <= C.DEFAULT_FLIP_THR and
                    float(row["kendall_tau_mean"]) >= C.DEFAULT_TAU_THR):
                orb = float(row[stress_col])
            else:
                break
        for _, row in sub.iterrows():
            if float(row["winner_flip_pct"]) > 0:
                first_flip = float(row[stress_col]); break
        result[split] = {"orb": orb,
                         "flip_onset": first_flip if first_flip else float("nan")}
    return result


def fmt_duration(secs: float) -> str:
    h, rem = divmod(int(secs), 3600)
    m, s   = divmod(rem, 60)
    if h > 0: return f"{h}h {m:02d}m {s:02d}s"
    if m > 0: return f"{m}m {s:02d}s"
    return f"{s}s"


# ─────────────────────────────────────────────────────────────────────────────
# Real-time streaming subprocess
# ─────────────────────────────────────────────────────────────────────────────

def stream_subprocess(cmd, cwd, run_id, phase, outer_bar):
    """
    Run subprocess and stream its stdout/stderr line-by-line.
    Returns (returncode, elapsed_seconds).
    """
    t0 = time.time()
    prefix = f"  [Phase {phase} | Run {run_id:02d}]"

    proc = subprocess.Popen(
        cmd, cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,        # merge stderr into stdout
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,                        # line-buffered
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    last_line = ""
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        # Print inner progress lines (tqdm or print statements)
        # Overwrite the same console line for tqdm-style updates
        if "\r" in line or line.startswith("\x1b"):
            # tqdm carriage-return style — print cleanly
            clean = re.sub(r'\x1b\[[0-9;]*[mK]', '', line)
            clean = clean.strip().split("\r")[-1].strip()
            if clean:
                tqdm.write(f"{prefix}  {clean}")
        else:
            tqdm.write(f"{prefix}  {line}")
        last_line = line

    proc.wait()
    elapsed = time.time() - t0
    return proc.returncode, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Run batch for one phase
# ─────────────────────────────────────────────────────────────────────────────

def run_phase_batch(phase: str) -> pd.DataFrame:
    script_src  = SCRIPTS[phase]
    stress_col  = STRESS_COL[phase]
    env_file    = ENVELOPE_FILE[phase]
    out_var     = OUT_DIR_VAR[phase]

    if not script_src.exists():
        print(f"[10] ERROR: {script_src} not found"); return pd.DataFrame()

    phase_out = C.OUT_N25 / phase
    phase_out.mkdir(parents=True, exist_ok=True)

    # Create unseeded base script
    unseeded_base = script_src.parent / f"_lure_unseeded_{phase}.py"
    create_unseeded_script(script_src, unseeded_base)

    orb_rows   = []
    failed     = []
    run_times  = []
    t_phase    = time.time()

    # ── Outer progress bar: one tick per run ─────────────────────────────────
    outer_bar = tqdm(
        range(1, N_RUNS + 1),
        desc=f"Phase {phase} unseeded batch",
        unit="run",
        position=0,
        leave=True,
        bar_format="{l_bar}{bar}| {n}/{total} runs [elapsed {elapsed} | ETA {remaining}] {postfix}",
    )

    for run_id in outer_bar:
        run_dir     = phase_out / f"run_{run_id:02d}"
        run_dir.mkdir(exist_ok=True)
        run_out_dir = str(run_dir / out_var)

        # Patch OUT_DIR in temp script
        per_run = script_src.parent / "_lure_run_temp.py"
        patched = patch_out_dir(read_script(unseeded_base), run_out_dir)
        per_run.write_text(patched, encoding="utf-8")

        # Update bar description
        eta_str = ""
        if run_times:
            avg = sum(run_times) / len(run_times)
            remaining = avg * (N_RUNS - run_id + 1)
            eta_str = f"  avg/run={fmt_duration(avg)}  ETA≈{fmt_duration(remaining)}"
        outer_bar.set_postfix_str(
            f"run {run_id}/{N_RUNS}  failed={len(failed)}{eta_str}",
            refresh=True
        )

        tqdm.write(f"\n{'─'*60}")
        tqdm.write(f"  Phase {phase} | Run {run_id}/{N_RUNS}  "
                   f"[{datetime.now().strftime('%H:%M:%S')}]")
        tqdm.write(f"{'─'*60}")

        rc, elapsed = stream_subprocess(
            [sys.executable, str(per_run)],
            cwd=str(script_src.parent),
            run_id=run_id,
            phase=phase,
            outer_bar=outer_bar,
        )
        run_times.append(elapsed)

        if rc != 0:
            tqdm.write(f"  ✗ Run {run_id} FAILED  ({fmt_duration(elapsed)})")
            failed.append(run_id)
        else:
            env_path = Path(run_out_dir) / env_file
            orb_data = compute_orb(env_path, stress_col)
            for split, data in orb_data.items():
                orb_rows.append({
                    "phase": phase, "run_id": run_id, "split": split,
                    "orb_estimate": data["orb"],
                    "flip_onset":   data["flip_onset"],
                })
            orb_s = {s: orb_rows[-1]["orb_estimate"]
                     for s in C.SPLITS
                     if any(r["run_id"]==run_id and r["split"]==s for r in orb_rows)}
            orb_str = "  ".join(f"{s}={v:.3f}" for s,v in orb_s.items())
            tqdm.write(f"  ✓ Run {run_id} OK  ({fmt_duration(elapsed)})  ORB: {orb_str}")

        outer_bar.update(0)   # refresh postfix

    outer_bar.close()

    # Cleanup temp files
    for tmp in [unseeded_base, script_src.parent / "_lure_run_temp.py"]:
        if tmp.exists():
            try: tmp.unlink()
            except: pass

    # ── Phase summary ─────────────────────────────────────────────────────────
    df = pd.DataFrame(orb_rows)
    total_elapsed = time.time() - t_phase
    print(f"\n{'='*60}")
    print(f"Phase {phase} complete:  {N_RUNS - len(failed)}/{N_RUNS} succeeded  "
          f"({fmt_duration(total_elapsed)} total)")
    if failed:
        print(f"  Failed runs: {failed}")
    if not df.empty:
        df.to_csv(phase_out / "orb_distribution.csv", index=False)
        for split in C.SPLITS:
            sub = df[df["split"] == split]["orb_estimate"].dropna()
            if len(sub) > 0:
                p5, p95 = np.percentile(sub, [5, 95])
                print(f"  {split}: ORB={sub.mean():.3f}±{sub.std():.3f}  "
                      f"90%range=[{p5:.3f},{p95:.3f}]  "
                      f"vs Run A={0.70:.2f}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate + figure
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_and_plot(df_all: pd.DataFrame):
    if df_all.empty:
        print("[10] No data to aggregate."); return

    agg_rows = []
    for (phase, split), grp in df_all.groupby(["phase","split"]):
        orb   = grp["orb_estimate"].dropna()
        onset = grp["flip_onset"].dropna()
        if len(orb) == 0: continue
        p5, p95 = np.percentile(orb, [5, 95])
        agg_rows.append({
            "phase":phase,"split":split,"n_runs":len(orb),
            "orb_mean":orb.mean(),"orb_std":orb.std(),
            "orb_p5":p5,"orb_p95":p95,
            "orb_min":orb.min(),"orb_max":orb.max(),
            "onset_mean":onset.mean() if len(onset)>0 else float("nan"),
            "onset_std":onset.std()   if len(onset)>0 else float("nan"),
            "run_A_orb":0.70,
        })
    df_agg = pd.DataFrame(agg_rows)
    df_agg.to_csv(C.OUT_N25 / "aggregate_summary.csv", index=False)

    # ── Figure ───────────────────────────────────────────────────────────────
    phases = sorted(df_all["phase"].unique())
    fig, axes = plt.subplots(len(phases), 2,
                              figsize=(13, 4.5*len(phases)),
                              constrained_layout=True)
    if len(phases) == 1: axes = axes[np.newaxis, :]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})

    for pi, phase in enumerate(phases):
        sl = "MCAR Rate (δ)" if phase=="4A" else "Target Prevalence (π)"
        for si, split in enumerate(C.SPLITS):
            ax  = axes[pi, si]
            sub = df_all[(df_all["phase"]==phase)&(df_all["split"]==split)][
                "orb_estimate"].dropna()
            if sub.empty:
                ax.text(0.5,0.5,"No data",transform=ax.transAxes,ha="center"); continue

            ax.hist(sub, bins=min(12,len(sub)), color=C.COLOR_RUN_B,
                    alpha=0.75, edgecolor="white", linewidth=0.8)
            ax.axvline(sub.mean(), color="black", lw=2.0, ls="--",
                       label=f"Run B mean = {sub.mean():.3f}")
            ax.axvline(0.70, color=C.COLOR_RUN_A, lw=2.0, ls=":",
                       label=f"Run A (seeded) = 0.70")
            p5, p95 = np.percentile(sub, [5, 95])
            ax.axvspan(p5, p95, alpha=0.08, color=C.COLOR_RUN_B,
                       label=f"90% range [{p5:.3f},{p95:.3f}]")
            ax.set_xlabel(f"ORB Estimate ({sl})", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.set_title(f"Phase {phase} | {split}  (N={len(sub)} runs)\n"
                         f"mean={sub.mean():.3f}±{sub.std():.3f}  "
                         f"vs Run A=0.70",
                         fontsize=11, pad=8)
            ax.legend(fontsize=9, loc="upper left")
            ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"LURE ORB Distribution — N={N_RUNS} Unseeded Runs\n"
        "Confirms stochastic ORB uncertainty under uncontrolled RNG state",
        fontsize=12, fontweight="bold"
    )
    out = C.OUT_FIGURES / "FigS4_LURE_unseeded_ORB_distribution.jpg"
    fig.savefig(str(out), dpi=200, bbox_inches="tight",
                format="jpg", pil_kwargs={"quality":95})
    plt.close(fig)
    print(f"\nFigure saved: {out.name}")

    # ── Paper-ready summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PAPER-READY NUMBERS (N={N_RUNS} unseeded runs)")
    print(f"{'='*60}")
    for _, row in df_agg.iterrows():
        print(f"\nPhase {row['phase']} {row['split']}:")
        print(f"  ORB unseeded : {row['orb_mean']:.3f} ± {row['orb_std']:.3f}  "
              f"90%=[{row['orb_p5']:.3f},{row['orb_p95']:.3f}]")
        print(f"  ORB seeded   : {row['run_A_orb']:.2f}")
        print(f"  Gap          : {row['run_A_orb'] - row['orb_mean']:.3f}")
    df_agg.to_csv(C.OUT_N25/"aggregate_summary.csv", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    C.OUT_N25.mkdir(parents=True, exist_ok=True)
    C.OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print(f"LURE Unseeded Batch Runner")
    print(f"N={N_RUNS} runs × 2 phases = {N_RUNS*2} total executions")
    print("="*60)

    all_dfs = []
    for phase in tqdm(["4A","4B"], desc="Overall phases", unit="phase",
                      position=0, leave=True):
        print(f"\n{'═'*60}")
        print(f"  Starting Phase {phase}")
        print(f"{'═'*60}")
        df = run_phase_batch(phase)
        all_dfs.append(df)

    df_all = pd.concat([d for d in all_dfs if not d.empty], ignore_index=True)
    aggregate_and_plot(df_all)
    pr