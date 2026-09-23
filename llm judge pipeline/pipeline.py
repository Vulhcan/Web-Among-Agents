"""
Automated overnight pipeline for the Among Us deception benchmark.

Steps:
  1. Discover all *-human-* data folders at repo root
  2. Label impostor speeches in parallel across folders (llm_judge.py --all-experiments)
  3. Compute metrics for each folder (analyze_metrics.py)
  4. Compute ELO ratings (elo.py)

Usage:
    python pipeline.py                    # label missing only, run metrics + ELO
    python pipeline.py --force            # re-label everything (asks for confirmation)
    python pipeline.py --force --yes      # re-label without confirmation prompt
    python pipeline.py --label-only       # labelling step only, skip metrics + ELO
    python pipeline.py --metrics-only     # skip labelling, run metrics + ELO only
    python pipeline.py --workers 4        # parallel workers for labelling (default: 3)
    python pipeline.py --folders claude-haiku-4.5-human-imp gemini-3-flash-human-crew

Error log: pipeline_errors.log  (appended on each run)
"""

import argparse
import re
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: tqdm not installed — progress bars disabled. pip install tqdm")

ROOT       = Path(__file__).parent
WEB_APP    = ROOT / "web-app"
METRICS    = ROOT / "metrics"
METRICS.mkdir(exist_ok=True)
ERROR_LOG  = METRICS / "pipeline_errors.log"


# ── Logging ───────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_error(step: str, folder: str, exc: Exception, extra: str = ""):
    entry = (
        f"\n[{_ts()}] ERROR in {step} — folder: {folder}\n"
        f"  {type(exc).__name__}: {exc}\n"
    )
    if extra:
        entry += f"  {extra}\n"
    entry += "-" * 60
    with ERROR_LOG.open("a", encoding="utf-8") as f:
        f.write(entry + "\n")
    print(entry)


def log_info(msg: str):
    line = f"[{_ts()}] {msg}"
    print(line)


# ── Folder discovery ──────────────────────────────────────────────────────────

def discover_folders(root: Path) -> list:
    return sorted(
        d for d in root.iterdir()
        if d.is_dir() and re.match(r".+-human-(imp|crew)$", d.name)
    )


# ── Step 1: Labelling ─────────────────────────────────────────────────────────

def label_folder(folder: Path, force: bool) -> tuple:
    """Run llm_judge.py for one folder. Returns (folder_name, success, duration_s)."""
    t0  = time.time()
    cmd = [
        sys.executable,
        str(WEB_APP / "llm_judge.py"),
        "--folder", str(folder),
        "--all-experiments",
    ]
    if force:
        cmd.append("--force")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=7200,   # 2-hour timeout per folder
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            err = RuntimeError(
                f"llm_judge.py exited with code {result.returncode}\n"
                f"stderr: {result.stderr[-1000:]}"
            )
            log_error("labelling", folder.name, err)
            return (folder.name, False, elapsed)
        return (folder.name, True, elapsed)
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        err = TimeoutError(f"llm_judge.py timed out after {elapsed/60:.0f} min")
        log_error("labelling", folder.name, err)
        return (folder.name, False, elapsed)
    except Exception as exc:
        elapsed = time.time() - t0
        log_error("labelling", folder.name, exc, traceback.format_exc())
        return (folder.name, False, elapsed)


def run_labelling(folders: list, workers: int, force: bool):
    log_info(f"── STEP 1: LABELLING ({len(folders)} folders, {workers} workers) ──")

    successes, failures = [], []

    if HAS_TQDM:
        pbar = tqdm(total=len(folders), desc="Labelling", unit="folder", ncols=80)
    else:
        pbar = None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(label_folder, f, force): f for f in folders}
        for future in as_completed(futures):
            folder_name, ok, elapsed = future.result()
            status = "OK" if ok else "FAILED"
            msg    = f"  {folder_name:40s} [{status}] {elapsed:.0f}s"
            if pbar:
                pbar.set_postfix_str(f"{folder_name} {status}")
                pbar.update(1)
            log_info(msg)
            (successes if ok else failures).append(folder_name)

    if pbar:
        pbar.close()

    log_info(f"Labelling complete: {len(successes)} OK, {len(failures)} failed")
    if failures:
        log_info(f"Failed folders: {failures}")
    return successes, failures


# ── Step 2: Metrics ───────────────────────────────────────────────────────────

def run_metrics_folder(folder: Path) -> tuple:
    """Run analyze_metrics.py for one folder. Returns (folder_name, success, elapsed)."""
    t0  = time.time()
    out = METRICS / f"{folder.name}_metrics_v2.json"
    cmd = [
        sys.executable,
        str(WEB_APP / "analyze_metrics.py"),
        "--folder",      str(folder),
        "--output",      str(out),
        "--judge-model", "llama-3.3-70b",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            err = RuntimeError(
                f"analyze_metrics.py exited with code {result.returncode}\n"
                f"stderr: {result.stderr[-500:]}"
            )
            log_error("metrics", folder.name, err)
            return (folder.name, False, elapsed)
        return (folder.name, True, elapsed)
    except Exception as exc:
        elapsed = time.time() - t0
        log_error("metrics", folder.name, exc, traceback.format_exc())
        return (folder.name, False, elapsed)


def run_metrics(folders: list):
    log_info(f"── STEP 2: METRICS ({len(folders)} folders) ──")
    successes, failures = [], []

    iter_folders = (
        tqdm(folders, desc="Metrics", unit="folder", ncols=80)
        if HAS_TQDM else folders
    )
    for folder in iter_folders:
        name, ok, elapsed = run_metrics_folder(folder)
        log_info(f"  {name:40s} [{'OK' if ok else 'FAILED'}] {elapsed:.0f}s")
        (successes if ok else failures).append(name)

    log_info(f"Metrics complete: {len(successes)} OK, {len(failures)} failed")
    return successes, failures


# ── Step 3: ELO ──────────────────────────────────────────────────────────────

def run_elo():
    log_info("── STEP 3: ELO RATINGS ──")
    cmd = [
        sys.executable,
        str(WEB_APP / "elo.py"),
        "--output", str(WEB_APP / "elo_ratings.json"),
        "--plot",   str(WEB_APP / "elo_scatter.png"),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        if result.returncode != 0:
            err = RuntimeError(
                f"elo.py exited with code {result.returncode}\n"
                f"stderr: {result.stderr[-500:]}"
            )
            log_error("elo", "all", err)
            return False
        print(result.stdout)
        log_info("ELO complete.")
        return True
    except Exception as exc:
        log_error("elo", "all", exc, traceback.format_exc())
        return False


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Automated overnight pipeline for Among Us benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--folders",      nargs="*", default=None,
                        help="Specific folder names to process (default: all *-human-* folders)")
    parser.add_argument("--force",        action="store_true",
                        help="Re-label experiments that already have label files")
    parser.add_argument("--yes", "-y",    action="store_true",
                        help="Skip confirmation prompt for --force")
    parser.add_argument("--workers",      type=int, default=3,
                        help="Parallel workers for labelling (default: 3)")
    parser.add_argument("--label-only",   action="store_true",
                        help="Run labelling step only, skip metrics and ELO")
    parser.add_argument("--metrics-only", action="store_true",
                        help="Skip labelling, run metrics and ELO only")
    parser.add_argument("--no-elo",       action="store_true",
                        help="Skip ELO computation")
    args = parser.parse_args()

    # Discover or validate folders
    if args.folders:
        folders = []
        for name in args.folders:
            p = ROOT / name
            if not p.exists():
                print(f"ERROR: folder {p} does not exist.")
                sys.exit(1)
            folders.append(p)
    else:
        folders = discover_folders(ROOT)

    if not folders:
        print("ERROR: no *-human-* folders found at repo root.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"AMONG US BENCHMARK PIPELINE")
    print(f"{'='*60}")
    print(f"Repo root:  {ROOT}")
    print(f"Folders:    {len(folders)}")
    for f in folders:
        print(f"  • {f.name}")
    print(f"Force:      {args.force}")
    print(f"Workers:    {args.workers}")
    print(f"Error log:  {ERROR_LOG}")
    print(f"{'='*60}\n")

    # Confirmation for --force
    if args.force and not args.yes:
        existing = sum(
            1 for f in folders
            if (ROOT / "web-app" / "preserved_labels_v2" / f.name).exists()
               and any((ROOT / "web-app" / "preserved_labels_v2" / f.name).iterdir())
        )
        if existing > 0:
            print(f"WARNING: --force will re-label {existing} folder(s) that already have labels.")
            resp = input("Proceed? [y/N] ").strip().lower()
            if resp != "y":
                print("Aborted.")
                sys.exit(0)

    pipeline_start = time.time()

    label_ok, label_fail = [], []
    metrics_ok, metrics_fail = [], []
    elo_ok = None

    # Step 1: Labelling
    if not args.metrics_only:
        label_ok, label_fail = run_labelling(folders, args.workers, args.force)
    else:
        log_info("Skipping labelling (--metrics-only)")
        label_ok = [f.name for f in folders]

    # Step 2: Metrics (run for all folders, even if some labelling failed)
    if not args.label_only:
        metrics_ok, metrics_fail = run_metrics(folders)

        # Step 3: ELO
        if not args.no_elo:
            elo_ok = run_elo()

    elapsed_total = time.time() - pipeline_start

    # Summary
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE — {elapsed_total/60:.1f} min")
    print(f"{'='*60}")
    if not args.metrics_only:
        print(f"Labelling : {len(label_ok)} OK, {len(label_fail)} failed")
        if label_fail:
            print(f"  Failed: {label_fail}")
    if not args.label_only:
        print(f"Metrics   : {len(metrics_ok)} OK, {len(metrics_fail)} failed")
        if metrics_fail:
            print(f"  Failed: {metrics_fail}")
        if not args.no_elo:
            print(f"ELO       : {'OK' if elo_ok else 'FAILED'}")
    if label_fail or metrics_fail or elo_ok is False:
        print(f"\nSee {ERROR_LOG} for details.")
    else:
        print("\nAll steps completed successfully.")

    outputs = []
    if not args.label_only:
        outputs += [f"metrics/{f.name}_metrics_v2.json" for f in folders]
        if not args.no_elo:
            outputs += ["web-app/elo_ratings.json", "web-app/elo_scatter.png"]
    if outputs:
        print(f"\nOutputs:")
        for o in outputs:
            p = ROOT / o
            if p.exists():
                print(f"  [OK] {p}")


if __name__ == "__main__":
    main()
