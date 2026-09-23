"""
Measure inter-judge agreement between two sets of LLM-as-judge label files.

For each field in the v2 label schema the appropriate agreement metric is
computed on all speeches labeled by BOTH judges (inner join on event ID):

  Binary fields              Cohen's κ
  Ordinal fields (1-5)       Weighted Cohen's κ (quadratic) + Spearman ρ
  Categorical single-label   Cohen's κ
  lie_claim_types (multi)    Mean per-speech Jaccard similarity

Fields with κ < 0.4 (or Jaccard < 0.4) are flagged as LOW RELIABILITY.

Usage:
    python judge_agreement.py \\
        --judge1 gemini-flash-3.1-lite-preview \\
        --judge2 llama-3.3-70b

    python judge_agreement.py \\
        --judge1 gemini-flash-3.1-lite-preview \\
        --judge2 llama-3.3-70b \\
        --labels-dir /path/to/preserved_labels_v2

Output:
    stdout — formatted summary table
    web-app/judge_agreement_<judge1>_vs_<judge2>.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from sklearn.metrics import cohen_kappa_score
except ImportError:
    print("ERROR: scikit-learn is required.  pip install scikit-learn")
    sys.exit(1)

try:
    from scipy.stats import spearmanr
except ImportError:
    print("ERROR: scipy is required.  pip install scipy")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).parent

# ── Field definitions ─────────────────────────────────────────────────────────

BINARY_FIELDS   = ["contains_lie", "unverifiable", "dead_player_confusion"]
ORDINAL_FIELDS  = ["score_awareness", "score_lying", "score_sophistication", "score_planning"]
ORDINAL_NULLABLE = ["thinking_planning_score"]   # skip pair when either value is null

VALID_MECHANISM      = ["none", "factual_lie", "omission", "ambiguity",
                        "misdirection", "false_accusation"]
VALID_STRATEGIC_GOALS = ["defend", "deflect", "control", "passive"]
CATEGORICAL_FIELDS  = [
    ("deception_mechanism", VALID_MECHANISM),
    ("strategic_goal",      VALID_STRATEGIC_GOALS),
]
MULTILABEL_FIELDS   = ["lie_claim_types"]

LOW_RELIABILITY_THRESHOLD = 0.4


# ── Helpers ───────────────────────────────────────────────────────────────────

def _set_jaccard(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    union  = sa | sb
    inter  = sa & sb
    return len(inter) / len(union) if union else 1.0


def _safe_kappa(y1: list, y2: list, weights=None, labels=None):
    """Return Cohen's κ, or None if the column is constant (undefined)."""
    kwargs = {}
    if weights is not None:
        kwargs["weights"] = weights
    if labels is not None:
        kwargs["labels"] = labels
    try:
        return cohen_kappa_score(y1, y2, **kwargs)
    except ValueError:
        return None


# ── Metric computation ────────────────────────────────────────────────────────

def _compute(pairs: list, field_type: str, cat_labels=None) -> dict | None:
    if not pairs:
        return None

    if field_type == "binary":
        y1, y2 = zip(*pairs)
        y1, y2 = [int(v) for v in y1], [int(v) for v in y2]
        kappa  = _safe_kappa(y1, y2)
        low    = kappa is not None and kappa < LOW_RELIABILITY_THRESHOLD
        return {"type": "binary", "n": len(pairs), "kappa": _r(kappa),
                "constant": kappa is None, "low_reliability": low}

    if field_type == "ordinal":
        y1, y2 = zip(*pairs)
        y1, y2 = list(y1), list(y2)
        kappa  = _safe_kappa(y1, y2, weights="quadratic")
        rho, p = (None, None)
        if len(set(y1)) > 1 and len(set(y2)) > 1:
            rho, p = spearmanr(y1, y2)
        low = kappa is not None and kappa < LOW_RELIABILITY_THRESHOLD
        return {
            "type":            "ordinal",
            "n":               len(pairs),
            "kappa_quadratic": _r(kappa),
            "spearman_r":      _r(rho),
            "spearman_p":      float(p) if p is not None else None,
            "constant":        kappa is None,
            "low_reliability": low,
        }

    if field_type == "categorical":
        y1, y2 = zip(*pairs)
        kappa  = _safe_kappa(list(y1), list(y2), labels=cat_labels)
        low    = kappa is not None and kappa < LOW_RELIABILITY_THRESHOLD
        return {"type": "categorical", "n": len(pairs), "kappa": _r(kappa),
                "constant": kappa is None, "low_reliability": low}

    if field_type == "multilabel":
        jaccards  = [_set_jaccard(a, b) for a, b in pairs]
        mean_j    = sum(jaccards) / len(jaccards)
        low       = mean_j < LOW_RELIABILITY_THRESHOLD
        return {"type": "multilabel", "n": len(pairs),
                "mean_jaccard": _r(mean_j), "low_reliability": low}

    return None


def _r(v, d=4):
    return round(v, d) if v is not None else None


# ── Label loading ─────────────────────────────────────────────────────────────

def load_paired_labels(labels_dir: Path, judge1: str, judge2: str):
    """
    Walk preserved_labels_v2/ and inner-join label files by event ID.

    Returns:
        all_pairs   dict[field_name -> list[(v1, v2)]]
        per_folder  dict[folder_name -> int matched]
    """
    all_pairs  = defaultdict(list)
    per_folder = {}

    for folder in sorted(labels_dir.iterdir()):
        if not folder.is_dir():
            continue

        folder_pairs = defaultdict(list)
        n_matched    = 0

        for j1_file in sorted(folder.glob(f"*_{judge1}.json")):
            # Strip the known "_<judge1>" suffix to recover the safe exp_id
            safe_eid = j1_file.stem[: -len(f"_{judge1}")]
            j2_file  = folder / f"{safe_eid}_{judge2}.json"
            if not j2_file.exists():
                continue

            j1_labels = json.loads(j1_file.read_text(encoding="utf-8", errors="replace"))
            j2_labels = json.loads(j2_file.read_text(encoding="utf-8", errors="replace"))

            for eid in j1_labels:
                if eid not in j2_labels:
                    continue
                l1 = j1_labels[eid]
                l2 = j2_labels[eid]
                n_matched += 1

                for f in BINARY_FIELDS:
                    if f in l1 and f in l2:
                        folder_pairs[f].append((l1[f], l2[f]))

                for f in ORDINAL_FIELDS:
                    v1, v2 = l1.get(f), l2.get(f)
                    if v1 is not None and v2 is not None:
                        folder_pairs[f].append((int(v1), int(v2)))

                for f in ORDINAL_NULLABLE:
                    v1, v2 = l1.get(f), l2.get(f)
                    if v1 is not None and v2 is not None:
                        folder_pairs[f].append((int(v1), int(v2)))

                for f, _ in CATEGORICAL_FIELDS:
                    if f in l1 and f in l2:
                        folder_pairs[f].append((l1[f], l2[f]))

                for f in MULTILABEL_FIELDS:
                    if f in l1 and f in l2:
                        folder_pairs[f].append(
                            (l1[f] if isinstance(l1[f], list) else [],
                             l2[f] if isinstance(l2[f], list) else [])
                        )

        if n_matched > 0:
            per_folder[folder.name] = n_matched
            for field, vals in folder_pairs.items():
                all_pairs[field].extend(vals)

    return dict(all_pairs), per_folder


# ── Output formatting ─────────────────────────────────────────────────────────

def _fmt_row(field: str, res: dict | None) -> str:
    if res is None:
        return f"  {field:<28} {'n/a':<10} {'—':>5}   (no matched speeches)"

    t   = res["type"]
    n   = res["n"]
    low = "⚠ LOW RELIABILITY" if res.get("low_reliability") else "OK"

    if t == "binary":
        k  = res["kappa"]
        ks = f"κ={k:.3f}" if k is not None else "κ=n/a (constant)"
        return f"  {field:<28} {'binary':<10} {n:5d}   {ks:<24}  {low}"

    if t == "ordinal":
        k  = res["kappa_quadratic"]
        r  = res["spearman_r"]
        p  = res["spearman_p"]
        ks = f"κ={k:.3f}" if k is not None else "κ=n/a"
        rs = ""
        if r is not None:
            ps = "<0.001" if p < 0.001 else f"{p:.3f}"
            rs = f"  ρ={r:.3f} (p={ps})"
        return f"  {field:<28} {'ordinal':<10} {n:5d}   {ks:<10}{rs:<26}  {low}"

    if t == "categorical":
        k  = res["kappa"]
        ks = f"κ={k:.3f}" if k is not None else "κ=n/a (constant)"
        return f"  {field:<28} {'categ.':<10} {n:5d}   {ks:<24}  {low}"

    if t == "multilabel":
        j = res["mean_jaccard"]
        return f"  {field:<28} {'multi':<10} {n:5d}   Jaccard={j:.3f}              {low}"

    return f"  {field:<28} {t:<10} {n:5d}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute inter-judge agreement metrics (v2 label schema).")
    parser.add_argument("--judge1",     required=True,
                        help="First judge model slug (e.g. gemini-flash-3.1-lite-preview)")
    parser.add_argument("--judge2",     required=True,
                        help="Second judge model slug (e.g. llama-3.3-70b)")
    parser.add_argument("--labels-dir", default=None,
                        help="Path to preserved_labels_v2/ (default: auto-detected)")
    args = parser.parse_args()

    labels_dir = (
        Path(args.labels_dir) if args.labels_dir
        else SCRIPT_DIR / "preserved_labels_v2"
    )
    if not labels_dir.exists():
        print(f"ERROR: labels directory {labels_dir} does not exist.")
        sys.exit(1)

    print(f"\nScanning {labels_dir} ...")
    all_pairs, per_folder = load_paired_labels(labels_dir, args.judge1, args.judge2)
    n_total = sum(per_folder.values()) if per_folder else 0

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\nJudge 1  : {args.judge1}")
    print(f"Judge 2  : {args.judge2}")
    print(f"Matched  : {n_total:,} speeches  (across {len(per_folder)} folder(s))")

    if n_total == 0:
        print("\nNo matched speeches found — check that both judges have labeled the same folders.")
        sys.exit(0)

    print(f"\nPer-folder breakdown:")
    for fname, cnt in sorted(per_folder.items()):
        print(f"  {fname:<45}  {cnt:5d}")

    # ── Compute metrics ────────────────────────────────────────────────────────
    results = {}

    for f in BINARY_FIELDS:
        results[f] = _compute(all_pairs.get(f, []), "binary")

    for f in ORDINAL_FIELDS:
        results[f] = _compute(all_pairs.get(f, []), "ordinal")

    for f in ORDINAL_NULLABLE:
        results[f] = _compute(all_pairs.get(f, []), "ordinal")

    for f, cat_labels in CATEGORICAL_FIELDS:
        results[f] = _compute(all_pairs.get(f, []), "categorical", cat_labels=cat_labels)

    for f in MULTILABEL_FIELDS:
        results[f] = _compute(all_pairs.get(f, []), "multilabel")

    # ── Summary table ──────────────────────────────────────────────────────────
    sep = "─" * 92
    print(f"\n{sep}")
    print(f"  {'Field':<28} {'Type':<10} {'N':>5}   {'κ / ρ / Jaccard':<36}  Reliability")
    print(sep)
    for field in (BINARY_FIELDS + ORDINAL_FIELDS + ORDINAL_NULLABLE
                  + [f for f, _ in CATEGORICAL_FIELDS] + MULTILABEL_FIELDS):
        print(_fmt_row(field, results.get(field)))
    print(f"{sep}\n")

    # ── JSON output ────────────────────────────────────────────────────────────
    s1 = args.judge1.replace("/", "-")
    s2 = args.judge2.replace("/", "-")
    out_path = SCRIPT_DIR / f"judge_agreement_{s1}_vs_{s2}.json"
    out_path.write_text(
        json.dumps({
            "judge1":               args.judge1,
            "judge2":               args.judge2,
            "n_matched_total":      n_total,
            "n_matched_per_folder": per_folder,
            "metrics":              results,
        }, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
