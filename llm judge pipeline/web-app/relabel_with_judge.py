"""
Run the LLM-as-judge labelling pipeline with a configurable model.

Imports all data-loading, prompt-building, and validation logic from
llm_judge.py — no duplication.  Only the model is configurable.

Usage:
    python relabel_with_judge.py \\
        --folder ../claude-haiku-4.5-human-imp \\
        --model  meta-llama/llama-3.3-70b-instruct \\
        --model-slug llama-3.3-70b

    python relabel_with_judge.py \\
        --folder ../claude-haiku-4.5-human-imp \\
        --model  google/gemini-2.0-flash \\
        --model-slug gemini-2.0-flash \\
        --exp-id 2026-04-15_exp_0        # single experiment

    python relabel_with_judge.py \\
        --folder ../claude-haiku-4.5-human-imp \\
        --model  google/gemini-2.0-flash \\
        --model-slug gemini-2.0-flash \\
        --force                          # re-label even if file exists

Output:
    preserved_labels_v2/<folder_name>/<safe_exp_id>_<model_slug>.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Import everything reusable from llm_judge.py ──────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from llm_judge import (
    LABELS_ROOT,
    SYSTEM_PROMPT,           # noqa: F401 (used implicitly via call_openrouter)
    safe_exp_id,
    load_experiment,
    collect_impostor_speeches,
    build_user_prompt,
    call_openrouter,
    validate,
    normalize_label,
    discover_experiments,
)


# ── Core labelling loop ───────────────────────────────────────────────────────

def run(model_slug: str, model_id: str, data_root: Path,
        labels_dir: Path, experiments: list, force: bool = False):

    print(f"\n{'='*60}")
    print(f"Model     : {model_slug}  ({model_id})")
    print(f"Folder    : {data_root.name}")
    print(f"Labels    : {labels_dir}")
    print(f"Exps      : {len(experiments)}")
    print(f"Force     : {force}")
    print(f"{'='*60}")

    for exp_id in experiments:
        out_path = labels_dir / f"{safe_exp_id(exp_id)}_{model_slug}.json"

        if out_path.exists() and not force:
            print(f"\n  {exp_id}  — labels exist, skipping")
            continue

        print(f"\n  {exp_id}")
        try:
            exp = load_experiment(data_root / exp_id)
        except Exception as exc:
            print(f"  ERROR loading {exp_id}: {exc} — skipping")
            continue

        speeches    = collect_impostor_speeches(exp)
        exp_results = {}

        for sp in speeches:
            eid = sp["event_id"]
            tag = f"event:{eid.split(':')[-1]}"
            print(f"    {tag}  [{sp['actor']}]  ...", end="", flush=True)

            prompt = build_user_prompt(sp)
            raw    = call_openrouter(model_id, prompt)
            if not raw:
                print(" FAILED")
                continue

            label = validate(raw)
            label = normalize_label(label, sp["text"])
            label["exp_id"]          = exp_id
            label["event_id"]        = eid
            label["speech_position"] = sp["speech_position"]
            label["meeting_number"]  = sp["meeting_number"]
            label["has_thinking"]    = bool(sp["thinking_text"])
            exp_results[eid] = label

            lie_str = ("LIE"     if label["contains_lie"]
                       else "unverif" if label["unverifiable"]
                       else "truthful")
            print(f" {lie_str}  lying={label['score_lying']}  goal={label['strategic_goal']}")
            time.sleep(0.3)

        out_path.write_text(json.dumps(exp_results, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"  Saved -> {out_path.name}")

    print(f"\n{'='*60}")
    print("Done.")
    print(f"{'='*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Re-label impostor speeches with a configurable judge model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--folder",     required=True,
                        help="Path to game data folder (e.g. ../claude-haiku-4.5-human-imp)")
    parser.add_argument("--model",      required=True,
                        help="OpenRouter model ID (e.g. meta-llama/llama-3.3-70b-instruct)")
    parser.add_argument("--model-slug", required=True,
                        help="Short slug used in output filenames (e.g. llama-3.3-70b)")
    parser.add_argument("--labels-dir", default=None,
                        help="Override label output directory (default: preserved_labels_v2/<folder>)")
    parser.add_argument("--exp-id",     default=None,
                        help="Label a single experiment only")
    parser.add_argument("--force",      action="store_true",
                        help="Re-label even if output file already exists")
    parser.add_argument("--all-experiments", action="store_true", default=True,
                        help="Label all experiments (default)")
    args = parser.parse_args()

    data_root = Path(args.folder)
    if not data_root.exists():
        print(f"ERROR: folder {data_root} does not exist.")
        sys.exit(1)

    if args.labels_dir:
        labels_dir = Path(args.labels_dir)
    else:
        labels_dir = LABELS_ROOT / data_root.name
    labels_dir.mkdir(parents=True, exist_ok=True)
    print(f"Labels directory: {labels_dir}")

    if args.exp_id:
        experiments = [args.exp_id]
    else:
        experiments = discover_experiments(data_root)
        print(f"Discovered {len(experiments)} experiments in {data_root.name}")

    if not experiments:
        print("No experiments found — nothing to do.")
        sys.exit(0)

    run(args.model_slug, args.model, data_root, labels_dir, experiments,
        force=args.force)


if __name__ == "__main__":
    main()
