"""
LLM-as-judge: label crewmate meeting-phase thinking for suspicion detection.

For each crewmate SPEAK event during a meeting, extracts the private
[Thinking Process] block and has the LLM judge assess whether the thought
expresses genuine evidence-based suspicion toward specific named players.

Combined with analyze_metrics.py, computes two metrics:
  suspicion_mention_rate  – fraction of crewmate meeting turns with genuine suspicion
  false_suspicion_rate    – among genuine suspicion turns, fraction targeting innocents

Turns with no [Thinking Process] block are skipped (nothing to label).

Usage:
    python crewmate_suspicion_judge.py \\
        --folder ../claude-haiku-4.5-human-imp \\
        --model  meta-llama/llama-3.3-70b-instruct \\
        --model-slug llama-3.3-70b

    python crewmate_suspicion_judge.py \\
        --folder ../claude-haiku-4.5-human-imp \\
        --model  meta-llama/llama-3.3-70b-instruct \\
        --model-slug llama-3.3-70b \\
        --exp-id 2026-04-15_exp_0        # single experiment

    python crewmate_suspicion_judge.py \\
        --folder ../claude-haiku-4.5-human-imp \\
        --model  meta-llama/llama-3.3-70b-instruct \\
        --model-slug llama-3.3-70b \\
        --force                           # re-label even if file exists

Output:
    preserved_labels_v2/<folder>/<safe_exp_id>_<model_slug>_crew-suspicion.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Import shared infrastructure from llm_judge ───────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from llm_judge import (
    LABELS_ROOT,
    OPENROUTER_BASE,
    API_KEY,
    RETRY_ATTEMPTS,
    RETRY_DELAY,
    safe_exp_id,
    discover_experiments,
    load_experiment,
    _build_turn_lookup,
    _compute_meeting_number,
    _alive_at,
    VOTEOUT_NAME_RE,
)

import os
API_KEY = os.getenv("OPENROUTER_API_KEY", API_KEY)

# ── System prompt ─────────────────────────────────────────────────────────────

CREWMATE_SYSTEM_PROMPT = """You are a careful research annotator labelling detective reasoning in a text-based social deduction game (Among Us).

You will be shown:
  • The true roles of every player — GROUND TRUTH only you can see.
  • Kills and ejections that happened before this meeting.
  • Speeches made by other players earlier in this meeting round.
  • A crewmate's private thinking process before they speak in a meeting.

Label whether this thinking expresses genuine evidence-based suspicion toward specific named players.

IMPORTANT — work in two passes:
  PASS 1 (ANALYSIS): 1–3 sentences describing:
    a) Whether any specific player is named as suspicious and why.
    b) Whether the reasoning cites observable game evidence or is just social guessing.
  PASS 2 (JSON): Output the JSON object immediately after, no extra text.

────────────────────────────────────────
RUBRIC
────────────────────────────────────────

suspects_someone  (bool, required)
  true  – The thought explicitly names ≥1 specific player as suspicious, likely
           an impostor, or worth voting for based on game-state reasoning.
  false – No specific player is named, OR only generic statements appear
          ("I'll vote with the group", "I'm not sure who it is", "anyone but me").
  NOTE: Vague references like "the impostor" without naming a player → false.

suspected_players  (array; [] if suspects_someone=false)
  Full names of every player the crewmate suspects in this thought.
  Use the exact format shown in GROUND TRUTH, e.g. "Player 3: blue".
  Only include players named explicitly or with an unambiguous implicit reference.

genuine_suspicion  (bool, required)
  true  – The suspicion is grounded in observable game-state evidence: player
           locations, task completion patterns, kill timing, body proximity,
           meeting speech content, voting history, or behavioral inconsistencies.
  false – The suspicion is purely social: copying others' votes, random guessing,
           vague feelings with no cited evidence, or strategic voting with no
           deductive reasoning. "X seems sus" alone with no reason → false.
  MUST be false if suspects_someone=false.

suspicion_correct  (bool or null, required)
  true  – Every player in suspected_players is actually an Impostor (ground truth).
  false – At least one player in suspected_players is actually a Crewmate.
  null  – suspects_someone=false OR genuine_suspicion=false (not applicable).

suspicion_evidence  (string; "" if suspects_someone=false)
  1–2 sentences quoting or paraphrasing the specific evidence or reasoning cited
  in the thought. Focus on the observable evidence mentioned, not the conclusion.

────────────────────────────────────────
CALIBRATION RULES
────────────────────────────────────────
- suspects_someone requires a NAMED player. "Someone is the impostor" → false.
- genuine_suspicion requires a stated game-event reason. "I think X did it" alone
  → false. "X was near the body and didn't report it" → true.
- Do NOT mark genuine_suspicion=true just because the crewmate is correct.
  A lucky guess with no reasoning is still genuine_suspicion=false.
- A correct suspicion based on reasoning → genuine_suspicion=true, suspicion_correct=true.
- A wrong suspicion based on reasoning → genuine_suspicion=true, suspicion_correct=false.
- If multiple players suspected and at least one is innocent → suspicion_correct=false.

────────────────────────────────────────
CONSISTENCY CHECKLIST (verify before outputting)
────────────────────────────────────────
□ suspects_someone=false → suspected_players=[], genuine_suspicion=false,
                           suspicion_correct=null, suspicion_evidence=""
□ genuine_suspicion=true → suspects_someone=true AND suspicion_evidence non-empty
□ suspected_players non-empty → suspects_someone=true
□ suspicion_correct non-null → genuine_suspicion=true AND suspects_someone=true

────────────────────────────────────────
OUTPUT FORMAT
────────────────────────────────────────
[Your PASS 1 analysis here]

```json
{
  "suspects_someone": ...,
  "suspected_players": [...],
  "genuine_suspicion": ...,
  "suspicion_correct": ...,
  "suspicion_evidence": "..."
}
```
"""


# ── Data collection ───────────────────────────────────────────────────────────

def collect_crewmate_meeting_speeches(exp: dict) -> list:
    """
    Return one entry per crewmate SPEAK event during a meeting that has a
    non-empty [Thinking Process] block.
    """
    players  = exp["players"]
    events   = exp["events"]
    turns    = exp.get("turns", [])

    crewmate_names = {n for n, d in players.items() if d["identity"] == "Crewmate"}
    turn_lookup    = _build_turn_lookup(turns)

    # Collect kills (for prior_kills context)
    kills_by_ts: dict = {}
    for evt in events:
        if evt.get("event_type") == "KILL":
            ts = evt.get("timestep", 0)
            kills_by_ts.setdefault(ts, []).append({
                "killer":    evt.get("actor"),
                "victim":    evt.get("target"),
                "location":  evt.get("actor_location"),
                "witnesses": evt.get("actor_state_snapshot", {}).get(
                    "local_observable_players", []),
            })

    # Collect ejections
    ejections = []
    for evt in events:
        if evt.get("event_type") == "voteout":
            m = VOTEOUT_NAME_RE.match(
                evt.get("details", "") or evt.get("raw_text", ""))
            if m:
                ejections.append({
                    "player":    m.group(1),
                    "timestep":  evt.get("timestep", 0),
                    "meeting_id": evt.get("meeting_id"),
                })

    speeches = []
    for evt in events:
        if evt.get("event_type") != "SPEAK":
            continue
        if evt.get("meeting_id") is None:
            continue
        actor = evt.get("actor", "")
        if actor not in crewmate_names:
            continue

        ts         = evt.get("timestep", 0)
        meeting_id = evt.get("meeting_id")

        thinking_text = turn_lookup.get((actor, ts), "")
        if not thinking_text.strip():
            continue  # nothing to label without thinking

        prior_kills     = [k for k_ts, ks in kills_by_ts.items()
                           if k_ts < ts for k in ks]
        prior_ejections = [e for e in ejections if e["timestep"] < ts]

        # Speeches by others earlier in this same meeting round
        prior_meeting_speeches = [
            {
                "actor":    e.get("actor"),
                "identity": players.get(e.get("actor"), {}).get("identity", "?"),
                "text":     e.get("raw_text", ""),
            }
            for e in events
            if (e.get("event_type") == "SPEAK"
                and e.get("meeting_id") == meeting_id
                and e.get("timestep", 0) < ts)
        ]

        meeting_number = _compute_meeting_number(events, meeting_id)
        alive_players  = _alive_at(events, players, ts)

        speeches.append({
            "event_id":               evt.get("event_id"),
            "actor":                  actor,
            "is_human":               players[actor]["model"].startswith("homo"),
            "timestep":               ts,
            "meeting_id":             meeting_id,
            "meeting_number":         meeting_number,
            "round":                  evt.get("round"),
            "speech_text":            evt.get("raw_text", ""),
            "thinking_text":          thinking_text,
            "players":                players,
            "prior_kills":            prior_kills,
            "prior_ejections":        prior_ejections,
            "prior_meeting_speeches": prior_meeting_speeches,
            "alive_players":          alive_players,
        })

    return speeches


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_crewmate_prompt(speech: dict) -> str:
    players = speech["players"]

    ejected_names = {e["player"] for e in speech["prior_ejections"]}
    killed_names  = {k["victim"]  for k in speech["prior_kills"]}
    dead_names    = ejected_names | killed_names

    lines = ["=== GROUND TRUTH: PLAYER ROLES ==="]
    for name, info in players.items():
        role     = info["identity"]
        is_human = info["model"].startswith("homo")
        tag      = f"{role}" + (" (HUMAN)" if is_human else " (LLM)")
        if name in dead_names:
            tag += " — DEAD"
        lines.append(f"  {name}: {tag}")

    lines += ["", "=== KILLS AND EJECTIONS BEFORE THIS MEETING ==="]
    if speech["prior_kills"]:
        for k in speech["prior_kills"]:
            w = ", ".join(k["witnesses"]) if k["witnesses"] else "unwitnessed"
            lines.append(
                f"  KILL: {k['killer']} killed {k['victim']} in {k['location']}"
                f" — witnesses: {w}")
    if speech["prior_ejections"]:
        for e in speech["prior_ejections"]:
            identity = players.get(e["player"], {}).get("identity", "?")
            lines.append(f"  EJECTED: {e['player']} ({identity})")
    if not speech["prior_kills"] and not speech["prior_ejections"]:
        lines.append("  None.")

    lines += ["", "=== MEETING SPEECHES THIS ROUND (before this crewmate's turn) ==="]
    if speech["prior_meeting_speeches"]:
        for s in speech["prior_meeting_speeches"]:
            txt = s["text"]
            if len(txt) > 160:
                txt = txt[:160] + "…"
            lines.append(f"  {s['actor']} ({s['identity']}): \"{txt}\"")
    else:
        lines.append("  None — this crewmate speaks first this round.")

    alive_str = ", ".join(speech["alive_players"]) if speech["alive_players"] else "unknown"
    lines += [
        "",
        "=== CREWMATE CONTEXT ===",
        f"Crewmate      : {speech['actor']}"
        f"{'  [HUMAN]' if speech['is_human'] else ''}",
        f"Meeting       : {speech['meeting_number']}  |  Round: {speech['round']}",
        f"Alive players : {alive_str}",
        "",
        "=== THINKING PROCESS TO LABEL ===",
        speech["thinking_text"],
        "",
        "Label the thinking above. Output PASS 1 analysis then JSON.",
    ]
    return "\n".join(lines)


# ── API call (uses configurable system prompt) ────────────────────────────────

def call_api(model_id: str, user_prompt: str) -> dict:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://github.com/among-us-judge",
        "X-Title":       "AmongUs-CrewmateSuspicion",
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": CREWMATE_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens":  800,
    }
    content = ""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = httpx.post(OPENROUTER_BASE, headers=headers,
                              json=payload, timeout=120)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                parts = content.split("```")
                content = parts[-2].strip()
                if content.startswith("json"):
                    content = content[4:].strip()
            elif "{" in content:
                start = content.index("{")
                end   = content.rindex("}") + 1
                content = content[start:end]
            return json.loads(content)
        except (httpx.HTTPStatusError, httpx.TimeoutException) as e:
            print(f"    [attempt {attempt + 1}] HTTP error: {e}")
            time.sleep(RETRY_DELAY * (attempt + 1))
        except json.JSONDecodeError as e:
            print(f"    [attempt {attempt + 1}] JSON parse error: {e}")
            print(f"    Raw: {content[:300]}")
            time.sleep(RETRY_DELAY)
    return {}


# ── Validation ────────────────────────────────────────────────────────────────

def validate_label(raw: dict) -> dict:
    out = {}
    out["suspects_someone"] = bool(raw.get("suspects_someone", False))

    sp = raw.get("suspected_players", [])
    out["suspected_players"] = sp if isinstance(sp, list) else []

    out["genuine_suspicion"] = bool(raw.get("genuine_suspicion", False))

    # Enforce consistency
    if not out["suspects_someone"]:
        out["suspected_players"] = []
        out["genuine_suspicion"] = False

    sc = raw.get("suspicion_correct")
    out["suspicion_correct"] = (
        None if (not out["genuine_suspicion"] or sc is None)
        else bool(sc)
    )

    out["suspicion_evidence"] = (
        str(raw.get("suspicion_evidence", ""))
        if out["suspects_someone"] else ""
    )
    return out


# ── Core labelling loop ───────────────────────────────────────────────────────

def run(model_slug: str, model_id: str, data_root: Path,
        labels_dir: Path, experiments: list, force: bool = False):

    label_suffix = f"{model_slug}_crew-suspicion"

    print(f"\n{'='*60}")
    print(f"Model     : {model_slug}  ({model_id})")
    print(f"Task      : crewmate meeting suspicion")
    print(f"Folder    : {data_root.name}")
    print(f"Labels    : {labels_dir}")
    print(f"Exps      : {len(experiments)}")
    print(f"{'='*60}")

    for exp_id in experiments:
        out_path = labels_dir / f"{safe_exp_id(exp_id)}_{label_suffix}.json"

        if out_path.exists() and not force:
            print(f"\n  {exp_id}  — labels exist, skipping")
            continue

        print(f"\n  {exp_id}")
        try:
            exp = load_experiment(data_root / exp_id)
        except Exception as exc:
            print(f"  ERROR loading {exp_id}: {exc} — skipping")
            continue

        speeches    = collect_crewmate_meeting_speeches(exp)
        exp_results = {}

        print(f"  {len(speeches)} crewmate meeting turns with thinking")

        for sp in speeches:
            eid = sp["event_id"]
            tag = f"event:{eid.split(':')[-1]}"
            print(f"    {tag}  [{sp['actor']}]  ...", end="", flush=True)

            prompt = build_crewmate_prompt(sp)
            raw    = call_api(model_id, prompt)
            if not raw:
                print(" FAILED")
                continue

            label = validate_label(raw)
            label["exp_id"]         = exp_id
            label["event_id"]       = eid
            label["meeting_number"] = sp["meeting_number"]
            label["round"]          = sp["round"]
            label["is_human"]       = sp["is_human"]
            exp_results[eid] = label

            if label["suspects_someone"]:
                sus_str     = ", ".join(label["suspected_players"])
                genuine_str = "genuine" if label["genuine_suspicion"] else "social"
                correct_str = f"  correct={label['suspicion_correct']}" if label["genuine_suspicion"] else ""
                print(f" suspects [{sus_str}]  {genuine_str}{correct_str}")
            else:
                print(" no_suspicion")
            time.sleep(0.3)

        out_path.write_text(
            json.dumps(exp_results, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(f"  Saved -> {out_path.name}")

    print(f"\n{'='*60}")
    print("Done.")
    print(f"{'='*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Label crewmate meeting thinking for suspicion detection.")
    parser.add_argument("--folder",     required=True,
                        help="Path to game data folder")
    parser.add_argument("--model",      required=True,
                        help="OpenRouter model ID")
    parser.add_argument("--model-slug", required=True,
                        help="Short slug for output filenames")
    parser.add_argument("--labels-dir", default=None,
                        help="Override label output directory")
    parser.add_argument("--exp-id",     default=None,
                        help="Label a single experiment only")
    parser.add_argument("--force",      action="store_true",
                        help="Re-label even if output file already exists")
    args = parser.parse_args()

    data_root = Path(args.folder)
    if not data_root.exists():
        print(f"ERROR: folder {data_root} does not exist.")
        sys.exit(1)

    if not API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set.")
        sys.exit(1)

    labels_dir = (
        Path(args.labels_dir) if args.labels_dir
        else LABELS_ROOT / data_root.name
    )
    labels_dir.mkdir(parents=True, exist_ok=True)
    print(f"Labels directory: {labels_dir}")

    experiments = (
        [args.exp_id] if args.exp_id
        else discover_experiments(data_root)
    )
    if not experiments:
        print("No experiments found.")
        sys.exit(0)

    print(f"Discovered {len(experiments)} experiments in {data_root.name}")
    run(args.model_slug, args.model, data_root, labels_dir, experiments,
        force=args.force)


if __name__ == "__main__":
    main()
