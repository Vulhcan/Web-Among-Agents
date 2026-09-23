import json
import os
import random
import re
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, session

sys.path.insert(0, str(Path(__file__).parent))
from llm_judge import (
    collect_impostor_speeches,
    build_user_prompt,
    load_experiment as lj_load_experiment,
    safe_exp_id as lj_safe_exp_id,
    discover_experiments as lj_discover_experiments,
)

app = Flask(__name__)
app.secret_key = "among-us-judge-2026"

# ── Verification constants ────────────────────────────────────────────────────
REPO_ROOT          = Path(__file__).parent.parent
LABELS_V2_DIR_BASE = Path(__file__).parent / "preserved_labels_v2"
LLAMA_SLUG         = "llama-3.3-70b"
VERIFY_LABELS_DIR  = Path(__file__).parent / "human_verify_labels"
VERIFY_LABELS_DIR.mkdir(exist_ok=True)
VERIFY_SELECTED_FILE = VERIFY_LABELS_DIR / "_selected.json"
VERIFY_SEED          = 2026
VERIFY_N_PER_FOLDER  = 4

MECH_CHOICES  = ["none", "factual_lie", "omission", "ambiguity", "misdirection", "false_accusation"]
GOAL_CHOICES  = ["defend", "deflect", "control", "passive"]
CLAIM_CHOICES = ["positional", "observational", "activity", "accusation"]
SCORE_FIELDS  = ["score_awareness", "score_lying", "score_sophistication", "score_planning"]


# ── Verification helpers ──────────────────────────────────────────────────────

def _all_data_folders():
    return sorted(
        d for d in REPO_ROOT.iterdir()
        if d.is_dir() and re.match(r".+-human-(imp|crew)$", d.name)
    )


def get_verify_selected() -> dict:
    if VERIFY_SELECTED_FILE.exists():
        return json.loads(VERIFY_SELECTED_FILE.read_text(encoding="utf-8"))

    rng = random.Random(VERIFY_SEED)
    selected = {}
    for folder in _all_data_folders():
        exps = lj_discover_experiments(folder)
        labeled = [
            e for e in exps
            if (LABELS_V2_DIR_BASE / folder.name /
                f"{lj_safe_exp_id(e)}_{LLAMA_SLUG}.json").exists()
        ]
        n = min(VERIFY_N_PER_FOLDER, len(labeled))
        if n:
            selected[folder.name] = rng.sample(labeled, n)

    VERIFY_SELECTED_FILE.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    return selected


def _load_llama_labels(folder_name: str, safe_eid: str) -> dict:
    path = LABELS_V2_DIR_BASE / folder_name / f"{safe_eid}_{LLAMA_SLUG}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    return {}


def load_verify_human_labels(folder_name: str, exp_id: str) -> dict:
    path = VERIFY_LABELS_DIR / folder_name / f"{lj_safe_exp_id(exp_id)}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_verify_human_label(folder_name: str, exp_id: str, event_id: str, label: dict):
    d = VERIFY_LABELS_DIR / folder_name
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{lj_safe_exp_id(exp_id)}.json"
    labels = load_verify_human_labels(folder_name, exp_id)
    labels[event_id] = label
    path.write_text(json.dumps(labels, indent=2), encoding="utf-8")


def load_all_verify_pairs() -> tuple:
    """Return (human_labels, llama_labels) dicts keyed by event_id."""
    selected = get_verify_selected()
    human, llama = {}, {}
    for folder_name, exp_ids in selected.items():
        for exp_id in exp_ids:
            h = load_verify_human_labels(folder_name, exp_id)
            l = _load_llama_labels(folder_name, lj_safe_exp_id(exp_id))
            for eid, lbl in h.items():
                human[eid] = lbl
                if eid in l:
                    llama[eid] = l[eid]
    return human, llama


def _spearman(xs: list, ys: list):
    n = len(xs)
    if n < 2:
        return None
    def rank(arr):
        s = sorted(enumerate(arr), key=lambda x: x[1])
        r = [0] * n
        for i, (idx, _) in enumerate(s):
            r[idx] = i + 1
        return r
    rx, ry = rank(xs), rank(ys)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    denom = n * (n * n - 1)
    return round(1 - 6 * d2 / denom, 3) if denom else None


def compute_verify_correlation(human: dict, llama: dict) -> dict:
    paired = [e for e in human if e in llama]
    n = len(paired)
    if n == 0:
        return {"n": 0}

    result = {"n": n}

    # Binary — % agreement
    for f in ["contains_lie", "unverifiable", "dead_player_confusion"]:
        vals = [(human[e].get(f), llama[e].get(f)) for e in paired
                if human[e].get(f) is not None and llama[e].get(f) is not None]
        if vals:
            result[f"{f}_pct"] = round(sum(bool(h) == bool(l) for h, l in vals) / len(vals), 3)

    # Ordinal — Spearman ρ + MAE
    for f in SCORE_FIELDS:
        vals = [(human[e].get(f), llama[e].get(f)) for e in paired
                if human[e].get(f) is not None and llama[e].get(f) is not None]
        if len(vals) >= 2:
            h_s, l_s = [v[0] for v in vals], [v[1] for v in vals]
            result[f"{f}_rho"] = _spearman(h_s, l_s)
            result[f"{f}_mae"] = round(sum(abs(h - l) for h, l in vals) / len(vals), 2)

    # Categorical — % agreement
    for f in ["deception_mechanism", "strategic_goal"]:
        vals = [(human[e].get(f), llama[e].get(f)) for e in paired
                if human[e].get(f) and llama[e].get(f)]
        if vals:
            result[f"{f}_pct"] = round(sum(h == l for h, l in vals) / len(vals), 3)

    # Multi-label — mean Jaccard
    jacs = []
    for e in paired:
        hs = set(human[e].get("lie_claim_types", []))
        ls = set(llama[e].get("lie_claim_types", []))
        u = hs | ls
        jacs.append(len(hs & ls) / len(u) if u else 1.0)
    if jacs:
        result["lie_claim_types_jaccard"] = round(sum(jacs) / len(jacs), 3)

    return result

DATA_ROOT = Path(__file__).parent.parent / "claude-haiku-4.5-human-imp"

# ── LLM review constants ──────────────────────────────────────────────────────
LABELS_V2_DIR = Path(__file__).parent / "preserved_labels_v2" / "claude-haiku-4.5-human-imp"
JUDGES = [
    ("Gemini Flash 3.1 Lite", "gemini-flash-3.1-lite-preview"),
    ("Llama 3.3 70B",         "llama-3.3-70b"),
]


def _load_v2_labels(exp_id: str, slug: str) -> dict:
    path = LABELS_V2_DIR / f"{lj_safe_exp_id(exp_id)}_{slug}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    return {}


def _get_outcome(exp_id: str) -> dict:
    try:
        f = DATA_ROOT / exp_id / "structured-v1" / "outcomes_v1.jsonl"
        lines = [l for l in f.read_text().splitlines() if l.strip()]
        return json.loads(lines[0]) if lines else {}
    except Exception:
        return {}
LABELS_DIR = Path(__file__).parent / "labels"
LABELS_DIR.mkdir(exist_ok=True)
SELECTED_FILE = LABELS_DIR / "selected_experiments.json"

RANDOM_SEED = 42


def get_selected_experiments():
    if SELECTED_FILE.exists():
        return json.loads(SELECTED_FILE.read_text())
    all_exps = sorted([d.name for d in DATA_ROOT.iterdir() if d.is_dir()])
    rng = random.Random(RANDOM_SEED)
    selected = rng.sample(all_exps, 5)
    SELECTED_FILE.write_text(json.dumps(selected, indent=2))
    return selected


def load_experiment(exp_id):
    exp_dir = DATA_ROOT / exp_id
    structured = exp_dir / "structured-v1"

    summary = json.loads((exp_dir / "summary.json").read_text())
    game_key = list(summary.keys())[0]
    game_info = summary[game_key]

    players = {}
    for k, v in game_info.items():
        if k.startswith("Player"):
            players[v["name"]] = {
                "color": v["color"],
                "identity": v["identity"],
                "model": v.get("model", ""),
                "tasks": v.get("tasks", []),
            }

    events = []
    events_file = structured / "events_v1.jsonl"
    if events_file.exists():
        for line in events_file.read_text().splitlines():
            if line.strip():
                events.append(json.loads(line))

    agent_turns = []
    turns_file = structured / "agent_turns_v1.jsonl"
    if turns_file.exists():
        for line in turns_file.read_text().splitlines():
            if line.strip():
                agent_turns.append(json.loads(line))

    outcomes = []
    outcomes_file = structured / "outcomes_v1.jsonl"
    if outcomes_file.exists():
        for line in outcomes_file.read_text().splitlines():
            if line.strip():
                outcomes.append(json.loads(line))

    return {
        "exp_id": exp_id,
        "summary": game_info,
        "players": players,
        "events": events,
        "agent_turns": agent_turns,
        "outcomes": outcomes,
    }


def build_ground_truth_timeline(events, players):
    """Build a timeline of factual ground-truth events for the human judge."""
    timeline = []
    player_locations = {}
    player_alive = {p: True for p in players}

    for evt in events:
        actor = evt.get("actor", "")
        etype = evt.get("event_type", "")
        ts = evt.get("timestep", 0)
        phase = evt.get("phase", "")

        if etype == "MOVE":
            player_locations[actor] = evt.get("to_location", "")
            timeline.append({
                "type": "move",
                "timestep": ts,
                "phase": phase,
                "actor": actor,
                "identity": players.get(actor, {}).get("identity", "?"),
                "from": evt.get("from_location"),
                "to": evt.get("to_location"),
                "text": f"{actor} moved {evt.get('from_location')} → {evt.get('to_location')}",
            })
        elif etype == "KILL":
            target = evt.get("target", "")
            player_alive[target] = False
            timeline.append({
                "type": "kill",
                "timestep": ts,
                "phase": phase,
                "actor": actor,
                "identity": players.get(actor, {}).get("identity", "?"),
                "target": target,
                "location": evt.get("actor_location"),
                "witnesses": evt.get("actor_state_snapshot", {}).get("local_observable_players", []),
                "text": f"KILL: {actor} killed {target} in {evt.get('actor_location')}",
            })
        elif etype in ("COMPLETE TASK", "COMPLETE FAKE TASK"):
            timeline.append({
                "type": "task",
                "timestep": ts,
                "phase": phase,
                "actor": actor,
                "identity": players.get(actor, {}).get("identity", "?"),
                "fake": etype == "COMPLETE FAKE TASK",
                "action": evt.get("action_repr", ""),
                "text": f"{'FAKE ' if etype == 'COMPLETE FAKE TASK' else ''}TASK: {actor} — {evt.get('action_repr', '')}",
            })
        elif etype == "VENT":
            timeline.append({
                "type": "vent",
                "timestep": ts,
                "phase": phase,
                "actor": actor,
                "identity": players.get(actor, {}).get("identity", "?"),
                "text": f"VENT: {actor} vented",
            })
        elif etype == "CALL MEETING":
            timeline.append({
                "type": "meeting",
                "timestep": ts,
                "phase": phase,
                "actor": actor,
                "text": f"MEETING called by {actor}",
            })
        elif etype == "voteout":
            timeline.append({
                "type": "voteout",
                "timestep": ts,
                "phase": phase,
                "actor": actor,
                "identity": players.get(actor, {}).get("identity", "?"),
                "text": f"EJECTED: {actor} ({players.get(actor, {}).get('identity', '?')})",
            })

    return timeline


def build_player_timelines(events, players):
    """Build per-player event timeline (all actions up to each timestep)."""
    player_events = {p: [] for p in players}

    type_labels = {
        "MOVE": "move",
        "KILL": "kill",
        "COMPLETE TASK": "task",
        "COMPLETE FAKE TASK": "fake_task",
        "VENT": "vent",
        "CALL MEETING": "meeting",
        "voteout": "voteout",
        "VOTE": "vote",
        "SPEAK": "speak",
    }

    for evt in events:
        actor = evt.get("actor", "")
        etype = evt.get("event_type", "")
        ts = evt.get("timestep", 0)

        if actor not in player_events:
            continue

        entry = {"ts": ts, "type": type_labels.get(etype, etype)}

        if etype == "MOVE":
            entry["text"] = f"Moved {evt.get('from_location')} → {evt.get('to_location')}"
        elif etype == "KILL":
            entry["text"] = f"KILLED {evt.get('target')} in {evt.get('actor_location')}"
            entry["highlight"] = "kill"
        elif etype == "COMPLETE TASK":
            entry["text"] = f"Task: {evt.get('action_repr', '')}"
            entry["highlight"] = "task"
        elif etype == "COMPLETE FAKE TASK":
            entry["text"] = f"FAKE task: {evt.get('action_repr', '')}"
            entry["highlight"] = "fake"
        elif etype == "VENT":
            entry["text"] = f"VENTED at {evt.get('actor_location', '')}"
            entry["highlight"] = "vent"
        elif etype == "CALL MEETING":
            entry["text"] = f"Called meeting at {evt.get('actor_location', '')}"
            entry["highlight"] = "meeting"
        elif etype == "voteout":
            entry["text"] = f"Was EJECTED"
            entry["highlight"] = "voteout"
        elif etype == "VOTE":
            target = evt.get("target") or "skip"
            entry["text"] = f"Voted: {target}"
        elif etype == "SPEAK":
            text = evt.get("raw_text", "")
            entry["text"] = f'Said: "{text[:80]}{"…" if len(text) > 80 else ""}"'
        else:
            entry["text"] = etype

        player_events[actor].append(entry)

    return player_events


def normalize_speech_text(text):
    """Collapse whitespace so event text can be matched against logged speech."""
    return re.sub(r"\s+", " ", (text or "").strip())


def extract_turn_thinking(turn):
    raw = turn.get("raw_response_text", "")
    if "[Thinking Process]" not in raw:
        return ""
    start = raw.index("[Thinking Process]") + len("[Thinking Process]")
    end = raw.index("[Action]") if "[Action]" in raw else len(raw)
    return raw[start:end].strip()


def extract_turn_speech(turn):
    speech = (turn.get("speak_message") or "").strip()
    if speech:
        return speech

    raw = turn.get("raw_response_text", "")
    marker = "FINAL_SPEAK_MESSAGE:"
    if marker in raw:
        return raw.split(marker, 1)[1].strip()
    return ""


def build_meetings(events, players, agent_turns):
    """Group SPEAK events by meeting, with per-player timeline context."""
    meetings = {}

    # Build full per-player timeline (used for per-player activity buttons)
    all_player_timelines = build_player_timelines(events, players)

    # Build turn lookup for thinking process. Multiple meeting speeches can share the
    # same actor+timestep, so keep all candidates and match them by spoken text.
    turn_lookup = {}
    for t in agent_turns:
        agent_name = t.get("agent", {}).get("name", "")
        step = t.get("step", -1)
        speech_text = extract_turn_speech(t)
        if not agent_name or step < 0 or not speech_text:
            continue
        turn_lookup.setdefault((agent_name, step), []).append({
            "speech_key": normalize_speech_text(speech_text),
            "thinking": extract_turn_thinking(t),
        })

    # Collect kills by timestep
    kills_by_ts = {}
    for evt in events:
        if evt.get("event_type") == "KILL":
            ts = evt.get("timestep", 0)
            kills_by_ts.setdefault(ts, []).append({
                "killer": evt.get("actor"),
                "victim": evt.get("target"),
                "location": evt.get("actor_location"),
                "witnesses": evt.get("actor_state_snapshot", {}).get("local_observable_players", []),
            })

    for evt in events:
        if evt.get("event_type") != "SPEAK":
            continue

        mid = evt.get("meeting_id")
        if mid is None:
            continue

        if mid not in meetings:
            meetings[mid] = {
                "meeting_id": mid,
                "round": evt.get("round"),
                "timestep": evt.get("timestep"),
                "speeches": [],
            }

        actor = evt.get("actor", "")
        ts = evt.get("timestep", 0)
        identity = players.get(actor, {}).get("identity", "Unknown")

        # Get thinking process if available (LLM only)
        turn_data = turn_lookup.get((actor, ts), [])
        thinking = ""
        if turn_data:
            speech_key = normalize_speech_text(evt.get("raw_text", ""))
            match_idx = next((i for i, item in enumerate(turn_data)
                              if item["speech_key"] == speech_key), 0)
            matched = turn_data.pop(match_idx)
            thinking = matched.get("thinking", "")

        # Per-player timelines up to this timestep
        player_timelines_snapshot = {
            p: [e for e in tl if e["ts"] <= ts]
            for p, tl in all_player_timelines.items()
        }

        # Kills before this meeting
        prior_kills = []
        for k_ts, kills in kills_by_ts.items():
            if k_ts < ts:
                prior_kills.extend(kills)

        speech = {
            "event_id": evt.get("event_id"),
            "actor": actor,
            "identity": identity,
            "is_human": players.get(actor, {}).get("model", "").startswith("homo"),
            "model": players.get(actor, {}).get("model", ""),
            "timestep": ts,
            "round": evt.get("round"),
            "text": evt.get("raw_text", ""),
            "thinking": thinking,
            "player_timelines": player_timelines_snapshot,
            "prior_kills": prior_kills,
        }
        meetings[mid]["speeches"].append(speech)

    return sorted(meetings.values(), key=lambda m: m["meeting_id"])


def load_labels(exp_id):
    label_file = LABELS_DIR / f"{exp_id}.json"
    if label_file.exists():
        return json.loads(label_file.read_text())
    return {}


def save_labels(exp_id, labels):
    label_file = LABELS_DIR / f"{exp_id}.json"
    label_file.write_text(json.dumps(labels, indent=2))


@app.route("/")
def index():
    experiments = get_selected_experiments()
    exp_data = []
    for exp_id in experiments:
        labels = load_labels(exp_id)
        exp = load_experiment(exp_id)
        outcome = exp["outcomes"][0] if exp["outcomes"] else {}
        winner_map = {1: "Impostor Win", 2: "Impostor Win (timeout)", 3: "Crewmate Win (tasks)", 4: "Crewmate Win (ejection)"}
        exp_data.append({
            "exp_id": exp_id,
            "winner": outcome.get("winner"),
            "winner_reason": outcome.get("winner_reason", ""),
            "labeled_count": len(labels),
            "human_player": next((p for p, d in exp["players"].items() if "homosapiens" in d.get("model", "")), "?"),
        })
    return render_template("index.html", experiments=exp_data)


@app.route("/experiment/<exp_id>")
def experiment(exp_id):
    selected = get_selected_experiments()
    if exp_id not in selected:
        return "Experiment not found", 404

    exp = load_experiment(exp_id)
    timeline = build_ground_truth_timeline(exp["events"], exp["players"])
    meetings = build_meetings(exp["events"], exp["players"], exp["agent_turns"])
    labels = load_labels(exp_id)
    outcome = exp["outcomes"][0] if exp["outcomes"] else {}

    total_speeches = sum(len(m["speeches"]) for m in meetings)
    labeled_speeches = len(labels)

    return render_template(
        "experiment.html",
        exp_id=exp_id,
        players=exp["players"],
        timeline=timeline,
        meetings=meetings,
        labels=labels,
        outcome=outcome,
        total_speeches=total_speeches,
        labeled_speeches=labeled_speeches,
        experiments=selected,
    )


@app.route("/api/save_label", methods=["POST"])
def save_label():
    data = request.json
    exp_id = data.get("exp_id")
    event_id = data.get("event_id")
    label_data = data.get("labels")

    if not all([exp_id, event_id, label_data is not None]):
        return jsonify({"error": "Missing required fields"}), 400

    labels = load_labels(exp_id)
    labels[event_id] = label_data
    save_labels(exp_id, labels)

    return jsonify({"status": "ok", "saved": event_id})


@app.route("/api/get_labels/<exp_id>")
def get_labels(exp_id):
    return jsonify(load_labels(exp_id))


@app.route("/summary")
def summary():
    experiments = get_selected_experiments()
    all_labels = {}
    for exp_id in experiments:
        all_labels[exp_id] = load_labels(exp_id)
    return render_template("summary.html", experiments=experiments, all_labels=all_labels)


@app.route("/api/export")
def export_labels():
    experiments = get_selected_experiments()
    export = {}
    for exp_id in experiments:
        export[exp_id] = load_labels(exp_id)
    from flask import Response
    return Response(
        json.dumps(export, indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=human_labels.json"},
    )


# ── LLM review routes ─────────────────────────────────────────────────────────

@app.route("/llm")
def llm_index():
    all_exps = lj_discover_experiments(DATA_ROOT)
    exp_data = []
    for exp_id in all_exps:
        outcome = _get_outcome(exp_id)
        judge_counts = {name: len(_load_v2_labels(exp_id, slug))
                        for name, slug in JUDGES}
        exp_data.append({
            "exp_id":       exp_id,
            "winner":       outcome.get("winner"),
            "winner_reason": outcome.get("winner_reason", ""),
            "judge_counts": judge_counts,
        })
    return render_template("llm_index.html",
                           experiments=exp_data,
                           judges=[j[0] for j in JUDGES])


@app.route("/llm/<exp_id>")
def llm_experiment_view(exp_id):
    all_exps = lj_discover_experiments(DATA_ROOT)
    if exp_id not in all_exps:
        return "Experiment not found", 404

    exp     = lj_load_experiment(DATA_ROOT / exp_id)
    speeches = collect_impostor_speeches(exp)

    all_labels = {name: _load_v2_labels(exp_id, slug) for name, slug in JUDGES}

    cards = []
    for sp in speeches:
        cards.append({
            "event_id":       sp["event_id"],
            "actor":          sp["actor"],
            "is_human":       sp["is_human"],
            "timestep":       sp["timestep"],
            "meeting_number": sp["meeting_number"],
            "round":          sp["round"],
            "text":           sp["text"],
            "thinking":       sp.get("thinking_text", ""),
            "prompt":         build_user_prompt(sp),
            "labels":         {name: all_labels[name].get(sp["event_id"])
                               for name, _ in JUDGES},
        })

    idx     = all_exps.index(exp_id)
    outcome = _get_outcome(exp_id)

    return render_template(
        "llm_experiment.html",
        exp_id    = exp_id,
        cards     = cards,
        judges    = [j[0] for j in JUDGES],
        prev_exp  = all_exps[idx - 1] if idx > 0 else None,
        next_exp  = all_exps[idx + 1] if idx < len(all_exps) - 1 else None,
        outcome   = outcome,
        exp_idx   = idx + 1,
        total_exps = len(all_exps),
    )


# ── Verification routes ───────────────────────────────────────────────────────

@app.route("/verify")
def verify_index():
    selected  = get_verify_selected()
    human, llama = load_all_verify_pairs()
    corr      = compute_verify_correlation(human, llama)
    exp_rows  = []
    for folder_name, exp_ids in selected.items():
        for exp_id in exp_ids:
            h = load_verify_human_labels(folder_name, exp_id)
            ll = _load_llama_labels(folder_name, lj_safe_exp_id(exp_id))
            exp_rows.append({
                "folder_name": folder_name,
                "exp_id":      exp_id,
                "safe_eid":    lj_safe_exp_id(exp_id),
                "n_llama":     len(ll),
                "n_human":     len(h),
            })
    return render_template("verify_index.html",
                           experiments=exp_rows,
                           correlation=corr,
                           total_human=len(human))


@app.route("/verify/<folder_name>/<safe_eid>")
def verify_experiment_view(folder_name, safe_eid):
    selected = get_verify_selected()
    if folder_name not in selected:
        return "Folder not found", 404

    folder_path = REPO_ROOT / folder_name
    exps   = lj_discover_experiments(folder_path)
    exp_id = next((e for e in exps if lj_safe_exp_id(e) == safe_eid), None)
    if not exp_id:
        return "Experiment not found", 404

    exp      = lj_load_experiment(folder_path / exp_id)
    speeches = collect_impostor_speeches(exp)

    llama_labels = _load_llama_labels(folder_name, safe_eid)
    human_labels = load_verify_human_labels(folder_name, exp_id)

    cards = []
    for sp in speeches:
        eid = sp["event_id"]
        ll  = llama_labels.get(eid)
        if not ll:
            continue
        cards.append({
            "event_id":       eid,
            "actor":          sp["actor"],
            "is_human":       sp["is_human"],
            "timestep":       sp["timestep"],
            "meeting_number": sp["meeting_number"],
            "round":          sp["round"],
            "text":           sp["text"],
            "thinking":       sp.get("thinking_text", ""),
            "prompt":         build_user_prompt(sp),
            "llama_label":    ll,
            "human_label":    human_labels.get(eid),
        })

    all_exps = [(fn, eid) for fn, eids in selected.items() for eid in eids]
    curr_idx = next((i for i, (fn, eid) in enumerate(all_exps)
                     if fn == folder_name and eid == exp_id), 0)

    human, llama = load_all_verify_pairs()
    corr = compute_verify_correlation(human, llama)

    try:
        outcome = _get_outcome(exp_id)
    except Exception:
        outcome = {}

    return render_template("verify_experiment.html",
                           folder_name   = folder_name,
                           exp_id        = exp_id,
                           safe_eid      = safe_eid,
                           cards         = cards,
                           correlation   = corr,
                           mech_choices  = MECH_CHOICES,
                           goal_choices  = GOAL_CHOICES,
                           claim_choices = CLAIM_CHOICES,
                           score_fields  = SCORE_FIELDS,
                           prev_exp      = all_exps[curr_idx - 1] if curr_idx > 0 else None,
                           next_exp      = all_exps[curr_idx + 1] if curr_idx < len(all_exps) - 1 else None,
                           exp_idx       = curr_idx + 1,
                           total_exps    = len(all_exps))


@app.route("/api/verify/save_label", methods=["POST"])
def verify_save_label():
    data       = request.json
    folder_name = data.get("folder_name")
    exp_id      = data.get("exp_id")
    event_id    = data.get("event_id")
    label       = data.get("label")
    if not all([folder_name, exp_id, event_id, label is not None]):
        return jsonify({"error": "Missing fields"}), 400
    save_verify_human_label(folder_name, exp_id, event_id, label)
    human, llama = load_all_verify_pairs()
    return jsonify({"status": "ok", "correlation": compute_verify_correlation(human, llama)})


# ── Metrics dashboard ─────────────────────────────────────────────────────────

_FOLDER_LABELS = {
    "claude-haiku-4.5-human-crew": "Haiku HC",
    "claude-haiku-4.5-human-imp":  "Haiku HI",
    "claude-opus-4.6-human-crew":  "Opus HC",
    "claude-opus-4.6-human-imp":   "Opus HI",
    "gemini-3-flash-human-crew":   "Gemini HC",
    "gemini-3-flash-human-imp":    "Gemini HI",
    "openai-gpt-4o-mini-human-crew": "4o-mini HC",
    "openai-gpt-4o-mini-human-imp":  "4o-mini HI",
    "openai-gpt-5.4-human-crew":   "GPT-5.4 HC",
    "openai-gpt-5.4-human-imp":    "GPT-5.4 HI",
}


@app.route("/metrics")
def metrics_dashboard():
    entries = []
    for path in sorted((REPO_ROOT / "metrics").glob("*_metrics_v2.json")):
        folder_name = path.stem.replace("_metrics_v2", "")
        try:
            d = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            entries.append({
                "folder":  folder_name,
                "label":   _FOLDER_LABELS.get(folder_name, folder_name),
                "n_games": d.get("n_games", 0),
                "agg":     d.get("aggregate", {}),
            })
        except Exception:
            pass
    return render_template("metrics.html", entries=entries)


if __name__ == "__main__":
    app.run(debug=True, port=5050)
