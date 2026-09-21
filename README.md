# Among Us Data Collection Sandbox

A text-based *Among Us* environment for collecting game transcripts from LLM
agents and from human players competing against them.

The repository is a **data collection tool**, not an analysis pipeline. It runs
games and writes structured logs; what you do with those logs is up to you.

Two collection modes are supported:

- **LLM vs LLM** — run many games headlessly and in parallel across any models
  reachable through OpenRouter.
- **Human vs LLM** — a browser client where a person plays one seat in a game of
  LLM agents, backed by a small FastAPI server.

Both modes write the same log format.

---

## Setup

Requires Python 3.10 or newer.

```bash
git clone <your-fork-url>
cd amongus-data-collection

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Then create your `.env`:

```bash
cp .env.example .env
```

and set `OPENROUTER_API_KEY` to a key from [OpenRouter](https://openrouter.ai/).
Every option in `.env.example` is documented inline.

> All agent traffic goes through OpenRouter, so you pay per token. Start with a
> small `--num_games` and a cheap model to confirm your setup before scaling up.

---

## Mode A — LLM vs LLM

```bash
python main.py --num_games 10 \
  --crewmate_llm "openai/gpt-4o-mini" \
  --impostor_llm "openai/gpt-4o-mini"
```

| Flag | Default | Meaning |
|---|---|---|
| `--num_games` | `2` | Games to play. They run concurrently, capped by an internal semaphore. |
| `--crewmate_llm` | from `.env` | Model id driving every crewmate agent. |
| `--impostor_llm` | from `.env` | Model id driving every impostor agent. |
| `--tournament_style` | `random` | `random` uses the two flags above for all games. `1on1` ignores them and samples a fresh model pair per game from `DEFAULT_MODELS` in `main.py`. |
| `--name` | auto | Ignored for directory naming; runs are always named `<date>_exp_<n>`. |
| `--display_ui` | `False` | Tkinter map view. Only applies to single-game runs. |

Games are seven-player by default (`SEVEN_MEMBER_GAME`); switch to
`FIVE_MEMBER_GAME` in `main.py` for shorter, cheaper games.

`run_games.sh` holds worked examples you can copy and edit.

---

## Mode B — Human vs LLM

### Web client

Start the backend:

```bash
uvicorn server.app:app --port 8000
```

Then open <http://localhost:8000/>. The server serves the client from `web/` and
exposes three endpoints: `POST /create_game`, `GET /game_state`, and
`POST /human_action`.

Pick the human's role (`crewmate`, `impostor`, or `random`) and the agent models
in the UI when creating a game. Up to 8 games can run concurrently per server.

To point the client at a backend on another host, copy `web/config.example.js` to
`web/config.js` and set `window.API_BASE_URL`. `web/config.js` is gitignored.

### Terminal client

For a single game without the browser:

```bash
python run_human_vs_llms.py
```

This reads `OPENROUTER_CREWMATE_MODEL` and `OPENROUTER_IMPOSTOR_MODEL` from
`.env`, validates your key against OpenRouter before starting, and plays one
seven-player game in the terminal.

---

## What gets collected

Each run creates `expt-logs/<YYYY-MM-DD>_exp_<n>/`, numbered upward within a day:

```
expt-logs/2026-01-15_exp_0/
├── experiment-details.txt      # run config, date, commit, model + label metadata
├── agent-logs.json             # full agent interactions, one JSON object per line
├── agent-logs-compact.json     # the same turns, trimmed
├── api-calls.jsonl             # raw request/response pairs sent to OpenRouter
├── api-errors.jsonl            # failed calls (only written if errors occur)
├── summary.json                # per-game roster, winner, and winner reason
└── structured-v1/
    ├── runs.jsonl              # one record per run: args, runtime, labels, env snapshot
    ├── agent_turns_v1.jsonl    # one record per agent turn
    ├── api_calls_v1.jsonl      # one record per model call
    ├── events_v1.jsonl         # one record per in-game event
    └── outcomes_v1.jsonl       # one record per finished game
```

`structured-v1/` is the track to build analysis on. The legacy files beside it are
kept for continuity and are less regular.

Fields worth knowing when joining records:

- **Identifiers.** `run_id`, `game_id`, `event_id`, `round_id`, `meeting_id`, and
  `turn_id` thread the tables together. `game_id` has the form
  `<run_id>:game:<index>`.
- **Reproducibility.** `agent_turns_v1.jsonl` records `system_prompt`,
  `system_prompt_hash`, `prompt`, `full_response`, `model`, `prompt_profile`, and
  `aggression_level` for every turn. Stratify on `system_prompt_hash` before
  pooling runs.
- **Game state.** `events_v1.jsonl` carries a `phase_context` block (alive players
  by role, config limits), an `actor_state_snapshot` with its hash, and
  `audit_flags` marking parser errors and unknown event types. Drop or inspect
  flagged rows rather than trusting them silently.
- **Placeholders.** Event records include empty `deception_*` fields. These are
  slots for your own annotation pass; nothing in this repo populates them.

`expt-logs/` is gitignored. Collected data is yours to store and publish as you
see fit.

### Labelling runs

Five optional variables in `.env` — `TOURNAMENT_ID`, `TOURNAMENT_CELL`,
`TOURNAMENT_RUN_LABEL`, `TOURNAMENT_HUMAN_ROLE`, and `TOURNAMENT_NOTES` — are
recorded verbatim into `experiment-details.txt` and `structured-v1/runs.jsonl`.
They are free-form. Use them to tag a collection campaign so you can slice it
later without reconstructing which run was which.

### Prompt profiles

`PROMPT_PROFILE` selects the agent prompting condition:

- `baseline_v1` — the default prompts.
- `aggressive_v1` — adds role and phase directives on top of the baseline while
  preserving the output schema and parser contract. `AGGRESSION_LEVEL` (1–5)
  controls the intensity.

Both values are logged per turn. Treat them as separate conditions: do not pool
runs that used different profiles.

---

## Deployment

To host the human-facing client for remote participants:

```bash
docker build -t amongus-collect .
docker run -p 8000:8000 --env-file .env amongus-collect
```

`fly.toml` is a ready Fly.io config — set `app` to your own name before
deploying. When `FLY_APP_NAME` is present the server **requires**
`BACKEND_API_KEY`, and clients must send it as an `X-API-Key` header; set
`ALLOWED_ORIGINS` to restrict CORS. `vercel.json` covers deploying `web/` as a
static frontend against a separately hosted backend.

---

## Layout

```
.
├── main.py                 # LLM-vs-LLM batch runner
├── run_human_vs_llms.py    # single human-vs-LLM game in the terminal
├── utils.py                # run directory setup and log helpers
├── among-agents/           # game engine: agents, environment, prompts, configs
├── server/                 # FastAPI backend for the web client
├── web/                    # browser client
└── tests/
```

## Tests

```bash
pytest
```

## License

CC0 1.0 Universal — see [LICENSE](LICENSE).

## Acknowledgments

The game logic derives from
[AmongAgents](https://github.com/cyzus/among-agents) (Chi et al.), a text-based
Among Us environment for evaluating language models in social deduction games.

The sandbox framing — using the game as a model organism for studying agentic
deception — follows *Among Us: A Sandbox for Measuring and Detecting Agentic
Deception* ([arXiv:2504.04072](https://arxiv.org/abs/2504.04072)).
