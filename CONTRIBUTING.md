# Contributing

## Setup

```bash
pip install -r requirements.txt
pre-commit install
```

## Before opening a pull request

- `pytest` passes.
- `pre-commit run --all-files` is clean.
- No collected data is staged. `expt-logs/` is gitignored; keep it that way, and
  do not commit `.env` or `web/config.js`.

## Changing agent prompts

Prompt changes affect the comparability of collected data. If you modify
`among-agents/amongagents/agent/agent.py` or the prompt modules beside it, say so
explicitly in the pull request description so downstream users can tell which
runs are poolable. Every run already records a `system_prompt_hash`,
`prompt_profile`, and `aggression_level` in `structured-v1/`, so labelled data
stays interpretable across changes.
