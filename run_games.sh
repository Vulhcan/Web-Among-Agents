#!/usr/bin/env bash
# Example batch collection runs. Edit the model ids and counts for your setup.
#
# --num_games        how many games to play (they run concurrently)
# --crewmate_llm     model id driving every crewmate agent
# --impostor_llm     model id driving every impostor agent
# --tournament_style "random" uses the flags above for all games;
#                    "1on1" ignores them and samples a fresh model pair per game
#                    from DEFAULT_MODELS in main.py
#
# Each invocation writes a new directory under expt-logs/.

set -euo pipefail

python main.py --num_games 10 \
  --crewmate_llm "openai/gpt-4o-mini" \
  --impostor_llm "openai/gpt-4o-mini"

# Self-play with a different model:
# python main.py --num_games 10 \
#   --crewmate_llm "meta-llama/llama-3.3-70b-instruct" \
#   --impostor_llm "meta-llama/llama-3.3-70b-instruct"

# Mixed-model sampling across the roster in main.py:
# python main.py --num_games 50 --tournament_style 1on1
