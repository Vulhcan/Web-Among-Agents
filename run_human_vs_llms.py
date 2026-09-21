#!/usr/bin/env python3
import asyncio
import datetime
import os
import subprocess
import sys

import requests
from dotenv import load_dotenv

ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(ROOT_PATH, "among-agents"))

from amongagents.envs.configs.game_config import SEVEN_MEMBER_GAME
from amongagents.envs.game import AmongUs
from utils import setup_experiment


def die(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def get_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .strip()
            .decode("utf-8")
        )
    except Exception:
        return "unknown"


def validate_env() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        die("OPENROUTER_API_KEY is not set. Add it to .env or your environment.")
    return api_key


def validate_openrouter(api_key: str) -> None:
    url = "https://openrouter.ai/api/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
    except requests.RequestException as exc:
        die(f"OpenRouter unreachable: {exc}")

    if response.status_code in (401, 403):
        die(f"OpenRouter rejected the API key (status {response.status_code}).")
    if response.status_code >= 500:
        die(f"OpenRouter server error (status {response.status_code}).")
    if response.status_code >= 400:
        die(f"OpenRouter request failed (status {response.status_code}).")


async def run_game() -> None:
    load_dotenv()

    api_key = validate_env()
    validate_openrouter(api_key)

    os.environ["FLASK_ENABLED"] = "False"

    crewmate_model = os.getenv("OPENROUTER_CREWMATE_MODEL", "openrouter/free").strip()
    impostor_model = os.getenv("OPENROUTER_IMPOSTOR_MODEL", "openrouter/free").strip()
    if not crewmate_model:
        die("OPENROUTER_CREWMATE_MODEL is empty. Set it or remove it to use the default.")
    if not impostor_model:
        die("OPENROUTER_IMPOSTOR_MODEL is empty. Set it or remove it to use the default.")

    args = {
        "game_config": SEVEN_MEMBER_GAME,
        "include_human": True,
        "test": False,
        "personality": False,
        "agent_config": {
            "Impostor": "LLM",
            "Crewmate": "LLM",
            "IMPOSTOR_LLM_CHOICES": [impostor_model],
            "CREWMATE_LLM_CHOICES": [crewmate_model],
        },
        "UI": False,
    }

    logs_path = os.path.join(ROOT_PATH, "expt-logs")
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    commit_hash = get_commit_hash()

    setup_experiment(None, logs_path, date, commit_hash, args)

    game = AmongUs(
        game_config=args["game_config"],
        include_human=args["include_human"],
        test=args["test"],
        personality=args["personality"],
        agent_config=args["agent_config"],
        UI=None,
        game_index=1,
    )

    try:
        game.initialize_game()
    except Exception as exc:
        die(f"Agent initialization failed: {exc}")

    game_over = game.check_game_over()
    while not game_over:
        await game.game_step()
        game_over = game.check_game_over()

    if game.interviewer is not None:
        for agent in game.agents:
            await game.interviewer.auto_question(game, agent)

    game.report_winner(game_over)


if __name__ == "__main__":
    try:
        asyncio.run(run_game())
    except KeyboardInterrupt:
        print("\nGame stopped by user.")
