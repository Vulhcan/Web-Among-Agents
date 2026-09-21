import json
from pandas import DataFrame, json_normalize
from functools import reduce
from typing import List, Dict
import pandas as pd
import os
from datetime import datetime
import platform
import sys


def setup_experiment(experiment_name, LOGS_PATH, DATE, COMMIT_HASH, DEFAULT_ARGS):
    """Set up experiment directory and files with an index-based system."""
    
    os.makedirs(LOGS_PATH, exist_ok=True)

    # Find the next available index for the current date
    next_index = 0
    while os.path.exists(os.path.join(LOGS_PATH, f"{DATE}_exp_{next_index}")):
        next_index += 1
    
    # Create the experiment name with the next index
    experiment_name = f"{DATE}_exp_{next_index}"
    
    experiment_path = os.path.join(LOGS_PATH, experiment_name)
    # Logged in place of the absolute path so run records stay portable
    # and do not embed the collector's home directory.
    try:
        experiment_ref = os.path.relpath(experiment_path, os.path.dirname(os.path.abspath(__file__)))
    except ValueError:  # different drive on Windows
        experiment_ref = os.path.join("expt-logs", experiment_name)
    experiment_ref = experiment_ref.replace(os.sep, "/")
    os.makedirs(experiment_path, exist_ok=True)
    
    # delete everything in the experiment path
    for file in os.listdir(experiment_path):
        os.remove(os.path.join(experiment_path, file))

    with open(
        os.path.join(experiment_path, "experiment-details.txt"), "w"
    ) as experiment_file:
        experiment_file.write(f"Experiment {experiment_ref}\n")
        experiment_file.write(f"Date: {DATE}\n")
        experiment_file.write(f"Commit: {COMMIT_HASH}\n")
        experiment_file.write(f"Experiment args: {DEFAULT_ARGS}\n")
        experiment_file.write(
            f"Tournament meta: id={os.getenv('TOURNAMENT_ID','')}, cell={os.getenv('TOURNAMENT_CELL','')}, run_label={os.getenv('TOURNAMENT_RUN_LABEL','')}, human_role={os.getenv('TOURNAMENT_HUMAN_ROLE','')}\n"
        )
        experiment_file.write(
            f"Model env: crewmate={os.getenv('OPENROUTER_CREWMATE_MODEL','')}, impostor={os.getenv('OPENROUTER_IMPOSTOR_MODEL','')}, timeout={os.getenv('OPENROUTER_TIMEOUT_SECONDS','60')}\n"
        )
        experiment_file.write(f"Logger module: {os.path.basename(__file__)}\n")
        experiment_file.write(f"Experiment index: {next_index}\n")

    os.environ["EXPERIMENT_PATH"] = experiment_path
    os.environ["EXPERIMENT_NAME"] = experiment_name
    os.environ["STREAMLIT"] = str(DEFAULT_ARGS.get("streamlit", False))
    os.environ["EXPERIMENT_INDEX"] = str(next_index)

    # Parallel structured logging track (v1) that does not replace legacy logs.
    structured_v1_path = os.path.join(experiment_path, "structured-v1")
    os.makedirs(structured_v1_path, exist_ok=True)
    os.environ["EXPERIMENT_PATH_STRUCTURED_V1"] = structured_v1_path

    run_record = {
        "schema_version": "v1",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "run_id": f"{experiment_name}",
        "experiment_path": experiment_ref,
        "commit_hash": COMMIT_HASH,
        "date": DATE,
        "experiment_index": next_index,
        "args": DEFAULT_ARGS,
        "runtime": {
            "python_version": sys.version,
            "platform": platform.platform(),
        },
        "tournament": {
            "id": os.getenv("TOURNAMENT_ID", ""),
            "cell": os.getenv("TOURNAMENT_CELL", ""),
            "run_label": os.getenv("TOURNAMENT_RUN_LABEL", ""),
            "human_role": os.getenv("TOURNAMENT_HUMAN_ROLE", ""),
            "notes": os.getenv("TOURNAMENT_NOTES", ""),
        },
        "env_snapshot": {
            "openrouter_crewmate_model": os.getenv("OPENROUTER_CREWMATE_MODEL", ""),
            "openrouter_impostor_model": os.getenv("OPENROUTER_IMPOSTOR_MODEL", ""),
            "openrouter_timeout_seconds": os.getenv("OPENROUTER_TIMEOUT_SECONDS", "60"),
            "openrouter_top_p": os.getenv("OPENROUTER_TOP_P", ""),
            "openrouter_temperature": os.getenv("OPENROUTER_TEMPERATURE", ""),
        },
    }
    with open(os.path.join(structured_v1_path, "runs.jsonl"), "a", encoding="utf-8") as f:
        json.dump(run_record, f, separators=(",", ": "))
        f.write("\n")
    
    return experiment_name

def load_game_summary(filepath: str) -> pd.DataFrame:
    # Read each line of the JSONL file
    with open(filepath, 'r') as file:
        data = [json.loads(line.strip()) for line in file]
    
    # Extract Game, Winner, and Winner Reason
    games_summary = [
        {
            "Game": game_id,
            "Winner": game_details.get("winner"),
            "Winner Reason": game_details.get("winner_reason")
        }
        for entry in data
        for game_id, game_details in entry.items()
    ]
    
    # Create DataFrame
    return pd.DataFrame(games_summary)

def read_jsonl_as_json(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def load_agent_logs_df(path: str) -> DataFrame:

    df: DataFrame = json_normalize(read_jsonl_as_json(path))
    
    action_cols = [
        "interaction.response.Action",
        "interaction.response.Action.action",
        "interaction.response.SPEAK Strategy.action",
        "interaction.response.ACTION",
        "interaction.response.Thinking Process.action",
    ]
    
    thinking_cols = [
        "interaction.response.Thinking Process",
        "interaction.response.Thinking Process.thought",
        "interaction.response.SPEAK Strategy.thought",
        "interaction.response.SPEAK Strategy",
        "interaction.response",
        "interaction.response.Action.thought",
    ]
    

    df["action"] = reduce(
        lambda x, y: x.combine_first(df[y]) if y in df else x,
        action_cols,
        df.assign(action=None)["action"]  # Start with a column of None
    )
    
    df["thought"] = reduce(
        lambda x, y: x.combine_first(df[y]) if y in df else x,
        thinking_cols,
        df.assign(thought=None)["thought"]  # Start with a column of None
    )
    
    df = df.drop(columns=(action_cols + thinking_cols), errors='ignore')

    return df
