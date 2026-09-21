import asyncio
import datetime
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "among-agents"))

from amongagents.agent.agent import HumanAgent, human_action_futures
from amongagents.envs.configs.game_config import SEVEN_MEMBER_GAME
from amongagents.envs.game import AmongUs
from utils import setup_experiment


@dataclass
class GameRecord:
    game_id: int
    game: AmongUs
    status: str
    task: Optional[asyncio.Task] = None
    error: Optional[str] = None
    winner: Optional[int] = None
    winner_reason: Optional[str] = None


MAX_CONCURRENT_GAMES = 8


class CapacityError(RuntimeError):
    """Raised when the server has reached the maximum number of concurrent games."""


class GameManager:
    def __init__(self) -> None:
        # Keep deployed secret management authoritative (Fly secrets > local .env).
        # Local .env values are loaded only for missing keys.
        load_dotenv(ROOT_DIR / ".env", override=False)
        os.environ["FLASK_ENABLED"] = "True"
        self._records: Dict[int, GameRecord] = {}
        self._next_game_id = 1
        self._init_lock = asyncio.Lock()
        self._logs_path = ROOT_DIR / "expt-logs"

    def _get_commit_hash(self) -> str:
        try:
            return (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, stderr=subprocess.DEVNULL
                )
                .strip()
                .decode("utf-8")
            )
        except Exception:
            return "unknown"

    def _find_human_agent(self, game: AmongUs) -> Optional[HumanAgent]:
        agents = getattr(game, "agents", None) or []
        for agent in agents:
            if isinstance(agent, HumanAgent):
                return agent
        return None

    def _find_current_agent(self, game: AmongUs) -> Optional[Any]:
        current_player_name = getattr(game, "current_player", None)
        agents = getattr(game, "agents", None) or []
        if not current_player_name:
            return None
        for agent in agents:
            if getattr(agent, "player", None) and agent.player.name == current_player_name:
                return agent
        return None

    async def _run_game(self, game_id: int) -> None:
        record = self._records[game_id]
        try:
            record.status = "running"
            await record.game.run_game()
            record.status = "completed"
            summary = record.game.summary_json.get(f"Game {game_id}", {})
            record.winner = summary.get("winner")
            record.winner_reason = summary.get("winner_reason")
        except Exception as exc:
            record.status = "error"
            record.error = str(exc)
        finally:
            # Best-effort cleanup for stale futures.
            future = human_action_futures.get(game_id)
            if future is not None and not future.done():
                future.cancel()
            human_action_futures.pop(game_id, None)

    async def _wait_for_human_agent_ready(self, game_id: int, timeout_seconds: float = 10.0) -> None:
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while asyncio.get_running_loop().time() < deadline:
            record = self._records[game_id]
            if record.task and record.task.done():
                exc = record.task.exception()
                if exc:
                    raise RuntimeError(f"Game task failed during initialization: {exc}") from exc
                raise RuntimeError("Game finished before human initialization completed.")

            human_agent = self._find_human_agent(record.game)
            if human_agent is not None:
                return
            await asyncio.sleep(0.05)

        raise TimeoutError("Timed out waiting for HumanAgent initialization.")

    @staticmethod
    def _resolve_human_role(requested: Optional[str]) -> str:
        """
        Resolve the human player's role.

        Priority:
          1. HUMAN_ROLE env var (if set to 'crewmate' or 'impostor') — locks role server-side.
          2. requested value from the frontend ('crewmate' or 'impostor').
          3. Falls back to '' (random).

        Set HUMAN_ROLE=crewmate or HUMAN_ROLE=impostor in .env to lock the role.
        Leave unset (or set to 'random') to let the UI / default randomness decide.
        """
        env_role = os.getenv("HUMAN_ROLE", "").strip().lower()
        if env_role in {"crewmate", "impostor"}:
            return env_role
        req = (requested or "").strip().lower()
        if req in {"crewmate", "impostor"}:
            return req
        return ""  # random

    async def create_game(
        self,
        crewmate_model: Optional[str] = None,
        impostor_model: Optional[str] = None,
        human_role: Optional[str] = None,
    ) -> tuple:
        if not os.getenv("OPENROUTER_API_KEY"):
            raise RuntimeError("OPENROUTER_API_KEY is required.")

        # Tournament mode: model selection is sourced strictly from .env, not request payloads.
        # This avoids accidental model drift across runs.
        crewmate_model = os.getenv("OPENROUTER_CREWMATE_MODEL", "").strip()
        impostor_model = os.getenv("OPENROUTER_IMPOSTOR_MODEL", "").strip()
        if not crewmate_model or not impostor_model:
            raise RuntimeError(
                "OPENROUTER_CREWMATE_MODEL and OPENROUTER_IMPOSTOR_MODEL must be set in .env"
            )

        resolved_role = self._resolve_human_role(human_role)
        # TOURNAMENT_HUMAN_ROLE is read by AmongUs.initialize_players() via os.getenv.
        os.environ["TOURNAMENT_HUMAN_ROLE"] = resolved_role

        async with self._init_lock:
            active = sum(1 for r in self._records.values() if r.status not in {"completed", "error"})
            if active >= MAX_CONCURRENT_GAMES:
                raise CapacityError(
                    f"Server is at capacity ({MAX_CONCURRENT_GAMES} active games). Try again later."
                )

            game_id = self._next_game_id
            self._next_game_id += 1

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

            date = datetime.datetime.now().strftime("%Y-%m-%d")
            setup_experiment(
                None,
                str(self._logs_path),
                date,
                self._get_commit_hash(),
                args,
            )

            game = AmongUs(
                game_config=args["game_config"],
                include_human=args["include_human"],
                test=args["test"],
                personality=args["personality"],
                agent_config=args["agent_config"],
                UI=None,
                game_index=game_id,
            )
            record = GameRecord(game_id=game_id, game=game, status="initializing")
            self._records[game_id] = record

            record.task = asyncio.create_task(self._run_game(game_id))
            try:
                await self._wait_for_human_agent_ready(game_id)
            except Exception as exc:
                record.status = "error"
                record.error = str(exc)
                if record.task and not record.task.done():
                    record.task.cancel()
                raise
            # Game task may already be running or waiting on human input.
            if record.status not in {"completed", "error"}:
                record.status = "running"

            return game_id, resolved_role or "random"

    def _serialize_player_positions(self, game: AmongUs) -> List[Dict[str, Any]]:
        positions: List[Dict[str, Any]] = []
        for player in getattr(game, "players", []) or []:
            positions.append(
                {
                    "name": player.name,
                    "room": player.location,
                    "color": player.color,
                    "is_alive": bool(player.is_alive),
                }
            )
        return positions

    def _serialize_meeting_messages(self, game: AmongUs) -> List[Dict[str, Any]]:
        def _clean_meeting_message(raw_action_text: str) -> str:
            action_text = str(raw_action_text or "")
            payload_match = re.match(
                r"^SPEAK\s*:?\s*(.*)$",
                action_text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            message = payload_match.group(1).strip() if payload_match else action_text.strip()

            # If parser leakage included an [Action] block, keep only the last action payload.
            action_markers = list(re.finditer(r"\[Action\]", message, flags=re.IGNORECASE))
            if action_markers:
                message = message[action_markers[-1].end():].strip()

            # Drop markdown/emphasis wrappers around SPEAK directives.
            message = re.sub(r"^\s*\**\s*SPEAK\s*\**\s*:?\s*", "", message, flags=re.IGNORECASE)

            # Remove leaked reasoning blocks that may follow the actual message.
            message = re.split(r"\n\s*\[(Reasoning|Thinking Process)\]\s*", message, maxsplit=1, flags=re.IGNORECASE)[0].strip()
            message = re.sub(r"\[(Condensed Memory|Thinking Process|Action)\]", "", message, flags=re.IGNORECASE)
            message = re.sub(r"\bFINAL_SPEAK_MESSAGE\s*:\s*", "", message, flags=re.IGNORECASE)
            message = re.sub(r"\bFINAL_ACTION_INDEX\s*:\s*\d+\b", "", message, flags=re.IGNORECASE)
            message = re.split(r"\bFINAL_[A-Z_]+\s*:", message, maxsplit=1, flags=re.IGNORECASE)[0].strip()
            message = re.sub(r"\s+", " ", message).strip()

            if message.startswith('"') and message.endswith('"') and len(message) >= 2:
                message = message[1:-1].strip()
            if message.startswith("'") and message.endswith("'") and len(message) >= 2:
                message = message[1:-1].strip()
            return message or "..."

        messages: List[Dict[str, Any]] = []
        pending_announcements: List[Dict[str, Any]] = []

        def _meeting_announcement(entry: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
            player_obj = entry.get("player")
            action = entry.get("action")
            if player_obj is None or action is None:
                return None
            action_text = str(action)
            if not (
                re.match(r"^\s*CALL MEETING\b", action_text, flags=re.IGNORECASE)
                or re.match(r"^\s*REPORT DEAD BODY\b", action_text, flags=re.IGNORECASE)
            ):
                return None

            location = getattr(action, "current_location", None)
            location_text = f" at {location}" if location else ""
            if re.match(r"^\s*REPORT DEAD BODY\b", action_text, flags=re.IGNORECASE):
                reason = f"reported a dead body{location_text}"
            else:
                reason = f"pressed the emergency button{location_text}"

            timestep = entry.get("timestep")
            return {
                "id": f"announcement:{idx}:{timestep}:{player_obj.name}",
                "timestep": timestep,
                "round": "announcement",
                "player": "SYSTEM",
                "text": f"Meeting called by {player_obj.name}: {reason}.",
                "system": True,
            }

        for idx, entry in enumerate(getattr(game, "activity_log", []) or []):
            phase = entry.get("phase")
            if phase != "meeting":
                if phase == "task":
                    announcement = _meeting_announcement(entry, idx)
                    if announcement is not None:
                        pending_announcements.append(announcement)
                continue
            player_obj = entry.get("player")
            action = entry.get("action")
            if player_obj is None or action is None:
                continue
            action_text = str(action)
            if not re.match(r"^\s*SPEAK\b", action_text, flags=re.IGNORECASE):
                continue
            if pending_announcements:
                messages.extend(pending_announcements)
                pending_announcements = []
            message_text = _clean_meeting_message(action_text)
            round_number = entry.get("round")
            timestep = entry.get("timestep")
            message_id = f"{idx}:{timestep}:{round_number}:{player_obj.name}:{message_text}"
            messages.append(
                {
                    "id": message_id,
                    "timestep": timestep,
                    "round": round_number,
                    "player": player_obj.name,
                    "text": message_text,
                    "system": False,
                }
            )
        # Flush any pending meeting announcements even if no one has spoken yet.
        if pending_announcements:
            messages.extend(pending_announcements)
        return messages

    def _serialize_progress_snapshot(self, completed: int, total: int) -> Dict[str, Any]:
        safe_completed = max(0, int(completed))
        safe_total = max(0, int(total))
        safe_completed = min(safe_completed, safe_total)
        remaining = max(0, safe_total - safe_completed)
        ratio = (safe_completed / safe_total) if safe_total > 0 else 0.0
        return {
            "completed": safe_completed,
            "total": safe_total,
            "remaining": remaining,
            "ratio": ratio,
        }

    def _compute_crewmate_task_progress(self, game: AmongUs, human_agent=None) -> List[Dict[str, Any]]:
        result = []
        try:
            players = getattr(game, "players", None)
            if not players:
                return result
            human_player_name = (
                human_agent.player.name
                if (human_agent and getattr(human_agent, "player", None))
                else None
            )
            print(f"[crewmate_tasks] {len(players)} players, identities: {[getattr(p, 'identity', 'N/A') for p in players]}")
            for player in players:
                identity = getattr(player, "identity", "")
                if identity != "Crewmate":
                    continue
                if not getattr(player, "is_alive", True):
                    continue
                tasks_raw = getattr(player, "tasks", None)
                if tasks_raw is None:
                    tasks = []
                else:
                    try:
                        tasks = list(tasks_raw)
                    except Exception:
                        tasks = []
                total = len(tasks)
                completed = 0
                for task in tasks:
                    try:
                        if task.check_completion():
                            completed += 1
                    except Exception:
                        continue
                result.append({
                    "name": str(player.name),
                    "completed": int(completed),
                    "total": int(total),
                    "remaining": int(max(0, total - completed)),
                    "is_self": bool(str(player.name) == str(human_player_name)),
                })
            print(f"[crewmate_tasks] result: {result}")
        except Exception as exc:
            print(f"[crewmate_tasks ERROR] {type(exc).__name__}: {exc}")
        return result

    def _compute_global_task_progress(self, game: AmongUs) -> Dict[str, Any]:
        task_assignment = getattr(game, "task_assignment", None)
        assigned_tasks = getattr(task_assignment, "assigned_tasks", None) if task_assignment else None
        completed = 0
        total = 0
        if assigned_tasks is None:
            return self._serialize_progress_snapshot(completed=0, total=0)
        for task in assigned_tasks:
            assigned_player = getattr(task, "assigned_player", None)
            if assigned_player is not None and not bool(getattr(assigned_player, "is_alive", True)):
                # Keep behavior aligned with env.task_assignment.check_task_completion():
                # dead players' tasks are excluded from the global progress bar.
                continue
            total += 1
            try:
                if bool(task.check_completion()):
                    completed += 1
            except Exception:
                continue
        return self._serialize_progress_snapshot(completed=completed, total=total)

    def _compute_human_task_progress(self, human_agent: Optional[HumanAgent]) -> Dict[str, Any]:
        if human_agent is None or getattr(human_agent, "player", None) is None:
            return self._serialize_progress_snapshot(completed=0, total=0)

        tasks = getattr(human_agent.player, "tasks", None)
        if tasks is None:
            tasks = []
        completed = 0
        total = 0
        for task in tasks:
            total += 1
            try:
                if bool(task.check_completion()):
                    completed += 1
            except Exception:
                continue
        snapshot = self._serialize_progress_snapshot(completed=completed, total=total)
        snapshot["is_alive"] = bool(getattr(human_agent.player, "is_alive", True))
        return snapshot

    def get_state(self, game_id: int) -> Dict[str, Any]:
        record = self._records.get(game_id)
        if record is None:
            raise KeyError(f"Unknown game_id={game_id}")

        game = record.game
        game_config = getattr(game, "game_config", {}) or {}
        max_timesteps = game_config.get("max_timesteps")
        current_phase = getattr(game, "current_phase", None)
        timestep = getattr(game, "timestep", None)
        current_player = getattr(game, "current_player", None)

        current_agent = self._find_current_agent(game)
        is_human_turn = isinstance(current_agent, HumanAgent)
        human_agent = self._find_human_agent(game)

        available_actions = []
        player_info = None
        current_step = None
        if human_agent is not None:
            # Always expose human-perspective information to the web UI.
            try:
                player_info = human_agent.player.all_info_prompt()
            except Exception:
                player_info = None

        if is_human_turn and human_agent is not None:
            human_state = human_agent.get_current_state_for_web()
            current_step = human_state.get("current_step")
            raw_actions = human_state.get("available_actions") or []
            try:
                monitor_rooms = sorted(list(game.map.ship_map.nodes)) if hasattr(game, "map") and hasattr(game.map, "ship_map") else []
            except Exception:
                monitor_rooms = []
            for idx, action in enumerate(raw_actions):
                action_name = action.get("name", "")
                requires_location = bool(action.get("requires_location", False))
                available_actions.append(
                    {
                        "index": idx,
                        "name": action_name,
                        "requires_message": bool(action.get("requires_message", False)),
                        "requires_location": requires_location,
                        "location_options": monitor_rooms if requires_location else [],
                    }
                )

        player_positions = self._serialize_player_positions(game)
        meeting_messages = self._serialize_meeting_messages(game)
        task_progress = {
            "global": self._compute_global_task_progress(game),
            "human": self._compute_human_task_progress(human_agent),
        }
        crewmate_tasks = self._compute_crewmate_task_progress(game, human_agent)

        human_tasks: List[Dict[str, Any]] = []
        monitor_results: List[str] = []
        if human_agent and getattr(human_agent, "player", None):
            tasks_raw = getattr(human_agent.player, "tasks", None)
            if tasks_raw is not None:
                for task in tasks_raw:
                    try:
                        human_tasks.append({
                            "name": str(task.name),
                            "location": str(task.location),
                            "task_type": str(task.task_type),
                            "is_done": bool(task.check_completion()),
                        })
                    except Exception:
                        continue
            for entry in (getattr(human_agent.player, "observation_history", None) or []):
                s = str(entry)
                if "Monitor Record:" in s:
                    monitor_results.append(s)

        kill_cooldown: Optional[int] = None
        if human_agent and getattr(getattr(human_agent, "player", None), "identity", "") == "Impostor":
            kill_cooldown = int(getattr(human_agent.player, "kill_cooldown", 0) or 0)

        return {
            "game_id": game_id,
            "status": record.status,
            "error": record.error,
            "winner": record.winner,
            "winner_reason": record.winner_reason,
            "initialized": bool(getattr(game, "agents", None)),
            "has_human": human_agent is not None,
            "timestep": timestep,
            "max_timesteps": max_timesteps,
            "current_phase": current_phase,
            "current_player": current_player,
            "is_human_turn": is_human_turn,
            "human_player_name": human_agent.player.name if human_agent else None,
            "human_player_identity": human_agent.player.identity if human_agent else None,
            "human_impostor_teammates": (
                getattr(human_agent, "known_impostor_teammates", []) if human_agent else []
            ),
            "current_step": current_step,
            "player_info": player_info,
            "available_actions": available_actions,
            "player_positions": player_positions,
            "meeting_messages": meeting_messages,
            "task_progress": task_progress,
            "crewmate_tasks": crewmate_tasks,
            "human_tasks": human_tasks,
            "monitor_results": monitor_results,
            "kill_cooldown": kill_cooldown,
        }

    def submit_human_action(
        self,
        game_id: int,
        action_index: int,
        speech_text: Optional[str],
        monitor_room: Optional[str] = None,
    ) -> None:
        record = self._records.get(game_id)
        if record is None:
            raise KeyError(f"Unknown game_id={game_id}")
        if record.status in {"completed", "error"}:
            raise RuntimeError(f"Game {game_id} is already {record.status}.")

        human_agent = self._find_human_agent(record.game)
        if human_agent is None:
            raise RuntimeError(f"Game {game_id} has no human agent.")

        future = human_action_futures.get(game_id)
        if future is None or future.done():
            raise RuntimeError("Game is not currently waiting for a human action.")

        current_actions = human_agent.current_available_actions or []
        if action_index < 0 or action_index >= len(current_actions):
            raise ValueError(f"action_index out of range: {action_index}")

        payload = {
            "action_index": action_index,
            "message": speech_text or "",
            "monitor_room": monitor_room or "",
        }
        future.set_result(payload)
