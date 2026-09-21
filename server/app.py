from pathlib import Path
import os
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from .game_manager import CapacityError, GameManager
except ImportError:
    from game_manager import CapacityError, GameManager

ROOT_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT_DIR / "web"


def _allowed_origins() -> list[str]:
    configured = os.getenv("ALLOWED_ORIGINS", "").strip()
    if not configured:
        return ["*"]
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


app = FastAPI(title="AmongUs Thin Web Wrapper")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")

manager = GameManager()

if os.getenv("FLY_APP_NAME") and not os.getenv("BACKEND_API_KEY"):
    raise RuntimeError("BACKEND_API_KEY must be set for Fly deployments.")


class CreateGameRequest(BaseModel):
    crewmate_model: Optional[str] = None
    impostor_model: Optional[str] = None
    human_role: Optional[str] = None  # "crewmate" | "impostor" | "random" | None


class CreateGameResponse(BaseModel):
    game_id: int
    status: str
    human_role: str  # resolved role that was actually applied


class HumanActionRequest(BaseModel):
    game_id: int
    action_index: int = Field(ge=0)
    speech_text: Optional[str] = ""
    monitor_room: Optional[str] = ""


def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    api_access_key = os.getenv("BACKEND_API_KEY", "").strip()
    if not api_access_key:
        return
    if x_api_key != api_access_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/favicon.ico")
async def favicon() -> FileResponse:
    return FileResponse(str(WEB_DIR / "assets" / "skeld.png"))


@app.post("/create_game", response_model=CreateGameResponse)
async def create_game(
    request: CreateGameRequest,
    _: None = Depends(require_api_key),
) -> CreateGameResponse:
    try:
        game_id, resolved_role = await manager.create_game(
            crewmate_model=request.crewmate_model,
            impostor_model=request.impostor_model,
            human_role=request.human_role,
        )
    except CapacityError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CreateGameResponse(game_id=game_id, status="running", human_role=resolved_role)


@app.get("/game_state")
async def game_state(game_id: int, _: None = Depends(require_api_key)):
    try:
        return manager.get_state(game_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/human_action")
async def human_action(request: HumanActionRequest, _: None = Depends(require_api_key)):
    try:
        manager.submit_human_action(
            game_id=request.game_id,
            action_index=request.action_index,
            speech_text=request.speech_text,
            monitor_room=request.monitor_room,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "accepted"}
