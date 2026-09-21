/*
  Thin web client for the AmongUs backend.
  Backend contract:
  - POST /create_game
  - GET /game_state?game_id=<id>
  - POST /human_action
*/

let gameId = null;
let previousState = null;
let pollTimer = null;
let pollInFlight = false;
let consecutivePollErrors = 0;
let actionSubmitInFlight = false;
let selectedAction = null;
let lastIsHumanTurn = null;
let lastActionSignature = "";
let lastActiveRoom = null;
const seenMeetingMessages = new Set();
const seenVisibleLogKeys = new Set();
const playerState = new Map();
const playerTokenEls = new Map();
const roomPlayerContainers = new Map();
let missionBriefCaptured = false;
const hiddenAliasByName = new Map();
let hiddenAliasCounter = 1;
let previousPhase = null;
const knownImpostorTeammates = new Set();

const ROOM_ORDER = [
  "Cafeteria",
  "Weapons",
  "Navigation",
  "O2",
  "Shields",
  "Communications",
  "Storage",
  "Admin",
  "Electrical",
  "Lower Engine",
  "Security",
  "Reactor",
  "Upper Engine",
  "Medbay",
  "Unknown",
];

const ROOM_COORDS = {
  Cafeteria: { x: 52, y: 17 },
  Weapons: { x: 76, y: 29 },
  Navigation: { x: 91, y: 45 },
  O2: { x: 69, y: 43 },
  Shields: { x: 79, y: 73 },
  Communications: { x: 64, y: 87 },
  Storage: { x: 49, y: 64 },
  Admin: { x: 51, y: 51 },
  Electrical: { x: 35, y: 61 },
  "Lower Engine": { x: 14, y: 74 },
  Security: { x: 22, y: 52 },
  Reactor: { x: 5, y: 44 },
  "Upper Engine": { x: 15, y: 29 },
  Medbay: { x: 30, y: 38 },
  Unknown: { x: 89, y: 11 },
};

const ROOM_PATHS = [
  ["Cafeteria", "Weapons"],
  ["Cafeteria", "Admin"],
  ["Cafeteria", "Upper Engine"],
  ["Cafeteria", "Medbay"],
  ["Weapons", "Navigation"],
  ["Weapons", "O2"],
  ["Navigation", "Shields"],
  ["O2", "Shields"],
  ["O2", "Admin"],
  ["Shields", "Communications"],
  ["Shields", "Storage"],
  ["Communications", "Storage"],
  ["Storage", "Admin"],
  ["Storage", "Electrical"],
  ["Storage", "Lower Engine"],
  ["Admin", "Electrical"],
  ["Electrical", "Lower Engine"],
  ["Lower Engine", "Reactor"],
  ["Lower Engine", "Security"],
  ["Lower Engine", "Upper Engine"],
  ["Security", "Reactor"],
  ["Security", "Upper Engine"],
  ["Reactor", "Upper Engine"],
  ["Upper Engine", "Medbay"],
];

const COLOR_MAP = {
  red: "#e05a4f",
  blue: "#4f8de0",
  green: "#2f8f4e",
  yellow: "#e6b93f",
  orange: "#d8843b",
  purple: "#8a63d2",
  pink: "#d56aa6",
  brown: "#8a5a3a",
  black: "#5e6675",
  white: "#d9e0ea",
  cyan: "#4fb9c9",
  lime: "#a7d94f",
  gray: "#7f8794",
};

const createGameBtn = document.getElementById("create-game-btn");
const createStatus = document.getElementById("create-status");
const roleSelectEl = document.getElementById("role-select");
const winnerBannerEl = document.getElementById("winner-banner");
const soundToggleBtn = document.getElementById("sound-toggle-btn");
const soundVolumeEl = document.getElementById("sound-volume");
const errorBanner = document.getElementById("error-banner");
const gameView = document.getElementById("game-view");
const logView = document.getElementById("log-view");
const mapGrid = document.getElementById("map-grid");

const gameIdEl = document.getElementById("game-id");
const statusEl = document.getElementById("status");
const phaseEl = document.getElementById("phase");
const timestepEl = document.getElementById("timestep");
const currentPlayerEl = document.getElementById("current-player");
const turnStateEl = document.getElementById("turn-state");
const waitingMessageEl = document.getElementById("waiting-message");
const actionsEl = document.getElementById("actions");
const latestLogEl = document.getElementById("latest-log");
const missionBriefEl = document.getElementById("mission-brief");
const notesBoxEl = document.getElementById("notes-box");
const roleBannerEl = document.getElementById("role-banner");

const tabLiveLogsBtn = document.getElementById("tab-live-logs");
const tabNotesBtn = document.getElementById("tab-notes");
const panelLiveLogsEl = document.getElementById("panel-live-logs");
const panelNotesEl = document.getElementById("panel-notes");
const monitorFeedEl = document.getElementById("monitor-feed");
const missionBriefPanelEl = document.getElementById("mission-brief-panel");
const tabMapBriefBtn = document.getElementById("tab-map-brief");
const tabMapMonitorBtn = document.getElementById("tab-map-monitor");
const panelMapBriefEl = document.getElementById("panel-map-brief");
const panelMapMonitorEl = document.getElementById("panel-map-monitor");
const globalTaskBarGroupEl = document.getElementById("global-task-bar-group");
const globalTaskProgressTextEl = document.getElementById("global-task-progress-text");
const globalTaskProgressFillEl = document.getElementById("global-task-progress-fill");
const humanTaskBarGroupEl = document.getElementById("human-task-bar-group");

const gameOverModalEl = document.getElementById("game-over-modal");
const gameOverReasonEl = document.getElementById("game-over-reason");
const gameOverCloseBtn = document.getElementById("game-over-close-btn");
let gameOverModalShown = false;

const speechBox = document.getElementById("speech-box");
const speechTextInput = document.getElementById("speech-text");
const submitSpeechBtn = document.getElementById("submit-speech-btn");
const monitorBox = document.getElementById("monitor-box");
const monitorRoomSelect = document.getElementById("monitor-room");
const submitMonitorBtn = document.getElementById("submit-monitor-btn");

const meetingPanel = document.getElementById("meeting-panel");
const meetingFeed = document.getElementById("meeting-feed");
const taskFeed = document.getElementById("task-feed");
const humanTaskProgressTextEl = document.getElementById("human-task-progress-text");
const humanTaskProgressFillEl = document.getElementById("human-task-progress-fill");
const humanTasksListEl = document.getElementById("human-tasks-list");
const crewmateTasksListEl = document.getElementById("crewmate-tasks-list");
const impostorPanelEl = document.getElementById("impostor-panel");
const killCooldownValueEl = document.getElementById("kill-cooldown-value");
const seenTaskEvents = new Set();
const PLAYER_NOTES_STORAGE_KEY = "amongus_player_notes";
const API_BASE_STORAGE_KEY = "amongus_api_base_url";
const API_AUTH_STORAGE_KEY = "amongus_api_auth_key";
const SOUND_ENABLED_KEY = "amongus_sound_enabled";
const SOUND_VOLUME_KEY = "amongus_sound_volume";

const API_BASE_URL = (() => {
  const params = new URLSearchParams(window.location.search);
  const queryApiBase = (params.get("api_base") || "").trim();
  if (queryApiBase) {
    window.localStorage.setItem(API_BASE_STORAGE_KEY, queryApiBase);
  }
  const configured = String(window.API_BASE_URL || "").trim();
  const persisted = String(window.localStorage.getItem(API_BASE_STORAGE_KEY) || "").trim();
  const raw = queryApiBase || configured || persisted || "";
  return raw.replace(/\/+$/, "");
})();

const API_AUTH_KEY = (() => {
  const params = new URLSearchParams(window.location.search);
  const queryApiKey = (params.get("api_key") || "").trim();
  if (queryApiKey) {
    window.localStorage.setItem(API_AUTH_STORAGE_KEY, queryApiKey);
  }
  const configured = String(window.API_AUTH_KEY || "").trim();
  const persisted = String(window.localStorage.getItem(API_AUTH_STORAGE_KEY) || "").trim();
  return queryApiKey || configured || persisted || "";
})();

function apiUrl(path) {
  const suffix = String(path || "");
  if (!API_BASE_URL) {
    return suffix;
  }
  if (suffix.startsWith("http://") || suffix.startsWith("https://")) {
    return suffix;
  }
  return `${API_BASE_URL}${suffix}`;
}

function initMapSkeleton() {
  mapGrid.innerHTML = "";
  roomPlayerContainers.clear();

  const mapStage = document.createElement("div");
  mapStage.className = "map-stage";

  const overlay = document.createElement("div");
  overlay.className = "map-overlay";
  const pathLayer = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  pathLayer.setAttribute("class", "map-path-layer");
  pathLayer.setAttribute("viewBox", "0 0 100 100");
  pathLayer.setAttribute("preserveAspectRatio", "none");

  ROOM_PATHS.forEach(([fromRoom, toRoom]) => {
    const from = ROOM_COORDS[fromRoom];
    const to = ROOM_COORDS[toRoom];
    if (!from || !to) {
      return;
    }
    const path = document.createElementNS("http://www.w3.org/2000/svg", "line");
    path.setAttribute("x1", String(from.x));
    path.setAttribute("y1", String(from.y));
    path.setAttribute("x2", String(to.x));
    path.setAttribute("y2", String(to.y));
    path.setAttribute("class", "room-path");
    pathLayer.appendChild(path);
  });

  ROOM_ORDER.forEach((roomName) => {
    const anchor = document.createElement("div");
    anchor.className = "room-anchor";
    anchor.dataset.room = roomName;
    const point = ROOM_COORDS[roomName] || ROOM_COORDS.Unknown;
    anchor.style.left = `${point.x}%`;
    anchor.style.top = `${point.y}%`;

    const titleEl = document.createElement("div");
    titleEl.className = "room-name";
    titleEl.textContent = roomName;

    const playersEl = document.createElement("div");
    playersEl.className = "room-players";

    anchor.appendChild(titleEl);
    anchor.appendChild(playersEl);
    overlay.appendChild(anchor);
    roomPlayerContainers.set(roomName, playersEl);
  });

  mapStage.appendChild(pathLayer);
  mapStage.appendChild(overlay);
  mapGrid.appendChild(mapStage);
}

const audioState = {
  ctx: null,
  enabled: true,
  volume: 0.35,
};

function ensureAudioContext() {
  if (audioState.ctx) {
    return audioState.ctx;
  }
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) {
    return null;
  }
  audioState.ctx = new Ctx();
  return audioState.ctx;
}

function playTone({ freq = 440, duration = 0.12, type = "sine", startAt = 0, gain = 1 }) {
  if (!audioState.enabled) {
    return;
  }
  const ctx = ensureAudioContext();
  if (!ctx) {
    return;
  }
  if (ctx.state === "suspended") {
    ctx.resume();
  }
  const osc = ctx.createOscillator();
  const amp = ctx.createGain();
  osc.type = type;
  osc.frequency.value = freq;
  amp.gain.value = 0.0001;
  const now = ctx.currentTime + startAt;
  const scaledGain = Math.max(0.0001, Math.min(1, audioState.volume * gain));
  amp.gain.exponentialRampToValueAtTime(scaledGain, now + 0.01);
  amp.gain.exponentialRampToValueAtTime(0.0001, now + duration);
  osc.connect(amp);
  amp.connect(ctx.destination);
  osc.start(now);
  osc.stop(now + duration + 0.02);
}

function playSfx(name) {
  if (!audioState.enabled) {
    return;
  }
  if (name === "meeting-start") {
    playTone({ freq: 523, duration: 0.08, type: "triangle", startAt: 0, gain: 0.8 });
    playTone({ freq: 659, duration: 0.1, type: "triangle", startAt: 0.09, gain: 0.8 });
    playTone({ freq: 784, duration: 0.12, type: "triangle", startAt: 0.2, gain: 0.9 });
    return;
  }
  if (name === "message") {
    playTone({ freq: 700, duration: 0.05, type: "sine", startAt: 0, gain: 0.45 });
    return;
  }
  if (name === "game-over") {
    playTone({ freq: 392, duration: 0.12, type: "square", startAt: 0, gain: 0.75 });
    playTone({ freq: 330, duration: 0.12, type: "square", startAt: 0.13, gain: 0.75 });
    playTone({ freq: 262, duration: 0.18, type: "square", startAt: 0.28, gain: 0.85 });
    return;
  }
  if (name === "human-turn") {
    playTone({ freq: 659, duration: 0.08, type: "triangle", startAt: 0, gain: 0.8 });
    playTone({ freq: 880, duration: 0.1, type: "triangle", startAt: 0.09, gain: 0.9 });
  }
}

function loadSoundPrefs() {
  const enabledRaw = window.localStorage.getItem(SOUND_ENABLED_KEY);
  const volumeRaw = window.localStorage.getItem(SOUND_VOLUME_KEY);
  if (enabledRaw !== null) {
    audioState.enabled = enabledRaw === "true";
  }
  if (volumeRaw !== null) {
    const parsed = Number(volumeRaw);
    if (!Number.isNaN(parsed)) {
      audioState.volume = Math.max(0, Math.min(1, parsed));
    }
  }
  if (soundToggleBtn) {
    soundToggleBtn.textContent = `Sound: ${audioState.enabled ? "On" : "Off"}`;
  }
  if (soundVolumeEl) {
    soundVolumeEl.value = String(Math.round(audioState.volume * 100));
  }
}

function setGameActive(active) {
  if (roleSelectEl) roleSelectEl.classList.toggle("hidden", active);
}

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.classList.remove("hidden");
}

function clearError() {
  errorBanner.textContent = "";
  errorBanner.classList.add("hidden");
}

function setActiveJournalTab(tabName) {
  [tabLiveLogsBtn, tabNotesBtn].forEach((t) => t && t.classList.remove("active"));
  panelLiveLogsEl.classList.add("hidden");
  panelNotesEl.classList.add("hidden");
  if (tabName === "notes") {
    tabNotesBtn.classList.add("active");
    panelNotesEl.classList.remove("hidden");
    return;
  }
  tabLiveLogsBtn.classList.add("active");
  panelLiveLogsEl.classList.remove("hidden");
}

function setActiveMapTab(tabName) {
  [tabMapBriefBtn, tabMapMonitorBtn].forEach((t) => t && t.classList.remove("active"));
  if (panelMapBriefEl) panelMapBriefEl.classList.add("hidden");
  if (panelMapMonitorEl) panelMapMonitorEl.classList.add("hidden");
  if (tabName === "monitor") {
    if (tabMapMonitorBtn) tabMapMonitorBtn.classList.add("active");
    if (panelMapMonitorEl) panelMapMonitorEl.classList.remove("hidden");
    return;
  }
  if (tabMapBriefBtn) tabMapBriefBtn.classList.add("active");
  if (panelMapBriefEl) panelMapBriefEl.classList.remove("hidden");
}

function loadNotes() {
  const saved = window.localStorage.getItem(PLAYER_NOTES_STORAGE_KEY) || "";
  notesBoxEl.value = saved;
}

function stopPolling() {
  if (pollTimer !== null) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
}

function scheduleNextPoll(delayMs = 1000) {
  if (gameId === null) {
    return;
  }
  if (pollTimer !== null) {
    clearTimeout(pollTimer);
  }
  pollTimer = setTimeout(() => {
    pollGameState();
  }, Math.max(200, Number(delayMs) || 1000));
}

function startPollingNow() {
  stopPolling();
  scheduleNextPoll(50);
}

function appendStatusLine(text) {
  createStatus.textContent = text;
}

async function fetchJson(url, options = {}) {
  const requestOptions = { ...options };
  const headers = new Headers(options.headers || {});
  if (API_AUTH_KEY) {
    headers.set("X-API-Key", API_AUTH_KEY);
  }
  // Bypass ngrok's free-tier browser interstitial page
  headers.set("ngrok-skip-browser-warning", "1");
  requestOptions.headers = headers;

  const response = await fetch(url, requestOptions);
  const raw = await response.text();
  let payload = {};
  if (raw) {
    try {
      payload = JSON.parse(raw);
    } catch (_) {
      payload = { detail: raw };
    }
  }
  if (!response.ok) {
    const detail = payload && payload.detail ? payload.detail : `HTTP ${response.status}`;
    throw new Error(detail);
  }
  return payload;
}

function hideSpeechInput() {
  selectedAction = null;
  speechTextInput.value = "";
  speechBox.classList.add("hidden");
  if (monitorRoomSelect) {
    monitorRoomSelect.innerHTML = "";
  }
  monitorBox.classList.add("hidden");
}

function setActionButtonsEnabled(enabled) {
  const buttons = actionsEl.querySelectorAll("button");
  buttons.forEach((button) => {
    button.disabled = !enabled;
  });
}

function clearActionButtons() {
  actionsEl.innerHTML = "";
}

function buildActionSignature(actions) {
  return actions
    .map(
      (action) =>
        `${action.index}|${action.name}|${action.requires_message ? 1 : 0}|${action.requires_location ? 1 : 0}`
    )
    .join("||");
}

function isMeetingPhase(phase) {
  return String(phase || "").toLowerCase().includes("meeting");
}

function canonicalRoomName(roomName) {
  const normalized = String(roomName || "").trim().toLowerCase();
  if (!normalized) {
    return "Unknown";
  }
  const match = ROOM_ORDER.find((room) => room.toLowerCase() === normalized);
  return match || "Unknown";
}

function parseColorName(playerName) {
  const parts = String(playerName).split(":");
  if (parts.length < 2) {
    return "white";
  }
  return parts[1].trim().toLowerCase();
}

function colorHexFromPlayerName(playerName) {
  const colorName = parseColorName(playerName);
  return COLOR_MAP[colorName] || COLOR_MAP.gray;
}

function hexToRgba(hex, alpha) {
  const normalized = String(hex || "").replace("#", "");
  if (normalized.length !== 6) {
    return `rgba(127, 135, 148, ${alpha})`;
  }
  const r = parseInt(normalized.slice(0, 2), 16);
  const g = parseInt(normalized.slice(2, 4), 16);
  const b = parseInt(normalized.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function extractPlayerLabelFromText(text) {
  const match = String(text || "").match(/(Player\s+\d+:\s*[A-Za-z]+)/);
  return match ? match[1] : null;
}

function parsePlayersLine(value) {
  if (!value || value.toLowerCase() === "none") {
    return [];
  }
  return value
    .split(",")
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0);
}

function ensurePlayerRecord(playerName) {
  if (!playerState.has(playerName)) {
    playerState.set(playerName, {
      name: playerName,
      displayName: playerName,
      room: "Unknown",
      colorName: parseColorName(playerName),
      isVisible: false,
      isHuman: false,
      isDead: false,
    });
  }
  return playerState.get(playerName);
}

function extractCurrentRoomFromInfo(state) {
  const info = String(state.player_info || "");
  const locationMatch = info.match(/Current Location:\s*([^\n]+)/);
  if (!locationMatch) {
    return null;
  }
  return canonicalRoomName(locationMatch[1]);
}

function extractCurrentRoomOccupantsFromInfo(state, roomName) {
  const occupants = new Set();
  const info = String(state.player_info || "");
  const roomLinePattern = /Players in ([^:]+):\s*([^\n]+)/g;
  let match;
  while ((match = roomLinePattern.exec(info)) !== null) {
    const listedRoom = canonicalRoomName(match[1]);
    if (listedRoom !== roomName) {
      continue;
    }
    const players = parsePlayersLine(match[2]);
    players.forEach((rawName) => {
      const cleanName = String(rawName).replace(/\s*\(dead\)\s*/g, "").trim();
      if (cleanName) {
        occupants.add(cleanName);
      }
    });
  }
  return occupants;
}

function getHiddenAlias(playerName) {
  if (!hiddenAliasByName.has(playerName)) {
    hiddenAliasByName.set(playerName, `Unknown ${hiddenAliasCounter}`);
    hiddenAliasCounter += 1;
  }
  return hiddenAliasByName.get(playerName);
}

function buildPositionMapFromSnapshot(state) {
  const positionMap = new Map();
  const positions = Array.isArray(state.player_positions) ? state.player_positions : [];
  const humanRoom = extractCurrentRoomFromInfo(state);
  const visibleNames = humanRoom ? extractCurrentRoomOccupantsFromInfo(state, humanRoom) : new Set();
  if (state.human_player_name) {
    visibleNames.add(state.human_player_name);
  }

  positions.forEach((entry) => {
    if (!entry || !entry.name) {
      return;
    }
    const isVisible = visibleNames.has(entry.name);
    positionMap.set(entry.name, {
      room: isVisible ? canonicalRoomName(entry.room) : "Unknown",
      colorName: isVisible ? String(entry.color || parseColorName(entry.name)).toLowerCase() : "gray",
      displayName: isVisible ? entry.name : getHiddenAlias(entry.name),
      isVisible,
      isAlive: entry.is_alive !== false,
    });
  });
  return positionMap;
}

function ensureTokenEl(playerName) {
  if (playerTokenEls.has(playerName)) {
    return playerTokenEls.get(playerName);
  }
  const record = ensurePlayerRecord(playerName);
  const token = document.createElement("div");
  token.className = "player-token";
  token.dataset.player = playerName;
  token.title = record.displayName || record.name;
  const numberMatch = String(record.displayName || "").match(/Player\s+(\d+)/i);
  token.textContent = numberMatch ? `P${numberMatch[1]}` : "?";
  playerTokenEls.set(playerName, token);
  return token;
}

function setHumanTag(token, isHuman) {
  token.classList.toggle("human", isHuman);
  const existing = token.querySelector(".you-tag");
  if (isHuman && !existing) {
    const youTag = document.createElement("span");
    youTag.className = "you-tag";
    youTag.textContent = "YOU";
    token.appendChild(youTag);
  }
  if (!isHuman && existing) {
    existing.remove();
  }
}

function styleToken(token, record) {
  token.style.background = COLOR_MAP[record.colorName] || "#f3f6fb";
  token.style.opacity = record.isDead ? "0.45" : "1";
  token.title = record.isVisible || record.isHuman ? (record.displayName || record.name) : "Unknown player";
  if (record.isVisible || record.isHuman) {
    const numberMatch = String(record.displayName || record.name).match(/Player\s+(\d+)/i);
    token.textContent = numberMatch ? `P${numberMatch[1]}` : "P";
  } else {
    token.textContent = "?";
  }
  setHumanTag(token, record.isHuman && (record.isVisible || record.name === record.displayName));
  const isTeammate = knownImpostorTeammates.has(record.name);
  token.classList.toggle("impostor-teammate", isTeammate);
}

function moveTokenToRoom(playerName, roomName) {
  const token = ensureTokenEl(playerName);
  const targetRoom = canonicalRoomName(roomName);
  const container = roomPlayerContainers.get(targetRoom);
  if (!container) {
    return;
  }
  if (token.parentElement !== container) {
    container.appendChild(token);
    token.classList.add("moving");
    window.setTimeout(() => token.classList.remove("moving"), 230);
  }
}

function snapshotPositions() {
  const positions = {};
  playerState.forEach((record, name) => {
    positions[name] = record.room;
  });
  return positions;
}

function extractKnownPositionsFromInfo(state) {
  const info = String(state.player_info || "");
  if (!info) {
    return;
  }

  const roomOccupants = new Map();
  const roomLinePattern = /Players in ([^:]+):\s*([^\n]+)/g;
  let roomMatch;
  while ((roomMatch = roomLinePattern.exec(info)) !== null) {
    const room = canonicalRoomName(roomMatch[1]);
    const players = parsePlayersLine(roomMatch[2]);
    const occupantSet = new Set();
    players.forEach((rawName) => {
      const isDead = rawName.includes("(dead)");
      const cleanName = rawName.replace(/\s*\(dead\)\s*/g, "").trim();
      if (!cleanName) {
        return;
      }
      occupantSet.add(cleanName);
      const record = ensurePlayerRecord(cleanName);
      record.room = room;
      record.isDead = isDead;
      record.colorName = parseColorName(cleanName);
    });
    roomOccupants.set(room, occupantSet);
  }

  // Handle observed moves.
  const movePattern = /(Player \d+: [^\n]+?) MOVE from ([A-Za-z ]+) to ([A-Za-z ]+)/g;
  let moveMatch;
  while ((moveMatch = movePattern.exec(info)) !== null) {
    const playerName = moveMatch[1].trim();
    const destinationRoom = canonicalRoomName(moveMatch[3]);
    const record = ensurePlayerRecord(playerName);
    record.room = destinationRoom;
    record.colorName = parseColorName(playerName);
  }

  // Human location snapshot.
  const locationMatch = info.match(/Current Location:\s*([^\n]+)/);
  if (state.human_player_name && locationMatch) {
    const record = ensurePlayerRecord(state.human_player_name);
    record.room = canonicalRoomName(locationMatch[1]);
    record.colorName = parseColorName(state.human_player_name);
  }

  // Immediate leave/enter detection for currently visible room lists.
  roomOccupants.forEach((occupants, room) => {
    playerState.forEach((record) => {
      if (record.room === room && !occupants.has(record.name)) {
        record.room = "Unknown";
      }
    });
  });
}

function updateMap(previous, current) {
  const oldPositions = snapshotPositions();
  const previousPositionMap = previous ? buildPositionMapFromSnapshot(previous) : new Map();
  const currentPositionMap = buildPositionMapFromSnapshot(current);

  playerState.forEach((record) => {
    record.isHuman = false;
  });
  if (current.human_player_name) {
    ensurePlayerRecord(current.human_player_name).isHuman = true;
  }
  if (current.current_player) {
    ensurePlayerRecord(current.current_player);
  }

  if (currentPositionMap.size > 0) {
    currentPositionMap.forEach((position, playerName) => {
      const record = ensurePlayerRecord(playerName);
      record.room = position.room;
      record.colorName = position.colorName;
      record.displayName = position.displayName || playerName;
      record.isVisible = Boolean(position.isVisible);
      record.isDead = !position.isAlive;
    });
  } else {
    // Fallback for older snapshots with no explicit positions.
    extractKnownPositionsFromInfo(current);
  }

  const movedPlayers = new Set();
  if (currentPositionMap.size > 0) {
    currentPositionMap.forEach((position, playerName) => {
      const oldRoom = previousPositionMap.has(playerName)
        ? previousPositionMap.get(playerName).room
        : oldPositions[playerName] || "Unknown";
      if (oldRoom !== position.room) {
        movedPlayers.add(playerName);
      }
    });
  } else {
    Object.keys(oldPositions).forEach((playerName) => {
      const record = playerState.get(playerName);
      if (!record) {
        return;
      }
      if (oldPositions[playerName] !== canonicalRoomName(record.room)) {
        movedPlayers.add(playerName);
      }
    });
  }

  playerState.forEach((record, playerName) => {
    const newRoom = canonicalRoomName(record.room);
    if (movedPlayers.has(playerName) || !playerTokenEls.has(playerName) || !previous) {
      moveTokenToRoom(playerName, newRoom);
    }
    styleToken(ensureTokenEl(playerName), record);
  });

  // Room highlighting for current player.
  if (lastActiveRoom) {
    const prevRoomContainer = roomPlayerContainers.get(lastActiveRoom);
    if (prevRoomContainer) {
      prevRoomContainer.parentElement.classList.remove("active");
    }
  }
  let activeRoom = null;
  if (current.human_player_name && playerState.has(current.human_player_name)) {
    activeRoom = canonicalRoomName(playerState.get(current.human_player_name).room);
    const currentContainer = roomPlayerContainers.get(activeRoom);
    if (currentContainer) {
      currentContainer.parentElement.classList.add("active");
    }
  }
  lastActiveRoom = activeRoom;
}

function isHumanPlayerDead(current) {
  const humanName = current.human_player_name;
  if (!humanName) return false;
  const positions = Array.isArray(current.player_positions) ? current.player_positions : [];
  const entry = positions.find((p) => p.name === humanName);
  return entry ? entry.is_alive === false : false;
}

function renderActionButtons(current) {
  const humanDead = isHumanPlayerDead(current);
  if (humanDead) {
    clearActionButtons();
    hideSpeechInput();
    waitingMessageEl.textContent =
      "You have been eliminated. Please keep this tab open until the game finishes.";
    waitingMessageEl.classList.add("eliminated-msg");
    lastIsHumanTurn = false;
    lastActionSignature = "";
    return;
  }
  waitingMessageEl.classList.remove("eliminated-msg");

  const actions = Array.isArray(current.available_actions) ? current.available_actions : [];
  const actionSignature = buildActionSignature(actions);
  const turnChanged = lastIsHumanTurn !== current.is_human_turn;
  const actionListChanged = lastActionSignature !== actionSignature;
  const shouldRerender = turnChanged || actionListChanged;

  if (!shouldRerender) {
    if (!current.is_human_turn) {
      waitingMessageEl.textContent = "Waiting for your turn...";
    } else if (speechBox.classList.contains("hidden")) {
      waitingMessageEl.textContent = "Your turn. Choose an action.";
    }
    setActionButtonsEnabled(current.is_human_turn && !actionSubmitInFlight);
    return;
  }

  const prevSelectedActionName = selectedAction ? selectedAction.name : null;
  const monitorWasOpen = selectedAction && !monitorBox.classList.contains("hidden");
  const speechWasOpen = selectedAction && !speechBox.classList.contains("hidden");

  clearActionButtons();
  hideSpeechInput();

  if (!current.is_human_turn) {
    waitingMessageEl.textContent = "Waiting for your turn...";
    lastIsHumanTurn = current.is_human_turn;
    lastActionSignature = actionSignature;
    return;
  }

  waitingMessageEl.textContent = "Your turn. Choose an action.";
  if (actions.length === 0) {
    waitingMessageEl.textContent = "Your turn, but no available actions were provided.";
    lastIsHumanTurn = current.is_human_turn;
    lastActionSignature = actionSignature;
    return;
  }

  function openMonitorBox(action) {
    selectedAction = action;
    speechBox.classList.add("hidden");
    monitorBox.classList.remove("hidden");
    monitorRoomSelect.innerHTML = "";
    const opts = Array.isArray(action.location_options) && action.location_options.length > 0
      ? action.location_options
      : ROOM_ORDER.filter((r) => r !== "Unknown");
    opts.forEach((room) => {
      const option = document.createElement("option");
      option.value = room;
      option.textContent = room;
      monitorRoomSelect.appendChild(option);
    });
    waitingMessageEl.textContent = "Select a room to monitor, then submit.";
    if (monitorRoomSelect.options.length > 0) {
      monitorRoomSelect.selectedIndex = 0;
    }
    monitorRoomSelect.focus();
  }

  actions.forEach((action) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = action.name;
    button.disabled = actionSubmitInFlight || !current.is_human_turn;
    button.addEventListener("click", async () => {
      if (actionSubmitInFlight) {
        return;
      }
      if (action.requires_message) {
        selectedAction = action;
        speechBox.classList.remove("hidden");
        monitorBox.classList.add("hidden");
        waitingMessageEl.textContent = "Enter speech text, then submit.";
        speechTextInput.focus();
      } else if (action.requires_location) {
        openMonitorBox(action);
      } else {
        await submitAction(action.index, "", "");
      }
    });
    actionsEl.appendChild(button);
  });

  // Restore monitor/speech box if the same action is still available after a re-render.
  if (prevSelectedActionName && current.is_human_turn) {
    const match = actions.find((a) => a.name === prevSelectedActionName);
    if (match) {
      if (match.requires_location && monitorWasOpen) {
        openMonitorBox(match);
      } else if (match.requires_message && speechWasOpen) {
        selectedAction = match;
        speechBox.classList.remove("hidden");
        monitorBox.classList.add("hidden");
        waitingMessageEl.textContent = "Enter speech text, then submit.";
      }
    }
  }

  lastIsHumanTurn = current.is_human_turn;
  lastActionSignature = actionSignature;
}

function updateSidebar(previous, current) {
  renderActionButtons(current);
}

function updateStatus(previous, current) {
  gameIdEl.textContent = String(gameId ?? "-");
  statusEl.textContent = String(current.status ?? "-");
  phaseEl.textContent = String(current.current_phase ?? "-");
  timestepEl.textContent = String(current.timestep ?? "-");
  currentPlayerEl.textContent = current.is_human_turn ? "No (your turn now)" : "Yes";
  turnStateEl.textContent = current.is_human_turn ? "Human turn" : "Waiting";
}

function updateRoleBanner(current) {
  const role = String(current.human_player_identity || "").trim();
  if (!role) {
    roleBannerEl.classList.add("hidden");
    roleBannerEl.classList.remove("crewmate", "impostor");
    roleBannerEl.textContent = "";
    return;
  }
  const roleLower = role.toLowerCase();
  roleBannerEl.classList.remove("hidden", "crewmate", "impostor");
  if (roleLower.includes("impostor")) {
    roleBannerEl.classList.add("impostor");
    roleBannerEl.textContent = `Role: IMPOSTOR`;
  } else {
    roleBannerEl.classList.add("crewmate");
    roleBannerEl.textContent = `Role: CREWMATE`;
  }
}

function updateWinnerBanner(current) {
  const finished = String(current.status || "") !== "running";
  if (!finished || current.winner == null) {
    winnerBannerEl.classList.add("hidden");
    winnerBannerEl.classList.remove("shown");
    winnerBannerEl.textContent = "";
    return;
  }
  const winnerText = String(current.winner_reason || `Winner: ${current.winner}`);
  winnerBannerEl.textContent = `Game Over - ${winnerText}`;
  winnerBannerEl.classList.remove("hidden");
  winnerBannerEl.classList.add("shown");
}

function showGameOverModal(current) {
  if (!gameOverModalEl || gameOverModalShown) {
    return;
  }
  const winnerText = String(current.winner_reason || `Winner: ${current.winner}`);
  gameOverReasonEl.textContent = winnerText;
  gameOverModalEl.classList.remove("hidden");
  gameOverModalShown = true;
}

function extractNewLogLines(previousInfo, currentInfo) {
  const prevLines = new Set(
    String(previousInfo || "")
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0)
  );
  return String(currentInfo || "")
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0 && !prevLines.has(line));
}

function parseVisibleEvent(line) {
  const value = String(line || "").trim();
  if (!value) {
    return null;
  }
  const normalized = value.replace(/^\d+\.\s*/, "").trim();

  if (/^Timestep\s+\d+:/i.test(normalized)) {
    const timed = normalized.match(/^Timestep\s+(\d+):\s*\[([^\]]+)\]\s*(.*)$/i);
    if (timed) {
      const playerLabel = extractPlayerLabelFromText(timed[3] || "");
      return {
        key: `timestep|${timed[1]}|${timed[2]}|${timed[3]}`,
        meta: `T${timed[1]} | ${timed[2]}`,
        text: timed[3] || "(no detail)",
        tintHex: playerLabel ? colorHexFromPlayerName(playerLabel) : null,
      };
    }
    return {
      key: `timestep|${value}`,
      meta: "Event",
      text: value,
      tintHex: null,
    };
  }

  if (/^(Current Location|Players in|Dead players):/i.test(value)) {
    const label = value.split(":")[0];
    const body = value.slice(label.length + 1).trim();
    return {
      key: `snapshot|${value}`,
      meta: "Current View",
      text: `${label}: ${body}`,
      tintHex: null,
    };
  }

  return null;
}

function appendVisibleLogEntry(entry) {
  if (!entry || seenVisibleLogKeys.has(entry.key)) {
    return false;
  }
  seenVisibleLogKeys.add(entry.key);

  if (latestLogEl.textContent.trim() === "No visible events yet.") {
    latestLogEl.textContent = "";
  }

  const item = document.createElement("div");
  item.className = "log-entry";
  if (entry.tintHex) {
    item.classList.add("tinted");
    item.style.setProperty("--player-tint-bg", hexToRgba(entry.tintHex, 0.14));
    item.style.setProperty("--player-tint-border", hexToRgba(entry.tintHex, 0.42));
  }

  const meta = document.createElement("div");
  meta.className = "log-meta";
  meta.textContent = entry.meta;

  const text = document.createElement("div");
  text.className = "log-text";
  text.textContent = entry.text;

  item.appendChild(meta);
  item.appendChild(text);
  latestLogEl.appendChild(item);
  return true;
}

function extractSection(text, header, nextHeader) {
  const source = String(text || "");
  const start = source.indexOf(header);
  if (start < 0) {
    return "";
  }
  const startIdx = start + header.length;
  const rest = source.slice(startIdx);
  const end = rest.indexOf(nextHeader);
  const content = end >= 0 ? rest.slice(0, end) : rest;
  return content.trim();
}

function extractMissionBrief(playerInfo, humanIdentity, impostorTeammates = []) {
  const locationSection = extractSection(playerInfo, "Current Location:", "Observation history:");
  const tasksSection = extractSection(playerInfo, "Your Assigned Tasks:", "Available actions:");
  if (!tasksSection) {
    return "";
  }

  const lines = [];
  if (humanIdentity) {
    lines.push(`Role: ${humanIdentity}`);
    if (String(humanIdentity).toLowerCase() === "impostor") {
      const teammates = Array.isArray(impostorTeammates)
        ? impostorTeammates.filter((name) => String(name || "").trim().length > 0)
        : [];
      lines.push(`Impostor Teammates: ${teammates.length > 0 ? teammates.join(", ") : "None"}`);
    }
    lines.push("");
  }
  lines.push("Your Assigned Tasks and Suggested Paths:");
  lines.push(tasksSection);
  return lines.join("\n").trim();
}

function captureMissionBrief(current) {
  if (missionBriefCaptured) {
    return;
  }
  const brief = extractMissionBrief(
    current.player_info || "",
    current.human_player_identity || "",
    current.human_impostor_teammates || []
  );
  if (!brief) {
    return;
  }
  missionBriefEl.textContent = brief;
  if (missionBriefPanelEl) {
    missionBriefPanelEl.classList.remove("hidden");
  }
  missionBriefCaptured = true;
}

function updateLog(previous, current) {
  const currentInfo = current.player_info || "";
  if (!currentInfo) {
    return;
  }
  captureMissionBrief(current);

  const prevInfo = previous ? previous.player_info || "" : "";
  const newLines = previous ? extractNewLogLines(prevInfo, currentInfo) : currentInfo.split("\n");
  const visibleEvents = newLines.map(parseVisibleEvent).filter(Boolean);
  const wasNearBottom =
    latestLogEl.scrollHeight - latestLogEl.scrollTop - latestLogEl.clientHeight < 24;
  let appendedCount = 0;
  visibleEvents.forEach((event) => {
    if (appendVisibleLogEntry(event)) {
      appendedCount += 1;
    }
  });
  if (appendedCount > 0 && wasNearBottom) {
    latestLogEl.scrollTop = latestLogEl.scrollHeight;
  }
}

function parseMeetingMessages(playerInfo) {
  const lines = String(playerInfo || "").split("\n");
  const messages = [];
  const pattern =
    /(?:Timestep\s+(\d+):\s*)?\[(meeting[^\]]*)\]\s*(Player\s+\d+:\s*[^\s]+)\s+SPEAK\s*:?\s*(.*)$/i;
  lines.forEach((line) => {
    const text = line.trim();
    const match = text.match(pattern);
    if (!match) {
      return;
    }
    messages.push({
      timestep: match[1] || "",
      phase: match[2] || "meeting",
      player: match[3].trim(),
      text: normalizeMeetingText(match[4].trim() || "..."),
    });
  });
  return messages;
}

function normalizeMeetingText(text) {
  let normalized = String(text || "").trim();
  if (!normalized) {
    return "...";
  }

  const actionMarkers = [...normalized.matchAll(/\[Action\]/gi)];
  if (actionMarkers.length > 0) {
    const last = actionMarkers[actionMarkers.length - 1];
    normalized = normalized.slice(last.index + last[0].length).trim();
  }

  normalized = normalized.replace(/^\s*\**\s*SPEAK\s*\**\s*:?\s*/i, "");
  normalized = normalized.replace(/^\*+\s*:?\s*/, "");
  normalized = normalized.replace(/\[(Condensed Memory|Thinking Process|Action)\]/gi, "");
  normalized = normalized.replace(/\bFINAL_SPEAK_MESSAGE\s*:\s*/gi, "");
  normalized = normalized.replace(/\bFINAL_ACTION_INDEX\s*:\s*\d+\b/gi, "");
  normalized = normalized.split(/\bFINAL_[A-Z_]+\s*:/i)[0].trim();

  const leakMarker = normalized.search(/\n\s*\[(Reasoning|Thinking Process)\]\s*/i);
  if (leakMarker >= 0) {
    normalized = normalized.slice(0, leakMarker).trim();
  }

  if (normalized.startsWith('"') && normalized.endsWith('"') && normalized.length >= 2) {
    normalized = normalized.slice(1, -1).trim();
  }
  if (normalized.startsWith("'") && normalized.endsWith("'") && normalized.length >= 2) {
    normalized = normalized.slice(1, -1).trim();
  }
  return normalized || "...";
}

function appendMeetingMessage(message, isHuman) {
  const group = document.createElement("div");
  const isSystem = Boolean(message.system) || message.player === "SYSTEM";
  const isTeammate = !isSystem && knownImpostorTeammates.has(message.player);
  group.className = `meeting-group${isHuman ? " human" : ""}${isSystem ? " system" : ""}${isTeammate ? " impostor-teammate-msg" : ""}`;
  if (!isSystem) {
    if (isTeammate) {
      group.classList.add("tinted");
      group.style.setProperty("--player-tint-bg", "rgba(200,40,40,0.18)");
      group.style.setProperty("--player-tint-border", "rgba(220,60,60,0.65)");
    } else {
      const speakerColor = colorHexFromPlayerName(message.player);
      group.classList.add("tinted");
      group.style.setProperty("--player-tint-bg", hexToRgba(speakerColor, 0.2));
      group.style.setProperty("--player-tint-border", hexToRgba(speakerColor, 0.62));
    }
  }

  const speaker = document.createElement("div");
  speaker.className = "speaker";
  speaker.textContent = isTeammate ? `${message.player} [IMPOSTOR]` : message.player;
  group.appendChild(speaker);

  const msgEl = document.createElement("div");
  msgEl.className = "meeting-msg";
  msgEl.textContent = message.timestep
    ? `T${message.timestep}: ${message.text}`
    : message.text;
  group.appendChild(msgEl);
  group.classList.add("new-msg");
  window.setTimeout(() => group.classList.remove("new-msg"), 380);
  meetingFeed.appendChild(group);
}

function updateMeetingPanel(previous, current) {
  const meeting = isMeetingPhase(current.current_phase);
  const wasMeeting = isMeetingPhase(previous ? previous.current_phase : null);
  meetingPanel.classList.toggle("hidden", !meeting);
  if (!meeting) {
    meetingPanel.classList.remove("revealed");
    return;
  }
  if (!wasMeeting) {
    meetingPanel.classList.add("revealed");
    playSfx("meeting-start");
  }

  let messages = [];
  if (Array.isArray(current.meeting_messages) && current.meeting_messages.length > 0) {
    messages = current.meeting_messages.map((entry) => ({
      id: entry.id ?? null,
      timestep: entry.timestep ?? "",
      round: entry.round ?? "",
      phase: "meeting",
      player: entry.player,
      text: normalizeMeetingText(String(entry.text || "").trim() || "..."),
      system: Boolean(entry.system),
    }));
  } else {
    messages = parseMeetingMessages(current.player_info || "");
  }
  const wasNearBottom =
    meetingFeed.scrollHeight - meetingFeed.scrollTop - meetingFeed.clientHeight < 24;
  let appendedCount = 0;
  messages.forEach((message) => {
    const key =
      message.id ||
      `${message.timestep}|${message.round ?? ""}|${message.player}|${message.text}`;
    if (seenMeetingMessages.has(key)) {
      return;
    }
    seenMeetingMessages.add(key);
    appendMeetingMessage(message, message.player === current.human_player_name);
    playSfx("message");
    appendedCount += 1;
  });

  if (appendedCount > 0 && wasNearBottom) {
    meetingFeed.scrollTop = meetingFeed.scrollHeight;
  }
}

function parseTaskEventLine(line, current) {
  const normalized = String(line || "").replace(/^\d+\.\s*/, "").trim();
  const observerPattern = /Timestep\s+(\d+):\s*\[task\]\s*(Player\s+\d+:\s*[^\s]+)\s+Seemingly doing task/i;
  const selfPattern = /Timestep\s+(\d+):\s*\[task phase\]\s*Seemingly doing task/i;

  const observerMatch = normalized.match(observerPattern);
  if (observerMatch) {
    return {
      key: `${observerMatch[1]}|${observerMatch[2]}|seemingly-doing-task`,
      text: `T${observerMatch[1]}: ${observerMatch[2]} seemingly doing task`,
      tintHex: colorHexFromPlayerName(observerMatch[2]),
    };
  }

  const selfMatch = normalized.match(selfPattern);
  if (selfMatch) {
    const playerName = current.current_player || "Unknown player";
    return {
      key: `${selfMatch[1]}|${playerName}|seemingly-doing-task`,
      text: `T${selfMatch[1]}: ${playerName} seemingly doing task`,
      tintHex: colorHexFromPlayerName(playerName),
    };
  }

  return null;
}

function appendTaskEvent(text, tintHex = null) {
  if (taskFeed.textContent.trim() === "No task actions yet.") {
    taskFeed.textContent = "";
  }
  const item = document.createElement("div");
  item.className = "task-event";
  if (tintHex) {
    item.classList.add("tinted");
    item.style.setProperty("--player-tint-bg", hexToRgba(tintHex, 0.14));
    item.style.setProperty("--player-tint-border", hexToRgba(tintHex, 0.42));
  }
  item.textContent = text;
  taskFeed.appendChild(item);
}

function updateTaskFeed(previous, current) {
  const currentInfo = current.player_info || "";
  if (!currentInfo) {
    return;
  }
  const prevInfo = previous ? previous.player_info || "" : "";
  const newLines = previous ? extractNewLogLines(prevInfo, currentInfo) : currentInfo.split("\n");
  newLines.forEach((rawLine) => {
    const line = String(rawLine || "").trim();
    const normalized = line.replace(/^\d+\.\s*/, "").trim();
    if (!normalized.includes("Seemingly doing task")) {
      return;
    }
    const event = parseTaskEventLine(normalized, current);
    if (!event || seenTaskEvents.has(event.key)) {
      return;
    }
    seenTaskEvents.add(event.key);
    appendTaskEvent(event.text, event.tintHex || null);
  });
  if (taskFeed.children.length > 0) {
    taskFeed.scrollTop = taskFeed.scrollHeight;
  }
}

function applyProgressBar(fillEl, textEl, progress) {
  if (!fillEl || !textEl) {
    return;
  }
  const totalRaw = Number(progress && progress.total);
  const completedRaw = Number(progress && progress.completed);
  const ratioRaw = Number(progress && progress.ratio);
  const total = Number.isFinite(totalRaw) && totalRaw >= 0 ? Math.floor(totalRaw) : 0;
  const completed = Number.isFinite(completedRaw) && completedRaw >= 0
    ? Math.min(total, Math.floor(completedRaw))
    : 0;
  const remainingRaw = Number(progress && progress.remaining);
  const remaining = Number.isFinite(remainingRaw) && remainingRaw >= 0
    ? Math.min(total, Math.floor(remainingRaw))
    : Math.max(0, total - completed);
  const ratio = Number.isFinite(ratioRaw)
    ? Math.max(0, Math.min(1, ratioRaw))
    : (total > 0 ? completed / total : 0);

  fillEl.style.width = `${(ratio * 100).toFixed(1)}%`;
  textEl.textContent = `${remaining} left (${completed}/${total})`;
}

function updateTaskProgress(current) {
  const progress = (current && current.task_progress) || {};
  const isImpostor = String(current && current.human_player_identity || "").toLowerCase().includes("impostor");
  applyProgressBar(humanTaskProgressFillEl, humanTaskProgressTextEl, progress.human || {});
  if (humanTaskBarGroupEl) humanTaskBarGroupEl.classList.toggle("hidden", isImpostor);
  if (globalTaskBarGroupEl) globalTaskBarGroupEl.classList.toggle("hidden", !isImpostor);
  if (isImpostor) {
    applyProgressBar(globalTaskProgressFillEl, globalTaskProgressTextEl, progress.global || {});
  }
}

function resetTaskProgress() {
  if (humanTaskProgressFillEl) {
    humanTaskProgressFillEl.style.width = "0%";
  }
  if (humanTaskProgressTextEl) {
    humanTaskProgressTextEl.textContent = "-";
  }
  if (humanTasksListEl) {
    humanTasksListEl.textContent = "-";
  }
  if (impostorPanelEl) {
    impostorPanelEl.classList.add("hidden");
  }
  if (killCooldownValueEl) {
    killCooldownValueEl.textContent = "-";
    killCooldownValueEl.classList.remove("ready");
  }
}

const seenMonitorResults = new Set();

function parseMonitorPlayers(raw) {
  const names = [];
  const re = /\(([^)]+)\):/g;
  let m;
  while ((m = re.exec(raw)) !== null) {
    const name = m[1].trim().replace(/^Player\s+\d+:\s*/i, "");
    if (name) names.push(name);
  }
  return names;
}

function updateMonitorResults(current) {
  if (!monitorFeedEl) return;
  const results = Array.isArray(current.monitor_results) ? current.monitor_results : [];
  if (results.length === 0) return;
  let addedNew = false;
  results.forEach((raw) => {
    if (seenMonitorResults.has(raw)) return;
    seenMonitorResults.add(raw);
    addedNew = true;
    if (monitorFeedEl.textContent.includes("No monitor results")) {
      monitorFeedEl.textContent = "";
    }
    const card = document.createElement("div");
    card.className = "monitor-result";

    const loc = raw.match(/Location:\s*([^,}]+)/i);
    const hdr = document.createElement("div");
    hdr.className = "monitor-result-header";
    hdr.textContent = loc ? `Camera: ${loc[1].trim()}` : "Monitor Result";
    card.appendChild(hdr);

    const body = document.createElement("div");
    body.className = "monitor-result-body";
    const players = parseMonitorPlayers(raw);
    body.textContent = players.length > 0 ? players.join(", ") : "No one observed.";
    card.appendChild(body);
    monitorFeedEl.appendChild(card);
  });
  if (addedNew) {
    setActiveMapTab("monitor");
    monitorFeedEl.scrollTop = monitorFeedEl.scrollHeight;
  }
}

function updateHumanTasks(current) {
  if (!humanTasksListEl) return;
  const isImpostor = String(current && current.human_player_identity || "").toLowerCase().includes("impostor");
  const panel = document.getElementById("crewmate-tasks-panel");
  if (panel) panel.classList.toggle("hidden", isImpostor);
  const tasks = current.human_tasks;
  if (!tasks || !tasks.length) {
    humanTasksListEl.textContent = "-";
    return;
  }
  let html = "";
  for (let i = 0; i < tasks.length; i++) {
    const t = tasks[i];
    const doneCls = t.is_done ? " done" : "";
    const typeBadge = t.task_type ? ` [${t.task_type}]` : "";
    html += `<div class="human-task-row${doneCls}">`;
    html += `<span class="human-task-name">${t.name}${typeBadge}</span>`;
    html += `<span class="human-task-loc">${t.location}</span>`;
    html += `</div>`;
  }
  humanTasksListEl.innerHTML = html;
}

function updateCrewmateTasks(current) {
  if (!crewmateTasksListEl) {
    return;
  }
  const tasks = current.crewmate_tasks;
  if (!tasks || !tasks.length) {
    crewmateTasksListEl.textContent = "-";
    return;
  }
  let html = "";
  for (let i = 0; i < tasks.length; i++) {
    const entry = tasks[i];
    const rawName = String(entry.name || "");
    const shortName = rawName.replace(/^Player\s+\d+:\s*/i, "");
    const label = entry.is_self ? shortName + " (you)" : shortName;
    const remaining = Number(entry.remaining) || 0;
    const total = Number(entry.total) || 0;
    const countText = remaining === 0 ? "Done" : remaining + "/" + total + " left";
    const selfCls = entry.is_self ? " is-self" : "";
    const doneCls = remaining === 0 ? " done" : "";
    const safeName = rawName.replace(/"/g, "&quot;");
    html += `<div class="crewmate-task-row${selfCls}">`;
    html += `<span class="crewmate-task-name" title="${safeName}">${label}</span>`;
    html += `<span class="crewmate-task-count${doneCls}">${countText}</span>`;
    html += `</div>`;
  }
  crewmateTasksListEl.innerHTML = html;
}

function updateKillCooldown(current) {
  if (!impostorPanelEl || !killCooldownValueEl) {
    return;
  }
  const identity = String(current.human_player_identity || "").toLowerCase();
  if (!identity.includes("impostor")) {
    impostorPanelEl.classList.add("hidden");
    return;
  }
  impostorPanelEl.classList.remove("hidden");
  const cooldown = current.kill_cooldown;
  if (cooldown === null || cooldown === undefined) {
    killCooldownValueEl.textContent = "-";
    killCooldownValueEl.classList.remove("ready");
    return;
  }
  const val = Number(cooldown);
  if (val === 0) {
    killCooldownValueEl.textContent = "READY";
    killCooldownValueEl.classList.add("ready");
  } else {
    killCooldownValueEl.textContent = `${val} turn${val === 1 ? "" : "s"}`;
    killCooldownValueEl.classList.remove("ready");
  }
}

function renderState(current) {
  gameView.classList.remove("hidden");
  logView.classList.remove("hidden");
  const becameHumanTurn =
    Boolean(current && current.is_human_turn) &&
    !(previousState && previousState.is_human_turn);

  // Track impostor teammates for red highlighting
  const teammates = Array.isArray(current.human_impostor_teammates)
    ? current.human_impostor_teammates
    : [];
  teammates.forEach((name) => {
    if (name) knownImpostorTeammates.add(String(name));
  });

  updateMap(previousState, current);
  updateStatus(previousState, current);
  updateRoleBanner(current);
  updateWinnerBanner(current);
  updateSidebar(previousState, current);
  updateMeetingPanel(previousState, current);
  updateTaskProgress(current);
  updateMonitorResults(current);
  updateHumanTasks(current);
  updateKillCooldown(current);
  updateTaskFeed(previousState, current);
  updateLog(previousState, current);
  if (becameHumanTurn && String(current.status || "") === "running") {
    playSfx("human-turn");
  }

  if (current.status !== "running") {
    const wasRunning = previousState && String(previousState.status || "") === "running";
    if (wasRunning) {
      playSfx("game-over");
      showGameOverModal(current);
    } else if (!previousState && current.winner != null) {
      showGameOverModal(current);
    }
    stopPolling();
    hideSpeechInput();
    setActionButtonsEnabled(false);
    setGameActive(false);
    waitingMessageEl.textContent = "Game finished.";
    appendStatusLine(
      `Game ${gameId} finished. Winner: ${current.winner ?? "n/a"} | Reason: ${current.winner_reason ?? "n/a"}`
    );
  }
}

async function createGame() {
  clearError();
  createGameBtn.disabled = true;
  appendStatusLine("Creating game...");

  try {
    const selectedRole = roleSelectEl ? roleSelectEl.value : "random";
    const response = await fetchJson(apiUrl("/create_game"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ human_role: selectedRole === "random" ? null : selectedRole }),
    });

    if (typeof response.game_id !== "number") {
      throw new Error("Invalid /create_game response: missing game_id");
    }

    gameId = response.game_id;
    setGameActive(true);
    const confirmedRole = String(response.human_role || "random");
    if (roleSelectEl) roleSelectEl.value = confirmedRole === "random" ? "random" : confirmedRole;
    previousState = null;
    consecutivePollErrors = 0;
    previousPhase = null;
    winnerBannerEl.classList.add("hidden");
    winnerBannerEl.textContent = "";
    roleBannerEl.classList.add("hidden");
    roleBannerEl.classList.remove("crewmate", "impostor");
    roleBannerEl.textContent = "";
    gameOverModalShown = false;
    if (gameOverModalEl) gameOverModalEl.classList.add("hidden");
    knownImpostorTeammates.clear();
    seenMeetingMessages.clear();
    seenVisibleLogKeys.clear();
    hiddenAliasByName.clear();
    hiddenAliasCounter = 1;
    playerState.clear();
    playerTokenEls.clear();
    lastActiveRoom = null;
    initMapSkeleton();
    meetingFeed.innerHTML = "";
    seenTaskEvents.clear();
    seenMonitorResults.clear();
    taskFeed.textContent = "No task actions yet.";
    if (monitorFeedEl) monitorFeedEl.textContent = "No monitor results yet. Go to Security and use VIEW MONITOR.";
    resetTaskProgress();
    latestLogEl.textContent = "No visible events yet.";
    missionBriefEl.textContent = "No mission briefing captured yet.";
    missionBriefCaptured = false;
    if (missionBriefPanelEl) {
      missionBriefPanelEl.classList.add("hidden");
    }
    appendStatusLine(`Game created (game_id=${gameId}). Polling state...`);
    startPollingNow();
  } catch (error) {
    showError(`Create game failed: ${error.message}`);
    appendStatusLine("Create game failed.");
    stopPolling();
  }
}

async function pollGameState() {
  if (gameId === null || pollInFlight) {
    if (gameId !== null && !pollInFlight) {
      scheduleNextPoll(1000);
    }
    return;
  }

  pollInFlight = true;
  try {
    clearError();
    const current = await fetchJson(apiUrl(`/game_state?game_id=${encodeURIComponent(gameId)}`));
    renderState(current);
    previousState = current;
    consecutivePollErrors = 0;
    scheduleNextPoll(1000);
  } catch (error) {
    consecutivePollErrors += 1;
    showError(`Polling failed: ${error.message}`);
    const retryDelayMs = Math.min(15000, 1000 * Math.pow(2, Math.min(consecutivePollErrors - 1, 4)));
    appendStatusLine(
      `Polling failed (attempt ${consecutivePollErrors}). Retrying in ${Math.round(retryDelayMs / 1000)}s...`
    );
    scheduleNextPoll(retryDelayMs);
  } finally {
    pollInFlight = false;
  }
}

async function submitAction(actionIndex, speechText = "", monitorRoom = "") {
  if (gameId === null) {
    showError("No active game. Create a game first.");
    return;
  }
  if (actionSubmitInFlight) {
    return;
  }

  actionSubmitInFlight = true;
  clearError();
  setActionButtonsEnabled(false);

  // Keep existing behavior unchanged: clear actions immediately after submit.
  clearActionButtons();
  hideSpeechInput();
  lastIsHumanTurn = null;
  lastActionSignature = "";
  waitingMessageEl.textContent = "Submitting action...";

  try {
    await fetchJson(apiUrl("/human_action"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        game_id: gameId,
        action_index: actionIndex,
        speech_text: speechText,
        monitor_room: monitorRoom,
      }),
    });
    appendStatusLine(`Action ${actionIndex} submitted. Waiting for next state...`);
    waitingMessageEl.textContent = "Waiting for your turn...";
    startPollingNow();
  } catch (error) {
    showError(`Submit action failed: ${error.message}`);
    appendStatusLine("Action submission failed.");
  } finally {
    actionSubmitInFlight = false;
  }
}

if (gameOverCloseBtn) {
  gameOverCloseBtn.addEventListener("click", () => {
    if (gameOverModalEl) gameOverModalEl.classList.add("hidden");
  });
}
if (gameOverModalEl) {
  gameOverModalEl.querySelector(".game-over-backdrop")?.addEventListener("click", () => {
    gameOverModalEl.classList.add("hidden");
  });
}

createGameBtn.addEventListener("click", createGame);
if (soundToggleBtn) {
  soundToggleBtn.addEventListener("click", () => {
    audioState.enabled = !audioState.enabled;
    window.localStorage.setItem(SOUND_ENABLED_KEY, String(audioState.enabled));
    soundToggleBtn.textContent = `Sound: ${audioState.enabled ? "On" : "Off"}`;
    if (audioState.enabled) {
      playSfx("message");
    }
  });
}
if (soundVolumeEl) {
  soundVolumeEl.addEventListener("input", () => {
    const value = Number(soundVolumeEl.value || 0);
    audioState.volume = Math.max(0, Math.min(1, value / 100));
    window.localStorage.setItem(SOUND_VOLUME_KEY, String(audioState.volume));
  });
}
tabLiveLogsBtn.addEventListener("click", () => setActiveJournalTab("live-logs"));
tabNotesBtn.addEventListener("click", () => setActiveJournalTab("notes"));
if (tabMapBriefBtn) tabMapBriefBtn.addEventListener("click", () => setActiveMapTab("brief"));
if (tabMapMonitorBtn) tabMapMonitorBtn.addEventListener("click", () => setActiveMapTab("monitor"));
notesBoxEl.addEventListener("input", () => {
  window.localStorage.setItem(PLAYER_NOTES_STORAGE_KEY, notesBoxEl.value);
});

submitSpeechBtn.addEventListener("click", async () => {
  clearError();
  if (!selectedAction) {
    showError("Select an action that requires a message first.");
    return;
  }

  const speechText = speechTextInput.value.trim();
  if (speechText.length === 0) {
    showError("Speech text is required for this action.");
    return;
  }

  await submitAction(selectedAction.index, speechText);
});

submitMonitorBtn.addEventListener("click", async () => {
  clearError();
  if (!selectedAction) {
    showError("Select VIEW MONITOR first.");
    return;
  }
  if (!selectedAction.requires_location) {
    showError("Selected action does not require a room.");
    return;
  }
  const monitorRoom = (monitorRoomSelect.value || "").trim();
  if (!monitorRoom) {
    showError("Select a room to monitor.");
    return;
  }
  await submitAction(selectedAction.index, "", monitorRoom);
});

initMapSkeleton();
resetTaskProgress();
setActiveJournalTab("live-logs");
loadNotes();
loadSoundPrefs();
