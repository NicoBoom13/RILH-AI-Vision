#!/usr/bin/env python3
"""
RILH Scoreboard — single-file Python server for OBS overlay + remote control.

Run:
    python3 scoreboard.py

Then in OBS:    Browser Source → http://localhost:7788/scoreboard
On your iPad:   open               http://<mac-ip>:7788/control

No dependencies, just Python 3.
"""

import http.server
import json
import mimetypes
import socket
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse


PORT = 7788
HERE = Path(__file__).resolve().parent
LOGO_PATH = HERE / "N2.png"      # défaut : N2 niveau

PRESET_LOGOS = {
    "elite": HERE / "elite.png",
    "n1":    HERE / "N1.png",
    "n2":    HERE / "N2.png",
    "n3":    HERE / "N3.png",
}

DEFAULT_PERIOD_SEC = 25 * 60
DEFAULT_OT_SEC     = 5 * 60
PENALTY_OPTIONS    = (120, 300, 600)
MAX_PENALTIES_PER_TEAM = 3

# Couleurs reprises du fond du logo N2
COLOR_PURPLE = "#3a2880"  # gauche du losange
COLOR_RED    = "#c8334d"  # droite du losange


# --- State -------------------------------------------------------------------

state_lock = threading.Lock()
state = {
    "teamA": {
        "name": "Team 1", "short": "TEAM 1",
        "score": 0,
        "color": COLOR_PURPLE, "textColor": "#ffffff",
        "bgMode": "gradient",   # 'gradient' | 'color' | 'none'
        "logo": "",
        "penalties": [],
    },
    "teamB": {
        "name": "Team 2", "short": "TEAM 2",
        "score": 0,
        "color": COLOR_RED, "textColor": "#ffffff",
        "bgMode": "gradient",
        "logo": "",
        "penalties": [],
    },
    "periodLabel": "P1",
    "periodDuration": DEFAULT_PERIOD_SEC,
    "otDuration":     DEFAULT_OT_SEC,
    "mainLogo": "",   # "" = N2-1.webp par défaut, sinon URL ou data: URL fournie par l'utilisateur
    "clock": {
        "running": False,
        "direction": "down",
        "lastValue": float(DEFAULT_PERIOD_SEC),
        "lastUpdate": time.time(),
        "limit": DEFAULT_PERIOD_SEC,
    },
    "visible": True,
    "rev": 0,
}

sse_clients = []
sse_lock = threading.Lock()


def bump_and_broadcast():
    state["rev"] += 1
    payload = ("data: " + json.dumps(state) + "\n\n").encode("utf-8")
    with sse_lock:
        dead = []
        for client in sse_clients:
            try:
                client.write(payload); client.flush()
            except Exception:
                dead.append(client)
        for d in dead:
            try: sse_clients.remove(d)
            except ValueError: pass


# --- Clock helpers -----------------------------------------------------------

def clock_now():
    c = state["clock"]
    if not c["running"]:
        return c["lastValue"]
    elapsed = time.time() - c["lastUpdate"]
    if c["direction"] == "up":
        return c["lastValue"] + elapsed
    return max(0.0, c["lastValue"] - elapsed)


def clock_freeze():
    state["clock"]["lastValue"] = clock_now()
    state["clock"]["lastUpdate"] = time.time()


# --- Penalty helpers ---------------------------------------------------------

def penalty_remaining(p):
    if not p["running"]:
        return p["lastValue"]
    elapsed = time.time() - p["lastUpdate"]
    return max(0.0, p["lastValue"] - elapsed)


def penalty_freeze(p):
    p["lastValue"] = penalty_remaining(p)
    p["lastUpdate"] = time.time()


def set_penalties_running(running):
    now = time.time()
    for team_key in ("teamA", "teamB"):
        for p in state[team_key]["penalties"]:
            if p["running"] != running:
                if running:
                    p["lastUpdate"] = now
                    p["running"] = True
                else:
                    penalty_freeze(p)
                    p["running"] = False


def prune_expired_penalties():
    changed = False
    for team_key in ("teamA", "teamB"):
        kept = []
        for p in state[team_key]["penalties"]:
            if penalty_remaining(p) > 0.05:
                kept.append(p)
            else:
                changed = True
        state[team_key]["penalties"] = kept
    return changed


def penalty_ticker():
    while True:
        time.sleep(0.5)
        with state_lock:
            if prune_expired_penalties():
                bump_and_broadcast()


# --- Period helpers ----------------------------------------------------------

def is_overtime_label(label):
    return label.upper().startswith("PROL")


def reset_clock_for_current_period():
    target = state["otDuration"] if is_overtime_label(state["periodLabel"]) else state["periodDuration"]
    c = state["clock"]
    c["running"] = False
    c["direction"] = "down"
    c["lastValue"] = float(target)
    c["lastUpdate"] = time.time()
    c["limit"] = int(target)
    set_penalties_running(False)


# --- Handler -----------------------------------------------------------------

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_args, **_kwargs): return

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/scoreboard"):
            return self._html(SCOREBOARD_HTML)
        if path == "/control":
            return self._html(CONTROL_HTML)
        if path == "/state":
            with state_lock:
                return self._json(state)
        if path == "/events":
            return self._sse()
        if path == "/logo":
            return self._serve_file(LOGO_PATH)
        if path.startswith("/preset-logo/"):
            name = path[len("/preset-logo/"):].lower()
            p = PRESET_LOGOS.get(name)
            if p is None:
                return self.send_error(404)
            return self._serve_file(p)
        return self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw or b"{}")
        except Exception:
            return self.send_error(400, "bad json")
        with state_lock:
            try:
                self._dispatch(path, data)
            except KeyError:
                return self.send_error(404)
            bump_and_broadcast()
        return self._json({"ok": True})

    def _dispatch(self, path, data):
        c = state["clock"]
        if path == "/score":
            key = "teamA" if data["team"] == "A" else "teamB"
            state[key]["score"] = max(0, state[key]["score"] + int(data.get("delta", 0)))
        elif path == "/score/set":
            key = "teamA" if data["team"] == "A" else "teamB"
            state[key]["score"] = max(0, int(data["value"]))
        elif path == "/team":
            key = "teamA" if data["team"] == "A" else "teamB"
            for f in ("name", "short", "color", "textColor"):
                if f in data:
                    state[key][f] = data[f]
            if "bgMode" in data and data["bgMode"] in ("gradient", "color", "none"):
                state[key]["bgMode"] = data["bgMode"]
        elif path == "/team/logo":
            key = "teamA" if data["team"] == "A" else "teamB"
            state[key]["logo"] = str(data.get("value", ""))
        elif path == "/main/logo":
            state["mainLogo"] = str(data.get("value", ""))
        elif path == "/period":
            label = str(data.get("label", state["periodLabel"])).upper()
            state["periodLabel"] = label
            if data.get("resetClock", True):
                reset_clock_for_current_period()
        elif path == "/period/duration":
            state["periodDuration"] = max(1, int(data.get("value", DEFAULT_PERIOD_SEC)))
            if not is_overtime_label(state["periodLabel"]) and not c["running"]:
                reset_clock_for_current_period()
        elif path == "/period/ot_duration":
            state["otDuration"] = max(1, int(data.get("value", DEFAULT_OT_SEC)))
            if is_overtime_label(state["periodLabel"]) and not c["running"]:
                reset_clock_for_current_period()
        elif path == "/visible":
            state["visible"] = bool(data.get("value", True))
        elif path == "/clock/start":
            if not c["running"]:
                clock_freeze()
                c["running"] = True
                c["lastUpdate"] = time.time()
                set_penalties_running(True)
        elif path == "/clock/pause":
            if c["running"]:
                clock_freeze()
                c["running"] = False
                set_penalties_running(False)
        elif path == "/clock/reset":
            if "seconds" in data:
                target = float(data["seconds"])
                c["running"] = False
                c["lastValue"] = max(0.0, target)
                c["lastUpdate"] = time.time()
                set_penalties_running(False)
            else:
                reset_clock_for_current_period()
        elif path == "/clock/direction":
            d = data.get("value", "up")
            if d not in ("up", "down"): raise KeyError
            clock_freeze()
            c["direction"] = d
        elif path == "/clock/adjust":
            clock_freeze()
            c["lastValue"] = max(0.0, c["lastValue"] + float(data.get("delta", 0)))
        elif path == "/penalty/add":
            key = "teamA" if data["team"] == "A" else "teamB"
            duration = int(data.get("duration", 120))
            if duration not in PENALTY_OPTIONS: duration = 120
            if len(state[key]["penalties"]) >= MAX_PENALTIES_PER_TEAM: raise KeyError
            running = bool(c["running"])
            state[key]["penalties"].append({
                "id": uuid.uuid4().hex[:8],
                "duration": duration,
                "lastValue": float(duration),
                "lastUpdate": time.time(),
                "running": running,
            })
        elif path == "/penalty/remove":
            key = "teamA" if data["team"] == "A" else "teamB"
            pid = str(data.get("id", ""))
            state[key]["penalties"] = [p for p in state[key]["penalties"] if p["id"] != pid]
        elif path == "/penalty/clear":
            key = "teamA" if data["team"] == "A" else "teamB"
            state[key]["penalties"] = []
        elif path == "/swap":
            state["teamA"], state["teamB"] = state["teamB"], state["teamA"]
        else:
            raise KeyError

    def _html(self, body):
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers(); self.wfile.write(encoded)

    def _json(self, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def _serve_file(self, p: Path):
        if not p.exists() or not p.is_file():
            return self.send_error(404)
        mime, _ = mimetypes.guess_type(p.name)
        if not mime and p.suffix.lower() == ".webp":
            mime = "image/webp"
        data = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Cache-Control", "public, max-age=3600")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)

    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        with state_lock:
            initial = ("data: " + json.dumps(state) + "\n\n").encode("utf-8")
        try:
            self.wfile.write(initial); self.wfile.flush()
        except Exception:
            return
        with sse_lock: sse_clients.append(self.wfile)
        try:
            while True:
                time.sleep(15)
                try:
                    self.wfile.write(b": keepalive\n\n"); self.wfile.flush()
                except Exception:
                    return
        finally:
            with sse_lock:
                if self.wfile in sse_clients:
                    sse_clients.remove(self.wfile)


class ThreadedHTTPServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# --- HTML pages embedded -----------------------------------------------------

SCOREBOARD_HTML = r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Scoreboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:ital,wght@0,800;0,900;1,800;1,900&display=swap" rel="stylesheet">
<style>
  :root {
    --teamA: #3a2880;
    --teamB: #c8334d;
    --teamA-text: #ffffff;
    --teamB-text: #ffffff;
    --ink: #000000;
    --bar-h: 30px;
    --pen-h: 22px;
    --logo-side: 96px;
    --clock-w: 86px;
    --clock-h: 26px;
    --team-min-w: 110px;       /* réduit : team auto-shrink quand pas de pénalité */
    --logo-grad: linear-gradient(90deg, #3a2880 0%, #8a2960 55%, #c8334d 100%);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { width: 100%; height: 100%; background: transparent; overflow: hidden; }
  body { font-family: 'Barlow Condensed', sans-serif; font-style: italic; font-weight: 900; color: var(--ink); }

  .stage {
    position: fixed;
    top: 4px;
    left: 4px;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 2px;
    transition: opacity .35s ease, transform .35s ease;
  }
  .stage.hidden { opacity: 0; transform: translateY(-6px); pointer-events: none; }

  /* Top : logo en absolu par-dessus le bandeau qui s'étend jusqu'en dessous du logo */
  .top {
    position: relative;
    display: block;
    height: var(--logo-side);
  }
  .logo {
    position: absolute;
    top: 0; left: 0;
    width: var(--logo-side); height: var(--logo-side);
    z-index: 3;
    display: flex; align-items: center; justify-content: center;
  }
  .logo img {
    width: var(--logo-side); height: var(--logo-side);
    object-fit: contain; display: block;
    filter: drop-shadow(0 3px 6px rgba(0,0,0,.4));
  }

  /* Board : grille 4 colonnes (col 1 = sous le logo, cols 2/3/4 = team1/score/team2) */
  .board {
    display: grid;
    grid-template-columns: var(--logo-side) auto auto auto;
    grid-template-rows: var(--pen-h) var(--bar-h);
    row-gap: 2px;
    align-content: center;
    height: var(--logo-side);
    position: relative;
    isolation: isolate;
  }
  /* Bandeau de fond unique : démarre au milieu du logo et s'étend vers la droite */
  .bandeau-bg {
    grid-column: 1 / -1;
    grid-row: 2;
    background: var(--logo-grad);
    border-radius: 4px;
    align-self: stretch;
    height: var(--bar-h);
    margin-left: calc(var(--logo-side) / 2);   /* s'arrête au milieu du logo à gauche */
    /* liseret blanc intérieur + anneau gris extérieur, façon N2 */
    border: 2px solid rgba(255,255,255,.92);
    box-shadow: 0 0 0 1.5px rgba(70,70,70,.85), 0 2px 4px rgba(0,0,0,.35);
  }

  .pens { grid-row: 1; display: flex; align-items: end; gap: 4px; height: var(--pen-h); }
  .pens.pa { grid-column: 2; justify-content: flex-start; padding: 0 12px 0 2px; }   /* 2px côté logo */
  .pens.pb { grid-column: 4; justify-content: flex-end;   padding: 0 8px  0 12px; }  /* 8px côté bord */
  .score-block { grid-column: 3; }
  /* Pénalités : onglets collés au bandeau (haut arrondi, bas droit, sans liseret en bas) */
  .pen-chip {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 900; font-style: italic;
    font-size: 16px; line-height: 1;
    color: #ffffff;
    padding: 2px 7px;
    letter-spacing: .2px;
    font-feature-settings: "tnum";
    background: var(--logo-grad);
    border: 2px solid rgba(255,255,255,.92);
    border-bottom: none;
    border-radius: 4px 4px 0 0;
    box-shadow: 0 0 0 1.5px rgba(70,70,70,.85), 0 1px 2px rgba(0,0,0,.4);
    margin-bottom: -2px;        /* chevauche le bandeau pour cacher l'anneau gris du bas */
    position: relative;
    z-index: 1;
  }
  .pen-chip.warn { background: #f59e0b !important; color: #1a1300; }

  /* All bar cells share the same row, height, and centered alignment.
     Backgrounds are transparent: the parent .bandeau-bg gradient shows through,
     giving one continuous bandeau across team A | score | team B | period. */
  .cell {
    grid-row: 2;
    height: var(--bar-h);
    display: flex; align-items: center; justify-content: center;
    color: #ffffff;
    font-weight: 900; font-style: italic;
    line-height: 1;
    white-space: nowrap;
    background: transparent;
    position: relative;
    text-shadow: 0 1px 2px rgba(0,0,0,.45);
  }

  .team {
    font-size: 26px;
    letter-spacing: .3px;
    gap: 9px;
    min-width: var(--team-min-w);
  }
  /* Anchor team contents toward the score (centre) so names stay tight to the score block.
     Outer side (logo / bord) = 8px ; inner side (vers le score) = 16px. */
  .team-a { grid-column: 2; justify-content: flex-end;   padding: 0 16px 0 2px; }   /* 2px côté logo */
  .team-b { grid-column: 4; justify-content: flex-start; padding: 0 8px  0 16px; }   /* 8px côté bord */

  /* Background mode overrides applied via classes set by JS.
     'gradient' (default) = transparent (laisse passer le bandeau-bg).
     'color' = couvre le bandeau-bg avec une couleur unie. */
  .team.bg-color-a { background: var(--teamA); color: var(--teamA-text); }
  .team.bg-color-b { background: var(--teamB); color: var(--teamB-text); }

  .team-logo {
    width: 24px; height: 24px; object-fit: contain; display: block;
    filter: drop-shadow(0 1px 2px rgba(0,0,0,.45));
    flex: 0 0 auto;
  }
  .team-logo.hidden { display: none; }

  .score-block {
    padding: 0 14px;
    font-size: 26px;
    font-feature-settings: "tnum";
    gap: 9px;
  }
  .score-block .dash { opacity: .7; font-weight: 700; }
  .score-num { transition: transform .25s ease; display: inline-block; min-width: 16px; text-align: center; }
  .score-num.pulse { animation: pulse .5s ease; }
  @keyframes pulse {
    0%   { transform: scale(1); color: #ffffff; }
    40%  { transform: scale(1.4); color: #ffd166; }
    100% { transform: scale(1); color: #ffffff; }
  }

  /* Bottom row : un seul bandeau combiné [PÉRIODE  CHRONO], aligné sur le bord gauche du logo */
  .bottom-row {
    grid-column: 1 / span 2;
    grid-row: 2;
    display: flex;
    align-items: center;
    margin-top: 2px;
    padding-left: 0;
  }
  .pill {
    height: var(--clock-h);
    background: var(--logo-grad);
    color: #ffffff;
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 900; font-style: italic;
    font-size: 18px;
    line-height: 1;
    letter-spacing: 1px;
    font-feature-settings: "tnum";
    display: inline-flex; align-items: center;
    border-radius: 3px;
    /* même liseret blanc + anneau gris que le bandeau */
    border: 2px solid rgba(255,255,255,.92);
    box-shadow: 0 0 0 1.5px rgba(70,70,70,.85), 0 2px 4px rgba(0,0,0,.35);
    text-shadow: 0 1px 1px rgba(0,0,0,.35);
  }
  .pill.combined {
    padding: 0 12px;
    gap: 12px;
  }
  /* Période et chrono : strictement le même formatage et même blanc pur */
  .pill .pill-period,
  .pill .pill-clock {
    font-family: inherit;
    font-weight: inherit;
    font-style: inherit;
    font-size: inherit;
    letter-spacing: inherit;
    font-feature-settings: "tnum";
    color: #ffffff;
    text-shadow: 0 1px 1px rgba(0,0,0,.35);
  }
  .pill .pill-sep { opacity: .55; font-weight: 700; }
  .pill.combined.paused .pill-clock { opacity: 1; }   /* pas de dim → reste blanc franc */
</style>
</head>
<body>
  <div class="stage" id="stage">
    <div class="top">
      <div class="board">
        <div class="bandeau-bg"></div>
        <div class="pens pa" id="pensA"></div>
        <div class="pens pb" id="pensB"></div>
        <div class="cell team team-a" id="teamABar">
          <img class="team-logo hidden" id="logoA" alt="">
          <span id="nameA">—</span>
        </div>
        <div class="cell score-block">
          <span class="score-num" id="scoreA">0</span>
          <span class="dash">−</span>
          <span class="score-num" id="scoreB">0</span>
        </div>
        <div class="cell team team-b" id="teamBBar">
          <span id="nameB">—</span>
          <img class="team-logo hidden" id="logoB" alt="">
        </div>
      </div>
      <div class="logo"><img id="mainLogo" src="/logo" alt=""></div>
    </div>

    <div class="bottom-row">
      <div class="pill combined" id="combinedPill">
        <span class="pill-period" id="period">P1</span>
        <span class="pill-sep">·</span>
        <span class="pill-clock"  id="clock">25:00</span>
      </div>
    </div>
  </div>

<script>
const $ = (id) => document.getElementById(id);
let lastState = null;
let prevA = 0, prevB = 0;

function pad(n){return String(n).padStart(2,'0');}
function fmt(s){s=Math.max(0,Math.floor(s));return pad(Math.floor(s/60))+':'+pad(s%60);}
function clockSeconds(c){
  if(!c.running) return c.lastValue;
  const e=(Date.now()/1000)-c.lastUpdate;
  return c.direction==='up'?c.lastValue+e:Math.max(0,c.lastValue-e);
}
function penaltySeconds(p){
  if(!p.running) return p.lastValue;
  const e=(Date.now()/1000)-p.lastUpdate;
  return Math.max(0,p.lastValue-e);
}

function setLogo(imgEl, src){
  if (src && src.length > 0) {
    imgEl.src = src;
    imgEl.classList.remove('hidden');
  } else {
    imgEl.classList.add('hidden');
    imgEl.removeAttribute('src');
  }
}

function renderPenalties(elId, list) {
  const el = $(elId);
  el.innerHTML = '';
  for (const p of (list || [])) {
    const sec = penaltySeconds(p);
    const chip = document.createElement('span');
    chip.className = 'pen-chip' + (sec <= 30 ? ' warn' : '');
    chip.textContent = fmt(sec);
    el.appendChild(chip);
  }
}

function applyState(s) {
  if (!s) return;
  if (s.teamA.score !== prevA) {
    $('scoreA').classList.remove('pulse'); void $('scoreA').offsetWidth;
    $('scoreA').classList.add('pulse');
    prevA = s.teamA.score;
  }
  if (s.teamB.score !== prevB) {
    $('scoreB').classList.remove('pulse'); void $('scoreB').offsetWidth;
    $('scoreB').classList.add('pulse');
    prevB = s.teamB.score;
  }
  $('nameA').textContent  = s.teamA.name;
  $('nameB').textContent  = s.teamB.name;
  $('scoreA').textContent = s.teamA.score;
  $('scoreB').textContent = s.teamB.score;
  $('period').textContent = s.periodLabel;
  const root = document.documentElement.style;
  root.setProperty('--teamA',      s.teamA.color);
  root.setProperty('--teamB',      s.teamB.color);
  root.setProperty('--teamA-text', s.teamA.textColor || '#ffffff');
  root.setProperty('--teamB-text', s.teamB.textColor || '#ffffff');

  // Apply bg mode per team (gradient = laisse passer le bandeau, color = recouvre)
  const tA = $('teamABar');
  tA.classList.toggle('bg-color-a', s.teamA.bgMode === 'color');
  const tB = $('teamBBar');
  tB.classList.toggle('bg-color-b', s.teamB.bgMode === 'color');

  setLogo($('logoA'), s.teamA.logo);
  setLogo($('logoB'), s.teamB.logo);
  $('mainLogo').src = (s.mainLogo && s.mainLogo.length > 0) ? s.mainLogo : '/logo';
  $('stage').classList.toggle('hidden', !s.visible);
  $('combinedPill').classList.toggle('paused', !s.clock.running);
  lastState = s;
  renderPenalties('pensA', s.teamA.penalties);
  renderPenalties('pensB', s.teamB.penalties);
}

function tick() {
  if (!lastState) return;
  $('clock').textContent = fmt(clockSeconds(lastState.clock));
  for (const [elId, list] of [['pensA', lastState.teamA.penalties], ['pensB', lastState.teamB.penalties]]) {
    const chips = $(elId).querySelectorAll('.pen-chip');
    chips.forEach((chip, i) => {
      if (!list || !list[i]) return;
      const sec = penaltySeconds(list[i]);
      chip.textContent = fmt(sec);
      chip.classList.toggle('warn', sec <= 30);
    });
  }
}
setInterval(tick, 100);

function connect() {
  const es = new EventSource('/events');
  es.onmessage = (e) => { try { applyState(JSON.parse(e.data)); } catch (_) {} };
  es.onerror = () => { es.close(); setTimeout(connect, 1500); };
}
connect();
</script>
</body>
</html>"""


CONTROL_HTML = r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>RILH Console</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:ital,wght@0,800;0,900;1,800;1,900&family=DM+Sans:wght@500;700&family=JetBrains+Mono:wght@700&display=swap" rel="stylesheet">
<style>
  :root {
    --teamA: #3a2880;
    --teamB: #c8334d;
    --bg: #0a0820;
    --panel: #14112e;
    --panel-2: #1e1a3f;
    --accent: #c8334d;
    --ok: #16a34a;
    --warn: #f59e0b;
    --ink: #f1f5f9;
    --ink-dim: rgba(241,245,249,.6);
    --line: rgba(255,255,255,.08);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
  html, body { background: var(--bg); color: var(--ink); font-family: 'DM Sans', system-ui, sans-serif; min-height: 100vh; }
  body { padding: 16px; padding-bottom: 60px; }

  h1 { font-family: 'Barlow Condensed', sans-serif; font-weight: 900; font-size: 28px; letter-spacing: 1px; text-transform: uppercase; }
  h2 { font-family: 'Barlow Condensed', sans-serif; font-weight: 800; font-size: 14px; letter-spacing: 2px; text-transform: uppercase; color: var(--ink-dim); margin-bottom: 10px; }

  .top {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 18px; padding-bottom: 14px; border-bottom: 1px solid var(--line);
  }
  .top .status { font-size: 11px; letter-spacing: 1.5px; text-transform: uppercase; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--ok); margin-right: 6px; vertical-align: middle; }
  .dot.off { background: var(--accent); }

  .grid { display: grid; gap: 14px; }
  @media (min-width: 720px) { .grid-2 { grid-template-columns: 1fr 1fr; } }

  .panel {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 14px;
  }

  .team-card { position: relative; overflow: hidden; }
  .team-card.A::before, .team-card.B::before {
    content:''; position: absolute; left: 0; top: 0; bottom: 0; width: 6px;
  }
  .team-card.A::before { background: var(--teamA); }
  .team-card.B::before { background: var(--teamB); }

  .scoreline {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 64px;
    text-align: center;
    line-height: 1;
    margin: 6px 0 14px;
  }

  .row { display: flex; gap: 8px; flex-wrap: wrap; }
  .row.tight { gap: 6px; }

  button, .btn {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800;
    font-size: 18px;
    letter-spacing: 1px;
    text-transform: uppercase;
    border: 1px solid var(--line);
    background: var(--panel-2);
    color: var(--ink);
    padding: 14px 16px;
    border-radius: 10px;
    cursor: pointer;
    transition: transform .06s ease, background .15s ease, border-color .15s ease;
    flex: 1;
    min-height: 52px;
  }
  button:active { transform: scale(.97); }
  button:disabled { opacity: .35; cursor: not-allowed; }
  button.primary { background: var(--accent); border-color: var(--accent); }
  button.ok      { background: var(--ok); border-color: var(--ok); }
  button.warn    { background: var(--warn); border-color: var(--warn); color: #1a1300; }
  button.ghost   { background: transparent; }
  button.blue    { background: #2563eb; border-color: #2563eb; }
  button.red     { background: #dc2626; border-color: #dc2626; }
  button.big     { font-size: 28px; min-height: 64px; }
  button.small   { font-size: 14px; min-height: 38px; padding: 8px 10px; }

  input[type=text], input[type=color], input[type=number], input[type=url] {
    width: 100%; padding: 10px 12px; border-radius: 8px;
    background: var(--panel-2); color: var(--ink);
    border: 1px solid var(--line); font: inherit; font-size: 16px;
  }
  input[type=color] { padding: 4px; height: 44px; cursor: pointer; }
  input[type=file] { color: var(--ink-dim); font-size: 13px; }

  label { display: block; font-size: 11px; letter-spacing: 1.5px; text-transform: uppercase; color: var(--ink-dim); margin: 12px 0 4px; }
  label:first-child { margin-top: 0; }

  .chrono {
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 56px;
    padding: 10px 0;
  }
  .chrono.paused { color: var(--accent); }

  .periods { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
  .periods button { padding: 12px 8px; font-size: 16px; min-height: 44px; }
  .periods button.active { background: var(--accent); border-color: var(--accent); }

  .duration-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }

  .pen-list { display: flex; flex-direction: column; gap: 6px; margin-top: 10px; }
  .pen-row {
    display: flex; align-items: center; gap: 8px;
    background: var(--panel-2);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 8px 10px;
  }
  .pen-time { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 22px; flex: 1; }
  .pen-time.warn { color: var(--warn); }
  .pen-rm {
    background: transparent;
    border: 1px solid var(--accent);
    color: var(--accent);
    padding: 6px 10px;
    min-height: 36px;
    font-size: 14px;
    flex: 0 0 auto;
  }
  .pen-empty { font-size: 12px; color: var(--ink-dim); padding: 4px 0; }

  .logo-preview {
    display: flex; align-items: center; gap: 10px; margin-top: 8px;
    background: var(--panel-2); border: 1px solid var(--line);
    border-radius: 8px; padding: 8px 10px;
  }
  .logo-preview img {
    width: 40px; height: 40px; object-fit: contain; background: rgba(255,255,255,.05);
    border-radius: 4px;
  }
  .logo-preview .none { color: var(--ink-dim); font-size: 12px; flex: 1; }

  .small-text { font-size: 12px; color: var(--ink-dim); }
  details { margin-top: 12px; }
  details > summary { cursor: pointer; padding: 8px 0; color: var(--ink-dim); font-size: 12px; letter-spacing: 1.5px; text-transform: uppercase; list-style: none; }
  details > summary::-webkit-details-marker { display: none; }
  details > summary::after { content: ' ▾'; opacity: .6; }
  details[open] > summary::after { content: ' ▴'; }
  summary:hover { color: var(--ink); }

  /* Live preview */
  .preview-wrap {
    background: #1a1f2e;
    border: 1px solid var(--line);
    border-radius: 12px;
    margin-bottom: 14px;
    padding: 0;
    overflow: hidden;
    position: relative;
  }
  .preview-wrap::before {
    content: 'APERÇU LIVE';
    position: absolute; top: 6px; right: 8px;
    font-size: 9px; letter-spacing: 1.5px;
    color: rgba(255,255,255,.4);
    text-transform: uppercase;
    z-index: 2;
    pointer-events: none;
  }
  .preview-frame {
    width: 100%; height: 150px;
    border: 0; display: block;
    background: #1a1f2e;
  }

  .color-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .bg-modes button.active,
  #presetRow button.active { background: var(--accent); border-color: var(--accent); }
</style>
</head>
<body>
  <div class="top">
    <h1>Console RILH</h1>
    <div class="status"><span class="dot" id="connDot"></span><span id="connTxt">connecté</span></div>
  </div>

  <!-- Aperçu live -->
  <div class="preview-wrap">
    <iframe class="preview-frame" id="previewFrame" src="/scoreboard"></iframe>
  </div>

  <!-- Chronomètre -->
  <div class="panel">
    <h2>Chronomètre</h2>
    <div class="chrono" id="chrono">00:00</div>
    <div class="row">
      <button id="toggleClockBtn" class="ok big" onclick="toggleClock()">▶ Start</button>
    </div>
    <details>
      <summary>Réglages chrono</summary>
      <div class="row" style="margin-top: 8px;">
        <button class="ghost small" onclick="resetDown()">↺ Reset (décompte)</button>
        <button class="ghost small" onclick="resetUp()">↺ 0:00 (comptage ↑)</button>
      </div>
      <div class="row" style="margin-top: 8px;">
        <button class="ghost small" onclick="post('/clock/adjust', {delta: -1})">−1s</button>
        <button class="ghost small" onclick="post('/clock/adjust', {delta: -10})">−10s</button>
        <button class="ghost small" onclick="post('/clock/adjust', {delta: 10})">+10s</button>
        <button class="ghost small" onclick="post('/clock/adjust', {delta: 60})">+1min</button>
      </div>
      <div class="small-text" id="chronoMode" style="margin-top: 8px; text-align: center;"></div>
    </details>
  </div>

  <!-- Equipes -->
  <div class="grid grid-2">
    <div class="panel team-card A">
      <label>Nom équipe 1</label>
      <input type="text" id="nameA" onchange="setTeam('A')">
      <div class="scoreline" id="sA">0</div>
      <div class="row">
        <button class="big blue" onclick="post('/score', {team:'A', delta:1})">+1</button>
        <button class="big red"  onclick="post('/score', {team:'A', delta:-1})">−1</button>
      </div>

      <label style="margin-top:14px;">Pénalités</label>
      <div class="row tight">
        <button class="small" id="penA2"  onclick="post('/penalty/add', {team:'A', duration:120})">+ 2:00</button>
        <button class="small" id="penA5"  onclick="post('/penalty/add', {team:'A', duration:300})">+ 5:00</button>
        <button class="small" id="penA10" onclick="post('/penalty/add', {team:'A', duration:600})">+ 10:00</button>
      </div>
      <div class="pen-list" id="penListA"><div class="pen-empty">Aucune pénalité</div></div>

      <details>
        <summary>Logo & couleurs</summary>
        <label>Logo (URL)</label>
        <input type="url" id="logoUrlA" placeholder="https://… .png/.jpg/.svg" onchange="setLogoUrl('A')">
        <label>… ou fichier image</label>
        <input type="file" id="logoFileA" accept="image/*" onchange="setLogoFile('A')">
        <div class="logo-preview" id="logoPrevA"><span class="none">Pas de logo</span></div>
        <label style="margin-top:10px;">Fond derrière le nom</label>
        <div class="row tight bg-modes" data-team="A">
          <button class="small" data-mode="gradient" onclick="setBgMode('A','gradient')">Bandeau chrono</button>
          <button class="small" data-mode="color"    onclick="setBgMode('A','color')">Couleur unie</button>
        </div>
        <div class="color-pair" style="margin-top: 10px;">
          <div>
            <label>Couleur fond</label>
            <input type="color" id="colorA" onchange="setTeam('A')">
          </div>
          <div>
            <label>Couleur police</label>
            <input type="color" id="textColorA" onchange="setTeam('A')">
          </div>
        </div>
      </details>
    </div>

    <div class="panel team-card B">
      <label>Nom équipe 2</label>
      <input type="text" id="nameB" onchange="setTeam('B')">
      <div class="scoreline" id="sB">0</div>
      <div class="row">
        <button class="big blue" onclick="post('/score', {team:'B', delta:1})">+1</button>
        <button class="big red"  onclick="post('/score', {team:'B', delta:-1})">−1</button>
      </div>

      <label style="margin-top:14px;">Pénalités</label>
      <div class="row tight">
        <button class="small" id="penB2"  onclick="post('/penalty/add', {team:'B', duration:120})">+ 2:00</button>
        <button class="small" id="penB5"  onclick="post('/penalty/add', {team:'B', duration:300})">+ 5:00</button>
        <button class="small" id="penB10" onclick="post('/penalty/add', {team:'B', duration:600})">+ 10:00</button>
      </div>
      <div class="pen-list" id="penListB"><div class="pen-empty">Aucune pénalité</div></div>

      <details>
        <summary>Logo & couleurs</summary>
        <label>Logo (URL)</label>
        <input type="url" id="logoUrlB" placeholder="https://… .png/.jpg/.svg" onchange="setLogoUrl('B')">
        <label>… ou fichier image</label>
        <input type="file" id="logoFileB" accept="image/*" onchange="setLogoFile('B')">
        <div class="logo-preview" id="logoPrevB"><span class="none">Pas de logo</span></div>
        <label style="margin-top:10px;">Fond derrière le nom</label>
        <div class="row tight bg-modes" data-team="B">
          <button class="small" data-mode="gradient" onclick="setBgMode('B','gradient')">Bandeau chrono</button>
          <button class="small" data-mode="color"    onclick="setBgMode('B','color')">Couleur unie</button>
        </div>
        <div class="color-pair" style="margin-top: 10px;">
          <div>
            <label>Couleur fond</label>
            <input type="color" id="colorB" onchange="setTeam('B')">
          </div>
          <div>
            <label>Couleur police</label>
            <input type="color" id="textColorB" onchange="setTeam('B')">
          </div>
        </div>
      </details>
    </div>
  </div>

  <!-- Divers -->
  <div class="panel" style="margin-top: 14px;">
    <h2>Affichage</h2>
    <div class="row">
      <button id="visBtn" onclick="toggleVisible()">Cacher l'overlay</button>
      <button onclick="post('/swap')">⇄ Inverser équipes</button>
    </div>
    <p class="small-text" style="margin-top: 12px;">
      OBS Browser Source : <code id="obsUrl"></code>
    </p>
  </div>

  <!-- Logo -->
  <div class="panel">
    <details>
      <summary>Logo</summary>
      <label>Niveau</label>
      <div class="row tight" id="presetRow">
        <button class="small" data-preset="elite" onclick="setMainLogoPreset('elite')">Elite</button>
        <button class="small" data-preset="n1"    onclick="setMainLogoPreset('n1')">N1</button>
        <button class="small" data-preset="n2"    onclick="setMainLogoPreset('n2')">N2</button>
        <button class="small" data-preset="n3"    onclick="setMainLogoPreset('n3')">N3</button>
      </div>
      <label>Lien (URL)</label>
      <input type="url" id="mainLogoUrl" placeholder="https://… .png/.jpg/.svg/.webp" onchange="setMainLogoUrl()">
      <label>… ou fichier image</label>
      <input type="file" id="mainLogoFile" accept="image/*" onchange="setMainLogoFile()">
      <div class="logo-preview" id="mainLogoPrev"><span class="none">Logo N2 par défaut</span></div>
      <div class="row" style="margin-top: 10px;">
        <button class="ghost small" onclick="post('/main/logo', {value: ''})">Restaurer logo N2</button>
      </div>
    </details>
  </div>

  <!-- Période -->
  <div class="panel">
    <details>
      <summary>Période & durées</summary>
      <div class="periods" style="margin-top: 8px;">
        <button onclick="post('/period', {label:'P1'})" data-p="P1">P1</button>
        <button onclick="post('/period', {label:'P2'})" data-p="P2">P2</button>
        <button onclick="post('/period', {label:'PROL'})" data-p="PROL">PROL</button>
      </div>
      <div class="duration-grid" style="margin-top: 12px;">
        <div>
          <label>Durée période (min)</label>
          <input type="number" id="periodMin" min="1" max="99" step="1">
        </div>
        <div>
          <label>Durée prolongation (min)</label>
          <input type="number" id="otMin" min="1" max="60" step="1">
        </div>
      </div>
      <div class="row" style="margin-top: 10px;">
        <button class="ghost small" onclick="setDurations()">Appliquer durées</button>
      </div>
    </details>
  </div>

<script>
const $ = (id) => document.getElementById(id);
let cur = null;
let visible = true;

function pad(n){return String(n).padStart(2,'0');}
function fmt(s){s=Math.max(0,Math.floor(s));return pad(Math.floor(s/60))+':'+pad(s%60);}

async function post(path, body) {
  try {
    await fetch(path, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(body || {})
    });
  } catch (e) { console.error(e); }
}

function clockSeconds(c){
  if(!c.running) return c.lastValue;
  const e=(Date.now()/1000)-c.lastUpdate;
  return c.direction==='up'?c.lastValue+e:Math.max(0,c.lastValue-e);
}
function penaltySeconds(p){
  if(!p.running) return p.lastValue;
  const e=(Date.now()/1000)-p.lastUpdate;
  return Math.max(0,p.lastValue-e);
}

function renderPenList(team, list) {
  const el = $(team === 'A' ? 'penListA' : 'penListB');
  el.innerHTML = '';
  if (!list || list.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'pen-empty';
    empty.textContent = 'Aucune pénalité';
    el.appendChild(empty);
    return;
  }
  for (const p of list) {
    const row = document.createElement('div');
    row.className = 'pen-row';
    const sec = penaltySeconds(p);
    const t = document.createElement('div');
    t.className = 'pen-time' + (sec <= 30 ? ' warn' : '');
    t.textContent = fmt(sec) + '  /  ' + fmt(p.duration);
    const rm = document.createElement('button');
    rm.className = 'pen-rm';
    rm.textContent = '× retirer';
    rm.onclick = () => post('/penalty/remove', {team: team, id: p.id});
    row.appendChild(t); row.appendChild(rm);
    el.appendChild(row);
  }
}

function renderLogoPreview(team, src) {
  const el = $('logoPrev' + team);
  el.innerHTML = '';
  if (!src) {
    const span = document.createElement('span');
    span.className = 'none';
    span.textContent = 'Pas de logo';
    el.appendChild(span);
    return;
  }
  const img = document.createElement('img');
  img.src = src;
  const txt = document.createElement('div');
  txt.style.flex = '1';
  txt.style.fontSize = '12px';
  txt.style.color = 'var(--ink-dim)';
  txt.style.overflow = 'hidden';
  txt.style.textOverflow = 'ellipsis';
  txt.style.whiteSpace = 'nowrap';
  txt.textContent = src.startsWith('data:') ? '(fichier intégré)' : src;
  const rm = document.createElement('button');
  rm.className = 'small ghost';
  rm.style.flex = '0 0 auto';
  rm.style.minHeight = '32px';
  rm.style.padding = '4px 10px';
  rm.textContent = '×';
  rm.onclick = () => post('/team/logo', {team: team, value: ''});
  el.appendChild(img); el.appendChild(txt); el.appendChild(rm);
}

function apply(s) {
  cur = s;
  $('sA').textContent  = s.teamA.score;
  $('sB').textContent  = s.teamB.score;
  document.documentElement.style.setProperty('--teamA', s.teamA.color);
  document.documentElement.style.setProperty('--teamB', s.teamB.color);

  document.querySelectorAll('.periods button').forEach(b => {
    b.classList.toggle('active', b.dataset.p === s.periodLabel);
  });

  if (document.activeElement.id !== 'nameA')   $('nameA').value   = s.teamA.name;
  if (document.activeElement.id !== 'nameB')   $('nameB').value   = s.teamB.name;
  if (document.activeElement.id !== 'colorA')  $('colorA').value  = s.teamA.color;
  if (document.activeElement.id !== 'colorB')  $('colorB').value  = s.teamB.color;
  if (document.activeElement.id !== 'textColorA') $('textColorA').value = s.teamA.textColor || '#ffffff';
  if (document.activeElement.id !== 'textColorB') $('textColorB').value = s.teamB.textColor || '#ffffff';
  if (document.activeElement.id !== 'periodMin') $('periodMin').value = Math.round(s.periodDuration/60);
  if (document.activeElement.id !== 'otMin')     $('otMin').value     = Math.round(s.otDuration/60);
  if (document.activeElement.id !== 'logoUrlA') $('logoUrlA').value = s.teamA.logo && !s.teamA.logo.startsWith('data:') ? s.teamA.logo : '';
  if (document.activeElement.id !== 'logoUrlB') $('logoUrlB').value = s.teamB.logo && !s.teamB.logo.startsWith('data:') ? s.teamB.logo : '';

  $('chronoMode').textContent = (s.clock.direction === 'up')
    ? 'Mode comptage ↑ (depuis 0:00)'
    : 'Mode décompte ↓';

  // Toggle button state: green "Start" when paused, yellow "Pause" when running
  const tbtn = $('toggleClockBtn');
  if (s.clock.running) {
    tbtn.textContent = '❙❙ Pause';
    tbtn.classList.remove('ok');
    tbtn.classList.add('warn');
  } else {
    tbtn.textContent = '▶ Start';
    tbtn.classList.remove('warn');
    tbtn.classList.add('ok');
  }

  // Active bg-mode buttons per team
  document.querySelectorAll('.bg-modes').forEach(group => {
    const team = group.dataset.team;
    const mode = (team === 'A') ? (s.teamA.bgMode || 'gradient') : (s.teamB.bgMode || 'gradient');
    group.querySelectorAll('button').forEach(b => {
      b.classList.toggle('active', b.dataset.mode === mode);
    });
  });

  renderLogoPreview('A', s.teamA.logo);
  renderLogoPreview('B', s.teamB.logo);
  renderMainLogoPreview(s.mainLogo);
  if (document.activeElement.id !== 'mainLogoUrl') {
    const v = s.mainLogo || '';
    $('mainLogoUrl').value = (v && !v.startsWith('data:') && !v.startsWith('/preset-logo/')) ? v : '';
  }
  // Highlight active preset button
  document.querySelectorAll('#presetRow button').forEach(b => {
    const expected = '/preset-logo/' + b.dataset.preset;
    b.classList.toggle('active', s.mainLogo === expected);
  });

  const capA = (s.teamA.penalties || []).length >= 3;
  const capB = (s.teamB.penalties || []).length >= 3;
  ['penA2','penA5','penA10'].forEach(id => $(id).disabled = capA);
  ['penB2','penB5','penB10'].forEach(id => $(id).disabled = capB);

  renderPenList('A', s.teamA.penalties);
  renderPenList('B', s.teamB.penalties);

  visible = s.visible;
  $('visBtn').textContent = visible ? "Cacher l'overlay" : "Afficher l'overlay";
}

function tick() {
  if (!cur) return;
  const sec = clockSeconds(cur.clock);
  $('chrono').textContent = fmt(sec);
  $('chrono').classList.toggle('paused', !cur.clock.running);
  for (const team of ['A','B']) {
    const list = cur['team'+team].penalties || [];
    const el = $(team === 'A' ? 'penListA' : 'penListB');
    el.querySelectorAll('.pen-row').forEach((row, i) => {
      const p = list[i];
      if (!p) return;
      const t = row.querySelector('.pen-time');
      const r = penaltySeconds(p);
      t.textContent = fmt(r) + '  /  ' + fmt(p.duration);
      t.classList.toggle('warn', r <= 30);
    });
  }
}
setInterval(tick, 100);

function setTeam(t) {
  const name = $('name'+t).value.trim();
  const color = $('color'+t).value;
  const textColor = $('textColor'+t).value;
  post('/team', {team: t, name: name, color: color, textColor: textColor});
}

async function resetDown() {
  await post('/clock/direction', {value: 'down'});
  await post('/clock/reset');
}
async function resetUp() {
  await post('/clock/direction', {value: 'up'});
  await post('/clock/reset', {seconds: 0});
}

function toggleClock() {
  if (cur && cur.clock.running) post('/clock/pause');
  else post('/clock/start');
}

function setBgMode(team, mode) {
  post('/team', {team: team, bgMode: mode});
}

function setLogoUrl(t) {
  const v = $('logoUrl'+t).value.trim();
  post('/team/logo', {team: t, value: v});
}

function setLogoFile(t) {
  const input = $('logoFile'+t);
  const f = input.files && input.files[0];
  if (!f) return;
  if (f.size > 800 * 1024) {
    alert('Image trop volumineuse (>800 KB). Préférez une URL ou redimensionnez.');
    input.value = '';
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    post('/team/logo', {team: t, value: reader.result});
    input.value = '';
  };
  reader.readAsDataURL(f);
}

function setMainLogoUrl() {
  const v = $('mainLogoUrl').value.trim();
  post('/main/logo', {value: v});
}

function setMainLogoPreset(name) {
  post('/main/logo', {value: '/preset-logo/' + name});
}

function setMainLogoFile() {
  const input = $('mainLogoFile');
  const f = input.files && input.files[0];
  if (!f) return;
  if (f.size > 800 * 1024) {
    alert('Image trop volumineuse (>800 KB). Préférez une URL ou redimensionnez.');
    input.value = '';
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    post('/main/logo', {value: reader.result});
    input.value = '';
  };
  reader.readAsDataURL(f);
}

function renderMainLogoPreview(src) {
  const el = $('mainLogoPrev');
  el.innerHTML = '';
  if (!src) {
    const span = document.createElement('span');
    span.className = 'none';
    span.textContent = 'Logo N2 par défaut';
    el.appendChild(span);
    return;
  }
  const img = document.createElement('img');
  img.src = src;
  const txt = document.createElement('div');
  txt.style.flex = '1';
  txt.style.fontSize = '12px';
  txt.style.color = 'var(--ink-dim)';
  txt.style.overflow = 'hidden';
  txt.style.textOverflow = 'ellipsis';
  txt.style.whiteSpace = 'nowrap';
  txt.textContent = src.startsWith('data:') ? '(fichier intégré)' : src;
  el.appendChild(img); el.appendChild(txt);
}

function setDurations() {
  const pm = parseInt($('periodMin').value, 10);
  const om = parseInt($('otMin').value, 10);
  if (pm > 0)  post('/period/duration',    {value: pm * 60});
  if (om > 0)  post('/period/ot_duration', {value: om * 60});
}

function toggleVisible() { post('/visible', {value: !visible}); }

function connect() {
  const es = new EventSource('/events');
  es.onmessage = (e) => {
    $('connDot').classList.remove('off');
    $('connTxt').textContent = 'connecté';
    try { apply(JSON.parse(e.data)); } catch (_) {}
  };
  es.onerror = () => {
    $('connDot').classList.add('off');
    $('connTxt').textContent = 'reconnexion…';
    es.close();
    setTimeout(connect, 1200);
  };
}
connect();

$('obsUrl').textContent = window.location.origin + '/scoreboard';

if ('wakeLock' in navigator) {
  navigator.wakeLock.request('screen').catch(() => {});
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
      navigator.wakeLock.request('screen').catch(() => {});
    }
  });
}
</script>
</body>
</html>"""


def get_lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]; s.close(); return ip
    except Exception:
        return "127.0.0.1"


def main():
    if not LOGO_PATH.exists():
        print(f"  [!] Logo missing: {LOGO_PATH} — /logo will 404.")
    server = ThreadedHTTPServer(("0.0.0.0", PORT), Handler)
    threading.Thread(target=penalty_ticker, daemon=True).start()
    ip = get_lan_ip()
    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  RILH Scoreboard server                                     │")
    print("  ├─────────────────────────────────────────────────────────────┤")
    print(f"  │  OBS Browser Source :  http://localhost:{PORT}/scoreboard    │")
    print(f"  │  Console (Mac)      :  http://localhost:{PORT}/control       │")
    print(f"  │  Console (iPad LAN) :  http://{ip}:{PORT}/control".ljust(63) + "│")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()
    print("  Ctrl+C pour arrêter.")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stop.")
        server.shutdown()


if __name__ == "__main__":
    main()
