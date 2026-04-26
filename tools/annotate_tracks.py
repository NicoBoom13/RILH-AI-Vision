"""
RILH-AI-Vision — track-level truth annotation tool.

Localhost web UI for labelling each player TRACK with its ground-truth
team (A / B) plus an `is_referee` flag. Used to build the truth set
that `tools/bench_team_engines.py` measures the four team engines
(hsv / osnet / siglip / contrastive) against.

Difference from `tools/annotate_crops.py`:
  * `annotate_crops.py` labels each per-frame jersey crop with its
    digit number (or X). Used to train the PARSeq fine-tune.
  * `annotate_tracks.py` labels the WHOLE track at once with its team
    + ref flag. Used to score whichever team engine you run.

Usage:
  python tools/annotate_tracks.py runs/run24 [runs/run25 ...] [--port 8001]

Pass one or more run folders (each must contain p1_a_detections.json).
The UI walks every player track in those runs, shows up to 6 thumbnail
crops sampled from the track, and lets you pick A / B / Referee /
Other / Skip with single-key shortcuts.

Annotations save incrementally to data/jersey_numbers/track_truth.json:
  {
    "tracks": {
      "run24/123": {"team": "A", "is_referee": false},
      "run24/187": {"team": null, "is_referee": true},
      "run25/42":  {"team": "B", "is_referee": false},
      ...
    },
    "metadata": {"produced_by": "...", "n_total": ..., ...}
  }

Track labels:
  "A"  team A (whichever team is on the left in the first frame)
  "B"  team B
  "X"  not a player (referee, spectator, mis-detection)
  ""   unlabeled — comes back later

Keyboard:
  A     team A + next
  B     team B + next
  R     mark as referee + next  (writes team=null, is_referee=true)
  X     mark as not-a-player + next (team=null, is_referee=false → spectator/junk)
  S     skip + next
  ←     prev
  →     next without saving
"""

import argparse
import json
import threading
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np


PERSON_CLASS = 0
THUMBS_PER_TRACK = 6
THUMB_W, THUMB_H = 96, 144  # roller torso aspect ~3:2 vertical


HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>RILH track-level annotation</title>
<style>
  body { font: 14px -apple-system, sans-serif; max-width: 1100px; margin: 1em auto; padding: 0 1em; background: #0a0d12; color: #cfeaff; }
  h2 { margin: 0.4em 0; color: #7fe0ff; font-weight: 600; letter-spacing: 0.04em; }
  .progress { color: #88a; margin-bottom: 0.5em; font-variant-numeric: tabular-nums; }
  .header { color: #aaa; font-size: 11px; margin-bottom: 0.6em; }
  .strip { display: flex; gap: 6px; margin: 1em 0; padding: 8px; background: #0f1620; border-radius: 6px; min-height: 160px; align-items: center; flex-wrap: wrap; }
  .strip img { image-rendering: pixelated; height: 144px; border: 1px solid rgba(0,220,255,0.2); border-radius: 3px; background: #06080c; }
  .controls { display: flex; gap: 0.4em; align-items: center; flex-wrap: wrap; margin: 8px 0; }
  button { font: 600 13px -apple-system, sans-serif; padding: 8px 14px; cursor: pointer; border: 1px solid rgba(0,220,255,0.3); background: rgba(0,30,55,0.7); color: #cfeaff; border-radius: 5px; transition: background 0.1s; }
  button:hover { background: rgba(0,60,90,0.9); }
  button.team-a { border-color: #06ffa5; color: #06ffa5; }
  button.team-b { border-color: #4cc9f0; color: #4cc9f0; }
  button.ref    { border-color: #ffd60a; color: #ffd60a; }
  button.skip   { border-color: #888; color: #aaa; }
  .hint { color: #888; font-size: 12px; margin-top: 0.4em; }
  kbd { background: #1a1f28; border: 1px solid #333; border-radius: 3px; padding: 1px 6px; font-size: 11px; color: #cfeaff; }
  .stats { margin-top: 1.5em; padding: 0.7em 1em; background: #0f1620; border-radius: 6px; font-size: 12px; }
  .stats b { color: #6affff; }
  .label-current { display: inline-block; padding: 2px 10px; border-radius: 3px; font-weight: 600; margin-left: 8px; }
  .label-A   { background: rgba(6,255,165,0.2);   color: #06ffa5; }
  .label-B   { background: rgba(76,201,240,0.2); color: #4cc9f0; }
  .label-REF { background: rgba(255,214,10,0.2); color: #ffd60a; }
  .label-X   { background: rgba(160,160,160,0.2); color: #aaa; }
</style>
</head>
<body>
  <h2>RILH track-level annotation</h2>
  <div class="progress" id="progress">Loading…</div>
  <div class="header" id="header"></div>
  <div class="strip" id="strip"></div>
  <div class="controls">
    <button id="btnA"   class="team-a">A — Team A</button>
    <button id="btnB"   class="team-b">B — Team B</button>
    <button id="btnRef" class="ref">R — Referee</button>
    <button id="btnX"   class="skip">X — Not a player</button>
    <button id="btnSkip" class="skip">S — Skip</button>
    <button id="btnPrev" class="skip">← Prev</button>
    <button id="btnNext" class="skip">Next →</button>
  </div>
  <p class="hint">
    <kbd>A</kbd> team A · <kbd>B</kbd> team B · <kbd>R</kbd> referee ·
    <kbd>X</kbd> not a player · <kbd>S</kbd> skip ·
    <kbd>←</kbd>/<kbd>→</kbd> navigate.
    Convention: Team A = whichever team is on the left in this clip's
    very first detected frame; Team B = the other one. Stay consistent
    within a clip; absolute team identity across clips doesn't matter
    for the bench (only intra-clip purity is scored).
  </p>
  <div class="stats" id="stats"></div>

<script>
let tracks = [];
let cur = 0;

async function loadList() {
  const r = await fetch('/api/list');
  const data = await r.json();
  tracks = data.tracks;
  if (tracks.length === 0) {
    document.getElementById('progress').textContent = 'No tracks found';
    return;
  }
  const idx = tracks.findIndex(t => !t.label);
  cur = idx >= 0 ? idx : 0;
  show();
  updateStats();
}

function show() {
  if (tracks.length === 0) return;
  const t = tracks[cur];
  const strip = document.getElementById('strip');
  strip.innerHTML = '';
  for (const i of t.thumb_indices) {
    const img = document.createElement('img');
    img.src = `/thumb/${encodeURIComponent(t.run)}/${t.tid}/${i}`;
    img.alt = '';
    strip.appendChild(img);
  }
  let lblHTML = '';
  if (t.label) {
    const cls = t.label === 'REF' ? 'REF' : t.label;
    lblHTML = `<span class="label-current label-${cls}">${t.label}</span>`;
  }
  document.getElementById('progress').innerHTML =
    'Track ' + (cur + 1) + ' / ' + tracks.length + lblHTML;
  document.getElementById('header').textContent =
    `${t.run} · track ${t.tid} · ${t.n_dets} detections, ${t.n_thumbs} thumbnails`;
}

async function setLabel(label) {
  const t = tracks[cur];
  await fetch('/api/annotate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({run: t.run, tid: t.tid, label: label}),
  });
  t.label = label;
  if (cur < tracks.length - 1) { cur++; show(); }
  else { document.getElementById('progress').textContent = 'Done!'; }
  updateStats();
}

function updateStats() {
  const total = tracks.length;
  const counts = {A: 0, B: 0, REF: 0, X: 0};
  let done = 0;
  for (const t of tracks) {
    if (!t.label) continue;
    done++;
    counts[t.label] = (counts[t.label] || 0) + 1;
  }
  const pct = total > 0 ? Math.round(100 * done / total) : 0;
  document.getElementById('stats').innerHTML =
    '<b>Progress:</b> ' + done + ' / ' + total + ' (' + pct + '%) — ' +
    '<b>A:</b> ' + counts.A + ' — <b>B:</b> ' + counts.B + ' — ' +
    '<b>Ref:</b> ' + counts.REF + ' — <b>X:</b> ' + counts.X;
}

document.getElementById('btnA').onclick   = () => setLabel('A');
document.getElementById('btnB').onclick   = () => setLabel('B');
document.getElementById('btnRef').onclick = () => setLabel('REF');
document.getElementById('btnX').onclick   = () => setLabel('X');
document.getElementById('btnSkip').onclick = () => {
  if (cur < tracks.length - 1) { cur++; show(); }
};
document.getElementById('btnPrev').onclick = () => {
  if (cur > 0) { cur--; show(); }
};
document.getElementById('btnNext').onclick = () => {
  if (cur < tracks.length - 1) { cur++; show(); }
};

document.addEventListener('keydown', e => {
  const k = e.key.toLowerCase();
  if (k === 'a') { e.preventDefault(); setLabel('A'); }
  else if (k === 'b') { e.preventDefault(); setLabel('B'); }
  else if (k === 'r') { e.preventDefault(); setLabel('REF'); }
  else if (k === 'x') { e.preventDefault(); setLabel('X'); }
  else if (k === 's') { e.preventDefault(); if (cur < tracks.length - 1) { cur++; show(); } }
  else if (e.key === 'ArrowLeft')  { e.preventDefault(); if (cur > 0) { cur--; show(); } }
  else if (e.key === 'ArrowRight') { e.preventDefault(); if (cur < tracks.length - 1) { cur++; show(); } }
});

loadList();
</script>
</body>
</html>
"""


def stream_frame_indices(video_path, indices):
    """Yield (frame_index, BGR ndarray) for the requested indices via a
    single linear scan (much faster than seek-per-frame on long clips)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    needed = set(indices)
    max_needed = max(needed) if needed else -1
    current = 0
    while current <= max_needed:
        ok, frame = cap.read()
        if not ok:
            break
        if current in needed:
            yield current, frame
        current += 1
    cap.release()


def build_thumbnails(run_dir: Path, video_path: Path, out_root: Path):
    """For every player track in run_dir/p1_a_detections.json, dump up to
    THUMBS_PER_TRACK thumbnail BGR PNGs sampled from the highest-conf
    detections. Thumbnails are cached on disk so subsequent UI loads
    don't re-decode the video."""
    det_path = run_dir / "p1_a_detections.json"
    if not det_path.exists():
        print(f"  WARN: {det_path} missing — skipping {run_dir.name}")
        return {}

    detections = json.loads(det_path.read_text())
    by_tid = defaultdict(list)  # tid → [(frame, xyxy, conf)]
    for fr in detections["frames"]:
        for b in fr["boxes"]:
            if b["class_id"] == PERSON_CLASS and b["track_id"] >= 0:
                by_tid[b["track_id"]].append((fr["frame"], b["xyxy"], b["conf"]))

    out_root.mkdir(parents=True, exist_ok=True)
    work = defaultdict(list)  # frame_idx → [(tid, idx, xyxy)]
    track_thumb_indices = {}
    for tid, dets in by_tid.items():
        dets = sorted(dets, key=lambda d: -d[2])[:THUMBS_PER_TRACK]
        track_thumb_indices[tid] = list(range(len(dets)))
        for idx, (fi, xyxy, _) in enumerate(dets):
            thumb_path = out_root / f"{tid}_{idx}.png"
            if not thumb_path.exists():
                work[fi].append((tid, idx, xyxy))

    if work:
        print(f"  decoding {sum(len(v) for v in work.values())} thumbnails "
              f"from {len(work)} unique frames…")
        needed = sorted(work.keys())
        for fi, frame in stream_frame_indices(video_path, needed):
            for tid, idx, xyxy in work[fi]:
                x1, y1, x2, y2 = (int(v) for v in xyxy)
                h, w = frame.shape[:2]
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)
                if x2 <= x1 + 4 or y2 <= y1 + 4:
                    continue
                crop = frame[y1:y2, x1:x2]
                # Letterbox-resize to fixed THUMB_W × THUMB_H so the UI
                # strip aligns; preserves aspect by padding rather than
                # stretching jersey colour.
                ar = crop.shape[1] / max(crop.shape[0], 1)
                target_ar = THUMB_W / THUMB_H
                if ar > target_ar:
                    new_w, new_h = THUMB_W, int(THUMB_W / ar)
                else:
                    new_w, new_h = int(THUMB_H * ar), THUMB_H
                resized = cv2.resize(crop, (max(new_w, 1), max(new_h, 1)))
                tile = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
                yy, xx = (THUMB_H - resized.shape[0]) // 2, (THUMB_W - resized.shape[1]) // 2
                tile[yy:yy + resized.shape[0], xx:xx + resized.shape[1]] = resized
                cv2.imwrite(str(out_root / f"{tid}_{idx}.png"), tile)

    return {
        tid: {
            "n_dets": len(by_tid[tid]),
            "thumb_indices": track_thumb_indices[tid],
            "n_thumbs": len(track_thumb_indices[tid]),
        }
        for tid in by_tid
    }


class State:
    """Server-side annotation state — one instance covers ALL run folders
    passed on the CLI. The annotation file is keyed by `<run>/<tid>` so
    the same run can be reopened and resumed."""

    def __init__(self, run_dirs, video_root, output, thumbs_dir):
        self.run_dirs = run_dirs
        self.video_root = video_root
        self.output = output
        self.thumbs_dir = thumbs_dir
        self.lock = threading.Lock()
        self.annotations = {}
        if output.exists():
            data = json.loads(output.read_text())
            self.annotations = data.get("tracks", {})

        # Build the catalogue: walk each run, decode thumbnails, pin them
        # into self.thumbs_dir/<run>/<tid>_<idx>.png. The list returned
        # by /api/list joins this catalogue with the saved annotations.
        self.tracks = []
        for run in run_dirs:
            run_name = run.name
            video_match = self._guess_video(run)
            if video_match is None:
                print(f"  WARN: can't locate source video for {run_name} — "
                      f"skipping; place it under {video_root}/ first.")
                continue
            print(f"\nIndexing {run_name} (video {video_match.name})")
            tinfo = build_thumbnails(
                run, video_match, thumbs_dir / run_name,
            )
            for tid in sorted(tinfo.keys()):
                key = f"{run_name}/{tid}"
                meta = self.annotations.get(key, {})
                self.tracks.append({
                    "run": run_name,
                    "tid": tid,
                    "n_dets": tinfo[tid]["n_dets"],
                    "n_thumbs": tinfo[tid]["n_thumbs"],
                    "thumb_indices": tinfo[tid]["thumb_indices"],
                    "label": self._meta_to_label(meta),
                })

    @staticmethod
    def _meta_to_label(meta):
        if not meta:
            return ""
        if meta.get("is_referee"):
            return "REF"
        if meta.get("team") in ("A", "B"):
            return meta["team"]
        if meta.get("team") is None and meta.get("is_referee") is False \
                and "team" in meta:
            return "X"
        return ""

    def _guess_video(self, run_dir: Path):
        """Find the source mp4 referenced by p1_a_detections.json. The
        field is `video` in the canonical schema (older runs may have
        used `source_video`). Falls back to a name match in
        `video_root` if the absolute path in the JSON has moved."""
        det = json.loads((run_dir / "p1_a_detections.json").read_text())
        sv = det.get("video") or det.get("source_video") or ""
        p = Path(sv)
        if p.exists() and p.is_file():
            return p
        # Try matching by basename inside videos/
        if p.name:
            candidates = list(self.video_root.glob(f"*{p.name}*"))
            if candidates:
                return candidates[0]
        return None

    def list_tracks(self):
        # Refresh labels from the persisted annotations on every list
        # call, so multi-tab consistency holds.
        for t in self.tracks:
            key = f"{t['run']}/{t['tid']}"
            t["label"] = self._meta_to_label(self.annotations.get(key, {}))
        return self.tracks

    def annotate(self, run, tid, label):
        key = f"{run}/{tid}"
        if label == "A":
            meta = {"team": "A", "is_referee": False}
        elif label == "B":
            meta = {"team": "B", "is_referee": False}
        elif label == "REF":
            meta = {"team": None, "is_referee": True}
        elif label == "X":
            meta = {"team": None, "is_referee": False}
        else:
            return
        with self.lock:
            self.annotations[key] = meta
            self._save()

    def _save(self):
        meta = {
            "n_total": len(self.annotations),
            "n_a": sum(1 for v in self.annotations.values() if v.get("team") == "A"),
            "n_b": sum(1 for v in self.annotations.values() if v.get("team") == "B"),
            "n_referee": sum(1 for v in self.annotations.values() if v.get("is_referee")),
            "n_other": sum(1 for v in self.annotations.values()
                           if v.get("team") is None and not v.get("is_referee")),
            "produced_by": "tools/annotate_tracks.py",
        }
        payload = {"tracks": self.annotations, "metadata": meta}
        tmp = self.output.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(self.output)


class Handler(BaseHTTPRequestHandler):
    state = None

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self._respond(200, "text/html; charset=utf-8", HTML.encode("utf-8"))
        elif url.path == "/api/list":
            payload = json.dumps({"tracks": self.state.list_tracks()}).encode()
            self._respond(200, "application/json", payload)
        elif url.path.startswith("/thumb/"):
            # /thumb/<run>/<tid>/<idx>
            parts = url.path[len("/thumb/"):].split("/")
            if len(parts) != 3:
                self.send_error(404); return
            run = unquote(parts[0])
            tid = parts[1]
            idx = parts[2]
            f = (self.state.thumbs_dir / run / f"{tid}_{idx}.png").resolve()
            try:
                f.relative_to(self.state.thumbs_dir.resolve())
            except ValueError:
                self.send_error(403); return
            if not f.exists():
                self.send_error(404); return
            self._respond(200, "image/png", f.read_bytes(),
                          extra=[("Cache-Control", "max-age=3600")])
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/annotate":
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            self.state.annotate(data["run"], int(data["tid"]), data["label"])
            self.send_response(204); self.end_headers()
        else:
            self.send_error(404)

    def _respond(self, code, ctype, body, extra=()):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def main():
    """CLI entry point — parse run folders + start the localhost server."""
    p = argparse.ArgumentParser(
        description="Track-level truth annotation for the team-engine bench")
    p.add_argument("runs", nargs="+", type=str,
                   help="One or more run folders (each contains p1_a_detections.json)")
    p.add_argument("--video-root", default="videos", type=str,
                   help="Where to look up the source mp4 if its path in "
                        "p1_a_detections.json no longer resolves.")
    p.add_argument("--output", default="data/jersey_numbers/track_truth.json",
                   type=str, help="Annotation JSON output (incremental).")
    p.add_argument("--thumbs-dir", default="data/jersey_numbers/_track_thumbs",
                   type=str,
                   help="Where decoded thumbnails are cached. Reusable across "
                        "annotation sessions.")
    p.add_argument("--port", type=int, default=8001)
    args = p.parse_args()

    run_dirs = [Path(r).resolve() for r in args.runs]
    for r in run_dirs:
        if not r.exists():
            raise SystemExit(f"Run folder not found: {r}")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    thumbs_dir = Path(args.thumbs_dir).resolve()

    Handler.state = State(run_dirs, Path(args.video_root).resolve(),
                          output, thumbs_dir)
    n_tracks = len(Handler.state.tracks)
    n_done = sum(1 for t in Handler.state.tracks if t["label"])
    print(f"\n{n_tracks} tracks indexed across {len(run_dirs)} runs "
          f"({n_done} already labelled)")
    print(f"Annotations: {output}")
    print(f"Thumbnails:  {thumbs_dir}")
    print(f"\n→ Open http://localhost:{args.port} in your browser")
    print("  Ctrl+C to stop\n")

    httpd = HTTPServer(("127.0.0.1", args.port), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
