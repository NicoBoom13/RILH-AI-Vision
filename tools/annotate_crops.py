"""
RILH-AI-Vision — manual jersey-number annotation tool

Tiny localhost web app to label dorsal jersey crops collected during
Phase 6 identify (--debug-crops-dir). Pre-fills each crop's input with
TrOCR's guess (parsed from the filename) so the typical interaction is
just hitting Enter to confirm a correct guess.

Usage:
  python tools/annotate_crops.py runs/run19 [--port 8000]

Then open http://localhost:8000 in a browser.

Annotations save incrementally to <root>/annotations.json:
  {
    "annotations": {
      "video05/debug_crops/numbers/t0226_f00550_num-9_c73.png": "9",
      "clip60/debug_crops/numbers/t0001_f00010_num-X_c73.png":  "X",
      ...
    }
  }

Label conventions:
  "0"…"99"  visible jersey number
  "X"       no number visible (back not facing, occluded, blurred, ...)
  ""        unlabeled — comes back later

Keyboard:
  Enter   confirm + next
  X       fill "X" (no number)
  ←       prev
  →       next without saving
"""

import argparse
import json
import re
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse


HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>RILH crop annotation</title>
<style>
  body { font: 14px -apple-system, sans-serif; max-width: 900px; margin: 1em auto; padding: 0 1em; }
  h2 { margin: 0.5em 0; }
  .progress { color: #666; margin-bottom: 0.5em; font-variant-numeric: tabular-nums; }
  .filename { color: #aaa; font-size: 11px; word-break: break-all; }
  .crop-wrap { text-align: center; margin: 1em 0; min-height: 200px; }
  .crop {
    image-rendering: pixelated;
    border: 1px solid #ccc;
    background: #f8f8f8;
  }
  .controls { display: flex; gap: 0.5em; align-items: center; flex-wrap: wrap; }
  input[type=text] { font: 600 28px monospace; padding: 6px 12px; width: 110px; text-align: center; }
  button { font-size: 14px; padding: 6px 12px; cursor: pointer; }
  .hint { color: #888; font-size: 12px; margin-top: 0.5em; }
  kbd { background: #eee; border: 1px solid #ccc; border-radius: 3px; padding: 1px 5px; font-size: 11px; }
  .stats { margin-top: 2em; padding: 0.5em 1em; background: #f0f0f0; border-radius: 4px; font-size: 12px; }
</style>
</head>
<body>
  <h2>Jersey number annotation</h2>
  <div class="progress" id="progress">Loading…</div>
  <div class="filename" id="filename"></div>
  <div class="crop-wrap"><img id="crop" class="crop" alt="" /></div>
  <div class="controls">
    <label>Label: <input type="text" id="label" autofocus maxlength="3" /></label>
    <button id="save">Save &amp; Next</button>
    <button id="x">X (no number)</button>
    <button id="prev">← Prev</button>
    <button id="next">Next →</button>
  </div>
  <p class="hint">
    Type the visible jersey number, or "X" if no number is visible.
    <kbd>Enter</kbd> to save+next, <kbd>X</kbd>+<kbd>Enter</kbd> for no-number, <kbd>←</kbd>/<kbd>→</kbd> to navigate.
  </p>
  <div class="stats" id="stats"></div>

<script>
let items = [];
let cur = 0;

async function loadList() {
  const r = await fetch('/api/list');
  const data = await r.json();
  items = data.items;
  if (items.length === 0) {
    document.getElementById('progress').textContent = 'No crops found';
    return;
  }
  // Resume at first un-annotated
  const idx = items.findIndex(it => !it.label);
  cur = idx >= 0 ? idx : 0;
  show();
  updateStats();
}

function show() {
  if (items.length === 0) return;
  const it = items[cur];
  const img = document.getElementById('crop');
  img.src = '/crop/' + encodeURIComponent(it.path);
  img.onload = function() {
    // Upscale x4 for visibility, capped at 800px wide
    const w = Math.min(this.naturalWidth * 4, 800);
    this.style.width = w + 'px';
  };
  document.getElementById('label').value = it.label || it.hint || '';
  document.getElementById('label').focus();
  document.getElementById('label').select();
  document.getElementById('progress').textContent =
    'Crop ' + (cur+1) + ' / ' + items.length +
    (it.label ? '  [annotated: ' + it.label + ']' : it.hint ? '  [hint: ' + it.hint + ']' : '');
  document.getElementById('filename').textContent = it.path;
}

async function save() {
  const it = items[cur];
  const label = document.getElementById('label').value.trim().toUpperCase();
  if (!label) return;
  await fetch('/api/annotate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({path: it.path, label: label})
  });
  it.label = label;
  if (cur < items.length - 1) {
    cur++;
    show();
  } else {
    document.getElementById('progress').textContent = 'Done!';
  }
  updateStats();
}

function updateStats() {
  const total = items.length;
  const done = items.filter(it => it.label).length;
  const xs = items.filter(it => it.label === 'X').length;
  const nums = items.filter(it => it.label && it.label !== 'X').length;
  const pct = total > 0 ? Math.round(100 * done / total) : 0;
  document.getElementById('stats').innerHTML =
    '<b>Progress:</b> ' + done + ' / ' + total + ' (' + pct + '%) — ' +
    '<b>numbers:</b> ' + nums + ' — ' +
    '<b>X (no number):</b> ' + xs;
}

document.getElementById('save').onclick = save;
document.getElementById('x').onclick = () => {
  document.getElementById('label').value = 'X';
  save();
};
document.getElementById('prev').onclick = () => { if (cur > 0) { cur--; show(); } };
document.getElementById('next').onclick = () => { if (cur < items.length-1) { cur++; show(); } };

document.addEventListener('keydown', e => {
  if (e.key === 'Enter') {
    e.preventDefault();
    save();
  } else if (e.key === 'ArrowLeft' && document.activeElement.id !== 'label') {
    if (cur > 0) { cur--; show(); }
  } else if (e.key === 'ArrowRight' && document.activeElement.id !== 'label') {
    if (cur < items.length-1) { cur++; show(); }
  }
});

loadList();
</script>
</body>
</html>
"""


# Filename pattern: t{tid:04d}_f{fi:05d}_{kind}-{result}_c{conf:02d}.png
FNAME_RE = re.compile(r"t(\d+)_f(\d+)_(num|nam)-([^_]+)_c(\d+)\.png")


class State:
    def __init__(self, root: Path, output: Path):
        self.root = root
        self.output = output
        self.lock = threading.Lock()
        self.annotations = {}
        if output.exists():
            data = json.loads(output.read_text())
            self.annotations = data.get("annotations", {})
        self._items_cache = None

    def save(self):
        with self.lock:
            tmp = self.output.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(
                {"annotations": self.annotations}, indent=2, sort_keys=True))
            tmp.replace(self.output)

    def list_crops(self):
        if self._items_cache is not None:
            # Refresh labels from current annotations
            for it in self._items_cache:
                it["label"] = self.annotations.get(it["path"], "")
            return self._items_cache
        items = []
        for crop in sorted(self.root.glob("*/debug_crops/numbers/*.png")):
            rel = crop.relative_to(self.root).as_posix()
            m = FNAME_RE.match(crop.name)
            hint = m.group(4) if m else ""
            if hint == "X":
                hint = ""
            items.append({
                "path": rel,
                "hint": hint,
                "label": self.annotations.get(rel, ""),
            })
        self._items_cache = items
        return items

    def annotate(self, path: str, label: str):
        with self.lock:
            self.annotations[path] = label
        self.save()


class Handler(BaseHTTPRequestHandler):
    state = None  # set externally before starting the server

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/api/list":
            payload = json.dumps({"items": self.state.list_crops()}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif url.path.startswith("/crop/"):
            rel = unquote(url.path[len("/crop/"):])
            f = (self.state.root / rel).resolve()
            try:
                f.relative_to(self.state.root.resolve())
            except ValueError:
                self.send_error(403)
                return
            if not f.exists() or not f.is_file():
                self.send_error(404)
                return
            data = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/annotate":
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            self.state.annotate(data["path"], data["label"])
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # quiet


def main():
    p = argparse.ArgumentParser(description="Manual jersey-number annotation")
    p.add_argument("root", type=str,
                   help="Root dir containing <video>/debug_crops/numbers/*.png "
                        "(e.g. runs/run19)")
    p.add_argument("--output", default=None,
                   help="Annotations JSON path (default: <root>/annotations.json)")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output) if args.output else (root / "annotations.json")
    output = output.resolve()

    Handler.state = State(root, output)
    items = Handler.state.list_crops()
    n_done = sum(1 for it in items if it["label"])

    print(f"Crops root: {root}")
    print(f"Annotations: {output}")
    print(f"Crops found: {len(items)} ({n_done} already annotated)")
    print(f"\n→ Open http://localhost:{args.port} in your browser")
    print("  Ctrl+C to stop\n")

    httpd = HTTPServer(("127.0.0.1", args.port), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
