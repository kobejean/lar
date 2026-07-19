"""BEV relabeler — hand-correct footprint2d output + place semantic ground-contact keypoints.

`footprint2d.py` fuses per-frame seg+depth into a BEV raster of ground states
(FREE / FOOTPRINT / HIDDEN / UNKNOWN) plus a per-cell obstacle class. It gets the broad
structure right and the *edges* wrong: footprint boundaries bleed where the bottom-most
obstacle pixel is a shadow or a leaf, HIDDEN over/under-claims, and thin structures
(fences, hedges) come out speckled.

Fixing that in image space would mean painting 970 frames. Fixing it **in BEV means
painting once** — the park is one 201x230 grid — and every frame that sees a cell inherits
the correction when the raster is rendered back into that camera (the same DEM->camera
rasterisation `footprint_labels.py` already does). That asymmetry is the whole point of
editing here rather than per-frame.

Two annotation products come out:

  1. **corrected rasters** — `state` (ground state) and `structure` (per-cell class),
     same grid, same dtype as the input npz. Dense supervision for a ground/footprint head.
  2. **semantic ground-contact keypoints** — sparse (class, u, v) points marking where a
     discrete object actually meets the ground. This is the signal `base_points.py`
     triangulates for; hand-placed points are the ground truth to *score* it against, and
     the target for a contact-point head on frozen LingBot features.

Run (from repo root):
  uv run --extra segmentation python script/backbone/relabel.py \
      --npz output/maguro-park-after-itchy-footprint2d-mono2/footprint2d.npz

then open the printed URL. Edits save next to the npz as `relabel.npz` + `keypoints.json`;
the original is never overwritten. Re-running picks up where you left off.

GRID REGISTRATION CAVEAT: footprint2d's npz stores `cell_size` but not the DEM origin, so
the edits are registered to *that grid*, not to world coordinates. Consumers must rebuild
the GroundField with the same session/cell-size/dem-source (deterministic) and apply the
edits cell-wise. Adding `origin_u`/`origin_v` to footprint2d's `np.savez` would make this
self-describing; keypoints then carry true world (u, v).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SCRIPT_ROOT = _HERE.parent
sys.path.insert(0, str(_SCRIPT_ROOT))
sys.path.insert(0, str(_SCRIPT_ROOT / "semantic_bev"))

from taxonomy import Klass, color_lut  # noqa: E402

UNKNOWN, FREE, FOOTPRINT, HIDDEN = 0, 1, 2, 3
STATE_NAMES = ["UNKNOWN", "FREE", "FOOTPRINT", "HIDDEN"]
STATE_COLORS = [(35, 35, 35), (60, 170, 60), (220, 40, 40), (230, 200, 60)]  # RGB


def hillshade(dem: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """Grey relief of the DEM so the editor has real terrain context under the labels.

    Uncovered cells (no DEM support) render near-black, which is itself information: it
    shows where the raster is extrapolated guesswork rather than observed ground.
    """
    d = np.where(np.isfinite(dem), dem, np.nan)
    gy, gx = np.gradient(np.nan_to_num(d, nan=float(np.nanmedian(d)) if np.isfinite(d).any() else 0.0))
    shade = np.clip(0.5 + 1.6 * (gx + gy), 0.0, 1.0)
    lo, hi = (np.nanpercentile(d, [2, 98]) if np.isfinite(d).any() else (0.0, 1.0))
    height = np.clip((np.nan_to_num(d, nan=lo) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    v = (0.35 + 0.45 * height) * (0.55 + 0.45 * shade)
    v = np.where(coverage, v, v * 0.25)
    g = (np.clip(v, 0, 1) * 255).astype(np.uint8)
    return np.dstack([g, g, g])


def png_b64(rgb: np.ndarray) -> str:
    """Encode HxWx3 uint8 to a data URI. Prefers cv2, falls back to a raw-zlib PNG writer."""
    try:
        import cv2

        ok, buf = cv2.imencode(".png", rgb[:, :, ::-1])
        if ok:
            return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()
    except Exception:
        pass
    import struct
    import zlib

    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


class Store:
    """Holds the grids and owns persistence. Original npz is read-only."""

    def __init__(self, npz_path: Path, out_dir: Path | None):
        self.npz_path = npz_path
        z = np.load(npz_path)
        self.state0 = z["state"].astype(np.uint8)
        self.struct0 = z["structure"].astype(np.uint8)
        self.dem = z["dem"].astype(np.float32)
        self.coverage = z["coverage"].astype(bool)
        self.cell_size = float(z["cell_size"])
        self.free_count = z["free_count"] if "free_count" in z else np.zeros_like(self.state0, np.int32)
        self.foot_count = z["foot_count"] if "foot_count" in z else np.zeros_like(self.state0, np.int32)
        self.rows, self.cols = self.state0.shape

        self.out_dir = out_dir or npz_path.parent
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.relabel_npz = self.out_dir / "relabel.npz"
        self.keypoints_json = self.out_dir / "keypoints.json"

        # resume a previous session if present
        self.state = self.state0.copy()
        self.structure = self.struct0.copy()
        self.keypoints: list[dict] = []
        if self.relabel_npz.exists():
            r = np.load(self.relabel_npz)
            if r["state"].shape == self.state0.shape:
                self.state = r["state"].astype(np.uint8)
                self.structure = r["structure"].astype(np.uint8)
                print(f"[relabel] resumed rasters from {self.relabel_npz}")
        if self.keypoints_json.exists():
            self.keypoints = json.loads(self.keypoints_json.read_text()).get("keypoints", [])
            print(f"[relabel] resumed {len(self.keypoints)} keypoints from {self.keypoints_json}")

    def save(self, state, structure, keypoints):
        self.state = np.asarray(state, np.uint8).reshape(self.rows, self.cols)
        self.structure = np.asarray(structure, np.uint8).reshape(self.rows, self.cols)
        self.keypoints = keypoints
        edited = int((self.state != self.state0).sum() + (self.structure != self.struct0).sum())
        np.savez(self.relabel_npz,
                 state=self.state, structure=self.structure,
                 state_orig=self.state0, structure_orig=self.struct0,
                 dem=self.dem, coverage=self.coverage, cell_size=self.cell_size,
                 edited_cells=edited)
        self.keypoints_json.write_text(json.dumps({
            "source_npz": str(self.npz_path),
            "cell_size": self.cell_size,
            "grid": {"rows": self.rows, "cols": self.cols},
            "coords": "cell indices (col, row) in the source grid; +row = +v (north)",
            "classes": {int(k): k.name for k in Klass},
            "keypoints": keypoints,
        }, indent=2))
        return {"edited_cells": edited, "keypoints": len(keypoints),
                "npz": str(self.relabel_npz), "json": str(self.keypoints_json)}


PAGE = r"""<!doctype html><html><head><meta charset=utf-8><title>BEV relabeler</title>
<style>
 :root{--bg:#12141a;--panel:#1b1e26;--ink:#e6e8ee;--mut:#8b93a7;--line:#2b2f3a;--acc:#4da3ff}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);
   font:13px/1.45 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;display:flex;height:100vh;overflow:hidden}
 #side{width:260px;flex:none;background:var(--panel);border-right:1px solid var(--line);
   padding:14px;overflow-y:auto}
 #stage{flex:1;position:relative;overflow:hidden;background:#0b0d11}
 canvas{position:absolute;top:0;left:0;image-rendering:pixelated;cursor:crosshair}
 h1{font-size:14px;margin:0 0 4px} .sub{color:var(--mut);font-size:11px;margin-bottom:14px}
 h2{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--mut);
   margin:16px 0 7px;padding-bottom:5px;border-bottom:1px solid var(--line)}
 .row{display:flex;gap:5px;flex-wrap:wrap}
 button{background:#252a35;color:var(--ink);border:1px solid var(--line);border-radius:6px;
   padding:6px 9px;font:inherit;font-size:12px;cursor:pointer}
 button:hover{border-color:var(--acc)} button.on{background:var(--acc);border-color:var(--acc);color:#08121e;font-weight:600}
 .sw{display:flex;align-items:center;gap:7px;width:100%;padding:5px 7px;border-radius:6px;
   border:1px solid transparent;cursor:pointer}
 .sw:hover{background:#232833} .sw.on{border-color:var(--acc);background:#232833}
 .chip{width:13px;height:13px;border-radius:3px;flex:none;border:1px solid #0006}
 label{display:flex;align-items:center;gap:7px;margin:5px 0;color:var(--mut);cursor:pointer}
 input[type=range]{width:100%} kbd{background:#252a35;border:1px solid var(--line);
   border-radius:3px;padding:0 4px;font-size:11px}
 #status{position:absolute;left:10px;bottom:10px;background:#0d0f14dd;border:1px solid var(--line);
   border-radius:6px;padding:6px 10px;font-size:11px;color:var(--mut);pointer-events:none}
 #save{width:100%;background:#1f6f43;border-color:#2b8c57;font-weight:600;padding:9px}
 #save:hover{background:#268051}
</style></head><body>
<div id=side>
  <h1>BEV relabeler</h1><div class=sub id=meta></div>

  <h2>Tool</h2>
  <div class=row>
    <button class=tool data-t=state>Ground</button>
    <button class=tool data-t=struct>Class</button>
    <button class=tool data-t=kp>Keypoint</button>
  </div>

  <h2 id=palhdr>Ground state</h2>
  <div id=pal></div>

  <h2>Brush</h2>
  <input type=range id=brush min=0 max=12 value=1>
  <div class=sub id=brushlbl style=margin:0></div>

  <h2>Layers</h2>
  <label><input type=checkbox id=lbg checked> Terrain relief</label>
  <label><input type=checkbox id=lstate checked> Ground state</label>
  <label><input type=checkbox id=lstruct> Class overlay</label>
  <label><input type=checkbox id=lkp checked> Keypoints</label>
  <label><input type=checkbox id=ldiff> Highlight my edits</label>
  <label>Opacity <input type=range id=alpha min=10 max=100 value=70></label>

  <h2>Session</h2>
  <div class=row>
    <button id=undo>Undo <kbd>Z</kbd></button>
    <button id=reset>Revert all</button>
  </div>
  <div style=height:9px></div>
  <button id=save>Save annotations</button>
  <div class=sub id=saved style=margin:8px_0_0></div>

  <h2>Keys</h2>
  <div class=sub>
    <kbd>1</kbd>/<kbd>2</kbd>/<kbd>3</kbd> tool &nbsp; <kbd>Z</kbd> undo<br>
    <kbd>[</kbd><kbd>]</kbd> brush &nbsp; drag=paint<br>
    wheel=zoom &nbsp; <kbd>space</kbd>+drag or middle=pan<br>
    keypoint: click=add, <kbd>shift</kbd>+click=delete
  </div>
</div>
<div id=stage><canvas id=cv></canvas><div id=status></div></div>
<script>
const D = __DATA__;
const R = D.rows, C = D.cols;
const b64 = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
let state = b64(D.state), struct = b64(D.struct);
const state0 = b64(D.state), struct0 = b64(D.struct0);
let kps = D.keypoints.slice();

const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const stage = document.getElementById('stage');
const bg = new Image(); bg.src = D.bg;
// offscreen at grid resolution; scaled up with smoothing off => crisp cells
const off = document.createElement('canvas'); off.width = C; off.height = R;
const octx = off.getContext('2d'), oimg = octx.createImageData(C, R);

let tool = 'state', paint = 1, brush = 1, undoStack = [], stroke = null;
let view = {x:0, y:0, s:4}, panning = false, spaceDown = false, last = null;

// row 0 of the array is min-v (south); display flipped so north is up, matching
// footprint2d's own BEV render. All click math inverts through the same flip.
const cellAt = (px, py) => {
  const gx = Math.floor((px - view.x) / view.s), gy = Math.floor((py - view.y) / view.s);
  return (gx < 0 || gy < 0 || gx >= C || gy >= R) ? null : {c: gx, r: R - 1 - gy};
};

function draw() {
  const a = +document.getElementById('alpha').value / 100;
  const showState = document.getElementById('lstate').checked;
  const showStruct = document.getElementById('lstruct').checked;
  const showDiff = document.getElementById('ldiff').checked;
  const p = oimg.data;
  for (let r = 0; r < R; r++) {
    const dy = R - 1 - r;                       // flip for north-up display
    for (let c = 0; c < C; c++) {
      const i = r * C + c, o = (dy * C + c) * 4;
      let col = null;
      if (showStruct && struct[i]) col = D.klassColors[struct[i]];
      if (!col && showState && state[i] !== 0) col = D.stateColors[state[i]];
      if (showDiff && (state[i] !== state0[i] || struct[i] !== struct0[i])) col = [255, 0, 255];
      if (col) { p[o] = col[0]; p[o+1] = col[1]; p[o+2] = col[2]; p[o+3] = Math.round(255*a); }
      else p[o+3] = 0;
    }
  }
  octx.putImageData(oimg, 0, 0);

  cv.width = stage.clientWidth; cv.height = stage.clientHeight;
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (document.getElementById('lbg').checked && bg.complete)
    ctx.drawImage(bg, view.x, view.y, C*view.s, R*view.s);
  ctx.drawImage(off, view.x, view.y, C*view.s, R*view.s);

  if (document.getElementById('lkp').checked) {
    for (const k of kps) {
      const x = view.x + (k.col + 0.5)*view.s, y = view.y + (R - 1 - k.row + 0.5)*view.s;
      const col = D.klassColors[k.klass] || [255,255,255];
      ctx.beginPath(); ctx.arc(x, y, 5, 0, 7);
      ctx.fillStyle = `rgb(${col})`; ctx.fill();
      ctx.lineWidth = 1.5; ctx.strokeStyle = '#000'; ctx.stroke();
    }
  }
  document.getElementById('status').textContent =
    `${tool} · ${C}x${R} @ ${D.cell}m · ${kps.length} keypoints · zoom ${view.s.toFixed(1)}x`;
}

function apply(cell) {
  if (!cell) return;
  const rad = brush;
  for (let dr = -rad; dr <= rad; dr++) for (let dc = -rad; dc <= rad; dc++) {
    if (dr*dr + dc*dc > rad*rad + rad) continue;
    const r = cell.r + dr, c = cell.c + dc;
    if (r < 0 || c < 0 || r >= R || c >= C) continue;
    const i = r*C + c;
    const arr = tool === 'struct' ? struct : state;
    if (arr[i] === paint) continue;
    stroke.push([tool === 'struct' ? 1 : 0, i, arr[i]]);   // record prior value for undo
    arr[i] = paint;
  }
}

cv.addEventListener('mousedown', e => {
  if (e.button === 1 || spaceDown) { panning = true; last = [e.clientX, e.clientY]; return; }
  const cell = cellAt(e.offsetX, e.offsetY); if (!cell) return;
  if (tool === 'kp') {
    if (e.shiftKey) {
      let bi = -1, bd = 1e9;
      kps.forEach((k, i) => { const d = (k.col-cell.c)**2 + (k.row-cell.r)**2; if (d < bd) { bd = d; bi = i; } });
      if (bi >= 0 && bd < 64) { undoStack.push({kp: kps.slice()}); kps.splice(bi, 1); }
    } else {
      undoStack.push({kp: kps.slice()});
      kps.push({klass: paint, name: D.klassNames[paint], col: cell.c, row: cell.r,
                u: +( (cell.c + 0.5) * D.cell).toFixed(3), v: +((cell.r + 0.5) * D.cell).toFixed(3)});
    }
    draw(); return;
  }
  stroke = []; apply(cell); draw();
});
cv.addEventListener('mousemove', e => {
  if (panning) { view.x += e.clientX - last[0]; view.y += e.clientY - last[1];
                 last = [e.clientX, e.clientY]; draw(); return; }
  if (stroke) { apply(cellAt(e.offsetX, e.offsetY)); draw(); }
});
window.addEventListener('mouseup', () => {
  panning = false;
  if (stroke && stroke.length) undoStack.push({cells: stroke});
  stroke = null;
});
cv.addEventListener('contextmenu', e => e.preventDefault());
cv.addEventListener('wheel', e => {
  e.preventDefault();
  const f = e.deltaY < 0 ? 1.15 : 1/1.15, ns = Math.max(1, Math.min(40, view.s*f));
  view.x = e.offsetX - (e.offsetX - view.x) * (ns/view.s);
  view.y = e.offsetY - (e.offsetY - view.y) * (ns/view.s);
  view.s = ns; draw();
}, {passive:false});

function undo() {
  const u = undoStack.pop(); if (!u) return;
  if (u.kp) kps = u.kp;
  else for (let i = u.cells.length - 1; i >= 0; i--) {
    const [which, idx, prev] = u.cells[i];
    (which ? struct : state)[idx] = prev;
  }
  draw();
}

// ---- palettes -------------------------------------------------------------
function buildPalette() {
  const pal = document.getElementById('pal');
  const hdr = document.getElementById('palhdr');
  pal.innerHTML = '';
  const items = tool === 'state'
    ? D.stateNames.map((n, i) => [i, n, D.stateColors[i]])
    : D.klassNames.map((n, i) => [i, n, D.klassColors[i]]);
  hdr.textContent = tool === 'state' ? 'Ground state' : (tool === 'kp' ? 'Keypoint class' : 'Object class');
  if (!items.some(it => it[0] === paint)) paint = items[0][0];
  for (const [v, n, col] of items) {
    const d = document.createElement('div');
    d.className = 'sw' + (v === paint ? ' on' : '');
    d.innerHTML = `<span class=chip style="background:rgb(${col})"></span>${n}`;
    d.onclick = () => { paint = v; buildPalette(); draw(); };
    pal.appendChild(d);
  }
}
function setTool(t) {
  tool = t;
  document.querySelectorAll('.tool').forEach(b => b.classList.toggle('on', b.dataset.t === t));
  buildPalette(); draw();
}
document.querySelectorAll('.tool').forEach(b => b.onclick = () => setTool(b.dataset.t));

document.getElementById('brush').oninput = e => {
  brush = +e.target.value;
  document.getElementById('brushlbl').textContent =
    `radius ${brush} cell${brush===1?'':'s'} ≈ ${((2*brush+1)*D.cell).toFixed(1)} m across`;
  draw();
};
['lbg','lstate','lstruct','lkp','ldiff','alpha'].forEach(id =>
  document.getElementById(id).oninput = draw);
document.getElementById('undo').onclick = undo;
document.getElementById('reset').onclick = () => {
  if (!confirm('Discard all edits and keypoints, back to the footprint2d output?')) return;
  state = state0.slice(); struct = struct0.slice(); kps = []; undoStack = []; draw();
};
document.getElementById('save').onclick = async () => {
  const btn = document.getElementById('save'); btn.textContent = 'Saving…'; btn.disabled = true;
  const b = a => btoa(String.fromCharCode(...a));
  const res = await fetch('/save', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({state: b(state), structure: b(struct), keypoints: kps})});
  const j = await res.json();
  document.getElementById('saved').textContent =
    `saved ${j.edited_cells} edited cells, ${j.keypoints} keypoints`;
  btn.textContent = 'Save annotations'; btn.disabled = false;
};
window.addEventListener('keydown', e => {
  if (e.code === 'Space') { spaceDown = true; e.preventDefault(); }
  if (e.key === 'z') undo();
  if (e.key === '1') setTool('state');
  if (e.key === '2') setTool('struct');
  if (e.key === '3') setTool('kp');
  if (e.key === '[') { brush = Math.max(0, brush-1); document.getElementById('brush').value = brush; draw(); }
  if (e.key === ']') { brush = Math.min(12, brush+1); document.getElementById('brush').value = brush; draw(); }
});
window.addEventListener('keyup', e => { if (e.code === 'Space') spaceDown = false; });
window.addEventListener('resize', draw);

document.getElementById('meta').textContent = D.meta;
document.getElementById('brush').dispatchEvent(new Event('input'));
// fit the grid to the viewport on open
view.s = Math.max(1, Math.min(stage.clientWidth/C, stage.clientHeight/R) * 0.92);
view.x = (stage.clientWidth - C*view.s)/2; view.y = (stage.clientHeight - R*view.s)/2;
setTool('state');
bg.onload = draw;
</script></body></html>
"""


def build_page(store: Store) -> str:
    lut = color_lut()
    klass_names = [k.name for k in sorted(Klass, key=int)]
    klass_colors = [list(lut[int(k)]) if int(k) < len(lut) else [200, 200, 200]
                    for k in sorted(Klass, key=int)]
    data = {
        "rows": store.rows, "cols": store.cols, "cell": store.cell_size,
        "state": base64.b64encode(store.state.tobytes()).decode(),
        "struct": base64.b64encode(store.structure.tobytes()).decode(),
        "struct0": base64.b64encode(store.struct0.tobytes()).decode(),
        "bg": png_b64(hillshade(store.dem, store.coverage)),
        "stateNames": STATE_NAMES, "stateColors": [list(c) for c in STATE_COLORS],
        "klassNames": klass_names, "klassColors": klass_colors,
        "keypoints": store.keypoints,
        "meta": (f"{store.npz_path.parent.name} · {store.cols}×{store.rows} cells @ "
                 f"{store.cell_size} m · {store.coverage.mean()*100:.0f}% DEM coverage"),
    }
    return PAGE.replace("__DATA__", json.dumps(data))


def serve(store: Store, port: int, open_browser: bool):
    page = build_page(store).encode()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype="application/json"):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(page, "text/html; charset=utf-8")
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/save":
                return self.send_error(404)
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            info = store.save(np.frombuffer(base64.b64decode(req["state"]), np.uint8),
                              np.frombuffer(base64.b64decode(req["structure"]), np.uint8),
                              req["keypoints"])
            print(f"[relabel] saved: {info['edited_cells']} edited cells, "
                  f"{info['keypoints']} keypoints -> {info['npz']}")
            self._send(json.dumps(info).encode())

    url = f"http://127.0.0.1:{port}/"
    print(f"[relabel] {store.npz_path}\n[relabel] grid {store.cols}x{store.rows} @ "
          f"{store.cell_size} m\n[relabel] serving {url}  (ctrl-c to stop)")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    HTTPServer(("127.0.0.1", port), H).serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--npz", required=True, help="footprint2d.npz to correct")
    ap.add_argument("--out", default=None, help="where to write relabel.npz/keypoints.json "
                                                "(default: alongside --npz)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    npz = Path(args.npz).expanduser().resolve()
    if not npz.exists():
        raise SystemExit(f"no such npz: {npz}")
    store = Store(npz, Path(args.out).expanduser().resolve() if args.out else None)
    try:
        serve(store, args.port, not args.no_browser)
    except KeyboardInterrupt:
        print("\n[relabel] stopped")


if __name__ == "__main__":
    main()
