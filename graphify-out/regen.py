#!/usr/bin/env python3
"""
Regenerate graph3d.html from src/ + orchestration.json.

Usage (from project root):
    /Users/nico/.local/share/uv/tools/graphifyy/bin/python graphify-out/regen.py

What this does:
  1. Re-runs graphify detect + AST extraction on src/   (deterministic, free, fast)
  2. Loads graphify-out/orchestration.json              (extracted by an LLM subagent — see REGEN.md)
  3. Merges AST + orchestration into graph.json
  4. Builds the 3D pipeline visualization at graph3d.html

When to re-run the orchestration extraction:
  Only when README.md / CLAUDE.md / docs/phase_1_6_design.md / configs/*.yaml change in ways
  that affect the pipeline structure (new phase, new data file, new config). See REGEN.md.

When to re-run this script:
  Any time src/ changes — AST is fast and free, no LLM call.
"""
import json, sys
from pathlib import Path

ROOT = Path('/Users/nico/Documents/Claude/Projects/RILH-AI-Vision')
SRC = ROOT / 'src'
OUT = ROOT / 'graphify-out'

# ============================================================================
# 1. graphify AST extraction on src/
# ============================================================================
from graphify.detect import detect
from graphify.extract import collect_files, extract
from graphify.build import build_from_json
from graphify.cluster import cluster, score_all
from graphify.export import to_json

print('[1/4] Detecting + extracting AST from src/...')
det = detect(SRC)
code_files = []
for f in det.get('files', {}).get('code', []):
    code_files.extend(collect_files(Path(f)) if Path(f).is_dir() else [Path(f)])
ast = extract(code_files, cache_root=ROOT)
print(f'      AST: {len(ast["nodes"])} nodes, {len(ast["edges"])} edges')

# ============================================================================
# 2. Load orchestration layer (LLM-extracted, pre-saved)
# ============================================================================
ORCH_PATH = OUT / 'orchestration.json'
if not ORCH_PATH.exists():
    sys.exit(f'ERROR: {ORCH_PATH} not found. Re-extract via subagent — see REGEN.md.')
print(f'[2/4] Loading orchestration from {ORCH_PATH.name}...')
orch = json.loads(ORCH_PATH.read_text())
print(f'      Orchestration: {len(orch["nodes"])} nodes, {len(orch["edges"])} edges')

# ============================================================================
# 3. Merge AST + orchestration → graph.json
# ============================================================================
print('[3/4] Merging + clustering...')
seen = {n['id'] for n in ast['nodes']}
merged_nodes = list(ast['nodes'])
for n in orch['nodes']:
    if n['id'] not in seen:
        merged_nodes.append(n); seen.add(n['id'])
merged_edges = list(ast['edges'])
node_ids = {n['id'] for n in merged_nodes}
for e in orch['edges']:
    if e['source'] in node_ids and e['target'] in node_ids:
        merged_edges.append(e)

merged = {'nodes': merged_nodes, 'edges': merged_edges, 'hyperedges': [],
          'input_tokens': 0, 'output_tokens': 0}

G = build_from_json(merged)
communities = cluster(G)
to_json(G, communities, str(OUT / 'graph.json'))
print(f'      Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(communities)} communities')

# ============================================================================
# 4. Build graph3d.html (pinned-phase rail + orbital stations + volumetric haze)
# ============================================================================
print('[4/4] Building graph3d.html...')

g = json.loads((OUT / 'graph.json').read_text())

# Community labels — assigned manually based on inspection. If clustering shifts,
# the integer→name mapping will need to be re-checked.
COMMUNITY_LABELS = {
    0: 'Entity Re-ID Clustering', 1: 'Team Classification', 2: 'Video Annotation',
    3: 'Player Identification', 4: 'Rink Calibration & Blockers', 5: 'Follow-Cam Smoothing',
    6: 'OCR Engines (PARSeq / TrOCR)', 7: 'Detection & Tracker Configs',
    8: 'Pipeline Orchestration', 9: 'Jersey Crop Helpers', 10: 'Track→Jersey Merging',
    11: 'Jersey Number Filter', 12: 'Filename Sanitizer', 13: 'Name Text Filter',
    14: 'Letterbox Preprocessing', 15: 'OCR Batch Output', 16: 'Aspect Padding',
    17: 'Batch OCR Interface', 18: 'OCR Result Merger',
}

SPACING = 380       # horizontal gap between phases — breathing room around the rail
ZIGZAG = 55         # alternating Y offset by slot parity for compact zigzag layout
STAGE_VSPACING = 70 # vertical spacing between stages of the same phase
ORPHAN_GAP = 1.6 * SPACING   # empty space between Phase 7 and the orphan cluster

# All 7 project-level phases pinned on the rail (Phase 1-5 active in pipeline,
# Phase 6-7 external — same rail to keep the logical pipeline visible
# end-to-end, but rendered amber to make the visual difference obvious).
PHASE_ORDER = [
    # nid,                 short,      slot, kind ('active' = orchestrated, 'external' = consumes runs/)
    ('readme_p1',          'Phase 1',  -3,  'active'),
    ('readme_p2',          'Phase 2',  -2,  'active'),
    ('readme_p3',          'Phase 3',  -1,  'active'),
    ('readme_p4',          'Phase 4',   0,  'active'),
    ('readme_p5',          'Phase 5',   1,  'active'),
    ('readme_p6_web',      'Phase 6',   2,  'external'),
    ('readme_p7_multicam', 'Phase 7',   3,  'external'),
]

# Per-phase internal stages — each phase has 0..N stages that appear as a
# vertical sub-pipeline anchored under the phase station (same X, Y stepping
# down). Phase 6/7 have no scripts, so no stages. Stages are NOT phases —
# they get a smaller, dimmer rendering so the user can tell the levels apart.
PHASE_STAGES = {
    'readme_p1': [
        ('readme_p1_a_detect',     'Stage 1.a — Detect & track'),
        ('readme_p1_b_teams',      'Stage 1.b — Teams'),
        ('readme_p1_c_numbers',    'Stage 1.c — Numbers'),
        ('readme_p1_d_entities',   'Stage 1.d — Entities'),
        ('readme_p1_e_annotate',   'Stage 1.e — Annotate'),
    ],
    'readme_p2': [('readme_p2_a_followcam', 'Stage 2.a — Follow-cam')],
    'readme_p3': [('readme_p3_a_rink',      'Stage 3.a — Rink keypoints (parked)')],
    'readme_p4': [('readme_p4_a_events',    'Stage 4.a — Events (STUB)')],
    'readme_p5': [('readme_p5_a_stats',     'Stage 5.a — Stats (STUB)')],
    'readme_p6_web':      [],
    'readme_p7_multicam': [],
}

# Reverse map stage_id → parent phase_id, plus an ordered stage list (used
# below for stage-BFS attribution: producer stages run before consumer
# stages, so ties on equal hop-count go to the producer).
STAGE_PARENT = {sid: pid for pid, stages in PHASE_STAGES.items() for sid, _ in stages}
STAGE_LIST_ORDERED = [sid for pid, stages in PHASE_STAGES.items() for sid, _ in stages]

PINNED = {}
PHASE_NODE_IDS = set()  # only the 7 project phases (used for is_phase styling)
STAGE_NODE_IDS = set()  # only the 1.a..5.a stages (rendered smaller)
SCRIPT_NODE_IDS = set() # the pN_x_*.py file-level AST nodes (pinned next to their stage)
# Lookup for the orchestration-side label of any node id (used for the
# filter sidebar so we can show the long phase/stage name).
NODE_LABEL_BY_ID = {n['id']: n.get('label', n['id']) for n in g['nodes']}

PHASE_LABELS = []
STAGE_LABELS = []
for phase_idx, (nid, short, slot, kind) in enumerate(PHASE_ORDER):
    # Stage cascade direction alternates per phase: Phase 1 descends,
    # Phase 2 ascends, Phase 3 descends, etc. Y zigzag of the phase
    # itself follows the same direction so the station sits on the
    # same side as its stages and reads as one column.
    stage_dir = -1 if phase_idx % 2 == 0 else +1   # -1 = down, +1 = up
    y_offset = stage_dir * ZIGZAG
    px, py, pz = slot * SPACING, y_offset, 0
    PINNED[nid] = (px, py, pz)
    PHASE_NODE_IDS.add(nid)
    PHASE_LABELS.append({'id': nid, 'short': short,
                         'long': NODE_LABEL_BY_ID.get(nid, short),
                         'x': px, 'y': py, 'kind': kind})
    # Pin each child stage on the same X axis, stepping in stage_dir
    # away from the phase station.
    for i, (sid, slabel) in enumerate(PHASE_STAGES.get(nid, [])):
        sx = px
        sy = py + stage_dir * STAGE_VSPACING * (i + 1)
        sz = pz
        PINNED[sid] = (sx, sy, sz)
        STAGE_NODE_IDS.add(sid)
        STAGE_LABELS.append({'id': sid, 'short': slabel,
                             'long': NODE_LABEL_BY_ID.get(sid, slabel),
                             'x': sx, 'y': sy, 'parent': nid})
        # Track the script id (used by the filter sidebar attribution),
        # but DO NOT pin it — let it float into orbit around its stage
        # via the natural force layout, like every other child node.
        script_id = sid.replace('readme_', '') + '_py'
        SCRIPT_NODE_IDS.add(script_id)

RELATION_COLOR = {
    'precedes': '#00f5ff', 'feeds_into': '#06ffa5',
    'consumes_output_of': '#ff006e', 'implements': '#ffb703',
    'rationale_for': '#c77dff', 'configures': '#3a86ff',
    'references': '#ffffff',
}

# --- Pipeline reachability + per-node phase attribution ---------------------
# Build undirected adjacency so we can BFS from every pinned phase. Any node
# that is reachable from at least one phase is "in the pipeline" and gets an
# X-pull toward its closest phase. Anything unreachable is an orphan and is
# rendered far past Phase 7 with a visible gap so the user can spot dead /
# unused code at a glance.
adj = {}
for l in g['links']:
    s = l['source'] if isinstance(l['source'], str) else l['source'].get('id', l['source'])
    t = l['target'] if isinstance(l['target'], str) else l['target'].get('id', l['target'])
    adj.setdefault(s, []).append(t)
    adj.setdefault(t, []).append(s)

# BFS from each phase; record (phase, hop count). The closest phase wins the
# X-pull. Stages count as their parent phase so we don't fragment the pull.
def stage_to_phase(nid):
    for pid, stages in PHASE_STAGES.items():
        for sid, _ in stages:
            if sid == nid:
                return pid
    return nid if nid in PHASE_NODE_IDS else None

phase_of_node = {}
hop_of_node = {}
from collections import deque
for pid in PHASE_NODE_IDS:
    seen = {pid: 0}
    q = deque([pid])
    while q:
        cur = q.popleft()
        for nb in adj.get(cur, []):
            if nb in seen: continue
            seen[nb] = seen[cur] + 1
            q.append(nb)
    for nid, hop in seen.items():
        if nid in PHASE_NODE_IDS or nid in STAGE_NODE_IDS:
            continue
        # closest wins; ties keep first-seen
        if nid not in hop_of_node or hop < hop_of_node[nid]:
            hop_of_node[nid] = hop
            phase_of_node[nid] = pid

# --- Per-node STAGE attribution -------------------------------------------
# BFS from each stage, refusing to traverse THROUGH another stage's anchor
# script (so attribution doesn't bleed across stage boundaries via shared
# data files). Closest-hop wins; ties go to whichever stage is processed
# first — we walk PHASE_STAGES in declared order, so the producer of a
# data file (e.g. Stage 1.a → detections.json) wins over its consumers
# (Stage 1.b/1.c/1.d/1.e all reach detections.json at the same hop).
stage_of_node = {}
stage_hop_of_node = {}
for sid in STAGE_LIST_ORDERED:
    own_script = sid.replace('readme_', '') + '_py'
    seen = {sid: 0}
    q = deque([sid])
    while q:
        cur = q.popleft()
        for nb in adj.get(cur, []):
            if nb in seen: continue
            # Don't bleed into another phase's column or another stage's
            # internals — those are owned by their own anchor.
            if nb in PHASE_NODE_IDS: continue
            if nb in STAGE_NODE_IDS and nb != sid: continue
            if nb in SCRIPT_NODE_IDS and nb != own_script: continue
            seen[nb] = seen[cur] + 1
            q.append(nb)
    for nid, hop in seen.items():
        if nid in PHASE_NODE_IDS or nid in STAGE_NODE_IDS:
            continue
        if nid not in stage_hop_of_node or hop < stage_hop_of_node[nid]:
            stage_hop_of_node[nid] = hop
            stage_of_node[nid] = sid

# Orphans = nodes not pinned, not reachable from any phase. Pin them in a
# detached cluster past Phase 7 with ORPHAN_GAP empty space in between.
import math
all_ids = {n['id'] for n in g['nodes']}
orphan_ids = [nid for nid in all_ids
              if nid not in PINNED and nid not in phase_of_node]
orphan_ids.sort()
ORPHAN_X = (max(slot for _,_,slot,_ in PHASE_ORDER) * SPACING) + ORPHAN_GAP
ORPHAN_LAYOUT = {}
if orphan_ids:
    cols = max(1, math.ceil(math.sqrt(len(orphan_ids))))
    cell = 36
    for i, nid in enumerate(orphan_ids):
        col = i % cols; row = i // cols
        ox = ORPHAN_X + (col - cols / 2) * cell
        oy = (row - cols / 2) * cell
        oz = ((i * 37) % 120) - 60   # gentle Z scatter so the cluster has depth
        ORPHAN_LAYOUT[nid] = (ox, oy, oz)
print(f'      Pipeline-reachable: {len(phase_of_node)} nodes; orphans: {len(orphan_ids)}')

nodes_out = []
for n in g['nodes']:
    nid = n['id']
    is_phase = nid in PHASE_NODE_IDS
    is_stage = nid in STAGE_NODE_IDS
    is_script = nid in SCRIPT_NODE_IDS
    is_orphan = nid in ORPHAN_LAYOUT

    # Attribution. Phases own themselves, stages own themselves (with their
    # parent phase as a coarser owner). Everything else takes its closest
    # stage if any was reached, otherwise its closest phase (covers nodes
    # attached to stage-less phases like P6/P7).
    if is_phase:
        sid_attrib = None
        pid_attrib = nid
    elif is_stage:
        sid_attrib = nid
        pid_attrib = STAGE_PARENT.get(nid)
    elif is_script:
        # The script is pinned next to its stage. Reverse-engineer the stage
        # id from the script id (p1_a_detect_py → readme_p1_a_detect).
        sid_attrib = 'readme_' + nid[:-3] if nid.endswith('_py') else None
        if sid_attrib not in STAGE_PARENT:
            sid_attrib = None
        pid_attrib = STAGE_PARENT.get(sid_attrib) if sid_attrib else phase_of_node.get(nid)
    else:
        sid_attrib = stage_of_node.get(nid)
        pid_attrib = STAGE_PARENT.get(sid_attrib) if sid_attrib else phase_of_node.get(nid)

    # XY-pull anchor: stage-level if available (children of Stage 1.b sit
    # near 1.b's pinned (x,y), not at the midline of Phase 1's column),
    # phase-level otherwise. Orphans skip the pull entirely.
    if not is_orphan and sid_attrib and sid_attrib in PINNED:
        ax, ay, _ = PINNED[sid_attrib]
    elif not is_orphan and pid_attrib and pid_attrib in PINNED:
        ax, ay, _ = PINNED[pid_attrib]
    else:
        ax, ay = None, None

    # Pass the phase kind (active / external) on phase nodes so the
    # nodeObject can colour the station to match the filter sidebar dot.
    phase_kind = None
    if is_phase:
        for _nid, _short, _slot, _kind in PHASE_ORDER:
            if _nid == nid: phase_kind = _kind; break

    # File-artifact flag: every concrete file living under
    # models/, runs/, src/, tools/ or configs/ gets a permanent floating
    # label and rows in the Pipeline Filter sidebar under EVERY stage
    # that has a direct edge to it (producer + every consumer).
    src_file = n.get('source_file') or ''
    src_parts = src_file.replace('\\', '/').strip('/').split('/')
    is_in_target_dir = any(d in src_parts for d in ('models', 'runs', 'src', 'tools', 'configs'))
    is_pyfile = nid.endswith('_py')
    is_orch_file = nid.startswith(('data_', 'model_', 'config_'))
    is_file_artifact = (
        not is_phase and not is_stage and not is_orphan
        and (is_pyfile or is_orch_file)
        and is_in_target_dir
    )

    # `connected_stages`: every stage that has a direct edge to this file
    # (either to the file directly, or via the stage's script). This is
    # what populates the multi-location listing in the filter sidebar
    # (file appears at producer AND every consumer).
    connected_stages = []
    if is_file_artifact:
        seen_stages = set()
        for nb in adj.get(nid, []):
            if nb in STAGE_NODE_IDS and nb not in seen_stages:
                seen_stages.add(nb); connected_stages.append(nb)
            elif nb in SCRIPT_NODE_IDS:
                stage_for_script = 'readme_' + nb[:-3]
                if stage_for_script in STAGE_NODE_IDS and stage_for_script not in seen_stages:
                    seen_stages.add(stage_for_script); connected_stages.append(stage_for_script)

    out = {
        'id': nid, 'label': n.get('label', nid),
        'community': n.get('community', 0),
        'community_label': COMMUNITY_LABELS.get(n.get('community', 0), f"Community {n.get('community', 0)}"),
        'source_file': (n.get('source_file') or '').split('/')[-1],
        'source_location': n.get('source_location', ''),
        'file_type': n.get('file_type', 'code'),
        'is_phase': is_phase,
        'is_stage': is_stage,
        'is_script': is_script,
        'is_orphan': is_orphan,
        'is_file_artifact': is_file_artifact,
        'connected_stages': connected_stages,
        'phase_kind': phase_kind,
        'stage_target_id': sid_attrib,
        'phase_target_id': pid_attrib,
        'anchor_x': ax,
        'anchor_y': ay,
    }
    if nid in PINNED:
        x, y, z = PINNED[nid]
        out['fx'] = x; out['fy'] = y; out['fz'] = z
    elif is_orphan:
        x, y, z = ORPHAN_LAYOUT[nid]
        out['fx'] = x; out['fy'] = y; out['fz'] = z
    nodes_out.append(out)

links_out = [{
    'source': l['source'], 'target': l['target'],
    'relation': l.get('relation', ''), 'confidence': l.get('confidence', ''),
    'weight': l.get('weight', 1.0),
    'rel_color': RELATION_COLOR.get(l.get('relation', '')),
} for l in g['links']]

payload = {'nodes': nodes_out, 'links': links_out,
           'phase_labels': PHASE_LABELS,
           'stage_labels': STAGE_LABELS,
           'orphan_x': ORPHAN_X if orphan_ids else None,
           'orphan_count': len(orphan_ids)}
data_str = json.dumps(payload).replace('</', '<\\/')
# Rail spans the active+external phases (no need to extend to the orphan
# cluster — the gap should be visually obvious, not bridged by rail).
rail_min, rail_max = -4 * SPACING, 4 * SPACING

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RILH-AI-Vision — Pipeline Graph 3D</title>
<style>
html, body { margin:0; padding:0; overflow:hidden; background:#000; font-family:-apple-system,system-ui,Segoe UI,sans-serif; color:#e0f7ff; }
#graph { position:absolute; inset:0; }
.panel { background:rgba(5,10,20,0.72); backdrop-filter:blur(12px); border:1px solid rgba(0,220,255,0.25); border-radius:10px; z-index:10; box-shadow:0 0 28px rgba(0,200,255,0.18); }
#leftcol { position:absolute; top:16px; left:16px; width:300px; display:flex; flex-direction:column; gap:12px; max-height:calc(100vh - 32px); z-index:10; }
#overlay { padding:14px 18px; }
#overlay h1 { margin:0 0 6px 0; font-size:15px; font-weight:600; letter-spacing:0.04em; color:#7fe0ff; text-transform:uppercase; }
#overlay .sub { font-size:11px; opacity:0.6; margin-bottom:10px; }
#overlay .stat { font-size:12px; margin:2px 0; opacity:0.85; }
#overlay .stat b { color:#6affff; }
#filters { padding:12px 14px; overflow:auto; min-height:0; flex:1 1 auto; font-size:11px; }
#filters h2 { margin:0 0 8px 0; font-size:11px; font-weight:600; letter-spacing:0.06em; color:#7fe0ff; text-transform:uppercase; cursor:pointer; user-select:none; display:flex; align-items:center; }
#filters h2 .caret { display:inline-block; width:10px; margin-right:6px; transition: transform 0.15s; }
#filters .section.collapsed h2 .caret { transform: rotate(-90deg); }
#filters .section.collapsed .body { display:none; }
#filters .hint { font-size:10px; opacity:0.5; margin:0 0 10px 0; }
#filters .search { margin:0 0 10px 0; }
#filters .search input { width:100%; box-sizing:border-box; padding:6px 10px; background:rgba(0,12,24,0.7); border:1px solid rgba(0,220,255,0.25); border-radius:5px; color:#cfeaff; font-size:11px; outline:none; transition:border-color 0.15s, background 0.15s; }
#filters .search input:focus { border-color:rgba(0,245,255,0.55); background:rgba(0,18,32,0.85); }
#filters .search input::placeholder { color:rgba(207,234,255,0.35); }
#filters .row.search-hidden { display:none; }
#filters .row.search-hit .name { color:#ffd60a; }
#filters .row { display:flex; align-items:center; padding:3px 4px; cursor:pointer; user-select:none; border-radius:4px; transition: background 0.12s, opacity 0.12s; }
#filters .row:hover { background:rgba(0,220,255,0.08); }
#filters .row.hidden { opacity:0.32; text-decoration:line-through; }
#filters .row .dot { width:10px; height:10px; border-radius:50%; margin-right:8px; box-shadow:0 0 8px currentColor; flex:0 0 10px; cursor:pointer; transition:transform 0.12s, box-shadow 0.12s; }
#filters .row .dot:hover { transform:scale(1.45); box-shadow:0 0 14px currentColor; }
#filters .row .name { cursor:pointer; transition:color 0.12s; }
#filters .row .name:hover { color:#fff; text-shadow:0 0 6px rgba(0,245,255,0.7); }
#filters .row.phase { font-weight:600; color:#cfeaff; margin-top:6px; }
#filters .row.phase:first-child { margin-top:0; }
#filters .row.phase.external .dot { background:#ffb703; color:#ffb703; }
#filters .row.phase.active .dot { background:#00f5ff; color:#00f5ff; }
#filters .row.stage { padding-left:22px; opacity:0.85; font-size:10.5px; }
#filters .row.stage .dot { background:#cfeaff; color:#cfeaff; width:7px; height:7px; flex:0 0 7px; }
#filters .row.community { padding-left:40px; font-size:10px; opacity:0.78; }
#filters .row.community .dot { width:8px; height:8px; flex:0 0 8px; }
#filters .row.file { padding-left:60px; font-size:10px; opacity:0.85; font-style:italic; }
#filters .row.file .dot { width:6px; height:6px; flex:0 0 6px; box-shadow:none; border:1px solid currentColor; }
#filters .row.file .name { color:#cfeaff; font-style:italic; }
#filters .row .name { flex:1; }
#filters .row .count { opacity:0.4; font-size:9.5px; margin-left:6px; }
#filters .edges { margin-top:10px; padding-top:8px; border-top:1px solid rgba(0,220,255,0.12); }
#filters .edges .edge-row { display:flex; align-items:center; margin:3px 0; opacity:0.85; font-size:10px; line-height:1; }
#filters .edges .edge-row .line { width:22px; height:2px; margin-right:8px; box-shadow:0 0 6px currentColor; flex:0 0 22px; align-self:center; }
#filters .edges .edge-row span { line-height:1; }
#filters .files { margin-top:10px; padding-top:8px; border-top:1px solid rgba(0,220,255,0.12); }
#filters .files .file-row { display:flex; align-items:center; padding:3px 4px; cursor:pointer; user-select:none; border-radius:4px; transition: background 0.12s, opacity 0.12s; font-size:10px; }
#filters .files .file-row:hover { background:rgba(0,220,255,0.08); }
#filters .files .file-row.hidden { opacity:0.32; text-decoration:line-through; }
#filters .files .file-row .swatch { width:10px; height:10px; border-radius:2px; margin-right:8px; border:1px solid rgba(0,220,255,0.5); flex:0 0 10px; }
#filters .files .file-row .ext { font-family: monospace; color:#6affff; min-width:46px; flex:0 0 46px; font-weight:600; }
#filters .files .file-row .lbl { flex:1; color:#cfeaff; }
#filters .files .file-row .cnt { opacity:0.4; font-size:9.5px; }
#filters .files .file-subgroup { margin-top:6px; }
#filters .files .file-subgroup:first-child { margin-top:0; }
#filters .files .file-subhead { font-size:10px; font-weight:600; letter-spacing:0.05em; color:rgba(127,224,255,0.55); text-transform:uppercase; margin:6px 4px 4px; }
#filters .row.stage .solo { margin-left:6px; padding:1px 5px; border-radius:3px; font-size:10px; line-height:1; cursor:pointer; opacity:0.4; transition: opacity 0.12s, background 0.12s, color 0.12s; user-select:none; }
#filters .row.stage:hover .solo { opacity:0.85; }
#filters .row.stage .solo:hover { opacity:1; background:rgba(255,214,10,0.15); color:#ffd60a; }
#filters .row.stage.soloed .solo { opacity:1; background:#ffd60a; color:#000; box-shadow:0 0 6px rgba(255,214,10,0.65); }
#filters .row.stage.soloed > .name { color:#ffd60a; }
#filters .actions { display:flex; gap:6px; margin-top:8px; padding-top:8px; border-top:1px solid rgba(0,220,255,0.12); }
#filters .actions button { flex:1; background:rgba(0,220,255,0.1); border:1px solid rgba(0,220,255,0.25); color:#cfeaff; font-size:10px; padding:4px 8px; border-radius:4px; cursor:pointer; }
#filters .actions button:hover { background:rgba(0,220,255,0.2); }
#info { position:absolute; bottom:16px; left:16px; padding:12px 16px; max-width:520px; font-size:12px; display:none; }
#info .lbl { color:#6affff; font-size:14px; font-weight:600; margin-bottom:4px; }
#info .meta { opacity:0.7; font-size:11px; }
#controls { position:absolute; bottom:16px; right:16px; padding:8px 12px; font-size:11px; opacity:0.75; }
#controls span { color:#6affff; }
.scanline::after { content:""; position:absolute; inset:0; pointer-events:none; background:repeating-linear-gradient(0deg, rgba(0,220,255,0.02) 0px, rgba(0,220,255,0.02) 1px, transparent 1px, transparent 3px); z-index:5; }
.vignette { position:absolute; inset:0; pointer-events:none; z-index:6; background: radial-gradient(ellipse at center, transparent 40%, rgba(0,0,0,0.55) 100%); }
</style>
</head>
<body class="scanline">
<div id="graph"></div>
<div class="vignette"></div>
<div id="leftcol">
  <div id="overlay" class="panel">
    <h1>▲ RILH-AI-Vision · Pipeline</h1>
    <div class="sub">Phase rail · orbital stations · volumetric haze</div>
    <div class="stat"><b>Nodes:</b> <span id="nc"></span></div>
    <div class="stat"><b>Links:</b> <span id="lc"></span></div>
    <div class="stat"><b>Communities:</b> <span id="cc"></span></div>
    <div class="stat"><b>Phases:</b> <span id="pc"></span> pinned on X-axis</div>
  </div>
  <div id="filters" class="panel">
    <div id="pipelineSection" class="section collapsed">
      <h2 data-target="pipelineSection"><span class="caret">▾</span>Pipeline Filter</h2>
      <div class="body">
        <p class="hint">Click a <b>name</b> to flash a yellow ring around it in the graph · click the coloured <b>dot</b> to hide / show.</p>
        <div class="search"><input id="filterSearch" type="text" placeholder="Search files…" autocomplete="off" spellcheck="false" /></div>
        <div id="filterTree"></div>
        <div class="actions">
          <button id="filterAll">Show all</button>
          <button id="filterNone">Hide all</button>
        </div>
      </div>
    </div>
    <div id="filesSection" class="section files collapsed">
      <h2 data-target="filesSection"><span class="caret">▾</span>File Filter</h2>
      <div class="body">
        <p class="hint">Toggle by file kind / extension — click a row to hide every node of that kind.</p>
        <div id="fileItems"></div>
      </div>
    </div>
    <div id="edgesSection" class="section edges collapsed">
      <h2 data-target="edgesSection"><span class="caret">▾</span>Edge Types</h2>
      <div class="body">
        <div id="relItems"></div>
      </div>
    </div>
  </div>
</div>
<div id="info" class="panel"><div class="lbl" id="nodeLabel"></div><div class="meta" id="nodeMeta"></div></div>
<div id="controls" class="panel"><span>Drag</span> rotate · <span>Scroll</span> zoom · <span>← → ↑ ↓</span> pan · <span>Click</span> focus · <span>R</span> reset</div>
<script src="https://unpkg.com/three@0.152.2/build/three.min.js"></script>
<script src="https://unpkg.com/3d-force-graph@1.73.4/dist/3d-force-graph.min.js"></script>
<script src="https://unpkg.com/three@0.152.2/examples/js/postprocessing/EffectComposer.js"></script>
<script src="https://unpkg.com/three@0.152.2/examples/js/postprocessing/RenderPass.js"></script>
<script src="https://unpkg.com/three@0.152.2/examples/js/postprocessing/ShaderPass.js"></script>
<script src="https://unpkg.com/three@0.152.2/examples/js/postprocessing/UnrealBloomPass.js"></script>
<script src="https://unpkg.com/three@0.152.2/examples/js/shaders/CopyShader.js"></script>
<script src="https://unpkg.com/three@0.152.2/examples/js/shaders/LuminosityHighPassShader.js"></script>
<script>
const DATA = __DATA__;
const RAIL_MIN = __RAIL_MIN__, RAIL_MAX = __RAIL_MAX__;
const PALETTE = ["#00f5ff","#ff006e","#ffb703","#8338ec","#06ffa5","#fb5607","#3a86ff","#ff4d9d","#80ed99","#ffd60a","#c77dff","#4cc9f0","#f72585","#7209b7","#4361ee","#f15bb5","#00f5d4","#fee440","#ff9e00"];
const colorFor = c => PALETTE[c % PALETTE.length];
const RELATION_COLOR = {'precedes':'#00f5ff','feeds_into':'#06ffa5','consumes_output_of':'#ff006e','implements':'#ffb703','rationale_for':'#c77dff','configures':'#3a86ff','references':'#ffffff'};
const RELATION_LABEL = {'precedes':'precedes (phase order)','feeds_into':'feeds_into (data flow)','consumes_output_of':'consumes_output_of','implements':'implements (concept→code)','rationale_for':'rationale_for (why)','configures':'configures (yaml→code)','references':'references'};

document.getElementById('nc').textContent = DATA.nodes.length;
document.getElementById('lc').textContent = DATA.links.length;
const comms = [...new Set(DATA.nodes.map(n=>n.community))].sort((a,b)=>a-b);
document.getElementById('cc').textContent = comms.length;
document.getElementById('pc').textContent = DATA.nodes.filter(n=>n.is_phase).length;

const commLabels = {}; DATA.nodes.forEach(n => { if(!(n.community in commLabels)) commLabels[n.community] = n.community_label; });
const commCounts = {}; DATA.nodes.forEach(n => { commCounts[n.community] = (commCounts[n.community]||0)+1; });
const hiddenComms = new Set();
const hiddenPhases = new Set();
const hiddenStages = new Set();
const hiddenNodes = new Set();   // individual files toggled from the filter sidebar
// File-kind defaults: every real extension is shown; the synthetic
// '_function' / '_class' / '_rationale' / '_other_code' / '_other'
// buckets start hidden — together they're 180+ extra labels that
// dominate the scene unless the user explicitly opts in. Phase /
// stage anchors live under '_anchor' which is never user-toggleable
// so their labels always render.
const hiddenFileKinds = new Set(['_function', '_class', '_rationale', '_other_code', '_other']);
let soloStageId = null;             // when set, only this stage's content (+1-hop neighbours) is visible
let soloVisibleSet = null;          // Set<id> rebuilt when soloStageId changes; null otherwise

// Build the set of node ids that should remain visible while a stage is
// soloed: the stage anchor itself, every node attributed to the stage
// (including file artefacts whose `connected_stages` list contains the
// stage), and all 1-hop neighbours of those — so the soloed stage is
// shown together with what it consumes / produces / depends on (input
// JSONs from upstream stages, model weights, configs, etc.).
function rebuildSoloSet(stageId) {
  if (stageId === null) { soloVisibleSet = null; return; }
  const core = new Set();
  core.add(stageId);
  DATA.nodes.forEach(n => {
    if (n.stage_target_id === stageId) core.add(n.id);
    if ((n.connected_stages || []).includes(stageId)) core.add(n.id);
  });
  const visible = new Set(core);
  DATA.links.forEach(l => {
    const sid = typeof l.source === 'object' ? l.source.id : l.source;
    const tid = typeof l.target === 'object' ? l.target.id : l.target;
    if (core.has(sid)) visible.add(tid);
    if (core.has(tid)) visible.add(sid);
  });
  soloVisibleSet = visible;
}

// Pseudo-extension key for a node — used by the File Filter section to
// bucket nodes for the toggle. Synthetic categories take priority over
// extension because they're more precise:
//   _anchor       — phases / stages (never toggled, always shown)
//   _function     — AST function definitions (label ends with `()`)
//   _class        — AST class definitions (capitalized, no parens, no dot)
//   _rationale    — graphify-extracted rationale notes
//   _other_code   — code-typed AST node that didn't match fn/class shape
//   .py / .md / … — real file extensions for is_script + file_artifact nodes
//   _other        — fallback (no source_file, no extension)
function nodeKind(n) {
  if (n.is_phase || n.is_stage) return '_anchor';
  const ft = n.file_type || '';
  if (ft === 'rationale') return '_rationale';
  // AST sub-nodes (functions/classes inside a .py module). Script-level
  // .py files (is_script=true) and file-artifact data files fall through
  // to extension lookup so they're filterable as `.py` / `.json` / etc.
  if (ft === 'code' && !n.is_script && !n.is_file_artifact) {
    const lbl = n.label || '';
    if (lbl.endsWith('()')) return '_function';
    if (/^[A-Z]/.test(lbl) && !lbl.includes('.')) return '_class';
    return '_other_code';
  }
  const sf = n.source_file || '';
  const base = sf.split('/').pop() || '';
  const dot = base.lastIndexOf('.');
  if (dot < 0) return '_other';
  const ext = base.slice(dot).toLowerCase();
  return ext || '_other';
}

// Highlight helper — instead of trying to recentre the camera (which is
// unreliable while the force layout is still settling and during the
// auto-orbit), we attach a big bright pulsing ring + halo to the target
// node for 4 seconds so the user can spot it visually wherever it sits
// in the scene. Yellow `#ffd60a` so it pops against the cyan/amber palette.
function highlightNodeObj(n) {
  if (!n) return;
  const obj = n.__threeObj;
  if (!obj) return;
  const HL = 0xffd60a;
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(34, 40, 56),
    new THREE.MeshBasicMaterial({color: HL, side: THREE.DoubleSide, transparent: true, opacity: 0.95, depthWrite: false, fog: false})
  );
  ring.rotation.y = Math.PI / 2;
  obj.add(ring);
  const halo = new THREE.Mesh(
    new THREE.SphereGeometry(22, 20, 20),
    new THREE.MeshBasicMaterial({color: HL, transparent: true, opacity: 0.55, depthWrite: false, fog: false})
  );
  obj.add(halo);
  const ring2 = new THREE.Mesh(
    new THREE.RingGeometry(80, 86, 64),
    new THREE.MeshBasicMaterial({color: HL, side: THREE.DoubleSide, transparent: true, opacity: 0.45, depthWrite: false, fog: false})
  );
  ring2.rotation.x = Math.PI / 2;
  obj.add(ring2);
  const t0 = performance.now();
  const duration = 4000;
  (function tick() {
    const t = (performance.now() - t0) / duration;
    if (t >= 1) {
      obj.remove(ring); obj.remove(halo); obj.remove(ring2);
      ring.geometry.dispose(); ring.material.dispose();
      halo.geometry.dispose(); halo.material.dispose();
      ring2.geometry.dispose(); ring2.material.dispose();
      return;
    }
    const fade = 1 - t;
    const pulse = 1 + 0.55 * Math.sin(t * Math.PI * 8);
    ring.scale.setScalar(pulse);
    halo.scale.setScalar(1 + 0.4 * Math.sin(t * Math.PI * 8));
    ring2.scale.setScalar(1 + 0.25 * Math.sin(t * Math.PI * 6));
    ring.material.opacity = 0.95 * fade;
    halo.material.opacity = 0.55 * fade;
    ring2.material.opacity = 0.45 * fade;
    requestAnimationFrame(tick);
  })();
}
function highlightNodeId(id) {
  highlightNodeObj(DATA.nodes.find(x => x.id === id));
}
function highlightCommunity(c) {
  // Pulse every visible node in this community.
  DATA.nodes.forEach(n => { if (n.community === c) highlightNodeObj(n); });
}

// Combined visibility: a node is visible iff its community is visible AND
// it is not attributed to a hidden phase AND not attributed to a hidden
// stage AND not in the explicit hidden-nodes set. Phases / stages are
// also hidden when the user filters them out directly (the phase/stage
// node IS attributed to itself).
function isNodeVisible(n) {
  if (hiddenNodes.has(n.id)) return false;
  if (hiddenComms.has(n.community)) return false;
  if (n.phase_target_id && hiddenPhases.has(n.phase_target_id)) return false;
  if (n.stage_target_id && hiddenStages.has(n.stage_target_id)) return false;
  // (file-kind toggle no longer hides nodes — it only toggles label sprites,
  // see applyLabelVisibility())
  // Solo mode: when a stage is soloed, the visible set is the stage anchor
  // + nodes attributed to it + 1-hop neighbours (everything they connect
  // to or that connects to them). This shows the soloed stage WITH its
  // dependencies (upstream data files, model weights, configs, downstream
  // consumers). The set is precomputed in `rebuildSoloSet` and cached.
  if (soloStageId !== null && soloVisibleSet) {
    if (!soloVisibleSet.has(n.id)) return false;
  }
  return true;
}
function applyFilters() {
  Graph.nodeVisibility(isNodeVisible);
  Graph.linkVisibility(l => {
    const s = typeof l.source==='object'?l.source:DATA.nodes.find(n=>n.id===l.source);
    const t = typeof l.target==='object'?l.target:DATA.nodes.find(n=>n.id===l.target);
    if (!s || !t) return true;
    return isNodeVisible(s) && isNodeVisible(t);
  });
}
// Toggle floating label sprites (the text floating next to a node) by
// file-kind, without touching node visibility. Each label sprite is
// tagged at construction time with `userData.isLabel = true` so we can
// find it among the children of the node's Three.js group.
function applyLabelVisibility() {
  DATA.nodes.forEach(n => {
    const obj = n.__threeObj;
    if (!obj) return;
    const showLabel = !hiddenFileKinds.has(nodeKind(n));
    obj.children.forEach(c => {
      if (c.userData && c.userData.isLabel) c.visible = showLabel;
    });
  });
}

// --- Pipeline filter sidebar (Phase → Stage → Community tree) -------------
// Counts: how many *content* nodes are attributed to each phase / stage,
// and which communities those nodes belong to (so each stage shows its
// own list of communities below it). Phase / stage anchor nodes don't
// count — only the children the filter actually hides.
const phaseContentCount = {};
const stageContentCount = {};
const stageCommunities = {};   // stage_id → Set of community ids present
const phaseCommunities = {};   // phase_id → Set of community ids present (for nodes attached to phase but no stage)
DATA.nodes.forEach(n => {
  if (n.is_phase || n.is_stage) return;
  if (n.stage_target_id) {
    stageContentCount[n.stage_target_id] = (stageContentCount[n.stage_target_id]||0)+1;
    (stageCommunities[n.stage_target_id] = stageCommunities[n.stage_target_id] || new Set()).add(n.community);
  } else if (n.phase_target_id) {
    (phaseCommunities[n.phase_target_id] = phaseCommunities[n.phase_target_id] || new Set()).add(n.community);
  }
  if (n.phase_target_id) phaseContentCount[n.phase_target_id] = (phaseContentCount[n.phase_target_id]||0)+1;
});

// Per-(stage, community) and per-(phase, community) node counts, used for
// the (N) badges on community sub-rows.
const stageCommCount = {}, phaseCommCount = {};
DATA.nodes.forEach(n => {
  if (n.is_phase || n.is_stage) return;
  if (n.stage_target_id) {
    const k = n.stage_target_id + '|' + n.community;
    stageCommCount[k] = (stageCommCount[k]||0)+1;
  } else if (n.phase_target_id) {
    const k = n.phase_target_id + '|' + n.community;
    phaseCommCount[k] = (phaseCommCount[k]||0)+1;
  }
});

// File artifacts indexed by (stage, community) — a file is listed under
// every stage it has a direct edge to (producer + each consumer), nested
// inside that stage's community row. So `p1_detections.json` appears
// under Stage 1.a (created) AND under Stage 1.b/1.c/1.d/1.e (consumed),
// inside its community sub-row each time.
const fileByStageComm = {};      // 'stage|community' → [node, …]
const fileByPhaseComm = {};      // 'phase|community' → [node, …]  (fallback)
const stageHasFiles  = new Set();   // 'stage|community' keys that exist
DATA.nodes.forEach(n => {
  if (!n.is_file_artifact) return;
  const stages = n.connected_stages || [];
  if (stages.length === 0) {
    if (n.phase_target_id) {
      const k = n.phase_target_id + '|' + n.community;
      (fileByPhaseComm[k] = fileByPhaseComm[k] || []).push(n);
    }
    return;
  }
  stages.forEach(sid => {
    const k = sid + '|' + n.community;
    (fileByStageComm[k] = fileByStageComm[k] || []).push(n);
    stageHasFiles.add(k);
  });
});
Object.values(fileByStageComm).forEach(arr => arr.sort((a,b) => (a.label||a.id).localeCompare(b.label||b.id)));
Object.values(fileByPhaseComm).forEach(arr => arr.sort((a,b) => (a.label||a.id).localeCompare(b.label||b.id)));

// A community must include EVERY stage in which any of its files lives
// (so a file from `p1_a_detect` listed under Stage 1.b — because 1.b
// consumes detections.json — needs Stage 1.b's community list to surface
// the file's community even if no other 1.b-attributed node carries it).
DATA.nodes.forEach(n => {
  if (!n.is_file_artifact) return;
  (n.connected_stages || []).forEach(sid => {
    (stageCommunities[sid] = stageCommunities[sid] || new Set()).add(n.community);
  });
});

const filterTree = document.getElementById('filterTree');
const phaseRowEls = {};
const stageRowEls = {};
const commRowEls = [];   // every community row in the tree (multiple rows per community possible)

function makeCommRow(c, count, indentClass) {
  const row = document.createElement('div');
  row.className = 'row community ' + indentClass;
  row.dataset.community = c;
  row.innerHTML = '<div class="dot" style="background:'+colorFor(c)+';color:'+colorFor(c)+'"></div><span class="name">'+(commLabels[c]||('Community '+c))+'</span><span class="count">'+count+'</span>';
  if (hiddenComms.has(c)) row.classList.add('hidden');
  // Row click → focus the camera on this community (centroid of its nodes).
  row.addEventListener('click', () => highlightCommunity(c));
  // Dot click → toggle visibility. Stops propagation so it doesn't focus.
  row.querySelector('.dot').addEventListener('click', e => {
    e.stopPropagation();
    if (hiddenComms.has(c)) { hiddenComms.delete(c); }
    else { hiddenComms.add(c); }
    commRowEls.forEach(r => {
      if (Number(r.dataset.community) === c) r.classList.toggle('hidden', hiddenComms.has(c));
    });
    applyFilters();
  });
  commRowEls.push(row);
  return row;
}

function makeFileRow(node) {
  const row = document.createElement('div');
  row.className = 'row file';
  row.dataset.nodeId = node.id;
  const c = colorFor(node.community);
  row.innerHTML = '<div class="dot" style="color:'+c+'"></div><span class="name">'+(node.label || node.id)+'</span>';
  if (hiddenNodes.has(node.id)) row.classList.add('hidden');
  // Row click → focus camera on the file node.
  row.addEventListener('click', () => highlightNodeId(node.id));
  // Dot click → toggle visibility of the file node.
  row.querySelector('.dot').addEventListener('click', e => {
    e.stopPropagation();
    if (hiddenNodes.has(node.id)) { hiddenNodes.delete(node.id); row.classList.remove('hidden'); }
    else { hiddenNodes.add(node.id); row.classList.add('hidden'); }
    applyFilters();
  });
  return row;
}

DATA.phase_labels.forEach(p => {
  const phaseRow = document.createElement('div');
  phaseRow.className = 'row phase ' + (p.kind === 'external' ? 'external' : 'active');
  phaseRow.dataset.phaseId = p.id;
  const cnt = phaseContentCount[p.id] || 0;
  phaseRow.innerHTML = '<div class="dot"></div><span class="name">'+p.long+'</span><span class="count">'+cnt+'</span>';
  // Row click → focus camera on the phase station.
  phaseRow.addEventListener('click', () => highlightNodeId(p.id));
  // Dot click → toggle visibility of the phase + everything it contains.
  phaseRow.querySelector('.dot').addEventListener('click', e => {
    e.stopPropagation();
    if (hiddenPhases.has(p.id)) { hiddenPhases.delete(p.id); phaseRow.classList.remove('hidden'); }
    else { hiddenPhases.add(p.id); phaseRow.classList.add('hidden'); }
    applyFilters();
  });
  filterTree.appendChild(phaseRow);
  phaseRowEls[p.id] = phaseRow;

  DATA.stage_labels.filter(s => s.parent === p.id).forEach(s => {
    const stageRow = document.createElement('div');
    stageRow.className = 'row stage';
    stageRow.dataset.stageId = s.id;
    const scnt = stageContentCount[s.id] || 0;
    stageRow.innerHTML = '<div class="dot"></div><span class="name">'+s.long+'</span><span class="count">'+scnt+'</span><span class="solo" title="Show only this stage (solo / un-solo)">◉</span>';
    // Row click → focus camera on the stage anchor.
    stageRow.addEventListener('click', () => highlightNodeId(s.id));
    // Dot click → toggle visibility of the stage + its bucket.
    stageRow.querySelector('.dot').addEventListener('click', e => {
      e.stopPropagation();
      if (hiddenStages.has(s.id)) { hiddenStages.delete(s.id); stageRow.classList.remove('hidden'); }
      else { hiddenStages.add(s.id); stageRow.classList.add('hidden'); }
      applyFilters();
    });
    // Solo button — hide everything except this stage. Clicking again
    // un-solos. Only one stage can be soloed at a time.
    stageRow.querySelector('.solo').addEventListener('click', e => {
      e.stopPropagation();
      if (soloStageId === s.id) {
        soloStageId = null;
      } else {
        soloStageId = s.id;
      }
      rebuildSoloSet(soloStageId);
      // Refresh `.soloed` class on every stage row so the highlight tracks state.
      Object.entries(stageRowEls).forEach(([sid, row]) => {
        row.classList.toggle('soloed', sid === soloStageId);
      });
      applyFilters();
    });
    filterTree.appendChild(stageRow);
    stageRowEls[s.id] = stageRow;

    // Communities present under this stage, sorted by node count desc.
    // Each community row carries — nested under it — the files that
    // belong to (this stage, this community).
    const set = stageCommunities[s.id];
    if (set) {
      const arr = [...set].sort((a,b) => (stageCommCount[s.id+'|'+b]||0) - (stageCommCount[s.id+'|'+a]||0));
      arr.forEach(c => {
        filterTree.appendChild(makeCommRow(c, stageCommCount[s.id+'|'+c]||0, ''));
        const filesHere = fileByStageComm[s.id + '|' + c] || [];
        filesHere.forEach(n => filterTree.appendChild(makeFileRow(n)));
      });
    }
  });

  // Communities attached to this phase but to no stage (e.g. orchestration
  // nodes that connect only to readme_p6_web). Inline them under the phase.
  const phaseSet = phaseCommunities[p.id];
  if (phaseSet) {
    const arr = [...phaseSet].sort((a,b) => (phaseCommCount[p.id+'|'+b]||0) - (phaseCommCount[p.id+'|'+a]||0));
    arr.forEach(c => {
      filterTree.appendChild(makeCommRow(c, phaseCommCount[p.id+'|'+c]||0, ''));
      const filesHere = fileByPhaseComm[p.id + '|' + c] || [];
      filesHere.forEach(n => filterTree.appendChild(makeFileRow(n)));
    });
  }
});

document.getElementById('filterAll').addEventListener('click', () => {
  hiddenPhases.clear(); hiddenStages.clear(); hiddenComms.clear(); hiddenNodes.clear();
  hiddenFileKinds.clear(); soloStageId = null; soloVisibleSet = null;
  Object.values(phaseRowEls).forEach(r => r.classList.remove('hidden'));
  Object.values(stageRowEls).forEach(r => { r.classList.remove('hidden'); r.classList.remove('soloed'); });
  commRowEls.forEach(r => r.classList.remove('hidden'));
  document.querySelectorAll('#filterTree .row.file').forEach(r => r.classList.remove('hidden'));
  document.querySelectorAll('#fileItems .file-row').forEach(r => r.classList.remove('hidden'));
  applyFilters();
  applyLabelVisibility();
});
document.getElementById('filterNone').addEventListener('click', () => {
  DATA.phase_labels.forEach(p => { hiddenPhases.add(p.id); phaseRowEls[p.id].classList.add('hidden'); });
  DATA.stage_labels.forEach(s => { hiddenStages.add(s.id); stageRowEls[s.id].classList.add('hidden'); });
  applyFilters();
});
const relItems = document.getElementById('relItems');
Object.keys(RELATION_LABEL).forEach(r => { const el = document.createElement('div'); el.className = 'edge-row'; el.innerHTML = '<div class="line" style="background:'+RELATION_COLOR[r]+';color:'+RELATION_COLOR[r]+'"></div><span>'+RELATION_LABEL[r]+'</span>'; relItems.appendChild(el); });

// --- File Filter section ---------------------------------------------------
// Bucket every node by `nodeKind(n)` (its extension, or '_rationale' /
// '_other' for nodes without one). The toggle row controls **label sprite
// visibility only** — the underlying nodes stay in the graph. Two
// sub-sections: real file extensions (alphabetical) on top, synthetic
// buckets ('_rationale', '_other') below — also alphabetical.
const FILE_KIND_LABEL = {
  // Real extensions
  '.py'        : ['.py',     'Python file',       '#06ffa5'],
  '.md'        : ['.md',     'Documentation',     '#ffb703'],
  '.json'      : ['.json',   'Data / JSON',       '#3a86ff'],
  '.pt'        : ['.pt',     'Model weights',     '#ff006e'],
  '.yaml'      : ['.yaml',   'Configs',           '#fee440'],
  '.yml'       : ['.yml',    'Configs',           '#fee440'],
  '.mp4'       : ['.mp4',    'Video',             '#f72585'],
  '.png'       : ['.png',    'Image',             '#80ed99'],
  '.jpg'       : ['.jpg',    'Image',             '#80ed99'],
  '.csv'       : ['.csv',    'Tabular',           '#4cc9f0'],
  '.pdf'       : ['.pdf',    'PDF',               '#ff9e00'],
  // Synthetic categories (Other section)
  '_function'  : ['fn()',    'Functions / methods', '#06ffa5'],
  '_class'     : ['cls',     'Classes',           '#c77dff'],
  '_rationale' : ['note',    'Rationale notes',   '#a0a0a0'],
  '_other_code': ['code?',   'Other code',        '#ffffff'],
  '_other'     : ['(other)', 'Other',             '#ffffff'],
};
const fileKindCount = {};
DATA.nodes.forEach(n => {
  const k = nodeKind(n);
  fileKindCount[k] = (fileKindCount[k] || 0) + 1;
});
const fileItems = document.getElementById('fileItems');
const fileKindRowEls = {};

// Two sub-section containers (Extensions / Other), each with its own
// header, sorted alphabetically by key. Synthetic keys (those starting
// with '_') sort after real extensions thanks to ASCII order, but we
// route them to a separate group anyway for clarity.
const extGroup = document.createElement('div');
extGroup.className = 'file-subgroup';
extGroup.innerHTML = '<div class="file-subhead">Extensions</div>';
const otherGroup = document.createElement('div');
otherGroup.className = 'file-subgroup';
otherGroup.innerHTML = '<div class="file-subhead">Other</div>';
fileItems.appendChild(extGroup);
fileItems.appendChild(otherGroup);

function makeFileKindRow(k) {
  const meta = FILE_KIND_LABEL[k] || [k, k, '#ffffff'];
  const [ext, lbl, col] = meta;
  const row = document.createElement('div');
  row.className = 'file-row';
  row.dataset.kind = k;
  row.innerHTML = '<span class="swatch" style="background:'+col+'"></span>'
                + '<span class="ext">'+ext+'</span>'
                + '<span class="lbl">'+lbl+'</span>'
                + '<span class="cnt">'+fileKindCount[k]+'</span>';
  if (hiddenFileKinds.has(k)) row.classList.add('hidden');
  row.addEventListener('click', () => {
    if (hiddenFileKinds.has(k)) { hiddenFileKinds.delete(k); row.classList.remove('hidden'); }
    else { hiddenFileKinds.add(k); row.classList.add('hidden'); }
    applyLabelVisibility();
  });
  fileKindRowEls[k] = row;
  return row;
}

// '_anchor' is never user-toggleable (phase/stage labels are always visible).
const allKinds = Object.keys(fileKindCount).filter(k => k !== '_anchor').sort();
const realExtKinds = allKinds.filter(k => k.startsWith('.'));
const syntheticKinds = allKinds.filter(k => !k.startsWith('.'));
realExtKinds.forEach(k => extGroup.appendChild(makeFileKindRow(k)));
syntheticKinds.forEach(k => otherGroup.appendChild(makeFileKindRow(k)));

// Collapsible sections — click h2 to toggle.
document.querySelectorAll('#filters h2[data-target]').forEach(h => {
  h.addEventListener('click', () => {
    const sec = document.getElementById(h.dataset.target);
    if (sec) sec.classList.toggle('collapsed');
  });
});

// File-name search — case-insensitive substring match on file rows.
// When a query is typed:
//   - matching file rows are highlighted (yellow)
//   - non-matching file rows are hidden
//   - phase / stage / community rows that have NO matching descendant are
//     hidden too, so the tree stays scannable.
const searchInput = document.getElementById('filterSearch');
function applySearch() {
  const q = (searchInput.value || '').toLowerCase().trim();
  const rows = Array.from(document.querySelectorAll('#filterTree .row'));
  if (!q) {
    rows.forEach(r => { r.classList.remove('search-hidden'); r.classList.remove('search-hit'); });
    return;
  }
  // First pass: tag file rows as match/no-match.
  let lastPhase = null, lastStage = null, lastComm = null;
  // Buckets: phase → boolean (any hit anywhere under it).
  // Built by walking the flat row sequence and remembering current parents.
  const phaseHasHit = {};
  const stageHasHit = {};
  const commHasHit = {};
  rows.forEach(r => {
    if (r.classList.contains('phase'))     { lastPhase = r.dataset.phaseId; lastStage = null; lastComm = null; }
    else if (r.classList.contains('stage')){ lastStage = r.dataset.stageId; lastComm = null; }
    else if (r.classList.contains('community')) { lastComm = lastStage + '|' + r.dataset.community; }
    else if (r.classList.contains('file')) {
      const name = (r.querySelector('.name')?.textContent || '').toLowerCase();
      const hit = name.includes(q);
      r.classList.toggle('search-hit', hit);
      r.classList.toggle('search-hidden', !hit);
      if (hit) {
        if (lastPhase) phaseHasHit[lastPhase] = true;
        if (lastStage) stageHasHit[lastStage] = true;
        if (lastComm)  commHasHit[lastComm]   = true;
      }
    }
  });
  // Second pass: hide phase / stage / community rows with no hit underneath.
  lastPhase = null; lastStage = null; lastComm = null;
  rows.forEach(r => {
    if (r.classList.contains('phase')) {
      lastPhase = r.dataset.phaseId; lastStage = null; lastComm = null;
      r.classList.toggle('search-hidden', !phaseHasHit[lastPhase]);
    } else if (r.classList.contains('stage')) {
      lastStage = r.dataset.stageId; lastComm = null;
      r.classList.toggle('search-hidden', !stageHasHit[lastStage]);
    } else if (r.classList.contains('community')) {
      lastComm = lastStage + '|' + r.dataset.community;
      r.classList.toggle('search-hidden', !commHasHit[lastComm]);
    }
  });
}
searchInput.addEventListener('input', applySearch);
// Auto-expand the Pipeline Filter section when the user starts typing.
searchInput.addEventListener('focus', () => {
  document.getElementById('pipelineSection').classList.remove('collapsed');
});

const degree = {};
DATA.links.forEach(l => { const s = typeof l.source==='object'?l.source.id:l.source; const t = typeof l.target==='object'?l.target.id:l.target; degree[s] = (degree[s]||0)+1; degree[t] = (degree[t]||0)+1; });

function linkColor(l) { if (l.rel_color) return l.rel_color; const src = typeof l.source==='object'?l.source:DATA.nodes.find(n=>n.id===l.source); return src ? colorFor(src.community) : '#00f5ff'; }
// Adaptive label sprite: measures the text at a fixed font size, sizes
// the canvas to fit it (no clipping, no smushing), then scales the
// sprite so the text reads at a consistent height in world space while
// the WIDTH stretches with the actual character count. So a short name
// like "Phase 1" stays compact and a long one like
// "p3_rink_keypoints.json" gets a wider sprite without ever overflowing.
function makeLabelSprite(text, color, baseHeight) {
  const fontSize = 72;
  const probe = document.createElement('canvas').getContext('2d');
  probe.font = 'bold ' + fontSize + 'px -apple-system, system-ui, sans-serif';
  const textW = Math.ceil(probe.measureText(text).width);
  const padX = 64;
  const canvasH = 128;
  const canvasW = Math.max(192, textW + padX * 2);
  const canvas = document.createElement('canvas');
  canvas.width = canvasW; canvas.height = canvasH;
  const ctx = canvas.getContext('2d');
  ctx.font = 'bold ' + fontSize + 'px -apple-system, system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.shadowColor = color; ctx.shadowBlur = 28; ctx.fillStyle = color;
  ctx.fillText(text, canvasW / 2, canvasH / 2);
  ctx.shadowBlur = 0; ctx.fillStyle = '#ffffff'; ctx.globalAlpha = 0.95;
  ctx.fillText(text, canvasW / 2, canvasH / 2);
  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  const mat = new THREE.SpriteMaterial({map: tex, transparent: true, depthTest: false, fog: false});
  const sp = new THREE.Sprite(mat);
  // World height = baseHeight / 4 (matches the original 4:1 aspect at
  // baseHeight=240 for phase labels). World width = height × canvas
  // aspect, so longer text gets a proportionally wider sprite.
  const worldH = baseHeight * 0.25;
  const worldW = worldH * (canvasW / canvasH);
  sp.scale.set(worldW, worldH, 1);
  return sp;
}

function nodeObject(n) {
  const grp = new THREE.Group();
  if (n.is_phase) {
    // Top-level project phases: full station treatment (core + halo + 3
    // orbital rings + pulsing station ring + satellite + bright label).
    // Active phases (P1-P5) tint cyan, external phases (P6/P7) tint amber
    // — matches the filter sidebar dots so the user can map at a glance.
    const isExternal = n.phase_kind === 'external';
    const tint = isExternal ? 0xffb703 : 0x00f5ff;
    const tintHex = isExternal ? '#ffb703' : '#00f5ff';
    const r = 14;
    const core = new THREE.Mesh(new THREE.SphereGeometry(r, 28, 28), new THREE.MeshBasicMaterial({color: 0xffffff, fog: false})); grp.add(core);
    const halo = new THREE.Mesh(new THREE.SphereGeometry(r * 1.6, 24, 24), new THREE.MeshBasicMaterial({color: tint, transparent: true, opacity: 0.22, depthWrite: false, fog: false})); grp.add(halo);
    const ring1 = new THREE.Mesh(new THREE.RingGeometry(28, 32, 48), new THREE.MeshBasicMaterial({color: tint, side: THREE.DoubleSide, transparent: true, opacity: 0.65, depthWrite: false, fog: false})); ring1.rotation.y = Math.PI / 2; grp.add(ring1);
    const stationRing = new THREE.Mesh(new THREE.RingGeometry(95, 100, 64), new THREE.MeshBasicMaterial({color: tint, side: THREE.DoubleSide, transparent: true, opacity: 0.18, depthWrite: false, fog: false})); stationRing.rotation.y = Math.PI / 2; grp.add(stationRing);
    const orb1 = new THREE.Mesh(new THREE.TorusGeometry(48, 1.0, 8, 72), new THREE.MeshBasicMaterial({color: 0x06ffa5, transparent: true, opacity: 0.75, depthWrite: false, fog: false})); orb1.rotation.x = Math.PI / 4; orb1.rotation.y = Math.PI / 6; grp.add(orb1);
    const orb2 = new THREE.Mesh(new THREE.TorusGeometry(64, 0.8, 8, 72), new THREE.MeshBasicMaterial({color: 0xff4d9d, transparent: true, opacity: 0.6, depthWrite: false, fog: false})); orb2.rotation.x = -Math.PI / 3; orb2.rotation.z = Math.PI / 5; grp.add(orb2);
    const orb3 = new THREE.Mesh(new THREE.TorusGeometry(78, 0.55, 6, 64), new THREE.MeshBasicMaterial({color: 0xffb703, transparent: true, opacity: 0.4, depthWrite: false, fog: false})); orb3.rotation.x = Math.PI / 2.5; grp.add(orb3);
    const sat = new THREE.Mesh(new THREE.SphereGeometry(2.8, 12, 12), new THREE.MeshBasicMaterial({color: 0xffffff, fog: false})); grp.add(sat);
    const lbl = makeLabelSprite(n.label.split('—')[0].trim(), tintHex, 240); lbl.position.set(0, 130, 0); lbl.userData.isLabel = true; lbl.visible = !hiddenFileKinds.has(nodeKind(n)); grp.add(lbl);
    grp.userData.tick = (t) => {
      stationRing.scale.setScalar(1 + 0.08 * Math.sin(t * 2.5));
      core.scale.setScalar(1 + 0.04 * Math.sin(t * 3));
      orb1.rotation.z = t * 0.85; orb2.rotation.y = -t * 0.55; orb3.rotation.x = Math.PI / 2.5 + t * 0.35;
      const a = t * 1.4, tx = Math.PI/4, ty = Math.PI/6;
      const cx = Math.cos(a) * 48, cy = Math.sin(a) * 48;
      sat.position.set(cx * Math.cos(ty) - cy * Math.sin(tx) * Math.sin(ty), cy * Math.cos(tx), -cx * Math.sin(ty) - cy * Math.sin(tx) * Math.cos(ty));
    };
  } else if (n.is_stage) {
    // Internal stages of a phase: deliberately smaller + dimmer than the
    // parent phase so the visual hierarchy is unambiguous. One small core
    // + one thin perpendicular ring + a smaller, semi-transparent label.
    const r = 5;
    const core = new THREE.Mesh(new THREE.SphereGeometry(r, 18, 18), new THREE.MeshBasicMaterial({color: 0xcfeaff, transparent: true, opacity: 0.85, fog: false})); grp.add(core);
    const halo = new THREE.Mesh(new THREE.SphereGeometry(r * 1.7, 14, 14), new THREE.MeshBasicMaterial({color: 0x00f5ff, transparent: true, opacity: 0.18, depthWrite: false, fog: false})); grp.add(halo);
    const ring = new THREE.Mesh(new THREE.RingGeometry(10, 12, 36), new THREE.MeshBasicMaterial({color: 0x00f5ff, side: THREE.DoubleSide, transparent: true, opacity: 0.32, depthWrite: false, fog: false})); ring.rotation.y = Math.PI / 2; grp.add(ring);
    // Strip "Stage 1.a — " out of the label for the floating sprite
    const shortLabel = (n.label || '').replace(/^Stage\s+/, '').split('—')[0].trim();
    const lbl = makeLabelSprite(shortLabel, '#cfeaff', 130); lbl.position.set(0, 30, 0); lbl.material.opacity = 0.82; lbl.userData.isLabel = true; lbl.visible = !hiddenFileKinds.has(nodeKind(n)); grp.add(lbl);
  } else if (n.is_script) {
    // Stage-entry scripts (pN_x_*.py): soft mint sphere + a generously
    // sized floating label so the user can tell at a glance which file
    // each stage actually launches.
    const r = 4;
    const core = new THREE.Mesh(new THREE.SphereGeometry(r, 16, 16), new THREE.MeshBasicMaterial({color: 0x80ed99, transparent: true, opacity: 0.92, fog: false})); grp.add(core);
    const halo = new THREE.Mesh(new THREE.SphereGeometry(r * 1.6, 12, 12), new THREE.MeshBasicMaterial({color: 0x80ed99, transparent: true, opacity: 0.22, depthWrite: false, fog: false})); grp.add(halo);
    const lbl = makeLabelSprite(n.label || n.id, '#80ed99', 170); lbl.position.set(0, 16, 0); lbl.material.opacity = 0.92; lbl.userData.isLabel = true; lbl.visible = !hiddenFileKinds.has(nodeKind(n)); grp.add(lbl);
  } else {
    // Regular nodes (data files, models, configs, AST helpers): community
    // colour, size scaled by degree. Orphans get an extra red halo so the
    // dead-code cluster past Phase 7 reads at a glance. File artifacts
    // (data_/model_/config_ ids) carry a permanent floating label so
    // the user can read every produced file at a glance.
    const c = colorFor(n.community);
    const r = 2.2 + Math.min(10, degree[n.id] || 0) * 0.55;
    const core = new THREE.Mesh(new THREE.SphereGeometry(r, 18, 18), new THREE.MeshBasicMaterial({color: new THREE.Color(c), fog: true})); grp.add(core);
    const halo = new THREE.Mesh(new THREE.SphereGeometry(r * 1.9, 12, 12), new THREE.MeshBasicMaterial({color: new THREE.Color(c), transparent: true, opacity: 0.3, depthWrite: false, fog: true})); grp.add(halo);
    if (n.is_orphan) {
      const orphanRing = new THREE.Mesh(new THREE.RingGeometry(r * 2.4, r * 2.7, 24), new THREE.MeshBasicMaterial({color: 0xff4d4d, side: THREE.DoubleSide, transparent: true, opacity: 0.55, depthWrite: false, fog: false}));
      grp.add(orphanRing);
    }
    if (n.is_file_artifact) {
      const lbl = makeLabelSprite(n.label || n.id, c, 170);
      lbl.position.set(0, r + 12, 0);
      lbl.material.opacity = 0.95;
      lbl.userData.isLabel = true;
      lbl.visible = !hiddenFileKinds.has(nodeKind(n));
      grp.add(lbl);
    } else {
      // Non-artifact regular nodes: function/method definitions
      // (file_type='code'), rationale notes (file_type='rationale'),
      // and the rare un-categorised node. Labels are added but hidden
      // by default for the noisier kinds (_rationale / _other) — see
      // hiddenFileKinds defaults. Rationale labels can be very long
      // (extracted comment / docstring), so we truncate them.
      let txt = n.label || n.id;
      if ((n.file_type || '') === 'rationale') {
        txt = (n.source_location || '') + (txt ? ' · ' + txt : '');
        if (txt.length > 36) txt = txt.slice(0, 33) + '…';
      } else if (txt.length > 30) {
        txt = txt.slice(0, 28) + '…';
      }
      const lbl = makeLabelSprite(txt, c, 110);
      lbl.position.set(0, r + 8, 0);
      lbl.material.opacity = 0.78;
      lbl.userData.isLabel = true;
      lbl.visible = !hiddenFileKinds.has(nodeKind(n));
      grp.add(lbl);
    }
  }
  return grp;
}

const elem = document.getElementById('graph');
const Graph = ForceGraph3D()(elem)
  .graphData(DATA)
  .backgroundColor('#000008')
  .nodeLabel(n => '<div style="background:rgba(0,20,40,0.95);padding:6px 10px;border:1px solid '+colorFor(n.community)+';border-radius:6px;font-size:12px;color:#fff;box-shadow:0 0 16px '+colorFor(n.community)+'">'+(n.is_phase?'★ ':'')+n.label+'<br><span style="opacity:.6;font-size:10px">'+n.community_label+'</span></div>')
  .nodeThreeObject(nodeObject).nodeThreeObjectExtend(false)
  .linkColor(linkColor)
  .linkOpacity(l => l.relation==='precedes' ? 0.92 : (l.rel_color ? 0.75 : 0.32))
  .linkWidth(l => l.relation==='precedes' ? 6.5 : (l.rel_color ? 1.4 : 0.55))
  .linkDirectionalParticles(l => l.relation==='precedes' ? 8 : (l.rel_color ? 3 : (l.confidence==='EXTRACTED' ? 1 : 0)))
  .linkDirectionalParticleSpeed(l => l.relation==='precedes' ? 0.014 : (l.rel_color ? 0.008 : 0.004))
  .linkDirectionalParticleWidth(l => l.relation==='precedes' ? 6 : (l.rel_color ? 2.4 : 1.2))
  .linkDirectionalParticleColor(linkColor).showNavInfo(false)
  .onNodeClick(n => {
    const dist = 80, r = Math.hypot(n.x,n.y,n.z) || 1;
    Graph.cameraPosition({x:n.x*(1+dist/r), y:n.y*(1+dist/r), z:n.z*(1+dist/r)}, n, 1400);
    const info = document.getElementById('info');
    info.style.display = 'block'; info.style.borderColor = colorFor(n.community); info.style.boxShadow = '0 0 32px '+colorFor(n.community);
    document.getElementById('nodeLabel').textContent = (n.is_phase?'★ ':'')+n.label;
    document.getElementById('nodeLabel').style.color = n.is_phase ? '#00f5ff' : colorFor(n.community);
    document.getElementById('nodeMeta').innerHTML = n.community_label+' · '+(degree[n.id]||0)+' edges'+(n.is_phase?' · <b>Phase pinned on X-axis</b>':'')+'<br><span style="opacity:.5">'+n.source_file+' '+(n.source_location||'')+'</span>';
  })
  .onBackgroundClick(() => { document.getElementById('info').style.display='none'; });

// Pure force-directed layout — no artificial XY pull. Phases and stages
// are pinned (fx/fy/fz from the data); everything else floats freely and
// forms its own galaxy around the stage that links to it via the natural
// charge + link forces.
//
// Labelled nodes (file artifacts + stage scripts) get extra repulsion AND
// extra link-distance so 3+ of them can sit around the same stage
// without their floating labels overlapping into a smushed pile.
Graph.d3Force('charge').strength(n => (n.is_file_artifact || n.is_script) ? -780 : -260);
Graph.d3Force('link').distance(l => {
  const s = typeof l.source==='object' ? l.source : DATA.nodes.find(n=>n.id===l.source);
  const t = typeof l.target==='object' ? l.target : DATA.nodes.find(n=>n.id===l.target);
  const labeled = (s && (s.is_file_artifact || s.is_script)) ||
                  (t && (t.is_file_artifact || t.is_script));
  if (l.relation === 'precedes')   return 450;
  if (l.relation === 'implements') return labeled ? 160 : 100;
  if (labeled)                     return 150;
  if (l.rel_color)                 return 75;
  return 32;
});

const scene = Graph.scene();
scene.fog = new THREE.FogExp2(0x000a18, 0.00060);

// Central rail tube removed — phases are visually distinct enough on
// their own; the rail just blurred the read.
// Phase station ticks. Active phases (P1-P5, in pipeline) get cyan rail
// ticks; external phases (P6 web, P7 multi-cam) get amber ticks to make
// the visual distinction obvious without breaking the rail continuity.
DATA.phase_labels.forEach(p => {
  const isActive = p.kind !== 'external';
  const tickColor = isActive ? 0x00f5ff : 0xffb703;
  const tick = new THREE.Mesh(
    new THREE.BoxGeometry(4, 4, 220),
    new THREE.MeshBasicMaterial({color: tickColor, transparent: true,
                                  opacity: isActive ? 0.18 : 0.30, fog: true})
  );
  tick.position.set(p.x, -110, 0);
  scene.add(tick);
});

// Per-phase vertical sub-pipeline rails — one thin cyan line connecting
// each phase station to the column of stages anchored under it. Smaller
// radius than the horizontal rail so the hierarchy reads correctly.
DATA.stage_labels.forEach(s => {
  const parent = DATA.phase_labels.find(p => p.id === s.parent);
  if (!parent) return;
  // Length is |dy| because stages can be either above or below their
  // phase (alternating per phase index). Cylinder is always vertical.
  const dy = Math.abs(parent.y - s.y);
  const cx = parent.x;
  const cy = (parent.y + s.y) / 2;
  const subRail = new THREE.Mesh(
    new THREE.CylinderGeometry(0.9, 0.9, dy, 10),
    new THREE.MeshBasicMaterial({color: 0x00f5ff, transparent: true,
                                  opacity: 0.32, fog: false})
  );
  subRail.position.set(cx, cy, 0);
  scene.add(subRail);
});

// Orphan-cluster anchor: a small red marker on the floor far past Phase 7
// to telegraph "this stuff is NOT part of the pipeline". The gap between
// Phase 7 and this marker is ORPHAN_GAP wide, deliberately empty.
if (DATA.orphan_x !== null && DATA.orphan_count > 0) {
  const orphanMarker = new THREE.Mesh(
    new THREE.BoxGeometry(120, 4, 120),
    new THREE.MeshBasicMaterial({color: 0xff4d4d, transparent: true, opacity: 0.30, fog: true})
  );
  orphanMarker.position.set(DATA.orphan_x, -180, 0);
  scene.add(orphanMarker);
  const orphanLabel = makeLabelSprite('Orphans (' + DATA.orphan_count + ')', '#ff4d4d', 200);
  orphanLabel.position.set(DATA.orphan_x, -120, 0);
  scene.add(orphanLabel);
}

const dustCount = 1500;
const dustGeo = new THREE.BufferGeometry();
const dustPositions = new Float32Array(dustCount * 3);
const dustColors = new Float32Array(dustCount * 3);
const SPAN_X = (RAIL_MAX - RAIL_MIN) * 1.4, SPAN_Y = 700, SPAN_Z = 700;
for (let i = 0; i < dustCount; i++) {
  dustPositions[i*3] = (Math.random() - 0.5) * SPAN_X;
  dustPositions[i*3 + 1] = (Math.random() - 0.5) * SPAN_Y;
  dustPositions[i*3 + 2] = (Math.random() - 0.5) * SPAN_Z;
  const col = new THREE.Color().setHSL(0.5 + (Math.random() - 0.5) * 0.18, 0.55 + Math.random() * 0.4, 0.5 + Math.random() * 0.3);
  dustColors[i*3] = col.r; dustColors[i*3+1] = col.g; dustColors[i*3+2] = col.b;
}
dustGeo.setAttribute('position', new THREE.BufferAttribute(dustPositions, 3));
dustGeo.setAttribute('color', new THREE.BufferAttribute(dustColors, 3));
const dust = new THREE.Points(dustGeo, new THREE.PointsMaterial({size: 1.4, vertexColors: true, transparent: true, opacity: 0.55, sizeAttenuation: true, depthWrite: false, fog: true, blending: THREE.AdditiveBlending}));
scene.add(dust);

const bloomPass = new THREE.UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 1.45, 0.6, 0.20);
Graph.postProcessingComposer().addPass(bloomPass);

const clock = new THREE.Clock();
function tick() {
  const t = clock.getElapsedTime();
  scene.traverse(obj => { if (obj.userData && obj.userData.tick) obj.userData.tick(t); });
  dust.rotation.y = t * 0.012; dust.rotation.x = Math.sin(t * 0.05) * 0.05;
  requestAnimationFrame(tick);
}
tick();

// Default zoom adapts to the viewport so all 7 phases are always in frame
// regardless of window size / aspect ratio. We compute the camera Z that
// makes the pipeline width (with a little headroom) fit the camera's
// horizontal frustum at the current aspect ratio.
function fitZForAllPhases() {
  const xs = DATA.phase_labels.map(p => p.x);
  const ys = DATA.phase_labels.map(p => p.y);
  const width  = (Math.max(...xs) - Math.min(...xs));
  const height = (Math.max(...ys) - Math.min(...ys));
  const camera = Graph.camera();
  const vFov = ((camera && camera.fov) ? camera.fov : 50) * Math.PI / 180;
  const aspect = window.innerWidth / Math.max(1, window.innerHeight);
  const hFov = 2 * Math.atan(Math.tan(vFov / 2) * aspect);
  // Distance to fit width geometrically, with a 3× scale on top so we
  // also clear the orbital rings, the children galaxies and leave plenty
  // of black space around the rail.
  const zW = (width  / 2) / Math.tan(hFov / 2);
  const zH = (height / 2) / Math.tan(vFov / 2);
  const z  = Math.max(zW, zH) * 3.0 + 2500;
  return Math.max(z, 6500);   // hard floor — never closer than 6500
}
let fitZ = 6500;
function resetCamera() {
  fitZ = fitZForAllPhases();
  Graph.cameraPosition({x: 0, y: 380, z: fitZ}, {x:0,y:0,z:0}, 1500);
}
setTimeout(resetCamera, 120);

// Auto-rotate is OFF by default so the user starts on the resetCamera
// view and stays there. Press R to relaunch the gentle orbit.
let theta = 0, autoRotate = false;
setInterval(() => {
  if (!autoRotate) return;
  theta += 0.0014;
  // Orbit stays near fitZ — small ±8 % swing for parallax only.
  Graph.cameraPosition({
    x: (fitZ * 0.10) * Math.sin(theta * 0.45),
    y: 380 * Math.cos(theta) - 80 * Math.sin(theta),
    z: fitZ * (1 + 0.08 * Math.sin(theta * 0.6))
  });
}, 25);
elem.addEventListener('mousedown', () => autoRotate = false);
// Zoom-to-cursor: r152's OrbitControls predates the native `zoomToCursor`
// flag, so we replace its built-in wheel zoom with a custom handler that
// dollies the camera toward the world point under the mouse, then shifts
// the orbit target by the same delta. End result: the world point under
// the cursor stays under the cursor as you scroll.
Graph.controls().enableZoom = false;   // we own the wheel now
elem.addEventListener('wheel', (ev) => {
  ev.preventDefault();
  autoRotate = false;
  const ctrls = Graph.controls();
  const cam = Graph.camera();
  const rect = elem.getBoundingClientRect();
  const mx = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  const my = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  // World point under the cursor, projected to the orbit-target depth so
  // it sits on the visible scene shell rather than at infinity.
  const ndc = new THREE.Vector3(mx, my, 0.5).unproject(cam);
  const dir = ndc.sub(cam.position).normalize();
  const distToTarget = cam.position.distanceTo(ctrls.target);
  const cursorWorld = cam.position.clone().add(dir.multiplyScalar(distToTarget));
  // Per-tick zoom factor. <1 = zoom in, >1 = zoom out. Mouse wheels
  // typically deliver ev.deltaY ±100, trackpad pinch ±5..±30.
  const step = Math.min(0.5, Math.abs(ev.deltaY) * 0.0015);
  const factor = ev.deltaY < 0 ? (1 - step) : (1 + step);
  // Standard zoom-to-cursor: scale (camera - W) by factor; target shifts
  // by the same delta so the orbit pivot tracks the new view.
  const newCam = cursorWorld.clone().add(cam.position.clone().sub(cursorWorld).multiplyScalar(factor));
  const delta = newCam.clone().sub(cam.position);
  cam.position.copy(newCam);
  ctrls.target.add(delta);
  ctrls.update();
}, { passive: false });
elem.addEventListener('touchstart', () => autoRotate = false);
function panCamera(dx, dy) {
  autoRotate = false;
  const cam = Graph.camera();
  const ctrls = Graph.controls();
  cam.position.x += dx; cam.position.y += dy;
  ctrls.target.x += dx; ctrls.target.y += dy;
  ctrls.update();
}
window.addEventListener('keydown', e => {
  const step = e.shiftKey ? 220 : 90;
  if (e.key === 'ArrowLeft')      { e.preventDefault(); panCamera(-step, 0); }
  else if (e.key === 'ArrowRight'){ e.preventDefault(); panCamera(step, 0); }
  else if (e.key === 'ArrowUp')   { e.preventDefault(); panCamera(0, step); }
  else if (e.key === 'ArrowDown') { e.preventDefault(); panCamera(0, -step); }
  else if (e.key === 'r' || e.key === 'R') {
    autoRotate = true;
    const ctrls = Graph.controls(); ctrls.target.set(0, 0, 0); ctrls.update();
    resetCamera();
  }
});
window.addEventListener('resize', () => {
  Graph.width(window.innerWidth).height(window.innerHeight);
  // Re-fit the camera so all phases stay in view after a resize.
  fitZ = fitZForAllPhases();
  if (autoRotate) return;   // user is driving — don't yank their view
  Graph.cameraPosition({x: 0, y: 380, z: fitZ}, undefined, 600);
});
</script>
</body>
</html>
"""

HTML = HTML.replace('__DATA__', data_str).replace('__RAIL_MIN__', str(rail_min)).replace('__RAIL_MAX__', str(rail_max))
out_path = OUT / 'graph3d.html'
out_path.write_text(HTML)
print(f'      Wrote {out_path} ({out_path.stat().st_size} bytes)')
print()
print('Done. Open graphify-out/graph3d.html in a browser.')
