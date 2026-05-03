#!/usr/bin/env python3
"""
Regenerate graphParams3D.html — companion to graph3d.html.

Same visual idiom (phase rail + stage anchors + cyan/amber palette +
volumetric haze + bloom), but the floating satellites around each stage
are the **CLI parameters** that stage exposes, not its AST nodes.
Defaults render right next to the flag so the user can read the
"out-of-the-box" config in one glance.

Usage (from project root):
    /Users/nico/.local/share/uv/tools/graphifyy/bin/python graphify-out/regen_params.py

Both files share `graphify-out/` and link to each other (top-right
discrete button). Re-run this whenever a stage's CLI changes.

The parameter list below is curated by hand — argparse defaults can be
parsed automatically but help text and category tags would be lost, and
a hand-curated list is what reads cleanly in the visualisation. Keep it
in sync with src/p*_*.py and src/run_project.py when CLIs evolve.
"""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/Users/nico/Documents/Claude/Projects/RILH-AI-Vision')
OUT = ROOT / 'graphify-out'

# Phase + stage layout — same constants as regen.py so the two files
# read identically. Phase 6 / 7 have no orchestrated stages and are
# rendered as external references (amber).
SPACING = 380
ZIGZAG = 55
# Bigger vertical spacing than graph3d.html (70) because each stage
# now hosts its own galaxy of Input + Output sub-nodes plus a cloud
# of params on each side — at 70 they would visually overlap.
STAGE_VSPACING = 240
# Z offsets for the per-station Input / Output sub-nodes. Sub-nodes
# sit in front of (positive Z = toward the camera) and behind (negative
# Z) their parent station, so the param galaxies live in the Z axis
# and don't pollute the X-Y rail.
STAGE_IO_DZ   = 80
PHASE_IO_DZ   = 130
ORCH_IO_DZ    = 95
# Galaxy radius for the random-spherical param cloud around each I/O
# sub-node. Smaller than the I/O DZ so a stage's input cluster and
# output cluster remain visually separated.
PARAM_GALAXY_R       = 70
PHASE_DATA_GALAXY_R  = 90
ORCH_GALAXY_R        = 95

PHASE_ORDER = [
    ('readme_p1',          'Phase 1',  -3,  'active'),
    ('readme_p2',          'Phase 2',  -2,  'active'),
    ('readme_p3',          'Phase 3',  -1,  'active'),
    ('readme_p4',          'Phase 4',   0,  'active'),
    ('readme_p5',          'Phase 5',   1,  'active'),
    ('readme_p6_web',      'Phase 6',   2,  'external'),
    ('readme_p7_multicam', 'Phase 7',   3,  'external'),
]
PHASE_LONG = {
    'readme_p1':          'Phase 1 — Detect & track',
    'readme_p2':          'Phase 2 — Rink calibration',
    'readme_p3':          'Phase 3 — Entity recognition',
    'readme_p4':          'Phase 4 — Event detection (stub)',
    'readme_p5':          'Phase 5 — Statistics (stub)',
    'readme_p6_web':      'Phase 6 — Web platform (external)',
    'readme_p7_multicam': 'Phase 7 — Multi-cam / live (external)',
}
PHASE_STAGES = {
    'readme_p1': [
        ('p1_a_detect',     'Stage 1.a — Detect & track'),
        ('p1_pose_cache',   'Pose pre-extract (1.b/1.c)'),
        ('p1_b_teams',      'Stage 1.b — Teams'),
        ('p1_c_numbers',    'Stage 1.c — Numbers'),
    ],
    'readme_p2': [
        ('p2_a_rink',       'Stage 2.a — Rink keypoints'),
    ],
    'readme_p3': [
        ('p3_a_entities',   'Stage 3.a — Entities (Re-ID)'),
        ('p3_b_annotate',   'Stage 3.b — Annotate (final MP4)'),
    ],
    'readme_p4': [
        ('p4_a_events',     'Stage 4.a — Events (STUB)'),
    ],
    'readme_p5': [
        ('p5_a_stats',      'Stage 5.a — Stats (STUB)'),
    ],
    'readme_p6_web':      [],
    'readme_p7_multicam': [],
}

# Data flow at the phase level — what files / artefacts each phase
# consumes (Input) and produces (Output). Renders as a galaxy of
# small artefact nodes around the phase's Input / Output sub-nodes,
# mirroring the per-stage param view one level up.
PHASE_DATA = {
    'readme_p1': {
        'input':  ['video.mp4'],
        'output': ['p1_a_detections.json', 'p1_b_teams.json',
                   'p1_c_numbers.json', 'p1_pose_cache.pkl',
                   'teams_preview.png', 'annotated_raw.mp4'],
    },
    'readme_p2': {
        'input':  ['video.mp4'],
        'output': ['p2_a_rink_keypoints.json'],
    },
    'readme_p3': {
        'input':  ['video.mp4', 'p1_a_detections.json',
                   'p1_b_teams.json', 'p1_c_numbers.json',
                   'p3_a_entities.json'],
        'output': ['p3_a_entities.json', 'annotated.mp4'],
    },
    'readme_p4': {
        'input':  ['p1_a_detections.json', 'p3_a_entities.json'],
        'output': ['p4_a_events.json'],
    },
    'readme_p5': {
        'input':  ['p3_a_entities.json', 'p1_c_numbers.json'],
        'output': ['p5_a_stats.json'],
    },
    'readme_p6_web':      {
        'input':  ['runs/runNN/'],
        'output': ['Web UI / API'],
    },
    'readme_p7_multicam': {
        'input':  ['runs/runNN/', 'multi-cam streams'],
        'output': ['Live RTMP/HLS', 'Mobile app'],
    },
}


def spiral_offset(rank, base_radius, dir_vec,
                  az_step_deg=70.0, radial_step=0.15,
                  elev_offset_deg=15.0, elev_step_deg=29.0,
                  elev_max_deg=60.0):
    """Deterministic 3-D spiral around the I/O axis. Each rank
    advances:

      • azimuth (around `dir_vec`) by `az_step_deg` (default 70°,
        coprime with 360 so consecutive params never align)
      • radius by `radial_step` × base_radius
        (additive: r(n) = base × (1 + 0.15 n))
      • elevation off the perpendicular plane (toward the I/O axis)
        following a triangle wave in [0, `elev_max_deg`] — this is
        the bit that breaks the otherwise coplanar layout, putting
        each param at a different depth along the axis.

    Elevation stays ≥ 0, so the offset always points into the
    half-space away from the parent station — no extra hemisphere
    clamp needed."""
    radius = base_radius * (1.0 + radial_step * rank)
    az = math.radians(az_step_deg * rank)

    # Triangle wave elevation in [0, elev_max_deg]: lin grow, bounce
    # back, repeat. Coprime step (29° vs 60° max) keeps the cycle
    # from re-aligning with the azimuth wheel for at least a dozen
    # ranks, so the spiral looks "noisy" in 3-D rather than periodic.
    raw = (elev_offset_deg + elev_step_deg * rank) % (2.0 * elev_max_deg)
    elev_deg = raw if raw <= elev_max_deg else 2.0 * elev_max_deg - raw
    elev = math.radians(elev_deg)

    # Orthonormal basis (u, v) perpendicular to dir_vec.
    dx, dy, dz = dir_vec
    helper = (1.0, 0.0, 0.0) if abs(dy) > 0.9 else (0.0, 1.0, 0.0)
    ux = dy * helper[2] - dz * helper[1]
    uy = dz * helper[0] - dx * helper[2]
    uz = dx * helper[1] - dy * helper[0]
    norm = math.sqrt(ux * ux + uy * uy + uz * uz) or 1.0
    ux, uy, uz = ux / norm, uy / norm, uz / norm
    # v = dir_vec × u (orthonormal because dir_vec is a unit vector)
    vx = dy * uz - dz * uy
    vy = dz * ux - dx * uz
    vz = dx * uy - dy * ux

    cos_e = math.cos(elev)
    sin_e = math.sin(elev)
    cos_a = math.cos(az)
    sin_a = math.sin(az)

    # Direction = cos(elev) × (azimuthal vector in u-v plane)
    #           + sin(elev) × dir_vec  (push out along the axis)
    nx = cos_e * (cos_a * ux + sin_a * vx) + sin_e * dx
    ny = cos_e * (cos_a * uy + sin_a * vy) + sin_e * dy
    nz = cos_e * (cos_a * uz + sin_a * vz) + sin_e * dz

    return (radius * nx, radius * ny, radius * nz)


def io_axis_for_stage(stage_idx):
    """Return (input_dir, output_dir) unit vectors in the X-Z plane
    for a stage's I/O sub-nodes, given its index within its phase.

    Adjacent stages rotate 45° and flip input ↔ output so two stages
    of the same phase never share an I/O orientation:
       i=0 → input +Z (0°)        / output -Z
       i=1 → input rotated 45° AND on the opposite side (= 225°)
       i=2 → input  +X (90°)      / output -X
       i=3 → input at 315° (= -45°+180°) / output at 135°
    The pattern repeats every 8 stages but no real phase has more
    than 4 stages, so each phase column gets a unique constellation.
    """
    angle_deg = stage_idx * 45
    if stage_idx % 2 == 1:
        angle_deg += 180
    a = math.radians(angle_deg)
    in_dir  = (math.sin(a),  0.0,  math.cos(a))
    out_dir = (-in_dir[0],   0.0,  -in_dir[2])
    return in_dir, out_dir


def classify_io(flag, type_):
    """Bucket a CLI flag/positional into 'input' or 'output'.

    Heuristic, applied to every entry in STAGE_PARAMS / ORCHESTRATOR:
      - positional args (no leading `--`) → input
      - flags whose name contains `output`            → output
      - flags whose name contains `debug` AND `dir`   → output (write
        a folder of debug artefacts)
      - everything else (config: thresholds, modes,
        engines, model paths, sizes, …) → input
    """
    if not flag:
        return 'input'
    f = flag.lower()
    if not f.startswith('--'):
        return 'input'
    if 'output' in f:
        return 'output'
    if 'debug' in f and 'dir' in f:
        return 'output'
    return 'input'

# Each stage's CLI parameters. `kind` drives the color:
#   io       — input / output paths       (mint)
#   model    — model selection / weights  (pink)
#   tracker  — tracker config             (blue)
#   threshold— numeric thresholds         (yellow)
#   size     — image / batch sizes        (cyan)
#   mode     — boolean flags              (purple)
#   speed    — perf / sub-sampling        (green)
#   engine   — pluggable engines          (orange)
#   debug    — debug-only outputs         (grey)
# `default` is the literal CLI default. None = required argument with
# no default (typically a positional). For store_true flags we render
# default `False`. For choices, we list them under `choices`.
STAGE_PARAMS = {
    'p1_a_detect': [
        ('video',           None,           'io',        'pos', 'Source MP4 path'),
        ('--output',        'runs/latest',  'io',        None,  'Run folder'),
        ('--model',         'yolo11m.pt',   'model',     None,  'COCO YOLO (ignored if --hockey-model)'),
        ('--hockey-model',  False,          'model',     'flag','HockeyAI YOLOv8m (recommended)'),
        ('--tracker',       'bytetrack.yaml','tracker',  None,  'Tracker config (yaml)'),
        ('--conf',          0.3,            'threshold', None,  'Detection confidence'),
        ('--imgsz',         1280,           'size',      None,  'Inference image size'),
        ('--training-mode', False,          'mode',      'flag','Disable 1-puck-per-frame filter'),
        ('--detect-fps',    30.0,           'speed',     None,  'Target detection fps (stride)'),
    ],
    'p1_pose_cache': [
        ('detections_json', None,           'io',        'pos', 'p1_a_detections.json'),
        ('video',           None,           'io',        'pos', 'Source MP4'),
        ('--output',        '<dir>/p1_pose_cache.pkl', 'io', None, 'Cache file'),
        ('--pose-model',    'yolo11n-pose.pt','model',   None,  'YOLO pose weights'),
        ('--pose-imgsz',    1280,           'size',      None,  'Pose inference size'),
        ('--samples-per-track', 15,         'size',      None,  'Top-N frames per track (union)'),
    ],
    'p1_b_teams': [
        ('detections_json', None,           'io',        'pos', 'p1_a_detections.json'),
        ('video',           None,           'io',        'pos', 'Source MP4'),
        ('--output',        '<dir>/p1_b_teams.json','io',None,  ''),
        ('--samples-per-track', 8,          'size',      None,  'Crops per track'),
        ('--pose-model',    'yolo11n-pose.pt','model',   None,  ''),
        ('--pose-imgsz',    1280,           'size',      None,  ''),
        ('--space',         'hsv',          'engine',    'choice', 'Color space for HSV engine', ['hsv','bgr']),
        ('--grid',          '3x2',          'size',      None,  'Multi-point torso grid'),
        ('--team-engine',   'hsv',          'engine',    'choice', 'Team-classification backend', ['hsv','osnet','siglip','contrastive']),
        ('--contrastive-checkpoint', None,  'model',     None,  'models/contrastive_team_rilh.pt'),
        ('--ref-classifier',None,           'model',     None,  'models/ref_classifier_rilh.pt (post-hoc)'),
        ('--pose-cache',    '<dir>/p1_pose_cache.pkl', 'speed', None, 'Read pose from cache'),
        ('--preview-cols',  10,             'debug',     None,  'Preview PNG cols'),
    ],
    'p1_c_numbers': [
        ('detections_json', None,           'io',        'pos', 'p1_a_detections.json'),
        ('video',           None,           'io',        'pos', 'Source MP4'),
        ('--output',        '<dir>/p1_c_numbers.json','io',None,''),
        ('--pose-model',    'yolo11n-pose.pt','model',   None,  ''),
        ('--samples-per-track', 15,         'size',      None,  'OCR crops per track'),
        ('--frame-stride',  None,           'mode',      None,  'Dataset-mode: every Nth frame'),
        ('--ocr-min-conf',  0.30,           'threshold', None,  'Per-sample OCR confidence floor'),
        ('--pose-imgsz',    1280,           'size',      None,  ''),
        ('--ocr-batch',     32,             'size',      None,  'PARSeq batch size'),
        ('--parseq-checkpoint', None,       'model',     None,  'models/parseq_hockey_rilh.pt'),
        ('--pose-cache',    '<dir>/p1_pose_cache.pkl', 'speed', None, 'Read pose from cache'),
        ('--debug-crops-dir', None,         'debug',     None,  'Save every OCR crop'),
    ],
    'p2_a_rink': [
        ('video',           None,           'io',        'pos', 'Source MP4'),
        ('--output',        'runs/test05',  'io',        None,  ''),
        ('--samples',       5,              'size',      None,  'Frames sampled for keypoints'),
        ('--conf',          0.25,           'threshold', None,  'HockeyRink confidence'),
        ('--imgsz',         1280,           'size',      None,  ''),
        ('--min-kp-conf',   0.3,            'threshold', None,  'Per-keypoint conf floor'),
    ],
    'p3_a_entities': [
        ('detections_json', None,           'io',        'pos', 'p1_a_detections.json'),
        ('teams_json',      None,           'io',        'pos', 'p1_b_teams.json'),
        ('numbers_json',    None,           'io',        'pos', 'p1_c_numbers.json'),
        ('video',           None,           'io',        'pos', 'Source MP4'),
        ('--output',        '<dir>/p3_a_entities.json','io',None,''),
        ('--samples-per-track', 8,          'size',      None,  'Embeddings per track'),
        ('--batch-size',    16,             'size',      None,  ''),
        ('--sim-threshold', 0.65,           'threshold', None,  'Cosine sim floor for non-OCR merges'),
        ('--ocr-bonus',     10.0,           'threshold', None,  'OCR-seeded merge weight'),
        ('--ocr-conflict-conf-floor', 0.40, 'threshold', None,  'Conflict gate'),
        ('--goalie-bonus',  0.05,           'threshold', None,  ''),
        ('--team-conf-floor', 0.67,         'threshold', None,  'P1.b vote-confidence floor'),
        ('--max-overlap-frames', 0,         'threshold', None,  'Strict 0 = no overlap'),
        ('--osnet',         'osnet_x0_25',  'model',     None,  'Torchreid Re-ID backbone'),
    ],
    'p3_b_annotate': [
        ('detections_json', None,           'io',        'pos', 'p1_a_detections.json'),
        ('numbers_json',    None,           'io',        'pos', 'p1_c_numbers.json'),
        ('video',           None,           'io',        'pos', 'Source MP4'),
        ('--output',        None,           'io',        None,  'Annotated MP4 (required)'),
        ('--color-samples', 6,              'size',      None,  'Fallback when teams.json missing'),
        ('--debug-frames-dir', None,        'debug',     None,  'PNG every N frames'),
        ('--debug-frames-step', 10,         'debug',     None,  ''),
    ],
    'p4_a_events': [
        ('detections_json', None,           'io',        'pos', 'p1_a_detections.json'),
        ('entities_json',   None,           'io',        'pos', 'p3_a_entities.json'),
        ('--output',        None,           'io',        None,  'Required; stub marker'),
    ],
    'p5_a_stats': [
        ('entities_json',   None,           'io',        'pos', 'p3_a_entities.json'),
        ('numbers_json',    None,           'io',        'pos', 'p1_c_numbers.json'),
        ('--output',        None,           'io',        None,  'Required; stub marker'),
    ],
}

# run_project.py — the orchestrator. Lives ABOVE the phase rail, drives
# everything via pass-through. Rendered as a "Phase 0" station (no stage
# children, but its own parameter halo).
ORCHESTRATOR = {
    'id':     'run_project',
    'label':  'Pipeline orchestrator (run_project.py)',
    'params': [
        ('video',           None,           'io',        'pos', 'Input MP4'),
        ('--output',        None,           'io',        None,  'Run folder (required)'),
        ('--skip-p1',       False,          'mode',      'flag',''),
        ('--skip-p2',       False,          'mode',      'flag',''),
        ('--skip-p3',       False,          'mode',      'flag',''),
        ('--skip-p4',       False,          'mode',      'flag',''),
        ('--skip-p5',       False,          'mode',      'flag',''),
        ('--force',         False,          'mode',      'flag','Re-run all enabled stages'),
        ('--hockey-model',  False,          'model',     'flag','→ Stage 1.a'),
        ('--training-mode', False,          'mode',      'flag','→ Stage 1.a'),
        ('--detect-fps',    30.0,           'speed',     None,  '→ Stage 1.a (frame stride)'),
        ('--model',         None,           'model',     None,  '→ Stage 1.a (COCO YOLO)'),
        ('--conf',          0.3,            'threshold', None,  '→ Stage 1.a'),
        ('--imgsz',         1280,           'size',      None,  '→ Stage 1.a'),
        ('--tracker',       'bytetrack.yaml','tracker',  None,  '→ Stage 1.a'),
        ('--pose-model',    'yolo26l-pose.pt','model',   None,  '→ Pose pre-extract + 1.b + 1.c'),
        ('--team-engine',   'hsv',          'engine',    'choice','→ Stage 1.b', ['hsv','osnet','siglip','contrastive']),
        ('--contrastive-checkpoint', None,  'model',     None,  '→ Stage 1.b'),
        ('--ref-classifier',None,           'model',     None,  '→ Stage 1.b (opt-in)'),
        ('--parseq-checkpoint', None,       'model',     None,  '→ Stage 1.c'),
    ],
}

# Category → palette colour. Picked from the same palette graph3d.html
# uses so the two views feel like siblings.
KIND_COLOR = {
    'io':        '#06ffa5',  # mint
    'model':     '#ff006e',  # pink
    'tracker':   '#3a86ff',  # blue
    'threshold': '#fee440',  # yellow
    'size':      '#4cc9f0',  # cyan
    'mode':      '#c77dff',  # purple
    'speed':     '#80ed99',  # green
    'engine':    '#fb5607',  # orange
    'debug':     '#a0a0a0',  # grey
}
KIND_LABEL = {
    'io':        'I/O paths',
    'model':     'Models / weights',
    'tracker':   'Tracker config',
    'threshold': 'Thresholds / scores',
    'size':      'Image / batch size',
    'mode':      'Modes / boolean flags',
    'speed':     'Speed / frame stride',
    'engine':    'Pluggable engines',
    'debug':     'Debug outputs',
}


def random_galaxy_offset(seed_str, radius=110.0):
    """Deterministic random 3-D spherical offset around a parent
    station. Hash the seed string so the same param always lands at
    the same spot on regeneration (galaxy stays stable between
    refreshes), but the angular distribution is uncorrelated across
    siblings — no visible regular pattern, just a random cloud.

    Returns (dx, dy, dz). Caller adds the parent station's
    coordinates and pins the param with fx/fy/fz so the force solver
    never touches it. Radius factor in [0.50, 1.00] gives the cloud
    a bit of depth without ever collapsing onto the centre."""
    h = hashlib.sha1(seed_str.encode()).digest()
    theta = (int.from_bytes(h[0:4], 'big') / 0xffffffff) * 2.0 * math.pi
    # Uniform spherical distribution: cos(phi) drawn from [-1, +1]
    # rather than phi from [0, π] so points don't pile up at the poles.
    cos_phi = (int.from_bytes(h[4:8], 'big') / 0xffffffff) * 2.0 - 1.0
    sin_phi = math.sqrt(max(0.0, 1.0 - cos_phi * cos_phi))
    rfac = 0.50 + (int.from_bytes(h[8:12], 'big') / 0xffffffff) * 0.50
    r = radius * rfac
    return (r * sin_phi * math.cos(theta),
            r * cos_phi,
            r * sin_phi * math.sin(theta))


def fmt_default(d, kind, type_):
    """Render a parameter's default for the floating label. None becomes
    a clear placeholder so the user knows it's optional / required."""
    if type_ == 'flag':
        return 'False'
    if type_ == 'pos':
        return '(required)' if d is None else f'={d}'
    if d is None:
        return '(none)'
    if isinstance(d, str):
        return d
    if isinstance(d, bool):
        return str(d)
    return str(d)


# --- Build the layout payload ------------------------------------------------
phase_labels = []
stage_labels = []
nodes = []
links = []

PINNED = {}

# Phase + stage anchors — same zigzag as regen.py so the rail visually
# matches the AST view. Each phase, stage, and the orchestrator host
# 2 small Input + Output sub-nodes; CLI params (or data-file artefacts
# at the phase level) are the satellites that orbit those sub-nodes.
def _add_io_subnodes(parent_id, parent_label, parent_x, parent_y,
                     parent_z, dz, parent_phase=None, parent_stage=None,
                     in_dir=(0.0, 0.0, 1.0),
                     out_dir=(0.0, 0.0, -1.0)):
    """Create the Input + Output sub-nodes for a parent station.
    Pinned at parent ± dz along the given unit direction vectors so
    the I/O axis can rotate per stage (see io_axis_for_stage).
    parent_phase / parent_stage propagate so sidebar visibility
    cascades hide the sub-nodes when their owner is hidden. Returns
    (input_id, output_id, in_dir, out_dir) — the direction vectors
    are echoed back so the caller can re-use them when placing the
    param / data-file satellites."""
    in_id  = f'{parent_id}__input'
    out_id = f'{parent_id}__output'
    for sub_id, role, dvec in ((in_id, 'input', in_dir),
                                (out_id, 'output', out_dir)):
        nodes.append({
            'id': sub_id,
            'label': 'Input' if role == 'input' else 'Output',
            'kind': '_io',
            'io_role': role,
            'parent': parent_id,
            'parent_label': parent_label,
            'parent_phase': parent_phase,
            'parent_stage': parent_stage,
            'io_dir': list(dvec),
            'fx': parent_x + dvec[0] * dz,
            'fy': parent_y + dvec[1] * dz,
            'fz': parent_z + dvec[2] * dz,
        })
    return in_id, out_id, in_dir, out_dir


for phase_idx, (pid, short, slot, kind) in enumerate(PHASE_ORDER):
    stage_dir = -1 if phase_idx % 2 == 0 else +1
    py = stage_dir * ZIGZAG
    px = slot * SPACING
    pz = 0
    PINNED[pid] = (px, py, pz)
    phase_labels.append({
        'id': pid, 'short': short, 'long': PHASE_LONG[pid],
        'x': px, 'y': py, 'kind': kind,
    })
    nodes.append({
        'id': pid, 'label': PHASE_LONG[pid],
        'kind': '_phase', 'phase_kind': kind,
        'fx': px, 'fy': py, 'fz': pz,
    })

    # Phase-level Input / Output sub-nodes + their data-file satellites
    # (galaxy of small artefact nodes representing the data flow at
    # the phase level: what comes in, what goes out).
    p_in_id, p_out_id, p_in_dir, p_out_dir = _add_io_subnodes(
        pid, PHASE_LONG[pid], px, py, pz, PHASE_IO_DZ,
        parent_phase=pid)
    p_data = PHASE_DATA.get(pid, {'input': [], 'output': []})
    for role, parent_io_id, dvec in (('input', p_in_id, p_in_dir),
                                      ('output', p_out_id, p_out_dir)):
        items = p_data.get(role, [])
        for rank, fname in enumerate(items):
            data_id = f'{pid}__{role}__{fname.replace("/", "_")}'
            ox, oy, oz = spiral_offset(rank, PHASE_DATA_GALAXY_R, dvec)
            nodes.append({
                'id': data_id,
                'label': fname,
                'kind': '_phase_data',
                'io_role': role,
                'parent_io': parent_io_id,
                'parent_phase': pid,
                'fx': px + dvec[0] * PHASE_IO_DZ + ox,
                'fy': py + dvec[1] * PHASE_IO_DZ + oy,
                'fz': pz + dvec[2] * PHASE_IO_DZ + oz,
            })

    for i, (sid, slabel) in enumerate(PHASE_STAGES.get(pid, [])):
        # Stages stay stacked vertically below (or above) their phase.
        # Vertical spacing bumped to 240 so each stage's I/O galaxies
        # don't collide with the next one in the column.
        sx = px
        sy = py + stage_dir * STAGE_VSPACING * (i + 1)
        sz = 0
        full_sid = 'stage_' + sid
        PINNED[full_sid] = (sx, sy, sz)
        stage_labels.append({
            'id': full_sid, 'short': slabel, 'long': slabel,
            'x': sx, 'y': sy, 'parent': pid,
        })
        nodes.append({
            'id': full_sid, 'label': slabel, 'kind': '_stage',
            'parent': pid, 'fx': sx, 'fy': sy, 'fz': sz,
        })

        # Per-stage Input / Output sub-nodes — pinned at sx, sy with
        # ±STAGE_IO_DZ along Z. Every CLI flag of the stage attaches
        # to one of these depending on classify_io().
        # Per-stage I/O axis rotation. Each stage in a phase rotates
        # its Input/Output by 45° AND flips them so two consecutive
        # stages within a phase column never share an orientation.
        s_in_dir_v, s_out_dir_v = io_axis_for_stage(i)
        s_in_id, s_out_id, _s_in_dir, _s_out_dir = _add_io_subnodes(
            full_sid, slabel, sx, sy, sz, STAGE_IO_DZ,
            parent_phase=pid, parent_stage=full_sid,
            in_dir=s_in_dir_v, out_dir=s_out_dir_v)

        # Spiral layout per (stage, role): 120° angular step, +15 %
        # of base radius per rank. Ranks are independent for input
        # vs output so each side draws its own three-spoke spiral.
        stage_params = STAGE_PARAMS.get(sid, [])
        role_rank = {'input': 0, 'output': 0}

        for tup in stage_params:
            if len(tup) == 5:
                flag, default, kind_, type_, helptext = tup
                choices = None
            else:
                flag, default, kind_, type_, helptext, choices = tup
            role = classify_io(flag, type_)
            parent_io_id = s_in_id if role == 'input' else s_out_id
            dvec = s_in_dir_v if role == 'input' else s_out_dir_v
            param_id = f'{full_sid}__param__{flag.lstrip("-")}'
            rank = role_rank[role]
            role_rank[role] += 1
            ox, oy, oz = spiral_offset(rank, PARAM_GALAXY_R, dvec)
            nodes.append({
                'id': param_id,
                'label': flag,
                'default': fmt_default(default, kind_, type_),
                'help': helptext or '',
                'choices': choices,
                'kind': kind_,
                'param_type': type_,
                'io_role': role,
                'parent_io': parent_io_id,
                'parent_stage': full_sid,
                'parent_phase': pid,
                # Pinned position around the I/O sub-node along the
                # per-stage rotated axis (dvec). Spreads in 3-D —
                # sides, heights, depth — while staying off the I/O
                # axis side reserved for the parent station.
                'fx': sx + dvec[0] * STAGE_IO_DZ + ox,
                'fy': sy + dvec[1] * STAGE_IO_DZ + oy,
                'fz': sz + dvec[2] * STAGE_IO_DZ + oz,
            })

# Orchestrator station — pinned ABOVE Phase 1, slightly back in Z so it
# reads as "at the helm" without colliding with the phase rail.
ORCH_ID = ORCHESTRATOR['id']
ORCH_X = PHASE_ORDER[0][2] * SPACING - SPACING * 0.55   # left of Phase 1
ORCH_Y = 320
ORCH_Z = 0
PINNED[ORCH_ID] = (ORCH_X, ORCH_Y, ORCH_Z)
nodes.append({
    'id': ORCH_ID, 'label': ORCHESTRATOR['label'], 'kind': '_orchestrator',
    'fx': ORCH_X, 'fy': ORCH_Y, 'fz': ORCH_Z,
})
o_in_id, o_out_id, o_in_dir, o_out_dir = _add_io_subnodes(
    ORCH_ID, ORCHESTRATOR['label'],
    ORCH_X, ORCH_Y, ORCH_Z, ORCH_IO_DZ,
    parent_stage=ORCH_ID)
# Same 120° / +15 % spiral as the per-stage params.
_orch_role_rank = {'input': 0, 'output': 0}

for tup in ORCHESTRATOR['params']:
    if len(tup) == 5:
        flag, default, kind_, type_, helptext = tup
        choices = None
    else:
        flag, default, kind_, type_, helptext, choices = tup
    role = classify_io(flag, type_)
    parent_io_id = o_in_id if role == 'input' else o_out_id
    dvec = o_in_dir if role == 'input' else o_out_dir
    param_id = f'orch__param__{flag.lstrip("-")}'
    rank = _orch_role_rank[role]
    _orch_role_rank[role] += 1
    ox, oy, oz = spiral_offset(rank, ORCH_GALAXY_R, dvec)
    nodes.append({
        'id': param_id, 'label': flag,
        'default': fmt_default(default, kind_, type_),
        'help': helptext or '',
        'choices': choices,
        'kind': kind_, 'param_type': type_,
        'io_role': role,
        'parent_io': parent_io_id,
        'parent_stage': ORCH_ID,
        'parent_phase': None,
        'fx': ORCH_X + dvec[0] * ORCH_IO_DZ + ox,
        'fy': ORCH_Y + dvec[1] * ORCH_IO_DZ + oy,
        'fz': ORCH_Z + dvec[2] * ORCH_IO_DZ + oz,
    })

# Phase → Phase precedes edges — same visual idiom as graph3d.html
# (cyan, thick, with directional particles flowing P1 → P7).
for i in range(len(PHASE_ORDER) - 1):
    links.append({
        'source': PHASE_ORDER[i][0],
        'target': PHASE_ORDER[i + 1][0],
        'kind': 'precedes',
    })

# Origin edges — every node carries a thin link back to whatever
# station "owns" it, the same way graph3d.html shows every code node
# tied to its phase / stage / file. Hierarchy:
#   param        → its I/O sub-node       (parent_io)
#   phase_data   → its I/O sub-node       (parent_io)
#   _io sub-node → its parent station     (parent)
#   _stage       → its phase              (parent)
# Rendered dim so they don't compete with the cyan precedes rail.
for n in nodes:
    parent_id = None
    if n.get('parent_io'):
        parent_id = n['parent_io']
    elif n.get('kind') == '_io' and n.get('parent'):
        parent_id = n['parent']
    elif n.get('kind') == '_stage' and n.get('parent'):
        parent_id = n['parent']
    if parent_id:
        links.append({
            'source': parent_id,
            'target': n['id'],
            'kind': 'origin',
            'io_role': n.get('io_role'),
        })

payload = {
    'nodes': nodes,
    'links': links,
    'phase_labels': phase_labels,
    'stage_labels': stage_labels,
    'orchestrator_id': ORCH_ID,
    'kind_colors': KIND_COLOR,
    'kind_labels': KIND_LABEL,
}
data_str = json.dumps(payload).replace('</', '<\\/')
rail_min, rail_max = -4 * SPACING, 4 * SPACING

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RILH-AI-Vision — Pipeline Params 3D</title>
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
#filters .row { display:flex; align-items:center; padding:3px 4px; cursor:pointer; user-select:none; border-radius:4px; transition: background 0.12s, opacity 0.12s; }
#filters .row:hover { background:rgba(0,220,255,0.08); }
#filters .row.hidden { opacity:0.32; text-decoration:line-through; }
#filters .row .dot { width:10px; height:10px; border-radius:50%; margin-right:8px; box-shadow:0 0 8px currentColor; flex:0 0 10px; cursor:pointer; transition:transform 0.12s, box-shadow 0.12s; }
#filters .row .dot:hover { transform:scale(1.45); box-shadow:0 0 14px currentColor; }
#filters .row .name { flex:1; cursor:pointer; transition:color 0.12s; }
#filters .row .name:hover { color:#fff; text-shadow:0 0 6px rgba(0,245,255,0.7); }
#filters .row .count { opacity:0.4; font-size:9.5px; margin-left:6px; }
#filters .row.phase { font-weight:600; color:#cfeaff; margin-top:6px; }
#filters .row.phase:first-child { margin-top:0; }
#filters .row.phase.external .dot { background:#ffb703; color:#ffb703; }
#filters .row.phase.active .dot { background:#00f5ff; color:#00f5ff; }
#filters .row.stage { padding-left:22px; opacity:0.85; font-size:10.5px; }
#filters .row.stage .dot { background:#cfeaff; color:#cfeaff; width:7px; height:7px; flex:0 0 7px; }
#filters .row.cat { padding-left:40px; font-size:10px; opacity:0.78; }
#filters .row.cat .dot { width:8px; height:8px; flex:0 0 8px; }
#filters .row.param { padding-left:60px; font-size:10px; opacity:0.85; font-style:italic; }
#filters .row.param .dot { width:6px; height:6px; flex:0 0 6px; box-shadow:none; border:1px solid currentColor; }
#filters .actions { display:flex; gap:6px; margin-top:8px; padding-top:8px; border-top:1px solid rgba(0,220,255,0.12); }
#filters .actions button { flex:1; background:rgba(0,220,255,0.1); border:1px solid rgba(0,220,255,0.25); color:#cfeaff; font-size:10px; padding:4px 8px; border-radius:4px; cursor:pointer; }
#filters .actions button:hover { background:rgba(0,220,255,0.2); }
#filters .search { margin:0 0 10px 0; }
#filters .search input { width:100%; box-sizing:border-box; padding:6px 10px; background:rgba(0,12,24,0.7); border:1px solid rgba(0,220,255,0.25); border-radius:5px; color:#cfeaff; font-size:11px; outline:none; transition:border-color 0.15s, background 0.15s; }
#filters .search input:focus { border-color:rgba(0,245,255,0.55); background:rgba(0,18,32,0.85); }
#filters .search input::placeholder { color:rgba(207,234,255,0.35); }
#filters .row.search-hidden { display:none; }
#filters .row.search-hit .name { color:#ffd60a; }
#filters .row .solo { margin-left:6px; padding:1px 5px; border-radius:3px; font-size:10px; line-height:1; cursor:pointer; opacity:0.4; transition: opacity 0.12s, background 0.12s, color 0.12s; user-select:none; }
#filters .row:hover .solo { opacity:0.85; }
#filters .row .solo:hover { opacity:1; background:rgba(255,214,10,0.15); color:#ffd60a; }
#filters .row.soloed .solo { opacity:1; background:#ffd60a; color:#000; box-shadow:0 0 6px rgba(255,214,10,0.65); }
#filters .row.soloed > .name { color:#ffd60a; }
#info { position:absolute; bottom:16px; left:16px; padding:12px 16px; max-width:520px; font-size:12px; display:none; }
#info .lbl { color:#6affff; font-size:14px; font-weight:600; margin-bottom:4px; }
#info .meta { opacity:0.85; font-size:11px; line-height:1.5; }
#info .meta .row { margin:2px 0; }
#info .meta .key { color:#a0a0a0; min-width:60px; display:inline-block; }
#info .meta code { font-family: ui-monospace, SFMono-Regular, monospace; color:#6affff; background:rgba(0,30,55,0.5); padding:1px 5px; border-radius:3px; }
#controls { position:absolute; bottom:16px; right:16px; padding:8px 12px; font-size:11px; opacity:0.75; }
#controls span { color:#6affff; }
#navbtn { position:absolute; top:16px; right:16px; padding:8px 14px; font-size:11px; text-decoration:none; color:#cfeaff; opacity:0.85; transition:opacity 0.12s, border-color 0.12s, color 0.12s; }
#navbtn:hover { opacity:1; color:#6affff; border-color:rgba(0,245,255,0.55); }
#navbtn .arrow { color:#6affff; margin-right:6px; font-weight:600; }
.scanline::after { content:""; position:absolute; inset:0; pointer-events:none; background:repeating-linear-gradient(0deg, rgba(0,220,255,0.02) 0px, rgba(0,220,255,0.02) 1px, transparent 1px, transparent 3px); z-index:5; }
.vignette { position:absolute; inset:0; pointer-events:none; z-index:6; background: radial-gradient(ellipse at center, transparent 40%, rgba(0,0,0,0.55) 100%); }
</style>
</head>
<body class="scanline">
<div id="graph"></div>
<div class="vignette"></div>
<a id="navbtn" class="panel" href="graph3d.html"><span class="arrow">←</span>Pipeline Graph</a>
<div id="leftcol">
  <div id="overlay" class="panel">
    <h1>▲ RILH-AI-Vision · CLI Params</h1>
    <div class="sub">Each stage's flags + defaults · click a node for full help</div>
    <div class="stat"><b>Stages:</b> <span id="sc"></span></div>
    <div class="stat"><b>Parameters:</b> <span id="pc"></span></div>
    <div class="stat"><b>Orchestrator:</b> <span id="oc"></span> pass-through</div>
  </div>
  <div id="filters" class="panel">
    <div id="pipelineSection" class="section collapsed">
      <h2 data-target="pipelineSection"><span class="caret">▾</span>Pipeline Filter</h2>
      <div class="body">
        <p class="hint">Click <b>name</b> to flash · click <b>dot</b> to hide/show · click <b>◉</b> to solo (show only this + its content).</p>
        <div class="search"><input id="filterSearch" type="text" placeholder="Search params / files…" autocomplete="off" spellcheck="false" /></div>
        <div id="filterTree"></div>
        <div class="actions">
          <button id="filterAll">Show all</button>
          <button id="filterNone">Hide all</button>
        </div>
      </div>
    </div>
    <div id="categoriesSection" class="section collapsed">
      <h2 data-target="categoriesSection"><span class="caret">▾</span>Param Categories</h2>
      <div class="body">
        <p class="hint">Toggle a category to hide every parameter of that kind.</p>
        <div id="catItems"></div>
        <div class="actions">
          <button id="catAll">Show all</button>
          <button id="catNone">Hide all</button>
        </div>
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
const KIND_COLOR = DATA.kind_colors;
const KIND_LABEL = DATA.kind_labels;

document.getElementById('sc').textContent = DATA.stage_labels.length;
document.getElementById('pc').textContent = DATA.nodes.filter(n => n.kind && !n.kind.startsWith('_')).length;
document.getElementById('oc').textContent = DATA.nodes.filter(n => n.parent_stage === DATA.orchestrator_id).length;

// --- Adaptive label sprite (matches regen.py's makeLabelSprite shape) ----
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
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.shadowColor = color; ctx.shadowBlur = 28; ctx.fillStyle = color;
  ctx.fillText(text, canvasW / 2, canvasH / 2);
  ctx.shadowBlur = 0; ctx.fillStyle = '#ffffff'; ctx.globalAlpha = 0.95;
  ctx.fillText(text, canvasW / 2, canvasH / 2);
  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  const mat = new THREE.SpriteMaterial({map: tex, transparent: true, depthTest: false, fog: false});
  const sp = new THREE.Sprite(mat);
  const worldH = baseHeight * 0.25;
  const worldW = worldH * (canvasW / canvasH);
  sp.scale.set(worldW, worldH, 1);
  return sp;
}

// ---------------------------------------------------------------------------
// Pipeline filter sidebar — same UX as graph3d.html: phase → stage →
// param category → param. All sections start collapsed (`.section.collapsed`
// in the markup) so the user opens what they need.
// ---------------------------------------------------------------------------
const hiddenKinds = new Set();
const hiddenPhases = new Set();
const hiddenStages = new Set();
const hiddenNodes = new Set();    // individual params toggled from the tree

function isNodeVisible(n) {
  if (hiddenNodes.has(n.id)) return false;
  if (n.kind && !n.kind.startsWith('_') && hiddenKinds.has(n.kind)) return false;
  if (n.parent_phase && hiddenPhases.has(n.parent_phase)) return false;
  if (n.parent_stage && hiddenStages.has(n.parent_stage)) return false;
  // Phase / stage anchors are also subject to direct toggles.
  if (n.kind === '_phase' && hiddenPhases.has(n.id)) return false;
  if (n.kind === '_stage' && hiddenStages.has(n.id)) return false;
  return true;
}
function applyFilters() {
  Graph.nodeVisibility(isNodeVisible);
  Graph.linkVisibility(l => {
    const s = typeof l.source === 'object' ? l.source : DATA.nodes.find(n => n.id === l.source);
    const t = typeof l.target === 'object' ? l.target : DATA.nodes.find(n => n.id === l.target);
    if (!s || !t) return true;
    return isNodeVisible(s) && isNodeVisible(t);
  });
}

// --- Yellow-ring highlight (lifted verbatim from graph3d.html so the
// two views feel identical when the user clicks a name in the tree). --
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

// --- Pipeline tree (Phase → Stage → Input / Output → Param) ----------
// Params bucket by (stage, io_role) so each stage row expands into
// two sub-rows ("Input" + "Output"), each of which expands into its
// param list. Phases also expose Input + Output rows whose children
// are the data files declared in PHASE_DATA.
const paramsByStageAndIO = {};       // 'stage_id|input'  → [param, …]
const dataByPhaseAndIO   = {};       // 'phase_id|input'  → [phase_data, …]
const stageParamCount    = {};       // stage_id → total param count
const phaseDataCount     = {};       // phase_id → total data-file count
DATA.nodes.forEach(n => {
  if (!n.io_role) return;
  if (n.kind && !n.kind.startsWith('_') && n.parent_stage) {
    const key = n.parent_stage + '|' + n.io_role;
    (paramsByStageAndIO[key] = paramsByStageAndIO[key] || []).push(n);
    stageParamCount[n.parent_stage] = (stageParamCount[n.parent_stage] || 0) + 1;
  } else if (n.kind === '_phase_data' && n.parent_phase) {
    const key = n.parent_phase + '|' + n.io_role;
    (dataByPhaseAndIO[key] = dataByPhaseAndIO[key] || []).push(n);
    phaseDataCount[n.parent_phase] = (phaseDataCount[n.parent_phase] || 0) + 1;
  }
});

const filterTree = document.getElementById('filterTree');
const phaseRowEls = {};
const stageRowEls = {};

function makeParamRow(node) {
  const row = document.createElement('div');
  row.className = 'row param';
  row.dataset.nodeId = node.id;
  const col = KIND_COLOR[node.kind] || '#cfeaff';
  row.innerHTML = '<div class="dot" style="color:'+col+'"></div>'
                + '<span class="name">'+(node.label || node.id)+'</span>'
                + '<span class="count">'+(node.default || '')+'</span>';
  if (hiddenNodes.has(node.id)) row.classList.add('hidden');
  row.addEventListener('click', () => highlightNodeId(node.id));
  row.querySelector('.dot').addEventListener('click', e => {
    e.stopPropagation();
    if (hiddenNodes.has(node.id)) { hiddenNodes.delete(node.id); row.classList.remove('hidden'); }
    else { hiddenNodes.add(node.id); row.classList.add('hidden'); }
    applyFilters();
  });
  return row;
}

function makeIORow(io_id, role, count) {
  // Grouping row inside the tree — represents the Input or Output
  // sub-node for a phase / stage / orchestrator. Click name = highlight
  // the I/O sphere in 3D; click dot = toggle visibility of every param
  // (or data file) under it.
  const row = document.createElement('div');
  row.className = 'row cat';
  row.dataset.ioId = io_id;
  const col = role === 'input' ? '#06ffa5' : '#ff006e';
  const lbl = role === 'input' ? '▶ Input' : 'Output ▶';
  row.innerHTML = '<div class="dot" style="background:'+col+';color:'+col+'"></div>'
                + '<span class="name">'+lbl+'</span>'
                + '<span class="count">'+count+'</span>';
  row.addEventListener('click', () => highlightNodeId(io_id));
  row.querySelector('.dot').addEventListener('click', e => {
    e.stopPropagation();
    if (hiddenNodes.has(io_id)) {
      hiddenNodes.delete(io_id);
      row.classList.remove('hidden');
    } else {
      hiddenNodes.add(io_id);
      row.classList.add('hidden');
    }
    // Cascade to children (params or phase_data nodes) so the cluster
    // disappears as a unit.
    DATA.nodes.forEach(n => {
      if (n.parent_io !== io_id) return;
      if (hiddenNodes.has(io_id)) hiddenNodes.add(n.id);
      else hiddenNodes.delete(n.id);
    });
    applyFilters();
  });
  return row;
}

function appendIOBucket(parent_id, role, items) {
  // `items` is the list of param / phase_data nodes for this
  // (parent, role) bucket. Returns nothing — appends rows to the tree.
  if (!items || items.length === 0) return;
  filterTree.appendChild(makeIORow(parent_id + '__' + role, role, items.length));
  items.forEach(n => filterTree.appendChild(makeParamRow(n)));
}

DATA.phase_labels.forEach(p => {
  const phaseRow = document.createElement('div');
  phaseRow.className = 'row phase ' + (p.kind === 'external' ? 'external' : 'active');
  phaseRow.dataset.phaseId = p.id;
  // Count = data-file artefacts at the phase + every param under
  // every stage of the phase.
  let cnt = phaseDataCount[p.id] || 0;
  DATA.stage_labels.filter(s => s.parent === p.id).forEach(s => {
    cnt += stageParamCount[s.id] || 0;
  });
  phaseRow.innerHTML = '<div class="dot"></div><span class="name">'+p.long+'</span><span class="count">'+cnt+'</span>';
  phaseRow.addEventListener('click', () => highlightNodeId(p.id));
  phaseRow.querySelector('.dot').addEventListener('click', e => {
    e.stopPropagation();
    if (hiddenPhases.has(p.id)) { hiddenPhases.delete(p.id); phaseRow.classList.remove('hidden'); }
    else { hiddenPhases.add(p.id); phaseRow.classList.add('hidden'); }
    applyFilters();
  });
  filterTree.appendChild(phaseRow);
  phaseRowEls[p.id] = phaseRow;

  // Phase-level data flow: Input artefacts + Output artefacts.
  appendIOBucket(p.id, 'input',  dataByPhaseAndIO[p.id + '|input']  || []);
  appendIOBucket(p.id, 'output', dataByPhaseAndIO[p.id + '|output'] || []);

  DATA.stage_labels.filter(s => s.parent === p.id).forEach(s => {
    const stageRow = document.createElement('div');
    stageRow.className = 'row stage';
    stageRow.dataset.stageId = s.id;
    const scnt = stageParamCount[s.id] || 0;
    stageRow.innerHTML = '<div class="dot"></div><span class="name">'+s.long+'</span><span class="count">'+scnt+'</span>';
    stageRow.addEventListener('click', () => highlightNodeId(s.id));
    stageRow.querySelector('.dot').addEventListener('click', e => {
      e.stopPropagation();
      if (hiddenStages.has(s.id)) { hiddenStages.delete(s.id); stageRow.classList.remove('hidden'); }
      else { hiddenStages.add(s.id); stageRow.classList.add('hidden'); }
      applyFilters();
    });
    filterTree.appendChild(stageRow);
    stageRowEls[s.id] = stageRow;

    appendIOBucket(s.id, 'input',  paramsByStageAndIO[s.id + '|input']  || []);
    appendIOBucket(s.id, 'output', paramsByStageAndIO[s.id + '|output'] || []);
  });
});

// Orchestrator block — same idiom, with its own Input + Output buckets.
const orchPhaseRow = document.createElement('div');
orchPhaseRow.className = 'row phase active';
orchPhaseRow.dataset.phaseId = DATA.orchestrator_id;
const orchParamCount = DATA.nodes.filter(n => n.parent_stage === DATA.orchestrator_id && n.kind && !n.kind.startsWith('_')).length;
orchPhaseRow.innerHTML = '<div class="dot"></div><span class="name">Orchestrator (run_project.py)</span><span class="count">'+orchParamCount+'</span>';
orchPhaseRow.addEventListener('click', () => highlightNodeId(DATA.orchestrator_id));
orchPhaseRow.querySelector('.dot').addEventListener('click', e => {
  e.stopPropagation();
  if (hiddenStages.has(DATA.orchestrator_id)) { hiddenStages.delete(DATA.orchestrator_id); orchPhaseRow.classList.remove('hidden'); }
  else { hiddenStages.add(DATA.orchestrator_id); orchPhaseRow.classList.add('hidden'); }
  applyFilters();
});
filterTree.appendChild(orchPhaseRow);
appendIOBucket(DATA.orchestrator_id, 'input',
               paramsByStageAndIO[DATA.orchestrator_id + '|input']  || []);
appendIOBucket(DATA.orchestrator_id, 'output',
               paramsByStageAndIO[DATA.orchestrator_id + '|output'] || []);

// Show / hide all (Pipeline Filter actions)
document.getElementById('filterAll').addEventListener('click', () => {
  hiddenPhases.clear(); hiddenStages.clear(); hiddenNodes.clear(); hiddenKinds.clear();
  document.querySelectorAll('#filterTree .row').forEach(r => r.classList.remove('hidden'));
  document.querySelectorAll('#catItems .cat').forEach(r => r.classList.remove('hidden'));
  applyFilters();
});
document.getElementById('filterNone').addEventListener('click', () => {
  DATA.phase_labels.forEach(p => { hiddenPhases.add(p.id); phaseRowEls[p.id].classList.add('hidden'); });
  DATA.stage_labels.forEach(s => { hiddenStages.add(s.id); stageRowEls[s.id].classList.add('hidden'); });
  applyFilters();
});

// ---------------------------------------------------------------------------
// Param Categories legend (toggles category visibility globally).
// ---------------------------------------------------------------------------
const catItems = document.getElementById('catItems');
const catCount = {};
DATA.nodes.forEach(n => {
  if (!n.kind || n.kind.startsWith('_')) return;
  catCount[n.kind] = (catCount[n.kind] || 0) + 1;
});
Object.keys(KIND_LABEL).forEach(k => {
  const row = document.createElement('div');
  row.className = 'row cat';
  row.dataset.kind = k;
  const col = KIND_COLOR[k] || '#fff';
  row.innerHTML = '<div class="dot" style="background:'+col+';color:'+col+'"></div>'
                + '<span class="name">'+KIND_LABEL[k]+'</span>'
                + '<span class="count">'+(catCount[k] || 0)+'</span>';
  row.addEventListener('click', () => {
    if (hiddenKinds.has(k)) { hiddenKinds.delete(k); row.classList.remove('hidden'); }
    else { hiddenKinds.add(k); row.classList.add('hidden'); }
    document.querySelectorAll('#filterTree .row.cat[data-kind="'+k+'"]').forEach(r => r.classList.toggle('hidden', hiddenKinds.has(k)));
    applyFilters();
  });
  catItems.appendChild(row);
});
document.getElementById('catAll').addEventListener('click', () => {
  hiddenKinds.clear();
  document.querySelectorAll('#catItems .row.cat').forEach(r => r.classList.remove('hidden'));
  document.querySelectorAll('#filterTree .row.cat').forEach(r => r.classList.remove('hidden'));
  applyFilters();
});
document.getElementById('catNone').addEventListener('click', () => {
  Object.keys(KIND_LABEL).forEach(k => hiddenKinds.add(k));
  document.querySelectorAll('#catItems .row.cat').forEach(r => r.classList.add('hidden'));
  document.querySelectorAll('#filterTree .row.cat').forEach(r => r.classList.add('hidden'));
  applyFilters();
});

// Collapsible-section toggle (click h2 to fold / unfold).
document.querySelectorAll('#filters h2[data-target]').forEach(h => {
  h.addEventListener('click', () => {
    const sec = document.getElementById(h.dataset.target);
    if (sec) sec.classList.toggle('collapsed');
  });
});

// ---------------------------------------------------------------------------
// Solo / search — same idiom as graph3d.html. The ◉ icon next to a row
// solos that node + all its descendants in the scene; clicking again
// un-solos. Only one node can be soloed at a time. The search box in
// the Pipeline Filter section narrows the tree to rows whose name
// contains the query.
// ---------------------------------------------------------------------------
let soloId = null;
let soloVisibleSet = null;

// Build the descendant set of a node id. Walks the parent_*
// attributes the data layer carries so it works for phases, stages,
// I/O sub-nodes and the orchestrator uniformly.
function rebuildSoloSet(id) {
  if (id === null) { soloVisibleSet = null; return; }
  const visible = new Set([id]);
  // Multi-pass closure: a node belongs to the solo set if any of its
  // ancestor chain (parent / parent_io / parent_stage / parent_phase)
  // is already in the set. Cheap because we only have ~170 nodes.
  let added;
  do {
    added = false;
    DATA.nodes.forEach(n => {
      if (visible.has(n.id)) return;
      if ((n.parent && visible.has(n.parent)) ||
          (n.parent_io && visible.has(n.parent_io)) ||
          (n.parent_stage && visible.has(n.parent_stage)) ||
          (n.parent_phase && visible.has(n.parent_phase))) {
        visible.add(n.id);
        added = true;
      }
    });
  } while (added);
  soloVisibleSet = visible;
}

// Patch isNodeVisible so a non-empty solo set wins over everything else.
const _baseIsNodeVisible = isNodeVisible;
isNodeVisible = function(n) {
  if (soloId !== null && soloVisibleSet && !soloVisibleSet.has(n.id)) return false;
  return _baseIsNodeVisible(n);
};

// Attach a solo control (◉) to every row that represents a station-
// like node — phases, stages, the orchestrator, and the per-station
// Input / Output buckets. Clicking solos / un-solos. Only one row can
// be soloed at a time, and the soloed row glows yellow.
function attachSolo(row, nodeId) {
  const btn = document.createElement('span');
  btn.className = 'solo';
  btn.title = 'Show only this and what it contains';
  btn.textContent = '◉';
  row.appendChild(btn);
  btn.addEventListener('click', e => {
    e.stopPropagation();
    if (soloId === nodeId) { soloId = null; }
    else { soloId = nodeId; }
    rebuildSoloSet(soloId);
    document.querySelectorAll('#filterTree .row.soloed').forEach(r => r.classList.remove('soloed'));
    if (soloId !== null) row.classList.add('soloed');
    applyFilters();
  });
}
document.querySelectorAll('#filterTree .row.phase, #filterTree .row.stage, #filterTree .row.cat').forEach(r => {
  const id = r.dataset.phaseId || r.dataset.stageId || r.dataset.ioId;
  if (id) attachSolo(r, id);
});

// --- Search box (Pipeline Filter) ----------------------------------------
const searchInput = document.getElementById('filterSearch');
function applySearch() {
  const q = (searchInput.value || '').toLowerCase().trim();
  const rows = Array.from(document.querySelectorAll('#filterTree .row'));
  if (!q) {
    rows.forEach(r => { r.classList.remove('search-hidden'); r.classList.remove('search-hit'); });
    return;
  }
  // First pass: tag param / data rows as match / no-match. Track
  // ancestor rows that have at least one hit underneath so we can
  // hide unrelated phase / stage / I/O rows in the second pass.
  let lastPhase = null, lastStage = null, lastIO = null;
  const phaseHasHit = {}, stageHasHit = {}, ioHasHit = {};
  rows.forEach(r => {
    if (r.classList.contains('phase')) { lastPhase = r.dataset.phaseId; lastStage = null; lastIO = null; }
    else if (r.classList.contains('stage')) { lastStage = r.dataset.stageId; lastIO = null; }
    else if (r.classList.contains('cat')) { lastIO = r.dataset.ioId; }
    else if (r.classList.contains('param')) {
      const name = (r.querySelector('.name')?.textContent || '').toLowerCase();
      const hit = name.includes(q);
      r.classList.toggle('search-hit', hit);
      r.classList.toggle('search-hidden', !hit);
      if (hit) {
        if (lastPhase) phaseHasHit[lastPhase] = true;
        if (lastStage) stageHasHit[lastStage] = true;
        if (lastIO)    ioHasHit[lastIO] = true;
      }
    }
  });
  // Second pass: hide phase / stage / I/O rows that have no hit
  // underneath them.
  lastPhase = null; lastStage = null; lastIO = null;
  rows.forEach(r => {
    if (r.classList.contains('phase')) {
      lastPhase = r.dataset.phaseId; lastStage = null; lastIO = null;
      r.classList.toggle('search-hidden', !phaseHasHit[lastPhase]);
    } else if (r.classList.contains('stage')) {
      lastStage = r.dataset.stageId; lastIO = null;
      r.classList.toggle('search-hidden', !stageHasHit[lastStage]);
    } else if (r.classList.contains('cat')) {
      lastIO = r.dataset.ioId;
      r.classList.toggle('search-hidden', !ioHasHit[lastIO]);
    }
  });
}
searchInput.addEventListener('input', applySearch);
searchInput.addEventListener('focus', () => {
  document.getElementById('pipelineSection').classList.remove('collapsed');
});

// --- Per-node Three.js object ------------------------------------------
function nodeObject(n) {
  const grp = new THREE.Group();
  if (n.kind === '_phase') {
    // Same idiom as graph3d.html — full station treatment.
    const isExternal = n.phase_kind === 'external';
    const tint = isExternal ? 0xffb703 : 0x00f5ff;
    const tintHex = isExternal ? '#ffb703' : '#00f5ff';
    const r = 14;
    const core = new THREE.Mesh(new THREE.SphereGeometry(r, 28, 28), new THREE.MeshBasicMaterial({color: 0xffffff, fog: false})); grp.add(core);
    const halo = new THREE.Mesh(new THREE.SphereGeometry(r * 1.6, 24, 24), new THREE.MeshBasicMaterial({color: tint, transparent: true, opacity: 0.22, depthWrite: false, fog: false})); grp.add(halo);
    const ring1 = new THREE.Mesh(new THREE.RingGeometry(28, 32, 48), new THREE.MeshBasicMaterial({color: tint, side: THREE.DoubleSide, transparent: true, opacity: 0.65, depthWrite: false, fog: false})); ring1.rotation.y = Math.PI / 2; grp.add(ring1);
    const stationRing = new THREE.Mesh(new THREE.RingGeometry(95, 100, 64), new THREE.MeshBasicMaterial({color: tint, side: THREE.DoubleSide, transparent: true, opacity: 0.18, depthWrite: false, fog: false})); stationRing.rotation.y = Math.PI / 2; grp.add(stationRing);
    const lbl = makeLabelSprite(n.label.split('—')[0].trim(), tintHex, 240); lbl.position.set(0, 130, 0); grp.add(lbl);
    grp.userData.tick = (t) => {
      stationRing.scale.setScalar(1 + 0.08 * Math.sin(t * 2.5));
      core.scale.setScalar(1 + 0.04 * Math.sin(t * 3));
    };
  } else if (n.kind === '_stage') {
    const r = 5;
    const core = new THREE.Mesh(new THREE.SphereGeometry(r, 18, 18), new THREE.MeshBasicMaterial({color: 0xcfeaff, transparent: true, opacity: 0.85, fog: false})); grp.add(core);
    const halo = new THREE.Mesh(new THREE.SphereGeometry(r * 1.7, 14, 14), new THREE.MeshBasicMaterial({color: 0x00f5ff, transparent: true, opacity: 0.18, depthWrite: false, fog: false})); grp.add(halo);
    const ring = new THREE.Mesh(new THREE.RingGeometry(10, 12, 36), new THREE.MeshBasicMaterial({color: 0x00f5ff, side: THREE.DoubleSide, transparent: true, opacity: 0.32, depthWrite: false, fog: false})); ring.rotation.y = Math.PI / 2; grp.add(ring);
    const shortLabel = (n.label || '').replace(/^Stage\s+/, '').split('—')[0].trim();
    const lbl = makeLabelSprite(shortLabel, '#cfeaff', 130); lbl.position.set(0, 30, 0); lbl.material.opacity = 0.82; grp.add(lbl);
  } else if (n.kind === '_orchestrator') {
    // Pipeline orchestrator — yellow station to read as "the conductor".
    const tint = 0xffd60a;
    const r = 11;
    const core = new THREE.Mesh(new THREE.SphereGeometry(r, 24, 24), new THREE.MeshBasicMaterial({color: 0xffffff, fog: false})); grp.add(core);
    const halo = new THREE.Mesh(new THREE.SphereGeometry(r * 1.7, 18, 18), new THREE.MeshBasicMaterial({color: tint, transparent: true, opacity: 0.30, depthWrite: false, fog: false})); grp.add(halo);
    const ring = new THREE.Mesh(new THREE.RingGeometry(22, 25, 40), new THREE.MeshBasicMaterial({color: tint, side: THREE.DoubleSide, transparent: true, opacity: 0.55, depthWrite: false, fog: false})); ring.rotation.y = Math.PI / 2; grp.add(ring);
    const lbl = makeLabelSprite('run_project.py', '#ffd60a', 200); lbl.position.set(0, 38, 0); grp.add(lbl);
  } else if (n.kind === '_io') {
    // Input / Output sub-node: green for inputs, pink for outputs.
    // Slightly bigger than a param so it reads as the "anchor" of
    // its galaxy. No long label sprite — just a tiny tag so the
    // viewer can scan IN/OUT at a glance.
    const isIn = n.io_role === 'input';
    const col = isIn ? '#06ffa5' : '#ff006e';
    const r = 4.5;
    const core = new THREE.Mesh(new THREE.SphereGeometry(r, 18, 18), new THREE.MeshBasicMaterial({color: new THREE.Color(col), fog: false})); grp.add(core);
    const halo = new THREE.Mesh(new THREE.SphereGeometry(r * 1.8, 14, 14), new THREE.MeshBasicMaterial({color: new THREE.Color(col), transparent: true, opacity: 0.40, depthWrite: false, fog: false})); grp.add(halo);
    const ring = new THREE.Mesh(new THREE.RingGeometry(8, 9.5, 32), new THREE.MeshBasicMaterial({color: new THREE.Color(col), side: THREE.DoubleSide, transparent: true, opacity: 0.55, depthWrite: false, fog: false})); ring.rotation.y = Math.PI / 2; grp.add(ring);
    const lbl = makeLabelSprite(isIn ? '▶ INPUT' : 'OUTPUT ▶', col, 95); lbl.position.set(0, r + 6, 0); lbl.material.opacity = 0.95; grp.add(lbl);
  } else if (n.kind === '_phase_data') {
    // Data-file artefact orbiting a phase Input / Output sub-node.
    // Uses the same green / pink as the I/O parent so the cluster
    // reads as one group; small box icon (file-shape).
    const isIn = n.io_role === 'input';
    const col = isIn ? '#06ffa5' : '#ff006e';
    const box = new THREE.Mesh(new THREE.BoxGeometry(5.5, 7, 1.4), new THREE.MeshBasicMaterial({color: new THREE.Color(col), transparent: true, opacity: 0.85, fog: true})); grp.add(box);
    const halo = new THREE.Mesh(new THREE.SphereGeometry(5.2, 10, 10), new THREE.MeshBasicMaterial({color: new THREE.Color(col), transparent: true, opacity: 0.22, depthWrite: false, fog: true})); grp.add(halo);
    const lbl = makeLabelSprite(n.label, col, 80); lbl.position.set(0, 9, 0); lbl.material.opacity = 0.88; grp.add(lbl);
  } else {
    // Parameter node — small sphere coloured by category, label "--flag = default".
    const col = KIND_COLOR[n.kind] || '#cfeaff';
    const r = 2.5;
    const core = new THREE.Mesh(new THREE.SphereGeometry(r, 14, 14), new THREE.MeshBasicMaterial({color: new THREE.Color(col), fog: true})); grp.add(core);
    const halo = new THREE.Mesh(new THREE.SphereGeometry(r * 1.9, 10, 10), new THREE.MeshBasicMaterial({color: new THREE.Color(col), transparent: true, opacity: 0.30, depthWrite: false, fog: true})); grp.add(halo);
    // Flag label above the sphere; default value below — same sprite
    // shape, but the default reads quieter: smaller text height and a
    // dimmed grey/lower-opacity treatment so the colourful flag is
    // what the eye locks onto first.
    const flagLbl = makeLabelSprite(n.label, col, 90);
    flagLbl.position.set(0, r + 8, 0);
    flagLbl.material.opacity = 0.95;
    grp.add(flagLbl);
    if (n.default) {
      const defLbl = makeLabelSprite(n.default, '#5a5a5a', 55);
      defLbl.position.set(0, -(r + 5), 0);
      defLbl.material.opacity = 0.55;
      grp.add(defLbl);
    }
  }
  return grp;
}

const elem = document.getElementById('graph');
const Graph = ForceGraph3D()(elem)
  .graphData({nodes: DATA.nodes, links: DATA.links})
  .backgroundColor('#000008')
  .nodeLabel(n => {
    const col = KIND_COLOR[n.kind] || '#6affff';
    if (n.kind && n.kind.startsWith('_')) {
      return '<div style="background:rgba(0,20,40,0.95);padding:6px 10px;border:1px solid '+col+';border-radius:6px;font-size:12px;color:#fff">'+n.label+'</div>';
    }
    return '<div style="background:rgba(0,20,40,0.95);padding:6px 10px;border:1px solid '+col+';border-radius:6px;font-size:12px;color:#fff;box-shadow:0 0 16px '+col+'"><b>'+n.label+'</b><br><span style="opacity:.7;font-size:10px">default: '+(n.default||'(none)')+'</span></div>';
  })
  .nodeThreeObject(nodeObject).nodeThreeObjectExtend(false)
  // Two link kinds:
  //   precedes — phase → phase, cyan rail (bright, thick, particles)
  //   origin   — every other node back to its parent station, dim
  //              line tinted by I/O role (green = input, pink =
  //              output, soft cyan otherwise). No particles so the
  //              rail stays the visual focus.
  .linkColor(l => {
    if (l.kind === 'precedes') return '#00f5ff';
    if (l.io_role === 'input')  return '#06ffa5';
    if (l.io_role === 'output') return '#ff006e';
    return '#7fe0ff';
  })
  .linkOpacity(l => l.kind === 'precedes' ? 0.92 : 0.32)
  .linkWidth(l => l.kind === 'precedes' ? 6.5 : 0.6)
  .linkDirectionalParticles(l => l.kind === 'precedes' ? 8 : 0)
  .linkDirectionalParticleSpeed(_ => 0.014)
  .linkDirectionalParticleWidth(_ => 6)
  .linkDirectionalParticleColor(l => l.kind === 'precedes' ? '#00f5ff' : '#7fe0ff')
  .showNavInfo(false)
  .onNodeClick(n => {
    const dist = 80, r = Math.hypot(n.x, n.y, n.z) || 1;
    Graph.cameraPosition({x: n.x*(1+dist/r), y: n.y*(1+dist/r), z: n.z*(1+dist/r)}, n, 1400);
    const info = document.getElementById('info');
    info.style.display = 'block';
    const col = KIND_COLOR[n.kind] || '#6affff';
    info.style.borderColor = col; info.style.boxShadow = '0 0 32px '+col;
    document.getElementById('nodeLabel').textContent = n.label;
    document.getElementById('nodeLabel').style.color = col;
    let html = '';
    if (n.kind && !n.kind.startsWith('_')) {
      const kindLbl = KIND_LABEL[n.kind] || n.kind;
      html += '<div class="row"><span class="key">Category</span> '+kindLbl+'</div>';
      html += '<div class="row"><span class="key">Default</span> <code>'+(n.default||'(none)')+'</code></div>';
      if (n.choices) html += '<div class="row"><span class="key">Choices</span> '+n.choices.map(c => '<code>'+c+'</code>').join(' · ')+'</div>';
      if (n.help)    html += '<div class="row"><span class="key">Help</span> '+n.help+'</div>';
      const parent = DATA.nodes.find(x => x.id === n.parent_stage);
      if (parent) html += '<div class="row"><span class="key">Stage</span> '+parent.label+'</div>';
    } else if (n.kind === '_stage') {
      html += '<div class="row">Stage anchor — child params orbit around it.</div>';
    } else if (n.kind === '_phase') {
      html += '<div class="row">Phase station — see graph3d.html for code-level structure.</div>';
    } else if (n.kind === '_orchestrator') {
      html += '<div class="row">Top-level orchestrator. Every flag here is passed through to the matching stage.</div>';
    }
    document.getElementById('nodeMeta').innerHTML = html;
  })
  .onBackgroundClick(() => { document.getElementById('info').style.display = 'none'; });

// Every node is pinned (phases, stages, orchestrator, params), so the
// force solver has nothing to relax — the layout is deterministic and
// matches exactly what `regen_params.py` placed. We still keep the
// charge force at default-ish values so the link line lengths render
// at the geometric distance between the two pinned endpoints (no
// elastic stretching).
Graph.d3Force('charge').strength(0);
Graph.d3Force('link').distance(40);

const scene = Graph.scene();
scene.fog = new THREE.FogExp2(0x000a18, 0.00060);

// Phase rail ticks + sub-pipeline cylinders — same as graph3d.html.
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
// (No phase → stage sub-rail cylinders here — stages live inside their
// phase's galaxy via spatial proximity only, no lines.)

// Volumetric haze — same dust cloud as graph3d.html so the two views
// feel like the same space at different zoom levels.
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

// Camera fitting (same logic as graph3d.html so the two views start
// with comparable framing).
function fitZForAllPhases() {
  const xs = DATA.phase_labels.map(p => p.x);
  const ys = DATA.phase_labels.map(p => p.y);
  const width  = (Math.max(...xs) - Math.min(...xs));
  const height = (Math.max(...ys) - Math.min(...ys));
  const camera = Graph.camera();
  const vFov = ((camera && camera.fov) ? camera.fov : 50) * Math.PI / 180;
  const aspect = window.innerWidth / Math.max(1, window.innerHeight);
  const hFov = 2 * Math.atan(Math.tan(vFov / 2) * aspect);
  const zW = (width  / 2) / Math.tan(hFov / 2);
  const zH = (height / 2) / Math.tan(vFov / 2);
  const z  = Math.max(zW, zH) * 3.0 + 2500;
  return Math.max(z, 6500);
}
let fitZ = 6500;
function resetCamera() {
  fitZ = fitZForAllPhases();
  Graph.cameraPosition({x: 0, y: 380, z: fitZ}, {x:0,y:0,z:0}, 1500);
}
setTimeout(resetCamera, 120);

let theta = 0, autoRotate = false;
setInterval(() => {
  if (!autoRotate) return;
  theta += 0.0014;
  Graph.cameraPosition({
    x: (fitZ * 0.10) * Math.sin(theta * 0.45),
    y: 380 * Math.cos(theta) - 80 * Math.sin(theta),
    z: fitZ * (1 + 0.08 * Math.sin(theta * 0.6))
  });
}, 25);
elem.addEventListener('mousedown', () => autoRotate = false);

// Wheel zoom-to-cursor (same ergonomics as graph3d.html).
Graph.controls().enableZoom = false;
elem.addEventListener('wheel', (ev) => {
  ev.preventDefault();
  autoRotate = false;
  const ctrls = Graph.controls();
  const cam = Graph.camera();
  const rect = elem.getBoundingClientRect();
  const mx = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  const my = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  const ndc = new THREE.Vector3(mx, my, 0.5).unproject(cam);
  const dir = ndc.sub(cam.position).normalize();
  const distToTarget = cam.position.distanceTo(ctrls.target);
  const cursorWorld = cam.position.clone().add(dir.multiplyScalar(distToTarget));
  const step = Math.min(0.5, Math.abs(ev.deltaY) * 0.0015);
  const factor = ev.deltaY < 0 ? (1 - step) : (1 + step);
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
  fitZ = fitZForAllPhases();
  if (autoRotate) return;
  Graph.cameraPosition({x: 0, y: 380, z: fitZ}, undefined, 600);
});
</script>
</body>
</html>
"""

HTML = HTML.replace('__DATA__', data_str).replace('__RAIL_MIN__', str(rail_min)).replace('__RAIL_MAX__', str(rail_max))
out_path = OUT / 'graphParams3D.html'
out_path.write_text(HTML)
print(f'Wrote {out_path} ({out_path.stat().st_size} bytes — {len(payload["nodes"])} nodes, {len(payload["links"])} links)')
print('Open graphify-out/graphParams3D.html in a browser.')
