# Regenerating graph3d.html

This directory contains a 3D pipeline visualization (`graph3d.html`) and the data + script needed to rebuild it.

## Files

- `graph3d.html` — the visualization (open in any browser, no server)
- `graph.json` — merged knowledge graph (graphify AST + orchestration layer)
- `orchestration.json` — pipeline orchestration nodes/edges (phases, data flow, design rationale, blockers). Hand-curated from README.md / CLAUDE.md / docs/phase_1_6_design.md / configs.
- `regen.py` — single-command rebuild script
- `cache/` — graphify's content-hashed AST cache (incremental rebuilds)

## How to rebuild

```bash
/Users/nico/.local/share/uv/tools/graphifyy/bin/python graphify-out/regen.py
```

What it does (~2 seconds, no LLM tokens):

1. Re-runs graphify AST extraction on `src/` (deterministic, free, content-hashed cache)
2. Loads `orchestration.json`
3. Merges them, clusters, writes `graph.json`
4. Generates `graph3d.html` with the data inlined

## When to update what

| You changed... | What to do |
|---|---|
| `src/*.py` (any code) | `python regen.py` — AST re-extracts changed files via cache, rest is instant |
| README.md / CLAUDE.md / docs | Edit `orchestration.json` by hand (add nodes/edges for new phases, new data files, new rationales), then `python regen.py` |
| Want different visual style | Edit the HTML template inside `regen.py` (the `HTML = r"""..."""` block) |
| Community labels look wrong | Edit `COMMUNITY_LABELS` dict at the top of `regen.py` (clustering can renumber communities between runs) |

## orchestration.json schema

Same as graphify's extraction output. Keys per node: `id`, `label`, `file_type` (`document`/`code`), `source_file`, `source_location` (nullable). Keys per edge: `source`, `target`, `relation`, `confidence` (`EXTRACTED`/`INFERRED`), `confidence_score`, `weight`.

Relation types in use (each gets a distinct color in the viz):

- `precedes` — phase ordering (cyan)
- `feeds_into` / `consumes_output_of` — data flow between scripts (green / pink)
- `implements` — Phase concept → Python script (amber)
- `configures` — yaml → script (blue)
- `rationale_for` — design decision → concept it explains (purple)
- `references` — generic link (white)

**Critical:** node IDs must match graphify's deterministic AST IDs when linking to existing code. Pattern: `{stem}_{entity}` lowercase, `[a-z0-9_]` only. Examples: `phase1_detect_track_py`, `phase1_6_entities_unionfind`. Any edge whose source or target doesn't exist in the AST + orchestration union is silently dropped during merge.

## Phase pinning

The 10 Phase concept nodes are pinned along the X-axis in `regen.py` (`PHASE_ORDER` list). To add a new phase: append to that list with a slot integer, and add the corresponding node + edges to `orchestration.json`.

## Visual design (validated)

- Pinned-phase rail along X-axis with cyan tube backbone
- Each phase = white core + cyan halo + 4 rings (1 perpendicular station ring + 3 orbital rings green/magenta/amber rotating on different axes) + orbiting white satellite + floating sprite label
- Custom `xPull` d3 force pulls non-phase nodes toward their connected phase's X column (neural-network-layer effect)
- Volumetric haze: `THREE.FogExp2` density 0.00060, 1500-particle dust field with HSL cyan/teal variation, additive blending
- CSS radial vignette + scanlines, UnrealBloom 1.45/0.6/0.20
- Auto-orbit camera around X-axis (stays in YZ plane)

Press R to reset camera. Click any node to focus.
