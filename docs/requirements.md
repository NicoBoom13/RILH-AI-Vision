# Python dependencies
#
# Replaces the old top-level `requirements.txt`. Install with:
#
#     pip install -r docs/requirements.md
#
# This file is dual-purpose: pip ignores blank lines and lines starting
# with `#`, so the `#`-prefixed prose below doubles as Markdown headings
# / comments. Keep every non-comment line as a single PEP-508 requirement
# spec — no fenced code blocks, no indented prose.

# ## Core — detection, tracking, video I/O
#
# - ultralytics: YOLO11 / YOLO26 / HockeyAI inference + ByteTrack / BoT-SORT trackers
# - supervision: annotation primitives used by Stage 1.e (boxes, labels, traces)
# - opencv-python: frame I/O for every stage
# - numpy: universal
# - lap: Hungarian assignment used internally by the trackers

ultralytics>=8.3.0
supervision>=0.25.0
opencv-python>=4.10.0
numpy>=1.26.0
lap>=0.5.12

# ## Stage 1.c — Numbers (PARSeq OCR)
#
# PARSeq is loaded via `torch.hub` (no PyPI package); these three are
# its runtime deps. TrOCR was removed during the refactor — there is no
# `transformers` dependency anymore.

pytorch_lightning>=2.0
timm>=0.9
nltk>=3.8

# ## Stage 1.d — Entities (OSNet Re-ID)
#
# `gdown` and `tensorboard` are transitive deps that `torchreid`
# imports at import time but doesn't declare in its own metadata, so
# they have to be pinned explicitly here.

torchreid>=0.2.5
gdown>=5.0
tensorboard>=2.0

# ## Notes
#
# - `torch` is not pinned here: it comes in transitively via
#   `ultralytics` and `pytorch_lightning`. For a specific CUDA build,
#   install torch first from <https://pytorch.org/get-started/locally/>
#   before running `pip install -r docs/requirements.md`.
# - Model weights (YOLO, HockeyAI, PARSeq, OSNet) auto-download into
#   `models/` on first use — they are not Python packages.
