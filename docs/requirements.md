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

# ## Stage 3.a — Entities (OSNet Re-ID, formerly Stage 1.d)
#
# Also reused by the Stage 1.b `osnet` team engine. `gdown` and
# `tensorboard` are transitive deps that `torchreid` imports at
# import time but doesn't declare in its own metadata, so they have
# to be pinned explicitly here.

torchreid>=0.2.5
gdown>=5.0
tensorboard>=2.0

# ## Stage 1.b — Optional team engines (siglip / contrastive)
#
# `transformers` powers the SigLIP vision encoder used by the
# Roboflow-style siglip engine. `umap-learn` reduces SigLIP's pooled
# features to 3-D before k-means (better-conditioned than raw
# 768-D for cluster separation). `scikit-learn` is the PCA fallback
# when umap-learn isn't installed and supplies KMeans for the
# contrastive engine. None of these are loaded by the default hsv
# engine — install them only if you want to ablate.

transformers>=4.40.0
umap-learn>=0.5.5
scikit-learn>=1.3.0

# ## Notes
#
# - `torch` is not pinned here: it comes in transitively via
#   `ultralytics` and `pytorch_lightning`. For a specific CUDA build,
#   install torch first from <https://pytorch.org/get-started/locally/>
#   before running `pip install -r docs/requirements.md`.
# - Model weights (YOLO, HockeyAI, PARSeq, OSNet) auto-download into
#   `models/` on first use — they are not Python packages.
