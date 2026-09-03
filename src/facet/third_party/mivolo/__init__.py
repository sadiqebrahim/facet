"""Vendored MiVOLO model definition.

Source: https://github.com/WildChlamydia/MiVOLO  (Apache License 2.0)
Copyright the MiVOLO authors. Cite:
    Kuprashevich & Tolstykh, MiVOLO: Multi-input Transformer for Age and Gender
    Estimation, arXiv:2307.04616 (2023); arXiv:2403.02302 (2024).

ONLY the model definition is vendored (`mivolo_model.py`, `cross_bottleneck_attn.py`).
`yolo_detector.py` is deliberately NOT included: it depends on Ultralytics YOLOv8, which is
**AGPL-3.0** and would be network-viral for a deployed service. We already have SCRFD, so we
use MiVOLO's age/gender head with our own detector - the plan recorded in docs/LICENSING.md
section 1. Nothing here imports ultralytics; verified by test.

Importing this module registers `mivolo_d1_384` with timm's model registry.
"""
from . import mivolo_model  # noqa: F401
