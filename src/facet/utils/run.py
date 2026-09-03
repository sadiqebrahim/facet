"""Run manifests.

Every experiment writes one of these. It is the mechanism behind the reproducibility
requirement in docs/RESEARCH.md section 15.7: a metric without its full provenance is
not a result.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .hashing import hash_obj


def _gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 else "none"
    except Exception:
        return "unknown"


@dataclass
class RunManifest:
    """Everything needed to reproduce and interpret one experiment."""

    experiment: str
    description: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    seed: int = 1337
    dataset: str = ""
    dataset_version: str = ""
    split_methodology: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    # filled automatically
    started_at: str = ""
    duration_sec: float = 0.0
    hardware: dict[str, str] = field(default_factory=dict)
    software: dict[str, str] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        self._t0 = time.time()
        self.started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.hardware = {
            "gpu": _gpu_name(),
            "cpu": platform.processor() or platform.machine(),
            "platform": platform.platform(),
        }
        self.software = {"python": sys.version.split()[0]}
        for mod in ("numpy", "sklearn", "onnxruntime", "torch"):
            try:
                self.software[mod] = __import__(mod).__version__
            except Exception:
                pass

    def finish(self, out_dir: str | Path) -> Path:
        """Stamp timing + config hash and write manifest.json."""
        self.duration_sec = round(time.time() - self._t0, 2)
        self.config_hash = hash_obj(self.config)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "manifest.json"
        payload = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path
