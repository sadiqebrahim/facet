"""Content hashing - used for config provenance and for incremental indexing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def hash_obj(obj: Any) -> str:
    """Stable short hash of a JSON-serialisable object (e.g. a config dict)."""
    payload = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def hash_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Content hash of a file. Used to detect changed images during re-indexing."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()[:16]
