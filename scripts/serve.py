#!/usr/bin/env python
"""Serve the Facet API and UI.

    python scripts/serve.py --index facet.db --features feats/
    open http://127.0.0.1:8000

Binds to localhost by default. Everything stays on this machine: images are read from local
disk and served to a local browser, and nothing in the API makes an outbound request
(docs/LICENSING.md section 4). Face embeddings are biometric data, so the default is not to
expose them on a network interface.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(ROOT / "artifacts/index/facet.db"))
    ap.add_argument("--features", default=str(ROOT / "artifacts/index/features"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    if args.host not in ("127.0.0.1", "localhost"):
        print(f"WARNING: binding to {args.host} exposes face embeddings (biometric data) "
              f"beyond this machine. See docs/LICENSING.md section 4.", file=sys.stderr)

    import uvicorn
    from facet.api.app import create_app

    app = create_app(args.index, args.features)
    print(f"index    : {args.index}\nfeatures : {args.features}\n"
          f"UI       : http://{args.host}:{args.port}\n"
          f"API docs : http://{args.host}:{args.port}/docs\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
