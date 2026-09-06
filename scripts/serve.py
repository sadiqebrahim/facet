#!/usr/bin/env python
"""Serve the Facet API and UI.

    python scripts/serve.py --index facet.db --features feats/
    open http://127.0.0.1:8000

Binds to localhost by default. Nothing in the API makes an outbound request - images are read
from local disk and served to the browser (docs/LICENSING.md section 4).

To reach it from a phone or another machine over a VPN:

    python scripts/serve.py --host 0.0.0.0 --index facet.db --features feats/

That prints every address the box is reachable on, so a VPN interface is easy to spot. Create
an account first: with no accounts the app runs open, and on 0.0.0.0 that means anyone who can
reach the port has full access to biometric data. There is no TLS, so use a VPN or an HTTPS
reverse proxy rather than exposing it to the internet.
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
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 to reach it from other devices (VPN/LAN). Read the "
                         "warning it prints before doing that on an untrusted network.")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    import uvicorn
    from facet.api.app import create_app
    from facet.api.auth import Accounts
    from facet.pipeline.db import Index

    external = args.host not in ("127.0.0.1", "localhost")
    if external:
        n_users = 0
        try:
            ix = Index(args.index)
            n_users = Accounts(ix.conn).count()
            ix.close()
        except Exception:
            pass
        print("─" * 72, file=sys.stderr)
        print(f"  Binding to {args.host} — reachable from other machines on this network.",
              file=sys.stderr)
        print("  Face embeddings are biometric data (docs/LICENSING.md §4).", file=sys.stderr)
        if n_users == 0:
            print("  ⚠ NO ACCOUNTS EXIST, so the app is in open mode: anyone who can reach",
                  file=sys.stderr)
            print("    this port gets full access. Create an account in the UI to require",
                  file=sys.stderr)
            print("    a login.", file=sys.stderr)
        else:
            print(f"  {n_users} account(s) — a login is required.", file=sys.stderr)
        print("  There is no TLS here: on an untrusted network, passwords and session",
              file=sys.stderr)
        print("  tokens cross the wire in the clear. Use a VPN or an HTTPS proxy.",
              file=sys.stderr)
        print("─" * 72, file=sys.stderr)

    app = create_app(args.index, args.features)
    print(f"index    : {args.index}\nfeatures : {args.features}")
    print(f"UI       : http://{'localhost' if not external else _lan_ip()}:{args.port}")
    if external:
        for name, ip in _all_addresses():
            print(f"           http://{ip}:{args.port}   ({name})")
    print(f"API docs : /docs\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def _lan_ip() -> str:
    """Best-guess routable address, so the printed URL is one you can actually open."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))          # no packets are sent
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "0.0.0.0"


def _all_addresses() -> list[tuple[str, str]]:
    """Every IPv4 address on the box, so a VPN interface is easy to spot."""
    out = []
    try:
        import subprocess
        r = subprocess.run(["ip", "-4", "-o", "addr"], capture_output=True, text=True,
                           timeout=5)
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) > 3 and parts[2] == "inet":
                name, ip = parts[1], parts[3].split("/")[0]
                if not ip.startswith("127."):
                    out.append((name, ip))
    except Exception:
        pass
    return out


if __name__ == "__main__":
    raise SystemExit(main())
