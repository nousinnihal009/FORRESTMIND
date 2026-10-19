"""Run the local ForestMind workspace: python scripts/serve.py."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description="Serve the ForestMind local monitoring workspace")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: localhost; no authentication)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Reload on source changes during development")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("Hosted access is disabled: use a loopback host. Multi-user authentication and isolation require a separate deployment gate.")
    try:
        import uvicorn
    except ImportError:
        raise SystemExit('Install web dependencies: python -m pip install -e ".[web]"') from None
    request_log = logging.getLogger("forestmind.http")
    if not request_log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        request_log.addHandler(handler)
    request_log.setLevel(logging.INFO)
    request_log.propagate = False
    uvicorn.run("forestmind.api.app:create_app", factory=True, host=args.host, port=args.port,
                reload=args.reload, app_dir=str(Path(__file__).resolve().parents[1]))


if __name__ == "__main__":
    main()
