#!/usr/bin/env python3
"""
Launcher script for Person Down Detection Web Application (Phase 7).
Starts the FastAPI server with Uvicorn.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch Person Down Detection Application Server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind the server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print(" PERSON DOWN DETECTION SYSTEM - WEB APPLICATION (PHASE 7)")
    print("=" * 70)
    print(f" Binding URL    : http://{args.host}:{args.port}")
    print(f" Local Web UI   : http://localhost:{args.port}")
    print(f" OpenAPI Docs   : http://localhost:{args.port}/docs")
    print(f" Project Root   : {PROJECT_ROOT}")
    print(" Press Ctrl+C to terminate the application.")
    print("=" * 70)

    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
