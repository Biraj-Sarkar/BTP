"""
Launch script for the Anemia Detection Single-Image Inference Interface.
Usage:
    python run_interface.py [--host 127.0.0.1] [--port 8000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure root directory is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Anemia Detection Single-Image Web Interface")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    args = parser.parse_args()

    print("==================================================================")
    print("ANEMIA DETECTION BTP — SINGLE-IMAGE INFERENCE INTERFACE")
    print("==================================================================")
    print(f"Starting local server at: http://{args.host}:{args.port}")
    print("Academic Prototype Notice: This is an academic research prototype,")
    print("not a clinically validated diagnostic device.")
    print("Press Ctrl+C to terminate.")
    print("==================================================================")

    uvicorn.run("backend.app:app", host=args.host, port=args.port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
