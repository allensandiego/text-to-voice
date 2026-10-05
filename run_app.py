#!/usr/bin/env python3
"""Convenience launcher for the Text to Voice Gradio Web UI.

Usage:
    python run_app.py [--port 7860] [--host 0.0.0.0] [--share]
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app import create_app


def main():
    parser = argparse.ArgumentParser(
        description="Launch Text to Voice Studio Gradio Web Interface."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host interface to bind server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port number (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Generate a public gradio.live share link",
    )

    args = parser.parse_args()

    print("=" * 65)
    print("🎙️  Text to Voice Studio (Chatterbox-Turbo)")
    print("=" * 65)
    print(f"Local Access:   http://localhost:{args.port}")
    print(f"Network Access: http://{args.host}:{args.port}")
    print("=" * 65)

    demo = create_app()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
