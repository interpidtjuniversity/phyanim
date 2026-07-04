#!/usr/bin/env python
"""PhyAnim server launcher.

Start the Flask server with configurable LLM provider, TTS, and media directory.

Usage::

    # Basic
    python server.py --media_dir /home/phyanim/media

    # With specific LLM provider and TTS
    python server.py \\
        --media_dir /home/phyanim/media \\
        --llm_provider doubao \\
        --llm_api_key sk-... \\
        --tts_provider minimax \\
        --tts_api_key sk-api-... \\
        --tts_voice male-qn-qingse

    # Or use environment variables
    PHYANIM_LLM_PROVIDER=doubao PHYANIM_LLM_API_KEY=sk-... python server.py --media_dir /home/phyanim/media
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure phyanim is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from phyanim.server.config import ServerConfig, LLMProvider
from phyanim.server.app import create_app
from phyanim.voiceover.tts_config import TTSConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PhyAnim animation generation server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Server
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Listen port (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")

    # Media directory — ALL generated files go here
    parser.add_argument(
        "--media_dir",
        default="",
        help="Root directory for ALL generated files (scripts, videos, tex, audio, images). "
             "Defaults to <cwd>/media",
    )

    # LLM
    parser.add_argument(
        "--llm_provider",
        choices=["deepseek", "doubao"],
        default=None,
        help="LLM backend (default: from env or 'deepseek')",
    )
    parser.add_argument("--llm_api_key", default="", help="LLM API key")
    parser.add_argument("--llm_model", default="", help="Override LLM model name")
    parser.add_argument("--llm_base_url", default="", help="Override LLM base URL")

    # TTS
    parser.add_argument(
        "--tts_provider",
        choices=["minimax", "none"],
        default=None,
        help="TTS provider (default: from env or 'none')",
    )
    parser.add_argument("--tts_api_key", default="", help="MiniMax TTS API key")
    parser.add_argument("--tts_voice", default="male-qn-qingse", help="TTS voice ID")

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ServerConfig:
    """Build ServerConfig from CLI args, falling back to env vars."""
    # Start from env (handles all PHYANIM_* vars).
    config = ServerConfig.from_env()

    # Override with CLI args where provided.
    if args.media_dir:
        config.media_dir = args.media_dir
    if args.llm_provider:
        config.llm_provider = LLMProvider.DEEPSEEK if args.llm_provider == "deepseek" else LLMProvider.DOUBAO
    if args.llm_api_key:
        config.llm_api_key = args.llm_api_key
    if args.llm_model:
        config.llm_model = args.llm_model
    if args.llm_base_url:
        config.llm_base_url = args.llm_base_url
    if args.tts_provider:
        config.tts = TTSConfig(
            provider=args.tts_provider,
            api_key=args.tts_api_key or config.tts.api_key,
            voice_id=args.tts_voice,
        )

    return config


def main() -> None:
    logging.basicConfig(
        level=logging.DEBUG if "--debug" in sys.argv else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()
    config = build_config(args)

    # Print effective configuration.
    media_dir = config.resolved_media_dir()
    print("=" * 60)
    print(f"  PhyAnim Server")
    print(f"  LLM Provider : {config.llm_provider.value}")
    print(f"  LLM Model    : {config.resolved_model()}")
    print(f"  TTS Enabled  : {config.tts.enabled}")
    print(f"  Media Dir    : {media_dir}")
    print(f"  Code Dir     : {config.resolved_code_dir()}")
    print(f"  Manim Media  : {config.resolved_manim_media_dir()}")
    print(f"  Listen       : {args.host}:{args.port}")
    print("=" * 60)

    app = create_app(config)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
