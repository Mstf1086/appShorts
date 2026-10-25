#!/usr/bin/env python3
"""
YouTube Shorts Generator - CLI Entry Point

Usage:
    python generate_shorts.py "Your topic here"

Example:
    python generate_shorts.py "Yapay zeka nedir?"
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pipeline.shorts_pipeline import ShortsPipeline
import config
from services.audio_factory import get_voice_generator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="YouTube Shorts Generator - Create vertical short videos automatically",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_shorts.py "Yapay zeka nedir?"
        """
    )
    parser.add_argument(
        "topic",
        nargs="?",
        help="Topic/title for the YouTube Shorts video"
    )
    parser.add_argument(
        "--no-approve",
        action="store_true",
        help="Skip interactive image approval (auto-approve the first candidate)"
    )
    parser.add_argument(
        "--cf-login",
        action="store_true",
        help="Log in to Cloudflare with the cf CLI (OAuth, no API token)"
    )
    parser.add_argument(
        "--tts",
        type=str,
        default=None,
        help="Voice TTS engine to use (edge / cloudflare / ttsai / pockettts); prompts on startup if omitted"
    )
    return parser.parse_args()


def _choose_tts_engine(service: str = None) -> str:
    """Prompt the user (TTY only) to choose the TTS engine at startup."""
    provider = service or config.TTS_PROVIDER
    if provider:
        return provider
    if not sys.stdin.isatty():
        # Non-interactive: fall back to default edge-tts
        return "edge"
    print("\nSelect voice engine (free tier):")
    print("  [1] edge-tts    - free, best quality, needs network to services.google.com")
    print("  [2] cloudflare  - Workers AI Melotts, 'cf' OAuth girişi (Türkçe desteklemez)")
    print("  [3] ttsai       - tts.ai quality voices, 15k free chars/day, needs TTS_AI_API_KEY (get one free at tts.ai)")
    print("  [4] pockettts   - çevrimdışı Türkçe Pocket TTS (CPU), 6 ses + duygu + klonlama")
    print("  [5] skip        - use default (edge-tts) for this run")
    try:
        choice = input("  Enter 1-5 (default 1): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "edge"
    mapping = {"1": "edge", "2": "cloudflare", "3": "ttsai", "4": "pockettts"}
    return mapping.get(choice, "edge")


def main():
    args = parse_args()

    if args.cf_login:
        from services.cloudflare_cli import CloudflareCLI
        client = CloudflareCLI(
            command=config.CLOUDFLARE_CLI_COMMAND,
            profile=config.CLOUDFLARE_CLI_PROFILE,
            timeout=config.CLOUDFLARE_CLI_TIMEOUT,
        )
        print("Cloudflare CLI giriş akışı başlatılıyor (API anahtarı gerekmez)...")
        try:
            result = client.login()
        except FileNotFoundError:
            print("Hata: 'cf' bulunamadı. Önce 'npm install --global cf' çalıştırın.")
            sys.exit(1)
        if result != 0:
            print("Cloudflare girişi tamamlanamadı.")
            sys.exit(result)
        print("Cloudflare oturumu hazır; Workers AI çağrıları API anahtarı olmadan kullanılabilir.")
        sys.exit(0)

    if not args.topic:
        print("YouTube Shorts Generator")
        print("=" * 40)
        print("\nUsage: python generate_shorts.py <topic> [options]")
        print("\nExample:")
        print('  python generate_shorts.py "Yapay zeka nedir?"')
        print("\nMake sure to set your API keys in .env file:")
        print("  - GEMINI_API_KEY")
        print("\nRun with --help for more options.")
        sys.exit(1)

    topic = args.topic

    # A provider is required: Gemini key or an authenticated cf CLI session.
    from services.cloudflare_cli import CloudflareCLI
    cf_client = CloudflareCLI(
        command=config.CLOUDFLARE_CLI_COMMAND,
        profile=config.CLOUDFLARE_CLI_PROFILE,
        timeout=config.CLOUDFLARE_CLI_TIMEOUT,
    )
    if ((not config.GEMINI_API_KEY
         or config.GEMINI_API_KEY == "your_gemini_api_key_here")
            and not cf_client.available()):
        print("Error: AI provider unavailable.")
        print("Run '.venv/bin/python generate_shorts.py --cf-login' "
              "or set GEMINI_API_KEY in .env.")
        sys.exit(1)

    # Image approval mode (step 2): None = auto-detect TTY, False = never
    interactive = False if args.no_approve else None

    # --- Step 0: choose TTS engine ---
    tts_engine = _choose_tts_engine(args.tts)
    config.TTS_PROVIDER = tts_engine
    print(f"Voice engine: {tts_engine} ({config.TTS_PROVIDER})")

    # Run the pipeline
    try:
        pipeline = ShortsPipeline()
        output_path = pipeline.run(topic, interactive=interactive)

        if output_path and Path(output_path).exists():
            print(f"\n✓ Video successfully created: {output_path}")
        else:
            print("\n✗ Video generation failed - no output file created")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nGeneration cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
