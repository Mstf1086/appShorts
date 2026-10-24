"""Cloudflare Workers AI text-to-speech service.

Supports the authenticated ``cf`` CLI session and the plain REST fallback.
Workers AI Melotts returns base64 WAV audio; the CLI path requires no API key.
"""
import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

import config
from services.cloudflare_cli import CloudflareCLI
from services.cloudflare_model_catalog import (
    build_tts_payload,
    decode_tts_audio,
)
from services.scene_generator import Scene


class CloudflareTTS:
    """Generate voiceovers from Cloudflare Workers AI text-to-speech."""

    def __init__(self, account_id: str = None, api_token: str = None,
                 model: str = None):
        self.account_id = (account_id or config.CLOUDFLARE_ACCOUNT_ID or "").strip()
        self.api_token = (api_token or config.CLOUDFLARE_API_TOKEN or "").strip()
        self.cloudflare_cli = CloudflareCLI(
            command=getattr(config, "CLOUDFLARE_CLI_COMMAND", "cf"),
            profile=getattr(config, "CLOUDFLARE_CLI_PROFILE", ""),
            timeout=getattr(config, "CLOUDFLARE_CLI_TIMEOUT", 120),
        )
        if not self.cloudflare_cli.available() and (not self.account_id or not self.api_token):
            raise ValueError(
                "Cloudflare Workers AI requires a logged-in 'cf' CLI session "
                "or CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN.")
        # Dedicated TTS model or fall back to the image model token.
        self.model_name = (model or config.CLOUDFLARE_TTS_MODEL
                           or config.CLOUDFLARE_IMAGE_MODEL)
        # Preserve the legacy bare model alias mapping; full @cf Aura IDs are
        # now first-class selectable models and must not be remapped.
        if str(self.model_name).startswith("deepgram/"):
            self.model_name = "@cf/myshell-ai/melotts"
        self.model_names = [
            m.strip() for m in str(self.model_name).split(",") if m.strip()
        ]
        self._model_idx = 0
        self.output_dir = config.AUDIO_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._quota_limited = False

    # --------------------------------------------------------- worker API
    @staticmethod
    def _quota_error(exc: Exception) -> bool:
        text = str(exc)
        return ("429" in text or "RESOURCE_EXHAUSTED" in text
                or "quota" in text.lower())

    def _url(self) -> str:
        return (f"https://api.cloudflare.com/client/v4/accounts/"
                f"{self.account_id}/ai/run/{self.model_names[self._model_idx]}")

    def _generate_single_cli(self, text: str, output_path: str) -> Optional[str]:
        """Run one TTS request with the authenticated Cloudflare CLI."""
        model = self.model_names[self._model_idx]
        inputs = build_tts_payload(model, text)
        if "melotts" in model.lower():
            inputs["lang"] = getattr(config, "CLOUDFLARE_TTS_LANGUAGE", "en")
        if "melotts" in model.lower():
            raw = self.cloudflare_cli.run_ai(model, inputs)
        else:
            raw = self.cloudflare_cli.run_ai_binary(model, inputs)
        audio_bytes = decode_tts_audio(raw)
        if len(audio_bytes) < 1000:
            raise RuntimeError("Empty/too-small TTS audio response")
        Path(output_path).write_bytes(audio_bytes)
        print(f"  Workers AI TTS (CLI): {Path(output_path).name} "
              f"({len(audio_bytes)} bytes)")
        return output_path

    def _generate_single(self, text: str, output_path: str) -> Optional[str]:
        """Run one TTS request via CLI OAuth or Workers AI REST."""
        if self.cloudflare_cli.available():
            return self._generate_single_cli(text, output_path)
        model = self.model_names[self._model_idx]
        request_data = build_tts_payload(model, text)
        if "melotts" in model.lower():
            request_data["lang"] = getattr(config, "CLOUDFLARE_TTS_LANGUAGE", "en")
        body = json.dumps(request_data).encode("utf-8")
        req = urllib.request.Request(
            self._url(), data=body,
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            })
        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=config.CLOUDFLARE_TTS_TIMEOUT) as resp:
                data = resp.read()
        except urllib.error.HTTPError as e:
            try:
                err_text = e.read().decode("utf-8", errors="replace")
            except Exception:
                err_text = str(e)
            if "429" in str(e.code) or "RESOURCE_EXHAUSTED" in err_text:
                self._quota_limited = True
            raise RuntimeError(f"Workers AI {e.code}: {err_text[:200]}")
        elapsed = time.monotonic() - start
        try:
            audio_bytes = decode_tts_audio(data)
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected Workers AI response ({len(data)} bytes): {exc}") from exc
        if not audio_bytes or len(audio_bytes) < 1000:
            raise RuntimeError("Empty/too-small TTS audio response")
        Path(output_path).write_bytes(audio_bytes)
        print(f"  Workers AI TTS ({elapsed:.1f}s): {Path(output_path).name} "
              f"({len(audio_bytes)} bytes)")
        return output_path

    def generate(self, text: str, output_path: str = None,
                 max_retries: int = None) -> Optional[str]:
        """Generate voiceover for one scene with quota-based retry/failover."""
        if max_retries is None:
            try:
                max_retries = int(getattr(config, "TTS_MAX_RETRIES", 5))
            except (TypeError, ValueError):
                max_retries = 5
        filename = "scene_audio.wav"
        output_path = output_path or str(self.output_dir / filename)
        for attempt in range(max_retries):
            try:
                return self._generate_single(text, output_path)
            except Exception as e:
                if self._quota_error(e) and not self._quota_limited:
                    self._model_idx = (self._model_idx + 1) % len(self.model_names)
                    print(f"  Attempt {attempt + 1}/{max_retries} failed "
                          f"(model {self._model_idx + 1}): {e}")
                    time.sleep(5)
                else:
                    print(f"  Attempt {attempt + 1}/{max_retries} failed: {e}")
                    time.sleep(2 ** attempt)
        print(f"  ✗ Cloudflare TTS failed after {max_retries} attempts")
        return None

    def generate_all_voiceovers(self, scenes: List[Scene]) -> List[Scene]:
        """Generate voiceovers for all scenes, syncing duration like Edge TTS."""
        for i, scene in enumerate(scenes):
            print(f"\n[{i + 1}/{len(scenes)}] Generating voiceover "
                  f"for scene {scene.id} (Cloudflare TTS)")
            if i > 0:
                try:
                    scene_delay = float(getattr(config, "TTS_SCENE_DELAY", 1.5))
                except (TypeError, ValueError):
                    scene_delay = 1.5
                time.sleep(scene_delay)
            audio_path = self.generate(scene.text)
            if audio_path:
                scene.audio_path = audio_path
                from services.subtitle_sync import SubtitleSync
                sub = SubtitleSync()
                entries = sub.get_subtitle_entries([scene])
                words = []
                for _, _, text in entries:
                    for w in text.split():
                        if w:
                            words.append({"word": w, "start": 0.0, "end": 0.0})
                scene.word_timings = words
                if scene.word_timings:
                    print(f"  Word timings: {len(scene.word_timings)} words")
                n_words = len(scene.word_timings)
                scene.duration = round(max(n_words * 0.9, 1.0)
                                       + config.SCENE_AUDIO_PADDING, 2)
                print(f"  Scene {scene.id} duration synced: {scene.duration:.2f}s")
            else:
                print(f"  Warning: Could not generate voiceover "
                      f"for scene {scene.id} (Cloudflare TTS)")
        return scenes


if __name__ == "__main__":
    gen = CloudflareTTS()
    print("Workers AI TTS ready:", gen.model_name)