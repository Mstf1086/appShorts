"""Tests for the keyless Cloudflare Workers AI TTS service."""
import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from services.cloudflare_tts import CloudflareTTS


class TestCloudflareTTS(unittest.TestCase):
    """Melotts CLI requests use the current Workers AI input contract."""

    @patch("services.cloudflare_tts.CloudflareCLI.available", return_value=True)
    def test_cli_uses_melotts_prompt_and_decodes_audio(self, _available):
        audio = b"RIFF" + (b"\x00" * 1200)
        response = json.dumps({"audio": base64.b64encode(audio).decode()}).encode()
        with tempfile.TemporaryDirectory(prefix="cf-tts-") as tmp:
            output = Path(tmp) / "scene.wav"
            with patch("services.cloudflare_tts.config.CLOUDFLARE_TTS_MODEL",
                       "@cf/myshell-ai/melotts"), \
                 patch("services.cloudflare_tts.config.CLOUDFLARE_TTS_LANGUAGE",
                       "en"):
                generator = CloudflareTTS()
                generator.cloudflare_cli.run_ai = MagicMock(
                    return_value=response)
                result = generator._generate_single_cli("Hello", str(output))
            self.assertEqual(result, str(output))
            self.assertEqual(output.read_bytes(), audio)
            model, body = generator.cloudflare_cli.run_ai.call_args.args
            self.assertEqual(model, "@cf/myshell-ai/melotts")
            self.assertEqual(body, {"prompt": "Hello", "lang": "en"})

    @patch("services.cloudflare_tts.CloudflareCLI.available", return_value=True)
    def test_aura_uses_binary_request_and_wav_payload(self, _available):
        audio = b"RIFF" + (b"\x00" * 1200)
        with tempfile.TemporaryDirectory(prefix="cf-tts-") as tmp:
            output = Path(tmp) / "scene.wav"
            generator = CloudflareTTS(model="@cf/deepgram/aura-2-en")
            generator.cloudflare_cli.run_ai_binary = MagicMock(
                return_value=audio)
            result = generator._generate_single_cli("Hello", str(output))
            self.assertEqual(result, str(output))
            self.assertEqual(output.read_bytes(), audio)
        model, body = generator.cloudflare_cli.run_ai_binary.call_args.args
        self.assertEqual(model, "@cf/deepgram/aura-2-en")
        self.assertEqual(body["text"], "Hello")
        self.assertEqual(body["encoding"], "linear16")
        self.assertEqual(body["container"], "wav")
        self.assertEqual(body["speaker"], "luna")

    @patch("services.cloudflare_tts.CloudflareCLI.available", return_value=True)
    def test_legacy_deepgram_model_is_remapped(self, _available):
        with patch("services.cloudflare_tts.config.CLOUDFLARE_TTS_MODEL",
                   "deepgram/aura-2-en"):
            generator = CloudflareTTS()
        self.assertEqual(generator.model_name, "@cf/myshell-ai/melotts")


if __name__ == "__main__":
    unittest.main()
