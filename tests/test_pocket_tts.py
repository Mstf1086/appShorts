"""Tests for the offline Pocket TTS Turkish service.

The optional ``pocket-tts-turkish`` package is never imported: the engine is
mocked, so these tests also prove that the default edge-tts path keeps working
without the dependency installed.
"""
import io
import shutil
import sys
import tempfile
import unittest
import wave
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

import config
from services.audio_factory import get_voice_generator
from services.pocket_tts import PocketTTS
from services.scene_generator import Scene
from services.voice_generator import VoiceGenerator


def _engine(sample_rate: int = 24000, samples=None) -> MagicMock:
    """Return a fake pocket-tts-turkish engine."""
    engine = MagicMock()
    engine.sample_rate = sample_rate
    engine.generate.return_value = (
        np.zeros(2400, dtype=np.float32) if samples is None else samples)
    return engine


class PocketTTSTestCase(unittest.TestCase):
    """Shared setup: a real temp voice directory, never the project .tmp."""

    def setUp(self):
        self.voice_dir = Path(tempfile.mkdtemp(prefix="pocket-tts-"))
        self.addCleanup(shutil.rmtree, str(self.voice_dir), ignore_errors=True)

    def _generator(self, **kwargs) -> PocketTTS:
        generator = PocketTTS(**kwargs)
        generator.output_dir = self.voice_dir
        return generator

    @staticmethod
    def _scene(scene_id: int = 1) -> Scene:
        return Scene(
            id=scene_id,
            text="Sahne metni",
            visual_description="scene visual",
            search_query="query",
            duration=8.0,
        )


class TestConfiguration(PocketTTSTestCase):
    """Config-driven defaults, emotion/seed normalization."""

    @patch("services.pocket_tts.config")
    def test_defaults_come_from_config(self, mock_config):
        mock_config.POCKET_TTS_VOICE = "male_2"
        mock_config.POCKET_TTS_EMOTION = "mutlu"
        mock_config.POCKET_TTS_DEVICE = "cpu"
        mock_config.POCKET_TTS_MODEL = "wite-tech/pocket-tts-turkish"
        mock_config.POCKET_TTS_VOICE_CLONE = ""
        mock_config.POCKET_TTS_SEED = "7"
        mock_config.AUDIO_DIR = self.voice_dir

        generator = PocketTTS()

        self.assertEqual(generator.voice, "male_2")
        self.assertEqual(generator.emotion, "mutlu")
        self.assertEqual(generator.device, "cpu")
        self.assertEqual(generator.model_name, "wite-tech/pocket-tts-turkish")
        self.assertEqual(generator.seed, 7)

    def test_explicit_arguments_win_over_config(self):
        generator = self._generator(voice="female_3", emotion="sakin",
                                    device="cpu", model="custom/repo")
        self.assertEqual(generator.voice, "female_3")
        self.assertEqual(generator.emotion, "sakin")
        self.assertEqual(generator.device, "cpu")
        self.assertEqual(generator.model_name, "custom/repo")

    def test_unknown_emotion_warns_and_falls_back_to_neutral(self):
        captured = io.StringIO()
        with redirect_stdout(captured):
            generator = self._generator(emotion="heyecanli")
        self.assertEqual(generator.emotion, "")
        self.assertIn("Unknown Pocket TTS emotion", captured.getvalue())

    def test_english_emotion_alias_is_accepted(self):
        self.assertEqual(self._generator(emotion="happy").emotion, "mutlu")
        self.assertEqual(self._generator(emotion="sakin").emotion, "sakin")

    def test_seed_defaults_to_random_for_invalid_values(self):
        with patch("services.pocket_tts.config") as mock_config:
            mock_config.POCKET_TTS_SEED = "not-a-number"
            mock_config.AUDIO_DIR = self.voice_dir
            self.assertIsNone(PocketTTS().seed)

    def test_available_is_false_without_the_optional_package(self):
        # ``None`` in sys.modules makes ``import pocket_tts_turkish`` raise.
        with patch.dict(sys.modules, {"pocket_tts_turkish": None}):
            self.assertFalse(self._generator().available())



class TestSynthesis(PocketTTSTestCase):
    """WAV writing and argument plumbing."""

    def test_generate_writes_mono_16bit_wav(self):
        engine = _engine(sample_rate=24000)
        generator = self._generator(voice="male_1")
        target = self.voice_dir / "scene.wav"

        with patch.object(PocketTTS, "_load_engine", return_value=engine):
            result = generator.generate("Merhaba dünya", str(target))

        self.assertEqual(result, str(target))
        with wave.open(str(target), "rb") as handle:
            self.assertEqual(handle.getnchannels(), 1)
            self.assertEqual(handle.getsampwidth(), 2)
            self.assertEqual(handle.getframerate(), 24000)
        engine.generate.assert_called_once_with("Merhaba dünya", voice="male_1")

    def test_generate_voiceover_passes_emotion_seed_and_wav_name(self):
        engine = _engine()
        generator = self._generator(voice="female_2", emotion="üzgün")
        generator.seed = 5

        with patch.object(PocketTTS, "_load_engine", return_value=engine):
            result = generator.generate_voiceover(self._scene(3))

        self.assertTrue(result.endswith("scene_03.wav"))
        engine.generate.assert_called_once_with(
            "Sahne metni", voice="female_2", emotion="üzgün", seed=5)

    def test_empty_audio_returns_none_without_a_file(self):
        engine = _engine(samples=np.array([], dtype=np.float32))
        generator = self._generator()
        target = self.voice_dir / "empty.wav"

        with patch.object(PocketTTS, "_load_engine", return_value=engine):
            result = generator.generate("Merhaba", str(target))

        self.assertIsNone(result)
        self.assertFalse(target.exists())

    def test_cloned_reference_is_loaded_once_and_used_as_voice(self):
        engine = _engine()
        engine.voice_from_file.return_value = "cloned-voice-state"
        generator = self._generator(clone_reference="kayit.wav")

        with patch.object(PocketTTS, "_load_engine", return_value=engine):
            generator.generate("Birinci metin", str(self.voice_dir / "a.wav"))
            generator.generate("İkinci metin", str(self.voice_dir / "b.wav"))

        engine.voice_from_file.assert_called_once_with("kayit.wav")
        self.assertEqual(engine.generate.call_count, 2)
        for call in engine.generate.call_args_list:
            self.assertEqual(call.kwargs["voice"], "cloned-voice-state")


class TestRetryPolicy(PocketTTSTestCase):
    """Transient engine failures follow the shared retry/backoff policy."""

    def test_failure_is_retried_then_reported(self):
        engine = _engine()
        engine.generate.side_effect = RuntimeError("model download failed")
        generator = self._generator()
        captured = io.StringIO()

        with patch.object(PocketTTS, "_load_engine", return_value=engine), \
             patch("services.pocket_tts.config.TTS_RETRY_BASE_DELAY", 0.0), \
             redirect_stdout(captured):
            result = generator.generate_voiceover(self._scene(), max_retries=2)

        self.assertIsNone(result)
        self.assertEqual(engine.generate.call_count, 2)
        self.assertIn("model download failed", captured.getvalue())

    def test_transient_failure_then_success_returns_path(self):
        engine = _engine()
        engine.generate.side_effect = [
            RuntimeError("boom"), np.zeros(480, dtype=np.float32)]
        generator = self._generator()

        with patch.object(PocketTTS, "_load_engine", return_value=engine), \
             patch("services.pocket_tts.config.TTS_RETRY_BASE_DELAY", 0.0):
            result = generator.generate_voiceover(self._scene(2), max_retries=3)

        self.assertTrue(result.endswith("scene_02.wav"))
        self.assertEqual(engine.generate.call_count, 2)


class TestAllVoiceovers(PocketTTSTestCase):
    """The shared loop syncs durations and stops instead of emitting silence."""

    def test_duration_follows_audio_and_word_timings_stay_empty(self):
        engine = _engine()
        generator = self._generator()
        scenes = [self._scene(1)]

        with patch.object(PocketTTS, "_load_engine", return_value=engine), \
             patch.object(PocketTTS, "get_audio_duration", return_value=3.5):
            result = generator.generate_all_voiceovers(scenes)

        self.assertEqual(result, scenes)
        self.assertTrue(scenes[0].audio_path.endswith("scene_01.wav"))
        self.assertEqual(scenes[0].word_timings, [])
        self.assertAlmostEqual(scenes[0].duration, 3.9, places=2)

    def test_missing_voiceover_stops_the_pipeline(self):
        engine = _engine()
        engine.generate.side_effect = RuntimeError("no model")
        generator = self._generator()

        with patch.object(PocketTTS, "_load_engine", return_value=engine), \
             patch("services.pocket_tts.config.TTS_MAX_RETRIES", 1), \
             patch("services.pocket_tts.config.TTS_RETRY_BASE_DELAY", 0.0), \
             redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                generator.generate_all_voiceovers([self._scene(1)])


class TestFactoryIntegration(PocketTTSTestCase):
    """Provider routing for the ``pockettts`` value."""

    def test_factory_returns_pocket_engine_for_provider(self):
        with patch.object(config, "TTS_PROVIDER", "pockettts"), \
             patch.object(config, "AUDIO_DIR", self.voice_dir):
            generator = get_voice_generator()

        self.assertIsInstance(generator, PocketTTS)
        self.assertEqual(generator.output_dir, self.voice_dir)

    def test_voice_generator_delegates_when_provider_is_pockettts(self):
        with patch.object(config, "TTS_PROVIDER", "pockettts"), \
             patch.object(config, "AUDIO_DIR", self.voice_dir), \
             patch("services.pocket_tts.PocketTTS.generate_voiceover",
                   return_value="/tmp/pocket.wav") as delegated:
            result = VoiceGenerator().generate_voiceover(self._scene())

        self.assertEqual(result, "/tmp/pocket.wav")
        delegated.assert_called_once()


if __name__ == "__main__":
    unittest.main()

