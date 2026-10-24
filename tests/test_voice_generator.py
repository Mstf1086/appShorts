"""Tests for VoiceGenerator service."""
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import os
from pathlib import Path

# Ensure project root is in path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock google.genai before importing scene_generator
sys.modules['google'] = MagicMock()
sys.modules['google.genai'] = MagicMock()
sys.modules['google.genai.types'] = MagicMock()

from services.scene_generator import Scene
from services.voice_generator import VoiceGenerator


class TestVoiceGenerator(unittest.TestCase):
    """Test the VoiceGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_scene = Scene(
            id=1,
            text="Test metin seslendirme.",
            visual_description="Test visual",
            search_query="test query",
            duration=8.0
        )

    @patch('services.voice_generator.config')
    def test_voice_generator_initialization(self, mock_config):
        """Test VoiceGenerator initialization."""
        mock_config.VOICE_NAME = "tr-TR-TestNeural"
        mock_config.AUDIO_DIR = Path("/tmp/test_audio")

        generator = VoiceGenerator()
        self.assertEqual(generator.voice_name, "tr-TR-TestNeural")

    @patch('services.voice_generator.config')
    def test_voice_generator_custom_voice(self, mock_config):
        """Test VoiceGenerator with custom voice name."""
        mock_config.AUDIO_DIR = Path("/tmp/test_audio")

        generator = VoiceGenerator(voice_name="tr-TR-CustomNeural")
        self.assertEqual(generator.voice_name, "tr-TR-CustomNeural")

    @patch('services.voice_generator.config')
    def test_generate_voiceover_success(self, mock_config):
        """Test successful voiceover generation."""
        mock_config.VOICE_NAME = "tr-TR-TestNeural"
        mock_config.AUDIO_DIR = Path("/tmp/test_audio")
        mock_config.TTS_PROVIDER = "edge"
        mock_config.TTS_MAX_RETRIES = 3

        generator = VoiceGenerator()

        # Mock the async _generate_single method
        with patch.object(generator, '_generate_single', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "/tmp/test_audio/scene_01.mp3"

            # Create the audio file so the check passes
            os.makedirs("/tmp/test_audio", exist_ok=True)
            with open("/tmp/test_audio/scene_01.mp3", "wb") as f:
                f.write(b"fake audio data")

            result = generator.generate_voiceover(self.test_scene)

            self.assertEqual(result, "/tmp/test_audio/scene_01.mp3")

            # Clean up
            Path("/tmp/test_audio/scene_01.mp3").unlink()

    @patch('services.voice_generator.config')
    def test_generate_voiceover_failure(self, mock_config):
        """Test voiceover generation failure."""
        mock_config.VOICE_NAME = "tr-TR-TestNeural"
        mock_config.AUDIO_DIR = Path("/tmp/test_audio")
        mock_config.TTS_PROVIDER = "edge"
        mock_config.TTS_MAX_RETRIES = 3

        generator = VoiceGenerator()

        # Mock the async method to always fail
        with patch.object(generator, '_generate_single', new_callable=AsyncMock) as mock_gen:
            mock_gen.side_effect = Exception("TTS error")

            result = generator.generate_voiceover(self.test_scene, max_retries=2)

            self.assertIsNone(result)

    @patch('services.voice_generator.config')
    def test_generate_all_voiceovers(self, mock_config):
        """Test generating voiceovers for all scenes."""
        mock_config.VOICE_NAME = "tr-TR-TestNeural"
        mock_config.AUDIO_DIR = Path("/tmp/test_audio")

        scenes = [
            Scene(id=1, text="Metin 1", visual_description="V1", search_query="Q1", duration=5.0),
            Scene(id=2, text="Metin 2", visual_description="V2", search_query="Q2", duration=5.0)
        ]

        generator = VoiceGenerator()

        with patch.object(generator, 'generate_voiceover') as mock_gen:
            mock_gen.return_value = "/tmp/test_audio/scene.mp3"
            result = generator.generate_all_voiceovers(scenes)

            self.assertEqual(len(result), 2)
            self.assertEqual(mock_gen.call_count, 2)

    @patch('services.voice_generator.config')
    def test_get_audio_duration(self, mock_config):
        """Test getting audio duration with non-existent file returns 0.0."""
        mock_config.VOICE_NAME = "tr-TR-TestNeural"
        mock_config.AUDIO_DIR = Path("/tmp/test_audio")

        generator = VoiceGenerator()

        # Mock the AudioFileClip import inside get_audio_duration to raise exception
        mock_audio_clip_class = MagicMock()
        mock_audio_clip_class.side_effect = Exception("File not found")

        with patch.dict('sys.modules', {'moviepy': MagicMock(AudioFileClip=mock_audio_clip_class)}):
            # Re-import to pick up the mocked moviepy
            import importlib
            import services.voice_generator as vg_module
            # Patch the import inside the method
            with patch.object(vg_module, 'moviepy', create=True) as mock_mp:
                mock_mp.AudioFileClip.side_effect = Exception("File not found")
                duration = generator.get_audio_duration("/nonexistent/file.mp3")
                self.assertEqual(duration, 0.0)


    @patch('services.voice_generator.config')
    def test_generate_voiceover_retries_with_backoff(self, mock_config):
        """403/throttle: failed attempts wait with exponential backoff, then succeed."""
        mock_config.VOICE_NAME = "tr-TR-TestNeural"
        mock_config.AUDIO_DIR = Path("/tmp/test_audio")
        mock_config.TTS_PROVIDER = "edge"
        mock_config.TTS_MAX_RETRIES = 3
        mock_config.TTS_RETRY_BASE_DELAY = 3.0

        generator = VoiceGenerator()

        with patch.object(generator, '_generate_single', new_callable=AsyncMock) as mock_gen:
            mock_gen.side_effect = [Exception("403"), "/tmp/test_audio/scene_01.mp3"]
            with patch("services.voice_generator.time.sleep") as mock_sleep:
                result = generator.generate_voiceover(self.test_scene)

                self.assertEqual(result, "/tmp/test_audio/scene_01.mp3")
                self.assertEqual(mock_gen.await_count, 2)
                mock_sleep.assert_called_once_with(3.0)

    @patch('services.voice_generator.config')
    def test_edge_no_audio_fails_without_quality_fallback(self, mock_config):
        """An empty Edge response must fail instead of using a poor fallback."""
        mock_config.VOICE_NAME = "tr-TR-TestNeural"
        mock_config.AUDIO_DIR = Path("/tmp/test_audio")
        mock_config.TTS_PROVIDER = "edge"
        mock_config.TTS_MAX_RETRIES = 5
        mock_config.TTS_RETRY_BASE_DELAY = 3.0

        generator = VoiceGenerator()
        with patch.object(
            generator,
            "_generate_single",
            new_callable=AsyncMock,
            side_effect=Exception("No audio was received."),
        ) as edge_mock:
            result = generator.generate_voiceover(self.test_scene)

        self.assertIsNone(result)
        self.assertEqual(edge_mock.await_count, 1)

    @patch('services.voice_generator.config')
    def test_generate_all_voiceovers_rejects_missing_audio(self, mock_config):
        """The pipeline must not continue with a scene that has no audio."""
        mock_config.AUDIO_DIR = Path("/tmp/test_audio")
        scenes = [
            Scene(id=1, text="Metin 1", visual_description="V1", search_query="Q1", duration=5.0),
            Scene(id=2, text="Metin 2", visual_description="V2", search_query="Q2", duration=5.0),
        ]
        generator = VoiceGenerator()
        with patch.object(
            generator,
            "generate_voiceover",
            side_effect=["/tmp/test_audio/scene_01.mp3", None],
        ):
            with self.assertRaisesRegex(RuntimeError, "Seslendirme eksik"):
                generator.generate_all_voiceovers(scenes)

    @patch('services.voice_generator.config')
    def test_generate_voiceover_inside_running_event_loop(self, mock_config):
        """A running event loop must not cause an asyncio.run error."""
        import asyncio

        mock_config.VOICE_NAME = "tr-TR-TestNeural"
        mock_config.AUDIO_DIR = Path("/tmp/test_audio_loop")
        mock_config.TTS_PROVIDER = "edge"

        generator = VoiceGenerator()
        results = {}

        async def running_loop():
            # Sync call from inside a running loop.
            with patch.object(generator, '_generate_single', new_callable=AsyncMock) as mock_gen:
                mock_gen.return_value = "/tmp/test_audio_loop/scene_01.mp3"
                os.makedirs("/tmp/test_audio_loop", exist_ok=True)
                with open("/tmp/test_audio_loop/scene_01.mp3", "wb") as f:
                    f.write(b"fake audio data")
                results["result"] = generator.generate_voiceover(self.test_scene)

        asyncio.run(running_loop())

        self.assertEqual(results["result"], "/tmp/test_audio_loop/scene_01.mp3")
        Path("/tmp/test_audio_loop/scene_01.mp3").unlink()


if __name__ == "__main__":
    unittest.main()
