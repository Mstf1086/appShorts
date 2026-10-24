"""Tests for ShortsPipeline."""
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Ensure project root is in path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock google.genai before importing (moviepy must stay REAL: other test
# modules rely on services.video_assembler binding the genuine classes)
sys.modules['google'] = MagicMock()
sys.modules['google.genai'] = MagicMock()
sys.modules['google.genai.types'] = MagicMock()

from services.scene_generator import Scene
from pipeline.shorts_pipeline import ShortsPipeline


class TestShortsPipeline(unittest.TestCase):
    """Test the ShortsPipeline class."""

    def test_pipeline_initialization(self):
        """Test that pipeline initializes all components."""
        with patch('pipeline.shorts_pipeline.SceneGenerator'), \
             patch('pipeline.shorts_pipeline.ImageGenerator'), \
             patch('pipeline.shorts_pipeline.get_voice_generator'), \
             patch('pipeline.shorts_pipeline.SubtitleSync'), \
             patch('pipeline.shorts_pipeline.VideoAssembler'), \
             patch('pipeline.shorts_pipeline.config') as mock_config:
            mock_config.GEMINI_API_KEY = "test_key"
            mock_config.APPROVE_IMAGES = False
            mock_config.VIDEO_WIDTH = 1080
            mock_config.VIDEO_HEIGHT = 1920
            pipeline = ShortsPipeline()
            self.assertIsNotNone(pipeline.scene_generator)
            self.assertIsNotNone(pipeline.image_generator)
            self.assertIsNotNone(pipeline.voice_generator)
            self.assertIsNotNone(pipeline.subtitle_sync)
            self.assertIsNotNone(pipeline.video_assembler)
            self.assertEqual(pipeline.scenes, [])

    def test_selected_models_are_passed_to_services(self):
        """Explicit model choices reach each service constructor."""
        with patch('pipeline.shorts_pipeline.SceneGenerator') as scene_cls, \
             patch('pipeline.shorts_pipeline.ImageGenerator') as image_cls, \
             patch('pipeline.shorts_pipeline.get_voice_generator') as voice_factory:
            ShortsPipeline(
                voice_service="cloudflare",
                text_model="@cf/openai/gpt-oss-20b",
                image_model="@cf/black-forest-labs/flux-2-klein-4b",
                tts_model="@cf/deepgram/aura-2-en",
                text_provider="cloudflare",
                image_provider="cloudflare",
            )
        scene_cls.assert_called_once_with(
            model="@cf/openai/gpt-oss-20b", provider="cloudflare")
        image_cls.assert_called_once_with(
            cloudflare_model="@cf/black-forest-labs/flux-2-klein-4b",
            provider="cloudflare")
        voice_factory.assert_called_once_with(
            "cloudflare", model="@cf/deepgram/aura-2-en")

    @patch('pipeline.shorts_pipeline.config')
    def test_pipeline_run(self, mock_config):
        """Test the full pipeline run with mocked components."""
        mock_config.GEMINI_API_KEY = "test_key"
        mock_config.APPROVE_IMAGES = False
        mock_config.VIDEO_WIDTH = 1080
        mock_config.VIDEO_HEIGHT = 1920

        test_scenes = [
            Scene(id=1, text="Metin 1", visual_description="V1", search_query="Q1", duration=10.0),
            Scene(id=2, text="Metin 2", visual_description="V2", search_query="Q2", duration=10.0),
            Scene(id=3, text="Metin 3", visual_description="V3", search_query="Q3", duration=10.0),
            Scene(id=4, text="Metin 4", visual_description="V4", search_query="Q4", duration=10.0),
            Scene(id=5, text="Metin 5", visual_description="V5", search_query="Q5", duration=10.0),
        ]

        with patch('pipeline.shorts_pipeline.SceneGenerator') as MockSceneGen, \
             patch('pipeline.shorts_pipeline.ImageGenerator'), \
             patch('pipeline.shorts_pipeline.get_voice_generator') as MockVoiceGen, \
             patch('pipeline.shorts_pipeline.SubtitleSync') as MockSubSync, \
             patch('pipeline.shorts_pipeline.VideoAssembler') as MockAssembler:

                            # Mock scene generator
                            mock_scene_gen = MagicMock()
                            mock_scene_gen.generate_scenes.return_value = test_scenes
                            mock_scene_gen.validate_scenes.return_value = True
                            MockSceneGen.return_value = mock_scene_gen

                            # Mock image generator
                            mock_fetcher = MagicMock()
                            mock_fetcher.generate_all_images.return_value = test_scenes

                            # Mock voice generator
                            mock_voice = MagicMock()
                            mock_voice.generate_all_voiceovers.return_value = test_scenes

                            # Mock subtitle sync
                            mock_sub_sync = MagicMock()
                            mock_sub_sync.generate_srt.return_value = "/tmp/subtitles.srt"
                            MockSubSync.return_value = mock_sub_sync

                            # Mock video assembler
                            mock_assembler = MagicMock()
                            mock_assembler.assemble_video.return_value = "/tmp/output.mp4"
                            MockAssembler.return_value = mock_assembler

                            # Patch all components in __init__
                            with patch.object(ShortsPipeline, '__init__', lambda self: None):
                                pipeline = ShortsPipeline()
                                pipeline.scene_generator = mock_scene_gen
                                pipeline.image_generator = mock_fetcher
                                pipeline.voice_generator = mock_voice
                                pipeline.subtitle_sync = mock_sub_sync
                                pipeline.video_assembler = mock_assembler
                                pipeline.scenes = []

                                output = pipeline.run("Test topic")

                                self.assertEqual(output, "/tmp/output.mp4")
                                self.assertEqual(len(pipeline.scenes), 5)
                                mock_scene_gen.generate_scenes.assert_called_once_with("Test topic")

    def test_run_with_progress_delegates_to_run(self):
        """Test that run_with_progress delegates to run."""
        with patch.object(ShortsPipeline, 'run', return_value="/tmp/output.mp4") as mock_run:
            with patch.object(ShortsPipeline, '__init__', lambda self: None):
                pipeline = ShortsPipeline()

                result = pipeline.run_with_progress("Test topic")

                self.assertEqual(result, "/tmp/output.mp4")
                mock_run.assert_called_once_with("Test topic", interactive=None)


if __name__ == "__main__":
    unittest.main()
