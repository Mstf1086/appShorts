"""Tests for Scene dataclass and SceneGenerator."""
import unittest
from unittest.mock import patch, MagicMock
import json
import os
from dataclasses import dataclass

# Ensure project root is in path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock google.genai before importing scene_generator (not available on this platform)
sys.modules['google'] = MagicMock()
sys.modules['google.genai'] = MagicMock()
sys.modules['google.genai.types'] = MagicMock()

from services.scene_generator import Scene, SceneGenerator


class TestScene(unittest.TestCase):
    """Test the Scene dataclass."""

    def test_scene_creation(self):
        """Test creating a Scene with all required fields."""
        scene = Scene(
            id=1,
            text="Test metin",
            visual_description="Test visual",
            search_query="test query",
            duration=8.0
        )
        self.assertEqual(scene.id, 1)
        self.assertEqual(scene.text, "Test metin")
        self.assertEqual(scene.visual_description, "Test visual")
        self.assertEqual(scene.search_query, "test query")
        self.assertEqual(scene.duration, 8.0)
        self.assertEqual(scene.image_path, "")
        self.assertEqual(scene.image_candidates, [])
        self.assertEqual(scene.audio_path, "")

    def test_scene_with_optional_fields(self):
        """Test creating a Scene with optional fields."""
        scene = Scene(
            id=2,
            text="Test metin",
            visual_description="Test visual",
            search_query="test query",
            duration=10.0,
            image_path="/path/to/image.png",
            audio_path="/path/to/audio.mp3"
        )
        self.assertEqual(scene.image_path, "/path/to/image.png")
        self.assertEqual(scene.audio_path, "/path/to/audio.mp3")

    def test_scene_repr(self):
        """Test Scene string representation."""
        scene = Scene(
            id=1,
            text="Test",
            visual_description="Visual",
            search_query="query",
            duration=5.0
        )
        self.assertIn("id=1", repr(scene))
        self.assertIn("Test", repr(scene))


class TestSceneGenerator(unittest.TestCase):
    """Test the SceneGenerator class."""

    def test_scene_generator_missing_api_key(self):
        """Test that SceneGenerator raises ValueError when API key is missing."""
        # Save original env var
        original_key = os.environ.get("GEMINI_API_KEY")
        os.environ.pop("GEMINI_API_KEY", None)

        # Mock config to return empty key
        with patch('services.scene_generator.config') as mock_config:
            mock_config.GEMINI_API_KEY = ""
            with self.assertRaises(ValueError):
                SceneGenerator()

        # Restore original env var
        if original_key:
            os.environ["GEMINI_API_KEY"] = original_key

    def test_extract_json_plain(self):
        """Test _extract_json with plain JSON."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            generator = SceneGenerator(api_key="test_key")

            json_str = '{"scenes": [{"id": 1, "text": "test", "visual_description": "v", "search_query": "q", "duration": 5.0}]}'
            result = generator._extract_json(json_str)
            self.assertIn("scenes", result)
            self.assertEqual(len(result["scenes"]), 1)

    def test_extract_json_with_markdown(self):
        """Test _extract_json with markdown code blocks."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            generator = SceneGenerator(api_key="test_key")

            markdown_str = "```json\n{\"scenes\": [{\"id\": 1, \"text\": \"test\", \"visual_description\": \"v\", \"search_query\": \"q\", \"duration\": 5.0}]}\n```"
            result = generator._extract_json(markdown_str)
            self.assertIn("scenes", result)
            self.assertEqual(result["scenes"][0]["id"], 1)

    def test_extract_json_accepts_safe_python_literal(self):
        """Workers AI occasionally returns a Python-style dict instead of JSON."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            generator = SceneGenerator(api_key="test_key")
            result = generator._extract_json(
                "{'scenes': [{'id': 1, 'text': 'test'}]}")
        self.assertEqual(result["scenes"][0]["id"], 1)

    def test_validate_scenes_valid_count(self):
        """Test validate_scenes with valid scene count."""
        scenes = [
            Scene(id=i, text=f"Text {i}", visual_description=f"Visual {i}",
                  search_query=f"Query {i}", duration=8.0)
            for i in range(1, 7)  # 6 scenes
        ]

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            generator = SceneGenerator(api_key="test_key")
            result = generator.validate_scenes(scenes)
            self.assertTrue(result)

    def test_validate_scenes_too_few(self):
        """Test validate_scenes with too few scenes (should still return True with warning)."""
        scenes = [
            Scene(id=1, text="Text 1", visual_description="Visual 1",
                  search_query="Query 1", duration=8.0)
        ]

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            generator = SceneGenerator(api_key="test_key")
            result = generator.validate_scenes(scenes)
            self.assertTrue(result)  # validate_scenes always returns True

    def test_validate_scenes_duration_warning(self):
        """Test validate_scenes with duration outside range."""
        # Total duration too short
        scenes = [
            Scene(id=i, text=f"Text {i}", visual_description=f"Visual {i}",
                  search_query=f"Query {i}", duration=2.0)
            for i in range(1, 7)
        ]  # 6 * 2 = 12s < 40s min

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            generator = SceneGenerator(api_key="test_key")
            result = generator.validate_scenes(scenes)
            self.assertTrue(result)

    @patch('services.scene_generator.CloudflareCLI.available', return_value=True)
    @patch('services.scene_generator.genai.Client')
    def test_cloudflare_cli_generates_scenes_without_gemini_key(self, mock_client, _available):
        mock_client.assert_not_called()
        with patch('services.scene_generator.config.GEMINI_API_KEY', ''):
            generator = SceneGenerator(api_key="")
        generator.cloudflare_cli.run_ai = MagicMock(return_value=json.dumps({
            "response": json.dumps({
                "scenes": [
                    {"id": 1, "text": "Metin", "visual_description": "Visual",
                     "search_query": "query", "duration": 10.0},
                ]
            })
        }).encode())
        scenes = generator.generate_scenes("Test topic")
        self.assertEqual(len(scenes), 1)
        self.assertEqual(scenes[0].text, "Metin")
        generator.cloudflare_cli.run_ai.assert_called_once()

    @patch('services.scene_generator.genai.Client')
    def test_generate_scenes_success(self, mock_client):
        """Test successful scene generation."""
        # Mock the Gemini response
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "scenes": [
                {"id": 1, "text": "Metin 1", "visual_description": "Visual 1", "search_query": "query 1", "duration": 10.0},
                {"id": 2, "text": "Metin 2", "visual_description": "Visual 2", "search_query": "query 2", "duration": 10.0},
                {"id": 3, "text": "Metin 3", "visual_description": "Visual 3", "search_query": "query 3", "duration": 10.0},
                {"id": 4, "text": "Metin 4", "visual_description": "Visual 4", "search_query": "query 4", "duration": 10.0},
                {"id": 5, "text": "Metin 5", "visual_description": "Visual 5", "search_query": "query 5", "duration": 10.0},
            ]
        })

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = mock_response
        mock_client.return_value = mock_client_instance

        with patch('services.scene_generator.CloudflareCLI.available',
                   return_value=False):
            generator = SceneGenerator(api_key="test_key")
            scenes = generator.generate_scenes("Test topic")

        self.assertEqual(len(scenes), 5)
        self.assertEqual(scenes[0].id, 1)
        self.assertEqual(scenes[0].text, "Metin 1")
        self.assertAlmostEqual(sum(s.duration for s in scenes), 50.0)

    @patch('services.scene_generator.genai.Client')
    def test_generate_scenes_duration_adjustment(self, mock_client):
        """Test that durations are adjusted when outside range."""
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "scenes": [
                {"id": 1, "text": "Metin 1", "visual_description": "Visual 1", "search_query": "query 1", "duration": 2.0},
                {"id": 2, "text": "Metin 2", "visual_description": "Visual 2", "search_query": "query 2", "duration": 2.0},
                {"id": 3, "text": "Metin 3", "visual_description": "Visual 3", "search_query": "query 3", "duration": 2.0},
                {"id": 4, "text": "Metin 4", "visual_description": "Visual 4", "search_query": "query 4", "duration": 2.0},
                {"id": 5, "text": "Metin 5", "visual_description": "Visual 5", "search_query": "query 5", "duration": 2.0},
            ]
        })

        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = mock_response
        mock_client.return_value = mock_client_instance

        with patch('services.scene_generator.CloudflareCLI.available',
                   return_value=False):
            generator = SceneGenerator(api_key="test_key")
            scenes = generator.generate_scenes("Test topic")

        total = sum(s.duration for s in scenes)
        # Should be adjusted to ~50s (target duration)
        self.assertGreater(total, 10.0)  # Original was 10s, should be scaled up


if __name__ == "__main__":
    unittest.main()
