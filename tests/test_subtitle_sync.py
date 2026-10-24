"""Tests for SubtitleSync service."""
import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
from pathlib import Path

# Ensure project root is in path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock google.genai before importing scene_generator
sys.modules['google'] = MagicMock()
sys.modules['google.genai'] = MagicMock()
sys.modules['google.genai.types'] = MagicMock()

import config
from services.scene_generator import Scene
from services.subtitle_sync import SubtitleSync


class TestSubtitleSync(unittest.TestCase):
    """Test the SubtitleSync class."""

    def setUp(self):
        """Set up test fixtures."""
        self.sync = SubtitleSync()
        self.test_scenes = [
            Scene(
                id=1,
                text="Yapay zeka, makinelerin insan benzeri zekâ göstermesini sağlayan bir teknolojidir.",
                visual_description="AI technology",
                search_query="artificial intelligence",
                duration=8.0,
                audio_path=""
            ),
            Scene(
                id=2,
                text="Bu teknoloji, bilgisayarların öğrenme ve karar verme yeteneklerini geliştirir.",
                visual_description="Machine learning",
                search_query="machine learning",
                duration=7.0,
                audio_path=""
            )
        ]

    def test_split_text_short(self):
        """Test split_text with short text that fits in one line."""
        result = self.sync.split_text("Kısa metin")
        self.assertEqual(result, ["Kısa metin"])

    def test_split_text_long(self):
        """Test split_text with long text that needs multiple lines."""
        long_text = "Bu çok uzun bir metin birden fazla satıra bölünmelidir çünkü karakter sınırı aşılmıştır"
        result = self.sync.split_text(long_text, max_chars=30)
        self.assertGreater(len(result), 1)
        for line in result:
            self.assertLessEqual(len(line), 30)

    def test_split_text_exact_max_chars(self):
        """Test split_text with text exactly at max_chars."""
        text = "1234567890" * 4  # 40 chars
        result = self.sync.split_text(text, max_chars=40)
        self.assertEqual(result, [text])

    def test_format_time_zero(self):
        """Test format_time with zero seconds."""
        result = self.sync.format_time(0)
        self.assertEqual(result, "00:00:00,000")

    def test_format_time_hours(self):
        """Test format_time with hours."""
        result = self.sync.format_time(3661.5)  # 1:01:01,500
        self.assertEqual(result, "01:01:01,500")

    def test_format_time_minutes(self):
        """Test format_time with minutes."""
        result = self.sync.format_time(90.5)  # 00:01:30,500
        self.assertEqual(result, "00:01:30,500")

    def test_format_time_milliseconds(self):
        """Test format_time with milliseconds."""
        result = self.sync.format_time(1.123)
        self.assertEqual(result, "00:00:01,123")

    def test_calculate_subtitle_timing_no_audio(self):
        """Test subtitle timing calculation without audio files."""
        subtitles = self.sync.calculate_subtitle_timing(self.test_scenes)

        self.assertGreater(len(subtitles), 0)
        self.assertEqual(subtitles[0]["index"], 1)
        self.assertEqual(subtitles[0]["start"], 0.0)

        # Check that all scenes are covered
        scene_ids = [sub["scene_id"] for sub in subtitles]
        self.assertIn(1, scene_ids)
        self.assertIn(2, scene_ids)

    def test_calculate_subtitle_timing_with_audio(self):
        """Test subtitle timing calculation with mocked audio."""
        scenes_with_audio = [
            Scene(
                id=1,
                text="Test metin",
                visual_description="Visual",
                search_query="query",
                duration=5.0,
                audio_path="/fake/audio.mp3"
            )
        ]

        with patch.object(SubtitleSync, 'calculate_subtitle_timing') as mock_timing:
            mock_timing.return_value = [
                {"index": 1, "start": 0.0, "end": 5.0, "text": "Test metin", "scene_id": 1}
            ]
            subtitles = self.sync.calculate_subtitle_timing(scenes_with_audio)
            self.assertEqual(len(subtitles), 1)

    def test_generate_srt(self):
        """Test SRT file generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_subtitles.srt")

            srt_path = self.sync.generate_srt(self.test_scenes, output_path=output_path)

            self.assertEqual(srt_path, output_path)
            self.assertTrue(Path(output_path).exists())

            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check SRT format
            self.assertIn("--> ", content)
            self.assertIn("Yapay zeka", content)
            self.assertIn("1", content)

    def test_generate_srt_default_path(self):
        """Test SRT file generation with default path."""
        srt_path = self.sync.generate_srt(self.test_scenes)

        self.assertTrue(Path(srt_path).exists())
        self.assertTrue(srt_path.endswith(".srt"))

        # Clean up
        Path(srt_path).unlink()

    def test_get_subtitle_entries(self):
        """Test getting subtitle entries as tuples."""
        entries = self.sync.get_subtitle_entries(self.test_scenes)

        self.assertGreater(len(entries), 0)
        for entry in entries:
            self.assertEqual(len(entry), 3)
            start, end, text = entry
            self.assertGreaterEqual(end, start)
            self.assertIsInstance(text, str)

    def test_split_text_preserves_words(self):
        """Test that split_text doesn't split words in the middle."""
        text = "Kısa uzun kelimeler doğru bölünmeli"
        result = self.sync.split_text(text, max_chars=15)

        # Check that no word is split
        all_words = text.split()
        for line in result:
            for word in line.split():
                self.assertIn(word, all_words)


class TestSubtitleAudioAlignment(unittest.TestCase):
    """Subtitle timings must follow the spoken words, not character estimates."""

    def setUp(self):
        self.sync = SubtitleSync()
        self.words = [
            {"word": "Yapay", "start": 0.10, "end": 0.70},
            {"word": "zeka", "start": 0.70, "end": 1.05},
            {"word": "hayatımızın", "start": 1.05, "end": 1.78},
            {"word": "her", "start": 1.78, "end": 2.04},
            {"word": "yerinde", "start": 2.04, "end": 2.53},
        ]

    def _scene(self, scene_id=1, duration=6.0, words=True):
        return Scene(
            id=scene_id,
            text="Yapay zeka hayatımızın her yerinde",
            visual_description="v",
            search_query="q",
            duration=duration,
            word_timings=list(self.words) if words else [],
        )

    def test_line_boundaries_follow_word_timings(self):
        """Subtitle must start/end exactly when its words are spoken."""
        subs = self.sync.calculate_subtitle_timing([self._scene()])
        self.assertTrue(subs)
        first = subs[0]
        self.assertAlmostEqual(first["start"], 0.10, delta=0.01)
        self.assertAlmostEqual(first["end"], 2.53, delta=0.01)
        self.assertEqual(first["text"], "Yapay zeka hayatımızın her yerinde")

    def test_scene_offsets_account_for_crossfade(self):
        """Scene starts must mirror the assembler's crossfade timeline."""
        scenes = [self._scene(1, duration=5.0), self._scene(2, duration=5.0)]
        with patch.object(config, "SCENE_TRANSITION", 0.4):
            starts = self.sync.scene_start_times(scenes)
        self.assertAlmostEqual(starts[0], 0.0, delta=0.001)
        self.assertAlmostEqual(starts[1], 4.6, delta=0.001)

    def test_second_scene_subtitle_is_shifted_by_scene_start(self):
        """The second scene's first word must be timed on the global timeline."""
        scenes = [self._scene(1, duration=5.0), self._scene(2, duration=5.0)]
        with patch.object(config, "SCENE_TRANSITION", 0.4):
            subs = self.sync.calculate_subtitle_timing(scenes)
        second = [s for s in subs if s["scene_id"] == 2]
        self.assertTrue(second)
        self.assertAlmostEqual(second[0]["start"], 4.7, delta=0.01)

    def test_fallback_without_word_timings(self):
        """Without boundaries the speech duration is split proportionally."""
        subs = self.sync.calculate_subtitle_timing([self._scene(words=False)])
        self.assertTrue(subs)
        self.assertAlmostEqual(subs[0]["start"], 0.0, delta=0.01)
        self.assertAlmostEqual(subs[0]["end"], 6.0, delta=0.01)

    def test_srt_uses_word_timings(self):
        """The written SRT must contain the word-accurate timestamps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "aligned.srt")
            self.sync.generate_srt([self._scene()], output_path=output_path)
            content = Path(output_path).read_text(encoding="utf-8")
        self.assertIn("00:00:00,100 --> 00:00:02,530", content)


if __name__ == "__main__":
    unittest.main()
