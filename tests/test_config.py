"""Tests for config module."""
import unittest
import os
from pathlib import Path
from unittest.mock import patch

# Ensure project root is in path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfig(unittest.TestCase):
    """Test the config module settings."""

    def test_video_dimensions(self):
        """Test video dimension settings."""
        import config
        self.assertEqual(config.VIDEO_WIDTH, 1080)
        self.assertEqual(config.VIDEO_HEIGHT, 1920)
        # Verify 9:16 aspect ratio
        self.assertAlmostEqual(config.VIDEO_WIDTH / config.VIDEO_HEIGHT, 9/16, places=2)

    def test_duration_range(self):
        """Test duration range settings."""
        import config
        self.assertGreater(config.MAX_DURATION, config.MIN_DURATION)
        self.assertEqual(config.MIN_DURATION, 40)
        self.assertEqual(config.MAX_DURATION, 60)

    def test_scene_count_range(self):
        """Test scene count range settings."""
        import config
        self.assertGreater(config.MAX_SCENES, config.MIN_SCENES)
        self.assertEqual(config.MIN_SCENES, 5)
        self.assertEqual(config.MAX_SCENES, 8)

    def test_voice_settings(self):
        """Test voice settings."""
        import config
        self.assertEqual(config.VOICE_LANGUAGE, "tr-TR")
        self.assertEqual(config.VOICE_NAME, "tr-TR-AhmetNeural")
        self.assertEqual(config.TTS_MAX_RETRIES, 5)
        self.assertEqual(config.TTS_RETRY_BASE_DELAY, 3.0)
        self.assertEqual(config.TTS_SCENE_DELAY, 1.5)

    def test_subtitle_settings(self):
        """Test subtitle settings (TikTok karaoke defaults)."""
        import config
        self.assertIsInstance(config.SUBTITLE_FONT, str)
        self.assertGreater(len(config.SUBTITLE_FONT), 0)
        self.assertEqual(config.SUBTITLE_FONT_SIZE, 52)
        self.assertEqual(config.SUBTITLE_COLOR, "yellow")
        self.assertEqual(config.SUBTITLE_EMPHASIS_COLOR, "white")
        self.assertEqual(config.SUBTITLE_STROKE_COLOR, "black")
        self.assertEqual(config.SUBTITLE_STROKE_WIDTH, 4)
        self.assertIsInstance(config.SUBTITLE_POSITION, str)
        self.assertGreater(len(config.SUBTITLE_POSITION), 0)
        self.assertEqual(config.SUBTITLE_STYLE, "karaoke")
        self.assertEqual(config.SUBTITLE_WORDS_PER_GROUP, 3)
        self.assertGreaterEqual(config.SUBTITLE_BG_COLOR[3], 200)  # opaque black box

    def test_transition_and_audio_padding_settings(self):
        """Crossfade transition + audio padding knobs."""
        import config
        self.assertIsInstance(config.SCENE_TRANSITION, float)
        self.assertGreaterEqual(config.SCENE_TRANSITION, 0.0)
        self.assertIsInstance(config.SCENE_AUDIO_PADDING, float)
        self.assertGreater(config.SCENE_AUDIO_PADDING, 0.0)

    def test_directories_exist(self):
        """Test that required directories are created."""
        import config
        self.assertTrue(Path(config.OUTPUT_DIR).exists())
        self.assertTrue(Path(config.TMP_DIR).exists())
        self.assertTrue(Path(config.IMAGES_DIR).exists())
        self.assertTrue(Path(config.AUDIO_DIR).exists())

    def test_tmp_dir_is_project_root_dot_tmp(self):
        """Temporary artifacts must live in the project-root .tmp folder."""
        import config
        self.assertEqual(config.TMP_DIR, config.BASE_DIR / ".tmp")
        self.assertNotEqual(config.TMP_DIR, config.OUTPUT_DIR / "tmp")
        self.assertTrue(config.VIDEO_TMP_DIR.is_dir())
        self.assertTrue(str(config.VIDEO_TMP_DIR).startswith(str(config.TMP_DIR)))
        self.assertTrue(str(config.IMAGES_DIR).startswith(str(config.TMP_DIR)))
        self.assertTrue(str(config.AUDIO_DIR).startswith(str(config.TMP_DIR)))

    def test_gemini_model(self):
        """Test Gemini model setting."""
        import config
        self.assertIsInstance(config.GEMINI_MODEL, str)
        self.assertGreater(len(config.GEMINI_MODEL), 0)

    def test_image_generation_settings(self):
        """Test image generation settings + one-image/no-approval defaults."""
        import config
        self.assertIsInstance(config.GEMINI_IMAGE_MODEL, str)
        self.assertGreater(len(config.GEMINI_IMAGE_MODEL), 0)
        self.assertIsInstance(config.IMAGE_CANDIDATES, int)
        # One image per scene, approval loop off by default (auto pipeline)
        self.assertEqual(config.IMAGE_CANDIDATES, 1)
        self.assertIsInstance(config.APPROVE_IMAGES, bool)
        self.assertFalse(config.APPROVE_IMAGES)

    def test_subtitle_font_is_resolved(self):
        """SUBTITLE_FONT must be a non-empty path resolved by the resolver."""
        import config
        self.assertIsInstance(config.SUBTITLE_FONT, str)
        self.assertTrue(config.SUBTITLE_FONT.endswith((".ttf", ".otf")))
        self.assertEqual(config._resolve_subtitle_font(), config.SUBTITLE_FONT)

    def test_subtitle_font_env_override(self):
        """SUBTITLE_FONT env var wins when the file exists."""
        import tempfile
        import config
        with tempfile.NamedTemporaryFile(suffix=".ttf") as fake_font:
            with patch.dict(os.environ, {"SUBTITLE_FONT": fake_font.name}):
                self.assertEqual(config._resolve_subtitle_font(), fake_font.name)

    def test_subtitle_font_fallback_when_missing(self):
        """With no candidate on disk the resolver returns "" (never a fake path)."""
        import config
        with patch.dict(os.environ, {"SUBTITLE_FONT": ""}):
            with patch.object(config.Path, "is_file", return_value=False):
                with patch.dict('sys.modules', {'matplotlib': None, 'matplotlib.font_manager': None}):
                    resolved = config._resolve_subtitle_font()
        self.assertEqual(resolved, "")

    def test_resolver_never_returns_missing_path(self):
        """Regression: resolver must not hand Pillow a nonexistent path."""
        import config
        resolved = config._resolve_subtitle_font()
        self.assertTrue(resolved == "" or Path(resolved).is_file())

    def test_bundled_font_exists_and_loads_turkish(self):
        """Repo bundles NotoSans-Bold so font-less hosts still render subtitles."""
        import config
        from PIL import ImageFont
        self.assertTrue(config.BUNDLED_FONT.is_file())
        font = ImageFont.truetype(str(config.BUNDLED_FONT), 20)
        left, top, right, bottom = font.getbbox("Türkçe ÇĞİŞÖÜ")
        self.assertGreater(right - left, 0)
        self.assertGreater(bottom - top, 0)

    def test_get_subtitle_font_returns_resolved_path(self):
        import config
        self.assertEqual(config.get_subtitle_font(), config.SUBTITLE_FONT)

    def test_get_subtitle_font_downloads_once_when_missing(self):
        """When nothing is resolvable, first use downloads a font (then caches)."""
        import tempfile
        import config
        def fake_retrieve(url, dest):
            Path(dest).write_bytes(b"0" * 200_000)
        with tempfile.TemporaryDirectory(prefix="font-dl-") as tmp:
            with patch.object(config, "TMP_DIR", Path(tmp)):
                with patch.object(config, "_subtitle_font_cache", None):
                    with patch("config.urllib.request.urlretrieve",
                               side_effect=fake_retrieve) as mock_dl:
                        font = config.get_subtitle_font()
                        self.assertTrue(font.endswith("NotoSans-Bold.ttf"))
                        self.assertTrue(Path(font).is_file())
                        # Second call must hit the cache, not the network
                        self.assertEqual(config.get_subtitle_font(), font)
                        self.assertEqual(mock_dl.call_count, 1)

    def test_pollinations_fallback_settings(self):
        """Keyless fallback knobs (enabled by default, 90s timeout)."""
        import config
        self.assertIsInstance(config.IMAGE_FALLBACK_POLLINATIONS, bool)
        self.assertTrue(config.IMAGE_FALLBACK_POLLINATIONS)
        self.assertEqual(config.POLLINATIONS_TIMEOUT, 90)


    def test_image_provider_defaults_to_cloudflare(self):
        """Scene images come from Cloudflare Workers AI by default."""
        import config
        self.assertEqual(config.IMAGE_PROVIDER, "cloudflare")

    def test_cloudflare_fallback_settings(self):
        """Token-based Workers AI fallback knobs (flag on, 90s timeout)."""
        import config
        self.assertIsInstance(config.IMAGE_FALLBACK_CLOUDFLARE, bool)
        self.assertTrue(config.IMAGE_FALLBACK_CLOUDFLARE)
        self.assertIsInstance(config.CLOUDFLARE_IMAGE_MODEL, str)
        self.assertTrue(len(config.CLOUDFLARE_IMAGE_MODEL) > 0)
        self.assertIsInstance(config.CLOUDFLARE_ACCOUNT_ID, str)
        self.assertIsInstance(config.CLOUDFLARE_API_TOKEN, str)
        self.assertIsInstance(config.CLOUDFLARE_TIMEOUT, int)


if __name__ == "__main__":
    unittest.main()
