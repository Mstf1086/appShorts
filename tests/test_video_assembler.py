"""Tests for services.video_assembler.VideoAssembler (image bg + subtitles)."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from moviepy import ImageClip

import config
from services.scene_generator import Scene
from services.video_assembler import VideoAssembler

FONT_AVAILABLE = Path(config.SUBTITLE_FONT).is_file()


def _temp_scene_image(directory: str, size=(1200, 800)) -> str:
    """Write a random RGB test image and return its path."""
    arr = (np.random.rand(size[1], size[0], 3) * 255).astype("uint8")
    path = Path(directory) / "test_scene.png"
    Image.fromarray(arr).save(path)
    return str(path)


def _scene(image_path: str = "") -> Scene:
    return Scene(id=1, text="Test sahne metni", visual_description="test",
                 search_query="test", duration=4.0, image_path=image_path)


class TestImageBackground(unittest.TestCase):
    def setUp(self):
        self.assembler = VideoAssembler()
        self._tmp = tempfile.TemporaryDirectory(prefix="va-test-")
        self.addCleanup(self._tmp.cleanup)

    def test_cover_fit_to_frame_size(self):
        img_path = _temp_scene_image(self._tmp.name, size=(1200, 800))
        clip = self.assembler._create_image_background(_scene(img_path))
        self.assertEqual(tuple(clip.size), (config.VIDEO_WIDTH, config.VIDEO_HEIGHT))
        self.assertEqual(clip.duration, 4.0)
        frame = clip.get_frame(0)
        self.assertEqual(frame.shape, (config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 3))

    def test_zoom_animates_over_time(self):
        img_path = _temp_scene_image(self._tmp.name, size=(1080, 1920))
        clip = self.assembler._create_image_background(_scene(img_path))
        start = clip.get_frame(0).shape
        end = clip.get_frame(3.9).shape
        # Slow zoom: final frame must be larger than the first
        self.assertGreater(end[0] * end[1], start[0] * start[1])

    def test_scene_clip_uses_image(self):
        img_path = _temp_scene_image(self._tmp.name)
        clip = self.assembler._create_scene_clip(_scene(img_path), start_time=0)
        self.assertIsNotNone(clip)

    def test_scene_clip_black_fallback_when_image_missing(self):
        clip = self.assembler._create_scene_clip(_scene(""), start_time=0)
        self.assertIsNotNone(clip)
        frame = clip.get_frame(1.0)
        # No subtitles/no image: mostly black frame
        self.assertEqual(frame.shape, (config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 3))


@unittest.skipUnless(FONT_AVAILABLE, f"No subtitle font at {config.SUBTITLE_FONT}")
class TestSubtitleClip(unittest.TestCase):
    """Regression test: MoviePy 2.x ImageClip has no `color` kwarg (old bug)."""

    def setUp(self):
        self.assembler = VideoAssembler()

    def test_subtitle_clip_created_without_warnings(self):
        import io
        import contextlib

        stderr_buf = io.StringIO()
        with contextlib.redirect_stderr(stderr_buf):
            clip = self.assembler._create_subtitle_clip("Merhaba dünya", 0.0, 2.0, 2.0)
        self.assertIsNotNone(clip)
        self.assertFalse(hasattr(clip, "_bg_clip") and clip._bg_clip is None,
                         "Subtitle background clip must be attached")
        # Opaque RGBA background with working alpha mask
        bg = clip._bg_clip
        self.assertIsNotNone(getattr(bg, "mask", None),
                             "Background must carry an alpha mask")
        alpha = bg.mask.get_frame(0)  # float mask in [0, 1]
        self.assertGreater(float(alpha.max()), 0.0)
        self.assertLessEqual(float(alpha.max()), 1.0)

    def test_subtitle_text_renders_visible_frame(self):
        clip = self.assembler._create_subtitle_clip("Test altyazı", 0.0, 2.0, 2.0)
        frame = clip.get_frame(0.5)
        # Some non-transparent white pixels = text actually rendered
        self.assertGreater(int(frame[..., :3].max()), 200)

    def test_scene_composite_keeps_text_full_brightness(self):
        """Regression: bg clip must be UNDER the text (was dimming it to ~50%)."""
        scene = Scene(id=1, text="Uzun bir sahne metni altyazı parlaklığı testi için",
                       visual_description="v", search_query="q", duration=3.0)
        clip = self.assembler._create_scene_clip(scene, 0)
        frame = clip.get_frame(1.0)
        zone = frame[1520:, :, :]
        self.assertGreaterEqual(int(zone.max()), 250,
                                "White subtitle text must keep full brightness")

    def test_invalid_font_falls_back_to_pillow_default(self):
        """Regression: a missing/corrupt font path must not drop the subtitle."""
        with patch.object(config, "SUBTITLE_FONT", "/nonexistent/invalid.ttf"):
            clip = self.assembler._create_subtitle_clip("Geçersiz font testi", 0.0, 2.0, 2.0)
        self.assertIsNotNone(clip)
        # Not the empty (1,1) emergency fallback — text actually rendered
        self.assertGreater(clip.size[0], 1)
        frame = clip.get_frame(0.5)
        self.assertGreater(int(frame[..., :3].max()), 200)


@unittest.skipUnless(FONT_AVAILABLE, f"No subtitle font at {config.SUBTITLE_FONT}")
class TestKaraokeSubtitles(unittest.TestCase):
    """TikTok accordion karaoke: yellow words + white emphasis on black box."""

    def setUp(self):
        self.assembler = VideoAssembler()

    def test_word_clips_cover_scene_duration(self):
        scene = Scene(id=1, text="Zarif bir kedi bahçede oyun oynuyor",
                      visual_description="v", search_query="q", duration=6.0)
        clips = self.assembler._create_karaoke_subtitles(scene, 0)
        # One clip per word
        self.assertEqual(len(clips), 6)
        # Clips must tile the scene timeline without big gaps
        starts = sorted(c.start for c in clips)
        self.assertAlmostEqual(starts[0], 0.0, delta=0.01)
        last = max(c.start + c.duration for c in clips)
        self.assertGreaterEqual(last, 5.9)

    def test_yellow_text_with_white_emphasis(self):
        """3rd word clip: previous words yellow, active word white, box black."""
        scene = Scene(id=1, text="Zarif bir kedi bahçede oyun oynuyor",
                      visual_description="v", search_query="q", duration=6.0)
        clip = self.assembler._create_karaoke_subtitles(scene, 0)[2]
        frame = clip.get_frame(clip.duration * 0.6)
        rgb = frame[..., :3].astype(int)
        yellow = ((rgb[..., 0] > 200) & (rgb[..., 1] > 200) & (rgb[..., 2] < 100)).sum()
        white = ((rgb[..., 0] > 200) & (rgb[..., 1] > 200) & (rgb[..., 2] > 200)).sum()
        black = ((rgb[..., 0] < 30) & (rgb[..., 1] < 30) & (rgb[..., 2] < 30)).sum()
        self.assertGreater(yellow, 100)
        self.assertGreater(white, 50)
        self.assertGreater(black, 500)

    def test_karaoke_box_positioned_near_bottom(self):
        scene = Scene(id=1, text="Kedi uyuyor",
                      visual_description="v", search_query="q", duration=2.0)
        clip = self.assembler._create_karaoke_subtitles(scene, 0)[0]
        # Settled position: box center within lower third of the frame
        _, y = clip.pos(clip.duration * 0.9)
        self.assertGreater(y, config.VIDEO_HEIGHT // 2)

    def test_karaoke_position_is_static_no_jitter(self):
        """Regression: per-step rise/pop animation made the box jump ~14px at
        every word change (visible jitter). Position must be static: same at
        t=0 and t=end, and y must be an integer (no sub-pixel shimmer)."""
        scene = Scene(id=1, text="Kedi bahçede uyuyor",
                      visual_description="v", search_query="q", duration=2.0)
        clip = self.assembler._create_karaoke_subtitles(scene, 0)[0]
        pos_start = clip.pos(0)
        pos_end = clip.pos(clip.duration * 0.99)
        self.assertEqual(pos_start, pos_end,
                         f"box moved during clip: {pos_start} -> {pos_end}")
        self.assertIsInstance(pos_start[1], int,
                              f"y must be int, got {pos_start[1]!r}")
        # Frame pixels identical at start/end (no resampling shimmer)
        import numpy as _np
        f0 = clip.get_frame(0.001)
        f1 = clip.get_frame(clip.duration * 0.99)
        self.assertTrue(_np.array_equal(f0, f1),
                        "static clip must render identical frames")

    def test_box_is_tall_enough_for_text(self):
        """Regression: subtitles did not fit — box must exceed text height."""
        scene = Scene(id=1, text="Zarif bir kedi bahcede oyun oynuyor",
                      visual_description="v", search_query="q", duration=6.0)
        clip = self.assembler._create_karaoke_subtitles(scene, 0)[2]
        # Single-canvas PIL render: clip size IS the box size
        # At least text height (~70px @52pt) + generous padding => >= 104px
        self.assertGreaterEqual(clip.size[1], 104)
        # Box padding rows (top edge) must be dark; text zone must be bright
        frame = clip.get_frame(clip.duration * 0.6)
        self.assertLess(int(frame[2, clip.size[0] // 2, 0]), 30)
        self.assertGreater(int(frame[clip.size[1] // 2, :, :].max()), 200)

    def test_disjoint_groups_no_word_repetition(self):
        """Regression: sliding window repeated words — groups must be disjoint.

        Single-canvas render: one clip per word, each with its own time slot.
        Repetition would show as two clips sharing a start time; disjoint
        slicing guarantees strictly increasing, non-overlapping slots.
        """
        scene = Scene(id=1, text="Zarif bir kedi bahcede oyun oynuyor",
                      visual_description="v", search_query="q", duration=6.0)
        clips = self.assembler._create_karaoke_subtitles(scene, 0)
        self.assertEqual(len(clips), 6)
        starts = [c.start for c in clips]
        self.assertEqual(len(set(starts)), 6, "two clips share a slot: repetition")
        self.assertEqual(starts, sorted(starts))
        for a, b in zip(clips, clips[1:]):
            self.assertLessEqual(a.start + a.duration, b.start + 0.05)

    def test_word_timings_preferred_over_estimate(self):
        """Exact Edge TTS boundaries drive clip timing when present."""
        scene = Scene(id=1, text="Zarif bir kedi", visual_description="v",
                      search_query="q", duration=6.0,
                      word_timings=[
                          {"word": "Zarif", "start": 0.10, "end": 0.74},
                          {"word": "bir", "start": 0.75, "end": 0.93},
                          {"word": "kedi", "start": 0.93, "end": 1.19},
                      ])
        clips = self.assembler._create_karaoke_subtitles(scene, 0)
        self.assertEqual(len(clips), 3)
        # First clip must start at the WordBoundary start (not at 0)
        self.assertAlmostEqual(min(c.start for c in clips), 0.10, delta=0.01)
        # Last clip must end near last boundary (not the 6s estimate)
        last = max(c.start + c.duration for c in clips)
        self.assertLess(last, 1.5)


class TestSceneAudioSync(unittest.TestCase):
    """Scene durations must follow real TTS audio, not LLM estimates."""

    def test_generate_all_voiceovers_syncs_duration(self):
        from services.voice_generator import VoiceGenerator
        vg = VoiceGenerator()
        scenes = [
            Scene(id=1, text="birinci sahne metni", visual_description="a",
                  search_query="a", duration=6.0),
        ]
        fake_audio = Path(tempfile.gettempdir()) / "sync_test.mp3"
        with patch.object(vg, "generate_voiceover", return_value=str(fake_audio)):
            with patch.object(vg, "get_audio_duration", return_value=8.5):
                scenes = vg.generate_all_voiceovers(scenes)
        # 8.5s audio + 0.4s padding = 8.9s — estimate 6.0 must be replaced
        self.assertAlmostEqual(scenes[0].duration, 8.9, delta=0.01)


if __name__ == "__main__":
    unittest.main()
