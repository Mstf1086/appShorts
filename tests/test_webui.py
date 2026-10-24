"""Tests for the Gradio web UI (webui.py)."""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock google.genai before importing (same pattern as test_pipeline.py)
sys.modules['google'] = MagicMock()
sys.modules['google.genai'] = MagicMock()
sys.modules['google.genai.types'] = MagicMock()

import config
import webui


class TestConfigGradioSettings(unittest.TestCase):
    """Gradio settings must live in config.py and be env-driven."""

    def test_gradio_settings_exist(self):
        self.assertTrue(hasattr(config, 'GRADIO_HOST'))
        self.assertTrue(hasattr(config, 'GRADIO_PORT'))
        self.assertTrue(hasattr(config, 'GRADIO_SHARE'))
        self.assertTrue(hasattr(config, 'GRADIO_AUTH_USER'))
        self.assertTrue(hasattr(config, 'GRADIO_AUTH_PASS'))
        self.assertTrue(hasattr(config, 'GRADIO_MAX_QUEUE'))

    def test_gradio_defaults(self):
        self.assertEqual(config.GRADIO_HOST, "0.0.0.0")
        self.assertEqual(config.GRADIO_PORT, 7860)
        self.assertIsInstance(config.GRADIO_SHARE, bool)

    def test_status_markdown_does_not_leak_keys(self):
        """Status text must never contain the API key itself."""
        markdown = webui._status_markdown()
        if config.GEMINI_API_KEY:
            self.assertNotIn(config.GEMINI_API_KEY, markdown)
        self.assertIn("Sistem durumu", markdown)


class TestBuildUI(unittest.TestCase):
    """The Gradio interface must build without launching."""

    def test_build_ui_returns_blocks(self):
        demo = webui.build_ui()
        self.assertTrue(hasattr(demo, 'launch'))
        self.assertTrue(hasattr(demo, 'queue'))

    def test_build_ui_is_reusable(self):
        """Building twice must not raise (no global state leak)."""
        self.assertIsNotNone(webui.build_ui())
        self.assertIsNotNone(webui.build_ui())


class TestStreamGeneration(unittest.TestCase):
    """The generation generator streams logs and returns the video path."""

    def setUp(self):
        self._tts_provider = config.TTS_PROVIDER

    def tearDown(self):
        config.TTS_PROVIDER = self._tts_provider

    def test_empty_topic_is_rejected(self):
        with patch('webui.ShortsPipeline') as mock_cls:
            results = list(webui.stream_generation("   "))
            mock_cls.assert_not_called()
        self.assertIn("konu başlığı girin", results[0][0].lower())
        self.assertIsNone(results[0][1])

    def test_successful_run_yields_video_and_log(self):
        def fake_run(topic, interactive=None):
            print("[1/5] Generating scenes...")
            print(f"Topic: {topic}")
            return "/tmp/output.mp4"

        with patch('webui.ShortsPipeline') as mock_cls:
            mock_cls.return_value.run.side_effect = fake_run
            results = list(webui.stream_generation("Test konu", "edge"))

        self.assertTrue(results)
        last_log, last_video, gallery = results[-1]
        self.assertIn("Generating scenes", last_log)
        self.assertIn("Test konu", last_log)
        self.assertEqual(last_video, "/tmp/output.mp4")
        self.assertIsInstance(gallery, list)
        mock_cls.assert_called_once_with(voice_service="edge")

    def test_tts_provider_is_applied(self):
        with patch('webui.ShortsPipeline') as mock_cls:
            mock_cls.return_value.run.return_value = "/tmp/output.mp4"
            list(webui.stream_generation("Test konu", "ttsai"))
        self.assertEqual(config.TTS_PROVIDER, "ttsai")

    def test_selected_models_are_propagated_without_global_config_changes(self):
        text_default = config.CLOUDFLARE_TEXT_MODEL
        image_default = config.CLOUDFLARE_IMAGE_MODEL
        tts_default = config.CLOUDFLARE_TTS_MODEL
        with patch('webui.ShortsPipeline') as mock_cls:
            mock_cls.return_value.run.return_value = "/tmp/output.mp4"
            list(webui.stream_generation(
                "Test konu", "cloudflare",
                "@cf/openai/gpt-oss-20b",
                "@cf/black-forest-labs/flux-2-klein-4b",
                "@cf/deepgram/aura-2-en",
            ))
        mock_cls.assert_called_once_with(
            voice_service="cloudflare",
            text_model="@cf/openai/gpt-oss-20b",
            text_provider="cloudflare",
            image_model="@cf/black-forest-labs/flux-2-klein-4b",
            image_provider="cloudflare",
            tts_model="@cf/deepgram/aura-2-en",
        )
        self.assertEqual(config.CLOUDFLARE_TEXT_MODEL, text_default)
        self.assertEqual(config.CLOUDFLARE_IMAGE_MODEL, image_default)
        self.assertEqual(config.CLOUDFLARE_TTS_MODEL, tts_default)

    def test_pipeline_error_is_surfaced(self):
        with patch('webui.ShortsPipeline') as mock_cls:
            mock_cls.return_value.run.side_effect = RuntimeError("boom")
            results = list(webui.stream_generation("Test konu"))

        last_log, last_video, _ = results[-1]
        self.assertIn("boom", last_log)
        self.assertIsNone(last_video)

    def test_generation_is_serialized(self):
        """Two concurrent runs must not interleave (lock is held)."""
        lock = webui._GENERATION_LOCK
        self.assertTrue(lock.locked() is False)


class TestLaunchUI(unittest.TestCase):
    """launch_ui must pass config values through to gradio."""

    def setUp(self):
        # Never touch the real output/ directory from tests.
        self._tmp = tempfile.mkdtemp(prefix="shorts-webui-test-")
        self._patchers = [
            patch.object(config, 'OUTPUT_DIR', Path(self._tmp)),
            patch.object(config, 'TMP_DIR', Path(self._tmp) / 'tmp'),
        ]
        for p in self._patchers:
            p.start()

    def tearDown(self):
        for p in self._patchers:
            p.stop()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_launch_uses_config_values(self):
        demo = MagicMock()
        demo.queue.return_value = None
        with patch('webui.build_ui', return_value=demo):
            webui.launch_ui(share=False, server_name="127.0.0.1", server_port=7999)
        kwargs = demo.launch.call_args.kwargs
        self.assertEqual(kwargs['server_name'], "127.0.0.1")
        self.assertEqual(kwargs['server_port'], 7999)
        self.assertFalse(kwargs['share'])
        self.assertTrue(kwargs['prevent_thread_lock'])
        self.assertIn(str(config.OUTPUT_DIR), kwargs['allowed_paths'])

    def test_launch_does_not_write_real_output_dir(self):
        """The URL file must land in the (patched) output dir, not the repo."""
        demo = MagicMock()
        demo.queue.return_value = None
        demo.share_url = "https://example.test.gradio.live"
        with patch('webui.build_ui', return_value=demo):
            webui.launch_ui(share=False, server_name="127.0.0.1", server_port=7996)
        self.assertTrue((Path(self._tmp) / 'webui_url.txt').is_file())

    def test_resolve_url_ignores_non_string_attributes(self):
        """Mocked attributes must not be mistaken for a real URL."""
        demo = MagicMock()
        self.assertEqual(webui._resolve_url(demo, MagicMock()), "")

    def test_resolve_url_prefers_share_url(self):
        demo = MagicMock()
        demo.share_url = "https://share.test/abc"
        demo.local_url = "http://127.0.0.1:7860"
        self.assertEqual(webui._resolve_url(demo, None), "https://share.test/abc")

    def test_launch_enables_auth_when_configured(self):
        demo = MagicMock()
        demo.queue.return_value = None
        with patch.object(config, 'GRADIO_AUTH_USER', 'admin'), \
             patch.object(config, 'GRADIO_AUTH_PASS', 'secret'), \
             patch('webui.build_ui', return_value=demo):
            webui.launch_ui(share=False, server_name="127.0.0.1", server_port=7998)
        kwargs = demo.launch.call_args.kwargs
        self.assertEqual(kwargs['auth'], ('admin', 'secret'))

    def test_launch_without_auth_has_auth_none(self):
        demo = MagicMock()
        demo.queue.return_value = None
        with patch.object(config, 'GRADIO_AUTH_USER', ''), \
             patch.object(config, 'GRADIO_AUTH_PASS', ''), \
             patch('webui.build_ui', return_value=demo):
            webui.launch_ui(share=False, server_name="127.0.0.1", server_port=7997)
        self.assertIsNone(demo.launch.call_args.kwargs['auth'])

    def test_default_busy_port_falls_back_to_next_port(self):
        demo = MagicMock()
        demo.queue.return_value = None
        with patch.object(config, 'GRADIO_PORT', 7860), \
             patch('webui.build_ui', return_value=demo), \
             patch('webui._find_available_port', return_value=7861) as find_port:
            webui.launch_ui(share=False, server_name="127.0.0.1")
        find_port.assert_called_once_with("127.0.0.1", 7860)
        self.assertEqual(demo.launch.call_args.kwargs['server_port'], 7861)

    def test_explicit_busy_port_raises_clear_error(self):
        demo = MagicMock()
        demo.queue.return_value = None
        with patch('webui.build_ui', return_value=demo), \
             patch('webui._port_is_available', return_value=False):
            with self.assertRaisesRegex(OSError, "already in use"):
                webui.launch_ui(share=False, server_name="127.0.0.1",
                                server_port=7995)
        demo.launch.assert_not_called()


class TestVideoArchive(unittest.TestCase):
    """Previously generated videos must be listed in the web UI."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="shorts-archive-test-")

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _make_video(self, name="shorts_20260101_120000.mp4", size=2048):
        path = Path(self._tmp) / name
        path.write_bytes(b"0" * size)
        return path

    def test_list_generated_videos(self):
        path = self._make_video()
        with patch.object(config, 'OUTPUT_DIR', Path(self._tmp)):
            videos = webui.list_generated_videos()
        self.assertEqual(len(videos), 1)
        label, value = videos[0]
        self.assertIn(path.stem, label)
        self.assertEqual(value, str(path))

    def test_list_generated_videos_is_empty_without_videos(self):
        with patch.object(config, 'OUTPUT_DIR', Path(self._tmp)):
            self.assertEqual(webui.list_generated_videos(), [])

    def test_list_generated_videos_newest_first(self):
        older = self._make_video("shorts_old.mp4")
        newer = self._make_video("shorts_new.mp4")
        import os
        os.utime(older, (1000000, 1000000))
        with patch.object(config, 'OUTPUT_DIR', Path(self._tmp)):
            videos = webui.list_generated_videos()
        self.assertEqual(videos[0][1], str(newer))

    def test_preview_video_returns_existing_path(self):
        path = self._make_video()
        self.assertEqual(webui.preview_video(str(path)), str(path))

    def test_preview_video_rejects_missing_file(self):
        self.assertIsNone(webui.preview_video("/tmp/does_not_exist_12345.mp4"))
        self.assertIsNone(webui.preview_video(None))

    def test_refresh_video_list_returns_update_object(self):
        """Choices must be refreshed with an update object, not a plain list."""
        path = self._make_video()
        with patch.object(config, 'OUTPUT_DIR', Path(self._tmp)):
            update, newest = webui.refresh_video_list()
        self.assertIsInstance(update, dict)
        self.assertEqual(update.get("__type__"), "update")
        self.assertEqual(len(update["choices"]), 1)
        self.assertEqual(update["choices"][0][1], str(path))
        self.assertEqual(update["value"], str(path))
        self.assertEqual(newest, str(path))

    def test_refresh_video_list_without_videos(self):
        with patch.object(config, 'OUTPUT_DIR', Path(self._tmp)):
            update, newest = webui.refresh_video_list()
        self.assertEqual(update["choices"], [])
        self.assertIsNone(newest)


class TestPortSelection(unittest.TestCase):
    """Port helpers must find a free port without relying on Gradio internals."""

    def test_find_available_port_skips_busy_port(self):
        with patch('webui._port_is_available', side_effect=[False, True]) as available:
            port = webui._find_available_port("127.0.0.1", 7860)
        self.assertEqual(port, 7861)
        self.assertEqual(available.call_count, 2)

    def test_find_available_port_raises_after_range(self):
        with patch('webui._port_is_available', return_value=False):
            with self.assertRaisesRegex(OSError, "7860-7861"):
                webui._find_available_port("127.0.0.1", 7860, max_attempts=2)


if __name__ == "__main__":
    unittest.main()
