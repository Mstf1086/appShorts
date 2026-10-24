"""Tests for services.image_generator.ImageGenerator (Gemini images + approval)."""
import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import config
from services.image_generator import ImageGenerator
from services.scene_generator import Scene

# Minimal valid 1x1 PNG
TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6360606060000000050001"
    "15023a0a0000000049454e44ae426082"
)


def _image_response(data: bytes = TINY_PNG, mime: str = "image/png") -> MagicMock:
    part = MagicMock()
    part.inline_data.data = data
    part.inline_data.mime_type = mime
    resp = MagicMock()
    resp.candidates = [MagicMock()]
    resp.candidates[0].content.parts = [part]
    return resp


def _text_response(text: str = "no image here") -> MagicMock:
    part = MagicMock()
    part.inline_data = None
    part.text = text
    resp = MagicMock()
    resp.candidates = [MagicMock()]
    resp.candidates[0].content.parts = [part]
    return resp


def _quota_error() -> Exception:
    return Exception("429 RESOURCE_EXHAUSTED: Quota exceeded for metric")


def _scene() -> Scene:
    return Scene(id=1, text="metin", visual_description="cinematic castle",
                 search_query="medieval castle", duration=6.0)


class TestImageGeneratorInit(unittest.TestCase):
    @patch("services.image_generator.genai.Client")
    @patch("services.image_generator.config")
    def test_missing_api_key_raises_in_gemini_mode(self, mock_config, mock_client):
        mock_config.IMAGE_PROVIDER = "gemini"
        mock_config.GEMINI_API_KEY = ""
        with self.assertRaises(ValueError):
            ImageGenerator()

    @patch("services.image_generator.genai.Client")
    @patch("services.image_generator.config")
    def test_cloudflare_mode_needs_no_gemini_key(self, mock_config, mock_client):
        mock_config.IMAGE_PROVIDER = "cloudflare"
        mock_config.GEMINI_API_KEY = ""
        mock_config.GEMINI_IMAGE_MODEL = "unused"
        mock_config.IMAGES_DIR = Path(tempfile.mkdtemp(prefix="img-gen-"))
        gen = ImageGenerator()
        self.assertEqual(gen.provider, "cloudflare")
        mock_client.assert_not_called()

    @patch("services.image_generator.genai.Client")
    @patch("services.image_generator.config")
    def test_uses_config_model(self, mock_config, mock_client):
        mock_config.IMAGE_PROVIDER = "gemini"
        mock_config.GEMINI_API_KEY = "test_key"
        mock_config.GEMINI_IMAGE_MODEL = "custom-model"
        mock_config.IMAGES_DIR = Path(tempfile.mkdtemp(prefix="img-gen-"))
        gen = ImageGenerator()
        self.assertEqual(gen.model_name, "custom-model")
        mock_client.assert_called_once_with(api_key="test_key")


class TestGenerateCandidates(unittest.TestCase):
    def setUp(self):
        patcher = patch("services.image_generator.genai.Client")
        self.mock_client_cls = patcher.start()
        self.addCleanup(patcher.stop)
        self.mock_client = self.mock_client_cls.return_value
        # These tests isolate the Gemini chain; provider must be "gemini"
        prov_patcher = patch("services.image_generator.config.IMAGE_PROVIDER", "gemini")
        prov_patcher.start()
        self.addCleanup(prov_patcher.stop)
        # Existing tests use tiny synthetic blobs; blank checks have own tests
        blank_patcher = patch.object(ImageGenerator, "_is_blank_image",
                                     return_value=False)
        blank_patcher.start()
        self.addCleanup(blank_patcher.stop)
        # Existing tests isolate Gemini behavior; fallbacks have their own tests
        fb_patcher = patch("services.image_generator.config.IMAGE_FALLBACK_POLLINATIONS", False)
        fb_patcher.start()
        self.addCleanup(fb_patcher.stop)
        cf_patcher = patch("services.image_generator.config.IMAGE_FALLBACK_CLOUDFLARE", False)
        cf_patcher.start()
        self.addCleanup(cf_patcher.stop)
        self.gen = ImageGenerator(api_key="test_key")
        self._tmp = tempfile.TemporaryDirectory(prefix="img-gen-test-")
        self.addCleanup(self._tmp.cleanup)
        self.gen.output_dir = Path(self._tmp.name)

    def test_success_generates_files(self):
        self.mock_client.models.generate_content.return_value = _image_response()
        paths = self.gen.generate_candidates(_scene(), count=2, max_retries=1)
        self.assertEqual(len(paths), 2)
        for p in paths:
            self.assertTrue(Path(p).exists())
            self.assertEqual(Path(p).read_bytes(), TINY_PNG)
            self.assertIn("scene_01_", Path(p).name)

    def test_quota_error_fails_fast(self):
        self.mock_client.models.generate_content.side_effect = _quota_error()
        paths = self.gen.generate_candidates(_scene(), count=3, max_retries=3)
        self.assertEqual(paths, [])
        self.assertTrue(self.gen._quota_limited)
        # Fail-fast: exactly one API call, no retries after quota
        self.assertEqual(self.mock_client.models.generate_content.call_count, 1)
        # Subsequent calls short-circuit without touching the API
        again = self.gen.generate_candidates(_scene(), count=3)
        self.assertEqual(again, [])
        self.assertEqual(self.mock_client.models.generate_content.call_count, 1)

    def test_response_without_image_retries(self):
        self.mock_client.models.generate_content.return_value = _text_response()
        paths = self.gen.generate_candidates(_scene(), count=1, max_retries=2)
        self.assertEqual(paths, [])
        self.assertEqual(self.mock_client.models.generate_content.call_count, 2)

    def test_model_failover_on_quota(self):
        """Comma-separated model list: 429 on m1 must retry with m2."""
        self.gen.model_names = ["m1", "m2"]
        self.mock_client.models.generate_content.side_effect = [
            _quota_error(), _image_response()]
        paths = self.gen.generate_candidates(_scene(), count=1, max_retries=1)
        self.assertEqual(len(paths), 1)
        self.assertFalse(self.gen._quota_limited)
        called = [c.kwargs["model"] for c in
                  self.mock_client.models.generate_content.call_args_list]
        self.assertEqual(called, ["m1", "m2"])

    def test_quota_falls_back_to_pollinations(self):
        """After quota exhaustion the keyless fallback fills the candidates."""
        self.mock_client.models.generate_content.side_effect = _quota_error()
        with patch("config.IMAGE_FALLBACK_POLLINATIONS", True):
            with patch.object(self.gen, "_generate_pollinations",
                              return_value=["p1.jpg", "p2.jpg"]) as mock_fb:
                paths = self.gen.generate_candidates(_scene(), count=3, max_retries=1)
        self.assertEqual(paths, ["p1.jpg", "p2.jpg"])
        self.assertTrue(self.gen._quota_limited)
        self.assertEqual(mock_fb.call_args[0][1], 3)  # asked for all 3

    def test_pollinations_disabled_returns_empty_on_quota(self):
        self.mock_client.models.generate_content.side_effect = _quota_error()
        with patch("services.image_generator.config.IMAGE_FALLBACK_CLOUDFLARE", False):
            paths = self.gen.generate_candidates(_scene(), count=2, max_retries=1)
        self.assertEqual(paths, [])
        self.assertTrue(self.gen._quota_limited)

    def test_pollinations_saves_downloaded_file(self):
        class _FakeResp:
            def __init__(self, data):
                self._d = data
            def read(self):
                return self._d
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False

        fake = _FakeResp(b"\xff\xd8" + b"x" * 8000)  # > 5 KB
        with patch("services.image_generator.urllib.request.urlopen",
                   return_value=fake):
            paths = self.gen._generate_pollinations(_scene(), count=2)
        self.assertEqual(len(paths), 2)
        for p in paths:
            self.assertTrue(Path(p).is_file())
            self.assertEqual(Path(p).read_bytes(), b"\xff\xd8" + b"x" * 8000)

    def test_prompt_contains_scene_hints(self):
        prompt = self.gen.build_prompt(_scene())
        self.assertIn("cinematic castle", prompt)
        self.assertIn("medieval castle", prompt)
        self.assertIn("9:16", prompt)

class TestApproval(unittest.TestCase):
    def setUp(self):
        patcher = patch("services.image_generator.genai.Client")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.gen = ImageGenerator(api_key="test_key")
        open_patcher = patch.object(ImageGenerator, "_open_in_viewer", return_value=True)
        open_patcher.start()
        self.addCleanup(open_patcher.stop)

    def test_non_interactive_auto_approves_first(self):
        result = self.gen.approve_candidates(_scene(), ["a.png", "b.png"], interactive=False)
        self.assertEqual(result, "a.png")

    def test_enter_approves_current(self):
        with patch("builtins.input", return_value=""):
            result = self.gen.approve_candidates(_scene(), ["a.png", "b.png"], interactive=True)
        self.assertEqual(result, "a.png")

    def test_change_moves_to_next_candidate(self):
        with patch("builtins.input", side_effect=["d", ""]):
            result = self.gen.approve_candidates(_scene(), ["a.png", "b.png"], interactive=True)
        self.assertEqual(result, "b.png")

    def test_skip_returns_none(self):
        with patch("builtins.input", return_value="s"):
            result = self.gen.approve_candidates(_scene(), ["a.png"], interactive=True)
        self.assertIsNone(result)

    def test_quit_raises_keyboard_interrupt(self):
        with patch("builtins.input", return_value="q"):
            with self.assertRaises(KeyboardInterrupt):
                self.gen.approve_candidates(_scene(), ["a.png"], interactive=True)

    def test_change_exhausted_regenerates_batch(self):
        scene = _scene()
        with patch.object(self.gen, "generate_candidates",
                          return_value=["fresh1.png", "fresh2.png"]) as mock_gen:
            # d = change beyond single candidate -> new batch; Enter approves fresh1
            with patch("builtins.input", side_effect=["d", ""]):
                result = self.gen.approve_candidates(scene, ["only.png"], interactive=True)
        self.assertEqual(result, "fresh1.png")
        self.assertEqual(scene.image_candidates, ["fresh1.png", "fresh2.png"])
        mock_gen.assert_called_once()

    def test_no_candidates_returns_none(self):
        self.assertIsNone(self.gen.approve_candidates(_scene(), [], interactive=True))


class TestGenerateAllImages(unittest.TestCase):
    def setUp(self):
        patcher = patch("services.image_generator.genai.Client")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.gen = ImageGenerator(api_key="test_key")
        self._tmp = tempfile.TemporaryDirectory(prefix="img-gen-all-")
        self.addCleanup(self._tmp.cleanup)
        self.gen.output_dir = Path(self._tmp.name)

    def test_auto_approval_sets_image_path(self):
        with patch.object(self.gen, "generate_candidates", return_value=["x.png"]):
            scenes = self.gen.generate_all_images([_scene()], interactive=False)
        self.assertEqual(scenes[0].image_path, "x.png")
        self.assertEqual(scenes[0].image_candidates, ["x.png"])

    def test_no_candidates_falls_back_to_gradient(self):
        """Every provider failed → colored procedural bg, never plain black."""
        from PIL import Image
        with patch.object(self.gen, "generate_candidates", return_value=[]):
            scenes = self.gen.generate_all_images([_scene()], interactive=False)
        path = scenes[0].image_path
        self.assertTrue(path.endswith("_gradient.png"))
        self.assertTrue(Path(path).is_file())
        self.assertEqual(scenes[0].image_candidates, [])
        with Image.open(path) as im:
            self.assertEqual(im.size, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))

    def test_gradient_is_deterministic_per_topic(self):
        p1 = self.gen._generate_gradient(_scene())
        p2 = self.gen._generate_gradient(_scene())
        self.assertEqual(Path(p1).read_bytes(), Path(p2).read_bytes())

    def test_skip_in_interactive_mode(self):
        """Explicit user skip (s) must stay black — gradient is only a fallback."""
        with patch.object(self.gen, "generate_candidates", return_value=["x.png"]):
            with patch.object(self.gen, "_open_in_viewer", return_value=True):
                with patch("builtins.input", return_value="s"):
                    scenes = self.gen.generate_all_images([_scene()], interactive=True)
        self.assertEqual(scenes[0].image_path, "")


class TestCloudflareFallback(unittest.TestCase):
    def setUp(self):
        patcher = patch("services.image_generator.genai.Client")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.gen = ImageGenerator(api_key="test_key")
        # These tests exercise the REST fallback; a real local OAuth session
        # must not redirect them to the live CLI path.
        self.gen.cloudflare_cli.available = MagicMock(return_value=False)
        self._tmp = tempfile.TemporaryDirectory(prefix="img-cf-test-")
        self.addCleanup(self._tmp.cleanup)
        self.gen.output_dir = Path(self._tmp.name)
        # Synthetic blobs in these tests aren't decodable; blank tests below
        blank_patcher = patch.object(ImageGenerator, "_is_blank_image",
                                     return_value=False)
        blank_patcher.start()
        self.addCleanup(blank_patcher.stop)
        self._creds = patch.multiple(
            "services.image_generator.config",
            IMAGE_FALLBACK_CLOUDFLARE=True,
            CLOUDFLARE_ACCOUNT_ID="test-account",
            CLOUDFLARE_API_TOKEN="test-token",
        )
        self._creds.start()
        self.addCleanup(self._creds.stop)

    @staticmethod
    def _resp(data: bytes, ctype: str):
        class _FakeResp:
            headers = {"Content-Type": ctype}

            def __init__(self, d):
                self._d = d

            def read(self):
                return self._d

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        return _FakeResp(data)

    def test_binary_response_saved(self):
        blob = b"\xff\xd8" + b"x" * 8000
        with patch("services.image_generator.urllib.request.urlopen",
                   return_value=self._resp(blob, "image/jpeg")):
            paths = self.gen._generate_cloudflare(_scene(), count=1)
        self.assertEqual(len(paths), 1)
        self.assertTrue(Path(paths[0]).is_file())
        self.assertEqual(Path(paths[0]).read_bytes(), blob)

    def test_json_base64_response_saved(self):
        blob = b"\x89PNG" + b"y" * 8000
        body = json.dumps(
            {"result": {"image": base64.b64encode(blob).decode()}}).encode()
        with patch("services.image_generator.urllib.request.urlopen",
                   return_value=self._resp(body, "application/json")):
            paths = self.gen._generate_cloudflare(_scene(), count=1)
        self.assertEqual(len(paths), 1)
        self.assertEqual(Path(paths[0]).read_bytes(), blob)

    def test_missing_creds_returns_empty_without_http(self):
        with patch("services.image_generator.config.CLOUDFLARE_API_TOKEN", ""):
            with patch("services.image_generator.urllib.request.urlopen") as mock_url:
                self.assertEqual(self.gen._generate_cloudflare(_scene(), count=2), [])
                mock_url.assert_not_called()

    def test_http_error_returns_empty(self):
        with patch("services.image_generator.urllib.request.urlopen",
                   side_effect=Exception("401 Unauthorized")):
            self.assertEqual(self.gen._generate_cloudflare(_scene(), count=1), [])

    def test_quota_prefers_cloudflare_over_pollinations(self):
        """Full Cloudflare fill must not touch Pollinations."""
        self.gen._quota_limited = True
        with patch.object(self.gen, "_generate_cloudflare",
                          return_value=["c1.jpg", "c2.jpg"]) as mock_cf:
            with patch.object(self.gen, "_generate_pollinations") as mock_pol:
                paths = self.gen.generate_candidates(_scene(), count=2, max_retries=1)
        self.assertEqual(paths, ["c1.jpg", "c2.jpg"])
        self.assertEqual(mock_cf.call_args[0][1], 2)
        mock_pol.assert_not_called()


class TestCloudflareOnly(unittest.TestCase):
    """Default provider is cloudflare: Gemini is never touched."""

    def setUp(self):
        patcher = patch("services.image_generator.genai.Client")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.gen = ImageGenerator(api_key="test_key")
        self.gen.cloudflare_cli.available = MagicMock(return_value=False)
        self._tmp = tempfile.TemporaryDirectory(prefix="img-cf-only-")
        self.addCleanup(self._tmp.cleanup)
        self.gen.output_dir = Path(self._tmp.name)

    def test_generate_candidates_skips_gemini(self):
        with patch.object(ImageGenerator, "_cloudflare_available",
                          return_value=True):
            with patch.object(self.gen, "_generate_gemini") as mock_gem:
                with patch.object(self.gen, "_generate_cloudflare",
                                  return_value=["cf1.jpg"]) as mock_cf:
                    with patch.object(self.gen, "_generate_pollinations") as mock_pol:
                        paths = self.gen.generate_candidates(_scene(), count=1, max_retries=1)
        self.assertEqual(paths, ["cf1.jpg"])
        mock_gem.assert_not_called()
        mock_pol.assert_not_called()
        mock_cf.assert_called_once()

    def test_cloudflare_unavailable_skips_cloudflare_and_uses_fallbacks(self):
        """No Cloudflare credentials -> Cloudflare is skipped, fallbacks run.

        Regression: the old code returned an empty list immediately when
        Cloudflare was unavailable, so every scene fell back to the offline
        gradient even though keyless providers were reachable.
        """
        with patch.object(self.gen, "_generate_cloudflare") as mock_cf:
            with patch.object(self.gen, "_generate_pollinations",
                              return_value=[]) as mock_pol:
                with patch.object(self.gen, "_generate_gemini",
                                  return_value=[]) as mock_gem:
                    with patch("services.image_generator.config.CLOUDFLARE_API_TOKEN", ""):
                        paths = self.gen.generate_candidates(_scene(), count=2)
        self.assertEqual(paths, [])
        mock_cf.assert_not_called()
        mock_pol.assert_called_once()
        mock_gem.assert_called_once()

    def test_no_quota_limited_flag_needed(self):
        """Cloudflare mode works regardless of Gemini quota state."""
        self.gen._quota_limited = True  # leftover Gemini quota flag is ignored
        with patch.object(ImageGenerator, "_cloudflare_available",
                          return_value=True):
            with patch.object(self.gen, "_generate_cloudflare",
                              return_value=["cf.jpg"]):
                paths = self.gen.generate_candidates(_scene(), count=1)
        self.assertEqual(paths, ["cf.jpg"])

    def test_cli_uses_flux_steps_body(self):
        """Flux 1 Schnell expects steps, not the old num_steps field."""
        self.gen.cloudflare_cli.run_ai = MagicMock(return_value=b"x" * 6000)
        with patch.object(self.gen, "_is_blank_image", return_value=False), \
             patch.object(self.gen, "_save_image",
                          return_value=Path("/tmp/cf.png")), \
             patch("services.image_generator.config.CLOUDFLARE_IMAGE_MODEL",
                  "@cf/black-forest-labs/flux-1-schnell"), \
             patch("services.image_generator.config.CLOUDFLARE_IMAGE_STEPS", 4):
            paths = self.gen._generate_cloudflare_cli(_scene(), count=1)
        self.assertEqual(paths, ["/tmp/cf.png"])
        body = self.gen.cloudflare_cli.run_ai.call_args.args[1]
        self.assertEqual(body["steps"], 4)
        self.assertNotIn("num_steps", body)


class TestBlankImageDetection(unittest.TestCase):
    """Empty/solid images are rejected before being used as scene art."""

    @staticmethod
    def _solid_png(size=(64, 64), color=(128, 128, 128)) -> bytes:
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", size, color).save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _textured_png(size=(64, 64)) -> bytes:
        import io
        import random as _r
        from PIL import Image
        img = Image.new("RGB", size)
        img.putdata([(_r.randrange(256), _r.randrange(256), _r.randrange(256))
                     for _ in range(size[0] * size[1])])
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_solid_color_is_blank(self):
        self.assertTrue(ImageGenerator._is_blank_image(self._solid_png()))

    def test_textured_image_is_not_blank(self):
        self.assertFalse(ImageGenerator._is_blank_image(self._textured_png()))

    def test_undecodable_bytes_are_blank(self):
        self.assertTrue(ImageGenerator._is_blank_image(b"not an image" * 100))

    def test_cloudflare_blank_rejected_then_retried(self):
        """Blank first response -> retried with fresh seed; only good one saved."""
        patcher = patch("services.image_generator.genai.Client")
        patcher.start()
        self.addCleanup(patcher.stop)
        gen = ImageGenerator(api_key="test_key")
        gen.cloudflare_cli.available = MagicMock(return_value=False)
        tmp = tempfile.TemporaryDirectory(prefix="img-blank-")
        self.addCleanup(tmp.cleanup)
        gen.output_dir = Path(tmp.name)
        solid, textured = self._solid_png(), self._textured_png()

        class _Resp:
            def __init__(self, data):
                self.headers = {"Content-Type": "image/png"}
                self._d = data

            def read(self):
                return self._d

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        with patch.multiple(
                "services.image_generator.config",
                IMAGE_FALLBACK_CLOUDFLARE=True,
                CLOUDFLARE_ACCOUNT_ID="acc",
                CLOUDFLARE_API_TOKEN="tok"):
            with patch("services.image_generator.urllib.request.urlopen",
                       side_effect=[_Resp(solid), _Resp(textured)]) as mock_url:
                paths = gen._generate_cloudflare(_scene(), count=1)
        self.assertEqual(len(paths), 1)
        self.assertEqual(Path(paths[0]).read_bytes(), textured)
        self.assertEqual(mock_url.call_count, 2)

    def test_cloudflare_all_blank_returns_empty(self):
        """Both tries blank -> nothing saved, never raises."""
        patcher = patch("services.image_generator.genai.Client")
        patcher.start()
        self.addCleanup(patcher.stop)
        gen = ImageGenerator(api_key="test_key")
        gen.cloudflare_cli.available = MagicMock(return_value=False)
        tmp = tempfile.TemporaryDirectory(prefix="img-blank2-")
        self.addCleanup(tmp.cleanup)
        gen.output_dir = Path(tmp.name)
        solid = self._solid_png()

        class _Resp:
            def __init__(self, data):
                self.headers = {"Content-Type": "image/png"}
                self._d = data

            def read(self):
                return self._d

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        with patch.multiple(
                "services.image_generator.config",
                IMAGE_FALLBACK_CLOUDFLARE=True,
                CLOUDFLARE_ACCOUNT_ID="acc",
                CLOUDFLARE_API_TOKEN="tok"):
            with patch("services.image_generator.urllib.request.urlopen",
                       return_value=_Resp(solid)) as mock_url:
                paths = gen._generate_cloudflare(_scene(), count=1)
        self.assertEqual(paths, [])
        self.assertEqual(mock_url.call_count, 2)


class TestCloudflareFallbackChain(unittest.TestCase):
    """Cloudflare-first provider must fall back to Pollinations/Gemini."""

    def setUp(self):
        patcher = patch("services.image_generator.genai.Client")
        self.mock_client_cls = patcher.start()
        self.addCleanup(patcher.stop)
        self.mock_client = self.mock_client_cls.return_value
        prov = patch("services.image_generator.config.IMAGE_PROVIDER", "cloudflare")
        prov.start()
        self.addCleanup(prov.stop)
        blank = patch.object(ImageGenerator, "_is_blank_image", return_value=False)
        blank.start()
        self.addCleanup(blank.stop)
        key = patch("services.image_generator.config.GEMINI_API_KEY", "test_key")
        key.start()
        self.addCleanup(key.stop)
        self.gen = ImageGenerator(api_key="test_key")
        self.gen.cloudflare_cli.available = MagicMock(return_value=False)
        self._tmp = tempfile.TemporaryDirectory(prefix="img-cf-fb-")
        self.addCleanup(self._tmp.cleanup)
        self.gen.output_dir = Path(self._tmp.name)

    def test_cloudflare_failure_falls_back_to_pollinations(self):
        """A failing Cloudflare call must not end the chain at the gradient."""
        with patch.object(ImageGenerator, "_cloudflare_available", return_value=True), \
             patch.object(ImageGenerator, "_generate_cloudflare", return_value=[]), \
             patch.object(ImageGenerator, "_generate_pollinations",
                          return_value=["/tmp/poll.png"]) as pollinations, \
             patch.object(ImageGenerator, "_generate_gemini", return_value=[]), \
             patch("services.image_generator.config.IMAGE_FALLBACK_POLLINATIONS", True), \
             patch("services.image_generator.config.IMAGE_FALLBACK_GEMINI", True):
            paths = self.gen.generate_candidates(_scene(), count=1)
        self.assertEqual(paths, ["/tmp/poll.png"])
        pollinations.assert_called_once()

    def test_gemini_used_when_pollinations_disabled(self):
        with patch.object(ImageGenerator, "_cloudflare_available", return_value=True), \
             patch.object(ImageGenerator, "_generate_cloudflare", return_value=[]), \
             patch.object(ImageGenerator, "_generate_pollinations", return_value=[]), \
             patch.object(ImageGenerator, "_generate_gemini",
                          return_value=["/tmp/gemini.png"]) as gemini, \
             patch("services.image_generator.config.IMAGE_FALLBACK_POLLINATIONS", False), \
             patch("services.image_generator.config.IMAGE_FALLBACK_GEMINI", True):
            paths = self.gen.generate_candidates(_scene(), count=1)
        self.assertEqual(paths, ["/tmp/gemini.png"])
        gemini.assert_called_once()

    def test_missing_cloudflare_config_still_uses_fallbacks(self):
        """No Cloudflare credentials must not skip the alternative providers."""
        with patch.object(ImageGenerator, "_cloudflare_available", return_value=False), \
             patch.object(ImageGenerator, "_generate_pollinations",
                          return_value=["/tmp/poll.png"]) as pollinations, \
             patch("services.image_generator.config.IMAGE_FALLBACK_POLLINATIONS", True), \
             patch("services.image_generator.config.IMAGE_FALLBACK_GEMINI", True):
            paths = self.gen.generate_candidates(_scene(), count=1)
        self.assertEqual(paths, ["/tmp/poll.png"])
        pollinations.assert_called_once()

    def test_no_provider_available_returns_empty(self):
        """Everything failing must return an empty list (gradient takes over)."""
        with patch.object(ImageGenerator, "_cloudflare_available", return_value=False), \
             patch.object(ImageGenerator, "_generate_pollinations", return_value=[]), \
             patch.object(ImageGenerator, "_generate_gemini", return_value=[]), \
             patch("services.image_generator.config.IMAGE_FALLBACK_POLLINATIONS", True), \
             patch("services.image_generator.config.IMAGE_FALLBACK_GEMINI", True):
            paths = self.gen.generate_candidates(_scene(), count=1)
        self.assertEqual(paths, [])

    def test_cloudflare_401_stops_further_attempts(self):
        """An invalid token must not be retried for every candidate."""
        import urllib.error
        error = urllib.error.HTTPError("https://api.cloudflare.com/client/v4/x",
                                        401, "Unauthorized", {}, None)
        with patch("services.image_generator.urllib.request.urlopen",
                   side_effect=error) as urlopen:
            with patch.object(ImageGenerator, "_cloudflare_available", return_value=True), \
                 patch("services.image_generator.config.CLOUDFLARE_ACCOUNT_ID", "acct"), \
                 patch("services.image_generator.config.CLOUDFLARE_API_TOKEN", "token"), \
                 patch("services.image_generator.config.CLOUDFLARE_IMAGE_MODEL", "model"), \
                 patch("services.image_generator.config.CLOUDFLARE_TIMEOUT", 5):
                paths = self.gen._generate_cloudflare(_scene(), count=3)
        self.assertEqual(paths, [])
        self.assertEqual(urlopen.call_count, 1)


if __name__ == "__main__":
    unittest.main()
