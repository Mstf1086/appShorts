"""Tests for services.colab_runner.ColabRunner (Colab CLI integration)."""

import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from services.colab_runner import PIPELINE_EXEC_TIMEOUT, ColabRunner


def _ok(stdout: str = "", stderr: str = "") -> MagicMock:
    return MagicMock(returncode=0, stdout=stdout, stderr=stderr)


class TestColabRunner(unittest.TestCase):
    def setUp(self):
        self.runner = ColabRunner(session_name="test-session", gpu="T4")
        self._tmp = tempfile.TemporaryDirectory(prefix="colab-runner-test-")
        self.runner._temp_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    # ------------------------------------------------------------------
    # run_pipeline
    # ------------------------------------------------------------------
    @patch("services.colab_runner.subprocess.run")
    def test_run_pipeline_sets_explicit_long_timeout(self, mock_run):
        mock_run.return_value = _ok()

        self.runner.run_pipeline("Game of Thrones")

        cmd = mock_run.call_args[0][0]
        self.assertIn("--timeout", cmd)
        timeout = float(cmd[cmd.index("--timeout") + 1])
        # CLI default (30s) would kill the multi-minute pipeline run
        self.assertGreater(timeout, 30)
        self.assertEqual(timeout, float(PIPELINE_EXEC_TIMEOUT))
        self.assertIn("-f", cmd)

    @patch("services.colab_runner.subprocess.run")
    def test_run_pipeline_escapes_topic_and_parses_output_path(self, mock_run):
        mock_run.return_value = _ok(
            stdout="noise\nOUTPUT_PATH:/content/output/shorts_1.mp4\n"
        )

        topic = 'Konusu "tırnaklı" ve \'karışık\' konu'
        self.runner.run_pipeline(topic)

        script = (self.runner._temp_dir / "run_shorts.py").read_text()
        self.assertIn(f"topic = {topic!r}", script)
        self.assertIn('os.chdir("/content")', script)
        self.assertEqual(
            self.runner._remote_output, "/content/output/shorts_1.mp4"
        )

    # ------------------------------------------------------------------
    # install_dependencies
    # ------------------------------------------------------------------
    @patch("services.colab_runner.subprocess.run")
    def test_install_dependencies_passes_local_requirements_file(self, mock_run):
        mock_run.return_value = _ok()

        self.runner.install_dependencies()

        r_calls = [
            call[0][0] for call in mock_run.call_args_list if "-r" in call[0][0]
        ]
        self.assertTrue(r_calls, "install -r call missing")
        req_path = r_calls[0][r_calls[0].index("-r") + 1]
        # `-r` must point at an existing LOCAL file (CLI rejects remote paths)
        self.assertTrue(Path(req_path).is_file())
        self.assertNotIn("/content/", req_path)

    # ------------------------------------------------------------------
    # _prepare_bundle
    # ------------------------------------------------------------------
    def test_prepare_bundle_includes_core_files_and_env(self):
        bundle = self.runner._prepare_bundle()
        with tarfile.open(bundle) as tar:
            names = tar.getnames()
        for expected in ("pipeline", "services", "config.py", "requirements.txt"):
            self.assertIn(expected, names)
        # .env travels with the bundle so remote config.py can load API keys;
        # only the member name is inspected, contents are never read/printed.
        if (self.runner._project_root / ".env").exists():
            self.assertIn(".env", names)

    # ------------------------------------------------------------------
    # download_output
    # ------------------------------------------------------------------
    @patch("services.colab_runner.subprocess.run")
    def test_download_output_prefers_reported_path(self, mock_run):
        mock_run.return_value = _ok()
        self.runner._remote_output = "/content/output/shorts_1.mp4"

        local = self.runner.download_output()

        self.assertIsNotNone(local)
        self.assertTrue(local.endswith("shorts_1.mp4"))
        # Only the download command runs - no remote search needed
        self.assertEqual(mock_run.call_count, 1)
        self.assertIn("download", mock_run.call_args[0][0])

    @patch("services.colab_runner.subprocess.run")
    def test_download_output_falls_back_to_remote_search(self, mock_run):
        mock_run.side_effect = [
            _ok(stdout="FOUND_VIDEO:/content/output/shorts_2.mp4\n"),
            _ok(),
        ]

        local = self.runner.download_output()

        self.assertIsNotNone(local)
        self.assertTrue(local.endswith("shorts_2.mp4"))
        find_call = mock_run.call_args_list[0]
        self.assertIn("--timeout", find_call[0][0])
        # Search code prefers the final shorts_* video over source clips
        self.assertIn("shorts_*.mp4", find_call[1].get("input", ""))

    # ------------------------------------------------------------------
    # error handling
    # ------------------------------------------------------------------
    @patch("services.colab_runner.subprocess.run")
    def test_stop_session_never_raises(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="boom")

        # Cleanup must swallow CLI failures instead of masking the real error
        self.assertTrue(self.runner.stop_session())

    @patch("services.colab_runner.subprocess.run")
    def test_run_cmd_raises_runtime_error_with_auth_hint(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="Enter the authorization code:\nAborted."
        )

        with self.assertRaises(RuntimeError) as ctx:
            self.runner._run_cmd(["colab", "new", "-s", "x"])

        self.assertIn("not authorized yet", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
