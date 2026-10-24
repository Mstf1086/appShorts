"""Tests for keyless Cloudflare CLI authentication and Workers AI calls."""
import json
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from services.cloudflare_cli import CloudflareCLI


class TestCloudflareCLI(unittest.TestCase):
    """The cf CLI profile is the credential source for Workers AI."""

    def test_stale_environment_token_does_not_shadow_oauth_profile(self):
        client = CloudflareCLI()
        env = client._command_env()
        self.assertNotIn("CLOUDFLARE_API_TOKEN", env)
        self.assertNotIn("CLOUDFLARE_API_KEY", env)

    @patch("services.cloudflare_cli.shutil.which", return_value="/usr/bin/cf")
    @patch("services.cloudflare_cli.subprocess.run")
    def test_available_requires_valid_authenticated_session(self, mock_run, _which):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"authenticated": True, "tokenValid": True}),
        )
        client = CloudflareCLI()
        self.assertTrue(client.available())
        command = mock_run.call_args.args[0]
        self.assertEqual(command[:3], ["cf", "auth", "whoami"])
        self.assertEqual(mock_run.call_args.kwargs["cwd"], str(Path.home()))

    @patch("services.cloudflare_cli.shutil.which", return_value="/usr/bin/cf")
    @patch("services.cloudflare_cli.subprocess.run")
    def test_invalid_session_is_unavailable(self, mock_run, _which):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"authenticated": True, "tokenValid": False}),
        )
        self.assertFalse(CloudflareCLI().available())

    @patch("services.cloudflare_cli.subprocess.run")
    def test_list_models_returns_parsed_catalog(self, mock_run):
        catalog = [{"name": "@cf/example/model"}]
        mock_run.return_value = MagicMock(
            returncode=0, stdout=json.dumps(catalog).encode())
        result = CloudflareCLI().list_models()
        self.assertEqual(result, catalog)
        command = mock_run.call_args.args[0]
        self.assertEqual(command[:4], ["cf", "ai", "models", "list"])
        self.assertIn("--per-page", command)

    @patch("services.cloudflare_cli.subprocess.run")
    def test_get_model_schema_returns_parsed_schema(self, mock_run):
        schema = {"input": {"properties": {"prompt": {}}}}
        mock_run.return_value = MagicMock(
            returncode=0, stdout=json.dumps(schema).encode())
        result = CloudflareCLI().get_model_schema("@cf/example/model")
        self.assertEqual(result, schema)
        command = mock_run.call_args.args[0]
        self.assertEqual(command[:4], ["cf", "ai", "get-model-schema", "--model"])

    def test_run_ai_routes_multipart_payload_to_multipart_request(self):
        client = CloudflareCLI()
        client.run_ai_multipart = MagicMock(return_value=b"image")
        result = client.run_ai("@cf/example/model", {
            "multipart": {
                "body": {"prompt": "hello"},
                "contentType": "application/json",
            },
        })
        self.assertEqual(result, b"image")
        client.run_ai_multipart.assert_called_once_with(
            "@cf/example/model", {"prompt": "hello"})

    @patch("services.cloudflare_cli.subprocess.run")
    def test_run_ai_passes_body_and_returns_binary_stdout(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout=b"\x89PNGdata")
        result = CloudflareCLI().run_ai(
            "@cf/example/model", {"prompt": "hello"})
        self.assertEqual(result, b"\x89PNGdata")
        command = mock_run.call_args.args[0]
        self.assertEqual(command[:3], ["cf", "ai", "run"])
        self.assertIn("--body", command)
        body = command[command.index("--body") + 1]
        self.assertEqual(json.loads(body), {"prompt": "hello"})

    @patch("services.cloudflare_cli.subprocess.run")
    def test_run_ai_raises_with_safe_error_detail(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1, stdout=b"", stderr=b"authentication failed")
        with self.assertRaisesRegex(RuntimeError, "authentication failed"):
            CloudflareCLI().run_ai("@cf/example/model", {"prompt": "hello"})

    @patch("services.cloudflare_cli.subprocess.run")
    def test_login_uses_device_flow_without_browser(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        self.assertEqual(CloudflareCLI().login(), 0)
        command = mock_run.call_args.args[0]
        self.assertEqual(command[:3], ["cf", "auth", "login"])
        self.assertIn("--device", command)
        self.assertIn("--no-browser", command)


if __name__ == "__main__":
    unittest.main()
