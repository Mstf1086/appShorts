"""Cloudflare Workers AI access through the authenticated ``cf`` CLI.

The CLI stores OAuth credentials in its own profile/keyring, so this service
can use Workers AI without reading or copying an API token into the project.
"""
import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional


class CloudflareCLI:
    """Run Workers AI inference with a logged-in Cloudflare CLI profile."""

    def __init__(self, command: str = "cf", profile: str = None,
                 timeout: int = 120):
        self.command = command
        self.profile = (profile or "").strip()
        self.timeout = timeout
        # Run outside the project directory: cf loads a local .env file and a
        # stale project token would otherwise override the OAuth profile.
        self.working_dir = str(Path.home())
        self._available: Optional[bool] = None

    def _base_command(self) -> list[str]:
        """Return the CLI command without authentication flags."""
        return [self.command]

    def _command_env(self) -> dict[str, str]:
        """Return an environment that cannot shadow the OAuth profile."""
        env = os.environ.copy()
        # A stale environment token takes precedence over OAuth in the CLI.
        env.pop("CLOUDFLARE_API_TOKEN", None)
        env.pop("CLOUDFLARE_API_KEY", None)
        return env

    def _profile_args(self) -> list[str]:
        """Return optional named-profile arguments."""
        return ["--profile", self.profile] if self.profile else []

    def available(self) -> bool:
        """Return whether the CLI exists and has a valid authenticated session."""
        if self._available is not None:
            return self._available
        try:
            executable = shutil.which(self.command)
        except (OSError, TypeError):
            executable = None
        if executable is None:
            self._available = False
            return False
        try:
            result = subprocess.run(
                self._base_command() + ["auth", "whoami"]
                + self._profile_args(),
                capture_output=True,
                text=True,
                timeout=15,
                env=self._command_env(),
                cwd=self.working_dir,
                check=False,
            )
            payload = json.loads(result.stdout or "{}")
            self._available = bool(
                result.returncode == 0
                and payload.get("authenticated")
                and payload.get("tokenValid")
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            self._available = False
        return self._available

    def login(self) -> int:
        """Start the CLI OAuth/device login flow and return its exit code.

        A named profile is created when configured; otherwise the CLI's
        default profile is used. The command intentionally streams output so
        the user can open the verification URL and enter the device code.
        """
        if self.profile:
            command = (self._base_command() + ["auth", "create", self.profile,
                       "--device", "--no-browser"])
        else:
            command = self._base_command() + ["auth", "login", "--device",
                                             "--no-browser"]
        result = subprocess.run(
            command, env=self._command_env(), cwd=self.working_dir, check=False)
        return result.returncode

    def list_models(self, per_page: int = 100) -> list[dict]:
        """Return the Workers AI model catalog as parsed JSON."""
        command = (self._base_command() + ["ai", "models", "list", "--per-page",
                   str(per_page)] + self._profile_args())
        result = subprocess.run(
            command,
            capture_output=True,
            timeout=self.timeout,
            env=self._command_env(),
            cwd=self.working_dir,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Cloudflare CLI model list failed: {detail[:300]}")
        payload = json.loads(result.stdout or "[]")
        if not isinstance(payload, list):
            raise RuntimeError("Unexpected Workers AI model catalog response")
        return payload

    def get_model_schema(self, model: str) -> dict:
        """Return the input/output schema for one Workers AI model."""
        if not model:
            raise ValueError("Cloudflare model name is required")
        command = (self._base_command() + ["ai", "get-model-schema", "--model",
                   model] + self._profile_args())
        result = subprocess.run(
            command,
            capture_output=True,
            timeout=self.timeout,
            env=self._command_env(),
            cwd=self.working_dir,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Cloudflare CLI schema lookup failed: {detail[:300]}")
        payload = json.loads(result.stdout or "{}")
        if not isinstance(payload, dict):
            raise RuntimeError("Unexpected Workers AI model schema response")
        return payload

    def _oauth_credentials(self) -> tuple[str, str]:
        """Return the OAuth token and account ID from the CLI profile.

        Credentials are read only in memory and are never printed. This is
        needed for Workers AI endpoints whose binary response is not preserved
        by ``cf ai run`` (for example Deepgram Aura).
        """
        profile_name = self.profile or "default"
        config_path = (Path.home() / ".config" / "cloudflare" / "config"
                       / f"{profile_name}.json")
        try:
            profile = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Cloudflare OAuth profile is unavailable: {exc}") from exc
        token = str(profile.get("oauth_token", "")).strip()
        if not token:
            raise RuntimeError("Cloudflare OAuth token is unavailable")
        result = subprocess.run(
            self._base_command() + ["accounts", "list"] + self._profile_args(),
            capture_output=True, text=True, timeout=self.timeout,
            env=self._command_env(), cwd=self.working_dir, check=False)
        try:
            accounts = json.loads(result.stdout or "[]")
        except ValueError as exc:
            raise RuntimeError("Invalid Cloudflare account list") from exc
        if result.returncode != 0 or not accounts:
            raise RuntimeError("Cloudflare account list is unavailable")
        return token, str(accounts[0].get("id", ""))

    def run_ai_multipart(self, model: str, fields: dict) -> bytes:
        """Run a Workers AI model using a real multipart/form-data request."""
        token, account_id = self._oauth_credentials()
        boundary = "----appShortsWorkersAI"
        chunks = []
        for name, value in fields.items():
            chunks.extend([
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(
                    "utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ])
        chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
        request = urllib.request.Request(
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
            f"/ai/run/{model}",
            data=b"".join(chunks),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Cloudflare Workers AI multipart request failed ({exc.code}): "
                f"{detail[:300]}") from exc

    def run_ai_binary(self, model: str, inputs: dict) -> bytes:
        """Run a Workers AI model and preserve a binary response body."""
        token, account_id = self._oauth_credentials()
        body = json.dumps(inputs, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
            f"/ai/run/{model}",
            data=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Cloudflare Workers AI binary request failed ({exc.code}): "
                f"{detail[:300]}") from exc

    def run_ai(self, model: str, inputs: dict) -> bytes:
        """Run one Workers AI model and return its raw stdout response."""
        if not model:
            raise ValueError("Cloudflare model name is required")
        if isinstance(inputs, dict) and "multipart" in inputs:
            multipart = inputs["multipart"]
            body = multipart.get("body", {}) if isinstance(multipart, dict) else {}
            return self.run_ai_multipart(model, body)
        command = (self._base_command() + ["ai", "run", model, "--body",
                   json.dumps(inputs, separators=(",", ":"))]
                   + self._profile_args())
        result = subprocess.run(
            command,
            capture_output=True,
            timeout=self.timeout,
            env=self._command_env(),
            cwd=self.working_dir,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            if not detail:
                detail = result.stdout.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Cloudflare CLI failed ({result.returncode}): {detail[:300]}")
        return result.stdout
