#!/usr/bin/env python3
"""
Google Colab CLI Runner

Integrates the YouTube Shorts pipeline with Google Colab CLI for remote execution
on cloud-based GPU/TPU runtimes.

Key features:
  - Provision remote Colab VMs (CPU, GPU, TPU)
  - Install dependencies on remote runtime
  - Execute the Shorts pipeline remotely
  - Download generated videos back to local machine
  - Automatic session cleanup

Usage:
    python generate_shorts.py "Konu başlığı" --colab
    python generate_shorts.py "Konu başlığı" --colab --gpu T4
    python generate_shorts.py "Konu başlığı" --colab --gpu A100
"""

import subprocess
import sys
import shutil
import tempfile
import tarfile
import os
from pathlib import Path
from typing import Optional

# `colab exec` defaults to a 30s timeout - the full pipeline (scene
# generation + downloads + TTS + render) needs a much larger budget.
PIPELINE_EXEC_TIMEOUT = 1800


class ColabRunner:
    """Manages Google Colab CLI sessions for remote pipeline execution."""

    def __init__(
        self,
        session_name: str = "shorts-gen",
        gpu: Optional[str] = None,
        tpu: Optional[str] = None,
        high_mem: bool = False,
    ):
        self.session_name = session_name
        self.gpu = gpu
        self.tpu = tpu
        self.high_mem = high_mem
        self._project_root = Path(__file__).parent.parent
        self._temp_dir: Optional[Path] = None
        self._remote_output: str = ""

    @property
    def _colab_cmd(self) -> list[str]:
        """Return the colab CLI command using the venv Python."""
        return [sys.executable, "-m", "colab_cli.cli"]

    def _check_colab_cli(self) -> bool:
        """Check if the Colab CLI is installed and accessible."""
        try:
            result = subprocess.run(
                self._colab_cmd + ["--help"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _run_cmd(self, cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command and return the result."""
        # Replace 'colab' with our venv-based command
        if cmd[0] == "colab":
            cmd = self._colab_cmd + cmd[1:]
        print(f"  [colab] $ colab {' '.join(cmd[len(self._colab_cmd):] if cmd[:len(self._colab_cmd)] == self._colab_cmd else cmd[1:])}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout.strip():
            print(f"  [colab] {result.stdout.strip()}")
        if result.stderr.strip():
            # Filter out non-error stderr (some colab commands write info to stderr)
            stderr = result.stderr.strip()
            if "Error" in stderr or "error" in stderr or "Traceback" in stderr:
                print(f"  [colab] WARNING: {stderr}")
        if check and result.returncode != 0:
            message = (
                f"Colab CLI command failed (exit {result.returncode}): {' '.join(cmd)}\n"
                f"stderr: {result.stderr}"
            )
            if "authorization code" in (result.stderr or ""):
                message += (
                    "\nHint: Colab CLI is not authorized yet - open the URL above in a "
                    "browser, approve access and paste the authorization code."
                )
            raise RuntimeError(message)
        return result

    def _prepare_bundle(self) -> Path:
        """
        Create a tarball of the project files needed for remote execution.
        This bundles the pipeline, services, config, and requirements.
        """
        self._temp_dir = Path(tempfile.mkdtemp(prefix="colab-shorts-"))
        bundle_path = self._temp_dir / "shorts_project.tar.gz"

        # Files/dirs to include in the bundle
        include_paths = [
            "pipeline",
            "services",
            "config.py",
            "generate_shorts.py",
            "requirements.txt",
            ".env",  # API keys for the remote pipeline (contents never printed)
        ]

        with tarfile.open(bundle_path, "w:gz") as tar:
            for rel_path in include_paths:
                full_path = self._project_root / rel_path
                if full_path.exists():
                    tar.add(str(full_path), arcname=rel_path)

        return bundle_path

    def create_session(self) -> bool:
        """Provision a new Colab VM session."""
        if not self._check_colab_cli():
            print("  [colab] Error: 'colab' CLI not found.")
            print("  [colab] Install it with: uv tool install google-colab-cli")
            return False

        cmd = ["colab", "new", "-s", self.session_name]
        if self.gpu:
            cmd.extend(["--gpu", self.gpu])
        if self.tpu:
            cmd.extend(["--tpu", self.tpu])
        if self.high_mem:
            cmd.append("--high-mem")

        self._run_cmd(cmd)
        return True

    def install_dependencies(self) -> bool:
        """Install project dependencies on the remote VM."""
        # `-r` requires a LOCAL requirements file - the CLI uploads it itself
        # (colab_cli.commands.automation checks os.path.isfile locally).
        req_path = self._project_root / "requirements.txt"
        if req_path.exists():
            self._run_cmd([
                "colab", "install", "-s", self.session_name,
                "-r", str(req_path),
            ])

        # Also install edge-tts which may need special handling
        self._run_cmd([
            "colab", "install", "-s", self.session_name,
            "edge-tts", "moviepy", "google-genai", "python-dotenv", "requests"
        ])

        return True

    def upload_project(self) -> bool:
        """Upload the project bundle to the remote VM and extract it."""
        bundle_path = self._prepare_bundle()

        # Upload the tarball
        self._run_cmd([
            "colab", "upload", "-s", self.session_name,
            str(bundle_path), "/content/shorts_project.tar.gz"
        ])

        # Extract on remote VM (piped code is supported by `colab exec`)
        extract_code = "import tarfile; tarfile.open('/content/shorts_project.tar.gz', 'r:gz').extractall('/content/'); print('Extracted')"
        extract_cmd = self._colab_cmd + ["exec", "-s", self.session_name, "--timeout", "120"]
        result = subprocess.run(
            extract_cmd,
            input=extract_code,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(f"  [colab] Extract warning: {result.stderr}")

        return True

    def run_pipeline(self, topic: str) -> bool:
        """
        Execute the Shorts generation pipeline on the remote Colab VM.
        Returns True if the video was generated successfully.
        """
        # Build the execution script
        exec_script = self._temp_dir / "run_shorts.py"
        exec_script.write_text(f'''#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, "/content")
os.chdir("/content")

# Topic is injected via repr() so quotes/newlines stay safe
topic = {topic!r}

# Import and run pipeline (API keys come from the bundled /content/.env)
from pipeline.shorts_pipeline import ShortsPipeline

def main():
    pipeline = ShortsPipeline()
    output_path = pipeline.run(topic)
    print(f"OUTPUT_PATH:{{output_path}}")
    return output_path

if __name__ == "__main__":
    main()
''')

        # Run the pipeline (generous timeout: scenes + downloads + TTS + render;
        # plain `colab exec` would abort after its 30s default)
        result = self._run_cmd([
            "colab", "exec", "-s", self.session_name,
            "--timeout", str(PIPELINE_EXEC_TIMEOUT),
            "-f", str(exec_script),
        ])

        # Remember the exact remote output path reported by the pipeline
        self._remote_output = ""
        for line in (result.stdout or "").splitlines():
            if line.startswith("OUTPUT_PATH:"):
                self._remote_output = line.split("OUTPUT_PATH:", 1)[1].strip()

        return True

    def download_output(self, local_output_dir: Optional[Path] = None) -> Optional[str]:
        """
        Download the generated video from the remote VM.
        Returns the local path of the downloaded video.
        """
        if local_output_dir is None:
            local_output_dir = self._project_root / "output"
        local_output_dir.mkdir(parents=True, exist_ok=True)

        # 1) Prefer the exact path reported by the remote pipeline run
        video_path = getattr(self, "_remote_output", "") or None

        # 2) Fall back to searching the remote filesystem
        #    (final video first - source clips live in output/tmp/)
        if not video_path:
            find_video_code = '''
import os
import glob

candidates = glob.glob("/content/output/shorts_*.mp4")
if not candidates:
    candidates = glob.glob("/content/output/**/*.mp4", recursive=True)
if not candidates:
    candidates = glob.glob("/content/**/*.mp4", recursive=True)
if candidates:
    print("FOUND_VIDEO:" + max(candidates, key=os.path.getmtime))
else:
    print("NO_VIDEO_FOUND")
'''
            find_cmd = self._colab_cmd + ["exec", "-s", self.session_name, "--timeout", "120"]
            result = subprocess.run(
                find_cmd,
                input=find_video_code,
                capture_output=True,
                text=True,
            )

            for line in (result.stdout or "").split("\n"):
                if line.startswith("FOUND_VIDEO:"):
                    video_path = line.split("FOUND_VIDEO:", 1)[1].strip()
                    break

        if video_path:
            local_name = Path(video_path).name
            local_path = local_output_dir / local_name
            self._run_cmd([
                "colab", "download", "-s", self.session_name,
                video_path, str(local_path)
            ])
            print(f"  [colab] Downloaded video to: {local_path}")
            return str(local_path)
        else:
            print("  [colab] No video file found on remote VM")
            return None

    def stop_session(self) -> bool:
        """Terminate the Colab session and release resources."""
        try:
            # check=False: cleanup must never raise and mask the real error
            self._run_cmd(["colab", "stop", "-s", self.session_name], check=False)
        except Exception as exc:
            print(f"  [colab] Warning: could not stop session: {exc}")
        if self._temp_dir and self._temp_dir.exists():
            # Cleanup temp files
            import shutil as sh
            sh.rmtree(self._temp_dir, ignore_errors=True)
        return True

    def run_one_shot(self, topic: str) -> Optional[str]:
        """
        Run the entire pipeline in a single command using `colab run`.
        This provisions a VM, runs the script, and tears it down automatically.
        """
        if not self._check_colab_cli():
            print("  [colab] Error: 'colab' CLI not found.")
            print("  [colab] Install it with: uv tool install google-colab-cli")
            return None

        # Create a self-contained script
        self._temp_dir = Path(tempfile.mkdtemp(prefix="colab-shorts-"))
        script_path = self._temp_dir / "run_on_colab.py"

        # Read the project files to embed
        pipeline_code = (self._project_root / "pipeline" / "shorts_pipeline.py").read_text()
        config_code = (self._project_root / "config.py").read_text()

        # Escape triple quotes in config_code for embedding
        escaped_config = config_code.replace("'", "\\'")

        script_content = (
            '#!/usr/bin/env python3\n'
            '"""Self-contained Shorts generator for Colab remote execution."""\n'
            'import sys\n'
            'import os\n'
            'sys.path.insert(0, "/content")\n'
            '\n'
            '# Install dependencies\n'
            'import subprocess\n'
            'subprocess.check_call([sys.executable, "-m", "pip", "install",\n'
            '    "edge-tts", "moviepy", "google-genai", "python-dotenv", "requests", "decorator"])\n'
            '\n'
            '# Create config module\n'
            f"config_code = '{escaped_config}'\n"
            'exec(compile(config_code, "config.py", "exec"))\n'
            '\n'
            f'topic = "{topic}"\n'
            '\n'
            '# Import and run\n'
            'from pipeline.shorts_pipeline import ShortsPipeline\n'
            'pipeline = ShortsPipeline()\n'
            'output_path = pipeline.run(topic)\n'
            'print(f"OUTPUT:{output_path}")\n'
        )
        script_path.write_text(script_content)

        cmd = ["colab", "run"]
        if self.gpu:
            cmd.extend(["--gpu", self.gpu])
        if self.tpu:
            cmd.extend(["--tpu", self.tpu])
        if self.high_mem:
            cmd.append("--high-mem")
        cmd.append(str(script_path))

        print(f"  [colab] Running one-shot pipeline...")
        self._run_cmd(cmd)

        # Cleanup
        if self._temp_dir and self._temp_dir.exists():
            import shutil as sh
            sh.rmtree(self._temp_dir, ignore_errors=True)

        return None

    def run_full_pipeline(self, topic: str) -> Optional[str]:
        """
        Complete workflow: create session, install deps, upload, run, download, cleanup.
        Returns the local path of the generated video.
        """
        print(f"\n{'='*60}")
        print(f"Google Colab CLI - Remote Execution")
        print(f"Session: {self.session_name}")
        if self.gpu:
            print(f"GPU: {self.gpu}")
        if self.tpu:
            print(f"TPU: {self.tpu}")
        print(f"{'='*60}\n")

        try:
            # Step 1: Create session
            print("[1/5] Provisioning Colab VM...")
            if not self.create_session():
                return None
            print("  ✓ Session created\n")

            # Step 2: Install dependencies
            print("[2/5] Installing dependencies...")
            self.install_dependencies()
            print("  ✓ Dependencies installed\n")

            # Step 3: Upload project
            print("[3/5] Uploading project files...")
            self.upload_project()
            print("  ✓ Project uploaded\n")

            # Step 4: Run pipeline
            print("[4/5] Running Shorts pipeline on remote VM...")
            self.run_pipeline(topic)
            print("  ✓ Pipeline execution complete\n")

            # Step 5: Download output
            print("[5/5] Downloading generated video...")
            output_path = self.download_output()
            if output_path:
                print(f"  ✓ Video downloaded: {output_path}\n")
            else:
                print("  ✗ No video found\n")

            return output_path

        except Exception as e:
            print(f"\n  ✗ Colab execution error: {e}")
            return None

        finally:
            # Always cleanup - wrapped so cleanup can never mask the real error
            print("Cleaning up Colab session...")
            try:
                self.stop_session()
                print("  ✓ Session terminated\n")
            except Exception as exc:
                print(f"  [colab] Warning: cleanup failed: {exc}\n")


def run_with_colab(
    topic: str,
    gpu: Optional[str] = None,
    session_name: str = "shorts-gen",
) -> Optional[str]:
    """
    Convenience function to run the Shorts pipeline on Google Colab.

    Args:
        topic: The topic for the YouTube Shorts video.
        gpu: Optional GPU type (T4, L4, G4, H100, A100).
        session_name: Name for the Colab session.

    Returns:
        Path to the generated video, or None if failed.
    """
    runner = ColabRunner(
        session_name=session_name,
        gpu=gpu,
    )
    return runner.run_full_pipeline(topic)
