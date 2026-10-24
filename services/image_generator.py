"""Scene image generation (Cloudflare-only by default, Gemini opt-in).

Primary provider is Cloudflare Workers AI (`IMAGE_PROVIDER=cloudflare`);
Gemini chain (`IMAGE_PROVIDER=gemini`) is kept as opt-in. Every saved
image is validated (decodes + not blank) before acceptance. Approval flow
is provider-agnostic: candidates are plain file paths either way.
"""
import base64
import hashlib
import json
import random
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from google import genai
from google.genai import types as genai_types

import config
from services.cloudflare_cli import CloudflareCLI
from services.cloudflare_model_catalog import (
    build_image_payload,
    decode_image_response,
)
from services.scene_generator import Scene


class ImageGenerator:
    """Generates portrait scene images with Gemini and lets the user approve them.

    Approval loop (interactive mode): Enter=approve, d=change (next candidate /
    regenerate batch), s=skip scene (black background), q=cancel the whole run.
    When stdin is not a TTY (Colab/CI/remote) the first candidate is auto-approved.
    """

    def __init__(self, api_key: str = None, model: str = None,
                 cloudflare_model: str = None, provider: str = None):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.provider = (provider or getattr(
            config, "IMAGE_PROVIDER", "cloudflare") or "cloudflare").lower()
        if self.provider == "gemini" and not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set. Set it in .env file.")
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        # Comma-separated env value defines the failover order (config: "m1,m2")
        self.model_name = model or getattr(config, "GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
        self.model_names = [m.strip() for m in str(self.model_name).split(",") if m.strip()]
        self._model_idx = 0
        self.output_dir = config.IMAGES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Fail-fast flag: once every Gemini model quota is exhausted, stop calling the API
        self._quota_limited = False
        self.cloudflare_cli = CloudflareCLI(
            command=getattr(config, "CLOUDFLARE_CLI_COMMAND", "cf"),
            profile=getattr(config, "CLOUDFLARE_CLI_PROFILE", ""),
            timeout=getattr(config, "CLOUDFLARE_CLI_TIMEOUT", 120),
        )
        self.cloudflare_model = (cloudflare_model or getattr(
            config, "CLOUDFLARE_IMAGE_MODEL",
            "@cf/black-forest-labs/flux-1-schnell"))

    # ------------------------------------------------------------------ prompt
    def build_prompt(self, scene: Scene) -> str:
        """Build the Turkish image prompt for a scene (visual hints stay English)."""
        return (
            f"Bu sahneyi görselleştir: {scene.visual_description}. "
            f"Görsel anahtar kelimeler: {scene.search_query}. "
            f"YouTube Short için dikey 9:16 kompozisyon ({config.IMAGE_WIDTH}x{config.IMAGE_HEIGHT}), "
            "sinematik, yüksek detay, metin, filigran veya çerçeve içermeyen."
        )

    # -------------------------------------------------------------- generation
    @staticmethod
    def _is_quota_error(exc: Exception) -> bool:
        text = str(exc)
        return "RESOURCE_EXHAUSTED" in text or "429" in text

    @staticmethod
    def _extract_image_bytes(response) -> Optional[tuple]:
        """Return (bytes, mime_type) of the first inline image in the response."""
        try:
            parts = response.candidates[0].content.parts
        except (AttributeError, IndexError, TypeError):
            return None
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline and inline.data:
                return inline.data, inline.mime_type or "image/png"
        return None

    def _save_image(self, scene: Scene, data: bytes, mime_type: str, index: int) -> Path:
        ext = ".jpg" if "jpeg" in mime_type else ".png"
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        path = self.output_dir / f"scene_{scene.id:02d}_{stamp}_c{index}{ext}"
        path.write_bytes(data)
        return path

    @staticmethod
    def _is_blank_image(data: bytes) -> bool:
        """True when the image decodes but is blank (solid color / tiny variance).

        Downscales to 32x32 grayscale and checks pixel stddev; a real
        diffusion photo always has texture, a failed solid fill has ~0.
        Undecodable data is also treated as blank (rejected).
        """
        try:
            from PIL import Image
            import io
            import statistics
            with Image.open(io.BytesIO(data)) as im:
                small = im.convert("L").resize((32, 32))
                px = list(small.getdata())
            return statistics.pstdev(px) < 3.0
        except Exception:
            return True

    def _accept_image(self, scene: Scene, data: bytes, mime_type: str,
                      index: int, tag: str) -> Optional[str]:
        """Validate + save one image; returns path or None when rejected."""
        if self._is_blank_image(data):
            print(f"  {tag}: boş resim reddedildi (yeniden deneniyor)")
            return None
        saved = self._save_image(scene, data, mime_type, index=index)
        print(f"  {tag} image: {saved.name}")
        return str(saved)

    def generate_candidates(self, scene: Scene, count: int = None,
                            max_retries: int = None) -> List[str]:
        """Generate up to `count` candidate images for a scene.

        Default provider (`IMAGE_PROVIDER=cloudflare`): Cloudflare Workers AI
        first, then the keyless Pollinations fallback, then Gemini (both
        configurable). The offline gradient stays the last resort. With
        `IMAGE_PROVIDER=gemini` the Gemini-first chain runs (Cloudflare +
        Pollinations on quota). Returns the list of downloaded file paths (may
        be empty).
        """
        count = count or config.IMAGE_CANDIDATES
        max_retries = max_retries or config.IMAGE_MAX_RETRIES
        if self.provider != "gemini":
            paths: List[str] = []
            if self._cloudflare_available():
                paths = self._generate_cloudflare(scene, count)
            else:
                print("  ✗ Cloudflare kullanılamıyor → anahtarsız giriş yapın:\n"
                      "      .venv/bin/python generate_shorts.py --cf-login\n"
                      "    Alternatif: .env'e CLOUDFLARE_ACCOUNT_ID ve "
                      "CLOUDFLARE_API_TOKEN ekleyin.\n"
                      "    (config IMAGE_PROVIDER=cloudflare, flag "
                      f"{config.IMAGE_FALLBACK_CLOUDFLARE}, id var="
                      f"{bool(config.CLOUDFLARE_ACCOUNT_ID)}, token var="
                      f"{bool(config.CLOUDFLARE_API_TOKEN)})")
            if len(paths) < count and config.IMAGE_FALLBACK_POLLINATIONS:
                need = count - len(paths)
                print(f"  Cloudflare sonuç vermedi → Pollinations.ai fallback ({need} aday)...")
                paths += self._generate_pollinations(scene, need)
            if (len(paths) < count and config.IMAGE_FALLBACK_GEMINI
                    and config.GEMINI_API_KEY and not self._quota_limited):
                need = count - len(paths)
                print(f"  → Gemini görsel fallback ({need} aday)...")
                paths += self._generate_gemini(scene, need, max_retries)
            return paths
        paths: List[str] = []
        if not self._quota_limited:
            paths = self._generate_gemini(scene, count, max_retries)
        if self._quota_limited and len(paths) < count:
            if self._cloudflare_available():
                need = count - len(paths)
                print(f"  Gemini kotası dolu → Cloudflare Workers AI fallback ({need} aday)...")
                paths += self._generate_cloudflare(scene, need)
            if len(paths) < count and config.IMAGE_FALLBACK_POLLINATIONS:
                need = count - len(paths)
                print(f"  Gemini kotası dolu → Pollinations.ai fallback ({need} aday)...")
                paths += self._generate_pollinations(scene, need)
        return paths

    def _generate_gemini(self, scene: Scene, count: int, max_retries: int) -> List[str]:
        """Gemini generation with in-list model failover on quota errors."""
        paths: List[str] = []
        attempts = 0
        max_attempts = max_retries * count
        while (len(paths) < count and attempts < max_attempts
               and not self._quota_limited):
            attempts += 1
            try:
                response = self.client.models.generate_content(
                    model=self.model_names[self._model_idx],
                    contents=self.build_prompt(scene),
                    config=genai_types.GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"],
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - API errors vary by type
                if self._is_quota_error(exc):
                    self._model_idx += 1
                    if self._model_idx >= len(self.model_names):
                        self._quota_limited = True
                        print("  ✗ Gemini görsel kotası doldu (429 RESOURCE_EXHAUSTED) — "
                              "tüm modeller tükendi.")
                        return paths
                    nxt = self.model_names[self._model_idx]
                    print(f"  Kota doldu → sonraki model deneniyor: {nxt}")
                    attempts -= 1  # quota failover must not consume the budget
                    continue
                print(f"  Image generation error (attempt {attempts}): {exc}")
                time.sleep(2)
                continue

            image = self._extract_image_bytes(response)
            if image is None:
                print(f"  No image in model response (attempt {attempts})")
                continue

            data, mime_type = image
            accepted = self._accept_image(scene, data, mime_type,
                                          index=len(paths) + 1,
                                          tag="Generated")
            if accepted is None:
                continue
            paths.append(accepted)

        return paths

    # ------------------------------------------------------- pollinations fb
    def _build_pollinations_prompt(self, scene: Scene) -> str:
        """English-oriented prompt (Pollinations models prefer English)."""
        return (
            f"{scene.search_query}. Cinematic vertical 9:16 composition, "
            "high detail, no text, no watermark"
        )

    def _generate_pollinations(self, scene: Scene, count: int) -> List[str]:
        """Keyless fallback: pollinations.ai GET image API (no account)."""
        paths: List[str] = []
        prompt = urllib.parse.quote(self._build_pollinations_prompt(scene))
        for _ in range(count):
            seed = random.randint(1, 2 ** 31)
            url = (
                f"https://image.pollinations.ai/prompt/{prompt}"
                f"?width={config.IMAGE_WIDTH}&height={config.IMAGE_HEIGHT}"
                f"&nologo=true&seed={seed}"
            )
            for attempt in range(2):
                try:
                    req = urllib.request.Request(
                        url, headers={"User-Agent": "appShorts/1.0"})
                    with urllib.request.urlopen(
                            req, timeout=config.POLLINATIONS_TIMEOUT) as resp:
                        data = resp.read()
                    if len(data) < 5_000:
                        raise RuntimeError(f"tiny response ({len(data)} bytes)")
                    accepted = self._accept_image(scene, data, "image/jpeg",
                                                  index=len(paths) + 1,
                                                  tag="Pollinations")
                    if accepted is None:
                        break  # blank rejected: don't retry same seed twice
                    paths.append(accepted)
                    break
                except Exception as exc:  # noqa: BLE001 - urllib errors vary
                    if attempt == 0:
                        time.sleep(3)
                        continue
                    print(f"  Pollinations error: {exc}")
        return paths

    # -------------------------------------------------------- cloudflare fb
    def _cloudflare_available(self) -> bool:
        """Cloudflare works with a CLI OAuth session or REST credentials."""
        return (bool(config.IMAGE_FALLBACK_CLOUDFLARE)
                and (self.cloudflare_cli.available()
                     or (bool(config.CLOUDFLARE_ACCOUNT_ID)
                         and bool(config.CLOUDFLARE_API_TOKEN))))

    @staticmethod
    def _decode_cloudflare_json_image(raw: bytes) -> tuple[bytes, str]:
        """Decode a Workers AI image response into image bytes."""
        return decode_image_response(raw)

    def _generate_cloudflare_cli(self, scene: Scene, count: int) -> List[str]:
        """Generate images with an authenticated ``cf`` CLI session."""
        paths: List[str] = []
        base_prompt = self._build_pollinations_prompt(scene)
        model = self.cloudflare_model
        # Older .env values may point to a deprecated/no-response model.
        if model == "@cf/bytedance/stable-diffusion-xl-lightning":
            model = "@cf/black-forest-labs/flux-1-schnell"
        for _ in range(count):
            data, mime = None, "image/png"
            for retry in range(2):
                prompt = (base_prompt if retry == 0 else
                          f"{base_prompt}, seed variation {random.randint(1, 999999)}")
                try:
                    raw = self.cloudflare_cli.run_ai(
                        model,
                        build_image_payload(
                            model, prompt, config.CLOUDFLARE_IMAGE_STEPS),
                    )
                    if raw.lstrip().startswith(b"{"):
                        data, mime = self._decode_cloudflare_json_image(raw)
                    else:
                        data = raw
                    if self._is_blank_image(data):
                        print("  Cloudflare CLI: boş resim reddedildi (yeniden deneniyor)")
                        data = None
                        continue
                    if len(data) < 5_000:
                        raise RuntimeError(f"tiny response ({len(data)} bytes)")
                    break
                except Exception as exc:  # noqa: BLE001 - CLI errors vary
                    print(f"  Cloudflare CLI error: {exc}")
                    data = None
                    break
            if not data:
                continue
            saved = self._save_image(scene, data, mime, index=len(paths) + 1)
            paths.append(str(saved))
            print(f"  Cloudflare CLI image: {saved.name}")
        return paths

    def _generate_cloudflare(self, scene: Scene, count: int) -> List[str]:
        """Primary provider: Cloudflare Workers AI text-to-image REST API.

        POSTs to .../accounts/{id}/ai/run/{model} with a Bearer token.
        The response is a binary image (usual) or JSON holding base64.
        Blank images are rejected and retried once with a fresh prompt seed.
        Returns saved paths (may be empty on error); never raises.
        """
        paths: List[str] = []
        if not self._cloudflare_available():
            return paths
        if self.cloudflare_cli.available():
            return self._generate_cloudflare_cli(scene, count)
        base_prompt = self._build_pollinations_prompt(scene)
        model = self.cloudflare_model
        if model == "@cf/bytedance/stable-diffusion-xl-lightning":
            model = "@cf/black-forest-labs/flux-1-schnell"
        url = (f"https://api.cloudflare.com/client/v4/accounts/"
               f"{config.CLOUDFLARE_ACCOUNT_ID}/ai/run/{model}")
        auth_failed = False
        for _ in range(count):
            if auth_failed:
                break
            data, mime = None, "image/png"
            for retry in range(2):  # blank rejected -> 1 retry, new seed
                prompt = base_prompt if retry == 0 else f"{base_prompt}, seed variation {random.randint(1, 999999)}"
                body = json.dumps(build_image_payload(
                    model, prompt, config.CLOUDFLARE_IMAGE_STEPS)).encode("utf-8")
                try:
                    req = urllib.request.Request(
                        url, data=body,
                        headers={
                            "Authorization": f"Bearer {config.CLOUDFLARE_API_TOKEN}",
                            "Content-Type": "application/json",
                            "User-Agent": "appShorts/1.0",
                        })
                    with urllib.request.urlopen(
                            req, timeout=config.CLOUDFLARE_TIMEOUT) as resp:
                        ctype = resp.headers.get("Content-Type", "")
                        raw = resp.read()
                    if "json" in ctype:
                        data, mime = self._decode_cloudflare_json_image(raw)
                    else:
                        data = raw
                        mime = "image/png" if "png" in ctype else "image/jpeg"
                    if self._is_blank_image(data):
                        print("  Cloudflare: boş resim reddedildi (yeniden deneniyor)")
                        data = None
                        continue
                    if len(data) < 5_000:
                        raise RuntimeError(f"tiny response ({len(data)} bytes)")
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code in (401, 403):
                        print("  Cloudflare: yetkilendirme reddedildi (HTTP "
                              f"{exc.code}) → API token geçersiz/süresi dolmuş. "
                              "Cloudflare Dashboard > API Tokens bölümünden "
                              "yeni token üretip .env'e yazın (bu sahne "
                              "fallback sağlayıcıya geçiyor).")
                        auth_failed = True
                        data = None
                        break
                    print(f"  Cloudflare error: {exc}")
                    data = None
                    break
                except Exception as exc:  # noqa: BLE001 - urllib errors vary
                    print(f"  Cloudflare error: {exc}")
                    data = None
                    break
            if not data:
                continue
            saved = self._save_image(scene, data, mime,
                                     index=len(paths) + 1)
            paths.append(str(saved))
            print(f"  Cloudflare image: {saved.name}")
        return paths

    # ----------------------------------------------------- gradient (offline)
    def _generate_gradient(self, scene: Scene) -> str:
        """Offline last resort: deterministic two-color vertical gradient.

        Replaces the plain black background when every provider failed, so
        the video still has a colored, topic-seeded backdrop.
        """
        import colorsys
        from PIL import Image

        digest = hashlib.md5(scene.search_query.encode("utf-8")).digest()
        top = colorsys.hsv_to_rgb(digest[0] / 255.0, 0.55, 0.35)
        bottom = colorsys.hsv_to_rgb(digest[1] / 255.0, 0.75, 0.85)
        h, w = config.IMAGE_HEIGHT, config.IMAGE_WIDTH
        strip = Image.new("RGB", (1, h))
        for y in range(h):
            t = y / max(h - 1, 1)
            strip.putpixel((0, y), tuple(
                int(a + (b - a) * t) for a, b in zip(top, bottom)))
        img = strip.resize((w, h))
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        path = self.output_dir / f"scene_{scene.id:02d}_{stamp}_gradient.png"
        img.save(path)
        print(f"  Procedural gradient: {path.name}")
        return str(path)

    # ---------------------------------------------------------------- approval
    @staticmethod
    def _open_in_viewer(path: str) -> bool:
        """Try to open the image in the platform's default viewer."""
        for cmd in (["termux-open", str(path)], ["xdg-open", str(path)], ["open", str(path)]):
            try:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except (FileNotFoundError, OSError):
                continue
        return False

    def approve_candidates(self, scene: Scene, candidates: List[str],
                           interactive: bool = True) -> Optional[str]:
        """Interactive approval: returns approved path, None = skip (black bg).

        Raises KeyboardInterrupt when the user cancels with 'q'.
        """
        if not candidates:
            return None
        if not interactive:
            print(f"  Selected image: {Path(candidates[0]).name} (auto-approved)")
            return candidates[0]

        idx = 0
        extra_rounds = 0
        while True:
            path = candidates[idx]
            print(f"\n  Scene {scene.id} — candidate {idx + 1}/{len(candidates)}: {path}")
            self._open_in_viewer(path)
            try:
                choice = input(
                    "    [Enter] Onayla  ·  [d] Değiştir  ·  [s] Atla (siyah zemin)  ·  [q] İptal > "
                ).strip().lower()
            except EOFError:
                # Non-interactive stdin (piped input): accept what is on screen
                return path

            if choice in ("", "o", "onayla"):
                return path
            if choice in ("q", "iptal"):
                raise KeyboardInterrupt("Kullanıcı tarafından iptal edildi.")
            if choice in ("s", "atla"):
                print("  Scene skipped → black background.")
                return None
            if choice in ("d", "degistir", "değiştir"):
                idx += 1
                if idx >= len(candidates):
                    if extra_rounds < 2 and not self._quota_limited:
                        extra_rounds += 1
                        print("  All candidates shown — generating a new batch...")
                        fresh = self.generate_candidates(scene)
                        if fresh:
                            candidates = fresh
                            scene.image_candidates = fresh
                            idx = 0
                            continue
                    idx = 0  # cycle back through existing candidates
                continue
            print("  Unrecognized key — Enter/d/s/q use.")

    # ------------------------------------------------------------------- main
    def generate_all_images(self, scenes: List[Scene], interactive: bool = False) -> List[Scene]:
        """Generate images for all scenes with optional per-scene approval."""
        total = len(scenes)
        for i, scene in enumerate(scenes):
            print(f"\n[{i + 1}/{total}] Generating images for scene {scene.id}: {scene.search_query}")
            candidates = self.generate_candidates(scene)
            scene.image_candidates = candidates

            if not candidates:
                # Last resort: colored procedural background instead of black
                gradient = self._generate_gradient(scene)
                scene.image_path = gradient
                if self._quota_limited:
                    print(f"  Scene {scene.id}: quota exhausted → gradient background")
                else:
                    print(f"  Warning: No image generated for scene {scene.id} "
                          f"→ gradient background")
                continue

            selected = self.approve_candidates(scene, candidates, interactive=interactive)
            scene.image_path = selected or ""
            if selected:
                print(f"  ✓ Scene {scene.id}: {Path(selected).name}")
            else:
                print(f"  Scene {scene.id}: no approved image → black background")

        return scenes


if __name__ == "__main__":
    # Manual smoke test: python -m services.image_generator
    gen = ImageGenerator()
    demo = Scene(id=1, text="test", visual_description="cinematic medieval castle",
                 search_query="medieval castle", duration=6.0)
    print("Prompt:", gen.build_prompt(demo))
    print("Candidates:", gen.generate_candidates(demo, count=1))
