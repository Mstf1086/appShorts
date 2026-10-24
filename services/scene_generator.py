import ast
import json
import re
import time
from typing import List, Dict, Any
from dataclasses import dataclass, field, asdict

from google import genai
from google.genai import types

import config
from services.cloudflare_cli import CloudflareCLI
from services.cloudflare_model_catalog import (
    build_text_payload,
    extract_text_response,
)


@dataclass
class Scene:
    id: int
    text: str
    visual_description: str
    search_query: str
    duration: float
    image_path: str = ""
    image_candidates: List[str] = field(default_factory=list)
    audio_path: str = ""
    # Exact word timings from Edge TTS WordBoundary events (seconds,
    # audio-relative): [{"word": str, "start": float, "end": float}, ...]
    word_timings: List[Dict[str, Any]] = field(default_factory=list)


PROMPT_TEMPLATE = """Sen bir YouTube Shorts video senaryo yazarısın. "{topic}" konusunda kısa, ilgi çekici bir video senaryosu yaz.

Kriterler:
- Toplam 5-8 sahne oluşturun
- Her sahne benzersizdir; önceki sahnenin metnini veya anahtar kelimelerini tekrar etme
- Her sahne 6-12 saniye arasında olmalı
- Toplam süre 40-60 saniye arasında olmalı
- Her sahne için kısa, net bir metin yaz (Türkçe)
- Her sahne için görsel açıklama yaz (İngilizce, sahne görselini üretmek için)
- Her sahne için Gemini görsel üretiminde kullanılacak İngilizce anahtar kelimeler belirle
- Görsel dikey format (9:16) için uygun olmalı

Cevabını SADECE şu JSON formatında ver, başka hiçbir şey ekleme:
{{
  "scenes": [
    {{
      "id": 1,
      "text": "Sahne metni (Türkçe)",
      "visual_description": "Visual description in English",
      "search_query": "English keywords for scene image generation",
      "duration": 8.0
    }}
  ]
}}
"""


class SceneGenerator:
    def __init__(self, api_key: str = None, model: str = None,
                 provider: str = None):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.cloudflare_cli = CloudflareCLI(
            command=getattr(config, "CLOUDFLARE_CLI_COMMAND", "cf"),
            profile=getattr(config, "CLOUDFLARE_CLI_PROFILE", ""),
            timeout=getattr(config, "CLOUDFLARE_CLI_TIMEOUT", 120),
        )
        self.use_cloudflare = self.cloudflare_cli.available()
        if provider:
            self.provider = provider.lower()
        elif not self.api_key and self.use_cloudflare:
            self.provider = "cloudflare"
        else:
            self.provider = getattr(config, "TEXT_PROVIDER", "gemini").lower()
        if self.provider == "cloudflare" and not self.use_cloudflare:
            raise ValueError(
                "Cloudflare text provider selected, but the 'cf' CLI session is "
                "not authenticated. Run 'cf auth login'.")
        if self.provider != "cloudflare" and not self.api_key:
            raise ValueError(
                "No AI provider available: log in with 'cf auth login' or "
                "set GEMINI_API_KEY in your .env file.")
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        self.model_name = config.GEMINI_MODEL
        self.cloudflare_model = model or getattr(
            config, "CLOUDFLARE_TEXT_MODEL", "@cf/meta/llama-3.2-3b-instruct")
        if self.cloudflare_model in (
            "@cf/meta/llama-3.1-8b-instruct",
            "@cf/meta/llama-3.1-8b-instruct-v2",
        ):
            self.cloudflare_model = "@cf/meta/llama-3.2-3b-instruct"

    def _extract_json(self, text: str) -> dict:
        """Extract and parse a JSON object from a model response."""
        text = text.strip()
        json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError) as parse_exc:
                raise ValueError(
                    f"Invalid JSON response: {exc}; literal_eval: {parse_exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Model response is not a JSON object")
        return parsed

    def _parse_cloudflare_response(self, raw: bytes) -> str:
        """Extract generated text from a Workers AI CLI response."""
        return extract_text_response(raw)

    def _generate_cloudflare_scenes(self, topic: str, max_retries: int) -> List[Scene]:
        """Generate scenes with a keyless Workers AI text model."""
        prompt = PROMPT_TEMPLATE.format(topic=topic)
        last_error = None
        for attempt in range(max_retries):
            try:
                try:
                    schema = self.cloudflare_cli.get_model_schema(self.cloudflare_model)
                except Exception:
                    schema = None
                payload = build_text_payload(self.cloudflare_model, prompt, schema)
                raw = self.cloudflare_cli.run_ai(self.cloudflare_model, payload)
                content = self._parse_cloudflare_response(raw)
                data = self._extract_json(content)
                if "scenes" not in data:
                    raise ValueError("Response does not contain 'scenes' key")
                scenes = [Scene(
                    id=int(item["id"]),
                    text=str(item["text"]).strip(),
                    visual_description=str(item["visual_description"]).strip(),
                    search_query=str(item["search_query"]).strip(),
                    duration=float(item["duration"]),
                ) for item in data["scenes"]]
                total_duration = sum(s.duration for s in scenes)
                if total_duration < config.MIN_DURATION or total_duration > config.MAX_DURATION:
                    target = (config.MIN_DURATION + config.MAX_DURATION) / 2
                    scale = target / max(total_duration, 0.1)
                    for scene in scenes:
                        scene.duration = round(scene.duration * scale, 1)
                print(f"Generated {len(scenes)} scenes with Cloudflare CLI "
                      f"(total duration: {sum(s.duration for s in scenes):.1f}s)")
                return scenes
            except Exception as exc:  # noqa: BLE001 - provider errors vary
                last_error = exc
                print(f"Cloudflare CLI attempt {attempt + 1}/{max_retries} failed: {exc}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(
            f"Failed to generate scenes with Cloudflare after {max_retries} attempts: {last_error}")

    def generate_scenes(self, topic: str, max_retries: int = 3) -> List[Scene]:
        """Generate 5-8 scenes using the explicitly selected provider."""
        if self.provider == "cloudflare":
            return self._generate_cloudflare_scenes(topic, max_retries)
        prompt = PROMPT_TEMPLATE.format(topic=topic)

        config_obj = types.GenerateContentConfig(
            system_instruction="Sen bir YouTube Shorts video senaryo uzmanısın. Sadece JSON formatında cevap ver.",
            temperature=0.7,
            max_output_tokens=2000,
            response_mime_type="application/json"
        )

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config_obj
                )

                content = response.text
                data = self._extract_json(content)

                if "scenes" not in data:
                    raise ValueError("Response does not contain 'scenes' key")

                scenes = []
                for scene_data in data["scenes"]:
                    scene = Scene(
                        id=int(scene_data["id"]),
                        text=str(scene_data["text"]).strip(),
                        visual_description=str(scene_data["visual_description"]).strip(),
                        search_query=str(scene_data["search_query"]).strip(),
                        duration=float(scene_data["duration"])
                    )
                    scenes.append(scene)

                # Validate total duration
                total_duration = sum(s.duration for s in scenes)
                if total_duration < config.MIN_DURATION or total_duration > config.MAX_DURATION:
                    # Adjust durations proportionally
                    target_duration = (config.MIN_DURATION + config.MAX_DURATION) / 2  # 50 seconds
                    scale_factor = target_duration / total_duration
                    for scene in scenes:
                        scene.duration = round(scene.duration * scale_factor, 1)

                print(f"Generated {len(scenes)} scenes (total duration: {sum(s.duration for s in scenes):.1f}s)")
                return scenes

            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise RuntimeError(f"Failed to generate scenes after {max_retries} attempts: {e}")

    def validate_scenes(self, scenes: List[Scene]) -> bool:
        """Validate that scenes meet the requirements."""
        if len(scenes) < config.MIN_SCENES or len(scenes) > config.MAX_SCENES:
            print(f"Warning: Scene count {len(scenes)} is outside recommended range ({config.MIN_SCENES}-{config.MAX_SCENES})")

        total_duration = sum(s.duration for s in scenes)
        if total_duration < config.MIN_DURATION:
            print(f"Warning: Total duration {total_duration:.1f}s is below minimum ({config.MIN_DURATION}s)")
        elif total_duration > config.MAX_DURATION:
            print(f"Warning: Total duration {total_duration:.1f}s exceeds maximum ({config.MAX_DURATION}s)")

        return True


if __name__ == "__main__":
    # Test the scene generator
    generator = SceneGenerator()
    scenes = generator.generate_scenes("Galaksi kaç yaşında?")
    for scene in scenes:
        print(f"\nScene {scene.id}:")
        print(f"  Text: {scene.text}")
        print(f"  Visual: {scene.visual_description}")
        print(f"  Search: {scene.search_query}")
        print(f"  Duration: {scene.duration}s")
