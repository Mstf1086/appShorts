"""Workers AI model discovery and request-family adapters.

The catalog is fetched through the authenticated ``cf`` CLI. Only models that
do not require Workers Paid are exposed. A small built-in fallback keeps the
web UI usable when the CLI is temporarily unavailable.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Optional

from services.cloudflare_cli import CloudflareCLI


TEXT_GENERATION = "Text Generation"
TEXT_TO_IMAGE = "Text-to-Image"
TEXT_TO_SPEECH = "Text-to-Speech"

ACCESS_APPROVAL_MODELS = {
    "@cf/utter-project/eurollm-9b-it",
    "@cf/swiss-ai/apertus-v1.5-8b",
}

FALLBACK_MODELS = {
    TEXT_GENERATION: (
        "@cf/meta/llama-3.2-3b-instruct",
        "@cf/openai/gpt-oss-20b",
    ),
    TEXT_TO_IMAGE: ("@cf/black-forest-labs/flux-1-schnell",),
    TEXT_TO_SPEECH: ("@cf/myshell-ai/melotts",),
}


@dataclass(frozen=True)
class ModelInfo:
    """One selectable Workers AI model and its catalog metadata."""

    id: str
    task: str
    description: str = ""
    properties: tuple[str, ...] = ()
    requires_access_approval: bool = False
    is_lora: bool = False

    @property
    def label(self) -> str:
        """Return a UI label that clearly marks restricted models."""
        markers = []
        if self.requires_access_approval:
            markers.append("erişim onayı gerekli")
        if self.is_lora:
            markers.append("LoRA adapter gerekli")
        suffix = f" — {', '.join(markers)}" if markers else ""
        return f"{self.id}{suffix}"


def _properties(model: dict) -> tuple[str, ...]:
    """Extract property names from a Cloudflare catalog entry."""
    return tuple(
        str(item.get("property_id", ""))
        for item in model.get("properties", [])
        if item.get("property_id")
    )


def _requires_access_approval(model_id: str, description: str) -> bool:
    """Detect models whose description contains an access form."""
    text = description.lower()
    return model_id in ACCESS_APPROVAL_MODELS or (
        "fill out this form" in text or "access request" in text
    )


def filter_free_models(catalog: Iterable[dict]) -> list[ModelInfo]:
    """Return selectable catalog entries, excluding Workers Paid models."""
    result: list[ModelInfo] = []
    for entry in catalog:
        task = entry.get("task", {})
        task_name = str(task.get("name", "") if isinstance(task, dict) else task)
        if task_name not in {TEXT_GENERATION, TEXT_TO_IMAGE, TEXT_TO_SPEECH}:
            continue
        properties = _properties(entry)
        if "require_workers_paid" in properties:
            continue
        model_id = str(entry.get("name", ""))
        if not model_id:
            continue
        description = str(entry.get("description", "")).strip()
        result.append(ModelInfo(
            id=model_id,
            task=task_name,
            description=description,
            properties=properties,
            requires_access_approval=_requires_access_approval(model_id, description),
            is_lora="lora" in properties,
        ))
    return sorted(result, key=lambda item: item.id.lower())


@lru_cache(maxsize=1)
def load_model_catalog() -> tuple[ModelInfo, ...]:
    """Load the live catalog once, with a safe fallback on CLI failure."""
    cli = CloudflareCLI()
    try:
        models = filter_free_models(cli.list_models())
    except Exception as exc:  # noqa: BLE001 - CLI/network failures vary
        print(f"Workers AI katalogu alınamadı ({exc}); varsayılan modeller kullanılıyor.")
        models = []
    if not models:
        models = [
            ModelInfo(id=model_id, task=task)
            for task, model_ids in FALLBACK_MODELS.items()
            for model_id in model_ids
        ]
    return tuple(models)


def models_for_task(task: str) -> list[ModelInfo]:
    """Return the free-compatible models belonging to one catalog task."""
    return [model for model in load_model_catalog() if model.task == task]


def model_choices(task: str, default: str) -> list[tuple[str, str]]:
    """Return Gradio (label, model_id) choices with the default included."""
    choices = [(model.label, model.id) for model in models_for_task(task)]
    if default and not any(value == default for _, value in choices):
        choices.insert(0, (default, default))
    return choices


def _schema_property_names(schema: dict) -> set[str]:
    """Collect all property names from a nested JSON schema."""
    names: set[str] = set()
    input_schema = schema.get("input", {})
    names.update(input_schema.get("properties", {}).keys())
    for branch in input_schema.get("oneOf", []):
        names.update(branch.get("properties", {}).keys())
        for nested in branch.get("oneOf", []):
            names.update(nested.get("properties", {}).keys())
    return names


def build_text_payload(model: str, prompt: str, schema: Optional[dict] = None) -> dict:
    """Build a text-generation payload for the selected model family."""
    lowered = model.lower()
    if any(name in lowered for name in ("/clef", "/clairic", "llama-guard")):
        raise ValueError(
            f"{model} özel amaçlı bir güvenlik/karar modelidir; "
            "video senaryosu JSON üretmek için kullanılamaz.")
    properties = _schema_property_names(schema or {})
    if "messages" in properties or not properties:
        payload: dict[str, Any] = {
            "messages": [{
                "role": "system",
                "content": "Sen bir YouTube Shorts video senaryo uzmanısın. "
                           "Sadece geçerli JSON formatında cevap ver.",
            }, {"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 2000,
        }
        if "response_format" in properties:
            payload["response_format"] = {"type": "json_object"}
        if "reasoning_effort" in properties:
            payload["reasoning_effort"] = "low"
        elif "reasoning" in properties:
            payload["reasoning"] = {"effort": "low"}
        return payload
    if "input" in properties:
        payload = {"input": prompt}
        if "reasoning" in properties:
            payload["reasoning"] = {"effort": "low"}
        return payload
    return {"prompt": prompt}


def extract_text_response(raw: bytes) -> str:
    """Extract generated text from the common Workers AI response shapes."""
    payload = json.loads(raw.decode("utf-8", errors="replace"))
    result = payload.get("result", payload)
    if isinstance(result, str):
        return result
    if not isinstance(result, dict):
        return str(result)
    if result.get("response") is not None:
        response = result["response"]
        return response if isinstance(response, str) else json.dumps(response)
    choices = result.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message", {})
            if isinstance(message, dict) and message.get("content") is not None:
                content = message["content"]
                return content if isinstance(content, str) else json.dumps(content)
            if choice.get("text") is not None:
                return str(choice["text"])
    if result.get("output_text") is not None:
        return str(result["output_text"])
    raise ValueError(f"Workers AI yanıtında metin bulunamadı: {list(result.keys())}")


def build_tts_payload(model: str, text: str) -> dict:
    """Build a TTS payload for MeloTTS or Deepgram Aura."""
    if "melotts" in model.lower():
        return {"prompt": text, "lang": "en"}
    payload: dict[str, Any] = {
        "text": text,
        "encoding": "linear16",
        "container": "wav",
    }
    if "aura-2-es" in model.lower():
        payload["speaker"] = "aquila"
    elif "aura-2" in model.lower():
        payload["speaker"] = "luna"
    else:
        payload["speaker"] = "angus"
    return payload


def decode_tts_audio(raw: bytes) -> bytes:
    """Decode base64 JSON or binary Workers AI TTS output."""
    if not raw.startswith(b"{"):
        return raw
    payload = json.loads(raw.decode("utf-8", errors="replace"))
    result = payload.get("result", payload)
    if isinstance(result, dict):
        audio = result.get("audio") or result.get("audio_base64")
        if audio:
            return base64.b64decode(audio)
    raise ValueError("Workers AI TTS yanıtında ses bulunamadı")


def build_image_payload(model: str, prompt: str, steps: int = 4) -> dict:
    """Build a text-to-image payload, including Flux 2 multipart requests."""
    lowered = model.lower()
    if "flux-2" in lowered:
        return {
            "multipart": {
                "body": {
                    "prompt": prompt,
                    "width": 1080,
                    "height": 1920,
                },
                "contentType": "application/json",
            },
        }
    payload: dict[str, Any] = {"prompt": prompt}
    if "leonardo" in lowered or "sdxl" in lowered or "stable-diffusion" in lowered:
        payload["num_steps"] = steps
    else:
        payload["steps"] = steps
    return payload


def decode_image_response(raw: bytes) -> tuple[bytes, str]:
    """Decode JSON base64 or binary Workers AI image output."""
    if not raw.lstrip().startswith(b"{"):
        return raw, "image/png"
    payload = json.loads(raw.decode("utf-8", errors="replace"))
    result = payload.get("result", payload)
    if isinstance(result, dict):
        encoded = result.get("image")
        if encoded:
            return base64.b64decode(encoded), "image/png"
    raise ValueError("Workers AI görsel yanıtında image bulunamadı")
