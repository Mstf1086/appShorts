"""Tests for Workers AI model discovery and family adapters."""
import base64
import json
import unittest

from services.cloudflare_model_catalog import (
    TEXT_GENERATION,
    TEXT_TO_IMAGE,
    TEXT_TO_SPEECH,
    build_image_payload,
    build_text_payload,
    build_tts_payload,
    decode_image_response,
    decode_tts_audio,
    extract_text_response,
    filter_free_models,
    model_choices,
)


class TestModelCatalog(unittest.TestCase):
    """Catalog filtering and Gradio choice generation."""

    def _entry(self, name, task, properties=(), description=""):
        return {
            "name": name,
            "task": {"name": task},
            "description": description,
            "properties": [
                {"property_id": prop, "value": "true"}
                for prop in properties
            ],
        }

    def test_paid_models_are_filtered_and_tasks_are_separated(self):
        models = filter_free_models([
            self._entry("@cf/free/text", TEXT_GENERATION),
            self._entry("@cf/paid/text", TEXT_GENERATION,
                        ["require_workers_paid"]),
            self._entry("@cf/free/image", TEXT_TO_IMAGE),
            self._entry("@cf/free/tts", TEXT_TO_SPEECH),
            self._entry("@cf/other", "Object Detection"),
        ])
        self.assertEqual(
            [(model.id, model.task) for model in models],
            [
                ("@cf/free/image", TEXT_TO_IMAGE),
                ("@cf/free/text", TEXT_GENERATION),
                ("@cf/free/tts", TEXT_TO_SPEECH),
            ],
        )

    def test_access_and_lora_models_are_marked(self):
        models = filter_free_models([
            self._entry(
                "@cf/utter-project/eurollm-9b-it", TEXT_GENERATION,
                description="For access, please fill out this form: https://x"),
            self._entry("@cf/google/gemma-2b-it-lora", TEXT_GENERATION,
                        ["lora"]),
        ])
        by_id = {model.id: model for model in models}
        restricted = by_id["@cf/utter-project/eurollm-9b-it"]
        lora = by_id["@cf/google/gemma-2b-it-lora"]
        self.assertTrue(restricted.requires_access_approval)
        self.assertIn("erişim onayı gerekli", restricted.label)
        self.assertTrue(lora.is_lora)
        self.assertIn("LoRA adapter gerekli", lora.label)

    def test_model_choices_include_default(self):
        choices = model_choices(TEXT_TO_SPEECH, "@cf/example/tts")
        self.assertIn(("@cf/example/tts", "@cf/example/tts"), choices)


class TestTextAdapter(unittest.TestCase):
    """Text model inputs and response decoding."""

    def test_messages_payload_includes_json_mode(self):
        schema = {"input": {"oneOf": [{"properties": {
            "messages": {}, "response_format": {}, "temperature": {},
        }}]}}
        payload = build_text_payload("@cf/meta/llama", "topic", schema)
        self.assertIn("messages", payload)
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_responses_api_payload_uses_input(self):
        schema = {"input": {"oneOf": [{"properties": {
            "input": {}, "reasoning": {},
        }}]}}
        payload = build_text_payload("@cf/openai/gpt-oss", "topic", schema)
        self.assertEqual(payload["input"], "topic")
        self.assertEqual(payload["reasoning"], {"effort": "low"})

    def test_prompt_only_payload(self):
        schema = {"input": {"properties": {"prompt": {}}}}
        payload = build_text_payload("@cf/example", "topic", schema)
        self.assertEqual(payload, {"prompt": "topic"})

    def test_specialized_model_is_rejected_explicitly(self):
        with self.assertRaisesRegex(ValueError, "kullanılamaz"):
            build_text_payload("@cf/cloudflare/clef", "topic", {})

    def test_response_shapes_are_decoded(self):
        response = json.dumps({"result": {"response": "hello"}}).encode()
        self.assertEqual(extract_text_response(response), "hello")
        response = json.dumps({"result": {"choices": [
            {"message": {"content": "world"}}
        ]}}).encode()
        self.assertEqual(extract_text_response(response), "world")


class TestMediaAdapters(unittest.TestCase):
    """TTS and image family payloads."""

    def test_melotts_and_aura_payloads(self):
        self.assertEqual(
            build_tts_payload("@cf/myshell-ai/melotts", "hello"),
            {"prompt": "hello", "lang": "en"},
        )
        aura = build_tts_payload("@cf/deepgram/aura-2-en", "hello")
        self.assertEqual(aura["text"], "hello")
        self.assertEqual(aura["encoding"], "linear16")
        self.assertEqual(aura["container"], "wav")
        self.assertEqual(aura["speaker"], "luna")

    def test_binary_and_base64_audio_are_decoded(self):
        self.assertEqual(decode_tts_audio(b"RIFF1234"), b"RIFF1234")
        encoded = base64.b64encode(b"RIFF1234").decode()
        response = json.dumps({"result": {"audio": encoded}}).encode()
        self.assertEqual(decode_tts_audio(response), b"RIFF1234")

    def test_flux_2_uses_multipart_and_json_models_use_steps(self):
        payload = build_image_payload(
            "@cf/black-forest-labs/flux-2-klein-4b", "prompt", 4)
        self.assertIn("multipart", payload)
        self.assertEqual(payload["multipart"]["contentType"], "application/json")
        payload = build_image_payload(
            "@cf/leonardo/lucid-origin", "prompt", 4)
        self.assertEqual(payload["num_steps"], 4)
        payload = build_image_payload(
            "@cf/black-forest-labs/flux-1-schnell", "prompt", 4)
        self.assertEqual(payload["steps"], 4)

    def test_json_image_response_is_decoded(self):
        encoded = base64.b64encode(b"PNGDATA").decode()
        response = json.dumps({"result": {"image": encoded}}).encode()
        self.assertEqual(decode_image_response(response), (b"PNGDATA", "image/png"))


if __name__ == "__main__":
    unittest.main()
