"""Factory for voice generation services.

At program startup the user is asked which TTS engine to use (edge-tts or
Cloudflare Workers AI). The choice is stored in `config.TTS_PROVIDER` and the
corresponding service is returned here. This keeps pipeline code agnostic of
the actual TTS backend.
"""
from services.cloudflare_tts import CloudflareTTS
from services.voice_generator import VoiceGenerator
import config


def get_voice_generator(service: str = None, voice_name: str = None,
                        model: str = None) -> "VoiceGenerator | CloudflareTTS":
    """Return a voice generator bound to the chosen TTS backend.

    Service selection order: explicit `service` arg > `config.TTS_PROVIDER` >
    interactive prompt (TTY only, once per run).

    Supported providers: "edge" (edge-tts), "cloudflare" (Workers AI),
    "ttsai" (tts.ai).
    """
    provider = service or config.TTS_PROVIDER
    provider = (provider or "").lower().strip()

    if provider == "cloudflare":
        return CloudflareTTS(
            account_id=None,  # uses config.CLOUDFLARE_ACCOUNT_ID
            api_token=None,   # uses config.CLOUDFLARE_API_TOKEN
            model=model,      # explicit model or config.CLOUDFLARE_TTS_MODEL
        )

    # default: edge-tts
    return VoiceGenerator(voice_name=voice_name)
