"""Project configuration — all tunables live here (see .env.example)."""
import os
import urllib.request
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
# Temporary artifacts (scene images, voiceovers, subtitles, moviepy render
# temp files) live in the project-root ".tmp" folder. Only finished videos are
# written to OUTPUT_DIR.
TMP_DIR = BASE_DIR / ".tmp"
IMAGES_DIR = TMP_DIR / "images"
AUDIO_DIR = TMP_DIR / "audio"
VIDEO_TMP_DIR = TMP_DIR / "video"

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
TMP_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)
VIDEO_TMP_DIR.mkdir(exist_ok=True)

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Video settings
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
MIN_DURATION = 40  # seconds
MAX_DURATION = 60  # seconds

# Scene settings
MIN_SCENES = 5
MAX_SCENES = 8

# Voice settings
VOICE_LANGUAGE = "tr-TR"  # Turkish
VOICE_NAME = "tr-TR-AhmetNeural"  # Male Turkish voice
# Voice provider:
#   "edge"    - edge-tts, free, needs services.google.com, returns word timings
#   "cloudflare" - Workers AI Melotts via cf OAuth or REST token;
#     Turkish is unsupported by this model
#   "ttsai"   - tts.ai, better voice quality, free tier (15k chars/day) needs
#     TTS_AI_API_KEY; no word timings -> classic caption path
#   "pockettts" - offline Türkçe Pocket TTS (pocket-tts-turkish, CPU-friendly)
#     6 ses + 5 duygu etiketi + kısa kayıttan ses klonlama; model indirmesi
#     dışında ağ gerekmez, kelime zamanlaması yok -> klasik altyazı yolu
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "edge")  # "edge" | "cloudflare" | "ttsai" | "pockettts"
# Edge TTS 403/throttle backoff: rapid sequential WSS connects get rate-limited
TTS_MAX_RETRIES = int(os.getenv("TTS_MAX_RETRIES", "5"))
TTS_RETRY_BASE_DELAY = float(os.getenv("TTS_RATE_LIMIT_BASE_DELAY", "3.0"))  # s, exponential: base * 2**attempt
TTS_SCENE_DELAY = float(os.getenv("TTS_SCENE_DELAY", "1.5"))  # s between scenes

# TTS.ai (better quality, free tier: 15,000 chars/day with free account)
TTS_AI_API_KEY = os.getenv("TTS_AI_API_KEY", "")
TTS_AI_MODEL = os.getenv("TTS_AI_MODEL", "kokoro")  # free models: piper | vits | melotts | kokoro

# Cloudflare Workers AI text-to-speech settings
CLOUDFLARE_TTS_MODEL = os.getenv(
    "CLOUDFLARE_TTS_MODEL", "@cf/myshell-ai/melotts")
CLOUDFLARE_TTS_LANGUAGE = os.getenv("CLOUDFLARE_TTS_LANGUAGE", "en")
CLOUDFLARE_TTS_TIMEOUT = int(os.getenv("CLOUDFLARE_TTS_TIMEOUT", "120"))

# Pocket TTS Turkish — offline Türkçe seslendirme (pocket-tts-turkish, CC-BY-4.0)
# İlk çalıştırmada ~440 MB model indirilir, sonrası tamamen çevrimdışıdır.
# Kelime zamanı üretilmediği için klasik (karakter-oranlı) altyazı yolu kullanılır.
POCKET_TTS_VOICE = os.getenv("POCKET_TTS_VOICE", "female_1")  # female_1..3 | male_1..3
POCKET_TTS_EMOTION = os.getenv("POCKET_TTS_EMOTION", "")  # boş = nötr; bilinmeyen değer uyarı verip nötre döner
POCKET_TTS_DEVICE = os.getenv("POCKET_TTS_DEVICE", "")  # boş = auto (GPU varsa GPU); "cpu" | "cuda"
POCKET_TTS_MODEL = os.getenv("POCKET_TTS_MODEL", "")  # HF repo override (varsayılan: wite-tech/pocket-tts-turkish)
POCKET_TTS_VOICE_CLONE = os.getenv("POCKET_TTS_VOICE_CLONE", "")  # 5-10 sn referans kayıt (klonlama)
POCKET_TTS_SEED = os.getenv("POCKET_TTS_SEED", "")  # tamsayı -> aynı metin + tohum = aynı ses

# Subtitle settings - TikTok style (bottom, scroll-up animation)
_SUBTITLE_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
)


# Bundled font (repo asset) — works offline on any machine, incl. font-less hosts
BUNDLED_FONT = BASE_DIR / "assets" / "fonts" / "NotoSans-Bold.ttf"
# Last-resort download sources (first use only, when nothing else is found)
FONT_DOWNLOAD_URLS = (
    "https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSans/NotoSans-Bold.ttf",
    "https://raw.githubusercontent.com/JetBrains/JetBrainsMono/master/fonts/ttf/JetBrainsMono-Bold.ttf",
)


def _resolve_subtitle_font() -> str:
    """Resolve a font file that actually exists in the current environment.

    Priority: SUBTITLE_FONT env override -> known system paths -> matplotlib's
    bundled DejaVuSans-Bold (always present when matplotlib is installed) ->
    first font matplotlib can find -> bundled repo font.
    Returns "" when nothing is found (callers fall back to Pillow's default
    font); never returns a path that does not exist.
    """
    candidates = [os.getenv("SUBTITLE_FONT", "").strip(), *_SUBTITLE_FONT_CANDIDATES]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    try:
        import matplotlib
        bundled = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
        if bundled.is_file():
            return str(bundled)
    except Exception:
        pass
    try:
        from matplotlib import font_manager
        found = font_manager.findfont("DejaVu Sans:bold", fallback_to_default=True)
        if found and Path(found).is_file():
            return found
    except Exception:
        pass
    if BUNDLED_FONT.is_file():
        return str(BUNDLED_FONT)
    return ""


SUBTITLE_FONT = _resolve_subtitle_font()
# None = not tried yet, "" = tried and failed (never retry), str = resolved path
_subtitle_font_cache: "str | None" = SUBTITLE_FONT or None


def get_subtitle_font() -> str:
    """Return a usable font path, downloading one on first use as last resort.

    The download is attempted at most once per process. Returns "" when no
    font can be obtained — the caller then renders text with Pillow's default.
    """
    global _subtitle_font_cache
    if _subtitle_font_cache is None:
        dest = TMP_DIR / "fonts" / "NotoSans-Bold.ttf"
        if not dest.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            for url in FONT_DOWNLOAD_URLS:
                try:
                    urllib.request.urlretrieve(url, dest)
                    if dest.is_file() and dest.stat().st_size > 100_000:
                        break
                except Exception:
                    continue
        if dest.is_file() and dest.stat().st_size > 100_000:
            _subtitle_font_cache = str(dest)
        else:
            _subtitle_font_cache = ""
    return _subtitle_font_cache


# Subtitle style - TikTok karaoke/accordion (yellow on black, word pop)
SUBTITLE_STYLE = os.getenv("SUBTITLE_STYLE", "karaoke")  # karaoke | caption
SUBTITLE_FONT_SIZE = 52
SUBTITLE_COLOR = os.getenv("SUBTITLE_COLOR", "yellow")
SUBTITLE_EMPHASIS_COLOR = "white"  # currently spoken word highlight
SUBTITLE_STROKE_COLOR = "black"
SUBTITLE_STROKE_WIDTH = 4
SUBTITLE_POSITION = "center"  # x position
SUBTITLE_Y_OFFSET = 150  # Pixels from bottom for TikTok style
SUBTITLE_BG_COLOR = (0, 0, 0, 255)  # opaque black box (TikTok style)
SUBTITLE_WORDS_PER_GROUP = int(os.getenv("SUBTITLE_WORDS_PER_GROUP", "3"))

# Scene transitions (crossfade) + audio-scene sync padding
SCENE_TRANSITION = float(os.getenv("SCENE_TRANSITION", "0.4"))  # crossfade s
SCENE_AUDIO_PADDING = 0.4  # extra seconds appended to scene duration after TTS

# Gemini settings
GEMINI_MODEL = "gemini-3.5-flash-lite"
# Text scene provider: "gemini" or "cloudflare". The web UI explicitly passes
# "cloudflare" when a Workers AI text model is selected.
TEXT_PROVIDER = os.getenv("TEXT_PROVIDER", "gemini")

# Scene image generation settings (Cloudflare-only by default)
IMAGE_PROVIDER = os.getenv("IMAGE_PROVIDER", "cloudflare")  # "cloudflare" | "gemini"
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
IMAGE_CANDIDATES = int(os.getenv("IMAGE_CANDIDATES", "1"))  # candidates per scene (1 = one image each)
IMAGE_MAX_RETRIES = 3
# Interactive image approval (Enter=approve, d=change, s=skip) — off by default
APPROVE_IMAGES = os.getenv("APPROVE_IMAGES", "false").lower() == "true"
IMAGE_WIDTH = 1080
IMAGE_HEIGHT = 1920
# Fallback when Gemini quota is exhausted: keyless Pollinations.ai image API
IMAGE_FALLBACK_POLLINATIONS = os.getenv("IMAGE_FALLBACK_POLLINATIONS", "true").lower() == "true"
POLLINATIONS_TIMEOUT = int(os.getenv("POLLINATIONS_TIMEOUT", "90"))
# Fallback: Cloudflare Workers AI text-to-image (needs ACCOUNT_ID + API token;
# call is plain REST — no wrangler needed)
IMAGE_FALLBACK_CLOUDFLARE = os.getenv("IMAGE_FALLBACK_CLOUDFLARE", "true").lower() == "true"
# Cloudflare path (default provider) fallback order: Pollinations (keyless)
# then Gemini; both can be disabled independently.
IMAGE_FALLBACK_GEMINI = os.getenv("IMAGE_FALLBACK_GEMINI", "true").lower() == "true"
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")
CLOUDFLARE_IMAGE_MODEL = os.getenv(
    "CLOUDFLARE_IMAGE_MODEL", "@cf/black-forest-labs/flux-1-schnell")
CLOUDFLARE_IMAGE_STEPS = int(os.getenv("CLOUDFLARE_IMAGE_STEPS", "4"))
CLOUDFLARE_TIMEOUT = int(os.getenv("CLOUDFLARE_TIMEOUT", "90"))
# Cloudflare CLI OAuth session (preferred; no API token required)
CLOUDFLARE_CLI_COMMAND = os.getenv("CLOUDFLARE_CLI_COMMAND", "cf")
CLOUDFLARE_CLI_PROFILE = os.getenv("CLOUDFLARE_CLI_PROFILE", "")
CLOUDFLARE_CLI_TIMEOUT = int(os.getenv("CLOUDFLARE_CLI_TIMEOUT", "120"))
# Workers AI text model used for keyless scene generation
CLOUDFLARE_TEXT_MODEL = os.getenv(
    "CLOUDFLARE_TEXT_MODEL", "@cf/meta/llama-3.2-3b-instruct")

# Gradio web UI settings (webui.py)
# Host "0.0.0.0" -> erişilebilir tüm arayüzlere dinler (dışarıdan erişim).
GRADIO_HOST = os.getenv("GRADIO_HOST", "0.0.0.0")
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))
# GRADIO_SHARE=true -> Gradio public link (*.gradio.live) üretir (dış erişim).
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "true").lower() == "true"
# Basit parola koruması (public link için önerilir; ikisi de dolu olmalı).
GRADIO_AUTH_USER = os.getenv("GRADIO_AUTH_USER", "")
GRADIO_AUTH_PASS = os.getenv("GRADIO_AUTH_PASS", "")
GRADIO_TITLE = os.getenv("GRADIO_TITLE", "appShorts — YouTube Shorts Generator")
GRADIO_MAX_QUEUE = int(os.getenv("GRADIO_MAX_QUEUE", "4"))
