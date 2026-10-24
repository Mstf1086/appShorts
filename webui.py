#!/usr/bin/env python3
"""Gradio web UI for appShorts — generate a YouTube Short from a topic in the browser.

Usage:
    .venv/bin/python webui.py                 # config'deki ayarlarla (varsayılan: 0.0.0.0:7860 + share link)
    .venv/bin/python webui.py --no-share      # sadece yerel/ağ erişimi, public link yok
    .venv/bin/python webui.py --host 0.0.0.0 --port 8080

The UI streams the pipeline log (5 steps) live and shows the rendered video plus
the generated scene images. All tunables live in config.py (GRADIO_*).
"""
from __future__ import annotations

import argparse
import io
import queue
import socket
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr

import config
from pipeline.shorts_pipeline import ShortsPipeline
from services.cloudflare_model_catalog import (
    TEXT_GENERATION,
    TEXT_TO_IMAGE,
    TEXT_TO_SPEECH,
    model_choices,
)

# Only one generation at a time: the pipeline writes into shared output/tmp
# directories, so parallel runs would overwrite each other's scenes.
_GENERATION_LOCK = threading.Lock()


class _LogCapture(io.TextIOBase):
    """stdout proxy: forwards text to the console and to a queue line by line."""

    def __init__(self, original: io.TextIOBase, sink: "queue.Queue[Optional[str]]"):
        self._original = original
        self._sink = sink
        self._pending = ""

    def write(self, text: str) -> int:
        try:
            self._original.write(text)
        except Exception:  # pragma: no cover - console may already be closed
            pass
        self._pending += text
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            self._sink.put(line + "\n")
        return len(text)

    def flush(self) -> None:
        try:
            self._original.flush()
        except Exception:  # pragma: no cover
            pass

    def isatty(self) -> bool:
        return False


def _iter_scene_images(limit: int = 12) -> List[Tuple[str, str]]:
    """Return (path, caption) pairs for the most recent scene images."""
    images = sorted(config.IMAGES_DIR.glob("scene_*.png"))
    return [(str(path), path.stem) for path in images[-limit:]]


def list_generated_videos() -> List[Tuple[str, str]]:
    """Return (label, path) pairs for every previously generated video."""
    videos: List[Tuple[str, str]] = []
    try:
        candidates = list(config.OUTPUT_DIR.glob("*.mp4"))
    except OSError:
        return videos
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            stat = path.stat()
        except OSError:
            continue
        stamp = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        size_mb = stat.st_size / 1_048_576
        videos.append((f"{stamp} · {path.stem} · {size_mb:.1f} MB", str(path)))
    return videos


def preview_video(path: Optional[str]) -> Optional[str]:
    """Return the selected archive video path for the preview player."""
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    return str(candidate)


def refresh_video_list() -> Tuple[dict, Optional[str]]:
    """Refresh the archive dropdown and show the newest video.

    Gradio needs an update object (``gr.update``) to replace the choices of
    a Dropdown; returning a plain list would be interpreted as the selected
    value and the server-side choice list would stay stale (which also breaks
    the input validation of the preview event).
    """
    videos = list_generated_videos()
    newest = videos[0][1] if videos else None
    return gr.update(choices=videos, value=newest), newest


def _status_markdown() -> str:
    """Return a short markdown summary of the current configuration."""
    key_state = "✅ tanımlı" if config.GEMINI_API_KEY else "❌ eksik"
    auth_state = "✅ açık" if (config.GRADIO_AUTH_USER and config.GRADIO_AUTH_PASS) else "⚠️ parolasız"
    return (
        f"**Sistem durumu** — Gemini anahtarı: {key_state} · "
        f"Görsel sağlayıcı: `{config.IMAGE_PROVIDER}` · "
        f"TTS: `{config.TTS_PROVIDER}` · "
        f"Format: `{config.VIDEO_WIDTH}x{config.VIDEO_HEIGHT}` "
        f"({config.MIN_DURATION}-{config.MAX_DURATION} sn) · "
        f"Web koruması: {auth_state}"
    )



def stream_generation(topic: str, tts_provider: Optional[str] = None,
                      text_model: Optional[str] = None,
                      image_model: Optional[str] = None,
                      tts_model: Optional[str] = None):
    """Run the pipeline and yield (log_text, video_path, gallery) updates."""
    topic = (topic or "").strip()
    if not topic:
        yield ("❌ Lütfen bir konu başlığı girin.\n", None, [])
        return

    lines: List[str] = []
    sink: "queue.Queue[Optional[str]]" = queue.Queue()
    outcome: Dict[str, object] = {}
    original_stdout = sys.stdout

    def worker() -> None:
        try:
            sys.stdout = _LogCapture(original_stdout, sink)
            if tts_provider:
                config.TTS_PROVIDER = tts_provider
            pipeline_options = {"voice_service": tts_provider}
            if text_model:
                pipeline_options.update(
                    text_model=text_model, text_provider="cloudflare")
            if image_model:
                pipeline_options.update(
                    image_model=image_model, image_provider="cloudflare")
            if tts_model:
                pipeline_options["tts_model"] = tts_model
            pipeline = ShortsPipeline(**pipeline_options)
            outcome["video"] = pipeline.run(topic, interactive=False)
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI log
            outcome["error"] = exc
        finally:
            sys.stdout = original_stdout
            sink.put(None)

    with _GENERATION_LOCK:
        thread = threading.Thread(target=worker, name="shorts-generation", daemon=True)
        thread.start()
        while True:
            try:
                chunk = sink.get(timeout=0.5)
            except queue.Empty:
                if not thread.is_alive():
                    break
                continue
            if chunk is None:
                break
            lines.append(chunk)
            yield ("".join(lines), None, [])
        thread.join(timeout=5)

    error = outcome.get("error")
    video = outcome.get("video")
    if error is not None:
        lines.append(f"\n❌ Hata: {type(error).__name__}: {error}\n")
    elif video:
        lines.append(f"\n✅ Video hazır: {video}\n")
    else:
        lines.append("\n⚠️ Video üretilemedi.\n")
    yield ("".join(lines), video, _iter_scene_images())


def build_ui() -> gr.Blocks:
    """Build the Gradio interface (does not launch it)."""
    with gr.Blocks(title=config.GRADIO_TITLE) as demo:
        gr.Markdown(f"# {config.GRADIO_TITLE}")
        gr.Markdown("Konu gir → 5 adımda (sahne → görsel → ses → altyazı → montaj) dikey video üret.")
        gr.Markdown(_status_markdown())

        with gr.Row():
            topic = gr.Textbox(
                label="Konu",
                placeholder="Örn. Yapay zeka nedir?",
                value="Yapay zeka nedir?",
                scale=3,
            )
            tts = gr.Dropdown(
                choices=["edge", "cloudflare", "ttsai"],
                value=config.TTS_PROVIDER,
                label="Ses motoru",
                scale=1,
            )
            run_btn = gr.Button("Video üret", variant="primary", scale=1)

        with gr.Row():
            text_model = gr.Dropdown(
                choices=model_choices(
                    TEXT_GENERATION, config.CLOUDFLARE_TEXT_MODEL),
                value=config.CLOUDFLARE_TEXT_MODEL,
                label="Workers AI · Text Generation",
                scale=2,
            )
            image_model = gr.Dropdown(
                choices=model_choices(
                    TEXT_TO_IMAGE, config.CLOUDFLARE_IMAGE_MODEL),
                value=config.CLOUDFLARE_IMAGE_MODEL,
                label="Workers AI · Text-to-Image",
                scale=2,
            )
            tts_model = gr.Dropdown(
                choices=model_choices(
                    TEXT_TO_SPEECH, config.CLOUDFLARE_TTS_MODEL),
                value=config.CLOUDFLARE_TTS_MODEL,
                label="Workers AI · Text-to-Speech",
                scale=2,
            )

        gr.Markdown(
            "ℹ️ Workers AI metin ve görsel modelleri seçilen dropdown "
            "modellerini kullanır. Workers AI TTS modeli yalnızca TTS "
            "sağlayıcısı **Cloudflare** seçiliyken geçerlidir.")

        with gr.Row():
            video = gr.Video(label="Üretilen video", height=480)
            with gr.Column():
                log = gr.Textbox(
                    label="Canlı log",
                    lines=18,
                    max_lines=25,
                    interactive=False,
                )

        gallery = gr.Gallery(label="Sahne görselleri", columns=6, height=220)

        with gr.Accordion("Önceki videolar", open=True):
            video_list = gr.Dropdown(
                choices=list_generated_videos(),
                label="Üretilmiş videolar",
                scale=3,
            )
            archive_video = gr.Video(label="Arşiv önizlemesi", height=360)

        run_btn.click(
            fn=stream_generation,
            inputs=[topic, tts, text_model, image_model, tts_model],
            outputs=[log, video, gallery],
        ).then(
            fn=refresh_video_list,
            outputs=[video_list, archive_video],
        )

        video_list.change(
            fn=preview_video,
            inputs=video_list,
            outputs=archive_video,
        )

        demo.load(
            fn=refresh_video_list,
            outputs=[video_list, archive_video],
        )

    return demo


def _resolve_url(demo: "gr.Blocks", app: object) -> str:
    """Pick the best available URL (share link first, then local).

    Only real http(s) strings are accepted — mocked/None attributes are ignored.
    """
    for holder in (demo, app):
        for attr in ("share_url", "local_url"):
            value = getattr(holder, attr, None)
            if isinstance(value, str) and value.startswith("http"):
                return value
    return ""


def _port_is_available(host: str, port: int) -> bool:
    """Return whether *port* can be bound for the web server."""
    bind_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((bind_host, port))
        return True
    except OSError:
        return False


def _find_available_port(host: str, preferred_port: int, max_attempts: int = 100) -> int:
    """Find the first free port at or after *preferred_port*."""
    for port in range(preferred_port, preferred_port + max_attempts):
        if port > 65535:
            break
        if _port_is_available(host, port):
            return port
    raise OSError(
        f"Cannot find an available port in range: "
        f"{preferred_port}-{min(preferred_port + max_attempts - 1, 65535)}."
    )


def launch_ui(
    share: Optional[bool] = None,
    server_name: Optional[str] = None,
    server_port: Optional[int] = None,
):
    """Launch the web UI and print the reachable URLs."""
    demo = build_ui()
    share = config.GRADIO_SHARE if share is None else bool(share)
    server_name = server_name or config.GRADIO_HOST
    explicit_port = server_port is not None
    preferred_port = int(server_port if explicit_port else config.GRADIO_PORT)

    if explicit_port and not _port_is_available(server_name, preferred_port):
        raise OSError(
            f"Port {preferred_port} is already in use. "
            "Choose another port with --port or stop the existing web UI."
        )
    server_port = _find_available_port(server_name, preferred_port)
    if server_port != preferred_port:
        print(f"Port {preferred_port} is busy; using {server_port} instead.")

    auth = None
    if config.GRADIO_AUTH_USER and config.GRADIO_AUTH_PASS:
        auth = (config.GRADIO_AUTH_USER, config.GRADIO_AUTH_PASS)
    elif share:
        print("⚠️ GRADIO_SHARE=true ama GRADIO_AUTH_USER/PASS tanımlı değil — linke herkes erişebilir.")

    demo.queue(max_size=config.GRADIO_MAX_QUEUE, default_concurrency_limit=1)
    app = demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        auth=auth,
        theme=getattr(gr.themes, "Soft", lambda: None)(),
        allowed_paths=[str(config.OUTPUT_DIR), str(config.TMP_DIR)],
        show_error=True,
        quiet=True,
        # Non-blocking: launch() returns so the URL can be printed/saved;
        # the server keeps running in a background thread.
        prevent_thread_lock=True,
    )

    url = _resolve_url(demo, app)
    print("\n" + "=" * 60)
    print(f"  Web arayüzü hazır: {url or f'http://{server_name}:{server_port}'}")
    print(f"  Dinlenen adres   : {server_name}:{server_port}")
    if share:
        print("  Public link       : gradio.live (yukarıdaki adres)")
    print("  Durdurmak için   : Ctrl+C")
    print("=" * 60 + "\n")

    if url:
        try:
            config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            (config.OUTPUT_DIR / "webui_url.txt").write_text(url + "\n", encoding="utf-8")
        except OSError:
            pass
    return app


def main() -> None:
    """CLI entry point for the Gradio web UI."""
    parser = argparse.ArgumentParser(description="appShorts Gradio web arayüzü")
    parser.add_argument("--host", default=None, help="Dinlenecek adres (varsayılan config.GRADIO_HOST)")
    parser.add_argument("--port", type=int, default=None, help="Port (varsayılan config.GRADIO_PORT)")
    share_group = parser.add_mutually_exclusive_group()
    share_group.add_argument("--share", dest="share", action="store_true", help="Gradio public link üret")
    share_group.add_argument("--no-share", dest="share", action="store_false", help="Public link üretme")
    parser.set_defaults(share=None)
    args = parser.parse_args()

    launch_ui(share=args.share, server_name=args.host, server_port=args.port)

    # launch() is non-blocking — keep the process alive until Ctrl+C.
    print("Sunucu çalışıyor. Durdurmak için Ctrl+C")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\nSunucu durduruldu.")


if __name__ == "__main__":
    main()
