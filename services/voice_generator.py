import asyncio
import time
import edge_tts
from typing import Dict, List, Optional
from pathlib import Path

import config
from services.scene_generator import Scene


def _run_coro(coro):
    """Run a coroutine from sync code, even inside a running event loop.

    Some embedded runtimes keep a live event loop, where asyncio.run()
    raises RuntimeError. In that case the coroutine is executed on a fresh
    loop in a worker thread instead. Outside a loop the standard
    asyncio.run() path is used unchanged.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class VoiceGenerator:
    def __init__(self, voice_name: str = None):
        self.voice_name = voice_name or config.VOICE_NAME
        self.output_dir = config.AUDIO_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def _generate_single(self, text: str, output_path: str, rate: str = "+0%") -> Optional[str]:
        """Generate a single voiceover using Edge TTS.

        Streams the synthesis so WordBoundary events (exact per-word offsets)
        can be captured; they are written next to the audio as
        ``<name>.words.json`` and returned to the caller for subtitle sync.
        """
        import json as _json
        words: List[Dict] = []
        try:
            communicate = edge_tts.Communicate(
                text, self.voice_name, rate=rate, boundary="WordBoundary")
            audio = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio.extend(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    # offset/duration are in 100-nanosecond ticks
                    start = chunk["offset"] / 1e7
                    end = start + chunk["duration"] / 1e7
                    words.append({
                        "word": chunk["text"],
                        "start": round(start, 3),
                        "end": round(end, 3),
                    })

            if not audio:
                print(f"  Warning: Edge TTS returned no audio: {output_path}")
                return None
            Path(output_path).write_bytes(bytes(audio))
            if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
                if words:
                    words_path = str(Path(output_path).with_suffix(".words.json"))
                    try:
                        Path(words_path).write_text(
                            _json.dumps(words, ensure_ascii=False), encoding="utf-8")
                    except Exception as e:
                        print(f"  Warning: Could not save word timings: {e}")
                return output_path
            else:
                print(f"  Warning: Generated audio file is empty: {output_path}")
                return None
        except Exception as e:
            print(f"  Error generating voiceover: {e}")
            # Preserve the error for the retry policy in
            # ``generate_voiceover``. Returning None here would hide the
            # distinction between a transient Edge failure and a service
            # response that contains no audio at all.
            raise

    def _generate_single_ttsai(self, text: str, output_path: str, model: str = None) -> Optional[str]:
        """Generate a single voiceover using TTS.ai (free tier).

        TTS.ai offers better voice quality; the free tier gives 15,000 chars/day
        with a free account (TTS_AI_API_KEY required). No word timings are
        returned, so the classic (non-karaoke) caption path will be used.
        """
        try:
            from tts_ai import TTSClient
            api_key = model or config.TTS_AI_API_KEY or ""
            client = TTSClient(api_key=api_key) if api_key else TTSClient()
            audio = client.generate(text=text, model=model or config.TTS_AI_MODEL)
            client.save(audio, output_path)
            if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
                return output_path
            print(f"  Warning: Generated audio file is empty: {output_path}")
            return None
        except Exception as e:
            print(f"  Error generating voiceover (TTS.ai): {e}")
            return None

    def generate_voiceover(self, scene: Scene, max_retries: int = None) -> Optional[str]:
        """Generate voiceover for a single scene.

        Uses the TTS backend selected by TTS_PROVIDER:
          edge   -> Edge TTS (returns per-word timings for karaoke captions)
          ttsai  -> tts.ai quality voice (classic captions)
          cloudflare -> Cloudflare Workers AI (classic captions)
          pockettts -> offline Pocket TTS Turkish, delegated to PocketTTS
            (classic captions; local engine owns its own retry loop)

        Edge TTS intermittently returns HTTP 403 (rate-limit/throttle) on the
        WSS connect when scenes are synthesized back-to-back. Retries use
        exponential backoff so the throttle window can pass.
        """
        if max_retries is None:
            try:
                max_retries = int(getattr(config, "TTS_MAX_RETRIES", 5))
            except (TypeError, ValueError):
                max_retries = 5
        try:
            base_delay = float(getattr(config, "TTS_RETRY_BASE_DELAY", 3.0))
        except (TypeError, ValueError):
            base_delay = 3.0
        filename = f"scene_{scene.id:02d}.mp3"
        output_path = str(self.output_dir / filename)

        provider = getattr(config, "TTS_PROVIDER", "edge").lower().strip()

        if provider == "pockettts":
            # Offline Turkish engine (pocket-tts-turkish): PocketTTS owns its own
            # retry loop and writes ``scene_XX.wav``. No word timings are produced,
            # so the classic (character-proportional) caption path is used.
            from services.pocket_tts import PocketTTS
            return PocketTTS().generate_voiceover(scene, max_retries=max_retries)

        if provider == "cloudflare":
            # Delegate to the CloudflareTTS service class for the actual call.
            from services.cloudflare_tts import CloudflareTTS
            cf = CloudflareTTS()
            cf_output = cf.generate(scene.text, output_path)
            if cf_output:
                print(f"  Generated voiceover: {filename}")
                return cf_output
            # Cloudflare Workers AI is currently unreachable; fall through to Edge TTS.
            print(f"  [TTS: Cloudflare Workers AI] failed, falling back to Edge TTS")
            provider = "edge"

        if provider == "ttsai":
            gen_func = self._generate_single_ttsai
            name = "TTS.ai"
        else:
            gen_func = self._generate_single
            name = "Edge TTS"

        print(f"  [TTS: {name}] Generating voiceover for scene {scene.id}")
        # Log retries only when the first attempt already fails — this
        # surfaces the real error instead of 5 identical "attempt X failed"
        # lines, and makes "I couldn't generate sound in 5 tries" diagnosable.
        first_failed = None
        attempts_made = 0
        for attempt in range(max_retries):
            attempts_made = attempt + 1
            try:
                if provider == "edge":
                    # _generate_single is a coroutine; run it with the loop-aware helper.
                    result = _run_coro(gen_func(scene.text, output_path))
                else:
                    result = gen_func(scene.text, output_path)
                if result:
                    print(f"  Generated voiceover: {filename}")
                    return result
                print(f"  Attempt {attempt + 1}/{max_retries} returned no audio, retrying...")
            except Exception as e:
                if attempt == 0:
                    first_failed = e  # save the real error for final message
                print(f"  Attempt {attempt + 1}/{max_retries} failed: {e}")
                # Edge can establish the WebSocket successfully but receive an
                # empty response. Retrying the same endpoint only repeats the
                # failure, so stop this scene instead of producing silent video.
                if provider == "edge" and "no audio was received" in str(e).lower():
                    break

            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"  Waiting {delay:.0f}s before retry...")
                time.sleep(delay)

        if first_failed is not None:
            print(f"  ✗ Could not generate voiceover for scene {scene.id} "
                  f"after {attempts_made} attempt(s): {first_failed}")
            hints = {
                "edge": ("TTS needs no key", "network access to services.google.com"),
                "ttsai": ("set TTS_AI_API_KEY (free 15k chars/day at tts.ai)", "tts.ai API"),
                "cloudflare": ("CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN", "Cloudflare Workers AI"),
                "pockettts": ("'uv pip install pocket-tts-turkish' (ilk koşuda ~440 MB model indirir)",
                              "offline Pocket TTS Turkish"),
            }.get(provider, ("", ""))
            msg = f"  Hints: {' and '.join(h for h in hints if h)}; try again."
            print(msg)
        return None

    @staticmethod
    def _load_word_timings(audio_path: str) -> List[Dict]:
        """Load Edge TTS WordBoundary timings saved next to the audio file."""
        import json as _json
        words_path = str(Path(audio_path).with_suffix(".words.json"))
        try:
            if Path(words_path).is_file():
                data = _json.loads(Path(words_path).read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data
        except Exception as e:
            print(f"  Warning: Could not load word timings: {e}")
        return []

    def generate_all_voiceovers(self, scenes: List[Scene]) -> List[Scene]:
        """Generate voiceovers for all scenes and sync scene durations to audio.

        Edge TTS output length rarely matches the LLM-estimated scene.duration;
        using the estimate cuts speech off at scene transitions. After each
        successful TTS the scene duration becomes the real audio length plus
        a small padding (SCENE_AUDIO_PADDING) so transitions never interrupt.
        """
        failed_scenes: List[int] = []
        for i, scene in enumerate(scenes):
            print(f"\n[{i + 1}/{len(scenes)}] Generating voiceover for scene {scene.id}")
            if i > 0:
                try:
                    scene_delay = float(getattr(config, "TTS_SCENE_DELAY", 1.5))
                except (TypeError, ValueError):
                    scene_delay = 1.5
                time.sleep(scene_delay)
            audio_path = self.generate_voiceover(scene)
            if audio_path:
                scene.audio_path = audio_path
                scene.word_timings = self._load_word_timings(audio_path)
                if scene.word_timings:
                    print(f"  Word timings: {len(scene.word_timings)} words")
                actual = self.get_audio_duration(audio_path)
                if actual and actual > 0:
                    scene.duration = round(actual + getattr(config, "SCENE_AUDIO_PADDING", 0.4), 2)
                    print(f"  Scene {scene.id} duration synced to audio: {scene.duration:.2f}s")
            else:
                failed_scenes.append(scene.id)
                print(f"  ✗ Could not generate voiceover for scene {scene.id}")

        if failed_scenes:
            failed = ", ".join(str(scene_id) for scene_id in failed_scenes)
            raise RuntimeError(
                f"Seslendirme eksik ({failed}); sessiz video üretimini önlemek için "
                "pipeline durduruldu."
            )

        return scenes

    def get_audio_duration(self, audio_path: str) -> float:
        """Get the duration of an audio file in seconds."""
        try:
            from moviepy import AudioFileClip
            with AudioFileClip(audio_path) as clip:
                return clip.duration
        except Exception as e:
            print(f"  Error getting audio duration: {e}")
            return 0.0


if __name__ == "__main__":
    # Test the voice generator
    generator = VoiceGenerator()

    test_scene = Scene(
        id=1,
        text="Yapay zeka, makinelerin insan benzeri zekâ göstermesini sağlayan bir teknolojidir.",
        visual_description="AI technology visualization",
        search_query="artificial intelligence",
        duration=8.0
    )

    audio_path = generator.generate_voiceover(test_scene)
    if audio_path:
        duration = generator.get_audio_duration(audio_path)
        print(f"Audio duration: {duration:.2f}s")
