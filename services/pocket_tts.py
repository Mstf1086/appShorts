"""Offline Turkish text-to-speech based on Pocket TTS Turkish.

`pocket-tts-turkish` (CC-BY-4.0) runs a 6-layer Turkish Pocket TTS model on an
ordinary CPU: six built-in voices, five emotion tags and optional voice cloning
from a short recording. Everything runs locally — no API key and no network
access after the first model download.

The engine is imported lazily, so the default edge-tts path keeps working when
the optional dependency is not installed. Audio is written as 16-bit PCM WAV
(24 kHz mono) with the standard library, and no per-word timings are produced,
therefore the classic (non-karaoke) caption path is used.
"""
import time
import wave
from pathlib import Path
from typing import List, Optional

import config
from services.scene_generator import Scene
from services.voice_generator import VoiceGenerator

# Built-in voices of wite-tech/pocket-tts-turkish
VOICES = ("female_1", "female_2", "female_3", "male_1", "male_2", "male_3")
# Local emotion tag -> English alias accepted by the package
EMOTIONS = {
    "mutlu": "happy",
    "üzgün": "sad",
    "kızgın": "angry",
    "şaşkın": "surprised",
    "sakin": "calm",
}


class PocketTTS(VoiceGenerator):
    """Generate Turkish voiceovers offline with Pocket TTS Turkish."""

    def __init__(self, voice: str = None, emotion: str = None,
                 device: str = None, model: str = None,
                 clone_reference: str = None):
        super().__init__()
        self.voice = (
            voice or getattr(config, "POCKET_TTS_VOICE", VOICES[0]) or VOICES[0]
        ).strip()
        self.emotion = self._resolve_emotion(
            emotion if emotion is not None
            else getattr(config, "POCKET_TTS_EMOTION", ""))
        self.device = (
            device if device is not None
            else getattr(config, "POCKET_TTS_DEVICE", "")
        ).strip()
        self.model_name = (
            model if model is not None
            else getattr(config, "POCKET_TTS_MODEL", "")
        ).strip()
        self.clone_reference = (
            clone_reference if clone_reference is not None
            else getattr(config, "POCKET_TTS_VOICE_CLONE", "")
        ).strip()
        self.seed = self._resolve_seed(getattr(config, "POCKET_TTS_SEED", ""))
        self._engine = None
        self._cloned_voice = None
        self._sample_rate = 24000

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _resolve_emotion(emotion: str) -> str:
        """Normalize an emotion tag; unknown values fall back to neutral."""
        value = (emotion or "").strip().lower()
        if not value:
            return ""
        if value in EMOTIONS:
            return value
        for tag, alias in EMOTIONS.items():
            if value == alias:
                return tag
        print(f"  Warning: Unknown Pocket TTS emotion '{emotion}' -> neutral voice")
        return ""

    @staticmethod
    def _resolve_seed(seed) -> Optional[int]:
        """Return a non-negative int seed, or None for random audio."""
        try:
            value = int(str(seed).strip())
        except (TypeError, ValueError):
            return None
        return value if value >= 0 else None

    def available(self) -> bool:
        """True when the optional ``pocket-tts-turkish`` package is installed."""
        try:
            import pocket_tts_turkish  # noqa: F401
        except Exception:
            return False
        return True

    def _load_engine(self):
        """Import the optional engine lazily and build the model once."""
        if self._engine is None:
            from pocket_tts_turkish import TurkishTTS
            kwargs = {"device": self.device} if self.device else {}
            if self.model_name:
                self._engine = TurkishTTS.from_pretrained(self.model_name, **kwargs)
            else:
                self._engine = TurkishTTS.from_pretrained(**kwargs)
            self._sample_rate = int(
                getattr(self._engine, "sample_rate", 24000) or 24000)
            details = [f"voice={self.voice}"]
            if self.emotion:
                details.append(f"emotion={self.emotion}")
            print(f"  [TTS: Pocket TTS Turkish] model ready "
                  f"({', '.join(details)}, {self._sample_rate} Hz)")
        return self._engine

    def _voice_argument(self, engine):
        """Return the voice to synthesize with.

        A reference recording (``POCKET_TTS_VOICE_CLONE``) wins over a built-in
        voice name; it is converted to a voice state once and then cached, so a
        multi-scene run never re-reads the sample.
        """
        if self.clone_reference:
            if self._cloned_voice is None:
                self._cloned_voice = engine.voice_from_file(self.clone_reference)
                print("  [TTS: Pocket TTS Turkish] cloned voice: "
                      f"{Path(self.clone_reference).name}")
            return self._cloned_voice
        return self.voice

    def _synthesize(self, text: str):
        """Synthesize one text and return ``(float32 samples, sample_rate)``."""
        import numpy as np
        engine = self._load_engine()
        arguments = {"voice": self._voice_argument(engine)}
        if self.emotion:
            arguments["emotion"] = self.emotion
        if self.seed is not None:
            arguments["seed"] = self.seed
        audio = engine.generate(text, **arguments)
        return np.asarray(audio, dtype=np.float32), self._sample_rate

    def _save_wav(self, samples, sample_rate: int, output_path: str) -> str:
        """Write float32 mono samples as a 16-bit PCM WAV file (stdlib only)."""
        import numpy as np
        pcm = np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
        with wave.open(output_path, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(int(sample_rate))
            handle.writeframes(pcm.tobytes())
        return output_path

    # ---------------------------------------------------------- worker API
    def _generate_single(self, text: str, output_path: str) -> Optional[str]:
        """Generate one voiceover with the local engine."""
        samples, sample_rate = self._synthesize(text)
        if samples.size == 0:
            print(f"  Warning: Pocket TTS returned no audio: {output_path}")
            return None
        self._save_wav(samples, sample_rate, output_path)
        if Path(output_path).is_file() and Path(output_path).stat().st_size > 0:
            return output_path
        print(f"  Warning: Generated audio file is empty: {output_path}")
        return None

    def generate(self, text: str, output_path: str = None) -> Optional[str]:
        """Generate a single voiceover (parity with ``CloudflareTTS.generate``)."""
        target = output_path or str(self.output_dir / "scene_audio.wav")
        return self._generate_single(text, target)

    def generate_voiceover(self, scene: Scene,
                           max_retries: int = None) -> Optional[str]:
        """Generate the voiceover of one scene with the local engine.

        Retries mirror ``VoiceGenerator``: a failure (for example an interrupted
        model download) is retried with the shared exponential backoff, and the
        real error is printed on the first failure so a broken install stays
        diagnosable instead of turning into "failed 5 times".
        """
        try:
            max_retries = int(
                max_retries or getattr(config, "TTS_MAX_RETRIES", 5) or 5)
        except (TypeError, ValueError):
            max_retries = 5
        output_path = str(self.output_dir / f"scene_{scene.id:02d}.wav")
        print(f"  [TTS: Pocket TTS Turkish] Generating voiceover for scene {scene.id}")
        last_error = None
        for attempt in range(max_retries):
            try:
                return self._generate_single(scene.text, output_path)
            except Exception as e:  # noqa: BLE001 - reported after retry policy
                last_error = e
                if attempt == 0:
                    print(f"  Error generating voiceover (Pocket TTS): {e}")
                else:
                    print(f"  Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    try:
                        base_delay = float(
                            getattr(config, "TTS_RETRY_BASE_DELAY", 3.0))
                    except (TypeError, ValueError):
                        base_delay = 3.0
                    time.sleep(base_delay * (2 ** attempt))
        print(f"  ✗ Pocket TTS failed after {max_retries} attempts "
              f"(last error: {last_error})")
        return None

    def generate_all_voiceovers(self, scenes: List[Scene]) -> List[Scene]:
        """Generate voiceovers for all scenes and sync scene durations.

        Reuses ``VoiceGenerator``'s loop: the local engine returns no word
        timings, so scenes get classic (character-proportional) captions while
        the scene duration still follows the real audio length. Set
        ``TTS_SCENE_DELAY=0`` to skip the throttle pause between scenes — a
        local engine does not need it.
        """
        return super().generate_all_voiceovers(scenes)


if __name__ == "__main__":
    generator = PocketTTS()
    print("Pocket TTS Turkish available:", generator.available())
    test_scene = Scene(
        id=1,
        text="Yapay zeka, makinelerin insan benzeri zekâ göstermesini sağlayan bir teknolojidir.",
        visual_description="AI technology visualization",
        search_query="artificial intelligence",
        duration=8.0,
    )
    audio_path = generator.generate_voiceover(test_scene)
    if audio_path:
        print(f"Audio: {audio_path} "
              f"({generator.get_audio_duration(audio_path):.2f}s)")

