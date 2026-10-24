import re
from typing import List, Optional, Tuple
from pathlib import Path

import config
from services.scene_generator import Scene
from services.voice_generator import VoiceGenerator

_WORD_CLEAN_RE = re.compile(r"[^\w]", re.UNICODE)


def _normalize_word(word: str) -> str:
    """Normalize a word for alignment (case and punctuation insensitive)."""
    return _WORD_CLEAN_RE.sub("", word or "").lower()


class SubtitleSync:
    def __init__(self):
        self.voice_generator = VoiceGenerator()
        self.max_chars_per_line = 40  # Max characters per subtitle line

    def split_text(self, text: str, max_chars: int = None) -> List[str]:
        """Split long text into multiple lines for subtitles."""
        max_chars = max_chars or self.max_chars_per_line

        if len(text) <= max_chars:
            return [text]

        lines = []
        words = text.split()
        current_line = ""

        for word in words:
            if len(current_line) + len(word) + 1 <= max_chars:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
            else:
                lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines

    def format_time(self, seconds: float) -> str:
        """Format seconds to SRT time format (HH:MM:SS,mmm)."""
        # Round to whole milliseconds (truncation produced off-by-one values
        # such as 2.53 -> 2,529 and broke subtitle/audio alignment checks).
        total_ms = int(round(max(float(seconds or 0.0), 0.0) * 1000))
        hours = total_ms // 3_600_000
        minutes = (total_ms % 3_600_000) // 60_000
        secs = (total_ms % 60_000) // 1000
        millis = total_ms % 1000
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def scene_start_times(self, scenes: List[Scene]) -> List[float]:
        """Absolute start time of every scene in the final video timeline.

        Mirrors VideoAssembler.assemble_video(): scene clips are concatenated
        with a SCENE_TRANSITION second overlap, so scene i starts at
        `sum(duration of scenes before i) - i * transition`. Using the same
        formula keeps the SRT frame-accurate against the rendered video
        (previously the offsets only lined up by accident, because the audio
        padding happened to equal the crossfade length).
        """
        transition = float(getattr(config, "SCENE_TRANSITION", 0.0) or 0.0)
        starts: List[float] = []
        cursor = 0.0
        for index, scene in enumerate(scenes):
            starts.append(max(cursor - index * transition, 0.0))
            cursor += max(float(scene.duration or 0.0), 0.0)
        return starts

    def _speech_duration(self, scene: Scene) -> float:
        """Duration of the spoken audio of a scene (falls back to scene duration)."""
        if scene.audio_path and Path(scene.audio_path).exists():
            try:
                duration = self.voice_generator.get_audio_duration(scene.audio_path)
            except Exception:
                duration = 0.0
            if duration and duration > 0:
                return float(duration)
        return float(scene.duration or 0.0)

    def _align_words(self, text_words: List[str], timed_words: List[tuple]) -> Optional[List[tuple]]:
        """Align scene text words to TTS word timings (order is preserved)."""
        if not text_words or not timed_words:
            return None
        if len(text_words) == len(timed_words):
            return list(timed_words)
        aligned: List[tuple] = []
        cursor = 0
        for word in text_words:
            target = _normalize_word(word)
            found = None
            for pos in range(cursor, len(timed_words)):
                if _normalize_word(timed_words[pos][0]) == target:
                    found = timed_words[pos]
                    cursor = pos + 1
                    break
            if found is None:
                return None
            aligned.append(found)
        return aligned

    def _group_words(self, words: List[str], max_chars: int) -> List[List[str]]:
        """Group words into subtitle lines without splitting a word."""
        lines: List[List[str]] = []
        current: List[str] = []
        current_len = 0
        for word in words:
            extra = len(word) + (1 if current else 0)
            if current and current_len + extra > max_chars:
                lines.append(current)
                current = [word]
                current_len = len(word)
            else:
                current.append(word)
                current_len += extra
        if current:
            lines.append(current)
        return lines

    def calculate_subtitle_timing(self, scenes: List[Scene]) -> List[dict]:
        """Calculate subtitle timing for all scenes, aligned to the audio.

        When the TTS engine reports word boundaries (Edge TTS WordBoundary) the
        line boundaries are taken from the spoken words themselves: a subtitle
        line starts and ends exactly when its first/last word is spoken.
        Without boundaries the speech duration is distributed proportionally to
        the character count of each line (previous behaviour, kept as
        fallback).
        """
        subtitles = []
        starts = self.scene_start_times(scenes)

        for scene, scene_start in zip(scenes, starts):
            speech = self._speech_duration(scene)
            text_words = scene.text.split()
            timed = [
                (str(w.get("word", "")).strip(),
                 float(w.get("start", 0.0)),
                 float(w.get("end", 0.0)))
                for w in (scene.word_timings or [])
                if str(w.get("word", "")).strip()
            ]
            aligned = self._align_words(text_words, timed) if timed else None

            if aligned:
                cursor = 0
                for group in self._group_words(text_words, self.max_chars_per_line):
                    first = aligned[cursor]
                    last = aligned[cursor + len(group) - 1]
                    cursor += len(group)
                    start = scene_start + first[1]
                    end = scene_start + max(last[2], first[1] + 0.05)
                    if speech > 0:
                        end = min(end, scene_start + speech)
                    end = max(end, start + 0.05)
                    subtitles.append({
                        "index": len(subtitles) + 1,
                        "start": start,
                        "end": end,
                        "text": " ".join(group),
                        "scene_id": scene.id
                    })
            else:
                # Fallback: distribute the speech duration over the lines.
                lines = self.split_text(scene.text)
                time_per_line = speech / max(len(lines), 1)

                for i, line in enumerate(lines):
                    start = scene_start + (i * time_per_line)
                    end = start + time_per_line

                    subtitles.append({
                        "index": len(subtitles) + 1,
                        "start": start,
                        "end": end,
                        "text": line,
                        "scene_id": scene.id
                    })

        return subtitles

    def generate_srt(self, scenes: List[Scene], output_path: str = None) -> str:
        """Generate SRT subtitle file for all scenes."""
        subtitles = self.calculate_subtitle_timing(scenes)

        if not output_path:
            output_path = str(config.TMP_DIR / "subtitles.srt")

        srt_content = ""
        for sub in subtitles:
            srt_content += f"{sub['index']}\n"
            srt_content += f"{self.format_time(sub['start'])} --> {self.format_time(sub['end'])}\n"
            srt_content += f"{sub['text']}\n\n"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        print(f"Generated SRT file: {output_path}")
        return output_path

    def get_subtitle_entries(self, scenes: List[Scene]) -> List[Tuple[float, float, str]]:
        """Get subtitle entries as (start, end, text) tuples for MoviePy."""
        subtitles = self.calculate_subtitle_timing(scenes)
        return [(sub["start"], sub["end"], sub["text"]) for sub in subtitles]


if __name__ == "__main__":
    # Test the subtitle sync
    sync = SubtitleSync()

    test_scenes = [
        Scene(
            id=1,
            text="Yapay zeka, makinelerin insan benzeri zekâ göstermesini sağlayan bir teknolojidir.",
            visual_description="AI technology",
            search_query="artificial intelligence",
            duration=8.0,
            audio_path=""
        ),
        Scene(
            id=2,
            text="Bu teknoloji, bilgisayarların öğrenme ve karar verme yeteneklerini geliştirir.",
            visual_description="Machine learning",
            search_query="machine learning",
            duration=7.0,
            audio_path=""
        )
    ]

    srt_path = sync.generate_srt(test_scenes)
    with open(srt_path, "r", encoding="utf-8") as f:
        print(f.read())
