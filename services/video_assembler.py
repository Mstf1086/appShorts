import os
from typing import List, Optional
from pathlib import Path
from datetime import timedelta

import numpy as np

from moviepy import (
    AudioFileClip,
    TextClip,
    CompositeVideoClip,
    CompositeAudioClip,
    concatenate_videoclips,
    AudioClip,
    ImageClip,
    ColorClip,
)
from moviepy.video.fx.CrossFadeIn import CrossFadeIn
from moviepy.video.fx.CrossFadeOut import CrossFadeOut

import config
from services.scene_generator import Scene
from services.subtitle_sync import SubtitleSync


class VideoAssembler:
    def __init__(self):
        self.subtitle_sync = SubtitleSync()
        self.output_dir = config.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _create_subtitle_clip(self, text: str, start_time: float, end_time: float,
                                duration: float) -> TextClip:
        """Create a TikTok-style subtitle clip with scroll-up animation."""
        # Split text into lines if too long
        lines = self.subtitle_sync.split_text(text)

        # Calculate font size based on number of lines
        font_size = config.SUBTITLE_FONT_SIZE
        if len(lines) > 1:
            font_size = int(font_size * 0.8)

        # Resolve a real font; never hand Pillow a path that does not exist.
        font = config.SUBTITLE_FONT or config.get_subtitle_font() or None
        text_content = "\n".join(lines)

        def _make_text(font_path) -> TextClip:
            return TextClip(
                text=text_content,
                font=font_path,
                font_size=font_size,
                color=config.SUBTITLE_COLOR,
                stroke_color=config.SUBTITLE_STROKE_COLOR,
                stroke_width=config.SUBTITLE_STROKE_WIDTH,
                size=(config.VIDEO_WIDTH - 100, None),
                method="caption",
                text_align="center"
            )

        try:
            # Create the text clip to measure its size. If the resolved font
            # cannot be loaded (missing/corrupt), retry with Pillow's default
            # font so subtitles are never dropped.
            try:
                text_clip = _make_text(font)
            except Exception:
                text_clip = _make_text(None)

            # Get text clip dimensions
            text_w, text_h = text_clip.size

            # Semi-transparent background (MoviePy 2.x: ImageClip has no `color`
            # kwarg — build an RGBA frame whose alpha channel becomes the mask)
            bg_padding = 20
            bg_w = text_w + bg_padding * 2
            bg_h = text_h + bg_padding * 2
            bg_color = config.SUBTITLE_BG_COLOR
            bg_arr = np.zeros((bg_h, bg_w, 4), dtype=np.uint8)
            bg_arr[..., 0] = bg_color[0]
            bg_arr[..., 1] = bg_color[1]
            bg_arr[..., 2] = bg_color[2]
            bg_arr[..., 3] = bg_color[3] if len(bg_color) > 3 else 255
            bg_clip = (
                ImageClip(bg_arr)
                .with_duration(duration)
                .with_start(start_time)
            )

            # TikTok caption: static position (no scroll-up animation — a moving
            # box shimmers on every frame and re-triggers at each caption).
            # NOTE: it must never sit FULLY off-canvas — moviepy 2.x compose_mask
            # crashes on zero-overlap regions ("operands could not be broadcast").
            y_target = int(config.VIDEO_HEIGHT - config.SUBTITLE_Y_OFFSET - text_h)
            y_target = max(0, y_target)

            # Static position (integer y: no sub-pixel resampling shimmer)
            text_clip = text_clip.with_position(("center", y_target))

            # Background follows the text (static too)
            bg_clip = bg_clip.with_position(("center", y_target - bg_padding))

            # Set duration and start time
            text_clip = text_clip.with_duration(duration).with_start(start_time)

            # Return composite of bg + text
            from moviepy import CompositeVideoClip as _CVC
            # We return the text clip and handle bg separately
            # Store bg as a special attribute
            text_clip._bg_clip = bg_clip

            return text_clip

        except Exception as e:
            print(f"  Warning: Could not create subtitle clip: {e}")
            # Return empty clip as fallback
            return TextClip(
                text="",
                font=None,
                font_size=1,
                color="white",
                size=(1, 1),
                method="caption"
            ).with_duration(0.001).with_start(start_time)

    # ------------------------------------------------- karaoke (TikTok) subs
    @staticmethod
    def _measure_text_width(text: str, font_path: str, font_size: int) -> int:
        """Measure rendered text width in pixels with the actual TTF font."""
        from PIL import ImageFont
        try:
            font = ImageFont.truetype(font_path, font_size)
            return int(font.getlength(text))
        except Exception:
            return int(len(text) * font_size * 0.55)

    def _create_karaoke_word_clip(self, group_words: List[str], active_idx: int,
                                   start: float, duration: float, font: str):
        """One karaoke step: whole word group in yellow + active word in white.

        Single-canvas PIL render: box, base text and highlight are drawn onto
        ONE RGBA image, so the white word can never drift out of alignment.
        Static position: NO per-step rise/pop animation — re-triggering it on
        every word made the whole box jump (~14px) at each word change, which
        read as jitter. The box now sits still; only the white highlight
        moves between words.
        """
        from PIL import Image, ImageDraw, ImageFont

        font_size = config.SUBTITLE_FONT_SIZE
        words = list(group_words)

        try:
            pil_font = ImageFont.truetype(font, font_size) if font else ImageFont.load_default()
        except Exception:
            pil_font = ImageFont.load_default()

        stroke_w = config.SUBTITLE_STROKE_WIDTH
        tmp = Image.new("RGBA", (8, 8))
        tmp_draw = ImageDraw.Draw(tmp)

        def _w(t: str) -> float:
            try:
                return tmp_draw.textlength(t, font=pil_font, stroke_width=stroke_w)
            except TypeError:
                return tmp_draw.textlength(t, font=pil_font)

        try:
            asc, desc = pil_font.getmetrics()
            text_h = asc + desc
        except Exception:
            text_h = int(font_size * 1.25)
        try:
            space_w = _w(" ")
        except Exception:
            space_w = font_size * 0.3

        word_widths = [_w(wd) for wd in words]
        full_w = int(sum(word_widths) + space_w * max(len(words) - 1, 0))

        pad_x, pad_y = 34, 34
        box_w = int(full_w + pad_x * 2)
        box_h = max(int(text_h + pad_y * 2), int(font_size * 2.0))
        y_text = (box_h - text_h) // 2

        word_x = []
        x = float(pad_x)
        for i, wd in enumerate(words):
            word_x.append(x)
            x += word_widths[i] + (space_w if i < len(words) - 1 else 0)

        bg_color = config.SUBTITLE_BG_COLOR
        img = Image.new("RGBA", (box_w, box_h),
                        (bg_color[0], bg_color[1], bg_color[2],
                         bg_color[3] if len(bg_color) > 3 else 255))
        draw = ImageDraw.Draw(img)

        def _draw(t: str, tx: float, fill: str):
            try:
                draw.text((tx, y_text), t, font=pil_font, fill=fill,
                          stroke_width=stroke_w,
                          stroke_fill=config.SUBTITLE_STROKE_COLOR)
            except TypeError:
                draw.text((tx, y_text), t, font=pil_font, fill=fill)

        for i, wd in enumerate(words):
            color = (config.SUBTITLE_EMPHASIS_COLOR
                     if i == active_idx else config.SUBTITLE_COLOR)
            _draw(wd, word_x[i], color)

        arr = np.array(img)
        static = ImageClip(arr).with_duration(duration)

        # Integer position: float y (e.g. 1697.3) forces sub-pixel resampling
        # on every frame, which reads as shimmer. Truncate once here.
        y_final = int(config.VIDEO_HEIGHT - config.SUBTITLE_Y_OFFSET - box_h)
        y_final = max(0, y_final)

        word_clip = static.with_position(("center", y_final)).with_start(start)
        return word_clip

    def _create_karaoke_subtitles(self, scene: Scene, start_time: float) -> List:
        """TikTok accordion karaoke subtitles for one scene.

        Prefers exact Edge TTS WordBoundary timings (scene.word_timings) so
        subtitle words match the spoken audio 1:1; falls back to proportional
        estimation inside audio-synced lines when boundaries are unavailable.
        Words are grouped into fixed disjoint groups of SUBTITLE_WORDS_PER_GROUP
        (no sliding window) so no word repeats across steps.
        """
        clips = []
        font = config.SUBTITLE_FONT or config.get_subtitle_font() or None
        group_size = max(1, config.SUBTITLE_WORDS_PER_GROUP)

        # (word, start, end) tuples relative to scene audio start
        if scene.word_timings:
            word_spans = [
                (str(w.get("word", "")).strip(), float(w.get("start", 0.0)),
                 float(w.get("end", 0.0)))
                for w in scene.word_timings
                if str(w.get("word", "")).strip()
            ]
        else:
            word_spans = []
            for line_start, line_end, text in self.subtitle_sync.get_subtitle_entries([scene]):
                words = [w for w in text.split() if w]
                if not words:
                    continue
                line_dur = max(line_end - line_start, 0.001)
                weights = [max(len(w), 1) for w in words]
                total = float(sum(weights))
                t = line_start
                for i, word in enumerate(words):
                    w_dur = line_dur * weights[i] / total
                    word_spans.append((word, t, t + w_dur))
                    t += w_dur

        # Fixed disjoint groups: [0..2], [3..5], ... — zero repetition
        for gi in range(0, len(word_spans), group_size):
            group = word_spans[gi:gi + group_size]
            group_words = [g[0] for g in group]
            for j, (word, ws, we) in enumerate(group):
                w_dur = max(we - ws, 0.12)
                try:
                    clip = self._create_karaoke_word_clip(
                        group_words, active_idx=j,
                        start=start_time + ws,
                        duration=w_dur + 0.04,  # tiny overlap: no flicker
                        font=font,
                    )
                    clips.append(clip)
                except Exception as e:
                    print(f"  Error creating karaoke word '{word}': {e}")
        return clips

    def _create_image_background(self, scene: Scene) -> Optional[ImageClip]:
        """Cover-fit the generated scene image to the frame with a slow zoom.

        The image is resized to cover 1080x1920, center-cropped, then slowly
        zooms in (~6% over the scene duration) for a subtle Ken Burns effect.
        """
        img = ImageClip(scene.image_path)
        w, h = img.size
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid image size for scene {scene.id}: {img.size}")

        # Cover scale: fill the whole frame, then center-crop the overflow
        scale = max(config.VIDEO_WIDTH / w, config.VIDEO_HEIGHT / h)
        cover = img.resized(new_size=(int(round(w * scale)), int(round(h * scale))))
        cover = cover.cropped(
            x_center=cover.size[0] / 2,
            y_center=cover.size[1] / 2,
            width=config.VIDEO_WIDTH,
            height=config.VIDEO_HEIGHT,
        )

        # Slow zoom-in; oversized frames are clipped by the composite canvas
        duration = max(scene.duration, 0.001)

        def zoom_factor(t):
            return 1 + 0.06 * (t / duration)

        zoomed = cover.resized(zoom_factor)
        return (
            zoomed.with_duration(scene.duration)
            .with_position(("center", "center"))
        )

    def _create_scene_clip(self, scene: Scene, start_time: float) -> Optional[CompositeVideoClip]:
        """Create a composite clip for a single scene with image, audio, and subtitles."""
        clips = []

        # Background image (Gemini-generated scene image)
        bg_ok = False
        if scene.image_path and Path(scene.image_path).exists():
            try:
                bg_clip = self._create_image_background(scene).with_start(start_time)
                clips.append(bg_clip)
                bg_ok = True
            except Exception as e:
                print(f"  Error loading image for scene {scene.id}: {e}")
        if not bg_ok:
            # Black background fallback
            clips.append(
                ColorClip(
                    size=(config.VIDEO_WIDTH, config.VIDEO_HEIGHT),
                    color=(0, 0, 0),
                    duration=scene.duration
                ).with_start(start_time)
            )

        # Audio clip
        if scene.audio_path and Path(scene.audio_path).exists():
            try:
                audio_clip = AudioFileClip(scene.audio_path)

                # Trim to match scene duration
                if audio_clip.duration > scene.duration:
                    audio_clip = audio_clip.subclipped(0, scene.duration)

                audio_clip = audio_clip.with_start(start_time)

                # Add audio to video clip
                if clips:
                    clips[0] = clips[0].with_audio(audio_clip)
            except Exception as e:
                print(f"  Error loading audio for scene {scene.id}: {e}")

        # Subtitle clips — TikTok karaoke (accordion) or classic caption
        if config.SUBTITLE_STYLE == "karaoke":
            clips.extend(self._create_karaoke_subtitles(scene, start_time))
        else:
            subtitle_entries = self.subtitle_sync.get_subtitle_entries([scene])
            for start, end, text in subtitle_entries:
                try:
                    sub_clip = self._create_subtitle_clip(text, start, end, end - start)
                    # Semi-transparent background must come BEFORE the text clip,
                    # otherwise it composites on top and dims the text (~50%)
                    if hasattr(sub_clip, '_bg_clip') and sub_clip._bg_clip is not None:
                        clips.append(sub_clip._bg_clip)
                    clips.append(sub_clip)
                except Exception as e:
                    print(f"  Error creating subtitle: {e}")

        if len(clips) > 1:
            return CompositeVideoClip(clips, size=(config.VIDEO_WIDTH, config.VIDEO_HEIGHT))
        elif clips:
            # Single clip (video only)
            clip = clips[0]
            if not hasattr(clip, 'size') or clip.size != (config.VIDEO_WIDTH, config.VIDEO_HEIGHT):
                return CompositeVideoClip([clip], size=(config.VIDEO_WIDTH, config.VIDEO_HEIGHT))
            return clip

        return None

    def assemble_video(self, scenes: List[Scene], output_filename: str = None) -> str:
        """Assemble all scenes into a final video."""
        if not output_filename:
            timestamp = Path("shorts_").name
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"shorts_{timestamp}.mp4"

        output_path = str(self.output_dir / output_filename)

        print(f"\nAssembling video with {len(scenes)} scenes...")

        # Create clips for each scene (start at 0; concatenation handles offsets)
        scene_clips = []
        for i, scene in enumerate(scenes):
            print(f"  Processing scene {i + 1}/{len(scenes)}...")
            clip = self._create_scene_clip(scene, 0)
            if clip:
                scene_clips.append(clip)

        if not scene_clips:
            raise RuntimeError("No scene clips could be created")

        # Calculate total duration
        total_duration = sum(s.duration for s in scenes)
        print(f"  Total duration: {total_duration:.1f}s")

        # Validate duration
        if total_duration < config.MIN_DURATION:
            print(f"  Warning: Duration {total_duration:.1f}s is below minimum ({config.MIN_DURATION}s)")
        elif total_duration > config.MAX_DURATION:
            print(f"  Warning: Duration {total_duration:.1f}s exceeds maximum ({config.MAX_DURATION}s)")

        # Clamp each scene clip to its declared duration
        concat_clips = []
        for scene, clip in zip(scenes, scene_clips):
            if hasattr(clip, 'duration') and clip.duration > scene.duration:
                clip = clip.subclipped(0, scene.duration)
            concat_clips.append(clip)

        if not concat_clips:
            raise RuntimeError("No clips to concatenate")

        # Crossfade transition between scenes (SCENE_TRANSITION seconds).
        # Scenes carry SCENE_AUDIO_PADDING of silent tail after the speech, so
        # the overlap never cuts audio: next scene fades in while previous one
        # fades out over that silent padding.
        trans = config.SCENE_TRANSITION
        if trans > 0 and len(concat_clips) > 1:
            faded = []
            for i, clip in enumerate(concat_clips):
                effects = []
                if i > 0:
                    effects.append(CrossFadeIn(trans))
                if i < len(concat_clips) - 1:
                    effects.append(CrossFadeOut(trans))
                faded.append(clip.with_effects(effects) if effects else clip)
            concat_clips = faded
            # Negative padding = overlap clips by `trans` seconds (compose mode)
            final_clip = concatenate_videoclips(
                concat_clips, method="compose", padding=-trans)
        else:
            # Concatenate clips
            final_clip = concatenate_videoclips(concat_clips, method="compose")

        # Ensure correct size
        if final_clip.size != (config.VIDEO_WIDTH, config.VIDEO_HEIGHT):
            final_clip = final_clip.resized(new_size=(config.VIDEO_WIDTH, config.VIDEO_HEIGHT))

        # Write final video (moviepy's audio temp file goes to .tmp/video)
        print(f"  Rendering final video: {output_filename}")
        temp_dir = config.VIDEO_TMP_DIR
        temp_dir.mkdir(parents=True, exist_ok=True)
        final_clip.write_videofile(
            output_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            # MoviePy 2.x treats this as a directory for its TEMP_MPY artifacts
            temp_audiofile_path=str(temp_dir),
            logger=None
        )

        # Clean up clips
        for clip in concat_clips:
            try:
                clip.close()
            except:
                pass
        final_clip.close()

        print(f"  Video saved: {output_path}")
        return output_path


if __name__ == "__main__":
    # Test the video assembler
    assembler = VideoAssembler()

    test_scenes = [
        Scene(
            id=1,
            text="Yapay zeka, makinelerin insan benzeri zekâ göstermesini sağlayan bir teknolojidir.",
            visual_description="AI technology",
            search_query="artificial intelligence",
            duration=8.0,
            image_path="",
            audio_path=""
        )
    ]

    # This would create a test video with black background
    print("Video assembler initialized successfully")
