import sys
from typing import List, Optional
from pathlib import Path

from services.scene_generator import SceneGenerator, Scene
from services.image_generator import ImageGenerator
from services.audio_factory import get_voice_generator
from services.subtitle_sync import SubtitleSync
from services.video_assembler import VideoAssembler

import config


class ShortsPipeline:
    def __init__(self, voice_service: str = None, text_model: str = None,
                 image_model: str = None, tts_model: str = None,
                 text_provider: str = None, image_provider: str = None):
        self.scene_generator = SceneGenerator(
            model=text_model, provider=text_provider)
        self.image_generator = ImageGenerator(
            cloudflare_model=image_model, provider=image_provider)
        self.voice_generator = get_voice_generator(
            voice_service, model=tts_model)
        self.subtitle_sync = SubtitleSync()
        self.video_assembler = VideoAssembler()
        self.scenes: List[Scene] = []

    def run(self, topic: str, voice_service: str = None, interactive: Optional[bool] = None) -> str:
        """Run the complete pipeline to generate a YouTube Shorts video.

        interactive: image approval mode for step 2. None = auto-detect
        (interactive only when APPROVE_IMAGES is on and stdin is a TTY).
        """
        if interactive is None:
            interactive = config.APPROVE_IMAGES and sys.stdin.isatty()

        print(f"\n{'='*60}")
        print(f"YouTube Shorts Generator")
        print(f"Topic: {topic}")
        print(f"{'='*60}\n")

        # Step 1: Generate scenes
        print("[1/5] Generating scenes...")
        self.scenes = self.scene_generator.generate_scenes(topic)
        self.scene_generator.validate_scenes(self.scenes)
        print(f"  ✓ Generated {len(self.scenes)} scenes")

        # Step 2: Generate scene images (with optional interactive approval)
        print("\n[2/5] Generating scene images...")
        if interactive:
            print("  Approval: [Enter]=approve · [d]=change · [s]=skip · [q]=cancel")
        self.scenes = self.image_generator.generate_all_images(
            self.scenes, interactive=interactive
        )
        print(f"  ✓ Images prepared for all scenes")

        # Step 3: Generate voiceovers
        print("\n[3/5] Generating voiceovers...")
        self.scenes = self.voice_generator.generate_all_voiceovers(self.scenes)
        print(f"  ✓ Voiceovers generated for all scenes")

        # Step 4: Sync subtitles
        print("\n[4/5] Synchronizing subtitles...")
        srt_path = self.subtitle_sync.generate_srt(self.scenes)
        print(f"  ✓ Subtitles synchronized: {srt_path}")

        # Step 5: Assemble video
        print("\n[5/5] Assembling final video...")
        output_path = self.video_assembler.assemble_video(self.scenes)
        print(f"  ✓ Video assembled: {output_path}")

        # Summary
        total_duration = sum(s.duration for s in self.scenes)
        print(f"\n{'='*60}")
        print(f"Video Generation Complete!")
        print(f"  Output: {output_path}")
        print(f"  Duration: {total_duration:.1f}s")
        print(f"  Scenes: {len(self.scenes)}")
        print(f"  Format: {config.VIDEO_WIDTH}x{config.VIDEO_HEIGHT} (9:16)")
        print(f"{'='*60}\n")

        return output_path

    def run_with_progress(self, topic: str, interactive: Optional[bool] = None) -> str:
        """Run pipeline with detailed progress updates."""
        return self.run(topic, interactive=interactive)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.shorts_pipeline <topic>")
        sys.exit(1)

    topic = " ".join(sys.argv[1:])
    pipeline = ShortsPipeline()
    pipeline.run(topic)
