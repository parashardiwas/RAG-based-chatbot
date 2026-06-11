"""
Video processing: extract text from videos with or without audio.

For videos with audio: extracts audio track → Whisper transcription
For videos without audio: extracts key frames → OCR (Tesseract)
"""

import asyncio
import logging
import os
import tempfile
import time
from typing import Any

from fastapi import UploadFile

from app.config import get_settings

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Video processing service.
    
    Handles:
    - Videos with audio → audio extraction + Whisper transcription
    - Videos without audio → keyframe extraction + OCR
    """

    async def process_upload(self, file: UploadFile) -> dict[str, Any]:
        """Process an uploaded video file."""
        start_time = time.time()

        # Save to temp file
        suffix = os.path.splitext(file.filename or "video.mp4")[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            return await self.process_file(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    async def process_file(self, video_path: str) -> dict[str, Any]:
        """
        Process a video file — detect audio, transcribe or OCR.
        
        Returns:
            dict with: text, language, source (audio/ocr), latency_ms
        """
        start_time = time.time()

        # Check if video has audio
        has_audio = await self._has_audio_track(video_path)

        if has_audio:
            logger.info(f"Video has audio track, extracting and transcribing...")
            return await self._process_with_audio(video_path, start_time)
        else:
            logger.info(f"No audio track found, extracting frames for OCR...")
            return await self._process_without_audio(video_path, start_time)

    async def _has_audio_track(self, video_path: str) -> bool:
        """Check if a video file has an audio track using ffprobe."""
        try:
            process = await asyncio.create_subprocess_exec(
                "ffprobe", "-v", "error", "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0", video_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await process.communicate()
            return b"audio" in stdout
        except FileNotFoundError:
            logger.warning("ffprobe not found. Assuming video has audio.")
            return True
        except Exception as e:
            logger.warning(f"ffprobe check failed: {e}. Assuming video has audio.")
            return True

    async def _process_with_audio(
        self, video_path: str, start_time: float
    ) -> dict[str, Any]:
        """Extract audio from video and transcribe."""
        from app.services.media.audio_processor import AudioProcessor

        audio_processor = AudioProcessor()
        audio_path = await audio_processor.extract_audio_from_video(video_path)

        if not audio_path:
            # Failed to extract audio — fall back to OCR
            return await self._process_without_audio(video_path, start_time)

        try:
            result = await audio_processor.transcribe_file(audio_path)
            result["source"] = "audio_transcription"
            result["latency_ms"] = int((time.time() - start_time) * 1000)
            return result
        finally:
            try:
                os.unlink(audio_path)
            except OSError:
                pass

    async def _process_without_audio(
        self, video_path: str, start_time: float
    ) -> dict[str, Any]:
        """Extract key frames from video, deduplicate by image, and run OCR."""
        frames_dir = tempfile.mkdtemp()

        try:
            # Extract I-frames
            await self._extract_keyframes(video_path, frames_dir)

            texts = []
            frame_files = sorted(
                [f for f in os.listdir(frames_dir) if f.endswith(".png")]
            )

            last_unique_frame_path = None

            for frame_file in frame_files:
                frame_path = os.path.join(frames_dir, frame_file)
                
                # Check for image duplication
                if last_unique_frame_path and self._is_duplicate_frame(last_unique_frame_path, frame_path):
                    continue
                
                # It's unique, run OCR
                text = await self._ocr_frame(frame_path)
                last_unique_frame_path = frame_path
                
                if text.strip():
                    texts.append(text.strip())

            # Deduplicate similar consecutive texts (just in case OCR hallucinates differences)
            deduped = self._deduplicate_texts(texts)
            combined_text = "\n".join(deduped)

            latency_ms = int((time.time() - start_time) * 1000)

            return {
                "text": combined_text,
                "language": "en",  # OCR doesn't detect language well
                "confidence": 0.6 if combined_text else 0.0,
                "source": "ocr",
                "frames_processed": len(frame_files),
                "unique_text_blocks": len(deduped),
                "latency_ms": latency_ms,
            }

        finally:
            # Clean up frames
            import shutil
            shutil.rmtree(frames_dir, ignore_errors=True)

    def _is_duplicate_frame(self, path1: str, path2: str, threshold: float = 100.0) -> bool:
        """Calculate MSE between two frames. If below threshold, they are duplicates."""
        try:
            import numpy as np
            from PIL import Image

            # Open, convert to grayscale, and resize to ignore minor noise/artifacts
            img1 = np.array(Image.open(path1).convert('L').resize((128, 128)), dtype=np.float32)
            img2 = np.array(Image.open(path2).convert('L').resize((128, 128)), dtype=np.float32)

            mse = np.mean((img1 - img2) ** 2)
            return mse < threshold
        except Exception as e:
            logger.warning(f"Duplicate frame check failed: {e}")
            return False

    async def _extract_keyframes(self, video_path: str, output_dir: str):
        """Extract structural I-frames from video using ffmpeg."""
        try:
            process = await asyncio.create_subprocess_exec(
                "ffmpeg", "-i", video_path,
                "-vf", "select='eq(pict_type,PICT_TYPE_I)'",  # Select only I-frames
                "-vsync", "vfr",  # Variable framerate output
                "-q:v", "2",     # High quality
                os.path.join(output_dir, "frame_%04d.png"),
                "-y",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()
        except FileNotFoundError:
            logger.error("ffmpeg not found. Cannot extract video frames.")
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")

    async def _ocr_frame(self, frame_path: str) -> str:
        """Run OCR on a single frame image."""
        try:
            import pytesseract
            from PIL import Image

            image = Image.open(frame_path)
            text = await asyncio.to_thread(
                pytesseract.image_to_string, image, lang="eng+hin"
            )
            return text
        except ImportError:
            logger.warning("pytesseract or Pillow not installed")
            return ""
        except Exception as e:
            logger.debug(f"OCR failed for {frame_path}: {e}")
            return ""

    def _deduplicate_texts(self, texts: list[str]) -> list[str]:
        """Remove near-duplicate consecutive OCR texts."""
        if not texts:
            return []

        deduped = [texts[0]]
        for text in texts[1:]:
            # Simple dedup: skip if very similar to previous
            prev = deduped[-1]
            # Check if >80% of words overlap
            prev_words = set(prev.lower().split())
            curr_words = set(text.lower().split())
            if prev_words and curr_words:
                overlap = len(prev_words & curr_words) / max(
                    len(prev_words), len(curr_words)
                )
                if overlap < 0.8:
                    deduped.append(text)
            else:
                deduped.append(text)

        return deduped
