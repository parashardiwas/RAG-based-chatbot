"""
Audio processing: transcription using AssemblyAI.
Handles audio transcription from file uploads.
"""

import logging
import os
import tempfile
import time
import asyncio
from typing import Any

from fastapi import UploadFile
import assemblyai as aai
from app.config import get_settings

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Audio processing service using AssemblyAI for transcription.
    
    Features:
    - Transcribes audio files (mp3, wav, m4a, flac, ogg, etc.)
    - Returns transcript with confidence
    """

    def __init__(self):
        settings = get_settings()
        aai.settings.api_key = settings.assemblyai_api_key

    async def transcribe_upload(self, file: UploadFile) -> dict[str, Any]:
        """
        Transcribe an uploaded audio file.
        
        Returns:
            dict with: text, language, confidence, segments, latency_ms
        """
        start_time = time.time()

        # Save uploaded file to temp location
        suffix = os.path.splitext(file.filename or "audio.wav")[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            return await self.transcribe_file(tmp_path)
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    async def transcribe_file(self, file_path: str) -> dict[str, Any]:
        """
        Transcribe an audio file from disk using AssemblyAI.
        """
        start_time = time.time()
        
        try:
            transcriber = aai.Transcriber()
            # Wrap the synchronous SDK call in a thread
            transcript = await asyncio.to_thread(transcriber.transcribe, file_path)
            
            if transcript.error:
                logger.error(f"AssemblyAI Transcription Error: {transcript.error}")
                return self._error_result(f"API Error: {transcript.error}")

            text = transcript.text or ""
            confidence = transcript.confidence or 0.0
            
            segments = []
            if transcript.words:
                for word in transcript.words:
                    segments.append({
                        "start": word.start / 1000.0,
                        "end": word.end / 1000.0,
                        "text": word.text,
                        "confidence": word.confidence
                    })

            latency_ms = int((time.time() - start_time) * 1000)

            return {
                "text": text,
                "language": "en", # AssemblyAI auto-detects but returns english unless configured
                "confidence": confidence,
                "segments": segments,
                "latency_ms": latency_ms,
            }

        except Exception as e:
            logger.error(f"AssemblyAI Exception: {e}", exc_info=True)
            return self._error_result(str(e))

    def _error_result(self, error_msg: str) -> dict[str, Any]:
        return {
            "text": f"[Transcription Failed: {error_msg}]",
            "language": "en",
            "confidence": 0.0,
            "segments": [],
            "latency_ms": 0,
        }

    async def extract_audio_from_video(self, video_path: str) -> str | None:
        """Extract audio from video file to a temporary wav file."""
        import tempfile
        import asyncio
        import os
        
        fd, audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        
        try:
            process = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-i", video_path, 
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                audio_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"ffmpeg extraction failed: {stderr.decode()}")
                os.unlink(audio_path)
                return None
                
            return audio_path
        except Exception as e:
            logger.error(f"Failed to extract audio from video: {e}")
            try:
                os.unlink(audio_path)
            except OSError:
                pass
            return None
