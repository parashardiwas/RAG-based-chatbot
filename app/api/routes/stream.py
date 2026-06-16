import asyncio
import json
import logging
import os
import ssl
import certifi
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
import assemblyai as aai
from assemblyai.streaming.v3 import (
    StreamingClient,
    StreamingClientOptions,
    StreamingEvents,
    StreamingParameters,
)
from app.config import get_settings

# Fix SSL certificates for macOS Python
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

router = APIRouter(prefix="/api/v1", tags=["Stream"])
logger = logging.getLogger(__name__)

@router.websocket("/stream/dictation")
async def stream_dictation(websocket: WebSocket, token: str = Query(default="")):
    settings = get_settings()

    await websocket.accept()

    # Validate token after accepting so we can send error messages back
    if settings.api_key and token != settings.api_key:
        logger.warning("WebSocket auth failed: token mismatch")
        await websocket.send_json({"type": "error", "message": "Invalid API key. Please set your API key in Settings."})
        await websocket.close(code=4001)
        return
    
    if not settings.assemblyai_api_key:
        logger.error("AssemblyAI API key not configured")
        await websocket.send_json({"type": "error", "message": "AssemblyAI API key not configured"})
        await websocket.close(code=1011)
        return

    aai.settings.api_key = settings.assemblyai_api_key
    loop = asyncio.get_running_loop()

    # Set up the v3 StreamingClient
    client = StreamingClient(
        StreamingClientOptions(api_key=settings.assemblyai_api_key)
    )

    # v3 callbacks receive (client, event) - two arguments
    def on_turn(client_ref, event):
        """Called when AssemblyAI produces a transcript turn."""
        if not event.transcript:
            return
        
        async def send_transcript():
            try:
                await websocket.send_json({
                    "type": "transcript",
                    "text": event.transcript,
                    "is_final": event.end_of_turn
                })
            except Exception as e:
                logger.error(f"Error sending transcript: {e}")
        
        asyncio.run_coroutine_threadsafe(send_transcript(), loop)

    def on_error(client_ref, error):
        logger.error(f"AssemblyAI Streaming Error: {error}")
        
        async def send_error():
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Transcription error: {str(error)}"
                })
            except Exception:
                pass
        
        asyncio.run_coroutine_threadsafe(send_error(), loop)

    def on_begin(client_ref, event):
        logger.info(f"AssemblyAI session started")

    client.on(StreamingEvents.Turn, on_turn)
    client.on(StreamingEvents.Error, on_error)
    client.on(StreamingEvents.Begin, on_begin)

    try:
        logger.info("Connecting to AssemblyAI streaming service...")
        client.connect(StreamingParameters(sample_rate=16000))
        logger.info("AssemblyAI streaming connection established")
        
        while True:
            # Receive raw PCM data from the browser
            data = await websocket.receive_bytes()
            client.stream(data)
            
    except WebSocketDisconnect:
        logger.info("Browser disconnected from dictation stream")
    except Exception as e:
        logger.error(f"Error in streaming route: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": f"Transcription error: {str(e)}"})
        except Exception:
            pass
    finally:
        try:
            client.disconnect()
        except Exception:
            pass
