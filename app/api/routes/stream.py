import asyncio
import json
import logging
import hashlib
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
import assemblyai as aai
from app.config import get_settings

router = APIRouter(prefix="/api/v1", tags=["Stream"])
logger = logging.getLogger(__name__)

@router.websocket("/stream/dictation")
async def stream_dictation(websocket: WebSocket, token: str = Query(default="")):
    settings = get_settings()

    # Validate token before accepting the WebSocket upgrade
    # Use the server's API key if configured
    if settings.api_key and token != settings.api_key:
        await websocket.close(code=4001)
        return

    await websocket.accept()
    
    if not settings.assemblyai_api_key:
        await websocket.send_json({"type": "error", "message": "AssemblyAI API key not configured"})
        await websocket.close(code=1011)
        return

    aai.settings.api_key = settings.assemblyai_api_key

    loop = asyncio.get_running_loop()
    
    def on_data(transcript: aai.RealtimeTranscript):
        if not transcript.text:
            return
        
        # We need to send this back to the WebSocket asynchronously
        # from a synchronous callback
        async def send_transcript():
            try:
                is_final = isinstance(transcript, aai.RealtimeFinalTranscript)
                await websocket.send_json({
                    "type": "transcript",
                    "text": transcript.text,
                    "is_final": is_final
                })
            except Exception as e:
                logger.error(f"Error sending transcript: {e}")
        
        asyncio.run_coroutine_threadsafe(send_transcript(), loop)

    def on_error(error: aai.RealtimeError):
        logger.error(f"AssemblyAI Realtime Error: {error}")

    def on_open(session_opened: aai.RealtimeSessionOpened):
        logger.info(f"AssemblyAI Session Opened: {session_opened.session_id}")

    def on_close():
        logger.info("AssemblyAI Session Closed")

    transcriber = aai.RealtimeTranscriber(
        sample_rate=16000,
        on_data=on_data,
        on_error=on_error,
        on_open=on_open,
        on_close=on_close,
        end_utterance_silence_threshold=1000
    )

    try:
        # Start the connection
        transcriber.connect()
        
        while True:
            # Receive raw PCM data from the browser
            data = await websocket.receive_bytes()
            transcriber.stream(data)
            
    except WebSocketDisconnect:
        logger.info("Browser disconnected from dictation stream")
    except Exception as e:
        logger.error(f"Error in streaming route: {e}")
    finally:
        transcriber.close()
