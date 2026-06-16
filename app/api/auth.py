import logging
import secrets
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from app.config import get_settings

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key_header: str = Security(api_key_header)):
    """FastAPI Dependency for validating the X-API-Key header."""
    settings = get_settings()
    
    # If no API key is configured on the server, skip authentication entirely.
    # This is intended for local development and testing.
    if not settings.api_key:
        return "no-auth"

    if not secrets.compare_digest(api_key_header, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )
    return api_key_header
