import logging
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from app.config import get_settings

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key_header: str = Security(api_key_header)):
    """FastAPI Dependency for validating the X-API-Key header."""
    settings = get_settings()
    
    # If no API key is configured on the server, we still require one for mutating routes by default,
    # or we can allow bypass in debug mode. For safety, we enforce it unless it's explicitly disabled.
    if not settings.api_key:
        logger.warning("No API Key configured on server. Rejecting authenticated request.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Server authentication not configured."
        )

    if api_key_header != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )
    return api_key_header
