"""
RAG-Based Multimodal AI Answering Service

Main FastAPI application entry point.
"""

import logging
import os
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.core.orchestrator import get_orchestrator, shutdown_orchestrator
from app.db.database import init_db, async_session_maker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize and cleanup resources."""
    settings = get_settings()
    logger.info(f"🚀 Starting {settings.app_name} v{settings.app_version}")

    # ── Initialize Database ───────────────────────────────────
    logger.info("Initializing database...")
    await init_db()
    logger.info("✅ Database initialized")

    # ── Initialize Redis ──────────────────────────────────────
    redis_client = None
    try:
        redis_client = aioredis.from_url(
            settings.redis_url, decode_responses=True
        )
        await redis_client.ping()
        logger.info("✅ Redis connected")
    except Exception as e:
        logger.warning(f"⚠️ Redis unavailable: {e}. Running without cache.")
        redis_client = None

    # ── Initialize Orchestrator ───────────────────────────────
    logger.info("Initializing orchestrator...")
    orchestrator = await get_orchestrator()
    await orchestrator.init(
        db_session_factory=async_session_maker,
        redis_client=redis_client,
    )
    logger.info("✅ Orchestrator initialized")

    # ── Ensure upload directory exists ────────────────────────
    os.makedirs(settings.upload_dir, exist_ok=True)

    logger.info(f"✅ {settings.app_name} is ready!")
    logger.info(f"   📊 Max concurrent requests: {settings.max_concurrent_requests}")
    logger.info(f"   🧠 LLM model: {settings.openai_model}")
    logger.info(f"   📐 Embedding model: {settings.embedding_model}")

    yield

    # ── Cleanup ───────────────────────────────────────────────
    await shutdown_orchestrator()
    if redis_client:
        await redis_client.close()
    logger.info("👋 Shutdown complete")


# ── Create FastAPI App ────────────────────────────────────────
app = FastAPI(
    title="RAG Answering Service",
    description=(
        "AI-powered answering service with multimodal input support "
        "(text, voice, video, files), multilingual support "
        "(Hindi, English, Hinglish), and RAG-based retrieval across topics."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS Middleware ───────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rate Limiter Middleware ───────────────────────────────────
from app.api.middleware.rate_limiter import RateLimitMiddleware
app.add_middleware(RateLimitMiddleware, requests_per_minute=120)

# ── Register Routes ──────────────────────────────────────────
from app.api.routes import query, ingest, qa_pairs, health, stream

app.include_router(query.router)
app.include_router(ingest.router)
app.include_router(qa_pairs.router)
app.include_router(health.router)
app.include_router(stream.router)

# ── Serve Static Files (Web UI) ──────────────────────────────
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # Serve index.html at root
    from fastapi.responses import FileResponse

    @app.get("/", include_in_schema=False)
    async def serve_ui():
        return FileResponse(
            os.path.join(static_dir, "index.html"),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )


# ── Root redirect to docs (if no static) ─────────────────────
@app.get("/api", include_in_schema=False)
async def api_root():
    return {
        "service": "RAG Answering Service",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
