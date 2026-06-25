"""
Executable entry point for the RAG Answering Service.

This is the script PyInstaller bundles into the .exe. It:
  1. Loads a `.env` file sitting NEXT TO the executable (not baked in),
  2. Ensures an `uploads/` folder exists next to the executable,
  3. Launches the FastAPI app with uvicorn (single process, no reloader).

End users just place a `.env` next to RAGServer.exe and double-click it
(or run it from a terminal). PostgreSQL and Redis must be reachable via the
DATABASE_URL and REDIS_URL values in that `.env`.
"""

import os
import sys
import multiprocessing


def _base_dir() -> str:
    """Return the directory containing the executable (frozen) or this script (dev)."""
    if getattr(sys, "frozen", False):
        # Running as a PyInstaller bundle: sys.executable is the .exe path
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    # Required so PyInstaller-frozen apps don't spawn infinite subprocesses on Windows
    multiprocessing.freeze_support()

    base = _base_dir()

    # ── 1. Load .env located next to the executable ──────────────────────────
    env_path = os.path.join(base, ".env")
    try:
        from dotenv import load_dotenv
        if os.path.exists(env_path):
            load_dotenv(env_path, override=True)
            print(f"[startup] Loaded configuration from: {env_path}")
        else:
            print(f"[startup] WARNING: no .env file found next to the executable ({env_path}).")
            print("[startup] The server will use system environment variables / defaults,")
            print("[startup] and will fail if OPENAI_API_KEY is not set.")
    except Exception as e:  # pragma: no cover
        print(f"[startup] Could not load .env: {e}")

    # ── 2. Ensure an uploads directory exists next to the executable ─────────
    uploads_dir = os.path.join(base, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    # Only set if the user didn't explicitly configure one in .env
    os.environ.setdefault("UPLOAD_DIR", uploads_dir)

    # ── 3. Launch the server ─────────────────────────────────────────────────
    host = os.environ.get("HOST", "0.0.0.0")
    try:
        port = int(os.environ.get("APP_PORT", os.environ.get("PORT", "8000")))
    except ValueError:
        port = 8000

    # Import AFTER env is loaded so settings pick up the right values
    import uvicorn
    from app.main import app

    print("=" * 60)
    print("  RAG Answering Service")
    print(f"  Running at:  http://{host}:{port}")
    print(f"  Swagger UI:  http://{host}:{port}/docs")
    print("  Press CTRL+C to stop.")
    print("=" * 60)

    # Pass the app object directly (NOT an import string) so uvicorn does not
    # try to spawn a reloader/worker subprocess — which breaks under PyInstaller.
    uvicorn.run(app, host=host, port=port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
