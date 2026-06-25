RAG Answering Service — Windows Executable
==========================================

This package contains a self-contained Windows build of the RAG Answering
Service. You do NOT need to install Python or any libraries to run it.

--------------------------------------------------------------------------
WHAT YOU STILL NEED (on the machine that runs this)
--------------------------------------------------------------------------
The executable bundles the application and all its Python/ML libraries, but
it connects to two services that must be running and reachable:

  1. PostgreSQL 13+  with the "pgvector" extension enabled.
  2. Redis 6+.

These can run on the same machine or on another server. You point the app at
them using the .env file (see below).

You also need:
  - An OpenAI API key (for answer generation).
  - (Optional) An AssemblyAI API key (only for audio/voice features).
  - Internet access on FIRST launch: the app downloads two small ML models
    (~120 MB total) from HuggingFace and caches them for future runs.

--------------------------------------------------------------------------
SETUP (one time)
--------------------------------------------------------------------------
1. Make sure PostgreSQL and Redis are running and reachable.
   In PostgreSQL, create the database and enable pgvector once:

       CREATE DATABASE ragbot;
       \c ragbot
       CREATE EXTENSION IF NOT EXISTS vector;

2. In this folder you'll find ".env.example". Make a copy named ".env":

       copy .env.example .env

3. Open ".env" in Notepad and set at least these values:

       DATABASE_URL=postgresql+asyncpg://USER:PASSWORD@HOST:5432/ragbot
       REDIS_URL=redis://HOST:6379/0
       OPENAI_API_KEY=sk-...                 (your OpenAI key)
       API_KEY=choose-a-secret-key           (clients send this as X-API-Key)

   Examples for a local Postgres/Redis on the same PC:
       DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/ragbot
       REDIS_URL=redis://localhost:6379/0

   IMPORTANT: keep ".env" in the SAME folder as RAGServer.exe.

--------------------------------------------------------------------------
RUNNING
--------------------------------------------------------------------------
Double-click "RAGServer.exe", or run it from a Command Prompt:

       RAGServer.exe

A console window opens and shows startup logs. When you see:

       Running at:  http://0.0.0.0:8000

the service is live. Open a browser to:

       http://localhost:8000/docs        (interactive API documentation)
       http://localhost:8000/health      (status check)

To stop the server, close the console window or press CTRL+C in it.

--------------------------------------------------------------------------
CHANGING THE PORT
--------------------------------------------------------------------------
By default it listens on port 8000. To change it, add this to .env:

       APP_PORT=9000

--------------------------------------------------------------------------
TROUBLESHOOTING
--------------------------------------------------------------------------
* "OPENAI_API_KEY must be set" / app exits immediately
    -> Your .env is missing or OPENAI_API_KEY is empty. Check step 3 above,
       and confirm .env sits next to RAGServer.exe.

* Database connection errors
    -> Verify PostgreSQL is running and DATABASE_URL is correct, and that the
       "vector" extension was created (step 1).

* Redis connection errors
    -> Verify Redis is running and REDIS_URL is correct.

* First launch is slow
    -> It's downloading the ML models (~120 MB) once. Later launches are fast.

* Windows SmartScreen warns about an unknown publisher
    -> This is expected for unsigned executables. Choose "More info" ->
       "Run anyway" if you trust the source.

* Antivirus flags the .exe
    -> PyInstaller executables are sometimes false-positived. Whitelist the
       file if your organization's policy allows.

--------------------------------------------------------------------------
NOTES
--------------------------------------------------------------------------
- Uploaded files are stored in the "uploads" folder next to the executable.
- This is an x86-64 build. It runs on standard 64-bit Intel/AMD Windows.
- For full API details, see API_DOCUMENTATION.md (shipped separately).
