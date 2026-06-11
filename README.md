# RAG-Based Multimodal AI Answering Service

A production-grade AI backend using **Retrieval-Augmented Generation (RAG)** to answer questions across topics, with multimodal input support and multilingual capabilities.

## Features

- 🎤 **Multimodal Input**: Text, Audio (Voice), Video (with/without audio), Files
- 🌐 **Multilingual**: Hindi, English, Hinglish — responds in the same language as input
- 🔍 **Hybrid RAG**: Vector search (pgvector) + BM25 keyword search with reciprocal rank fusion
- ⚡ **Ultra-Fast Latency**: Aggressively optimized with parallel retrieval, intelligent BM25 corpus limits, and a fast hybrid language detection pipeline (`langdetect` + heuristics).
- 📊 **Confidence Scoring**: Multi-layer confidence checks with 80% match threshold
- ✏️ **Editable Q/A**: Full CRUD with versioning, soft delete, and audit trails
- 🚀 **Concurrency**: asyncio-based execution with background task dispatching for caching and logging.
- 💰 **Cost Tracking**: Monitor API-equivalent costs for LLM tokens
- 🔒 **Safety**: "I don't know" fallback with clarifying questions

## Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | FastAPI (async Python) |
| LLM | OpenAI API (gpt-3.5-turbo, gpt-4o, etc.) |
| Embeddings | sentence-transformers (multi-qa-MiniLM-L6-cos-v1) |
| ASR | faster-whisper (large-v3) |
| OCR | Tesseract |
| Database | PostgreSQL 16 + pgvector |
| Cache | Redis 7 |
| Language Detection | langdetect + custom Hinglish heuristics |

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for PostgreSQL + Redis)
- ffmpeg (for audio/video processing)
- Tesseract (for OCR)

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt
```

### 2. Start Infrastructure

```bash
# Start PostgreSQL + Redis
docker compose up -d
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env to add your OpenAI API Key and Model
# OPENAI_API_KEY="your-api-key"
# OPENAI_MODEL="gpt-3.5-turbo" 
```

### 4. Run the App

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Open the UI

Navigate to http://localhost:8000 for the web interface, or http://localhost:8000/docs for the API documentation.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/ask` | Ask a text question |
| POST | `/api/v1/ask/audio` | Ask via audio file |
| POST | `/api/v1/ask/video` | Ask via video file |
| POST | `/api/v1/ingest/file` | Upload file for ingestion |
| GET | `/api/v1/ingest/status/{id}` | Check ingestion status |
| GET | `/api/v1/qa` | List Q/A pairs |
| POST | `/api/v1/qa` | Create Q/A pair |
| PUT | `/api/v1/qa/{id}` | Update Q/A pair |
| DELETE | `/api/v1/qa/{id}` | Delete Q/A pair |
| POST | `/api/v1/qa/{id}/restore` | Restore deleted pair |
| POST | `/api/v1/qa/bulk` | Bulk upload Q/A pairs |
| GET | `/health` | System health check |
| GET | `/metrics` | System metrics |

## Architecture

```
Input (Text/Audio/Video/File)
    → Media Processing (Whisper/OCR/File Parser)
    → Fast Hybrid Language Detection (langdetect + heuristics)
    → Cache Check (Fast Path)
    → Query Embedding (sentence-transformers)
    → Parallel Hybrid Retrieval (pgvector + BM25)
    → Confidence Gate
    → Answer Generation (OpenAI LLM)
    → Final Translation (Respects original language)
    → Background Caching & Auditing
    → Response
```

## Project Structure

```
app/
├── main.py              # FastAPI app entry point
├── config.py            # Settings and thresholds
├── api/
│   ├── routes/          # API endpoints
│   └── middleware/       # Rate limiting
├── core/
│   ├── orchestrator.py  # Main pipeline orchestration
│   ├── queue_manager.py # Concurrency control
│   └── cost_tracker.py  # Cost monitoring
├── services/
│   ├── rag/             # Embedding, retrieval, reranking
│   ├── llm/             # Generation, confidence, model routing
│   ├── media/           # Audio, video, file processing
│   ├── language/        # Language detection and translation
│   └── qa/              # Q/A pair management
├── db/
│   ├── models.py        # SQLAlchemy ORM models
│   └── database.py      # DB connection
└── schemas/             # Pydantic request/response models
```

## License

MIT
