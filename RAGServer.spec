# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the RAG Answering Service.

Build with:
    pyinstaller RAGServer.spec --noconfirm

Produces a one-folder distribution under dist/RAGServer/ containing
RAGServer.exe (on Windows). The folder is what you ship; the user drops a
.env next to RAGServer.exe and runs it.

We use one-FOLDER (not one-file) mode because:
  - PyTorch + sentence-transformers load native libraries and data files that
    are far more reliable in folder mode,
  - startup is much faster (no per-launch unpacking of a multi-GB archive).
"""

from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

datas = []
binaries = []
hiddenimports = []

# ── Heavy third-party packages that need full collection ─────────────────────
# collect_all grabs submodules, data files, and binaries for each package.
for pkg in [
    "torch",
    "sentence_transformers",
    "transformers",
    "tokenizers",
    "safetensors",
    "huggingface_hub",
    "sklearn",
    "scipy",
    "tqdm",
    "regex",
]:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        # Package may be optional; skip if not installed
        pass

# ── Data files for packages that ship runtime resources ──────────────────────
datas += collect_data_files("langdetect")        # language profile data
datas += collect_data_files("certifi")           # CA bundle for TLS

# ── Bundle the application's own package and static web UI ───────────────────
# Many imports in app/ happen INSIDE functions (lazy imports), which PyInstaller
# static analysis can miss — so explicitly collect every submodule of `app`.
hiddenimports += collect_submodules("app")

# Static frontend (served at "/")
datas += [("static", "static")]

# ── Extra hidden imports PyInstaller often misses for async/db/ML stacks ─────
hiddenimports += [
    "uvicorn",
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "asyncpg",
    "pgvector",
    "pgvector.sqlalchemy",
    "sqlalchemy.dialects.postgresql.asyncpg",
    "redis.asyncio",
    "fitz",            # PyMuPDF
    "docx",            # python-docx
    "openpyxl",
    "pandas",
    "PIL",
    "pytesseract",
    "langdetect",
    "sympy",
    "assemblyai",
    "openai",
    "aiofiles",
    "filetype",
    "dotenv",
]

block_cipher = None

a = Analysis(
    ["run_server.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "PyQt5",
        "PySide2",
        "notebook",
        "IPython",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RAGServer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,           # keep a console window so users see logs/errors
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="RAGServer",
)
