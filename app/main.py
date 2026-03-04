import logging
import shutil
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from app.api.books import router as books_router
from app.core.config import DATA_DIR, INPUT_DIR, OUTPUT_DIR, BOOKS_DIR, TEMP_DIR, LOG_FILE

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

BASE_DIR = Path(__file__).parent

app = FastAPI(
    title="F5-TTS Audiobook API",
    description="REST API for converting FB2 ebooks to audiobooks using F5-TTS",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(books_router, prefix="/api/v1", tags=["books"])


@app.on_event("startup")
async def startup_event():
    for dir_path in [DATA_DIR, INPUT_DIR, OUTPUT_DIR, BOOKS_DIR, TEMP_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    if shutil.which("f5-tts_infer-cli") is None:
        raise RuntimeError("f5-tts_infer-cli not found in PATH")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = BASE_DIR / "templates" / "index.html"
    return FileResponse(html_path)


@app.get("/index.html", response_class=HTMLResponse)
async def index():
    html_path = BASE_DIR / "templates" / "index.html"
    return FileResponse(html_path)
