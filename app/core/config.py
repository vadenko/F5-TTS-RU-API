import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
ENV_PATH = BASE_DIR / ".env"

print(f"Looking for .env at: {ENV_PATH}")
print(f"Exists: {ENV_PATH.exists()}")

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

data_dir_env = os.getenv("DATA_DIR", "")
if data_dir_env:
    DATA_DIR = Path(data_dir_env)
else:
    DATA_DIR = BASE_DIR / "data"

# Лог-файл для отладки
LOG_FILE = DATA_DIR / "app.log"

INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
BOOKS_DIR = DATA_DIR / "books"
TEMP_DIR = DATA_DIR / "temp"

print(f"DATA_DIR: {DATA_DIR}")
print(f"OUTPUT_DIR: {OUTPUT_DIR}")

MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "2000"))
MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "1"))
DEVICE = os.getenv("DEVICE", "cpu")
NFE_STEPS = int(os.getenv("NFE_STEPS", "32"))  # Количество шагов денойзинга (меньше = быстрее)

MODEL_REPO = os.getenv("MODEL_REPO", "Misha24-10/F5-TTS_RUSSIAN")
MODEL_PATH = os.getenv("MODEL_PATH", "")
CKPT_FILE = os.getenv("CKPT_FILE", "")
VOCAB_FILE = os.getenv("VOCAB_FILE", "")

if MODEL_PATH:
    MODEL_PATH = MODEL_PATH.strip() or None
if CKPT_FILE:
    CKPT_FILE = CKPT_FILE.strip() or None
if VOCAB_FILE:
    VOCAB_FILE = VOCAB_FILE.strip() or None

print(f"=== Config loaded ===")
print(f"MODEL_PATH: {MODEL_PATH}")
print(f"CKPT_FILE: {CKPT_FILE}")
print(f"VOCAB_FILE: {VOCAB_FILE}")
print(f"DEVICE: {DEVICE}")
print(f"====================")

for dir_path in [DATA_DIR, INPUT_DIR, OUTPUT_DIR, BOOKS_DIR, TEMP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
