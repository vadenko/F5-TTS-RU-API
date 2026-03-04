from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class OutputFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"


class BookConversionRequest(BaseModel):
    """Запрос на конвертацию книги"""
    out_format: OutputFormat = OutputFormat.MP3
    ref_audio: str | None = None
    ref_text: str | None = None
    voice_speed: float | None = 1.0
    voice_pitch: float | None = 0.0
    voice_volume: float | None = 1.0
    chapters: list[int] | None = None
    merge_chapters: bool = True
    add_chapter_markers: bool = True
    notify_url: str | None = None
    
    class Config:
        use_enum_values = True


class ChapterInfo(BaseModel):
    """Информация о главе"""
    number: int
    title: str
    text_length: int
    chunks: int


class BookInfo(BaseModel):
    """Информация о загруженной книге"""
    id: str
    title: str
    author: str
    lang: str
    year: str
    series: str | None
    sequence_number: str | None
    description: str | None
    chapters: list[ChapterInfo]
    total_text_length: int
    estimated_duration_minutes: float
    has_cover: bool


class TaskProgress(BaseModel):
    """Прогресс задачи"""
    task_id: str
    status: TaskStatus
    book_id: str | None = None
    book_title: str | None = None
    current_chapter: int | None = None
    total_chapters: int | None = None
    current_chunk: int | None = None
    total_chunks: int | None = None
    progress_percent: float = 0.0
    message: str | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result_url: str | None = None


class AudioBookResult(BaseModel):
    """Результат конвертации"""
    task_id: str
    status: TaskStatus
    book_title: str
    author: str
    duration_seconds: float
    file_size_mb: float
    chapters_count: int
    output_format: OutputFormat
    download_url: str
    created_at: datetime


class HealthResponse(BaseModel):
    """Ответ проверки здоровья"""
    status: str
    version: str
    device: str
    models_loaded: bool
    queue_size: int
