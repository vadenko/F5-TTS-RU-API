import asyncio
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from app.models.schemas import TaskStatus


@dataclass
class Task:
    """Задача конвертации"""
    id: str
    book_id: str
    status: TaskStatus = TaskStatus.PENDING
    book_title: str = ""
    author: str = ""
    current_chapter: int = 0
    total_chapters: int = 0
    current_chunk: int = 0
    total_chunks: int = 0
    progress_percent: float = 0.0
    message: str = ""
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result_url: str | None = None
    output_path: Path | None = None
    progress_callback: Callable | None = None


class TaskManager:
    """Менеджер задач"""
    
    def __init__(self):
        self._tasks: dict[str, Task] = {}
        self._book_tasks: dict[str, str] = {}
        self._lock = asyncio.Lock()
    
    async def create_task(self, book_id: str, book_title: str = "", author: str = "") -> Task:
        """Создание новой задачи"""
        async with self._lock:
            task_id = str(uuid.uuid4())
            task = Task(
                id=task_id,
                book_id=book_id,
                book_title=book_title,
                author=author
            )
            self._tasks[task_id] = task
            self._book_tasks[book_id] = task_id
            return task
    
    async def get_task(self, task_id: str) -> Task | None:
        """Получение задачи по ID"""
        return self._tasks.get(task_id)
    
    async def get_task_by_book(self, book_id: str) -> Task | None:
        """Получение задачи по ID книги"""
        task_id = self._book_tasks.get(book_id)
        if task_id:
            return self._tasks.get(task_id)
        return None
    
    async def update_progress(
        self,
        task_id: str,
        status: TaskStatus | None = None,
        current_chapter: int | None = None,
        total_chapters: int | None = None,
        current_chunk: int | None = None,
        total_chunks: int | None = None,
        progress_percent: float | None = None,
        message: str | None = None,
        error: str | None = None,
        result_url: str | None = None
    ):
        """Обновление прогресса задачи"""
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            
            if status:
                task.status = status
                if status == TaskStatus.PROCESSING and task.started_at is None:
                    task.started_at = datetime.now()
                elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    task.completed_at = datetime.now()
            
            if current_chapter is not None:
                task.current_chapter = current_chapter
            if total_chapters is not None:
                task.total_chapters = total_chapters
            if current_chunk is not None:
                task.current_chunk = current_chunk
            if total_chunks is not None:
                task.total_chunks = total_chunks
            if progress_percent is not None:
                task.progress_percent = progress_percent
            if message is not None:
                task.message = message
            if error is not None:
                task.error = error
            if result_url is not None:
                task.result_url = result_url
    
    async def get_all_tasks(self) -> list[Task]:
        """Получение всех задач"""
        async with self._lock:
            return list(self._tasks.values())
    
    async def get_pending_tasks(self) -> list[Task]:
        """Получение ожидающих задач"""
        async with self._lock:
            return [t for t in self._tasks.values() if t.status == TaskStatus.PENDING]
    
    async def cancel_task(self, task_id: str) -> bool:
        """Отмена задачи"""
        async with self._lock:
            task = self._tasks.get(task_id)
            if task and task.status == TaskStatus.PENDING:
                task.status = TaskStatus.FAILED
                task.error = "Cancelled by user"
                return True
        return False
    
    async def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Очистка старых завершённых задач"""
        async with self._lock:
            now = datetime.now()
            to_remove = []
            for task_id, task in self._tasks.items():
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    if task.completed_at:
                        age = (now - task.completed_at).total_seconds() / 3600
                        if age > max_age_hours:
                            to_remove.append(task_id)
            for task_id in to_remove:
                del self._tasks[task_id]


task_manager = TaskManager()
