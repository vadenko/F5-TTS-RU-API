import asyncio
import os
import shutil
from pathlib import Path
from typing import Annotated

import aiofiles
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from app.core.config import BOOKS_DIR, DEVICE, OUTPUT_DIR
from app.models.schemas import (
    AudioBookResult,
    BookConversionRequest,
    BookInfo,
    ChapterInfo,
    HealthResponse,
    OutputFormat,
    TaskProgress,
    TaskStatus,
)
from app.services.audio_generator import audio_generator
from app.services.fb2_parser import FB2Parser, detect_encoding
from app.services.task_manager import task_manager


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка здоровья сервиса"""
    tasks = await task_manager.get_all_tasks()
    pending_count = len([t for t in tasks if t.status == TaskStatus.PENDING])
    
    return HealthResponse(
        status="ok",
        version="1.0.0",
        device=DEVICE,
        models_loaded=True,
        queue_size=pending_count
    )


@router.post("/books/upload", response_model=BookInfo)
async def upload_book(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Загрузка FB2 книги"""
    filename = file.filename or ""
    if not filename.lower().endswith('.fb2'):
        raise HTTPException(status_code=400, detail="Only .fb2 files are supported")
    
    book_id = os.urandom(8).hex()
    book_path = BOOKS_DIR / f"{book_id}.fb2"
    
    try:
        content = await file.read()
        encoding = detect_encoding(content)
        async with aiofiles.open(book_path, 'wb') as f:
            await f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    
    try:
        parser = FB2Parser()
        book = parser.parse_content(content.decode(encoding))
    except Exception as e:
        book_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid FB2 file: {e}")
    
    book.metadata.id = book_id
    
    chapters_info = []
    total_length = 0
    
    for chapter in book.chapters:
        chunks = parser.split_into_chunks(chapter.text)
        chapters_info.append(ChapterInfo(
            number=chapter.number,
            title=chapter.title,
            text_length=len(chapter.text),
            chunks=len(chunks)
        ))
        total_length += len(chapter.text)
    
    estimated_chars_per_minute = 1000
    estimated_duration = total_length / estimated_chars_per_minute
    
    return BookInfo(
        id=book_id,
        title=book.metadata.title,
        author=book.metadata.author,
        lang=book.metadata.lang,
        year=book.metadata.year,
        series=book.metadata.series or None,
        sequence_number=book.metadata.sequence_number or None,
        description=book.metadata.description or None,
        chapters=chapters_info,
        total_text_length=total_length,
        estimated_duration_minutes=estimated_duration,
        has_cover=book.metadata.cover_image is not None
    )


@router.get("/books/{book_id}", response_model=BookInfo)
async def get_book_info(book_id: str):
    """Получение информации о книге"""
    book_path = BOOKS_DIR / f"{book_id}.fb2"
    
    if not book_path.exists():
        raise HTTPException(status_code=404, detail="Book not found")
    
    try:
        parser = FB2Parser()
        # Читаем в бинарном режиме и определяем кодировку
        raw_content = book_path.read_bytes()
        from app.services.fb2_parser import detect_encoding
        encoding = detect_encoding(raw_content)
        content = raw_content.decode(encoding)
        book = parser.parse_content(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse book: {e}")
    
    chapters_info = []
    total_length = 0
    
    for chapter in book.chapters:
        chunks = parser.split_into_chunks(chapter.text)
        chapters_info.append(ChapterInfo(
            number=chapter.number,
            title=chapter.title,
            text_length=len(chapter.text),
            chunks=len(chunks)
        ))
        total_length += len(chapter.text)
    
    estimated_chars_per_minute = 1000
    estimated_duration = total_length / estimated_chars_per_minute
    
    return BookInfo(
        id=book_id,
        title=book.metadata.title,
        author=book.metadata.author,
        lang=book.metadata.lang,
        year=book.metadata.year,
        series=book.metadata.series or None,
        sequence_number=book.metadata.sequence_number or None,
        description=book.metadata.description or None,
        chapters=chapters_info,
        total_text_length=total_length,
        estimated_duration_minutes=estimated_duration,
        has_cover=book.metadata.cover_image is not None
    )


@router.delete("/books/{book_id}")
async def delete_book(book_id: str):
    """Удаление книги"""
    book_path = BOOKS_DIR / f"{book_id}.fb2"
    
    if not book_path.exists():
        raise HTTPException(status_code=404, detail="Book not found")
    
    book_path.unlink()
    return {"status": "deleted", "book_id": book_id}


@router.post("/books/{book_id}/convert", response_model=TaskProgress)
async def convert_book(
    book_id: str,
    request: BookConversionRequest,
    background_tasks: BackgroundTasks
):
    """Конвертация книги в аудио"""
    book_path = BOOKS_DIR / f"{book_id}.fb2"
    
    if not book_path.exists():
        raise HTTPException(status_code=404, detail="Book not found")
    
    existing_task = await task_manager.get_task_by_book(book_id)
    if existing_task and existing_task.status == TaskStatus.PROCESSING:
        raise HTTPException(status_code=409, detail="Book conversion already in progress")
    
    try:
        parser = FB2Parser()
        # Читаем в бинарном режиме и определяем кодировку
        raw_content = book_path.read_bytes()
        from app.services.fb2_parser import detect_encoding
        encoding = detect_encoding(raw_content)
        content = raw_content.decode(encoding)
        book = parser.parse_content(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse book: {e}")
    
    if not book.chapters:
        raise HTTPException(status_code=400, detail="No chapters found in book")
    
    task = await task_manager.create_task(
        book_id=book_id,
        book_title=book.metadata.title,
        author=book.metadata.author
    )
    
    background_tasks.add_task(
        process_conversion,
        task.id,
        book_path,
        request
    )
    
    return TaskProgress(
        task_id=task.id,
        status=TaskStatus.PENDING,
        book_id=book_id,
        book_title=book.metadata.title,
        message="Task queued"
    )


@router.get("/tasks/{task_id}", response_model=TaskProgress)
async def get_task_progress(task_id: str):
    """Получение прогресса задачи"""
    task = await task_manager.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return TaskProgress(
        task_id=task.id,
        status=task.status,
        book_id=task.book_id,
        book_title=task.book_title,
        current_chapter=task.current_chapter if task.total_chapters else None,
        total_chapters=task.total_chapters or None,
        current_chunk=task.current_chunk if task.total_chunks else None,
        total_chunks=task.total_chunks or None,
        progress_percent=task.progress_percent,
        message=task.message,
        error=task.error,
        started_at=task.started_at,
        completed_at=task.completed_at,
        result_url=task.result_url
    )


@router.get("/tasks", response_model=list[TaskProgress])
async def list_tasks(status: TaskStatus | None = None):
    """Список всех задач"""
    tasks = await task_manager.get_all_tasks()
    
    if status:
        tasks = [t for t in tasks if t.status == status]
    
    return [
        TaskProgress(
            task_id=t.id,
            status=t.status,
            book_id=t.book_id,
            book_title=t.book_title,
            current_chapter=t.current_chapter if t.total_chapters else None,
            total_chapters=t.total_chapters or None,
            progress_percent=t.progress_percent,
            message=t.message,
            error=t.error,
            started_at=t.started_at,
            completed_at=t.completed_at,
            result_url=t.result_url
        )
        for t in tasks
    ]


@router.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Отмена задачи"""
    success = await task_manager.cancel_task(task_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Task not found or cannot be cancelled")
    
    return {"status": "cancelled", "task_id": task_id}


@router.get("/download/{task_id}")
async def download_result(task_id: str):
    """Скачивание результата"""
    task = await task_manager.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Task not completed")
    
    if not task.result_url:
        raise HTTPException(status_code=404, detail="Result file not found")
    
    file_path = Path(task.result_url)
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    
    media_type = "audio/mpeg" if file_path.suffix == ".mp3" else "audio/wav"
    
    return FileResponse(
        file_path,
        media_type=media_type,
        filename=file_path.name
    )


@router.websocket("/ws/progress/{task_id}")
async def websocket_progress(websocket: WebSocket, task_id: str):
    """WebSocket для получения прогресса в реальном времени"""
    await websocket.accept()
    
    try:
        while True:
            task = await task_manager.get_task(task_id)
            
            if not task:
                await websocket.send_json({
                    "error": "Task not found"
                })
                break
            
            await websocket.send_json({
                "task_id": task.id,
                "status": task.status.value,
                "current_chapter": task.current_chapter,
                "total_chapters": task.total_chapters,
                "current_chunk": task.current_chunk,
                "total_chunks": task.total_chunks,
                "progress_percent": task.progress_percent,
                "message": task.message,
                "error": task.error
            })
            
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                break
            
            import asyncio
            await asyncio.sleep(2)
    
    except WebSocketDisconnect:
        pass


async def process_conversion(task_id: str, book_path: Path, request: BookConversionRequest):
    """Обработка конвертации в фоне"""
    try:
        task = await task_manager.get_task(task_id)
        
        if not task:
            return
        
        out_format = request.out_format
        if hasattr(out_format, 'value'):
            out_format = out_format.value
        
        print(f"Converting with out_format: {out_format}")
        
        output_path = await audio_generator.convert_book(
            book_path,
            task,
            out_format,
            request.ref_audio,
            request.ref_text,
            request.voice_speed or 1.0,
            request.voice_pitch or 0.0,
            request.voice_volume or 1.0,
            request.chapters,
            request.merge_chapters,
            request.add_chapter_markers
        )
        
        if output_path and output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            duration = audio_generator._get_audio_duration(output_path)
            
            await task_manager.update_progress(
                task_id,
                status=TaskStatus.COMPLETED,
                progress_percent=100.0,
                message="Conversion completed",
                result_url=str(output_path)
            )
            
            if request.notify_url:
                try:
                    import httpx
                    async with httpx.AsyncClient() as client:
                        await client.post(
                            request.notify_url,
                            json={
                                "task_id": task_id,
                                "status": "completed",
                                "download_url": f"/download/{task_id}",
                                "file_size_mb": file_size_mb,
                                "duration_seconds": duration
                            }
                        )
                except Exception:
                    pass
        else:
            await task_manager.update_progress(
                task_id,
                status=TaskStatus.FAILED,
                error="No output generated"
            )
    
    except Exception as e:
        await task_manager.update_progress(
            task_id,
            status=TaskStatus.FAILED,
            error=str(e)
        )
