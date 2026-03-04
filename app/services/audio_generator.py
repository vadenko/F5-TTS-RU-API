import asyncio
import glob
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import AsyncGenerator

import aiofiles
import requests
import subprocess

from app.core.config import (
    BOOKS_DIR,
    DEVICE,
    OUTPUT_DIR,
    TEMP_DIR,
    MODEL_REPO,
    MODEL_PATH,
    CKPT_FILE,
    VOCAB_FILE,
    NFE_STEPS,
)
from app.models.schemas import OutputFormat, TaskStatus
from app.services.fb2_parser import FB2Parser
from app.services.task_manager import Task, task_manager


class AudioGenerator:
    """Генератор аудио из текста"""
    
    def __init__(self, max_chunk_chars: int = 2000):
        self.parser = FB2Parser(max_chunk_chars=max_chunk_chars)
        self._check_dependencies()
        self._model_downloaded = False
    
    def _check_dependencies(self):
        """Проверка наличия необходимых утилит"""
        if shutil.which("f5-tts_infer-cli") is None:
            raise RuntimeError("f5-tts_infer-cli not found in PATH")
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found in PATH")
    
    def _get_model_paths(self) -> tuple[str, str]:
        """Получение путей к модели"""
        print(f"=== _get_model_paths ===")
        print(f"CKPT_FILE: {CKPT_FILE}")
        print(f"VOCAB_FILE: {VOCAB_FILE}")
        print(f"MODEL_PATH: {MODEL_PATH}")
        print(f"======================")
        
        if CKPT_FILE and VOCAB_FILE:
            ckpt_path = CKPT_FILE
            vocab_path = VOCAB_FILE
            
            if not os.path.isfile(ckpt_path):
                raise RuntimeError(f"Custom ckpt file not found: {ckpt_path}")
            
            if not os.path.isfile(vocab_path):
                print(f"vocab.txt not found at {vocab_path}, downloading...")
                vocab_path = self._download_vocab(os.path.dirname(ckpt_path))
            
            return ckpt_path, vocab_path
        
        if MODEL_PATH:
            base_path = Path(MODEL_PATH)
            ckpt_path = str(base_path / "model_last_inference.safetensors")
            vocab_path = str(base_path / "vocab.txt")
            if not os.path.isfile(ckpt_path):
                ckpt_path = str(base_path / "model_last.pt")
            if not os.path.isfile(ckpt_path):
                raise RuntimeError(f"Model file not found in: {MODEL_PATH}")
            if not os.path.isfile(vocab_path):
                print(f"vocab.txt not found at {vocab_path}, downloading...")
                vocab_path = self._download_vocab(base_path)
            return ckpt_path, vocab_path
        
        from huggingface_hub import snapshot_download
        
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        
        if not self._model_downloaded:
            print(f"Downloading model {MODEL_REPO}...")
            try:
                snapshot_download(
                    repo_id=MODEL_REPO,
                    cache_dir=cache_dir,
                    local_dir_use_symlinks=False
                )
                self._model_downloaded = True
                print("Model downloaded successfully")
            except Exception as e:
                print(f"Error downloading model: {e}")
        
        snapshot_glob = os.path.join(cache_dir, f"models--{MODEL_REPO.replace('/', '--')}/snapshots/*")
        snapshot_dirs = sorted(glob.glob(snapshot_glob), key=os.path.getmtime, reverse=True)
        
        if not snapshot_dirs:
            raise RuntimeError(f"Model snapshot not found in huggingface cache at {snapshot_glob}")
        
        snapshot_dir = snapshot_dirs[0]
        ckpt_path = os.path.join(snapshot_dir, "F5TTS_v1_Base_v2/model_last_inference.safetensors")
        vocab_path = os.path.join(snapshot_dir, "F5TTS_v1_Base/vocab.txt")
        
        if not os.path.isfile(ckpt_path):
            raise RuntimeError(f"model_last_inference.safetensors not found: {ckpt_path}")
        if not os.path.isfile(vocab_path):
            vocab_path = self._download_vocab(snapshot_dir)
        
        return ckpt_path, vocab_path
    
    def _download_vocab(self, target_dir: str | Path) -> str:
        """Скачивание vocab.txt из оригинального репозитория F5-TTS"""
        import urllib.request
        
        vocab_path = Path(target_dir) / "vocab.txt"
        
        if vocab_path.exists():
            return str(vocab_path)
        
        vocab_urls = [
            "https://huggingface.co/swivid/F5-TTS/resolve/main/F5TTS_Base/vocab.txt",
            "https://raw.githubusercontent.com/SWivid/F5-TTS/main/f5tts_base/vocab.txt",
        ]
        
        try:
            print("Downloading vocab.txt...")
            for url in vocab_urls:
                try:
                    import urllib.request
                    urllib.request.urlretrieve(url, vocab_path)
                    print(f"vocab.txt downloaded from {url}")
                    return str(vocab_path)
                except Exception as e1:
                    print(f"Failed to download from {url}: {e1}")
                    continue
            raise RuntimeError("Failed to download vocab.txt from all sources")
        except Exception as e:
            raise RuntimeError(f"Failed to download vocab.txt: {e}")
    
    def _clean_text_for_tts(self, text: str) -> str:
        """Очистка текста от markdown и спецсимволов для TTS"""
        import re
        # Убираем markdown-разметку
        text = re.sub(r'\*\*', '', text)  # **жирный**
        text = re.sub(r'__', '', text)    # __жирный__
        # Заменяем переносы строк на пробелы
        text = text.replace('\n', ' ').replace('\r', ' ')
        # Убираем лишние пробелы
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _process_with_ruaccent(self, text: str) -> str:
        """Обработка текста (ruaccent отключен из-за проблем совместимости)"""
        # Очищаем текст
        text = self._clean_text_for_tts(text)
        print(f"=== Processing text ({len(text)} chars) ===")
        print(f"First 100 chars: {text[:100]}")
        # Возвращаем текст без ударений (F5-TTS справляется и так)
        return text + ' '
    
    async def _generate_audio_chunk(
        self,
        text: str,
        output_path: Path,
        ckpt_path: str,
        vocab_path: str,
        ref_audio: str | None = None,
        ref_text: str | None = None,
        speed: float = 1.0
    ) -> bool:
        """Генерация аудио для одного чанка"""
        text_with_accents = self._process_with_ruaccent(text)
        
        output_path = output_path.absolute()
        
        # Ограничиваем длину текста для стабильности
        max_len = 200
        if len(text_with_accents) > max_len:
            print(f"Warning: Text too long ({len(text_with_accents)}), truncating to {max_len}")
            text_with_accents = text_with_accents[:max_len]

        cmd = [
            "f5-tts_infer-cli",
            "--ckpt_file", str(ckpt_path),
            "--vocab_file", str(vocab_path),
            "--gen_text", text_with_accents,
            "--output_dir", str(output_path.parent),
            "--output_file", output_path.name,
            "--device", DEVICE,
            "--nfe_step", str(NFE_STEPS)  # Количество шагов денойзинга
        ]

        # Не используем кастомный ref_audio для русского TTS
        # Модель F5-TTS_RUSSIAN имеет встроенный референс
        if speed != 1.0:
            cmd += ["--speed", str(speed)]
        
        print(f"Output path: {output_path}")
        print(f"Text length: {len(text_with_accents)} chars")

        # Создаем директорию заранее
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Directory created: {output_path.parent.exists()}")

        # Проверяем, что f5-tts_infer-cli доступен
        import shutil
        if not shutil.which("f5-tts_infer-cli"):
            print("ERROR: f5-tts_infer-cli not found in PATH")
            return False

        print(f"Running: {' '.join(cmd[:6])}...")
        print("Starting TTS generation (this may take several minutes on CPU)...")

        try:
            import subprocess
            import asyncio
            # Увеличиваем таймаут для CPU (2 часа на чанк)
            timeout = 7200 if DEVICE == "cpu" else 600

            # Запускаем в отдельном потоке, чтобы не блокировать event loop
            def run_tts():
                return subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=str(output_path.parent),
                    shell=True
                )

            # Запускаем в executor и периодически проверяем
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, run_tts)
            
            print(f"Return code: {result.returncode}")
            if result.stdout:
                print(f"STDOUT: {result.stdout[:500]}")
            if result.stderr:
                print(f"STDERR: {result.stderr[:500]}")
            
            if result.returncode != 0:
                print(f"TTS generation failed")
                return False
            
            print(f"Checking for file: {output_path}")
            print(f"File exists: {output_path.exists()}")
            
            if not output_path.exists():
                wav_files = list(Path(output_path.parent).glob("*.wav"))
                print(f"WAV files in dir: {wav_files}")
                for f in wav_files:
                    print(f"  {f.name} exists: {f.exists()}")
            
            return output_path.exists()
        except subprocess.TimeoutExpired:
            print("TTS generation timed out")
            return False
        except Exception as e:
            print(f"TTS generation error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _merge_audio_files(
        self,
        input_files: list[Path],
        output_path: Path,
        format: str = "mp3"
    ) -> Path:
        """Объединение аудиофайлов в один"""
        concat_list = TEMP_DIR / f"concat_{int(time.time())}.txt"
        
        with open(concat_list, 'w', encoding='utf-8') as f:
            for file_path in input_files:
                f.write(f"file '{file_path.absolute()}'\n")
        
        if format == "mp3":
            output_path = output_path.with_suffix('.mp3')
            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(concat_list),
                "-c:a", "libmp3lame", "-b:a", "192k",
                str(output_path)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(concat_list),
                "-c:a", "pcm_s16le",
                str(output_path)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        concat_list.unlink(missing_ok=True)
        return output_path
    
    def _get_audio_duration(self, audio_path: Path) -> float:
        """Получение длительности аудио в секундах"""
        try:
            result = subprocess.run([
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path)
            ], capture_output=True, text=True)
            return float(result.stdout.strip()) if result.stdout.strip() else 0.0
        except Exception:
            return 0.0
    
    async def convert_book(
        self,
        book_path: Path | str,
        task: Task,
        out_format: str = "mp3",
        ref_audio: str | None = None,
        ref_text: str | None = None,
        speed: float = 1.0,
        voice_pitch: float = 0.0,
        voice_volume: float = 1.0,
        chapters: list[int] | None = None,
        merge: bool = True,
        add_markers: bool = True
    ) -> Path | None:
        """Конвертация книги в аудио"""
        if isinstance(book_path, str):
            book_path = Path(book_path)
        
        print(f"Starting conversion: {book_path}")
        print(f"Out format: {out_format}, speed: {speed}, pitch: {voice_pitch}, volume: {voice_volume}")
        
        out_format_str = out_format if isinstance(out_format, str) else str(out_format)
        
        try:
            # Читаем в бинарном режиме и определяем кодировку
            raw_content = book_path.read_bytes()
            from app.services.fb2_parser import detect_encoding
            encoding = detect_encoding(raw_content)
            content = raw_content.decode(encoding)
            book = self.parser.parse_content(content)
        except Exception as e:
            print(f"Error parsing book: {e}")
            raise

        print(f"Book title: {book.metadata.title}, chapters: {len(book.chapters)}")

        # Debug: show first chapter text
        if book.chapters:
            print(f"=== First chapter debug ===")
            print(f"Chapter 1 title: {book.chapters[0].title}")
            sample_text = book.chapters[0].text[:200]
            print(f"Chapter 1 text (first 200 chars): {repr(sample_text)}")
            print(f"Chapter 1 text length: {len(book.chapters[0].text)}")

            # Debug: count <p> tags in first chapter
            import xml.etree.ElementTree as ET
            raw_content = book_path.read_bytes()
            encoding = detect_encoding(raw_content)
            fb2_content = raw_content.decode(encoding)
            root = ET.fromstring(fb2_content)
            p_tags = root.findall('.//{http://www.gribuser.ru/xml/fictionbook/2/0}p')
            if not p_tags:
                p_tags = root.findall('.//p')
            print(f"Total <p> tags in FB2: {len(p_tags)}")
            if p_tags:
                print(f"First <p> text: {p_tags[0].text[:100] if p_tags[0].text else 'None'}")
            print(f"===========================")
        
        if chapters:
            book.chapters = [ch for ch in book.chapters if ch.number in chapters]
        
        if not book.chapters:
            raise ValueError("No chapters to process")
        
        ckpt_path, vocab_path = self._get_model_paths()
        
        task.total_chapters = len(book.chapters)
        await task_manager.update_progress(
            task.id,
            status=TaskStatus.PROCESSING,
            total_chapters=len(book.chapters),
            message="Starting conversion..."
        )
        
        book_output_dir = TEMP_DIR / task.id
        book_output_dir.mkdir(parents=True, exist_ok=True)
        
        chapter_files = []
        total_chunks = 0
        
        for chapter_idx, chapter in enumerate(book.chapters):
            await task_manager.update_progress(
                task.id,
                current_chapter=chapter_idx + 1,
                message=f"Processing chapter: {chapter.title}"
            )
            
            chunks = self.parser.split_into_chunks(chapter.text)
            chapter_dir = book_output_dir / f"chapter_{chapter.number}"
            chapter_dir.mkdir(parents=True, exist_ok=True)
            
            chapter_audio_files = []
            task.total_chunks = len(chunks)
            
            for chunk_idx, chunk_text in enumerate(chunks):
                await task_manager.update_progress(
                    task.id,
                    current_chunk=chunk_idx + 1,
                    total_chunks=len(chunks),
                    progress_percent=((chapter_idx + chunk_idx/len(chunks)) / len(book.chapters)) * 100,
                    message=f"Chapter {chapter.number}, chunk {chunk_idx + 1}/{len(chunks)}"
                )
                
                chunk_file = chapter_dir / f"chunk_{chunk_idx:04d}.wav"
                success = await self._generate_audio_chunk(
                    chunk_text,
                    chunk_file,
                    ckpt_path,
                    vocab_path,
                    ref_audio,
                    ref_text,
                    speed
                )
                
                if success and chunk_file.exists():
                    chapter_audio_files.append(chunk_file)
                    total_chunks += 1
            
            if chapter_audio_files:
                chapter_output = book_output_dir / f"chapter_{chapter.number}.wav"
                merged = self._merge_audio_files(chapter_audio_files, chapter_output, OutputFormat.WAV)
                chapter_files.append((chapter.number, chapter.title, merged))
        
        if not chapter_files:
            await task_manager.update_progress(
                task.id,
                status=TaskStatus.FAILED,
                error="No audio files generated"
            )
            return None
        
        if merge and len(chapter_files) > 1:
            final_audio_path = OUTPUT_DIR / f"{book.metadata.id}_{int(time.time())}"
            
            if add_markers:
                await self._add_chapter_markers(chapter_files, final_audio_path, out_format_str)
            else:
                audio_files = [f[2] for f in chapter_files]
                final_audio_path = self._merge_audio_files(
                    audio_files,
                    final_audio_path,
                    out_format_str
                )
        else:
            _, _, single_file = chapter_files[0]
            final_audio_path = single_file.with_suffix(f".{out_format_str}")
            if single_file.suffix != f".{out_format_str}":
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(single_file),
                    str(final_audio_path)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        shutil.rmtree(book_output_dir, ignore_errors=True)
        
        return final_audio_path
    
    async def _add_chapter_markers(
        self,
        chapters: list[tuple[int, str, Path]],
        output_path: Path,
        out_format: str
    ):
        """Добавление маркеров глав"""
        concat_list = TEMP_DIR / f"concat_{int(time.time())}.txt"
        
        with open(concat_list, 'w', encoding='utf-8') as f:
            for num, title, path in chapters:
                f.write(f"file '{path.absolute()}'\n")
        
        if out_format == "mp3":
            output_path = output_path.with_suffix('.mp3')
            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(concat_list),
                "-c:a", "libmp3lame", "-b:a", "192k",
                str(output_path)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(concat_list),
                "-c:a", "pcm_s16le",
                str(output_path)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        concat_list.unlink(missing_ok=True)


audio_generator = AudioGenerator()
