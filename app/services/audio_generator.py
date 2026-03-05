import asyncio
import glob
import logging
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

# Простой логгер в файл для Colab
LOG_FILE_PATH = Path("/content/F5-TTS-RU-API/data/debug.log")

def log_msg(msg):
    """Запись лога в файл"""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {msg}\n"
    try:
        LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
            f.write(log_line)
    except:
        pass
    print(log_line.strip())

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
        
        from huggingface_hub import hf_hub_download

        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

        if not self._model_downloaded:
            log_msg(f"Downloading model {MODEL_REPO} (F5TTS_v1_Base_v2)...")
            try:
                # Скачиваем только нужные файлы из F5TTS_v1_Base_v2
                ckpt_path = hf_hub_download(
                    repo_id=MODEL_REPO,
                    filename="F5TTS_v1_Base_v2/model_last_inference.safetensors",
                    cache_dir=cache_dir,
                    local_dir_use_symlinks=False
                )
                vocab_path = hf_hub_download(
                    repo_id=MODEL_REPO,
                    filename="F5TTS_v1_Base/vocab.txt",
                    cache_dir=cache_dir,
                    local_dir_use_symlinks=False
                )
                log_msg(f"Downloaded ckpt: {ckpt_path}")
                log_msg(f"Downloaded vocab: {vocab_path}")
                log_msg(f"ckpt exists: {os.path.exists(ckpt_path)}")
                log_msg(f"vocab exists: {os.path.exists(vocab_path)}")
                self._model_downloaded = True
                log_msg("Model downloaded successfully")
                return ckpt_path, vocab_path
            except Exception as e:
                log_msg(f"Error downloading model: {e}")
                raise

        # Если уже скачено, ищем в кэше
        try:
            ckpt_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename="F5TTS_v1_Base_v2/model_last_inference.safetensors",
                cache_dir=cache_dir,
                local_dir_use_symlinks=False
            )
            vocab_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename="F5TTS_v1_Base/vocab.txt",
                cache_dir=cache_dir,
                local_dir_use_symlinks=False
            )
            return ckpt_path, vocab_path
        except Exception as e:
            raise RuntimeError(f"Model files not found: {e}")
    
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

        # Проверяем пути к модели
        ckpt_path = Path(ckpt_path).absolute()
        vocab_path = Path(vocab_path).absolute()
        log_msg(f"Using ckpt: {ckpt_path} (exists: {ckpt_path.exists()})")
        log_msg(f"Using vocab: {vocab_path} (exists: {vocab_path.exists()})")

        # Формируем команду (shell=False автоматически экранирует аргументы)
        cmd = [
            "f5-tts_infer-cli",
            "--ckpt_file", str(ckpt_path),
            "--vocab_file", str(vocab_path),
            "--gen_text", text_with_accents[:150],  # Ограничиваем длину текста
            "--output_dir", str(output_path.parent),
            "--output_file", output_path.name,
            "--device", DEVICE,
            "--nfe_step", str(NFE_STEPS)
        ]

        if speed != 1.0:
            cmd += ["--speed", str(speed)]

        log_msg(f"Command args: ckpt={ckpt_path.name}, vocab={vocab_path.name}, text_len={len(text_with_accents)}")
        
        log_msg(f"Output path: {output_path}")
        log_msg(f"Text length: {len(text_with_accents)} chars")

        # Создаем директорию заранее
        output_path.parent.mkdir(parents=True, exist_ok=True)
        log_msg(f"Directory created: {output_path.parent.exists()}")

        # Проверяем, что f5-tts_infer-cli доступен
        import shutil
        if not shutil.which("f5-tts_infer-cli"):
            log_msg("ERROR: f5-tts_infer-cli not found in PATH")
            return False

        print(f"Running: {' '.join(cmd[:6])}...")
        print("Starting TTS generation (this may take several minutes on CPU)...")

        try:
            import subprocess
            import asyncio
            # Увеличиваем таймаут для CPU (2 часа на чанк)
            timeout = 7200 if DEVICE == "cpu" else 600

            # Запускаем в отдельном потоке, чтобы не блокировать event loop
            # shell=False для правильной обработки аргументов с пробелами
            def run_tts():
                return subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=str(output_path.parent),
                    shell=False
                )

            # Запускаем в executor и периодически проверяем
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, run_tts)
            
            log_msg(f"Return code: {result.returncode}")
            if result.stdout:
                log_msg(f"STDOUT: {result.stdout[:200]}")
            if result.stderr:
                log_msg(f"STDERR: {result.stderr[:200]}")

            if result.returncode != 0:
                log_msg(f"TTS generation failed with code {result.returncode}")
                return False

            log_msg(f"Checking for file: {output_path}")
            log_msg(f"File exists: {output_path.exists()}")

            if not output_path.exists():
                wav_files = list(Path(output_path.parent).glob("*.wav"))
                log_msg(f"WAV files in dir: {wav_files}")

            return output_path.exists()
        except subprocess.TimeoutExpired:
            log_msg("TTS generation timed out")
            return False
        except Exception as e:
            log_msg(f"TTS generation error: {e}")
            import traceback
            log_msg(traceback.format_exc()[:500])
            return False
    
    def _merge_audio_files(
        self,
        input_files: list[Path],
        output_path: Path,
        format: str = "mp3"
    ) -> Path:
        """Объединение аудиофайлов в один"""
        log_msg(f"_merge_audio_files: {len(input_files)} files, format={format}")
        for i, f in enumerate(input_files):
            log_msg(f"  Input {i}: {f} (exists: {f.exists()})")

        concat_list = TEMP_DIR / f"concat_{int(time.time())}.txt"

        with open(concat_list, 'w', encoding='utf-8') as f:
            for file_path in input_files:
                f.write(f"file '{file_path.absolute()}'\n")

        log_msg(f"Concat list created: {concat_list}")

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
        
        log_msg(f"ffmpeg result: {output_path} (exists: {output_path.exists()})")
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
        
        log_msg(f"Starting conversion: {book_path}")
        log_msg(f"Out format: {out_format}, speed: {speed}")
        
        out_format_str = out_format if isinstance(out_format, str) else str(out_format)
        
        try:
            # Читаем в бинарном режиме и определяем кодировку
            raw_content = book_path.read_bytes()
            from app.services.fb2_parser import detect_encoding
            encoding = detect_encoding(raw_content)
            content = raw_content.decode(encoding)
            book = self.parser.parse_content(content)
        except Exception as e:
            log_msg(f"Error parsing book: {e}")
            raise

        log_msg(f"Book title: {book.metadata.title}, chapters: {len(book.chapters)}")

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
                
                print(f"Chunk {chunk_idx} result: success={success}, file_exists={chunk_file.exists()}")
                if success and chunk_file.exists():
                    chapter_audio_files.append(chunk_file)
                    total_chunks += 1
                    print(f"✓ Chunk {chunk_idx} added to chapter files")

            print(f"Chapter {chapter.number}: {len(chapter_audio_files)}/{len(chunks)} chunks successful")

            if chapter_audio_files:
                print(f"Merging {len(chapter_audio_files)} audio files...")
                chapter_output = book_output_dir / f"chapter_{chapter.number}.wav"
                merged = self._merge_audio_files(chapter_audio_files, chapter_output, OutputFormat.WAV)
                print(f"Merged chapter file: {merged} (exists: {merged.exists()})")
                chapter_files.append((chapter.number, chapter.title, merged))
            else:
                print(f"✗ No audio files generated for chapter {chapter.number}")
            
            if chapter_audio_files:
                chapter_output = book_output_dir / f"chapter_{chapter.number}.wav"
                merged = self._merge_audio_files(chapter_audio_files, chapter_output, OutputFormat.WAV)
                chapter_files.append((chapter.number, chapter.title, merged))
        
        log_msg(f"Total chapters processed: {len(chapter_files)}")
        for num, title, path in chapter_files:
            log_msg(f"  Chapter {num}: {path} (exists: {path.exists()})")

        if not chapter_files:
            await task_manager.update_progress(
                task.id,
                status=TaskStatus.FAILED,
                error="No audio files generated"
            )
            return None

        log_msg(f"Merging {len(chapter_files)} chapters, merge={merge}, add_markers={add_markers}")

        if merge and len(chapter_files) > 1:
            final_audio_path = OUTPUT_DIR / f"{book.metadata.id}_{int(time.time())}"
            log_msg(f"Final output path: {final_audio_path}")

            if add_markers:
                log_msg("Adding chapter markers...")
                final_audio_path = await self._add_chapter_markers(chapter_files, final_audio_path, out_format_str)
            else:
                log_msg("Merging audio files...")
                audio_files = [f[2] for f in chapter_files]
                final_audio_path = self._merge_audio_files(
                    audio_files,
                    final_audio_path,
                    out_format_str
                )
            log_msg(f"Merge result: {final_audio_path} (exists: {final_audio_path.exists()})")
        else:
            _, _, single_file = chapter_files[0]
            final_audio_path = single_file.with_suffix(f".{out_format_str}")
            log_msg(f"Single file output: {final_audio_path}")
            if single_file.suffix != f".{out_format_str}":
                log_msg(f"Converting {single_file} to {out_format_str}")
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(single_file),
                    str(final_audio_path)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        log_msg(f"Final file exists: {final_audio_path.exists()}")
        if final_audio_path.exists():
            log_msg(f"Final file size: {final_audio_path.stat().st_size} bytes")

        shutil.rmtree(book_output_dir, ignore_errors=True)
        log_msg(f"Cleaned up temp dir: {book_output_dir}")

        return final_audio_path if final_audio_path.exists() else None
    
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
