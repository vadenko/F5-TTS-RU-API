import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import aiofiles


def detect_encoding(data: bytes) -> str:
    """Определение кодировки файла"""
    try:
        data.decode('utf-8')
        return 'utf-8'
    except UnicodeDecodeError:
        pass
    
    try:
        data.decode('cp1251')
        return 'cp1251'
    except UnicodeDecodeError:
        pass
    
    try:
        data.decode('koi8-r')
        return 'koi8-r'
    except UnicodeDecodeError:
        pass
    
    return 'utf-8'


@dataclass
class Chapter:
    """Глава книги"""
    number: int
    title: str
    text: str
    sections: list[str] = field(default_factory=list)


@dataclass
class BookMetadata:
    """Метаданные книги"""
    id: str = ""
    title: str = "Без названия"
    author: str = "Неизвестный автор"
    author_first_name: str = ""
    author_last_name: str = ""
    lang: str = "ru"
    year: str = ""
    publisher: str = ""
    series: str = ""
    sequence: str = ""
    sequence_number: str = ""
    description: str = ""
    cover_image: bytes | None = None


@dataclass
class Book:
    """Структура книги"""
    metadata: BookMetadata
    chapters: list[Chapter] = field(default_factory=list)
    raw_text: str = ""


class FB2Parser:
    """Парсер FictionBook 2 формата"""
    
    FB_NS = 'http://www.gribuser.ru/xml/fictionbook/2/0'
    
    def __init__(self, max_chunk_chars: int = 2000):
        self.max_chunk_chars = max_chunk_chars
        self.ns = {'fb': self.FB_NS}
        self.ns_registered = False
        self.use_ns = True
    
    def _register_ns(self, root: ET.Element):
        """Определение namespace"""
        ns = root.tag.split('}')[0].strip('{') if '}' in root.tag else ''
        if ns:
            self.ns = {'fb': ns}
            self.ns_registered = True
            self.use_ns = True
        else:
            self.use_ns = False
    
    async def parse_file(self, file_path: str | Path) -> Book:
        """Парсинг FB2 файла с автоопределением кодировки"""
        async with aiofiles.open(file_path, 'rb') as f:
            raw_content = await f.read()

        encoding = detect_encoding(raw_content)
        content = raw_content.decode(encoding)
        return self.parse_content(content)
    
    def parse_content(self, content: str) -> Book:
        """Парсинг содержимого FB2"""
        root = ET.fromstring(content)
        
        self._register_ns(root)
        print(f"Using namespace: {self.use_ns}, ns: {self.ns}")
        
        metadata = self._parse_metadata(root)
        chapters = self._parse_body(root)
        
        return Book(
            metadata=metadata,
            chapters=chapters,
            raw_text=self._extract_raw_text(chapters)
        )
    
    def _find_all(self, element: ET.Element, tag: str) -> list[ET.Element]:
        """Поиск всех элементов с учётом/без namespace"""
        if self.use_ns:
            return element.findall(f'{{}}{tag}'.format(self.ns['fb']) if self.ns['fb'] else tag)
        else:
            return element.findall(tag)
    
    def _find_first(self, element: ET.Element, tag: str) -> ET.Element | None:
        """Поиск первого элемента с учётом/без namespace"""
        if self.use_ns:
            return element.find(f'{{}}{tag}'.format(self.ns['fb']) if self.ns['fb'] else tag)
        else:
            return element.find(tag)
    
    def _xpath(self, element: ET.Element, path: str) -> list[ET.Element]:
        """XPath с учётом namespace"""
        if self.use_ns:
            return element.findall(path, self.ns)
        else:
            return element.findall(path)
    
    def _find(self, element: ET.Element, path: str) -> ET.Element | None:
        """Find с учётом namespace"""
        if self.use_ns:
            return element.find(path, self.ns)
        else:
            return element.find(path)
    
    def _findtext(self, element: ET.Element, path: str, default: str = "") -> str:
        """Findtext с учётом namespace"""
        if self.use_ns:
            el = element.find(path, self.ns)
        else:
            el = element.find(path)
        if el is not None and el.text:
            return el.text
        return default
        if el is not None and el.text:
            return el.text
        return default
    
    def _parse_metadata(self, root: ET.Element) -> BookMetadata:
        """Извлечение метаданных из FB2"""
        metadata = BookMetadata(id=str(uuid.uuid4()))
        
        description = self._find(root, './/fb:description')
        if description is None:
            description = root.find('.//description')
        
        if description is None:
            return metadata
        
        title_info = self._find(description, './/fb:title-info')
        if title_info is None:
            title_info = description.find('.//title-info')
        
        if title_info is not None:
            metadata.lang = self._findtext(title_info, 'fb:lang', 'ru') or 'ru'
            
            author = self._find(title_info, './/fb:author')
            if author is None:
                author = title_info.find('.//author')
            
            if author is not None:
                first_name = self._findtext(author, 'fb:first-name', '') or ''
                last_name = self._findtext(author, 'fb:last-name', '') or ''
                
                if not first_name and not last_name:
                    author_text = author.text or ""
                    if author_text.strip():
                        parts = author_text.strip().split()
                        if len(parts) >= 2:
                            first_name = parts[0]
                            last_name = ' '.join(parts[1:])
                        else:
                            first_name = author_text.strip()
                
                metadata.author_first_name = first_name
                metadata.author_last_name = last_name
                metadata.author = f"{first_name} {last_name}".strip() or "Неизвестный автор"
            
            book_title = self._findtext(title_info, 'fb:book-title', '')
            if not book_title:
                title_el = title_info.find('.//book-title')
                if title_el is not None and title_el.text:
                    book_title = title_el.text
            
            metadata.title = book_title or "Без названия"
            
            metadata.year = self._findtext(title_info, 'fb:year', '')
            
            metadata.publisher = self._findtext(title_info, 'fb:publisher', '')
            
            sequence = self._find(title_info, './/fb:sequence')
            if sequence is None:
                sequence = title_info.find('.//sequence')
            
            if sequence is not None:
                metadata.series = sequence.get('name', '')
                metadata.sequence_number = sequence.get('number', '')
            
            annotation = self._find(title_info, './/fb:annotation')
            if annotation is None:
                annotation = title_info.find('.//annotation')
            
            if annotation is not None:
                metadata.description = self._flatten_text(annotation)
        
        binary = self._find(root, './/fb:binary[@type="image/jpeg"]')
        if binary is None:
            binary = root.find('.//binary[@type="image/jpeg"]')
        if binary is None:
            binary = self._find(root, './/fb:binary[@type="image/png"]')
        if binary is None:
            binary = root.find('.//binary[@type="image/png"]')
        
        if binary is not None and binary.text:
            try:
                import base64
                metadata.cover_image = base64.b64decode(binary.text)
            except Exception:
                pass
        
        return metadata
    
    def _parse_body(self, root: ET.Element) -> list[Chapter]:
        """Извлечение глав из тела книги"""
        chapters = []
        chapter_number = 0

        body = self._find(root, './/fb:body')
        if body is None:
            body = root.find('.//body')

        if body is None:
            return chapters

        # Собираем все секции рекурсивно
        all_sections = []

        def collect_sections(elem, depth=0):
            """Рекурсивный сбор секций с отслеживанием глубины"""
            for child in elem:
                tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                if tag == 'section':
                    all_sections.append((child, depth))
                    collect_sections(child, depth + 1)

        collect_sections(body)

        # Определяем структуру: если есть вложенные секции,
        # берем только самые глубокие (те, что содержат текст)
        if all_sections:
            max_depth = max(s[1] for s in all_sections)
            # Берем секции на максимальной глубине или с заголовками
            candidate_sections = []
            for section, depth in all_sections:
                text = self._extract_section_text(section)
                if text.strip():
                    has_title = self._extract_title(section) is not None
                    # Приоритет: секции с заголовками и текстом на любой глубине
                    # или секции с текстом на максимальной глубине
                    if has_title or depth == max_depth:
                        candidate_sections.append(section)

            # Убираем дубликаты (если родитель и ребенок оба подходят)
            valid_sections = []
            for section in candidate_sections:
                # Проверяем, не является ли эта секция родителем другой кандидатской
                is_parent = False
                for other in candidate_sections:
                    if other is not section:
                        # Проверяем, является ли other потомком section
                        parent = other
                        while parent is not None:
                            parent = self._get_parent(body, parent)
                            if parent is section:
                                is_parent = True
                                break
                if not is_parent:
                    valid_sections.append(section)

            for section in valid_sections:
                chapter_number += 1
                title = self._extract_title(section)
                if not title:
                    title = f"Глава {chapter_number}"
                text = self._extract_section_text(section)

                if text.strip():
                    chapters.append(Chapter(
                        number=chapter_number,
                        title=title,
                        text=text
                    ))

        if not chapters:
            # Fallback: весь текст как одна глава
            all_text = self._extract_section_text(body)
            if all_text.strip():
                chapters.append(Chapter(
                    number=1,
                    title="Полный текст",
                    text=all_text.strip()
                ))

        return chapters

    def _get_parent(self, root: ET.Element, child: ET.Element) -> ET.Element | None:
        """Найти родительский элемент"""
        for elem in root.iter():
            for subelem in elem:
                if subelem is child:
                    return elem
        return None
    
    def _extract_title(self, section: ET.Element) -> str | None:
        """Извлечение заголовка секции"""
        title_elem = self._find(section, './/fb:title')
        if title_elem is None:
            title_elem = section.find('.//title')
        
        if title_elem is not None:
            return self._flatten_text(title_elem).strip()
        
        p_elems = section.findall('fb:p', self.ns)
        if not p_elems:
            p_elems = section.findall('p')
        
        if p_elems:
            first_p = p_elems[0]
            text = self._flatten_text(first_p).strip()
            if text and len(text) < 100:
                return text
        
        return None
    
    def _extract_section_text(self, section: ET.Element) -> str:
        """Извлечение текста секции - рекурсивно"""
        text_parts = []

        def has_title(elem) -> bool:
            """Проверяет, есть ли у элемента заголовок title"""
            for child in elem:
                child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                if child_tag == 'title':
                    return True
            return False

        def process_element(elem, is_top_level=True):
            """Рекурсивная обработка элемента"""
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

            if tag == 'section':
                # Если это вложенная секция с заголовком - пропускаем (отдельная глава)
                if not is_top_level and has_title(elem):
                    return
                # Иначе обрабатываем содержимое (группировка без заголовка)
                for child in elem:
                    process_element(child, is_top_level=False)

            elif tag == 'p':
                text = self._flatten_text(elem).strip()
                if text:
                    text_parts.append(text)

            elif tag == 'empty-line':
                text_parts.append('')

            elif tag in ('cite', 'poem', 'stanza', 'epigraph'):
                # Обрабатываем цитаты, стихи, эпиграфы
                for child in elem:
                    process_element(child, is_top_level=False)

            elif tag == 'title':
                # Заголовки добавляем как отдельный параграф
                text = self._flatten_text(elem).strip()
                if text:
                    text_parts.append(f"**{text}**")

            else:
                # Остальные элементы - рекурсивно
                for child in elem:
                    process_element(child, is_top_level=False)

        # Начинаем обработку с прямых детей секции
        for elem in section:
            process_element(elem, is_top_level=True)

        return '\n\n'.join(text_parts)
    
    def _flatten_text(self, element: ET.Element) -> str:
        """Извлечение текста из элемента и его потомков"""
        if element.text:
            text = element.text
        else:
            text = ""
        
        for child in element:
            text += self._flatten_text(child)
            if child.tail:
                text += child.tail
        
        return text
    
    def _extract_raw_text(self, chapters: list[Chapter]) -> str:
        """Извлечение простого текста без разметки"""
        parts = []
        for ch in chapters:
            parts.append(f"{ch.title}\n\n{ch.text}")
        return '\n\n'.join(parts)
    
    def split_into_chunks(self, text: str) -> list[str]:
        """Разбиение текста на чанки для TTS"""
        if len(text) <= self.max_chunk_chars:
            return [text]
        
        chunks = []
        paragraphs = re.split(r'\n\s*\n', text)
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_chunk) + len(para) + 2 <= self.max_chunk_chars:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                if len(para) > self.max_chunk_chars:
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    current_chunk = ""
                    for sent in sentences:
                        if len(current_chunk) + len(sent) + 1 <= self.max_chunk_chars:
                            current_chunk += sent + " "
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sent + " "
                else:
                    current_chunk = para + "\n\n"
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks


def clean_text(text: str) -> str:
    """Очистка текста от FB2-разметки и артефактов"""
    text = re.sub(r'\[\[.*?\]\]', '', text)
    text = re.sub(r'\{[^}]+\}', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()
