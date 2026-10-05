"""EPUB 解析器：提取标题层级 + 段落。

对齐 RAG_SYSTEM_DESIGN §8.1：
- ebooklib / bs4 提取 OEBPS 目录结构；
- 遍历 .xhtml 文件 → 解析 <h1~h6> 标题层级 + <p>/<li> 段落；
- 按标题层级生成 Chapter / Section，段落收集到 raw_paragraphs；
- 无明确章节结构时降级（§8.3）：未命名章节 + has_structure=False。

输出：Document（含 Chapter→Section→raw_paragraphs 层级），供 chunker 切分。
"""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path

from app.rag.chunker import Chapter, Document, Section, detect_language

logger = logging.getLogger(__name__)

_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_PARAGRAPH_TAGS = {"p", "li", "blockquote"}

# 索引/术语表章节标题特征（Jung/Freud 英文版 INDEX、中文版术语表/词汇表）
_INDEX_HEADING_RE = re.compile(
    r"(术语表|术语汇编|词汇表|译名对照|关键词表|\bindex\b|\bglossary\b)",
    re.IGNORECASE,
)
# 索引条目段落 CSS 类特征（Jung EPUB：vol_01_indexmain1 / vol_01_indexsub 等）
_INDEX_P_CLASS_RE = re.compile(r"(indexmain|indexsub|glossary)", re.IGNORECASE)
# 术语后括注的英文对应词，如 "白板（tabula rasa）"、"阿波罗式的[Apollonian（Nietzsche）]"
# 按括号类型分别配对，避免嵌套括号（[Apollonian（Nietzsche）]）在内部全角括号处提前截断
_EN_PAREN_PATTERNS = [
    re.compile(r"\[([^\[\]]+)\]"),
    re.compile(r"（([^（）]+)）"),
    re.compile(r"\(([^()]+)\)"),
]
_CJK_RE = re.compile(r"[一-鿿]")


def _extract_index_entry(p_el) -> dict | None:
    """从索引/术语表段落 ``<p>`` 提取结构化术语条目。

    支持两种 EPUB 术语形态：
    - 页码索引型（Jung/Freud 英文 Collected Works）：术语文本 + ``<a>页码</a>`` 链接，
      如 ``<p class="vol_01_indexmain1">abnormality, emotional, <a>119</a>, <a>134</a></p>``；
    - 术语定义型（中文哲学书术语表）：``<b>术语</b>（英文）：定义``，
      如 ``<p><b>白板</b>（tabula rasa）：洛克哲学中的术语……</p>``。

    Returns:
        ``{"term", "synonym", "pages", "definition", "is_sub"}`` 或 None（字母分组/噪声行）。
    """
    from bs4 import BeautifulSoup

    full = " ".join(p_el.get_text(" ").split())
    if not full or len(full) < 2:
        return None

    cls = " ".join(p_el.get("class", []) or [])
    is_sub = "indexsub" in cls.lower()

    anchors = p_el.find_all("a")
    pages = [a.get_text(strip=True) for a in anchors if re.search(r"\d", a.get_text() or "")]

    if pages:
        # 页码索引型：移除 <a> 后剩余文本即术语本身
        clone = BeautifulSoup(str(p_el), "html.parser")
        for a in clone.find_all("a"):
            a.extract()
        term = " ".join(clone.get_text(" ").split()).strip(" ,:;，、")
        if not term or len(term) < 2:
            return None
        return {
            "term": term,
            "synonym": None,
            "pages": ", ".join(pages),
            "definition": "",
            "is_sub": is_sub,
        }

    # 无页码：术语定义型（或字母分组标题 A/B/C）
    bold = p_el.find(["b", "strong"])
    if not bold:
        return None
    term = " ".join(bold.get_text(" ").split()).strip(" ,:;，、")
    if not term or len(term) < 2:
        return None

    # 提取英文对应词：按文档顺序取第一个含拉丁字母的括注（跳过"（尼采）"这类纯中文括注）
    synonym = None
    candidates = []
    for pat in _EN_PAREN_PATTERNS:
        for m in pat.finditer(full):
            candidates.append((m.start(), m.group(1).strip()))
    for _, cand in sorted(candidates):
        if re.search(r"[A-Za-z]{2,}", cand) and not _CJK_RE.search(cand):
            # 归一化全角括号为半角
            synonym = cand.replace("（", "(").replace("）", ")")
            break

    return {
        "term": term,
        "synonym": synonym,
        "pages": "",
        "definition": full,
        "is_sub": is_sub,
    }


def _domain_from_path(file_path: str) -> str:
    """从文件所在目录名推断 domain（Freud/Jung/Philosophy → freud/jung/philosophy）。"""
    parent = Path(file_path).parent.name.lower()
    mapping = {"freud": "freud", "jung": "jung", "philosophy": "philosophy"}
    return mapping.get(parent, "philosophy")


def _clean_text(text: str) -> str:
    """清理 HTML 提取文本的空白。"""
    return " ".join(text.split())


def parse_epub(file_path: str, domain: str | None = None) -> Document:
    """解析 EPUB，返回层级 Document。

    Args:
        file_path: EPUB 文件路径
        domain: 领域覆盖（None 时从路径推断）
    """
    from ebooklib import epub, ITEM_DOCUMENT  # type: ignore
    from bs4 import BeautifulSoup

    path = Path(file_path)
    book = epub.read_epub(str(path), options={"ignore_ncx": True})

    domain = domain or _domain_from_path(file_path)
    # 书名：优先 metadata title，退化用文件名
    title = ""
    try:
        title = book.get_metadata("DC", "Title")[0][0] if book.get_metadata("DC", "Title") else ""
    except Exception:  # noqa: BLE001
        pass
    if not title:
        title = path.stem

    language = "zh"
    try:
        lang_meta = book.get_metadata("DC", "Language")
        if lang_meta:
            language = "zh" if str(lang_meta[0][0]).lower().startswith("zh") else "en"
    except Exception:  # noqa: BLE001
        pass

    doc = Document(
        document_id=f"doc_{uuid.uuid4().hex[:12]}",
        title=title,
        domain=domain,
        language=language,
        file_path=str(path),
    )

    current_chapter: Chapter | None = None
    current_section: Section | None = None
    has_any_heading = False
    # 是否处于 INDEX/术语表 章节：该区域段落不入正文，收为 index_entries
    in_index = False

    def _flush_section() -> None:
        nonlocal current_section
        if current_section is not None and current_section.raw_paragraphs:
            if current_chapter is not None:
                current_chapter.sections.append(current_section)
            else:
                # 无章节时创建未命名章节兜底
                fallback = Chapter(chapter_id=uuid.uuid4().hex, title="未命名章节")
                fallback.sections.append(current_section)
                doc.chapters.append(fallback)
        current_section = None

    def _flush_chapter() -> None:
        nonlocal current_chapter
        _flush_section()
        if current_chapter is not None and current_chapter.sections:
            doc.chapters.append(current_chapter)
        current_chapter = None

    for item in book.get_items_of_type(ITEM_DOCUMENT):
        try:
            content = item.get_content()
            soup = BeautifulSoup(content, "html.parser")
        except Exception as e:  # noqa: BLE001
            logger.warning("[epub_parser] 解析 item %s 失败：%s", getattr(item, "name", "?"), e)
            continue

        # 跳过空或纯导航页
        body = soup.find("body") or soup
        if body is None:
            continue

        for el in body.descendants:
            if not hasattr(el, "name") or el.name is None:
                continue

            if el.name in _HEADING_TAGS:
                has_any_heading = True
                heading_text = _clean_text(el.get_text())
                if not heading_text:
                    continue
                level = int(el.name[1])  # h1→1
                is_index_heading = bool(_INDEX_HEADING_RE.search(heading_text))

                if is_index_heading:
                    # 进入索引/术语表章节：收尾正文，后续段落收为 index_entries
                    _flush_chapter()
                    in_index = True
                    continue
                if in_index:
                    if level <= 2:
                        # 索引区结束（下一卷/下一章主标题），回落正常解析
                        in_index = False
                    else:
                        # 索引区内的子标题（如 GENERAL INDEX 卷号分节），保持模式
                        continue

                if level <= 2:
                    # 新章节：先 flush 旧章节
                    _flush_chapter()
                    current_chapter = Chapter(
                        chapter_id=uuid.uuid4().hex,
                        title=heading_text,
                    )
                    # 同步开一个新节
                    current_section = Section(
                        section_id=uuid.uuid4().hex,
                        title=heading_text,
                        chapter_id=current_chapter.chapter_id,
                    )
                else:
                    # h3+ 视为新节
                    _flush_section()
                    ch = current_chapter or Chapter(chapter_id=uuid.uuid4().hex, title="未命名章节")
                    if not current_chapter:
                        current_chapter = ch
                    current_section = Section(
                        section_id=uuid.uuid4().hex,
                        title=heading_text,
                        chapter_id=current_chapter.chapter_id,
                    )

            elif el.name in _PARAGRAPH_TAGS:
                text = _clean_text(el.get_text())
                if not text or len(text) < 2:
                    continue
                # 索引/术语表区域：不入正文，结构化提取为术语条目
                p_cls = " ".join(el.get("class", []) or [])
                if in_index or _INDEX_P_CLASS_RE.search(p_cls):
                    entry = _extract_index_entry(el)
                    if entry:
                        doc.index_entries.append(entry)
                        doc.index_text += text + "\n"
                    continue
                # 语言检测（首次段落决定文档语言，修正 metadata 不可靠的情况）
                if doc.language == "zh":
                    doc.language = detect_language(text)
                if current_section is None:
                    # 无标题结构：兜底未命名章节+节
                    if current_chapter is None:
                        current_chapter = Chapter(chapter_id=uuid.uuid4().hex, title="未命名章节")
                    current_section = Section(
                        section_id=uuid.uuid4().hex,
                        title="未命名节",
                        chapter_id=current_chapter.chapter_id,
                    )
                current_section.raw_paragraphs.append(text)

    # flush 收尾
    _flush_chapter()

    if not has_any_heading:
        doc.has_structure = False

    # 全空文档兜底
    if not doc.chapters:
        doc.has_structure = False
        ch = Chapter(chapter_id=uuid.uuid4().hex, title="未命名章节")
        sec = Section(section_id=uuid.uuid4().hex, title="未命名节", chapter_id=ch.chapter_id)
        ch.sections.append(sec)
        doc.chapters.append(ch)

    logger.info(
        "[epub_parser] %s 解析完成：%d 章，结构=%s，语言=%s，术语条目=%d",
        path.name, len(doc.chapters), doc.has_structure, doc.language, len(doc.index_entries),
    )
    return doc
