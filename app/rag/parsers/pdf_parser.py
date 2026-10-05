"""PDF 解析器：Outline 书签树 + 页文本 + 术语索引页。

对齐 RAG_SYSTEM_DESIGN §8.2：
- pymupdf 提取 Outline（书签树）→ 映射为 Chapter/Section 层级；
- 按页文本 + 页码 → 段落收集；
- 末尾术语索引页（供 term_builder 第一阶段）；
- 无 Outline 时降级（§8.3）：按页 + 段落，has_structure=False。

输出：Document（含 Chapter→Section→raw_paragraphs），doc.index_text 为术语索引文本。
"""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path

from app.rag.chunker import Chapter, Document, Section, detect_language

logger = logging.getLogger(__name__)


def _domain_from_path(file_path: str) -> str:
    parent = Path(file_path).parent.name.lower()
    mapping = {"freud": "freud", "jung": "jung", "philosophy": "philosophy"}
    return mapping.get(parent, "philosophy")


def _split_paragraphs(page_text: str) -> list[str]:
    """把一页文本切成段落（按空行/换行分段，过滤短行噪声）。"""
    if not page_text:
        return []
    # 先按双换行分段，再按单换行兜底
    chunks = re.split(r"\n\s*\n", page_text)
    paragraphs: list[str] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        # 合并换行内的短行为一段
        merged = " ".join(line.strip() for line in chunk.splitlines() if line.strip())
        merged = merged.strip()
        if len(merged) >= 2:
            paragraphs.append(merged)
    # 若双换行切不出段（整页连成块），按单换行兜底
    if not paragraphs and page_text.strip():
        lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
        buf = ""
        for ln in lines:
            if buf:
                buf += " " + ln
            else:
                buf = ln
        if buf:
            paragraphs.append(buf.strip())
    return paragraphs


def parse_pdf(file_path: str, domain: str | None = None) -> Document:
    """解析 PDF，返回层级 Document。

    doc.index_text 为末尾术语索引页文本（如有）。
    """
    import pymupdf  # noqa: F401  （避免 fitz 弃用警告，统一用 pymupdf 模块）

    path = Path(file_path)
    pdf = pymupdf.open(str(path))

    domain = domain or _domain_from_path(file_path)
    title = path.stem
    # 元数据 title
    try:
        meta_title = pdf.metadata.get("title") if pdf.metadata else None
        if meta_title and meta_title.strip():
            title = meta_title.strip()
    except Exception:  # noqa: BLE001
        pass

    language = "en"  # Freud/Jung 英文版默认 en，按内容修正
    doc = Document(
        document_id=f"doc_{uuid.uuid4().hex[:12]}",
        title=title,
        domain=domain,
        language=language,
        file_path=str(path),
    )

    total_pages = len(pdf)
    toc = pdf.get_toc()  # [[level, title, page_num], ...] page_num 从 1 开始

    # ---- 识别术语索引页（outline 含 "Index" 或末尾页）----
    index_pages: set[int] = set()
    index_section_start = None
    for level, ttext, pnum in toc:
        if re.search(r"\bindex\b", ttext, re.IGNORECASE) or "索引" in ttext:
            index_section_start = pnum
            break
    if index_section_start is not None:
        for p in range(index_section_start, total_pages + 1):
            index_pages.add(p)
    elif total_pages > 0:
        # 无 Index 标记时，检测末尾几页是否像索引（大量短行 + 页码引用）
        for p in range(max(1, total_pages - 20), total_pages + 1):
            try:
                txt = pdf[p - 1].get_text()
            except Exception:  # noqa: BLE001
                continue
            # 索引页特征：行短、含大量逗号/页码数字
            lines = [ln for ln in txt.splitlines() if ln.strip()]
            short_lines = sum(1 for ln in lines if len(ln.strip()) < 60)
            if len(lines) > 10 and short_lines / len(lines) > 0.7:
                index_pages.add(p)

    # ---- 构建 Chapter/Section 层级 ----
    if toc:
        doc.has_structure = True
        # outline 条目 → 章节
        # level 1 → Chapter，level 2+ → Section（同一 chapter 下）
        current_chapter: Chapter | None = None
        current_section: Section | None = None

        for i, (level, ttitle, pnum) in enumerate(toc):
            # 该条目覆盖的页范围 [pnum, next_pnum)
            next_pnum = total_pages + 1
            for j in range(i + 1, len(toc)):
                if toc[j][0] <= level:
                    next_pnum = toc[j][2]
                    break
            page_range = f"{pnum}-{next_pnum - 1}" if next_pnum > pnum + 1 else f"{pnum}"

            if level <= 1:
                if current_section is not None and current_chapter is not None:
                    current_chapter.sections.append(current_section)
                if current_chapter is not None:
                    doc.chapters.append(current_chapter)
                current_chapter = Chapter(
                    chapter_id=uuid.uuid4().hex,
                    title=ttitle.strip(),
                    page_range=page_range,
                )
                current_section = None
            else:
                if current_chapter is None:
                    current_chapter = Chapter(chapter_id=uuid.uuid4().hex, title="未命名章节")
                if current_section is not None:
                    current_chapter.sections.append(current_section)
                current_section = Section(
                    section_id=uuid.uuid4().hex,
                    title=ttitle.strip(),
                    page_range=page_range,
                    chapter_id=current_chapter.chapter_id,
                )

            # 收集该条目页的段落（非索引页）
            for p in range(pnum, next_pnum):
                if p in index_pages:
                    continue
                try:
                    page_text = pdf[p - 1].get_text()
                except Exception:  # noqa: BLE001
                    continue
                for para in _split_paragraphs(page_text):
                    if doc.language == "en":
                        doc.language = detect_language(para)
                    if current_section is None:
                        current_section = Section(
                            section_id=uuid.uuid4().hex,
                            title=ttitle.strip() or "未命名节",
                            chapter_id=current_chapter.chapter_id if current_chapter else uuid.uuid4().hex,
                        )
                    current_section.raw_paragraphs.append(para)

        # flush 收尾
        if current_section is not None and current_chapter is not None:
            current_chapter.sections.append(current_section)
        if current_chapter is not None:
            doc.chapters.append(current_chapter)
    else:
        # 无 Outline 降级：按页收集段落
        doc.has_structure = False
        ch = Chapter(chapter_id=uuid.uuid4().hex, title="未命名章节")
        sec = Section(section_id=uuid.uuid4().hex, title="未命名节", chapter_id=ch.chapter_id)
        for p in range(1, total_pages + 1):
            if p in index_pages:
                continue
            try:
                page_text = pdf[p - 1].get_text()
            except Exception:  # noqa: BLE001
                continue
            for para in _split_paragraphs(page_text):
                if doc.language == "en":
                    doc.language = detect_language(para)
                sec.raw_paragraphs.append(para)
        ch.sections.append(sec)
        doc.chapters.append(ch)

    # ---- 收集术语索引页文本 ----
    index_texts: list[str] = []
    for p in sorted(index_pages):
        try:
            index_texts.append(pdf[p - 1].get_text())
        except Exception:  # noqa: BLE001
            continue
    doc.index_text = "\n".join(index_texts)

    pdf.close()

    logger.info(
        "[pdf_parser] %s 解析完成：%d 章，结构=%s，索引页=%d，语言=%s",
        path.name, len(doc.chapters), doc.has_structure, len(index_pages), doc.language,
    )
    return doc
