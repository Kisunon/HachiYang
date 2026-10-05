"""层级分块：Document → Chapter → Section → ParagraphBlock。

对齐 RAG_SYSTEM_DESIGN §3.2：
- 尊重原书段落边界，不跨段落硬切；
- 单 ParagraphBlock 目标 300~500 中文字 / 80~150 英文词；
- 超长段在句号边界软切，子块保留父段落引用；
- 父子关系：ParagraphBlock.parent_section_id → Section.id → Chapter.id → Document.id。

数据结构（Document/Chapter/Section/ParagraphBlock）由 parser 构建，
chunk_document() 负责把 Section.raw_paragraphs 切分为合规 ParagraphBlock。
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# 目标块长度区间
_ZH_MIN, _ZH_MAX = 300, 500
_EN_MIN, _EN_MAX = 80, 150

# CJK 字符检测
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def detect_language(text: str) -> str:
    """粗略语言检测：含 CJK 字符判 zh，否则 en。"""
    if not text:
        return "zh"
    sample = text[:500]
    cjk_count = len(_CJK_RE.findall(sample))
    # CJK 占比 > 15% 判为中文
    return "zh" if cjk_count / max(len(sample), 1) > 0.15 else "en"


def _word_count_en(text: str) -> int:
    return len(text.split())


def _char_count_zh(text: str) -> int:
    return len(text)


# ==================== 层级数据结构 ====================

@dataclass
class ParagraphBlock:
    """向量化最小单元（对齐 RAG §3.2 元数据 Schema，v1.6 含 6 级溯源）。"""

    block_id: str
    document_id: str
    domain: str
    language: str
    chapter_title: str
    section_title: str
    page_range: str | None
    block_index_in_section: int
    char_count: int
    block_type: str = "text"          # text / image / section_summary / chapter_summary
    text: str = ""
    has_image: bool = False
    image_refs: list[str] = field(default_factory=list)
    has_structure: bool = True
    parent_block_ids: list[str] | None = None   # 摘要块：链式展开目标 id 列表
    # 段落级父引用（软切分子块时指向同一段落组）
    parent_section_id: str | None = None
    # ---- v1.6 三阶段流水线：6 级溯源 + 提取稿定位 ----
    volume_title: str = ""
    volume_index: int = 1
    subsection_title: str = ""
    source_anchor_in_md: str = ""
    extracted_md_path: str = ""
    parent_chapter_summary_id: str | None = None  # 节摘要 → 所属章摘要（检查页跳转）

    def to_metadata(self) -> dict:
        """生成 ChromaDB metadata + SQLite 存储所需的扁平字典。"""
        return {
            "block_id": self.block_id,
            "document_id": self.document_id,
            "domain": self.domain,
            "language": self.language,
            "volume_title": self.volume_title or "",
            "volume_index": self.volume_index or 1,
            "chapter_title": self.chapter_title or "",
            "section_title": self.section_title or "",
            "subsection_title": self.subsection_title or "",
            "page_range": self.page_range or "",
            "block_index_in_section": self.block_index_in_section,
            "char_count": self.char_count,
            "block_type": self.block_type,
            "has_structure": 1 if self.has_structure else 0,
            "source_anchor_in_md": self.source_anchor_in_md or "",
            "extracted_md_path": self.extracted_md_path or "",
            "parent_chapter_summary_id": self.parent_chapter_summary_id or "",
        }


@dataclass
class Section:
    """节 / 小节。"""

    section_id: str
    title: str
    page_range: str | None = None
    raw_paragraphs: list[str] = field(default_factory=list)
    has_structure: bool = True
    chapter_id: str | None = None

    def total_text(self) -> str:
        return "\n".join(p for p in self.raw_paragraphs if p.strip())


@dataclass
class Chapter:
    """章 / 卷。"""

    chapter_id: str
    title: str
    page_range: str | None = None
    sections: list[Section] = field(default_factory=list)


@dataclass
class Document:
    """整本书（解析后的中间结构）。"""

    document_id: str
    title: str
    domain: str
    language: str
    file_path: str
    content_hash: str = ""
    chapters: list[Chapter] = field(default_factory=list)
    has_structure: bool = True
    # 末尾术语索引页文本（Freud/Jung 英文版自带），供 term_builder 第一阶段提取；为空表示无
    index_text: str = ""
    # EPUB 结构化术语条目（parser 从 INDEX/术语表 章节提取，PDF 解析器留空走 index_text 正则）
    # 每项：{"term": str, "synonym": str|None, "pages": str, "definition": str, "is_sub": bool}
    index_entries: list[dict] = field(default_factory=list)


# ==================== 句子边界软切 ====================

# 中英文句末标点
_SENT_END_RE = re.compile(r"[。！？!?…]\s*")


def _soft_split(text: str, language: str) -> list[str]:
    """超长段落按句末边界软切，使每块落入目标长度区间。

    不会跨句子硬切；若单句本身超长，原样保留（不强制切）。
    """
    text = text.strip()
    if not text:
        return []

    if language == "zh":
        max_len = _ZH_MAX
    else:
        max_len = _EN_MAX * 5  # 英文按字符近似换算（词数×5）

    if len(text) <= max_len:
        return [text]

    # 按句末标点切句
    sentences: list[str] = []
    last = 0
    for m in _SENT_END_RE.finditer(text):
        sentences.append(text[last : m.end()].strip())
        last = m.end()
    if last < len(text):
        tail = text[last:].strip()
        if tail:
            sentences.append(tail)

    if len(sentences) <= 1:
        # 无法在句边界切，原样返回（不硬切）
        return [text]

    # 贪心合并句子到不超过 max_len
    chunks: list[str] = []
    buf = ""
    for s in sentences:
        if not s:
            continue
        if buf and len(buf) + len(s) > max_len:
            chunks.append(buf)
            buf = s
        else:
            buf = (buf + s) if buf else s
    if buf:
        chunks.append(buf)
    return chunks if chunks else [text]


# ==================== 主分块入口 ====================

def chunk_document(doc: Document) -> list[ParagraphBlock]:
    """把 Document 的 Section.raw_paragraphs 切分为合规 ParagraphBlock。

    返回全部 text 类型 block（不含 section_summary，后者由 summarizer 生成）。
    """
    blocks: list[ParagraphBlock] = []

    for chapter in doc.chapters:
        for section in chapter.sections:
            section_lang = doc.language or "zh"
            idx = 0
            for para in section.raw_paragraphs:
                para = para.strip()
                if not para:
                    continue
                # 语言检测（段落级，更精确）
                section_lang = detect_language(para)
                pieces = _soft_split(para, section_lang)
                for piece in pieces:
                    piece = piece.strip()
                    if not piece:
                        continue
                    blocks.append(
                        ParagraphBlock(
                            block_id=uuid.uuid4().hex,
                            document_id=doc.document_id,
                            domain=doc.domain,
                            language=section_lang,
                            chapter_title=chapter.title,
                            section_title=section.title,
                            page_range=section.page_range,
                            block_index_in_section=idx,
                            char_count=len(piece),
                            block_type="text",
                            text=piece,
                            has_structure=doc.has_structure,
                            parent_section_id=section.section_id,
                        )
                    )
                    idx += 1

            if not blocks and not section.raw_paragraphs:
                continue

    logger.info(
        "[chunker] %s 分块完成：%d 个 ParagraphBlock（%d 章）",
        doc.document_id, len(blocks), len(doc.chapters),
    )
    return blocks


def group_blocks_by_section(blocks: list[ParagraphBlock]) -> dict[str, list[ParagraphBlock]]:
    """按 (document_id, chapter_title, section_title) 分组，供章节摘要生成。"""
    groups: dict[str, list[ParagraphBlock]] = {}
    for b in blocks:
        key = f"{b.document_id}\x00{b.chapter_title}\x00{b.section_title}"
        groups.setdefault(key, []).append(b)
    return groups


# ==================== 阶段 2：切片落盘（v1.6 三阶段流水线） ====================
#
# 对齐 RAG_SYSTEM_DESIGN §3.1.1 阶段 2：
# - 输入：阶段 1 产物 knowledge_extracted/<域>/<卷名>.extracted.json
#   （outline_tree + contents，纯本地、零 API 配额）；
# - 输出：knowledge_chunks/<域>/<卷名>.chunks.json，每块携带 6 级溯源元数据
#   （书名 / 卷 / 章 / 节 / 小节 / 节内序号）+ 原文锚点 source_anchor_in_md；
# - 切片规则复用 _soft_split：尊重段落边界，超长段句号边界软切；
# - block_id 在本阶段生成（uuid），阶段 3 Embedding 入库时直接沿用，保证跨步骤一致。

CHUNKS_SCHEMA_VERSION = 2


@dataclass
class ChunkVolumeResult:
    """单卷切片结果。"""

    volume_title: str
    chunks_path: Path
    block_count: int = 0
    section_count: int = 0
    avg_chars: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "volume_title": self.volume_title,
            "chunks_path": str(self.chunks_path),
            "block_count": self.block_count,
            "section_count": self.section_count,
            "avg_chars": round(self.avg_chars, 1),
            "error": self.error,
        }


@dataclass
class ChunkReport:
    """阶段 2 总报告。"""

    volumes: list[ChunkVolumeResult] = field(default_factory=list)

    @property
    def total_blocks(self) -> int:
        return sum(v.block_count for v in self.volumes if not v.error)

    def to_dict(self) -> dict:
        return {
            "volume_count": len(self.volumes),
            "block_count": self.total_blocks,
            "volumes": [v.to_dict() for v in self.volumes],
        }


def _chunks_out_path(extracted_json_path: Path, out_root: Path, domain: str) -> Path:
    """由 extracted.json 路径推导 chunks.json 路径（文件名 .extracted.json → .chunks.json）。"""
    name = extracted_json_path.name
    if name.endswith(".extracted.json"):
        stem = name[: -len(".extracted.json")]
    else:
        stem = extracted_json_path.stem
    return out_root / domain / f"{stem}.chunks.json"


def chunk_extracted_volume(extracted_json_path: str | Path,
                           out_dir: str | Path | None = None,
                           settings=None) -> ChunkVolumeResult:
    """对单个 extracted.json 执行阶段 2 切片，落盘 chunks.json。

    纯本地、零 API 配额。
    """
    from app.config import PROJECT_ROOT, get_settings

    settings = settings or get_settings()
    src = Path(extracted_json_path)
    if not src.is_absolute():
        src = PROJECT_ROOT / src
    if not src.exists():
        raise FileNotFoundError(str(src))

    payload = json.loads(src.read_text(encoding="utf-8"))
    source_doc = payload.get("source_document", {})
    volume_meta = payload.get("volume", {})
    contents = payload.get("contents", [])

    book_title = source_doc.get("book_title", "")
    domain = source_doc.get("domain", "philosophy")
    content_hash = source_doc.get("content_hash", "")
    volume_index = volume_meta.get("volume_index", 1)
    volume_title = volume_meta.get("volume_title", book_title)

    out_root = Path(out_dir) if out_dir else settings.knowledge_chunks_path
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root
    out_path = _chunks_out_path(src, out_root, domain)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 按 (chapter, section) 分组，contents 已是树顺序（同组内顺序即原文顺序）
    blocks: list[dict] = []
    groups: dict[tuple[str, str], list[dict]] = {}
    group_order: list[tuple[str, str]] = []
    for item in contents:
        ref = item.get("path_ref", {})
        key = (ref.get("chapter", ""), ref.get("section", ""))
        if key not in groups:
            groups[key] = []
            group_order.append(key)
        groups[key].append(item)

    for key in group_order:
        chapter_title, section_title = key
        items = groups[key]
        block_index = 0
        for item in items:
            ref = item.get("path_ref", {})
            para_text = (item.get("text") or "").strip()
            if not para_text:
                continue
            language = detect_language(para_text)
            pieces = _soft_split(para_text, language)
            para_order = item.get("paragraph_order_in_section", 0)
            anchor = item.get("anchor", "")
            for piece in pieces:
                piece = piece.strip()
                if not piece:
                    continue
                blocks.append({
                    "block_id": uuid.uuid4().hex,
                    "block_type": "text",
                    "parent_ref": {
                        "book_title": book_title,
                        "volume_title": volume_title,
                        "volume_index": volume_index,
                        "chapter_title": chapter_title,
                        "section_title": section_title,
                        "subsection_title": ref.get("subsection", ""),
                    },
                    "block_index_in_section": block_index,
                    "text": piece,
                    "char_count": len(piece),
                    "language": language,
                    "source_anchor_in_md": anchor,
                    "soft_split_from_paragraphs": [para_order],
                })
                block_index += 1

    total_chars = sum(b["char_count"] for b in blocks)
    avg_chars = total_chars / len(blocks) if blocks else 0.0

    chunks_payload = {
        "schema_version": CHUNKS_SCHEMA_VERSION,
        "source_ref": {
            "book_title": book_title,
            "domain": domain,
            "volume_index": volume_index,
            "volume_title": volume_title,
            "content_hash": content_hash,
            "extracted_json": str(src.resolve()),
            "extracted_md": str(
                src.with_name(src.name[: -len(".extracted.json")] + ".extracted.md").resolve()
            ),
        },
        "blocks": blocks,
        "stats": {
            "total_blocks": len(blocks),
            "total_chapter_summary_blocks": 0,   # 阶段 3 生成
            "total_section_summary_blocks": 0,   # 阶段 3 生成
            "section_count": len(group_order),
            "avg_chars": round(avg_chars, 1),
        },
    }
    out_path.write_text(
        json.dumps(chunks_payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    logger.info(
        "[chunker] 阶段 2 切片完成：《%s》→ %d blocks / %d 节（avg %d 字）→ %s",
        volume_title, len(blocks), len(group_order), avg_chars, out_path.name,
    )
    return ChunkVolumeResult(
        volume_title=volume_title,
        chunks_path=out_path,
        block_count=len(blocks),
        section_count=len(group_order),
        avg_chars=avg_chars,
    )


def run_chunk_stage(extracted_dir: str | Path | None = None,
                    out_dir: str | Path | None = None,
                    settings=None) -> ChunkReport:
    """阶段 2：扫描 knowledge_extracted/ 下全部 *.extracted.json → 切片落盘。

    纯本地、零 API 配额。
    """
    from app.config import PROJECT_ROOT, get_settings

    settings = settings or get_settings()
    in_root = Path(extracted_dir) if extracted_dir else settings.knowledge_extracted_path
    if not in_root.is_absolute():
        in_root = PROJECT_ROOT / in_root
    if not in_root.exists():
        raise FileNotFoundError(f"阶段 1 产物目录不存在：{in_root}")

    json_files = sorted(in_root.rglob("*.extracted.json"))
    logger.info("[chunker] 阶段 2 开始：%d 个 extracted.json（%s）", len(json_files), in_root)

    report = ChunkReport(volumes=[])
    for jf in json_files:
        try:
            result = chunk_extracted_volume(jf, out_dir=out_dir, settings=settings)
            report.volumes.append(result)
        except Exception as e:  # noqa: BLE001
            logger.exception("[chunker] 切片失败：%s", jf.name)
            report.volumes.append(ChunkVolumeResult(
                volume_title=jf.stem, chunks_path=jf, error=str(e),
            ))

    logger.info(
        "[chunker] 阶段 2 完成：%d 卷 / %d blocks",
        len(report.volumes), report.total_blocks,
    )
    return report
