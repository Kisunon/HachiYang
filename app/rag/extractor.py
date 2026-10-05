"""阶段 1：文档解析 + 按卷提取落盘（三阶段流水线第一阶段，v1.6）。

对齐 RAG_SYSTEM_DESIGN §3.1.1：
- 纯本地、零 API 配额：EPUB 解析 → 标题层级 + 段落线性流；
- 多卷合订本自动检测并**按卷拆分**，每卷独立落盘：
  - ``knowledge_extracted/<域>/<卷名>.extracted.md``：人工可读提取稿（标题层级 + 段落原文）；
  - ``knowledge_extracted/<域>/<卷名>.extracted.json``：结构化产物（outline_tree + contents）；
- 检测不到卷级结构时**整书单卷落盘**（降级规则，不做猜测性拆分）；
- 卷首/卷末行政页（版权信息/编委会/编译说明/目录/参考文献/索引等）剔除并计数，
  不进入 contents（避免 12 卷合订本中编委会等重复内容污染后续切片与 Embedding）。

对外接口：
  run_extract_stage(kb_dir=None, out_dir=None) -> ExtractReport
  extract_book(file_path, out_dir=None) -> list[VolumeExtractResult]
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.config import PROJECT_ROOT, Settings, get_settings
from app.rag.chunker import detect_language

logger = logging.getLogger(__name__)

EXTRACTED_SCHEMA_VERSION = 2

# ==================== 标题/正文元素流 ====================

_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_PARAGRAPH_TAGS = {"p", "li", "blockquote"}


@dataclass
class _Elem:
    """线性文档流元素：标题或正文段落（按 EPUB 文档顺序）。"""

    kind: str  # "h" 标题 | "p" 段落
    text: str
    level: int = 0  # 标题级别 1~6（kind="h" 时有效）


# ==================== 卷检测规则（§3.1.1 自动识别） ====================

# A 类：显式卷/册标题（锚定开头，避免章标题中提及"第 N 卷"造成误匹配）
_VOLUME_TITLE_RE = re.compile(
    r"^\s*(第\s*[一二三四五六七八九十百零0-9]+\s*[卷册]|Volume\s*\d+|Book\s*\d+)",
    re.IGNORECASE,
)

# 卷首行政页标志（标题包含即判定）
_FRONT_MATTER_RE = re.compile(
    r"编委会|编译说明|编译者|版本出处|编译分工|卷次划分|出版说明|版权信息|版权页|请支持正版"
    r"|总目录|^目录$|目录$|插图目录|插页图目录|内文插图目录"
)
# 卷末/书末参考资料标志（标题包含即判定，其后正文直到下个标题/卷边界一并剔除）
_BACK_MATTER_RE = re.compile(
    r"参考文献|参考书目|推荐读物|推荐阅读|主要参考|术语表|术语汇编|词汇表|译名对照|关键词表"
    r"|^索\s*引$|索引页"
)
# C 类卷边界：h1 卷名后紧跟的编委会/编译说明标志
_VOLUME_FRONT_MARKER_RE = re.compile(r"编委会|编译说明|编译者")

# 判定为多卷的最少边界数
_MIN_VOLUME_BOUNDARIES = 2
# 卷区间有效正文字数低于此值视为空区间（前置版权页等并入丢弃）
_EMPTY_VOLUME_CHARS = 100


@dataclass
class _Boundary:
    """卷边界：元素流下标 + 卷名（无显式卷名时为 None）。"""

    elem_index: int
    title: str | None
    rule: str  # "A" 显式卷标题 | "B" 重复版权页 | "C" h1卷名+编委会


# ==================== 落盘数据结构 ====================

@dataclass
class VolumeExtractResult:
    """单卷提取结果。"""

    volume_index: int
    volume_title: str
    md_path: Path
    json_path: Path
    chapter_count: int = 0
    section_count: int = 0
    subsection_count: int = 0
    paragraph_count: int = 0
    total_chars: int = 0
    skipped_front_sections: int = 0
    skipped_back_sections: int = 0

    def to_dict(self) -> dict:
        return {
            "volume_index": self.volume_index,
            "volume_title": self.volume_title,
            "md_path": str(self.md_path),
            "json_path": str(self.json_path),
            "chapter_count": self.chapter_count,
            "section_count": self.section_count,
            "subsection_count": self.subsection_count,
            "paragraph_count": self.paragraph_count,
            "total_chars": self.total_chars,
            "skipped_front_sections": self.skipped_front_sections,
            "skipped_back_sections": self.skipped_back_sections,
        }


@dataclass
class BookExtractReport:
    """单书提取报告。"""

    file_path: str
    book_title: str
    domain: str
    content_hash: str
    multi_volume: bool
    volumes: list[VolumeExtractResult] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "book_title": self.book_title,
            "domain": self.domain,
            "content_hash": self.content_hash,
            "multi_volume": self.multi_volume,
            "volume_count": len(self.volumes),
            "volumes": [v.to_dict() for v in self.volumes],
            "error": self.error,
        }


@dataclass
class ExtractReport:
    """阶段 1 总报告。"""

    books: list[BookExtractReport] = field(default_factory=list)

    @property
    def total_volumes(self) -> int:
        return sum(len(b.volumes) for b in self.books)

    @property
    def total_paragraphs(self) -> int:
        return sum(v.paragraph_count for b in self.books for v in b.volumes)

    def to_dict(self) -> dict:
        return {
            "book_count": len(self.books),
            "volume_count": self.total_volumes,
            "paragraph_count": self.total_paragraphs,
            "books": [b.to_dict() for b in self.books],
        }


# ==================== EPUB → 线性元素流 ====================

def _clean_text(text: str) -> str:
    return " ".join(text.split())


def stream_epub_elements(file_path: str) -> tuple[str, str, list[_Elem]]:
    """解析 EPUB 为线性元素流（标题 + 段落），不做层级树构建。

    Returns:
        (book_title, language, elements)
    """
    from ebooklib import epub, ITEM_DOCUMENT  # type: ignore
    from bs4 import BeautifulSoup

    path = Path(file_path)
    book = epub.read_epub(str(path), options={"ignore_ncx": True})

    title = ""
    try:
        meta = book.get_metadata("DC", "Title")
        title = meta[0][0] if meta else ""
    except Exception:  # noqa: BLE001
        pass
    if not title:
        title = path.stem

    language = "zh"
    try:
        lang_meta = book.get_metadata("DC", "Language")
        if lang_meta and not str(lang_meta[0][0]).lower().startswith("zh"):
            language = "en"
    except Exception:  # noqa: BLE001
        pass

    elems: list[_Elem] = []
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        try:
            soup = BeautifulSoup(item.get_content(), "html.parser")
        except Exception as e:  # noqa: BLE001
            logger.warning("[extractor] 解析 item 失败 %s：%s", getattr(item, "name", "?"), e)
            continue
        body = soup.find("body") or soup
        if body is None:
            continue
        for el in body.descendants:
            if not hasattr(el, "name") or el.name is None:
                continue
            if el.name in _HEADING_TAGS:
                text = _clean_text(el.get_text())
                if text:
                    elems.append(_Elem(kind="h", text=text, level=int(el.name[1])))
            elif el.name in _PARAGRAPH_TAGS:
                text = _clean_text(el.get_text())
                if text and len(text) >= 2:
                    elems.append(_Elem(kind="p", text=text))

    # 用首批正文修正语言检测（metadata 可能不可靠）
    for e in elems:
        if e.kind == "p":
            language = detect_language(e.text)
            break

    logger.info(
        "[extractor] %s 元素流提取完成：%d 个元素（标题 %d）",
        path.name, len(elems), sum(1 for e in elems if e.kind == "h"),
    )
    return title, language, elems


# ==================== 卷边界检测 ====================

def _heading_indices(elems: list[_Elem]) -> list[int]:
    return [i for i, e in enumerate(elems) if e.kind == "h"]


def _detect_boundaries(elems: list[_Elem]) -> list[_Boundary]:
    """检测多卷合订本的卷边界（A/B/C 三类规则，见 §3.1.1）。"""
    h_indices = _heading_indices(elems)
    boundaries: list[_Boundary] = []

    for pos, idx in enumerate(h_indices):
        e = elems[idx]
        text = e.text.strip()

        # A 类：显式"第 N 卷/册 / Volume N / Book N"标题
        if _VOLUME_TITLE_RE.search(text):
            boundaries.append(_Boundary(idx, text, "A"))
            continue

        # B 类：重复出现的"版权信息/版权页"标题（哲学通史 1-5 卷模式）
        if text in ("版权信息", "版权页"):
            boundaries.append(_Boundary(idx, None, "B"))
            continue

        # C 类：h1 卷名，且其后 3 个标题内出现编委会/编译说明（弗洛伊德文集模式）
        if e.level == 1:
            window = h_indices[pos + 1 : pos + 4]
            for j in window:
                if _VOLUME_FRONT_MARKER_RE.search(elems[j].text):
                    boundaries.append(_Boundary(idx, text, "C"))
                    break

    # 多卷判定：任一类规则命中 ≥2 次
    rule_counts = {"A": 0, "B": 0, "C": 0}
    for b in boundaries:
        rule_counts[b.rule] += 1
    if not any(c >= _MIN_VOLUME_BOUNDARIES for c in rule_counts.values()):
        return []

    # D 类补充边界：显式卷名模式（A/C）下，同为 h1 的续编卷可能没有重复编委会页
    # （如《释梦（下）》直接以 h1 接上卷章节）；这类 h1 卷名标题补充为边界。
    # B 类（版权页重复）模式不适用——该模式下 h1 是章标题而非卷名（如哲学通史）。
    if rule_counts["A"] >= _MIN_VOLUME_BOUNDARIES or rule_counts["C"] >= _MIN_VOLUME_BOUNDARIES:
        existing = {b.elem_index for b in boundaries}
        for idx in h_indices:
            e = elems[idx]
            if e.level != 1 or idx in existing:
                continue
            text = e.text.strip()
            if _is_front_heading(text) or _is_back_heading(text):
                continue
            boundaries.append(_Boundary(idx, text, "D"))

    # 按位置排序去重（同一位置可能多规则命中，保留信息量最大的：A > C > B）
    boundaries.sort(key=lambda b: b.elem_index)
    merged: list[_Boundary] = []
    for b in boundaries:
        if merged and merged[-1].elem_index == b.elem_index:
            if (b.rule == "A") or (b.rule == "C" and merged[-1].rule in ("B", "D")):
                merged[-1] = b
        else:
            merged.append(b)

    # 空区间过滤：边界到下一边界之间有效正文极少（如全书前置版权页），降级丢弃
    filtered: list[_Boundary] = []
    for k, b in enumerate(merged):
        end = merged[k + 1].elem_index if k + 1 < len(merged) else len(elems)
        chars = _interval_content_chars(elems, b.elem_index + 1, end)
        if chars >= _EMPTY_VOLUME_CHARS:
            filtered.append(b)
        else:
            logger.info(
                "[extractor] 边界 @%d（%s，规则 %s）区间有效正文仅 %d 字，视为前置页丢弃",
                b.elem_index, (b.title or elems[b.elem_index].text)[:20], b.rule, chars,
            )
    return filtered


def _interval_content_chars(elems: list[_Elem], start: int, end: int) -> int:
    """统计卷区间内"有效正文"字数（行政页/标题前零散段落不计）。

    用于空区间过滤：边界后初始为跳过态，遇到第一个非行政标题才开始计数，
    遇到 front/back 行政标题则跳过其下全部段落。
    """
    skip = True
    chars = 0
    for i in range(start, end):
        e = elems[i]
        if e.kind == "h":
            skip = _is_front_heading(e.text) or _is_back_heading(e.text)
        elif e.kind == "p" and not skip:
            chars += len(e.text)
    return chars


# ==================== 卷内层级树构建 ====================

@dataclass
class _Node:
    """卷内标题节点（chapter/section/subsection）。"""

    kind: str  # chapter | section | subsection
    title: str
    level: int  # 原始标题级别
    order: int
    anchor: str
    paragraphs: list[str] = field(default_factory=list)
    para_orders: list[int] = field(default_factory=list)  # 段落在 section 内的序号
    children: list["_Node"] = field(default_factory=list)
    skipped: bool = False  # front/back matter 标记


def _is_front_heading(text: str) -> bool:
    return bool(_FRONT_MATTER_RE.search(text.strip()))


def _is_back_heading(text: str) -> bool:
    return bool(_BACK_MATTER_RE.search(text.strip()))


def _build_volume_tree(elems: list[_Elem], start: int, end: int) -> tuple[list[_Node], int, int]:
    """把卷区间 [start, end) 的元素流构建为 chapter → section → subsection 树。

    - 卷首/卷末行政页（front/back matter）整节剔除并计数；
    - 基础层级 = 卷内最浅标题级别：base→chapter，base+1→section，base+2+→subsection；
    - 章下直接出现段落/小节（无节标题）时自动挂一个**隐式空标题节**（title=""），
      保证树深度统一、path_ref 的 section 字段可为空字符串；
    - 相对深度 ≥3 的标题（如 h5（一））：其后有正文则作为更深小节（路径拼接），
      无正文（图注/空标题）由 _prune 剔除。

    Returns:
        (chapters, skipped_front_count, skipped_back_count)
    """
    # 第一遍：确定基础层级（仅统计非行政页标题）
    content_levels = [
        elems[i].level
        for i in range(start, end)
        if elems[i].kind == "h"
        and not _is_front_heading(elems[i].text)
        and not _is_back_heading(elems[i].text)
    ]
    base_level = min(content_levels) if content_levels else 2

    chapters: list[_Node] = []
    cur_chapter: _Node | None = None
    cur_section: _Node | None = None
    # subsection 栈：[(rel_depth, node)]，rel_depth 2 = subsection，3+ = 更深小节
    sub_stack: list[tuple[int, _Node]] = []

    ch_order = 0
    sec_order = 0
    ss_order = 0
    para_order_in_section = 0

    skipped_front = 0
    skipped_back = 0
    # 跳过模式：进入 front/back 标题后，其下段落与更深标题全部丢弃，直到遇到
    # 同级或更浅的非行政标题
    skip_depth: int | None = None
    # 卷区间第一个标题之前的零散段落（B 类边界排除版权页标题后残留的
    # CIP/书名/ISBN 等无标题碎片）直接丢弃，不进入"未命名章"兜底
    first_heading_seen = False
    dropped_lead_paras = 0

    def _ensure_section() -> _Node:
        """章下直接出现段落/小节时，补一个隐式空标题节（title=""）。"""
        nonlocal cur_section, sec_order, para_order_in_section
        if cur_section is not None:
            return cur_section
        sec_order += 1
        para_order_in_section = 0
        node = _Node(
            kind="section", title="", level=base_level + 1,
            order=sec_order, anchor=f"#ch-{ch_order}-sec-{sec_order}",
        )
        if cur_chapter is not None:
            cur_chapter.children.append(node)
        cur_section = node
        return node

    for i in range(start, end):
        e = elems[i]
        if e.kind == "h":
            first_heading_seen = True
            text = e.text.strip()
            front = _is_front_heading(text)
            back = _is_back_heading(text)
            if front or back:
                skip_depth = e.level
                if front:
                    skipped_front += 1
                else:
                    skipped_back += 1
                continue
            if skip_depth is not None and e.level > skip_depth:
                # 行政页区域内的子标题，继续跳过
                continue
            skip_depth = None

            rel = e.level - base_level
            if rel <= 0:
                # 新章
                ch_order += 1
                sec_order = 0
                ss_order = 0
                para_order_in_section = 0
                cur_chapter = _Node(
                    kind="chapter", title=text, level=e.level,
                    order=ch_order, anchor=f"#ch-{ch_order}",
                )
                chapters.append(cur_chapter)
                cur_section = None
                sub_stack = []
            elif rel == 1:
                # 新节
                if cur_chapter is None:
                    # 节先于章出现：兜底建章
                    ch_order += 1
                    cur_chapter = _Node(
                        kind="chapter", title="未命名章", level=base_level,
                        order=ch_order, anchor=f"#ch-{ch_order}",
                    )
                    chapters.append(cur_chapter)
                sec_order += 1
                ss_order = 0
                para_order_in_section = 0
                cur_section = _Node(
                    kind="section", title=text, level=e.level,
                    order=sec_order, anchor=f"#ch-{ch_order}-sec-{sec_order}",
                )
                cur_chapter.children.append(cur_section)
                sub_stack = []
            else:
                # 小节（rel >= 2）
                if cur_chapter is None:
                    ch_order += 1
                    cur_chapter = _Node(
                        kind="chapter", title="未命名章", level=base_level,
                        order=ch_order, anchor=f"#ch-{ch_order}",
                    )
                    chapters.append(cur_chapter)
                _ensure_section()
                # 弹栈到父级
                while sub_stack and sub_stack[-1][0] >= rel:
                    sub_stack.pop()
                ss_order += 1
                parent = sub_stack[-1][1] if sub_stack else cur_section
                anchor = f"{cur_section.anchor}-ss-{ss_order}"
                node = _Node(
                    kind="subsection", title=text, level=e.level,
                    order=ss_order, anchor=anchor,
                )
                parent.children.append(node)
                sub_stack.append((rel, node))
        elif e.kind == "p":
            if skip_depth is not None:
                continue
            if not first_heading_seen:
                # 卷首无标题碎片（CIP/版权页数据/封面文字）：丢弃
                dropped_lead_paras += 1
                continue
            if cur_chapter is None:
                # 标题前的零散段落：兜底建章承接
                ch_order += 1
                cur_chapter = _Node(
                    kind="chapter", title="未命名章", level=base_level,
                    order=ch_order, anchor=f"#ch-{ch_order}",
                )
                chapters.append(cur_chapter)
            target = sub_stack[-1][1] if sub_stack else _ensure_section()
            para_order_in_section += 1
            target.paragraphs.append(e.text)
            target.para_orders.append(para_order_in_section)

    # 第三遍：移除无正文且无含正文后代的空小节（图注/空标题）
    if dropped_lead_paras:
        logger.debug(
            "[extractor] 卷区间 @%d 丢弃卷首无标题碎片 %d 段", start, dropped_lead_paras,
        )

    def _prune(node: _Node) -> bool:
        """返回 node 是否保留（有正文或有保留的后代）。"""
        kept_children = [c for c in node.children if _prune(c)]
        node.children = kept_children
        return bool(node.paragraphs) or bool(kept_children)

    pruned_chapters = [c for c in chapters if _prune(c)]
    return pruned_chapters, skipped_front, skipped_back


# ==================== 产物序列化 ====================

def _outline_node(node: _Node) -> dict:
    out: dict = {
        "level": node.kind,
        "order": node.order,
        "title": node.title,
        "anchor": node.anchor,
    }
    if node.children:
        key = "sections" if node.kind == "chapter" else "subsections"
        out[key] = [_outline_node(c) for c in node.children]
    return out


def _collect_contents(chapters: list[_Node], volume_title: str) -> tuple[list[dict], dict]:
    """遍历树生成 contents 扁平列表 + 统计。

    path_ref 始终携带 卷/章/节/小节 四级路径；段落挂在树上最深节点。
    """
    contents: list[dict] = []
    stats = {"chapters": 0, "sections": 0, "subsections": 0, "paragraphs": 0, "chars": 0}

    def _walk(node: _Node, path: dict) -> None:
        if node.kind == "chapter":
            stats["chapters"] += 1
            path = {**path, "chapter": node.title, "chapter_order": node.order}
        elif node.kind == "section":
            stats["sections"] += 1
            path = {**path, "section": node.title, "section_order": node.order}
        else:
            stats["subsections"] += 1
            # 多级小节路径拼接（如 "一、关于性对象的变异 · （一）性变态"）
            prev_sub = path.get("subsection", "")
            sub_title = f"{prev_sub} · {node.title}" if prev_sub else node.title
            path = {**path, "subsection": sub_title, "subsection_order": node.order}

        for para, p_order in zip(node.paragraphs, node.para_orders):
            contents.append({
                "path_ref": {
                    "volume": volume_title,
                    "chapter": path.get("chapter", ""),
                    "section": path.get("section", ""),
                    "subsection": path.get("subsection", ""),
                },
                "anchor": node.anchor,
                "paragraph_order_in_section": p_order,
                "text": para,
            })
            stats["paragraphs"] += 1
            stats["chars"] += len(para)

        for child in node.children:
            _walk(child, path)

    for ch in chapters:
        _walk(ch, {"volume": volume_title})

    return contents, stats


def _render_markdown(book_title: str, volume_title: str, volume_index: int,
                     chapters: list[_Node], meta: dict) -> str:
    """渲染人工可读提取稿（标题层级 + 段落原文 + 锚点）。"""
    lines: list[str] = []
    lines.append(f"# {volume_title}")
    lines.append("")
    lines.append(
        f"> 来源书籍：{book_title} ｜ 卷次：第 {volume_index} 卷 ｜ "
        f"领域：{meta['domain']} ｜ 语言：{meta['language']} ｜ content_hash：{meta['content_hash'][:12]}"
    )
    lines.append("")

    def _render_node(node: _Node, depth: int) -> None:
        if node.kind == "chapter":
            lines.append(f"## {node.title}  {{#{node.anchor.lstrip('#')}}}" if node.anchor else f"## {node.title}")
            lines.append("")
        elif node.kind == "section":
            if node.title:
                lines.append(f"### {node.title}  {{#{node.anchor.lstrip('#')}}}")
                lines.append("")
        else:
            lines.append(f"#### {node.title}  {{#{node.anchor.lstrip('#')}}}")
            lines.append("")
        for para in node.paragraphs:
            lines.append(para)
            lines.append("")
        for child in node.children:
            _render_node(child, depth + 1)

    for ch in chapters:
        _render_node(ch, 0)

    return "\n".join(lines).rstrip() + "\n"


def _safe_filename(name: str, max_len: int = 60) -> str:
    """文件名安全化：去除非法字符，空白收敛。"""
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:max_len] or "untitled"


def _content_hash(file_path: str) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _domain_from_path(file_path: str) -> str:
    parent = Path(file_path).parent.name.lower()
    return {"freud": "freud", "jung": "jung", "philosophy": "philosophy"}.get(parent, "philosophy")


# ==================== 单书提取 ====================

def extract_book(file_path: str, out_dir: str | Path | None = None,
                 settings: Settings | None = None) -> list[VolumeExtractResult]:
    """提取单本书：解析 → 卷检测/拆分 → 每卷落盘 extracted.md + extracted.json。

    多卷合订本返回多个 VolumeExtractResult（每卷一个文件对）；
    单卷书返回 1 个。
    """
    settings = settings or get_settings()
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(file_path)

    out_root = Path(out_dir) if out_dir else settings.knowledge_extracted_path
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root

    book_title, language, elems = stream_epub_elements(str(path))
    domain = _domain_from_path(str(path))
    content_hash = _content_hash(str(path))

    boundaries = _detect_boundaries(elems)
    multi_volume = len(boundaries) >= _MIN_VOLUME_BOUNDARIES

    # 切分卷区间：(start, end, title)
    volumes: list[tuple[int, int, str | None]] = []
    if multi_volume:
        for k, b in enumerate(boundaries):
            end = boundaries[k + 1].elem_index if k + 1 < len(boundaries) else len(elems)
            # 边界标题本身（版权页/h1 卷名）不入正文
            volumes.append((b.elem_index + 1, end, b.title))
        logger.info(
            "[extractor] %s 检测为多卷合订本：%d 卷（规则 %s）",
            path.name, len(volumes),
            ",".join(sorted({b.rule for b in boundaries})),
        )
    else:
        volumes.append((0, len(elems), None))
        logger.info("[extractor] %s 未检测到卷级结构，整书单卷落盘", path.name)

    domain_dir = out_root / domain
    domain_dir.mkdir(parents=True, exist_ok=True)

    results: list[VolumeExtractResult] = []
    book_stem = _safe_filename(path.stem)

    for idx, (start, end, vol_title) in enumerate(volumes, start=1):
        chapters, skipped_front, skipped_back = _build_volume_tree(elems, start, end)

        if multi_volume and vol_title:
            volume_title = f"卷{idx} {vol_title}"
            file_base = f"卷{idx:02d}-{_safe_filename(vol_title)}"
        elif multi_volume:
            volume_title = f"卷{idx}"
            file_base = f"卷{idx:02d}-{book_stem}"
        else:
            volume_title = book_title
            file_base = book_stem

        contents, stats = _collect_contents(chapters, volume_title)

        md_path = domain_dir / f"{file_base}.extracted.md"
        json_path = domain_dir / f"{file_base}.extracted.json"

        meta = {"domain": domain, "language": language, "content_hash": content_hash}
        md_text = _render_markdown(book_title, volume_title, idx, chapters, meta)
        md_path.write_text(md_text, encoding="utf-8")

        payload = {
            "schema_version": EXTRACTED_SCHEMA_VERSION,
            "source_document": {
                "book_title": book_title,
                "book_file_path": str(path.resolve()),
                "domain": domain,
                "language": language,
                "content_hash": content_hash,
            },
            "volume": {
                "volume_index": idx,
                "volume_title": volume_title,
                "multi_volume": multi_volume,
                "chapter_count": stats["chapters"],
                "section_count": stats["sections"],
                "subsection_count": stats["subsections"],
                "paragraph_count": stats["paragraphs"],
                "total_raw_chars": stats["chars"],
                "skipped_front_sections": skipped_front,
                "skipped_back_sections": skipped_back,
            },
            "outline_tree": [_outline_node(c) for c in chapters],
            "contents": contents,
        }
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
        )

        result = VolumeExtractResult(
            volume_index=idx,
            volume_title=volume_title,
            md_path=md_path,
            json_path=json_path,
            chapter_count=stats["chapters"],
            section_count=stats["sections"],
            subsection_count=stats["subsections"],
            paragraph_count=stats["paragraphs"],
            total_chars=stats["chars"],
            skipped_front_sections=skipped_front,
            skipped_back_sections=skipped_back,
        )
        results.append(result)
        logger.info(
            "[extractor] 卷 %d《%s》落盘：%d 章 / %d 节 / %d 小节 / %d 段 / %d 字"
            "（剔除 front %d 节、back %d 节）",
            idx, volume_title, stats["chapters"], stats["sections"],
            stats["subsections"], stats["paragraphs"], stats["chars"],
            skipped_front, skipped_back,
        )

    return results


# ==================== 阶段 1 全库入口 ====================

def run_extract_stage(kb_dir: str | Path | None = None,
                      out_dir: str | Path | None = None,
                      settings: Settings | None = None) -> ExtractReport:
    """阶段 1：扫描知识库 → 逐书提取落盘。纯本地，零 API 配额。

    Args:
        kb_dir: 知识库源目录（默认 settings.knowledge_base_path）
        out_dir: 提取产物输出目录（默认 settings.knowledge_extracted_path）
    """
    settings = settings or get_settings()
    kb_path = Path(kb_dir) if kb_dir else settings.knowledge_base_path
    if not kb_path.is_absolute():
        kb_path = PROJECT_ROOT / kb_path

    if not kb_path.exists():
        raise FileNotFoundError(f"知识库目录不存在：{kb_path}")

    files = sorted(
        p for p in kb_path.rglob("*")
        if p.is_file() and p.suffix.lower() in (".epub", ".pdf")
    )
    logger.info("[extractor] 阶段 1 开始：扫描到 %d 个文档（%s）", len(files), kb_path)

    report = ExtractReport()
    for f in files:
        book_report = BookExtractReport(
            file_path=str(f), book_title=f.stem, domain=_domain_from_path(str(f)),
            content_hash="", multi_volume=False,
        )
        try:
            if f.suffix.lower() == ".pdf":
                # v1.6 全库为 EPUB；PDF 解析预留，暂不进入阶段 1
                logger.warning("[extractor] PDF 源暂不支持阶段 1 提取（预留）：%s", f.name)
                book_report.error = "PDF 阶段 1 提取暂未实现（v1.6 全库 EPUB）"
                report.books.append(book_report)
                continue
            volumes = extract_book(str(f), out_dir=out_dir, settings=settings)
            book_report.book_title = volumes[0].volume_title if volumes and len(volumes) == 1 else f.stem
            book_report.multi_volume = len(volumes) >= _MIN_VOLUME_BOUNDARIES
            book_report.volumes = volumes
            book_report.content_hash = _content_hash(str(f))
        except Exception as e:  # noqa: BLE001
            logger.exception("[extractor] 提取失败：%s", f.name)
            book_report.error = str(e)
        report.books.append(book_report)

    logger.info(
        "[extractor] 阶段 1 完成：%d 书 / %d 卷 / %d 段落",
        len(report.books), report.total_volumes, report.total_paragraphs,
    )
    return report
