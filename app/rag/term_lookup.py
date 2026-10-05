"""术语表查询接口（毫秒级 SQLite 查询）。

对齐 RAG_SYSTEM_DESIGN §5.2 / §6.2：
- 供 Query Rewrite 调用（命中术语 → 扩展同义词 + 定义位置 hint）；
- match_mode: "exact" 精确匹配 / "prefix" 前缀匹配（默认）；
- 不参与向量检索，仅精确映射。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.rag.stores import RAGStore

logger = logging.getLogger(__name__)


@dataclass
class TermEntry:
    """术语表命中条目。"""

    term_id: str
    canonical_term: str
    synonyms: list[str] = field(default_factory=list)
    language_pairs: dict = field(default_factory=dict)
    domain: str = ""
    definition_location: dict = field(default_factory=dict)
    occurrence_block_ids: list[str] = field(default_factory=list)
    related_terms: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "TermEntry":
        return cls(
            term_id=d.get("term_id", ""),
            canonical_term=d.get("canonical_term", ""),
            synonyms=d.get("synonyms", []) or [],
            language_pairs=d.get("language_pairs", {}) or {},
            domain=d.get("domain", ""),
            definition_location=d.get("definition_location", {}) or {},
            occurrence_block_ids=d.get("occurrence_block_ids", []) or [],
            related_terms=d.get("related_terms", []) or [],
        )


def TermLookup(query: str, match_mode: str = "prefix", store: RAGStore | None = None) -> list[TermEntry]:
    """术语表查询。

    Args:
        query: 用户查询文本
        match_mode: "exact" 精确匹配 / "prefix" 前缀/包含匹配（默认）
        store: RAGStore 实例（None 时新建）
    Returns:
        命中的术语列表
    """
    store = store or RAGStore()
    # 对查询做简单的分词：英文按空格，中文按字符级覆盖
    # 先整体查一次
    raw = store.query_terms(query, match_mode=match_mode)
    if not raw and match_mode == "prefix":
        # 退化：把查询拆词再查
        tokens = [t for t in query.split() if len(t) >= 2]
        for tok in tokens:
            raw.extend(store.query_terms(tok, match_mode="prefix"))
        # 去重
        seen: set[str] = set()
        deduped = []
        for t in raw:
            if t["term_id"] not in seen:
                seen.add(t["term_id"])
                deduped.append(t)
        raw = deduped
    return [TermEntry.from_dict(t) for t in raw]


def build_term_hints(entries: list[TermEntry]) -> str:
    """把术语命中条目格式化为 Query Rewrite prompt 的 hints 字符串。"""
    if not entries:
        return ""
    lines = []
    for e in entries:
        syns = ", ".join(e.synonyms) if e.synonyms else ""
        lang_pairs = " / ".join(f"{k}:{v}" for k, v in e.language_pairs.items()) if e.language_pairs else ""
        parts = [e.canonical_term]
        if syns:
            parts.append(f"同义词: {syns}")
        if lang_pairs:
            parts.append(f"跨语言: {lang_pairs}")
        if e.domain:
            parts.append(f"领域: {e.domain}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)
