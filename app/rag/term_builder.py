"""术语表构建器（两阶段）。

对齐 RAG_SYSTEM_DESIGN §3.6：
- 第一阶段：从原始文档索引页提取（Freud/Jung 英文版末尾 Index）→ 术语 → 页码映射；
- 第二阶段：LLM 从 ParagraphBlock 提取次级术语 → 补充术语表未收录的术语；
- 存储到 SQLite rag_terms（不进向量库），供 TermLookup 毫秒级查询；
- 接入 Query Rewrite（同义词扩展）与 Rerank（术语段落加权 ×1.1）。

术语表 Schema 对齐 §3.6：
  term_id / canonical_term / synonyms / language_pairs / domain /
  definition_location / occurrence_block_ids / related_terms
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid

from app.llm.client import LLMClient
from app.llm.prompts import build_term_extract_messages
from app.rag.chunker import ParagraphBlock
from app.rag.stores import RAGStore

logger = logging.getLogger(__name__)

# 第二阶段 LLM 提取的并发上限
_TERM_CONCURRENCY = 4
# 第二阶段最多抽取多少批段落（成本控制）
_MAX_EXTRACT_BATCHES = 60
# 每批段落数
_BATCH_SIZE = 5


def _extract_json_array(raw: str) -> list[dict]:
    """从 LLM 输出中提取 JSON 数组（容错）。"""
    if not raw:
        return []
    raw = raw.strip()
    # 去掉 ```json 围栏
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except Exception:  # noqa: BLE001
        pass
    # 兜底：正则提取 {...}
    items = []
    for m in re.finditer(r"\{[^{}]*\}", raw):
        try:
            items.append(json.loads(m.group()))
        except Exception:  # noqa: BLE001
            continue
    return items


# ==================== 第一阶段：索引页解析 ====================

# 索引页行形如："Repression, 14, 28n, 105-110" / "压抑 ......... 105"
_INDEX_LINE_RE = re.compile(r"^(?P<term>[^,\n]{2,80}?)[\s.,]*\(?\s*(?P<pages>[\d,\s\-–n]+)", re.M)


def parse_index_pages(index_text: str, domain: str) -> list[dict]:
    """从末尾术语索引页文本提取术语 → 页码映射（第一阶段）。

    返回 [{canonical_term, definition_location:{page_range}, domain, synonyms:[], ...}]
    """
    terms: list[dict] = []
    seen: set[str] = set()
    for line in index_text.splitlines():
        line = line.strip()
        if not line or len(line) < 3:
            continue
        m = _INDEX_LINE_RE.match(line)
        if not m:
            continue
        term = m.group("term").strip().strip(",").strip()
        pages_raw = m.group("pages").strip()
        if not term or len(term) < 2:
            continue
        # 清理术语尾部标点
        term = re.sub(r"[\s,]+$", "", term)
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append({
            "term_id": f"t_{uuid.uuid4().hex[:10]}",
            "canonical_term": term,
            "synonyms": [],
            "language_pairs": {"en": term},
            "domain": domain,
            "definition_location": {"page_range": pages_raw},
            "occurrence_block_ids": [],
            "related_terms": [],
        })
    logger.info("[term_builder] 第一阶段索引页提取：%d 个术语（domain=%s）", len(terms), domain)
    return terms


# CJK 检测（EPUB 术语条目语言对判断）
_CJK_RE = re.compile(r"[一-鿿]")


def parse_epub_index_entries(entries: list[dict], domain: str) -> list[dict]:
    """把 epub_parser 结构化提取的术语条目转换为 rag_terms 记录（第一阶段）。

    Args:
        entries: epub_parser._extract_index_entry 输出，每项含
            term / synonym / pages / definition / is_sub
        domain: freud / jung / philosophy
    Returns:
        rag_terms 记录列表（子条目 is_sub 跳过，它们依附于父术语）
    """
    terms: list[dict] = []
    seen: set[str] = set()
    for e in entries:
        if e.get("is_sub"):
            # 索引子条目（如 "in compulsives, 18"）非独立术语，跳过
            continue
        term = (e.get("term") or "").strip(" ,:;，、")
        if not term or len(term) < 2:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)

        synonym = (e.get("synonym") or "").strip()
        pages = (e.get("pages") or "").strip()
        definition = (e.get("definition") or "").strip()

        # 语言对：中文术语配 zh；英文术语配 en；有英文对应词时补全另一语言
        language_pairs: dict[str, str] = {}
        synonyms: list[str] = []
        if _CJK_RE.search(term):
            language_pairs["zh"] = term
        else:
            language_pairs["en"] = term
        if synonym:
            synonyms.append(synonym)
            if "en" not in language_pairs and not _CJK_RE.search(synonym):
                language_pairs["en"] = synonym
            elif "zh" not in language_pairs and _CJK_RE.search(synonym):
                language_pairs["zh"] = synonym

        definition_location: dict = {}
        if pages:
            definition_location["page_range"] = pages
        if definition:
            definition_location["source"] = "book_glossary"

        terms.append({
            "term_id": f"t_{uuid.uuid4().hex[:10]}",
            "canonical_term": term,
            "synonyms": synonyms,
            "language_pairs": language_pairs,
            "domain": domain,
            "definition_location": definition_location,
            "occurrence_block_ids": [],
            "related_terms": [],
        })

    logger.info(
        "[term_builder] 第一阶段 EPUB 结构化术语提取：%d 个术语（domain=%s）",
        len(terms), domain,
    )
    return terms


# ==================== 第二阶段：LLM 补充次级术语 ====================

class TermBuilder:
    """术语表构建器（两阶段编排）。"""

    def __init__(self, llm: LLMClient | None = None, store: RAGStore | None = None) -> None:
        self._llm = llm
        self._store = store or RAGStore()

    async def build_from_index(self, index_text: str, domain: str) -> list[dict]:
        """第一阶段：从索引页文本提取术语。"""
        return parse_index_pages(index_text, domain)

    async def build_from_blocks(self, blocks: list[ParagraphBlock]) -> list[dict]:
        """第二阶段：LLM 从段落块提取次级术语。

        只补充已有术语表未收录的术语。返回新增术语列表。
        """
        if self._llm is None:
            logger.info("[term_builder] 无 LLM，跳过第二阶段术语提取")
            return []

        # 已有术语名集合（用于去重）
        existing: set[str] = set()
        for domain in ("freud", "jung", "philosophy"):
            for t in self._store.query_terms("", match_mode="prefix"):
                existing.add(t.get("canonical_term", "").lower())
                for syn in t.get("synonyms", []):
                    existing.add(syn.lower())

        # 批量抽取
        batches = [
            blocks[i : i + _BATCH_SIZE]
            for i in range(0, min(len(blocks), _MAX_EXTRACT_BATCHES * _BATCH_SIZE), _BATCH_SIZE)
        ][:_MAX_EXTRACT_BATCHES]

        sem = asyncio.Semaphore(_TERM_CONCURRENCY)

        async def _extract(batch: list[ParagraphBlock]) -> list[dict]:
            async with sem:
                text = "\n\n".join(b.text for b in batch)[:2000]
                if len(text) < 50:
                    return []
                try:
                    raw = await self._llm.chat(
                        build_term_extract_messages(text),
                        temperature=0.1,
                        max_tokens=600,
                        json_mode=True,
                        timeout=12.0,
                    )
                    return _extract_json_array(raw)
                except Exception as e:  # noqa: BLE001
                    logger.warning("[term_builder] LLM 术语提取失败：%s", e)
                    return []

        results = await asyncio.gather(*[_extract(b) for b in batches], return_exceptions=True)

        new_terms: list[dict] = []
        seen_new: set[str] = set()
        for r in results:
            if isinstance(r, list):
                for item in r:
                    term_name = (item.get("term") or "").strip()
                    if not term_name or len(term_name) < 2:
                        continue
                    key = term_name.lower()
                    if key in existing or key in seen_new:
                        continue
                    seen_new.add(key)
                    new_terms.append({
                        "term_id": f"t_{uuid.uuid4().hex[:10]}",
                        "canonical_term": term_name,
                        "synonyms": item.get("synonyms", []) or [],
                        "language_pairs": {},
                        "domain": item.get("domain", "") or "",
                        "definition_location": {},
                        "occurrence_block_ids": [],
                        "related_terms": [],
                    })
            elif isinstance(r, Exception):
                logger.warning("[term_builder] 术语提取任务异常：%s", r)

        logger.info("[term_builder] 第二阶段 LLM 提取：%d 个新增术语", len(new_terms))
        return new_terms

    async def build_all(
        self,
        index_text: str,
        domain: str,
        blocks: list[ParagraphBlock],
        index_entries: list[dict] | None = None,
    ) -> list[dict]:
        """两阶段合并：索引页/术语表提取 + LLM 补充 → 存储到 rag_terms。

        Args:
            index_text: PDF 末尾索引页文本（正则行解析）；EPUB 通常为空字符串
            domain: freud / jung / philosophy
            blocks: 段落块（第二阶段 LLM 提取用）
            index_entries: EPUB 结构化术语条目（epub_parser 提取）；PDF 传 None
        """
        phase1_pdf = await self.build_from_index(index_text, domain)
        phase1_epub = parse_epub_index_entries(index_entries or [], domain)
        phase1 = phase1_pdf + phase1_epub
        phase2 = await self.build_from_blocks(blocks)

        all_terms = phase1 + phase2
        # 去重（按 canonical_term 小写）
        deduped: list[dict] = []
        seen: set[str] = set()
        for t in all_terms:
            key = t["canonical_term"].lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(t)

        for t in deduped:
            try:
                self._store.upsert_term(t)
            except Exception as e:  # noqa: BLE001
                logger.warning("[term_builder] 术语写入失败 %s：%s", t.get("canonical_term"), e)

        logger.info(
            "[term_builder] 术语表构建完成：阶段1=%d 阶段2=%d 去重后=%d（domain=%s）",
            len(phase1), len(phase2), len(deduped), domain,
        )
        return deduped
