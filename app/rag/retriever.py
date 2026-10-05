"""RAG 混合检索管线编排器。

对齐 RAG_SYSTEM_DESIGN §5：
- Query Rewrite（deepseek-v4-pro + 术语表辅助）→ 独立完整查询；
- Language Detection（zh/en）；
- 分支 A：段落级向量检索 Top-50；
- 分支 B：FTS5 关键词检索 Top-50；
- 分支 C：章节摘要级向量检索 Top-5；
- Merge & Deduplicate（保留双命中 hit_count）；
- Parent Retrieval（章节摘要命中 → 召回 parent_block_ids 全部段落）；
- Rerank（Top-10 + 同语言加权 + 术语表加权 + parent 附后）；
- Confidence Metadata Assembly（置信度信号）。

对外接口（对齐 §6.2）：
  RAGRetrieve(query, context, domain_filter=None, enable_image=False) -> RAGResult
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Callable

from app.config import Settings, get_settings
from app.core.models import ConfidenceSignals, RAGCandidate, RAGResult
from app.llm.client import LLMClient
from app.llm.embedding import EmbeddingClient
from app.llm.prompts import build_query_rewrite_messages
from app.rag.reranker import Reranker, _normalize_lang
from app.rag.stores import RAGStore
from app.rag.term_lookup import TermLookup, build_term_hints

logger = logging.getLogger(__name__)

# 各分支召回数量
_TOP_VECTOR = 50
_TOP_FTS = 50
_TOP_SUMMARY = 5

# CJK 检测
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _messages_to_text(messages: list[dict]) -> str:
    """将 messages 列表渲染为可读文本，用于链路追踪展示完整提示词。"""
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        parts.append(f"【{role}】\n{content}")
    return "\n\n".join(parts)


def _format_candidates_for_trace(candidates: list[dict], limit: int | None = None) -> str:
    """将候选列表格式化为可读文本，用于链路追踪展示完整召回条目。"""
    items = candidates if limit is None else candidates[:limit]
    lines: list[str] = []
    for i, c in enumerate(items, 1):
        bid = c.get("block_id", "")
        btype = c.get("block_type", "")
        text = c.get("text", "")
        vs = c.get("vector_score")
        ks = c.get("keyword_score")
        rs = c.get("rerank_score")
        src = c.get("source") or {}
        chapter = src.get("chapter_title", "")
        doc = src.get("document_title", "")
        lang = src.get("language", "")
        domain = src.get("domain", "")
        meta = []
        if vs is not None:
            meta.append(f"vec={vs:.4f}")
        if ks is not None:
            meta.append(f"fts={ks:.4f}")
        if rs is not None:
            meta.append(f"rerank={rs:.4f}")
        lines.append(
            f"[{i}] block_id={bid} type={btype} domain={domain} lang={lang}\n"
            f"    doc={doc} | chapter={chapter}\n"
            f"    scores: {', '.join(meta) if meta else '无'}\n"
            f"    text: {text}"
        )
    if limit is not None and len(candidates) > limit:
        lines.append(f"...（共 {len(candidates)} 条，仅展示前 {limit} 条）")
    return "\n\n".join(lines) if lines else "（空）"


def _detect_language(text: str) -> str:
    if not text:
        return "zh"
    sample = text[:200]
    cjk = len(_CJK_RE.findall(sample))
    return "zh" if cjk / max(len(sample), 1) > 0.15 else "en"


class RAGRetrieve:
    """RAG 混合检索管线。

    用法：
        retriever = RAGRetrieve(embedding, llm, store)
        result = await retriever.retrieve(query, context, domain_filter=None)
    """

    def __init__(
        self,
        embedding: EmbeddingClient | None = None,
        llm: LLMClient | None = None,
        store: RAGStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._embedding = embedding or EmbeddingClient(self.settings)
        self._llm = llm
        self._store = store or RAGStore(self.settings)
        self._reranker = Reranker(llm) if llm is not None else None

    async def retrieve(
        self,
        query: str,
        context: list[dict] | None = None,
        domain_filter: list[str] | None = None,
        enable_image: bool = False,
        *,
        trace: Callable[[str, str, str, str], None] | None = None,
    ) -> RAGResult:
        """执行完整混合检索管线。

        trace: 可选链路追踪回调 trace(name, input, output, notes)，
               用于记录各子环节（含 LLM 提示词与原始输出）。
        """
        if not query.strip():
            return RAGResult(candidates=[], confidence_signals=ConfidenceSignals())

        query_lang = _detect_language(query)
        working_window = context or []

        # 1) 术语表辅助（毫秒级，不降级）
        term_entries = TermLookup(query, match_mode="prefix", store=self._store)
        term_hints = build_term_hints(term_entries)
        term_block_ids: set[str] = set()
        for e in term_entries:
            term_block_ids.update(e.occurrence_block_ids)
        # 链路追踪：术语表命中
        if trace:
            term_lines = [
                f"term={e.term} | aliases={e.aliases} | block_ids={e.occurrence_block_ids}"
                for e in term_entries
            ]
            trace(
                "rag/term_lookup",
                input=f"query={query}",
                output=f"命中 {len(term_entries)} 个术语\n" + "\n".join(term_lines) if term_lines else "（无术语命中）",
                notes=f"term_hints={term_hints}",
            )

        # 2) Query Rewrite（可选，超时降级为原查询）
        rewritten = await self._query_rewrite(query, working_window, term_hints, trace=trace)
        logger.info(
            "[retriever] Query Rewrite: %r → %r (lang=%s, terms=%d)",
            query[:40], rewritten[:40], query_lang, len(term_entries),
        )

        # 3) 三分支召回（并行）
        vector_task = self._vector_search(rewritten, query_lang, domain_filter, enable_image)
        fts_task = asyncio.to_thread(
            self._fts_search, rewritten, domain_filter
        )
        summary_task = self._summary_search(rewritten, domain_filter)

        vector_hits, fts_hits, summary_hits = await asyncio.gather(
            vector_task, fts_task, summary_task, return_exceptions=True
        )

        # 异常分支降级为空
        if isinstance(vector_hits, Exception):
            logger.warning("[retriever] 向量检索失败：%s", vector_hits)
            vector_hits = []
        if isinstance(fts_hits, Exception):
            logger.warning("[retriever] FTS5 检索失败：%s", fts_hits)
            fts_hits = []
        if isinstance(summary_hits, Exception):
            logger.warning("[retriever] 章节摘要检索失败：%s", summary_hits)
            summary_hits = []

        # 链路追踪：三路召回完整结果（含完整文本）
        if trace:
            trace(
                "rag/vector_recall",
                input=f"embedding_query={rewritten}\nquery_lang={query_lang}\ndomain_filter={domain_filter}",
                output=f"共 {len(vector_hits)} 条向量召回（Top-{_TOP_VECTOR}）：\n\n"
                       + _format_candidates_for_trace(vector_hits),
                notes="分支 A：段落级向量检索",
            )
            trace(
                "rag/fts_recall",
                input=f"FTS 关键词（即重写后查询）：{rewritten}\ndomain_filter={domain_filter}",
                output=f"共 {len(fts_hits)} 条 FTS 召回（Top-{_TOP_FTS}）：\n\n"
                       + _format_candidates_for_trace(fts_hits),
                notes="分支 B：FTS5 关键词检索（关键词 = 重写后查询）",
            )
            trace(
                "rag/summary_recall",
                input=f"embedding_query={rewritten}\ndomain_filter={domain_filter}",
                output=f"共 {len(summary_hits)} 条章节摘要召回（Top-{_TOP_SUMMARY}）：\n\n"
                       + _format_candidates_for_trace(summary_hits),
                notes="分支 C：章节摘要级向量检索",
            )

        # 4) Merge & Deduplicate
        merged = self._merge_dedup(vector_hits, fts_hits, summary_hits)
        # 链路追踪：合并去重结果
        if trace:
            double_hits = [c for c in merged if c.get("hit_count", 1) >= 2]
            trace(
                "rag/merge_dedup",
                input=f"vector={len(vector_hits)} + fts={len(fts_hits)} + summary={len(summary_hits)}",
                output=f"合并去重后 {len(merged)} 条，其中双命中 {len(double_hits)} 条",
                notes="Merge & Deduplicate（双命中标记 hit_count>=2）",
            )

        # 5) Parent Retrieval
        parent_ctx = self._parent_retrieval(merged)
        if trace and parent_ctx:
            trace(
                "rag/parent_retrieval",
                input=f"merged 数={len(merged)}",
                output=f"扩展 parent 段落 {len(parent_ctx)} 条",
                notes="章节摘要命中 → 召回其 parent_block_ids 下全部段落",
            )

        # 6) Rerank
        all_candidates = merged + parent_ctx
        # 修正 is_term_hit（向量/关键词候选按 term_block_ids 标记）
        for c in all_candidates:
            if c.get("block_id") in term_block_ids:
                c["is_term_hit"] = True
        if self._reranker is not None and all_candidates:
            ranked = await self._reranker.rerank(
                rewritten, query_lang, all_candidates, term_block_ids, trace=trace
            )
        else:
            # 无 LLM 降级：按向量+关键词分数加权排序，并写 rerank_score
            # parent 候选 rerank_score 保持 None（不参与精排统计）
            from app.rag.reranker import _normalize_lang
            qlang = _normalize_lang(query_lang)

            def _fallback_score(c: dict) -> float:
                v = c.get("vector_score") or 0.0
                k = c.get("keyword_score") or 0.0
                s = 0.6 * v + 0.4 * k
                c_lang = _normalize_lang((c.get("source") or {}).get("language"))
                if c_lang == qlang:
                    s = min(1.0, s * 1.1)
                if c.get("block_id") in term_block_ids:
                    s = min(1.0, s * 1.1)
                c["rerank_score"] = s
                return s

            # parent 候选不参与排序，附在末尾
            main_cands = [c for c in all_candidates if c.get("retrieval_source") != "parent"]
            parent_cands = [c for c in all_candidates if c.get("retrieval_source") == "parent"]
            ranked = sorted(main_cands, key=_fallback_score, reverse=True) + parent_cands
            if trace:
                trace(
                    "rag/rerank",
                    input=f"query={rewritten}\n候选数={len(all_candidates)}",
                    output="降级：向量+关键词分数加权排序（无 LLM Reranker）",
                    notes="未配置 LLM，使用降级排序",
                )

        # 7) 组装 confidence_signals
        signals = self._assemble_signals(
            ranked, query_lang, domain_filter, term_block_ids
        )

        # 转 RAGCandidate
        candidates = [self._to_rag_candidate(c) for c in ranked[:20]]  # 限制返回量

        # 链路追踪：最终返回的完整候选列表（Top-20，含完整文本）
        if trace:
            final_text = _format_candidates_for_trace(
                [c.model_dump() if hasattr(c, "model_dump") else c for c in candidates]
            )
            trace(
                "rag/final_candidates",
                input=f"精排后总数={len(ranked)}，取 Top-20",
                output=f"最终返回 {len(candidates)} 条候选：\n\n{final_text}",
                notes=f"confidence_signals={signals.model_dump() if hasattr(signals, 'model_dump') else signals}",
            )

        logger.info(
            "[retriever] 检索完成：vec=%d fts=%d sum=%d merged=%d parent=%d → %d 候选",
            len(vector_hits), len(fts_hits), len(summary_hits),
            len(merged), len(parent_ctx), len(candidates),
        )
        return RAGResult(candidates=candidates, confidence_signals=signals)

    # ---- Query Rewrite ----

    async def _query_rewrite(
        self, query: str, working_window: list[dict], term_hints: str,
        *, trace: Callable | None = None,
    ) -> str:
        """Query Rewrite（LLM 重写查询），失败时返回原查询。"""
        if self._llm is None:
            return query
        # 组装发给 LLM 的完整提示词（用于链路追踪展示）
        rewrite_messages = build_query_rewrite_messages(query, working_window, term_hints)
        rewrite_prompt = _messages_to_text(rewrite_messages)
        try:
            raw = await asyncio.wait_for(
                self._llm.chat(
                    rewrite_messages,
                    temperature=0.2,
                    max_tokens=200,
                    timeout=2.0,
                ),
                timeout=3.0,
            )
            rewritten = raw.strip().splitlines()[0].strip() if raw.strip() else query
            # 链路追踪：记录 Query Rewrite 提示词 + LLM 原始输出
            if trace:
                trace(
                    "rag/query_rewrite",
                    input=f"原查询：{query}",
                    output=f"重写后查询：{rewritten}\n（LLM 原始输出：{raw}）",
                    notes=rewrite_prompt,
                )
            return rewritten or query
        except asyncio.TimeoutError:
            logger.info("[retriever] Query Rewrite 超时，用原查询")
            if trace:
                trace(
                    "rag/query_rewrite",
                    input=f"原查询：{query}",
                    output="（超时 >3s，使用原查询）",
                    notes=rewrite_prompt,
                )
            return query
        except Exception as e:  # noqa: BLE001
            logger.warning("[retriever] Query Rewrite 失败（%s），用原查询", e)
            if trace:
                trace(
                    "rag/query_rewrite",
                    input=f"原查询：{query}",
                    output=f"（失败：{e}，使用原查询）",
                    notes=rewrite_prompt,
                )
            return query

    # ---- 分支 A/C：向量检索 ----

    @staticmethod
    def _build_where(
        block_type: str | None = None,
        domain_filter: list[str] | None = None,
        enable_image: bool = False,
    ) -> dict | None:
        """构建 ChromaDB where 子句（单操作符约束：多条件用 $and 组合）。"""
        conditions: list[dict] = []
        if block_type:
            if enable_image:
                conditions.append({"$or": [{"block_type": "text"}, {"block_type": "image"}]})
            else:
                conditions.append({"block_type": block_type})
        if domain_filter:
            conditions.append({"domain": {"$in": domain_filter}})
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    async def _vector_search(
        self, query: str, query_lang: str,
        domain_filter: list[str] | None, enable_image: bool,
    ) -> list[dict]:
        """段落级向量检索 Top-50。"""
        try:
            embedding = await self._embedding.embed_query(query)
        except Exception as e:  # noqa: BLE001
            logger.warning("[retriever] 查询向量化失败：%s", e)
            return []

        where = self._build_where(
            block_type="text",
            domain_filter=domain_filter,
            enable_image=enable_image,
        )

        hits = self._store.query_vector(embedding, n=_TOP_VECTOR, where=where)
        out = []
        for h in hits:
            meta = h.get("metadata") or {}
            out.append({
                "block_id": h["block_id"],
                "block_type": meta.get("block_type", "text"),
                "text": h.get("text", ""),
                "source": {
                    "document_title": "",  # 从 rag_blocks 补
                    "chapter_title": meta.get("chapter_title", ""),
                    "section_title": meta.get("section_title", ""),
                    "page_range": meta.get("page_range", ""),
                    "language": meta.get("language", ""),
                    "domain": meta.get("domain", ""),
                },
                "vector_score": h.get("vector_score", 0.0),
                "keyword_score": None,
                "hit_count": 1,
                "retrieval_source": "vector",
                "is_term_hit": False,  # 在 retrieve 中按 term_block_ids 修正
            })
        # 补全 source.document_title + is_term_hit
        self._enrich_block_meta(out)
        return out

    async def _summary_search(
        self, query: str, domain_filter: list[str] | None
    ) -> list[dict]:
        """章节摘要级向量检索 Top-5。"""
        try:
            embedding = await self._embedding.embed_query(query)
        except Exception as e:  # noqa: BLE001
            return []

        where = self._build_where(
            block_type="section_summary",
            domain_filter=domain_filter,
        )

        hits = self._store.query_vector(embedding, n=_TOP_SUMMARY, where=where)
        out = []
        for h in hits:
            meta = h.get("metadata") or {}
            out.append({
                "block_id": h["block_id"],
                "block_type": "section_summary",
                "text": h.get("text", ""),
                "source": {
                    "document_title": "",
                    "chapter_title": meta.get("chapter_title", ""),
                    "section_title": meta.get("section_title", ""),
                    "page_range": meta.get("page_range", ""),
                    "language": meta.get("language", ""),
                    "domain": meta.get("domain", ""),
                },
                "vector_score": h.get("vector_score", 0.0),
                "keyword_score": None,
                "hit_count": 1,
                "retrieval_source": "vector",
                "is_term_hit": False,
                "parent_block_ids": [],  # 从 rag_blocks 补
            })
        self._enrich_block_meta(out, load_parent=True)
        return out

    # ---- 分支 B：FTS5 关键词检索 ----

    def _fts_search(self, query: str, domain_filter: list[str] | None) -> list[dict]:
        """FTS5 BM25 关键词检索 Top-50。"""
        rows = self._store.query_fts(query, n=_TOP_FTS, domains=domain_filter)
        out = []
        for r in rows:
            out.append({
                "block_id": r.get("block_id", ""),
                "block_type": "text",
                "text": r.get("text", ""),
                "source": {
                    "document_title": r.get("document_title", ""),
                    "chapter_title": r.get("chapter_title", ""),
                    "section_title": "",
                    "page_range": "",
                    "language": "",
                    "domain": r.get("domain", ""),
                },
                "vector_score": None,
                "keyword_score": r.get("bm25_score", 0.0),
                "hit_count": 1,
                "retrieval_source": "keyword",
                "is_term_hit": False,
            })
        # 补全 section_title / page_range / language
        self._enrich_block_meta(out)
        return out

    # ---- Merge & Deduplicate ----

    def _merge_dedup(
        self,
        vector_hits: list[dict],
        fts_hits: list[dict],
        summary_hits: list[dict],
    ) -> list[dict]:
        """合并三分支，去重，标记双命中 hit_count=2。"""
        by_id: dict[str, dict] = {}
        for c in vector_hits + fts_hits + summary_hits:
            bid = c.get("block_id")
            if not bid:
                continue
            if bid in by_id:
                existing = by_id[bid]
                existing["hit_count"] = existing.get("hit_count", 1) + 1
                # 合并分数
                if c.get("vector_score") is not None and existing.get("vector_score") is None:
                    existing["vector_score"] = c["vector_score"]
                if c.get("keyword_score") is not None and existing.get("keyword_score") is None:
                    existing["keyword_score"] = c["keyword_score"]
                # 检索来源合并
                srcs = {existing.get("retrieval_source"), c.get("retrieval_source")}
                existing["retrieval_source"] = "/".join(sorted(s for s in srcs if s))
            else:
                by_id[bid] = dict(c)
        return list(by_id.values())

    # ---- Parent Retrieval ----

    def _parent_retrieval(self, merged: list[dict]) -> list[dict]:
        """章节摘要命中 → 召回 parent_block_ids 全部段落（扩展上下文）。"""
        parent_ids: list[str] = []
        for c in merged:
            if c.get("block_type") == "section_summary":
                pids = c.get("parent_block_ids") or []
                parent_ids.extend(pids)
        if not parent_ids:
            return []
        # 已在 merged 中的 block 跳过
        existing = {c.get("block_id") for c in merged}
        new_ids = [pid for pid in parent_ids if pid not in existing]
        if not new_ids:
            return []
        blocks = self._store.get_blocks_by_ids(new_ids)
        out = []
        for bid, b in blocks.items():
            out.append({
                "block_id": bid,
                "block_type": b.get("block_type", "text"),
                "text": b.get("text", ""),
                "source": {
                    "document_title": "",
                    "chapter_title": b.get("chapter_title", ""),
                    "section_title": b.get("section_title", ""),
                    "page_range": b.get("page_range", ""),
                    "language": b.get("language", ""),
                    "domain": b.get("domain", ""),
                },
                "vector_score": None,
                "keyword_score": None,
                "hit_count": 1,
                "retrieval_source": "parent",
                "is_term_hit": False,
            })
        return out

    # ---- 元数据补全 ----

    def _enrich_block_meta(self, candidates: list[dict], load_parent: bool = False) -> None:
        """从 rag_blocks 补全 document_title / section_title / page_range / language / parent_block_ids。"""
        ids = [c.get("block_id") for c in candidates if c.get("block_id")]
        if not ids:
            return
        blocks = self._store.get_blocks_by_ids(ids)
        for c in candidates:
            bid = c.get("block_id")
            if not bid or bid not in blocks:
                continue
            b = blocks[bid]
            src = c.get("source") or {}
            if not src.get("section_title"):
                src["section_title"] = b.get("section_title", "") or src.get("section_title", "")
            if not src.get("page_range"):
                src["page_range"] = b.get("page_range", "") or src.get("page_range", "")
            if not src.get("language"):
                src["language"] = b.get("language", "") or src.get("language", "")
            c["source"] = src
            if load_parent:
                import json
                pids = b.get("parent_block_ids")
                if isinstance(pids, str):
                    try:
                        pids = json.loads(pids)
                    except Exception:  # noqa: BLE001
                        pids = []
                c["parent_block_ids"] = pids or []

    # ---- 置信度信号组装 ----

    def _assemble_signals(
        self,
        ranked: list[dict],
        query_lang: str,
        domain_filter: list[str] | None,
        term_block_ids: set[str],
    ) -> ConfidenceSignals:
        """组装置信度信号（对齐 §5.7）。"""
        main = [c for c in ranked if c.get("retrieval_source") != "parent" and c.get("rerank_score") is not None]
        total = len(main)
        double = sum(1 for c in main if c.get("hit_count", 1) >= 2)

        scores = [c.get("rerank_score") or 0.0 for c in main]
        max_score = max(scores) if scores else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0

        # source_diversity: 命中来源文档数（domain 维度 + block 不重复）
        domains = {(c.get("source") or {}).get("domain") for c in main if (c.get("source") or {}).get("domain")}
        source_diversity = len(domains)

        # section_coverage: 命中不同 Section 数
        sections = {
            (c.get("source") or {}).get("document_title", "") + "|" + (c.get("source") or {}).get("section_title", "")
            for c in main
        }
        section_coverage = len(sections)

        # languages_hit
        langs = {(c.get("source") or {}).get("language") for c in main if (c.get("source") or {}).get("language")}
        languages_hit = sorted(langs)

        # coverage_estimate
        if total >= 3 and max_score >= 0.85 and section_coverage >= 2:
            coverage = "high"
        elif total >= 1 and max_score >= 0.6:
            coverage = "medium"
        else:
            coverage = "low"

        # term_hits_count
        term_hits = sum(1 for c in main if c.get("block_id") in term_block_ids)

        return ConfidenceSignals(
            total_hits=total,
            double_hits=double,
            max_rerank_score=round(max_score, 4),
            avg_rerank_score=round(avg_score, 4),
            source_diversity=source_diversity,
            section_coverage=section_coverage,
            coverage_estimate=coverage,
            languages_hit=languages_hit,
            domain_filter_applied=domain_filter,
            term_hits_count=term_hits,
        )

    # ---- 转 RAGCandidate ----

    def _to_rag_candidate(self, c: dict) -> RAGCandidate:
        src = c.get("source") or {}
        return RAGCandidate(
            block_id=c.get("block_id", ""),
            block_type=c.get("block_type", "text"),
            text=c.get("text", ""),
            source=src,
            vector_score=c.get("vector_score"),
            keyword_score=c.get("keyword_score"),
            rerank_score=c.get("rerank_score"),
            hit_count=c.get("hit_count", 1),
            retrieval_source=c.get("retrieval_source", "vector"),
            is_term_hit=c.get("is_term_hit", False),
        )
