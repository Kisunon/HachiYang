"""LLM Rerank（精排）。

对齐 RAG_SYSTEM_DESIGN §5.6：
- 输入：query + 候选 block 的 document_title + chapter_title + block_type + text[:300]；
- 输出：按相关性排序的 Top-10 + 每条 0~1 相关性分；
- 同语言加权：查询语言 zh 时中文候选 ×1.1，en 时英文候选 ×1.1（上限 1.0）；
- 术语表加权：候选命中术语 occurrence_block_ids 时 ×1.1（上限 1.0）；
- 扩展上下文（retrieval_source=parent）不参与精排，附在主候选之后；
- 降级：>3s 或 LLM 失败 → 向量+关键词分数加权排序，跳过术语表加权。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Callable

from app.llm.client import LLMClient
from app.llm.prompts import build_rerank_messages

logger = logging.getLogger(__name__)

# 链路追踪回调签名：trace(name, input, output, notes)
TraceFn = Callable[[str, str, str, str], None]


def _messages_to_text(messages: list[dict]) -> str:
    """将 messages 列表渲染为可读文本，用于链路追踪展示完整提示词。"""
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        parts.append(f"【{role}】\n{content}")
    return "\n\n".join(parts)

# Rerank 超时阈值（秒），超过则降级
_RERANK_TIMEOUT = 3.0
# 精排保留数量
_RERANK_TOP = 10

# block_type → 展示标签
_TYPE_LABELS = {
    "text": "段落",
    "section_summary": "章节摘要",
    "image": "图片",
}


def _extract_rerank_json(raw: str) -> dict | None:
    """从 LLM 输出提取 {"ranked_ids": [...], "scores": [...]}。"""
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "ranked_ids" in data and "scores" in data:
            return data
    except Exception:  # noqa: BLE001
        pass
    # 兜底正则
    m = re.search(r"\{[^{}]*ranked_ids[^{}]*\}", raw, re.S)
    if m:
        try:
            return json.loads(m.group())
        except Exception:  # noqa: BLE001
            pass
    return None


def _normalize_lang(lang: str | None) -> str:
    return "zh" if (lang or "").lower().startswith("zh") else "en"


class Reranker:
    """LLM Rerank 精排器。"""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def rerank(
        self,
        query: str,
        query_lang: str,
        candidates: list[dict],
        term_block_ids: set[str] | None = None,
        *,
        trace: TraceFn | None = None,
    ) -> list[dict]:
        """对候选精排，返回 Top-N（含 rerank_score）。

        Args:
            query: 重写后的查询
            query_lang: 查询语言 zh/en
            candidates: [{block_id, block_type, text, source:{language,...}, vector_score, keyword_score, ...}]
            term_block_ids: 术语表 occurrence_block_ids 并集（命中加权用）
            trace: 可选链路追踪回调，用于记录精排提示词与 LLM 原始输出
        Returns:
            排序后的 candidates（每个含 rerank_score），扩展段落(parent)排在末尾且 rerank_score=None
        """
        # 分离扩展段落（parent）与主候选
        main = [c for c in candidates if c.get("retrieval_source") != "parent"]
        parent_ctx = [c for c in candidates if c.get("retrieval_source") == "parent"]

        if not main:
            # 无主候选，直接返回扩展段落
            for c in parent_ctx:
                c["rerank_score"] = None
            return parent_ctx

        # 尝试 LLM Rerank
        reranked = await self._llm_rerank(
            query, main, query_lang, term_block_ids or set(), trace=trace
        )
        if reranked is None:
            # 降级：向量+关键词分数加权
            logger.info("[reranker] LLM Rerank 降级为分数加权排序")
            reranked = self._fallback_sort(main, query_lang, term_block_ids or set())
            # 注：_llm_rerank 已在超时/失败时记录了 trace（含提示词与降级原因），
            # 此处不再重复记录 rag/rerank，避免链路追踪中出现重复环节。

        # 取 Top-N
        reranked = reranked[:_RERANK_TOP]

        # 扩展段落附在末尾
        for c in parent_ctx:
            c["rerank_score"] = None
        return reranked + parent_ctx

    async def _llm_rerank(
        self,
        query: str,
        candidates: list[dict],
        query_lang: str,
        term_block_ids: set[str],
        *,
        trace: TraceFn | None = None,
    ) -> list[dict] | None:
        """LLM 精排，失败返回 None。"""
        # 构造候选展示
        id_to_idx: dict[str, int] = {}
        display: list[dict] = []
        for i, c in enumerate(candidates):
            cid = str(i + 1)
            id_to_idx[cid] = i
            btype = _TYPE_LABELS.get(c.get("block_type", "text"), "段落")
            snippet = c.get("text", "")[:300]
            display.append({"id": cid, "type": btype, "snippet": snippet})

        # 组装发给 LLM 的完整提示词（用于链路追踪展示）
        rerank_messages = build_rerank_messages(query, display)
        rerank_prompt = _messages_to_text(rerank_messages)

        try:
            raw = await asyncio.wait_for(
                self._llm.chat(
                    rerank_messages,
                    temperature=0.1,
                    max_tokens=800,
                    json_mode=True,
                    timeout=_RERANK_TIMEOUT * 2,  # 给 LLM 留余量，外层 wait_for 控制
                ),
                timeout=_RERANK_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("[reranker] LLM Rerank 超时（>%.1fs）降级", _RERANK_TIMEOUT)
            if trace:
                trace(
                    "rag/rerank",
                    input=f"query={query}\n候选数={len(display)}",
                    output=f"（超时 >{_RERANK_TIMEOUT}s，降级）",
                    notes=rerank_prompt,
                )
            return None
        except Exception as e:  # noqa: BLE001
            logger.warning("[reranker] LLM Rerank 失败降级：%s", e)
            if trace:
                trace(
                    "rag/rerank",
                    input=f"query={query}\n候选数={len(display)}",
                    output=f"（LLM 调用失败：{e}，降级）",
                    notes=rerank_prompt,
                )
            return None

        # 链路追踪：记录精排提示词 + LLM 原始输出
        if trace:
            trace(
                "rag/rerank",
                input=f"query={query}\n候选数={len(display)}",
                output=raw,
                notes=rerank_prompt,
            )

        parsed = _extract_rerank_json(raw)
        if parsed is None:
            logger.warning("[reranker] Rerank 输出解析失败降级：%s", raw[:80])
            return None

        ranked_ids = parsed.get("ranked_ids", [])
        scores = parsed.get("scores", [])

        # 映射回候选 + 加权
        out: list[dict] = []
        for rank_pos, rid in enumerate(ranked_ids):
            idx = id_to_idx.get(str(rid))
            if idx is None:
                continue
            c = dict(candidates[idx])
            score = float(scores[rank_pos]) if rank_pos < len(scores) else 0.5
            score = max(0.0, min(1.0, score))
            # 同语言加权
            c_lang = _normalize_lang((c.get("source") or {}).get("language"))
            if c_lang == query_lang:
                score = min(1.0, score * 1.1)
            # 术语命中加权
            if c.get("block_id") in term_block_ids:
                score = min(1.0, score * 1.1)
            c["rerank_score"] = score
            out.append(c)
        # 补上未被排到的候选（LLM 漏列）
        covered = {id_to_idx[str(rid)] for rid in ranked_ids if str(rid) in id_to_idx}
        for i, c in enumerate(candidates):
            if i in covered:
                continue
            cc = dict(c)
            cc["rerank_score"] = 0.3
            out.append(cc)
        return out

    def _fallback_sort(
        self,
        candidates: list[dict],
        query_lang: str,
        term_block_ids: set[str],
    ) -> list[dict]:
        """降级排序：向量分 + 关键词分加权。"""
        def _score(c: dict) -> float:
            v = c.get("vector_score") or 0.0
            k = c.get("keyword_score") or 0.0
            s = 0.6 * v + 0.4 * k
            c_lang = _normalize_lang((c.get("source") or {}).get("language"))
            if c_lang == query_lang:
                s = min(1.0, s * 1.1)
            if c.get("block_id") in term_block_ids:
                s = min(1.0, s * 1.1)
            c["rerank_score"] = s
            return s
        return sorted(candidates, key=_score, reverse=True)
