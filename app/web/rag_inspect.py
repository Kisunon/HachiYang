"""知识库索引人工检查只读 API（Phase 3.5，对齐 IMPLEMENTATION_DESIGN §6.11.1）。

挂载方式：`create_debug_router()` 内 `router.include_router(create_rag_inspect_router())`，
父路由 prefix=/debug + 本路由 prefix=/rag → 最终路径 /debug/rag/*。

约束：
- 全部端点只读 GET，无任何写操作；仅 127.0.0.1 / ::1 / localhost / testclient 可访问；
- 浏览检查与 FTS 检索零 API 配额；vector 检索试跑消耗 Embedding 配额（界面显式提示）。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.rag.stores import RAGStore

logger = logging.getLogger(__name__)

_LIST_SNIPPET_LEN = 200


def _ensure_localhost(request: Request) -> None:
    """仅允许本地访问（与 server.py 调试台一致）。"""
    client = request.client
    if client is None:
        raise HTTPException(status_code=403, detail="无法识别客户端地址")
    if client.host not in ("127.0.0.1", "::1", "localhost", "testclient"):
        raise HTTPException(status_code=403, detail="知识库检查仅限本地访问（127.0.0.1）")


def _snippet(text: str, n: int = _LIST_SNIPPET_LEN) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n] + "…"


def _block_list_item(b: dict) -> dict:
    """切片列表项（text 截断，完整文本走 /blocks/{block_id}）。"""
    parent_ids = b.get("parent_block_ids")
    if isinstance(parent_ids, str):
        try:
            parent_ids = json.loads(parent_ids or "[]")
        except Exception:  # noqa: BLE001
            parent_ids = []
    return {
        "block_id": b["block_id"],
        "document_id": b["document_id"],
        "domain": b.get("domain"),
        "language": b.get("language"),
        "volume_title": b.get("volume_title") or "",
        "volume_index": b.get("volume_index"),
        "chapter_title": b.get("chapter_title") or "",
        "section_title": b.get("section_title") or "",
        "subsection_title": b.get("subsection_title") or "",
        "page_range": b.get("page_range") or "",
        "block_index_in_section": b.get("block_index_in_section"),
        "char_count": b.get("char_count"),
        "block_type": b.get("block_type"),
        "source_anchor_in_md": b.get("source_anchor_in_md") or "",
        "parent_chapter_summary_id": b.get("parent_chapter_summary_id") or "",
        "has_children": bool(parent_ids),
        "snippet": _snippet(b.get("text") or ""),
    }


def create_rag_inspect_router(embedding_client: Any = None) -> APIRouter:
    """创建知识库检查路由。

    embedding_client：可选注入（测试用 FakeEmbedding）；生产为 None，
    vector 检索试跑时惰性构造真实 EmbeddingClient。
    """
    router = APIRouter(prefix="/rag", tags=["debug-rag"])
    store = RAGStore(get_settings())
    _embedding = embedding_client

    def _get_embedding() -> Any:
        nonlocal _embedding
        if _embedding is None:
            from app.llm.embedding import EmbeddingClient
            _embedding = EmbeddingClient(get_settings())
        return _embedding

    # ---- 文档总览 ----

    @router.get("/docs")
    async def rag_docs(request: Request) -> JSONResponse:
        _ensure_localhost(request)
        try:
            docs = store.list_documents_overview()
        except Exception as e:  # noqa: BLE001
            logger.exception("[rag_inspect] 文档总览查询失败")
            raise HTTPException(status_code=500, detail=f"查询失败：{e}") from e
        return JSONResponse({
            "documents": docs,
            "term_feature_enabled": store.term_feature_enabled,
        })

    # ---- 章节列表 ----

    @router.get("/docs/{document_id}/chapters")
    async def rag_chapters(document_id: str, request: Request) -> JSONResponse:
        _ensure_localhost(request)
        if not store.get_document(document_id):
            raise HTTPException(status_code=404, detail=f"文档不存在：{document_id}")
        chapters = store.list_chapters(document_id)
        return JSONResponse({"document_id": document_id, "chapters": chapters})

    # ---- 切片列表（章内下钻 / 类型过滤 / FTS 关键词 / 分页）----

    @router.get("/docs/{document_id}/blocks")
    async def rag_blocks(
        document_id: str,
        request: Request,
        chapter: str | None = Query(default=None),
        type: str | None = Query(default=None, description="text / section_summary / chapter_summary"),
        q: str | None = Query(default=None, description="FTS5 关键词（BM25，零配额）"),
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> JSONResponse:
        _ensure_localhost(request)
        if type and type not in ("text", "section_summary", "chapter_summary", "image"):
            raise HTTPException(
                status_code=400,
                detail="type 仅支持 text / section_summary / chapter_summary / image",
            )
        if not store.get_document(document_id):
            raise HTTPException(status_code=404, detail=f"文档不存在：{document_id}")

        if q and q.strip():
            # FTS5 BM25 检索 → block_id 关联回本文档（FTS 表无 document_id 列）
            hits = store.query_fts(q.strip(), n=200)
            blocks_map = store.get_blocks_by_ids([h["block_id"] for h in hits]) if hits else {}
            matched: list[dict] = []
            for h in hits:
                b = blocks_map.get(h["block_id"])
                if not b or b["document_id"] != document_id:
                    continue
                if type and b["block_type"] != type:
                    continue
                matched.append(b)
            total = len(matched)
            page = matched[offset:offset + limit]
        else:
            total = store.count_blocks(document_id, chapter, type)
            page = store.paginate_blocks(document_id, chapter, type, limit, offset)

        return JSONResponse({
            "document_id": document_id,
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": [_block_list_item(b) for b in page],
        })

    # ---- 切片全文 + Parent Retrieval 互跳 ----

    @router.get("/blocks/{block_id}")
    async def rag_block_detail(block_id: str, request: Request) -> JSONResponse:
        _ensure_localhost(request)
        b = store.get_block(block_id)
        if b is None:
            raise HTTPException(status_code=404, detail=f"切片不存在：{block_id}")

        parent_ids: list[str] = []
        raw_parents = b.get("parent_block_ids")
        if isinstance(raw_parents, str) and raw_parents:
            try:
                parent_ids = json.loads(raw_parents)
            except Exception:  # noqa: BLE001
                parent_ids = []

        children: list[dict] = []
        parent_summary = None
        if b["block_type"] in ("section_summary", "chapter_summary") and parent_ids:
            # 两级摘要链式展开：章摘要 → 节摘要 → 正文块
            blocks_map = store.get_blocks_by_ids(parent_ids)
            for cid in parent_ids:
                cb = blocks_map.get(cid)
                if not cb:
                    continue
                children.append({
                    "block_id": cb["block_id"],
                    "block_type": cb.get("block_type"),
                    "volume_title": cb.get("volume_title") or "",
                    "chapter_title": cb.get("chapter_title") or "",
                    "section_title": cb.get("section_title") or "",
                    "subsection_title": cb.get("subsection_title") or "",
                    "block_index_in_section": cb.get("block_index_in_section"),
                    "char_count": cb.get("char_count"),
                    "source_anchor_in_md": cb.get("source_anchor_in_md") or "",
                    "snippet": _snippet(cb.get("text") or ""),
                })
        elif b["block_type"] not in ("section_summary", "chapter_summary"):
            parent_summary = store.find_parent_summary(b["document_id"], block_id)

        detail = dict(b)
        detail["parent_block_ids"] = parent_ids
        return JSONResponse({
            "block": detail,
            "children": children,
            "parent_summary": parent_summary,
        })

    # ---- 术语表（分页/搜索）----

    @router.get("/terms")
    async def rag_terms(
        request: Request,
        q: str | None = Query(default=None),
        domain: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> JSONResponse:
        _ensure_localhost(request)
        query = (q or "").strip()
        total = store.count_terms(query, domain)
        terms = store.paginate_terms(query, domain, limit, offset)
        items = [{
            "term_id": t["term_id"],
            "canonical_term": t["canonical_term"],
            "domain": t.get("domain"),
            "synonyms": t.get("synonyms") or [],
            "language_pairs": t.get("language_pairs") or {},
            "occurrence_count": len(t.get("occurrence_block_ids") or []),
            "related_count": len(t.get("related_terms") or []),
        } for t in terms]
        return JSONResponse({
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": items,
            "term_feature_enabled": store.term_feature_enabled,
        })

    # ---- 术语详情 + 出现片段 ----

    @router.get("/terms/{term_id}")
    async def rag_term_detail(term_id: str, request: Request) -> JSONResponse:
        _ensure_localhost(request)
        t = store.get_term(term_id)
        if t is None:
            raise HTTPException(status_code=404, detail=f"术语不存在：{term_id}")
        occ_ids = (t.get("occurrence_block_ids") or [])[:3]
        blocks_map = store.get_blocks_by_ids(occ_ids) if occ_ids else {}
        occurrences = []
        for bid in occ_ids:
            b = blocks_map.get(bid)
            if not b:
                continue
            occurrences.append({
                "block_id": b["block_id"],
                "document_id": b["document_id"],
                "chapter_title": b.get("chapter_title") or "",
                "section_title": b.get("section_title") or "",
                "snippet": _snippet(b.get("text") or ""),
            })
        return JSONResponse({"term": t, "occurrences": occurrences})

    # ---- 检索试跑（fts 零配额 / vector 消耗 Embedding 配额）----

    @router.get("/retrieve")
    async def rag_retrieve(
        request: Request,
        q: str = Query(..., min_length=1, description="查询词"),
        mode: str = Query(default="fts", description="fts（BM25，零配额）/ vector（语义，消耗 Embedding 配额）"),
        k: int = Query(default=10, ge=1, le=50),
        domain: str | None = Query(default=None, description="freud / jung / philosophy"),
    ) -> JSONResponse:
        _ensure_localhost(request)
        query = q.strip()
        if not query:
            raise HTTPException(status_code=400, detail="q 不能为空")
        if domain and domain not in ("freud", "jung", "philosophy"):
            raise HTTPException(status_code=400, detail="domain 仅支持 freud / jung / philosophy")

        if mode == "fts":
            hits = store.query_fts(query, n=k, domains=[domain] if domain else None)
            items = [{
                "block_id": h["block_id"],
                "chapter_title": h.get("chapter_title") or "",
                "document_title": h.get("document_title") or "",
                "domain": h.get("domain"),
                "snippet": _snippet(h.get("text") or ""),
                "score": round(h.get("bm25_score", 0.0), 4),
                "score_type": "bm25（绝对值越大越相关）",
            } for h in hits]
            return JSONResponse({"mode": "fts", "query": query, "items": items})

        if mode == "vector":
            try:
                vector = await _get_embedding().embed_query(query)
                where = {"domain": domain} if domain else None
                hits = store.query_vector(vector, n=k, where=where)
            except Exception as e:  # noqa: BLE001
                # 429 AccountQuotaExceeded 的错误信息含配额重置时间，原样透传便于前端展示
                msg = str(e)
                logger.warning("[rag_inspect] 向量检索试跑失败：%s", msg[:300])
                return JSONResponse(
                    status_code=503,
                    content={
                        "detail": "向量语义检索失败（调用 Embedding API 失败，可能配额超限/限流）",
                        "error": msg[:500],
                        "hint": "FTS 关键词检索零配额可随时使用；向量试跑消耗 doubao-embedding-vision 配额，配额恢复后再试。",
                    },
                )
            items = [{
                "block_id": h["block_id"],
                "chapter_title": (h.get("metadata") or {}).get("chapter_title", ""),
                "section_title": (h.get("metadata") or {}).get("section_title", ""),
                "document_id": (h.get("metadata") or {}).get("document_id", ""),
                "domain": (h.get("metadata") or {}).get("domain", ""),
                "snippet": _snippet(h.get("text") or ""),
                "score": round(h.get("vector_score", 0.0), 4),
                "score_type": "cosine 相似度（0~1，越大越相关）",
            } for h in hits]
            return JSONResponse({"mode": "vector", "query": query, "items": items})

        raise HTTPException(status_code=400, detail="mode 仅支持 fts / vector")

    return router
