"""Phase 3.5：知识库索引人工检查（调试台扩展）测试。

对齐 IMPLEMENTATION_DESIGN.md §6.11.1 / §8 Phase 3.5，验收 I-01 ~ I-05：
- I-01：文档总览（状态 / 切片数 / 摘要数 / 术语数 / 向量覆盖率 / 未入库标记）
- I-02：章节下钻 + 分页浏览 + 摘要 ↔ 父子段落 Parent Retrieval 互跳
- I-03：术语表分页 / 搜索 / 详情（同义词、出现片段）
- I-04：检索试跑 —— FTS 零配额；vector 分支用 FakeEmbedding 验证（真实 API 配额恢复后人工验收）
- I-05：全部只读 GET、非 127.0.0.1 返回 403、页面内置问题处理指引

策略：全部跑在本地真实库（.data/hachiyang.db，Freud PDF 已完成索引），
零 LLM / 零 Embedding API 消耗；若本机尚无 done 文档则 pytest.skip。

运行：python -m pytest tests/test_rag_inspector.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.rag.stores import RAGStore  # noqa: E402
from app.web.rag_inspect import create_rag_inspect_router  # noqa: E402


# ==================== 测试基础设施 ====================

class _FakeDimEmbedding:
    """维度对齐的确定性向量（不调 API）：验证 vector 试跑分支执行与 score 语义。"""

    def __init__(self, dim: int) -> None:
        self.calls = 0
        self.dim = dim

    async def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        # 确定性伪随机（同查询同向量），归一化到单位向量
        vec = []
        x = float(len(text))
        for i in range(self.dim):
            x = (x * 1103515245 + 12345) % (2 ** 31)
            vec.append(math.sin(x / 1e6) + 1.0)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class _RaisingEmbedding:
    """模拟 Embedding API 429 配额超限。"""

    async def embed_query(self, text: str) -> list[float]:
        raise RuntimeError(
            "Error code: 429 - {'error': {'code': 'AccountQuotaExceeded', "
            "'message': 'You have exceeded the 5-hour usage quota. It will reset at ...'}}"
        )


@pytest.fixture(scope="module")
def done_doc() -> dict:
    """取一个 status=done 的文档（优先块数最多的，如英文 Freud PDF）；无则跳过整个模块。"""
    settings = get_settings()
    store = RAGStore(settings)
    overview = store.list_documents_overview()
    done = [d for d in overview if d["status"] == "done" and d.get("sqlite_block_count", 0) > 0]
    if not done:
        pytest.skip("本地无 done 状态的索引文档（先完成至少一个文档的全量索引），跳过 Phase 3.5 测试")
    # 块数最多的 done 文档作为稳定测试对象（新入库的小卷文档不影响既有断言）
    return max(done, key=lambda d: d.get("sqlite_block_count", 0))


@pytest.fixture(scope="module")
def client():
    """直接挂载 /rag 路由（路径前缀 /rag/*）。"""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI(title="RAGInspectTest")
    app.include_router(create_rag_inspect_router())
    return TestClient(app)


@pytest.fixture(scope="module")
def debug_client():
    """经 server.py create_debug_router 嵌套挂载（最终路径 /debug/rag/*），验证接线。"""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.web.server import create_debug_router

    app = FastAPI(title="RAGInspectWiringTest")
    app.include_router(create_debug_router(lambda: None))  # rag 端点不依赖 orchestrator
    return TestClient(app)


# ==================== I-01：文档总览 ====================

def test_docs_overview(done_doc, client):
    """I-01：文档总览含状态徽标、计数、向量覆盖率、未入库标记。"""
    resp = client.get("/rag/docs")
    assert resp.status_code == 200
    docs = resp.json()["documents"]

    # 知识库磁盘文件（Freud / Jung / Philosophy 等 EPUB/PDF）+ DB 记录合并总览
    assert len(docs) >= 5, f"文档数异常：{len(docs)}"

    # done 文档字段完整（块数随入库文档不同，仅断言非空与向量全覆盖）
    freud = next(d for d in docs if d["document_id"] == done_doc["document_id"])
    assert freud["status"] == "done"
    assert freud["sqlite_block_count"] > 0
    assert freud["vector_count"] == freud["sqlite_block_count"], (
        f"Freud 向量覆盖率异常：{freud['vector_count']}/{freud['sqlite_block_count']}"
    )
    # v1.6：术语表弃置留壳（rag_enable_term=False），总览术语计数旁路为 0
    assert freud["term_feature_enabled"] is False
    assert freud["term_count"] == 0
    assert freud["sqlite_summary_count"] >= 1

    # 未入库文件（磁盘有文件但 DB 无记录）若存在，应标记 not_indexed 且无 document_id
    # （全量索引完成后所有文件均入库，not_indexed 可为空——不对其数量做硬断言）
    not_indexed = [d for d in docs if d["status"] == "not_indexed"]
    assert all(d["document_id"] is None for d in not_indexed)

    # 索引中途文档（Jung）覆盖率可被观测（0 向量 / 有 SQLite 块）
    indexing = [d for d in docs if d["status"] in ("indexing", "failed")]
    if indexing:
        j = indexing[0]
        assert j["vector_count"] < j["sqlite_block_count"], "索引中文档应暴露向量缺口"


# ==================== I-02：章节下钻 / 分页 / 摘要互跳 ====================

def test_chapters_and_blocks_pagination(done_doc, client):
    """I-02：章节列表 + 切片分页（limit/offset 正确）+ section_summary 过滤。"""
    doc_id = done_doc["document_id"]

    # 章节列表
    resp = client.get(f"/rag/docs/{doc_id}/chapters")
    assert resp.status_code == 200
    chapters = resp.json()["chapters"]
    assert len(chapters) >= 1, f"章节数异常：{len(chapters)}"
    total_blocks = sum(c["block_count"] for c in chapters)
    total_summaries = sum(
        (c.get("section_summary_count") or 0) + (c.get("chapter_summary_count") or 0)
        for c in chapters
    )
    assert total_blocks == done_doc["sqlite_block_count"]
    assert total_summaries == done_doc["sqlite_summary_count"]

    # 分页：第 1 页与第 2 页不重叠
    p1 = client.get(f"/rag/docs/{doc_id}/blocks", params={"limit": 50, "offset": 0}).json()
    p2 = client.get(f"/rag/docs/{doc_id}/blocks", params={"limit": 50, "offset": 50}).json()
    assert p1["total"] == done_doc["sqlite_block_count"]
    assert len(p1["items"]) == 50 and len(p2["items"]) == 50
    ids1 = {b["block_id"] for b in p1["items"]}
    ids2 = {b["block_id"] for b in p2["items"]}
    assert ids1.isdisjoint(ids2), "分页 offset 未生效"

    # 列表项是截断片段（全文走详情）
    assert all(len(b["snippet"]) <= 201 for b in p1["items"])
    assert all({"block_id", "chapter_title", "block_type", "char_count", "snippet"} <= set(b) for b in p1["items"])

    # 仅节摘要过滤（章摘要单独类型 chapter_summary）
    sm = client.get(f"/rag/docs/{doc_id}/blocks", params={"type": "section_summary"}).json()
    assert sm["total"] == done_doc.get("sqlite_section_summary_count", done_doc["sqlite_summary_count"])
    assert all(b["block_type"] == "section_summary" for b in sm["items"])

    # 章内下钻
    ch_title = chapters[0]["chapter_title"]
    ch = client.get(
        f"/rag/docs/{doc_id}/blocks", params={"chapter": ch_title, "limit": 5}
    ).json()
    assert ch["total"] == chapters[0]["block_count"]
    assert all(b["chapter_title"] == ch_title for b in ch["items"])

    # FTS 关键词（章内/文档内，零配额）
    # 中文语料 unicode61 分词下整串成词，需前缀通配；用实际章名片段命中 chapter_title 列
    # （done 文档随索引进度变化，查询词必须从真实章名动态构造，不能写死特定书章节）
    import re

    def _fts_term(title: str) -> str:
        words = [w for w in re.split(r"[\s第章卷]+", title) if len(w) >= 2]
        return (words[-1] if words else title) + "*"

    ft = client.get(f"/rag/docs/{doc_id}/blocks", params={"q": _fts_term(ch_title), "limit": 10}).json()
    assert ft["total"] >= 1, f"FTS 前缀查询未命中：q={_fts_term(ch_title)!r}"
    assert all(b["document_id"] == doc_id for b in ft["items"])

    # 非法 type → 400；不存在文档 → 404
    assert client.get(f"/rag/docs/{doc_id}/blocks", params={"type": "bogus"}).status_code == 400
    assert client.get("/rag/docs/doc_nonexistent/blocks").status_code == 404


def test_block_detail_parent_retrieval(done_doc, client):
    """I-02：切片全文 + 摘要↔子段落 Parent Retrieval 双向互跳。"""
    doc_id = done_doc["document_id"]
    sm = client.get(f"/rag/docs/{doc_id}/blocks", params={"type": "section_summary", "limit": 10}).json()
    summary_ids = [b["block_id"] for b in sm["items"]]
    assert summary_ids

    # 摘要详情 → 覆盖段落（children）
    resp = client.get(f"/rag/blocks/{summary_ids[0]}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["block"]["block_type"] == "section_summary"
    assert data["parent_summary"] is None
    assert len(data["block"]["parent_block_ids"]) >= 1, "摘要应含 parent_block_ids"
    assert len(data["children"]) >= 1
    assert data["block"]["text"], "摘要详情应含全文"

    # 子段落详情 → 反查所属摘要（parent_summary 回指同一摘要）
    child_id = data["children"][0]["block_id"]
    resp2 = client.get(f"/rag/blocks/{child_id}")
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["block"]["block_type"] != "section_summary"
    # 普通段落详情为全文（不截断）
    assert len(data2["block"]["text"]) >= len(data["children"][0]["snippet"])
    assert data2["parent_summary"] is not None, "子段落应反查到章节摘要"
    assert data2["parent_summary"]["block_id"] == summary_ids[0]

    # 404
    assert client.get("/rag/blocks/nonexistent_block_id").status_code == 404


# ==================== I-03：术语表 ====================

def test_terms_search_and_detail(done_doc, client):
    """I-03：术语分页 + 搜索 + 详情（同义词/出现片段）。

    v1.6 术语表弃置留壳（rag_enable_term=False）：功能关闭时端点返回空集 +
    term_feature_enabled=False，测试验证旁路行为；功能重新开启后恢复完整断言。
    """
    resp = client.get("/rag/terms", params={"limit": 100})
    assert resp.status_code == 200
    data = resp.json()

    if not data.get("term_feature_enabled", False):
        assert data["total"] == 0 and data["items"] == []
        # 详情端点同样旁路（404 而非 500）
        assert client.get("/rag/terms/any_term_id").status_code == 404
        return

    assert data["total"] == done_doc["term_count"], f"术语总数异常：{data['total']}"
    assert len(data["items"]) == 100

    first = data["items"][0]
    assert {"term_id", "canonical_term", "domain", "synonyms", "occurrence_count"} <= set(first)

    # 分页不重叠
    page2 = client.get("/rag/terms", params={"limit": 100, "offset": 100}).json()
    ids_a = {t["term_id"] for t in data["items"]}
    ids_b = {t["term_id"] for t in page2["items"]}
    assert ids_a.isdisjoint(ids_b)

    # 详情
    resp_t = client.get(f"/rag/terms/{first['term_id']}")
    assert resp_t.status_code == 200
    td = resp_t.json()
    assert td["term"]["term_id"] == first["term_id"]
    assert "canonical_term" in td["term"]
    assert isinstance(td["term"].get("synonyms"), list)
    assert isinstance(td["occurrences"], list)
    # 出现片段（若有）结构正确
    for occ in td["occurrences"]:
        assert "block_id" in occ and "snippet" in occ

    # 搜索：Freud 英文索引术语（canonical 或同义词命中）
    srch = client.get("/rag/terms", params={"q": "a", "limit": 5}).json()
    assert srch["total"] >= 1


# ==================== I-04：检索试跑 ====================

def test_retrieve_fts_zero_quota(done_doc, client):
    """I-04（FTS）：BM25 关键词检索零配额，返回分数与片段。"""
    # 中文语料 unicode61 整串成词，用章名前缀通配命中 chapter_title 列
    resp = client.get("/rag/retrieve", params={"q": "性学三论*", "mode": "fts", "k": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "fts"
    assert len(data["items"]) >= 1
    top = data["items"][0]
    assert top["score"] > 0, "BM25 分数应 > 0"
    assert "bm25" in top["score_type"]
    assert top["snippet"]

    # domain 过滤 + 参数校验
    assert client.get("/rag/retrieve", params={"q": "性学三论*", "domain": "bogus"}).status_code == 400
    assert client.get("/rag/retrieve", params={"q": "性学三论*", "mode": "bogus"}).status_code == 400


def test_retrieve_vector_branch_with_fake_embedding(done_doc):
    """I-04（vector 分支）：注入 FakeEmbedding 验证语义检索执行路径（零 API 配额）。

    真实 Embedding API 调用待配额恢复后在 Web 界面人工验收（界面有配额提示）。
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    store = RAGStore(get_settings())
    peek = store._get_collection().peek(1)
    emb = peek.get("embeddings")
    dim = len(emb[0]) if emb is not None and len(emb) > 0 else 0
    assert dim > 100, f"向量维度异常：{dim}"

    fake = _FakeDimEmbedding(dim)
    app = FastAPI()
    app.include_router(create_rag_inspect_router(embedding_client=fake))
    c = TestClient(app)

    resp = c.get("/rag/retrieve", params={"q": "dream", "mode": "vector", "k": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "vector"
    assert fake.calls == 1, "vector 模式应调用一次 embed_query"
    assert len(data["items"]) == 5
    for it in data["items"]:
        # cosine 相似度 0~1
        assert 0.0 <= it["score"] <= 1.0, f"cosine 分数量纲异常：{it['score']}"
        assert "cosine" in it["score_type"]
        assert it["block_id"] and it["snippet"]


def test_retrieve_vector_quota_error_is_503(done_doc):
    """I-04（降级提示）：Embedding 失败（如 429 配额超限）返回 503 + 重置提示，不 500。"""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(create_rag_inspect_router(embedding_client=_RaisingEmbedding()))
    c = TestClient(app)

    resp = c.get("/rag/retrieve", params={"q": "dream", "mode": "vector", "k": 5})
    assert resp.status_code == 503
    body = resp.json()
    assert "配额" in body["detail"] or "Embedding" in body["detail"]
    assert "AccountQuotaExceeded" in body["error"]
    assert body["hint"]  # 前端可展示的引导文案


# ==================== I-05：只读 / 403 / 处理指引 ====================

def test_non_localhost_blocked(done_doc):
    """I-05：非 127.0.0.1 访问全部 rag 端点返回 403。"""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(create_rag_inspect_router())
    remote = TestClient(app, client=("8.8.8.8", 12345))
    resp = remote.get("/rag/docs")
    assert resp.status_code == 403


def test_endpoints_are_readonly(done_doc, client):
    """I-05：全部端点仅 GET（POST/PUT/DELETE 不可用）。"""
    doc_id = done_doc["document_id"]
    for path in ("/rag/docs", f"/rag/docs/{doc_id}/chapters", "/rag/terms"):
        assert client.post(path, json={}).status_code in (404, 405), f"POST {path} 应被拒绝"
        assert client.delete(path).status_code in (404, 405), f"DELETE {path} 应被拒绝"


def test_debug_router_nested_mount(done_doc, debug_client):
    """I-05：经 create_debug_router 嵌套挂载后 /debug/rag/* 可用（验证 server.py 接线）。"""
    resp = debug_client.get("/debug/rag/docs")
    assert resp.status_code == 200
    assert len(resp.json()["documents"]) >= 5

    # 静态页面包含知识库检查视图与处理指引
    page = debug_client.get("/debug")
    assert page.status_code == 200
    html = page.text
    assert "知识库检查" in html
    assert "reindex_all.py" in html, "页面应内置索引问题处理指引"
    assert "/debug/rag" in html or "kb-reload-docs" in html
