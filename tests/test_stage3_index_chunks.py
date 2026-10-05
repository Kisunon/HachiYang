"""阶段 3（Embedding + 两级摘要入库）三阶段流水线测试（Phase 3.25，零 API 配额）。

对齐 RAG_SYSTEM_DESIGN §3.4 / §3.5 与验收项 R-12：
- chunks.json → 同书多卷合并为一个 rag_documents（卷列表 volumes_json）；
- block_id 沿用 chunks.json 的 uuid；正文块写入 rag_blocks / rag_fts / ChromaDB；
- 两级摘要：节摘要 parent_block_ids 全部指向存在的正文块；
  章摘要 parent_block_ids 全部指向存在的节摘要块（R-12）；
- 摘要块回填 chunks.json（blocks 追加 + stats 计数），重跑幂等；
- 容错状态：--no-summaries → indexed_but_summary_pending；章摘要失败 → summary_partial；
- 术语表旁路：rag_enable_term=False 时 term 查询全部返回空。

全部使用 Fake Embedding / Fake LLM 与临时 sqlite/chroma 目录，不调用任何真实 API。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import pytest

from app.config import Settings
from app.db.connection import init_db
from app.rag.indexer import indexer_from_chunks
from app.rag.stores import RAGStore

_BOOK = "测试文集（合成）"
_DOMAIN = "freud"

# 足量正文（>300 字/节，触发节摘要）；单段 <500 字不软切
_PARA_A = "精神分析学认为，人类的心理活动可以分为意识与无意识两个层次，二者相互转化。" * 8
_PARA_B = "力比多是一种驱动个体寻求快感的心理能量，它在不同发展阶段有不同的表现形式。" * 8
_PARA_SHORT = "短节，正文不足三百字，按设计应跳过节摘要且不计失败。" * 2


# ==================== Fake 客户端 ====================

class FakeEmbedding:
    """确定性假向量（8 维），零 API。"""

    def __init__(self) -> None:
        self.calls = 0

    @staticmethod
    def _vec(text: str) -> list[float]:
        import hashlib
        h = hashlib.md5(text.encode("utf-8")).digest()
        return [(h[i % len(h)] / 255.0) - 0.5 for i in range(8)]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vec(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FakeLLM:
    """按 system prompt 区分节/章摘要；可对含指定关键词的章摘要注入失败。"""

    def __init__(self, fail_chapter_keyword: str | None = None) -> None:
        self.fail_chapter_keyword = fail_chapter_keyword
        self.calls = 0

    async def chat(self, messages, temperature: float = 0.3, max_tokens: int = 400,
                   timeout: float = 15.0) -> str:
        self.calls += 1
        system = messages[0]["content"] if messages else ""
        user = messages[-1]["content"] if messages else ""
        is_chapter = "章摘要" in system or "全章" in system
        if is_chapter:
            if self.fail_chapter_keyword and self.fail_chapter_keyword in user:
                raise RuntimeError("fake chapter LLM failure（模拟章摘要限流）")
            return "本章综合各节摘要，围绕核心论题逐层展开，形成连贯的总体论述与结论。" * 3
        return "本节概括了该部分的核心论点、论证过程与主要结论，语言客观凝练。" * 3


# ==================== 合成 chunks.json ====================

def _make_block(book: str, vol_index: int, vol_title: str, chapter: str,
                section: str, text: str, block_index: int, anchor: str) -> dict:
    return {
        "block_id": uuid.uuid4().hex,
        "block_type": "text",
        "parent_ref": {
            "book_title": book,
            "volume_title": vol_title,
            "volume_index": vol_index,
            "chapter_title": chapter,
            "section_title": section,
            "subsection_title": "",
        },
        "block_index_in_section": block_index,
        "text": text,
        "char_count": len(text),
        "language": "zh",
        "source_anchor_in_md": anchor,
        "soft_split_from_paragraphs": [block_index],
    }


def _write_chunks(path: Path, vol_index: int, vol_title: str,
                  sections: list[tuple[str, str, list[str]]]) -> Path:
    """sections: [(chapter_title, section_title, [paragraph, ...]), ...]"""
    blocks: list[dict] = []
    for ch_idx, (chapter, section, paras) in enumerate(sections, start=1):
        for pi, para in enumerate(paras):
            blocks.append(_make_block(
                _BOOK, vol_index, vol_title, chapter, section, para,
                block_index=pi, anchor=f"#ch-{ch_idx}-sec-{pi + 1}",
            ))
    payload = {
        "schema_version": 2,
        "source_ref": {
            "book_title": _BOOK,
            "domain": _DOMAIN,
            "volume_index": vol_index,
            "volume_title": vol_title,
            "content_hash": f"fakehash{vol_index:02d}",
            "extracted_json": str(path.with_suffix(".extracted.json")),
            "extracted_md": str(path.with_suffix(".extracted.md")),
        },
        "blocks": blocks,
        "stats": {
            "total_blocks": len(blocks),
            "total_chapter_summary_blocks": 0,
            "total_section_summary_blocks": 0,
            "section_count": len(sections),
            "avg_chars": round(sum(b["char_count"] for b in blocks) / max(len(blocks), 1), 1),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _make_env(tmp_path: Path) -> tuple[Settings, Path, RAGStore]:
    settings = Settings(
        sqlite_path=str(tmp_path / "test.db"),
        chroma_dir=str(tmp_path / "chroma"),
        rag_data_dir=str(tmp_path / "knowledge_base"),
        rag_extracted_dir=str(tmp_path / "knowledge_extracted"),
        rag_chunks_dir=str(tmp_path / "knowledge_chunks"),
        rag_enable_term=False,
    )
    init_db(settings)
    store = RAGStore(settings)
    return settings, tmp_path / "knowledge_chunks", store


def _make_two_volume_book(chunks_dir: Path) -> dict:
    """卷1：第一章 甲论（2 个长节）；卷2：第二章 乙论（1 个长节 + 1 个短节）。"""
    v1 = _write_chunks(
        chunks_dir / _DOMAIN / "卷01-甲论.chunks.json", 1, "卷1 甲论",
        [("第一章 甲论", "第一节 总论", [_PARA_A, _PARA_B]),
         ("第一章 甲论", "第二节 分论", [_PARA_A, _PARA_B])],
    )
    v2 = _write_chunks(
        chunks_dir / _DOMAIN / "卷02-乙论.chunks.json", 2, "卷2 乙论",
        [("第二章 乙论", "第一节 源流", [_PARA_A, _PARA_B]),
         ("第二章 乙论", "第二节 短节", [_PARA_SHORT])],
    )
    return {"v1": v1, "v2": v2}


def _all_db_blocks(store: RAGStore, document_id: str) -> list[dict]:
    return store.paginate_blocks(document_id, limit=2000, offset=0)


# ==================== 测试 ====================

def test_stage3_full_two_level_pipeline(tmp_path):
    """R-12 主路径：两卷合并入库 + 两级摘要 + 父子指针全部有效 + chunks 回填。"""
    settings, chunks_dir, store = _make_env(tmp_path)
    files = _make_two_volume_book(chunks_dir)

    results = asyncio.run(indexer_from_chunks(
        chunks_root_dir=chunks_dir,
        embedding=FakeEmbedding(),
        llm=FakeLLM(),
        store=store,
        settings=settings,
    ))

    assert len(results) == 1
    r = results[0]
    assert r["status"] == "done", r
    assert r["volume_count"] == 2
    # 正文块：卷1 4 + 卷2 3（短节段落仍作为正文入库）= 7
    assert r["text_block_count"] == 7
    # 节摘要：3 个长节（短节 <300 字跳过，不计失败）
    assert r["section_summary_count"] == 3
    # 章摘要：2 章
    assert r["chapter_summary_count"] == 2
    assert r["failed_sections"] == 0 and r["failed_chapters"] == 0

    doc_id = r["document_id"]
    doc = store.get_document(doc_id)
    assert doc["status"] == "done"
    assert doc["volume_count"] == 2
    volumes = json.loads(doc["volumes_json"] or "[]")
    assert [v["volume_index"] for v in volumes] == [1, 2]
    assert doc["section_summary_count"] == 3 and doc["chapter_summary_count"] == 2

    blocks = _all_db_blocks(store, doc_id)
    by_id = {b["block_id"]: b for b in blocks}
    text_blocks = [b for b in blocks if b["block_type"] == "text"]
    sec_blocks = [b for b in blocks if b["block_type"] == "section_summary"]
    ch_blocks = [b for b in blocks if b["block_type"] == "chapter_summary"]
    assert len(text_blocks) == 7 and len(sec_blocks) == 3 and len(ch_blocks) == 2

    # R-12：节摘要 parent_block_ids 全部指向存在的正文块
    for sb in sec_blocks:
        parents = json.loads(sb["parent_block_ids"] or "[]")
        assert parents, "节摘要必须有 parent_block_ids"
        for pid in parents:
            assert pid in by_id, f"节摘要 {sb['block_id'][:8]} 指向不存在的块 {pid[:8]}"
            assert by_id[pid]["block_type"] == "text"
        assert sb["parent_chapter_summary_id"], "节摘要必须回填 parent_chapter_summary_id"
        assert sb["parent_chapter_summary_id"] in by_id

    # R-12：章摘要 parent_block_ids 全部指向存在的节摘要块
    for cb in ch_blocks:
        parents = json.loads(cb["parent_block_ids"] or "[]")
        assert parents, "章摘要必须有 parent_block_ids"
        for pid in parents:
            assert pid in by_id, f"章摘要 {cb['block_id'][:8]} 指向不存在的块 {pid[:8]}"
            assert by_id[pid]["block_type"] == "section_summary"

    # 溯源字段落库
    vol1_text = [b for b in text_blocks if b["volume_index"] == 1]
    assert vol1_text and all(b["volume_title"] == "卷1 甲论" for b in vol1_text)
    assert all(b["source_anchor_in_md"].startswith("#ch-") for b in text_blocks)
    assert all(b["extracted_md_path"].endswith(".extracted.md") for b in text_blocks)

    # block_id 沿用 chunks.json 的 uuid
    chunks_v1 = json.loads(files["v1"].read_text(encoding="utf-8"))
    chunk_ids = {b["block_id"] for b in chunks_v1["blocks"] if b["block_type"] == "text"}
    db_ids = {b["block_id"] for b in text_blocks if b["volume_index"] == 1}
    assert chunk_ids == db_ids, "正文 block_id 必须与 chunks.json 完全一致"

    # ChromaDB 向量数 == 正文 + 摘要
    assert store._get_collection().count() == 7 + 3 + 2

    # FTS5 可检索（unicode61 对中文按连续串成词，用前缀通配匹配段首词）
    hits = store.query_fts("精神分析*", n=10)
    assert hits, "FTS5 应能检索到正文"

    # chunks.json 回填：摘要块追加 + stats 计数
    for key, n_sec in (("v1", 2), ("v2", 1)):
        data = json.loads(files[key].read_text(encoding="utf-8"))
        assert data["stats"]["total_section_summary_blocks"] == n_sec
        assert data["stats"]["total_chapter_summary_blocks"] == 1
        appended = [b for b in data["blocks"] if b["block_type"] != "text"]
        assert len(appended) == n_sec + 1
        for b in appended:
            assert b["parent_block_ids"], "回填摘要块必须带 parent_block_ids"

    # 重跑幂等：status=done 且 hash 未变 → 跳过，chunks 摘要不重复累积
    results2 = asyncio.run(indexer_from_chunks(
        chunks_root_dir=chunks_dir,
        embedding=FakeEmbedding(), llm=FakeLLM(), store=store, settings=settings,
    ))
    assert results2[0]["skipped"] is True
    data = json.loads(files["v1"].read_text(encoding="utf-8"))
    assert sum(1 for b in data["blocks"] if b["block_type"] != "text") == 3


def test_stage3_no_summaries_pending_status(tmp_path):
    """enable_summaries=False：正文入库、零摘要块、状态 indexed_but_summary_pending。"""
    settings, chunks_dir, store = _make_env(tmp_path)
    _make_two_volume_book(chunks_dir)

    results = asyncio.run(indexer_from_chunks(
        chunks_root_dir=chunks_dir,
        embedding=FakeEmbedding(),
        llm=None,
        store=store,
        settings=settings,
        enable_summaries=False,
    ))
    r = results[0]
    assert r["status"] == "indexed_but_summary_pending"
    assert r["section_summary_count"] == 0 and r["chapter_summary_count"] == 0

    blocks = _all_db_blocks(store, r["document_id"])
    assert len(blocks) == 7 and all(b["block_type"] == "text" for b in blocks)
    assert store._get_collection().count() == 7


def test_stage3_chapter_failure_partial(tmp_path):
    """章摘要 LLM 失败 → summary_partial；节摘要照常，R-12 对已生成块仍成立。"""
    settings, chunks_dir, store = _make_env(tmp_path)
    _make_two_volume_book(chunks_dir)

    results = asyncio.run(indexer_from_chunks(
        chunks_root_dir=chunks_dir,
        embedding=FakeEmbedding(),
        llm=FakeLLM(fail_chapter_keyword="乙论"),
        store=store,
        settings=settings,
    ))
    r = results[0]
    assert r["status"] == "summary_partial", r
    assert r["chapter_summary_count"] == 1  # 甲论章成功、乙论章失败
    assert r["section_summary_count"] == 3
    assert r["failed_chapters"] == 1

    blocks = _all_db_blocks(store, r["document_id"])
    by_id = {b["block_id"]: b for b in blocks}
    for b in blocks:
        if b["block_type"] in ("section_summary", "chapter_summary"):
            for pid in json.loads(b["parent_block_ids"] or "[]"):
                assert pid in by_id


def test_stage3_only_file_indexes_whole_book(tmp_path):
    """--file 单卷：整书（含其他卷）合并重建。"""
    settings, chunks_dir, store = _make_env(tmp_path)
    files = _make_two_volume_book(chunks_dir)

    results = asyncio.run(indexer_from_chunks(
        only_file=files["v1"],
        embedding=FakeEmbedding(),
        llm=FakeLLM(),
        store=store,
        settings=settings,
    ))
    assert len(results) == 1
    r = results[0]
    assert r["status"] == "done"
    assert r["volume_count"] == 2
    blocks = _all_db_blocks(store, r["document_id"])
    assert {b["volume_index"] for b in blocks} == {1, 2}


def test_stage3_term_feature_bypassed(tmp_path):
    """rag_enable_term=False：术语查询全部旁路返回空（§3.6.1）。"""
    settings, chunks_dir, store = _make_env(tmp_path)
    _make_two_volume_book(chunks_dir)
    asyncio.run(indexer_from_chunks(
        chunks_root_dir=chunks_dir,
        embedding=FakeEmbedding(), llm=FakeLLM(), store=store, settings=settings,
    ))
    assert store.term_feature_enabled is False
    assert store.count_terms("anything") == 0
    assert store.paginate_terms() == []
    assert store.query_terms("精神分析") == []
    health = store.health_check()
    assert health["term_disabled"] is True
    assert health["chapter_summary_count"] == 2
    assert health["section_summary_count"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
