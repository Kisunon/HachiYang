"""配额超限自动等待 + 断点续跑测试（Phase 3.25 阶段 3，零 API 配额）。

覆盖：
- is_quota_exceeded：AccountQuotaExceeded（配额耗尽）vs AccountRateLimitExceeded
  （瞬时频控）的区分；
- run_with_quota_wait：配额错误等待后重试同一工厂直至成功，非配额错误直接抛出；
- 全链路「报错条目不遗漏」：Embedding 批次/节摘要首次调用抛配额错误 → 等待后重试成功，
  最终文档 done、块/向量 1:1 无缺失；
- 断点续跑：pending/failed 的同 hash 书重跑时复用正文向量（零重复 Embedding），
  摘要清除重建，FTS/向量无重复残留。

sleep 通过 monkeypatch asyncio.sleep 即时返回，测试不产生真实等待。
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.db.connection import db_conn
from app.llm.quota import is_quota_exceeded, run_with_quota_wait
from app.rag.indexer import indexer_from_chunks

from tests.test_stage3_index_chunks import (
    FakeEmbedding,
    FakeLLM,
    _all_db_blocks,
    _make_env,
    _make_two_volume_book,
)

_QUOTA_MSG = (
    "Error code: 429 - {'error': {'code': 'AccountQuotaExceeded', "
    "'message': 'You have exceeded the 5-hour usage quota for your coding plan.'}}"
)
_RATE_MSG = (
    "Error code: 429 - {'error': {'code': 'AccountRateLimitExceeded', "
    "'message': 'Requests are too frequent, please slow down.'}}"
)


# ==================== 配额识别 ====================

def test_is_quota_exceeded_classification():
    assert is_quota_exceeded(RuntimeError(_QUOTA_MSG)) is True
    assert is_quota_exceeded(RuntimeError("您的 Coding Plan 配额已超限，请稍后再试")) is True
    assert is_quota_exceeded(RuntimeError(_RATE_MSG)) is False
    assert is_quota_exceeded(RuntimeError("connection reset by peer")) is False

    e1 = RuntimeError("429")
    e1.body = {"error": {"code": "AccountQuotaExceeded", "message": "quota exceeded"}}
    assert is_quota_exceeded(e1) is True
    e2 = RuntimeError("429")
    e2.body = {"error": {"code": "AccountRateLimitExceeded"}}
    assert is_quota_exceeded(e2) is False


# ==================== run_with_quota_wait ====================

def test_run_with_quota_wait_retries_same_item_until_success():
    calls = []
    slept: list[float] = []

    async def _fake_sleep(s):
        slept.append(s)

    async def factory():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError(_QUOTA_MSG)
        return "ok"

    async def _main():
        return await run_with_quota_wait(
            factory, what="测试条目", wait_seconds=7200, sleep=_fake_sleep,
        )

    result = asyncio.run(_main())
    assert result == "ok"
    assert len(calls) == 3, "失败条目必须被重试到成功为止"
    # 两次等待，每次 7200s（60s 分段）
    assert sum(slept) == 2 * 7200


def test_run_with_quota_wait_passes_through_non_quota():
    calls = []
    slept: list[float] = []

    async def _fake_sleep(s):
        slept.append(s)

    async def factory():
        calls.append(1)
        raise ValueError("内容敏感，不可重试")

    async def _main():
        await run_with_quota_wait(
            factory, what="测试条目", wait_seconds=7200, sleep=_fake_sleep,
        )

    with pytest.raises(ValueError):
        asyncio.run(_main())
    assert len(calls) == 1, "非配额错误不重试"
    assert not slept


# ==================== 全链路：等待后无遗漏 ====================

@pytest.fixture
def instant_quota_sleep(monkeypatch):
    """把 asyncio.sleep 替换为即时 fake（run_with_quota_wait 经 asyncio.sleep 解析）。"""
    slept: list[float] = []

    async def _fake(seconds):
        slept.append(seconds)

    monkeypatch.setattr("asyncio.sleep", _fake)
    return slept


class FlakyEmbedding(FakeEmbedding):
    """前 fail_first 次 embed_texts 抛配额错误，之后正常。"""

    def __init__(self, fail_first: int = 1) -> None:
        super().__init__()
        self.fail_first = fail_first

    async def embed_texts(self, texts):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise RuntimeError(_QUOTA_MSG)
        return [self._vec(t) for t in texts]


class FlakyLLM(FakeLLM):
    """前 fail_first 次 chat 抛配额错误，之后正常（验证失败摘要条目被补回）。"""

    def __init__(self, fail_first: int = 2) -> None:
        super().__init__()
        self.fail_first = fail_first

    async def chat(self, messages, temperature=0.3, max_tokens=400, timeout=15.0):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise RuntimeError(_QUOTA_MSG)
        return await super().chat(messages, temperature, max_tokens, timeout)


def test_embedding_batch_quota_wait_no_item_lost(tmp_path, instant_quota_sleep):
    """首个 Embedding 批次配额失败 → 等待 2h（fake 即时）后重试；最终无缺失。"""
    settings, chunks_dir, store = _make_env(tmp_path)
    _make_two_volume_book(chunks_dir)

    results = asyncio.run(indexer_from_chunks(
        chunks_root_dir=chunks_dir,
        embedding=FlakyEmbedding(fail_first=1),
        llm=FakeLLM(),
        store=store,
        settings=settings,
    ))
    r = results[0]
    assert r["status"] == "done", r
    assert r["failed_sections"] == 0 and r["failed_chapters"] == 0
    # 7 正文 + 3 节摘要 + 2 章摘要 = 12，向量 1:1
    blocks = _all_db_blocks(store, r["document_id"])
    assert len(blocks) == 12
    assert store._get_collection().count() == 12
    # 触发过至少一次 2 小时等待（120 个 60s 分段）
    assert sum(instant_quota_sleep) >= 7200


def test_section_summary_quota_wait_then_all_summaries_present(tmp_path, instant_quota_sleep):
    """节摘要 LLM 配额失败 → 等待重试；最终 3 节 + 2 章摘要齐全（报错条目不遗漏）。"""
    settings, chunks_dir, store = _make_env(tmp_path)
    _make_two_volume_book(chunks_dir)

    results = asyncio.run(indexer_from_chunks(
        chunks_root_dir=chunks_dir,
        embedding=FakeEmbedding(),
        llm=FlakyLLM(fail_first=3),
        store=store,
        settings=settings,
    ))
    r = results[0]
    assert r["status"] == "done", r
    assert r["section_summary_count"] == 3, "配额失败的节摘要必须在重试后补齐"
    assert r["chapter_summary_count"] == 2
    assert r["failed_sections"] == 0
    assert sum(instant_quota_sleep) >= 7200


# ==================== 断点续跑 ====================

def _count_fts_and_blocks(settings, document_id: str) -> tuple[int, int]:
    with db_conn(settings) as conn:
        n_fts = conn.execute("SELECT COUNT(*) FROM rag_fts").fetchone()[0]
        n_blocks = conn.execute(
            "SELECT COUNT(*) FROM rag_blocks WHERE document_id=?", (document_id,)
        ).fetchone()[0]
    return n_fts, n_blocks


def test_resume_pending_book_reuses_text_vectors(tmp_path):
    """先 --no-summaries 入库（pending），再带摘要重跑：正文向量零重复 Embedding。"""
    settings, chunks_dir, store = _make_env(tmp_path)
    _make_two_volume_book(chunks_dir)

    r1 = asyncio.run(indexer_from_chunks(
        chunks_root_dir=chunks_dir,
        embedding=FakeEmbedding(), llm=None, store=store, settings=settings,
        enable_summaries=False,
    ))[0]
    assert r1["status"] == "indexed_but_summary_pending"
    assert store._get_collection().count() == 7

    emb2 = FakeEmbedding()
    r2 = asyncio.run(indexer_from_chunks(
        chunks_root_dir=chunks_dir,
        embedding=emb2, llm=FakeLLM(), store=store, settings=settings,
    ))[0]
    assert r2["status"] == "done", r2
    assert r2["resumed"] is True
    # 正文 7 块全部复用：仅 5 个摘要块构成 1 个 Embedding 批次
    assert emb2.calls == 1, f"续跑不应重复 Embedding 正文，实际调用 {emb2.calls} 次"
    assert store._get_collection().count() == 12

    blocks = _all_db_blocks(store, r2["document_id"])
    assert len(blocks) == 12
    n_fts, n_blocks = _count_fts_and_blocks(settings, r2["document_id"])
    assert n_blocks == 12 and n_fts == 12, "FTS 不得有重复行"


def test_resume_failed_book_rebuilds_summaries_keeps_text(tmp_path):
    """done 后人为置 failed（模拟中断于摘要阶段）：重跑清旧摘要、复用正文、无残留。"""
    settings, chunks_dir, store = _make_env(tmp_path)
    _make_two_volume_book(chunks_dir)

    r1 = asyncio.run(indexer_from_chunks(
        chunks_root_dir=chunks_dir,
        embedding=FakeEmbedding(), llm=FakeLLM(), store=store, settings=settings,
    ))[0]
    assert r1["status"] == "done"
    doc_id = r1["document_id"]
    old_blocks = _all_db_blocks(store, doc_id)
    old_summary_ids = {b["block_id"] for b in old_blocks if b["block_type"] != "text"}
    assert len(old_summary_ids) == 5
    text_ids = {b["block_id"] for b in old_blocks if b["block_type"] == "text"}

    # 模拟中断：状态打回 failed
    store.set_document_status(doc_id, "failed")

    emb2 = FakeEmbedding()
    r2 = asyncio.run(indexer_from_chunks(
        chunks_root_dir=chunks_dir,
        embedding=emb2, llm=FakeLLM(), store=store, settings=settings,
    ))[0]
    assert r2["resumed"] is True
    assert r2["status"] == "done", r2
    assert emb2.calls == 1, "续跑仅 Embedding 新生成的 5 个摘要块（1 批）"

    blocks = _all_db_blocks(store, doc_id)
    by_id = {b["block_id"]: b for b in blocks}
    new_summary_ids = {bid for bid, b in by_id.items() if b["block_type"] != "text"}
    # 旧摘要全部清除、正文全部保留
    assert not (new_summary_ids & old_summary_ids), "旧摘要块必须删除重建"
    assert text_ids <= set(by_id), "正文块必须原样复用"
    assert len(blocks) == 12 and len(new_summary_ids) == 5
    assert store._get_collection().count() == 12
    n_fts, n_blocks = _count_fts_and_blocks(settings, doc_id)
    assert n_blocks == 12 and n_fts == 12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
