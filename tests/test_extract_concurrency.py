"""术语提取并发调度离线测试（零 API）。

覆盖 scripts/extract_terms.py 的并发改造：
- 批次级并发不超过 AIMD 限流上限；
- 拆半子任务按 slot 保序合并，结果与串行一致；
- 频控 429 乘性降并发 + 冷却重放；配额门全队只等待一次；
- 瞬时错误尝试预算耗尽后拆半；内容审核拆半/单块跳过；空响应重试后拆半；
- 停止标志优雅收尾（失败节不落盘）；断点续跑跳过已完成节。

运行：python -m pytest tests/test_extract_concurrency.py -v
（项目惯例：同步测试 + asyncio.run，不依赖 pytest-asyncio）
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from unittest import mock

import openai
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import extract_terms as et  # noqa: E402


# ==================== 测试基础设施 ====================

def _terms_json(*zhs: str) -> str:
    return json.dumps(
        [{"zh": z, "en": f"EN-{z}", "aliases_zh": [], "aliases_en": [], "definition": f"{z}的简要解释"}
         for z in zhs],
        ensure_ascii=False,
    )


def _err(exc_cls, code: str | None = None, message: str = "boom", status: int = 429):
    """构造 openai 风格异常（response/body 用 mock，避免真实 httpx 对象）。"""
    if exc_cls in (openai.APITimeoutError, openai.APIConnectionError):
        return exc_cls(request=mock.MagicMock())
    resp = mock.MagicMock()
    resp.status_code = status
    body = {"error": {"code": code, "message": message}} if code else None
    return exc_cls(message, response=resp, body=body)


class FakeLLM:
    """按输入文本中的标记脚本化返回；记录并发峰值与调用次数。

    behavior: {marker: result}，result 为：
      - str：直接返回（JSON 字符串）；
      - Exception：抛出；
      - callable(seen_count) -> str|Exception：每次调用时动态决定。
    """

    def __init__(self, behavior: dict, delay: float = 0.02):
        self.behavior = behavior
        self.delay = delay
        self.calls = 0
        self.active = 0
        self.peak = 0
        self._seen: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def chat(self, messages, *, json_mode=False, temperature=0.2, max_tokens=8192,
                   timeout=240.0, max_retries=1, sdk_max_retries=0):
        content = messages[-1]["content"] if messages else ""
        marker = next((m for m in self.behavior if m in content), None)
        async with self._lock:
            self.calls += 1
            self.active += 1
            self.peak = max(self.peak, self.active)
            n = self._seen.get(marker, 0) + 1
            self._seen[marker] = n
        await asyncio.sleep(self.delay)
        try:
            if marker is None:
                return _terms_json("默认术语")
            res = self.behavior[marker]
            if callable(res):
                res = res(n)
            if isinstance(res, Exception):
                raise res
            return res
        finally:
            async with self._lock:
                self.active -= 1


def _make_chunks(tmp: Path, specs: list[tuple[str, str, list[str]]]) -> Path:
    """specs: [(chapter_title, section_title, [block_text, ...]), ...]"""
    blocks = []
    for chap, sec, texts in specs:
        for ti, t in enumerate(texts):
            blocks.append({
                "block_type": "text",
                "text": t,
                "parent_ref": {
                    "book_title": "测试书", "volume_title": "卷T", "volume_index": 1,
                    "chapter_title": chap, "section_title": sec, "subsection_title": "",
                },
                "block_index_in_section": ti,
            })
    p = tmp / "test.chunks.json"
    p.write_text(json.dumps({"source_ref": {"volume_title": "卷T"}, "blocks": blocks}, ensure_ascii=False),
                 encoding="utf-8")
    return p


def _make_ctx(llm, tmp: Path, *, concurrency: int = 3, wait_seconds: int = 0, sleep=None) -> et.RunCtx:
    return et.RunCtx(
        llm=llm,
        limiter=et.AdaptiveLimiter(concurrency),
        quota=et.QuotaGate(wait_seconds, sleep=sleep),
        stats=et.RunStats(),
        stop_flag=tmp / "no_such_stop.flag",
        timeout=240.0,
    )


async def _run_volume(tmp: Path, llm: FakeLLM, specs, *, concurrency=3, batch_chars=3500,
                      stop: bool = False, prebuilt_out: dict | None = None):
    chunks = _make_chunks(tmp, specs)
    out = tmp / "out.terms.json"
    if prebuilt_out is not None:
        out.write_text(json.dumps(prebuilt_out, ensure_ascii=False), encoding="utf-8")
    ctx = _make_ctx(llm, tmp, concurrency=concurrency)
    if stop:
        ctx.stop_flag.write_text("stop", encoding="utf-8")
    stat = await et.extract_volume(chunks, out, "freud", ctx,
                                   batch_chars=batch_chars, dry_run=False)
    data = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"sections": []}
    return stat, data, ctx


def _two_batch_section(marker1: str, marker2: str):
    """一节两块、各 2000 字符（batch_chars=500 时切成 2 批）。"""
    return ("章一", "节一", [f"{marker1}" + "内容" * 500, f"{marker2}" + "内容" * 500])


_WARN_BUFFER: list[str] = []


@pytest.fixture(autouse=True)
def _attach_warn_buffer(monkeypatch):
    handler = logging.Handler()
    handler.emit = lambda record: _WARN_BUFFER.append(record.getMessage())
    logger = logging.getLogger("extract_terms")
    logger.addHandler(handler)
    _WARN_BUFFER.clear()
    # 退避/冷却归零，测试不等待真实时间
    monkeypatch.setattr(et, "_BACKOFF", [0, 0, 0])
    monkeypatch.setattr(et, "_RATE_LIMIT_COOLDOWN", 0)
    yield
    logger.removeHandler(handler)


# ==================== 测试用例 ====================

def test_concurrency_respects_limit_and_runs_parallel(tmp_path):
    async def _main():
        specs = [
            ("章一", "节一", ["【M1】" + "甲" * 700, "【M2】" + "乙" * 700]),
            ("章二", "节二", ["【M3】" + "丙" * 700, "【M4】" + "丁" * 700]),
        ]
        llm = FakeLLM({m: _terms_json(f"术语{m}") for m in ["【M1】", "【M2】", "【M3】", "【M4】"]},
                      delay=0.15)
        loop = asyncio.get_event_loop()
        t0 = loop.time()
        stat, data, ctx = await _run_volume(tmp_path, llm, specs, concurrency=3, batch_chars=500)
        elapsed = loop.time() - t0
        return stat, data, llm, elapsed

    stat, data, llm, elapsed = asyncio.run(_main())
    assert stat["done"] == 2
    assert llm.peak >= 3, f"应发生并发，峰值={llm.peak}"
    assert llm.peak <= 3, f"并发不得超过上限，峰值={llm.peak}"
    # 4 批 × 0.15s：串行 ≈0.6s，3 并发应 <0.45s
    assert elapsed < 0.45, f"并发未提速：{elapsed:.2f}s"
    assert len(data["sections"]) == 2


def test_slot_order_merge_is_stable(tmp_path):
    async def _main():
        specs = [_two_batch_section("【M1】", "【M2】")]
        llm = FakeLLM({"【M1】": _terms_json("术语甲"), "【M2】": _terms_json("术语乙")})
        # batch_chars=500：两块各 1003 字符 → 切成 2 批，分别命中 M1/M2
        return await _run_volume(tmp_path, llm, specs, concurrency=3, batch_chars=500)

    stat, data, _ = asyncio.run(_main())
    assert stat["done"] == 1
    zh = {t["zh"] for t in data["sections"][0]["terms"]}
    assert zh == {"术语甲", "术语乙"}


def test_rate_limit_triggers_aimd_decrease(tmp_path):
    async def _main():
        specs = [_two_batch_section("【M1】", "【M2】")]
        rate_err = _err(openai.RateLimitError, code="AccountRateLimitExceeded", message="too frequent")

        def script(n):
            return rate_err if n == 1 else _terms_json("术语甲")

        llm = FakeLLM({"【M1】": script, "【M2】": _terms_json("术语乙")})
        return await _run_volume(tmp_path, llm, specs, concurrency=3, batch_chars=500)

    stat, data, ctx = asyncio.run(_main())
    assert ctx.limiter.limit == 1, "频控后并发应乘性减半至 1"
    assert ctx.stats.retry_calls >= 1
    assert stat["done"] == 1
    zh = {t["zh"] for t in data["sections"][0]["terms"]}
    assert "术语甲" in zh and "术语乙" in zh  # 重放后零丢失


def test_quota_gate_whole_team_waits_once():
    async def _main():
        slept = {"total": 0}

        async def fake_sleep(s):
            slept["total"] += s
            await asyncio.sleep(0)  # 让出事件循环，模拟真实等待的交错

        gate = et.QuotaGate(wait_seconds=120, sleep=fake_sleep)
        order = []

        async def triggerer(name):
            await gate.trigger_and_wait(name, RuntimeError("AccountQuotaExceeded"))
            order.append(name)

        await asyncio.gather(triggerer("A"), triggerer("B"))
        return slept, order, gate

    slept, order, gate = asyncio.run(_main())
    assert slept["total"] == 120, f"全队应只等一次 120s，实际 {slept['total']}"
    assert set(order) == {"A", "B"}

    async def _gate_ok():
        await gate.gate()  # 恢复后立即放行

    asyncio.run(_gate_ok())


def test_transient_timeout_budget_then_split(tmp_path):
    async def _main():
        specs = [("章一", "节一", ["【M1】" + "子块甲" * 100, "【M1】" + "子块乙" * 100])]
        timeout_err = _err(openai.APITimeoutError)

        def script(n):
            # 前 3 次（整批尝试）超时；拆半后的子任务成功
            return timeout_err if n <= 3 else _terms_json("拆分术语")

        llm = FakeLLM({"【M1】": script})
        return await _run_volume(tmp_path, llm, specs, concurrency=2, batch_chars=5000), llm

    (stat, data, ctx), llm = asyncio.run(_main())
    assert stat["done"] == 1
    assert llm.calls == 5, f"应为 3 次超时尝试 + 2 个拆半子任务，实际 {llm.calls}"
    assert any("拆分为 1+1" in m for m in _WARN_BUFFER)
    zh = {t["zh"] for t in data["sections"][0]["terms"]}
    assert zh == {"拆分术语"}


def test_sensitive_content_split_and_skip(tmp_path):
    async def _main():
        sens = _err(openai.BadRequestError, code="SensitiveContentDetected", status=400)
        results = {}

        # 场景1：一批 2 块触发审核 → 拆半，子任务成功
        tmp1 = tmp_path / "case1"
        tmp1.mkdir()
        specs1 = [("章一", "节一", ["【S1】" + "块甲" * 100, "【S1】" + "块乙" * 100])]
        llm1 = FakeLLM({"【S1】": lambda n: sens if n == 1 else _terms_json("安全术语")})
        results["case1"] = await _run_volume(tmp1, llm1, specs1, concurrency=2, batch_chars=5000)

        # 场景2：单块批触发审核 → 无法拆分，跳过该块，小节仍完成
        tmp2 = tmp_path / "case2"
        tmp2.mkdir()
        specs2 = [("章一", "节一", ["【S2】" + "独块" * 200])]
        llm2 = FakeLLM({"【S2】": sens})
        results["case2"] = await _run_volume(tmp2, llm2, specs2, concurrency=2, batch_chars=5000)
        return results

    results = asyncio.run(_main())
    stat1, data1, _ = results["case1"]
    assert stat1["done"] == 1
    assert data1["sections"][0]["skipped_blocks"] == 0
    assert {t["zh"] for t in data1["sections"][0]["terms"]} == {"安全术语"}

    stat2, data2, _ = results["case2"]
    assert stat2["done"] == 1
    assert data2["sections"][0]["skipped_blocks"] == 1
    assert data2["sections"][0]["terms"] == []


def test_empty_response_retries_then_split(tmp_path):
    async def _main():
        specs = [("章一", "节一", ["【E1】" + "块甲" * 200, "【E1】" + "块乙" * 200])]

        def script(n):
            # n=1,2 空响应重试；n=3 仍空 → 拆半；n=4,5 为拆半子任务，返回术语
            return "[]" if n <= 3 else _terms_json("补回术语")

        llm = FakeLLM({"【E1】": script})
        # batch_chars 很大 → 两块同批；3 次空响应（整批）后拆半为 2 个子任务
        return await _run_volume(tmp_path, llm, specs, concurrency=2, batch_chars=50000), llm

    (stat, data, ctx), llm = asyncio.run(_main())
    assert stat["done"] == 1
    assert llm.calls == 5, f"预期 3 次空响应 + 2 个拆半子任务，实际 {llm.calls}"
    assert {t["zh"] for t in data["sections"][0]["terms"]} == {"补回术语"}


def test_serial_and_concurrent_results_match(tmp_path):
    async def _main():
        specs = [
            _two_batch_section("【M1】", "【M2】"),
            ("章二", "节二", ["【M3】" + "丙" * 700]),
        ]
        behavior = {m: _terms_json(f"术语{m}") for m in ["【M1】", "【M2】", "【M3】"]}

        tmp_s = tmp_path / "serial"
        tmp_s.mkdir()
        stat_s, data_s, _ = await _run_volume(tmp_s, FakeLLM(behavior), specs, concurrency=1, batch_chars=500)
        tmp_c = tmp_path / "concurrent"
        tmp_c.mkdir()
        stat_c, data_c, _ = await _run_volume(tmp_c, FakeLLM(behavior), specs, concurrency=4, batch_chars=500)
        return stat_s, data_s, stat_c, data_c

    def zh_set(data):
        return sorted(t["zh"] for s in data["sections"] for t in s["terms"])

    stat_s, data_s, stat_c, data_c = asyncio.run(_main())
    assert zh_set(data_s) == zh_set(data_c)
    assert stat_s["done"] == stat_c["done"] == 2


def test_stop_flag_marks_sections_failed_and_persists_nothing(tmp_path):
    async def _main():
        specs = [
            _two_batch_section("【M1】", "【M2】"),
            ("章二", "节二", ["【M3】" + "丙" * 700]),
        ]
        llm = FakeLLM({m: _terms_json(f"术语{m}") for m in ["【M1】", "【M2】", "【M3】"]}, delay=0.05)
        return await _run_volume(tmp_path, llm, specs, concurrency=2, stop=True)

    stat, data, ctx = asyncio.run(_main())
    assert data["sections"] == []
    assert all(s.failed for s in ctx.states.values())


def test_resume_skips_done_sections(tmp_path):
    async def _main():
        specs = [
            ("章一", "节一", ["【M1】" + "甲" * 700]),
            ("章二", "节二", ["【M2】" + "乙" * 700]),
        ]
        prebuilt = {
            "schema_version": 1, "domain": "freud", "source_ref": {},
            "sections": [{
                "ref": {"book_title": "测试书", "volume_title": "卷T", "volume_index": 1,
                        "chapter_title": "章一", "section_title": "节一", "subsection_title": ""},
                "status": "done", "terms_cap": et._TERMS_CAP, "char_count": 999,
                "batch_count": 1, "skipped_blocks": 0,
                "terms": [{"zh": "已有术语", "en": "", "aliases_zh": [], "aliases_en": [], "definition": ""}],
            }],
            "stats": {},
        }
        llm = FakeLLM({"【M1】": _terms_json("不应被调用"), "【M2】": _terms_json("新术语")})
        return await _run_volume(tmp_path, llm, specs, concurrency=2,
                                 batch_chars=500, prebuilt_out=prebuilt), llm

    (stat, data, ctx), llm = asyncio.run(_main())
    assert stat["done"] == 2
    assert llm.calls == 1, "已完成节不应再调用 LLM"
    zh = {t["zh"] for s in data["sections"] for t in s["terms"]}
    assert zh == {"已有术语", "新术语"}
