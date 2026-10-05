"""Phase 5 置信度与对话策略整合测试（零 API 配额）。

覆盖（对齐 PRD A-03/A-04/A-05/A-07/A-08 与 IMPLEMENTATION_DESIGN Phase 5）：
- 5.1 confidence.assess：量化映射高/中/低（含 domain_filter 降级、低分/无命中）；
- 5.2 编排器接入：qa/discussion 走 RAG + 置信度链路，chat 不检索，RAG 失败降级低置信度；
- 5.3 策略块：讨论（盲点/追问）、问答（越界/不确定）注入 System Prompt；
- 5.4 敏感话题块：谨慎回应 + 专业帮助指引；
- 5.5 指令执行：提醒（登记 pending reminders）、记录、查询转知识链路、模糊追问、
  不可执行礼貌说明。

全部使用 FakeLLM / FakeEmbedding / FakeRAG 与临时 SQLite/Chroma，不调用真实 API。
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

import pytest

from app.config import Settings
from app.core.command import CommandExecutor, list_pending_reminders, parse_reminder_time
from app.core.confidence import assess
from app.core.models import (
    ConfidenceLevel,
    ConfidenceSignals,
    RAGCandidate,
    RAGResult,
    UnifiedRequest,
)
from app.db.connection import init_db

_GOOD_EXTRACT = json.dumps({
    "episode": {"summary": "用户讨论了强迫性重复", "topics": ["弗洛伊德"], "emotion_tag": "好奇"},
    "profile_facts": {},
}, ensure_ascii=False)


# ==================== 测试基础设施 ====================

def make_settings(tmp_path: Path) -> Settings:
    s = Settings(
        sqlite_path=str(tmp_path / "test.db"),
        chroma_dir=str(tmp_path / "chroma"),
    )
    init_db(s)
    return s


class FakeLLM:
    """按序返回预置回应（记录调用供断言）。"""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def chat(self, messages, **kwargs) -> str:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if not self.responses:
            raise RuntimeError("FakeLLM 预置回应已耗尽")
        return self.responses.pop(0)


class FakeEmbedding:
    async def embed_query(self, text: str) -> list[float]:
        return [0.0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


class FakeRAG:
    """可编程 RAG 检索桩：记录调用、返回预置结果或抛异常。"""

    def __init__(self, result: RAGResult | None = None, raise_exc: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self.result = result
        self.raise_exc = raise_exc

    async def retrieve(self, query, context=None, domain_filter=None, enable_image=False):
        self.calls.append({"query": query, "domain_filter": domain_filter})
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.result or RAGResult()


async def drain_tasks():
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


def _signals(**kw) -> ConfidenceSignals:
    return ConfidenceSignals(**kw)


# ==================== 5.1 置信度映射（A-07）====================

def test_assess_low_no_hits():
    r = assess(_signals(total_hits=0, max_rerank_score=0.0))
    assert r.level == ConfidenceLevel.LOW
    assert "可能" in r.wording_hint


def test_assess_low_score_below_threshold():
    r = assess(_signals(total_hits=5, max_rerank_score=0.55, source_diversity=2,
                        section_coverage=2, coverage_estimate="high"))
    assert r.level == ConfidenceLevel.LOW


def test_assess_medium_single_source():
    r = assess(_signals(total_hits=2, max_rerank_score=0.9, source_diversity=1,
                        section_coverage=1, coverage_estimate="medium"))
    assert r.level == ConfidenceLevel.MEDIUM


def test_assess_high_full_conditions():
    r = assess(_signals(total_hits=3, max_rerank_score=0.9, source_diversity=2,
                        section_coverage=2, coverage_estimate="high"))
    assert r.level == ConfidenceLevel.HIGH
    assert "绝对化" in r.wording_hint or "边界" in r.wording_hint


def test_assess_high_with_domain_filter_diversity_downgraded():
    """domain_filter 启用时 source_diversity 条件降级（PRD §5.2）。"""
    r = assess(_signals(total_hits=3, max_rerank_score=0.9, source_diversity=1,
                        section_coverage=2, coverage_estimate="high",
                        domain_filter_applied=["freud"]))
    assert r.level == ConfidenceLevel.HIGH


def test_assess_medium_hits_but_weak_coverage():
    r = assess(_signals(total_hits=4, max_rerank_score=0.9, source_diversity=2,
                        section_coverage=1, coverage_estimate="medium"))
    assert r.level == ConfidenceLevel.MEDIUM


# ==================== 知识参考块 / 置信度块渲染 ====================

def _make_rag_result() -> RAGResult:
    cand = RAGCandidate(
        block_id="b1", block_type="text",
        text="强迫性重复是指个体不断重演早期创伤经验的心理现象，弗洛伊德在《超越快乐原则》中提出。",
        source={"document_title": "弗洛伊德文集", "chapter_title": "超越快乐原则",
                "section_title": "第一节 强迫性重复", "domain": "freud"},
        rerank_score=0.88, retrieval_source="vector",
    )
    signals = _signals(total_hits=2, max_rerank_score=0.88, source_diversity=1,
                       section_coverage=1, coverage_estimate="medium")
    return RAGResult(candidates=[cand], confidence_signals=signals)


def test_knowledge_block_contains_source_and_text():
    from app.core.strategy import build_knowledge_block
    block = build_knowledge_block(_make_rag_result())
    assert "知识参考" in block
    assert "弗洛伊德文集" in block
    assert "超越快乐原则" in block
    assert "强迫性重复" in block


def test_knowledge_block_empty_returns_empty():
    from app.core.strategy import build_knowledge_block
    assert build_knowledge_block(None) == ""
    assert build_knowledge_block(RAGResult()) == ""


def test_confidence_block_per_level():
    from app.core.strategy import build_confidence_block
    from app.core.models import ConfidenceResult
    low = build_confidence_block(ConfidenceResult(level=ConfidenceLevel.LOW, wording_hint="低"))
    assert "低置信度" in low and "不确定" in low
    high = build_confidence_block(ConfidenceResult(level=ConfidenceLevel.HIGH, wording_hint="高"))
    assert "高置信度" in high and ("边界" in high or "绝对化" in high)
    assert build_confidence_block(None) == ""


# ==================== 5.3 / 5.4 策略块 ====================

def test_discussion_and_qa_and_sensitive_blocks():
    from app.core.strategy import build_discussion_block, build_qa_block, build_sensitive_block
    d = build_discussion_block()
    assert "盲点" in d and "追问" in d or "延伸" in d
    q = build_qa_block()
    assert "不确定" in q
    s = build_sensitive_block()
    assert "专业帮助" in s


# ==================== 5.5 提醒时间解析 ====================

def test_parse_reminder_time_variants():
    now = datetime(2026, 9, 5, 10, 0)
    t = parse_reminder_time("晚上10点", now)
    assert t is not None and t.hour == 22 and t.minute == 0
    t2 = parse_reminder_time("明天早上7点半", now)
    assert t2 is not None and (t2.year, t2.month, t2.day) == (2026, 9, 6) and t2.hour == 7 and t2.minute == 30
    t3 = parse_reminder_time("下午3点", now)
    assert t3 is not None and t3.hour == 15
    t4 = parse_reminder_time("18点", now)
    assert t4 is not None and t4.hour == 18
    # 已过时刻 → 下一天同时刻
    t5 = parse_reminder_time("8点", now)
    assert t5 is not None and t5.day == 6 and t5.hour == 8
    assert parse_reminder_time("吃药", now) is None


# ==================== 5.5 CommandExecutor ====================

def test_command_reminder_executed_and_persisted(tmp_path):
    s = make_settings(tmp_path)
    executor = CommandExecutor(s)
    outcome = executor.handle("提醒我晚上10点吃药", {"reminder_time": "10点"}, "local:t")
    assert outcome.status == "executed" and outcome.action == "reminder"
    assert "吃药" in outcome.detail
    rows = list_pending_reminders(s)
    assert len(rows) == 1
    assert rows[0]["content"] == "吃药"
    assert rows[0]["status"] == "pending"
    assert rows[0]["remind_at"] and "22:" in rows[0]["remind_at"]


def test_command_reminder_vague_asks_clarification(tmp_path):
    s = make_settings(tmp_path)
    executor = CommandExecutor(s)
    outcome = executor.handle("提醒我", {}, "local:t")
    assert outcome.status == "clarify" and outcome.action == "reminder"
    assert any("提醒时间" in m for m in outcome.missing)
    assert any("提醒内容" in m for m in outcome.missing)
    assert list_pending_reminders(s) == []


def test_command_record_executed(tmp_path):
    s = make_settings(tmp_path)
    executor = CommandExecutor(s)
    outcome = executor.handle("记录一下我喜欢夜跑", {}, "local:t")
    assert outcome.status == "executed" and outcome.action == "record"
    assert "夜跑" in outcome.detail


def test_command_query_routes_to_knowledge(tmp_path):
    s = make_settings(tmp_path)
    executor = CommandExecutor(s)
    outcome = executor.handle("帮我查一下什么是情结", {}, "local:t")
    assert outcome.action == "query" and outcome.is_query


def test_command_infeasible_boundary(tmp_path):
    s = make_settings(tmp_path)
    executor = CommandExecutor(s)
    outcome = executor.handle("帮我订一张明天去上海的机票", {}, "local:t")
    assert outcome.status == "infeasible"


# ==================== 5.2 编排器接入（A-03/A-04/A-07/A-08/A-05）====================

def _make_orchestrator(tmp_path, llm: FakeLLM, rag: FakeRAG | None):
    from app.core.command import CommandExecutor
    from app.core.intent import IntentParser
    from app.core.orchestrator import Orchestrator
    from app.core.persona import Persona
    from app.memory.manager import MemoryManager

    s = make_settings(tmp_path)
    mgr = MemoryManager(llm, FakeEmbedding(), s)
    return Orchestrator(
        llm_client=llm,
        persona=Persona(),
        intent_parser=IntentParser(llm),
        memory_manager=mgr,
        rag_retriever=rag,
        command_executor=CommandExecutor(s),
    ), s


def test_orchestrator_qa_uses_rag_and_confidence(tmp_path):
    """A-04/A-07：qa 走 RAG → 置信度映射 → 知识参考 + 措辞要求 + 问答策略注入。"""
    qa_intent = json.dumps({
        "intent": "qa", "scene": "academic", "requires_knowledge": True,
        "domain_filter": ["freud"], "is_sensitive": False, "slots": {},
    }, ensure_ascii=False)
    llm = FakeLLM([qa_intent, "或许强迫性重复正如弗洛伊德所言……", _GOOD_EXTRACT])
    rag = FakeRAG(result=_make_rag_result())
    orch, _s = _make_orchestrator(tmp_path, llm, rag)

    async def run():
        req = UnifiedRequest(user_id="local:t", chat_id="local", open_id="t", text="什么是强迫性重复？")
        resp = await orch.handle(req)
        await drain_tasks()
        return resp

    resp = asyncio.run(run())
    assert resp.text == "或许强迫性重复正如弗洛伊德所言……"
    # RAG 被调用且 domain_filter 透传
    assert len(rag.calls) == 1
    assert rag.calls[0]["domain_filter"] == ["freud"]
    # System Prompt（calls[1]）注入知识参考 + 置信度 + 问答策略
    system = llm.calls[1]["messages"][0]["content"]
    assert "知识参考" in system and "强迫性重复" in system
    assert "置信度措辞要求" in system
    assert "知识问答策略" in system


def test_orchestrator_chat_skips_rag(tmp_path):
    """闲聊不走知识链路：零检索、无知识/置信度块。"""
    chat_intent = json.dumps({
        "intent": "chat", "scene": "daily", "requires_knowledge": False,
        "domain_filter": None, "is_sensitive": False, "slots": {},
    }, ensure_ascii=False)
    llm = FakeLLM([chat_intent, "晚上好呀……漂泊者。", _GOOD_EXTRACT])
    rag = FakeRAG()
    orch, _s = _make_orchestrator(tmp_path, llm, rag)

    async def run():
        req = UnifiedRequest(user_id="local:t", chat_id="local", open_id="t", text="晚上好呀")
        await orch.handle(req)
        await drain_tasks()

    asyncio.run(run())
    assert rag.calls == []
    system = llm.calls[1]["messages"][0]["content"]
    assert "知识参考" not in system
    assert "置信度措辞要求" not in system


def test_orchestrator_sensitive_injects_block(tmp_path):
    """A-08：敏感话题注入谨慎回应 + 专业帮助块（不走 RAG）。"""
    llm = FakeLLM(["真的撑不下去了……", _GOOD_EXTRACT])
    rag = FakeRAG()
    orch, _s = _make_orchestrator(tmp_path, llm, rag)

    async def run():
        req = UnifiedRequest(user_id="local:t", chat_id="local", open_id="t", text="我不想活了")
        await orch.handle(req)
        await drain_tasks()

    asyncio.run(run())
    assert rag.calls == []  # 敏感话题关键词路径 requires_knowledge=False
    system = llm.calls[0]["messages"][0]["content"]  # 关键词快速路径：calls[0] 即主生成
    assert "敏感话题处理" in system
    assert "专业帮助" in system


def test_orchestrator_discussion_strategy_injected(tmp_path):
    """A-03：观点讨论注入讨论策略块（盲点/追问延伸）。"""
    disc_intent = json.dumps({
        "intent": "discussion", "scene": "sadness", "requires_knowledge": True,
        "domain_filter": ["freud"], "is_sensitive": False, "slots": {},
    }, ensure_ascii=False)
    llm = FakeLLM([disc_intent, "或许这正是压抑在起作用……你觉得呢？", _GOOD_EXTRACT])
    rag = FakeRAG(result=_make_rag_result())
    orch, _s = _make_orchestrator(tmp_path, llm, rag)

    async def run():
        req = UnifiedRequest(
            user_id="local:t", chat_id="local", open_id="t",
            text="我觉得有氧拳击的意义被某部分拒绝承认，你怎么看？",
        )
        await orch.handle(req)
        await drain_tasks()

    asyncio.run(run())
    system = llm.calls[1]["messages"][0]["content"]
    assert "观点讨论策略" in system
    assert "盲点" in system
    assert "知识参考" in system


def test_orchestrator_rag_failure_degrades_low_confidence(tmp_path):
    """RAG 异常 → 降级空信号 → 低置信度措辞块，不阻塞对话。"""
    qa_intent = json.dumps({
        "intent": "qa", "scene": "academic", "requires_knowledge": True,
        "domain_filter": None, "is_sensitive": False, "slots": {},
    }, ensure_ascii=False)
    llm = FakeLLM([qa_intent, "这个我了解得还不深……或许我们可以一起想想。", _GOOD_EXTRACT])
    rag = FakeRAG(raise_exc=RuntimeError("embedding quota exceeded"))
    orch, _s = _make_orchestrator(tmp_path, llm, rag)

    async def run():
        req = UnifiedRequest(user_id="local:t", chat_id="local", open_id="t", text="什么是集体潜意识？")
        resp = await orch.handle(req)
        await drain_tasks()
        return resp

    resp = asyncio.run(run())
    assert "了解得还不深" in resp.text
    system = llm.calls[1]["messages"][0]["content"]
    assert "置信度措辞要求" in system
    assert "低置信度" in system
    assert "【知识参考" not in system  # 无候选不注入知识块（问答策略文案不含块头）


def test_orchestrator_command_reminder_end_to_end(tmp_path):
    """A-05：提醒指令（关键词快速路径）→ CommandExecutor 登记 → 确认指引注入。"""
    llm = FakeLLM(["好的……今晚十点我会轻轻叫你的。到时记得吃药哦。", _GOOD_EXTRACT])
    orch, s = _make_orchestrator(tmp_path, llm, FakeRAG())

    async def run():
        req = UnifiedRequest(
            user_id="local:t", chat_id="local", open_id="t", text="提醒我晚上10点吃药",
        )
        resp = await orch.handle(req)
        await drain_tasks()
        return resp

    resp = asyncio.run(run())
    assert "吃药" in resp.text or "叫你" in resp.text
    rows = list_pending_reminders(s)
    assert len(rows) == 1 and rows[0]["content"] == "吃药"
    system = llm.calls[0]["messages"][0]["content"]  # 快速路径：无意图 LLM 调用
    assert "指令执行指引" in system
    assert "22:" in system  # 提醒时间已解析注入


def test_orchestrator_command_infeasible_polite_boundary(tmp_path):
    """A-05：不可执行指令 → 能力边界指引注入，礼貌说明不假装执行。"""
    llm = FakeLLM(["这个……我真的做不到呢，订票需要在现实里操作……不过我可以陪你查查攻略。"])
    orch, _s = _make_orchestrator(tmp_path, llm, FakeRAG())

    async def run():
        req = UnifiedRequest(
            user_id="local:t", chat_id="local", open_id="t", text="帮我订一张明天去上海的机票",
        )
        resp = await orch.handle(req)
        await drain_tasks()
        return resp

    resp = asyncio.run(run())
    assert "做不到" in resp.text
    system = llm.calls[0]["messages"][0]["content"]
    assert "能力边界指引" in system


def test_orchestrator_command_query_goes_rag(tmp_path):
    """A-05：查询类指令（帮我查）转 RAG 知识链路 + 置信度。"""
    llm = FakeLLM(["或许可以说，情结是荣格理论中围绕某一主题的情绪-观念群……", _GOOD_EXTRACT])
    rag = FakeRAG(result=_make_rag_result())
    orch, _s = _make_orchestrator(tmp_path, llm, rag)

    async def run():
        req = UnifiedRequest(
            user_id="local:t", chat_id="local", open_id="t", text="帮我查一下什么是情结",
        )
        await orch.handle(req)
        await drain_tasks()

    asyncio.run(run())
    assert len(rag.calls) == 1
    system = llm.calls[0]["messages"][0]["content"]
    assert "知识参考" in system
    assert "置信度措辞要求" in system
    assert "指令执行指引" in system  # 查询指引 + 知识链路并存


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
