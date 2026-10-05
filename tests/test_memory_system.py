"""Phase 4 记忆系统测试（M-01 ~ M-09 验收项）。

策略：与 RAG 全量索引任务共享 ARK 配额，主体测试用 Fake LLM/Embedding 覆盖
管线逻辑与存储行为（不耗配额、不互相挤压限流）；最后附真实 API 冒烟
（配置缺失自动跳过），验证 M-01/M-05/M-09 端到端。

运行：python -m pytest tests/test_memory_system.py -v
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path

import pytest

from app.config import Settings
from app.db.connection import init_db
from app.memory.episodic import EpisodicMemory, decay_score_at
from app.memory.manager import MemoryManager, _extract_json
from app.memory.profile import UserProfileStore, merge_profile_facts, default_profile
from app.memory.working import WorkingMemory


# ==================== 测试基础设施 ====================

def make_settings(tmp_path: Path) -> Settings:
    """隔离的临时存储配置（SQLite + Chroma 均指向临时目录）。"""
    s = Settings(
        sqlite_path=str(tmp_path / "test.db"),
        chroma_dir=str(tmp_path / "chroma"),
    )
    init_db(s)
    return s


class FakeEmbedding:
    """确定性词法向量：命中共享关键词则相似度高，用于验证检索/隔离逻辑。"""

    def __init__(self) -> None:
        self.calls = 0

    def _vec(self, text: str) -> list[float]:
        dims = ["情结", "梦境", "焦虑", "情绪", "作息", "睡觉", "哲学", "弗洛伊德", "荣格", "其他"]
        lowered = text
        return [1.0 if d in lowered else 0.0 for d in dims]

    async def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return self._vec(text)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


class FakeLLM:
    """可编程 LLM：按顺序返回预置回应（记录收到的 messages 供断言）。"""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def chat(self, messages, **kwargs) -> str:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if not self.responses:
            raise RuntimeError("FakeLLM 预置回应已耗尽")
        return self.responses.pop(0)


async def drain_tasks():
    """等待所有 pending 的 create_task 完成（MemoryWrite 异步提取）。"""
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


# ==================== WorkingMemory（M-03）====================

def test_working_memory_window_and_persist(tmp_path):
    """M-03：窗口淘汰 + SQLite 持久化 + 懒加载恢复 + 表不无界增长。"""
    s = make_settings(tmp_path)
    wm = WorkingMemory(window_size=3, settings=s)  # 3 轮 = 6 条消息
    uid = "test:wm_user"

    async def run():
        for i in range(10):  # 写 10 轮
            await wm.write(uid, "user", f"u{i}")
            await wm.write(uid, "assistant", f"a{i}")
        return await wm.read(uid)

    window = asyncio.run(run())
    # 只保留最近 3 轮（6 条）
    assert len(window) == 6
    assert window[0] == {"role": "user", "content": "u7"}
    assert window[-1] == {"role": "assistant", "content": "a9"}

    # SQLite 也只保留 6 条
    from app.db.connection import db_conn
    with db_conn(s) as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM working_memory WHERE user_id = ?", (uid,)
        ).fetchone()[0]
    assert n == 6

    # 新实例（模拟重启）懒加载恢复窗口
    wm2 = WorkingMemory(window_size=3, settings=s)
    window2 = asyncio.run(wm2.read(uid))
    assert len(window2) == 6
    assert window2[0]["content"] == "u7"

    # 清空
    asyncio.run(wm2.clear(uid))
    assert asyncio.run(wm2.read(uid)) == []


# ==================== decay 公式（M-04）====================

def test_decay_formula():
    """M-04：decay(t) = exp(-λ·Δt)；λ=0.005 时 14 天约衰减至 0.93（优先级降低）。"""
    now = int(time.time())
    d_now = decay_score_at(1.0, now, lam=0.005, now_ts=now)
    assert abs(d_now - 1.0) < 1e-9
    d_14d = decay_score_at(1.0, now - 14 * 86400, lam=0.005, now_ts=now)
    assert 0.90 < d_14d < 0.95  # exp(-0.07) ≈ 0.932，优先级"降低"
    d_140d = decay_score_at(1.0, now - 140 * 86400, lam=0.005, now_ts=now)
    assert 0.45 < d_140d < 0.55  # 半衰期约 139 天


# ==================== EpisodicMemory（M-02 / M-04 / M-07）====================

def test_episodic_isolation_and_search(tmp_path):
    """M-02 多用户隔离：A 的情景记忆 B 检索不到；同用户可命中。"""
    s = make_settings(tmp_path)
    em = EpisodicMemory(s, embedding=FakeEmbedding())

    async def run():
        await em.add(
            user_id="chat:alice",
            summary="用户讨论了荣格的情结理论，对梦境分析很感兴趣",
            topics=["荣格", "情结"],
        )
        await em.add(
            user_id="chat:bob",
            summary="用户聊到了哲学中的自由意志问题",
            topics=["哲学"],
        )
        hits_a = await em.search("情结理论怎么看", "chat:alice", k=3)
        hits_b = await em.search("情结理论怎么看", "chat:bob", k=3)
        return hits_a, hits_b

    hits_a, hits_b = asyncio.run(run())
    assert len(hits_a) == 1
    assert "情结" in hits_a[0]["summary"]
    assert hits_a[0]["relevance"] > 0.5
    assert hits_b == []  # B 检索不到 A 的记忆


def test_episodic_hit_resets_decay(tmp_path):
    """M-04：检索命中后 decay_score 重置为 1.0、last_hit_at 更新。"""
    s = make_settings(tmp_path)
    em = EpisodicMemory(s, embedding=FakeEmbedding())

    async def run():
        ep = await em.add(
            user_id="chat:alice",
            summary="用户讨论情结理论",
            timestamp=int(time.time()) - 100 * 86400,  # 100 天前
        )
        hits = await em.search("情结", "chat:alice", k=1)
        return ep, hits

    ep, hits = asyncio.run(run())
    assert len(hits) == 1
    # 100 天前的条目衰减后 relevance 明显降低
    assert hits[0]["relevance"] < 0.75
    # 命中后重置
    row = em._get_episode_row(ep.episode_id)
    assert row["decay_score"] == pytest.approx(1.0)
    assert row["last_hit_at"] == pytest.approx(int(time.time()), abs=60)


def test_episodic_cleanup_and_forget(tmp_path):
    """M-04/M-07：衰减 <0.1 周期清理；指定遗忘 + 敏感一键遗忘。"""
    s = make_settings(tmp_path)
    em = EpisodicMemory(s, embedding=FakeEmbedding())

    async def run():
        now = int(time.time())
        # 300 天前写入（decay ≈ exp(-0.005*300)=0.22 > 0.1 不清除）
        old = await em.add(user_id="chat:alice", summary="旧的情结讨论", timestamp=now - 300 * 86400)
        # 600 天前（decay ≈ 0.049 < 0.1 → 清除）
        stale = await em.add(user_id="chat:alice", summary="更旧的梦境讨论", timestamp=now - 600 * 86400)
        sensitive = await em.add(
            user_id="chat:alice", summary="用户提到近期情绪困扰", is_sensitive=True,
        )
        return old, stale, sensitive

    old, stale, sensitive = asyncio.run(run())

    removed = em.cleanup_decayed(threshold=0.1)
    assert removed >= 1
    assert em._get_episode_row(old.episode_id) is not None
    assert em._get_episode_row(stale.episode_id) is None

    # 指定遗忘
    em.forget_episode(old.episode_id)
    assert em._get_episode_row(old.episode_id) is None

    # 敏感一键遗忘
    n = em.forget_all_sensitive("chat:alice")
    assert n == 1
    assert em._get_episode_row(sensitive.episode_id) is None


# ==================== UserProfile merge（M-05 / M-06）====================

def test_profile_merge_rules():
    """M-05/M-06：sleep_pattern 更新、去重合并、theory upsert、敏感仅存标签。"""
    profile = default_profile("chat:alice")
    facts = {
        "basic_info": {"nickname": "漂泊者", "known_identity_tags": ["心理学爱好者", "心理学爱好者"]},
        "sleep_pattern": {"avg_bedtime": "23:30", "avg_wake_time": "07:15", "irregular_days": ["2026-08-30"]},
        "emotional_baseline": {"recurring_themes": ["梦境模糊"], "current_mood_hint": "略困惑"},
        "discussed_theories": [{"name": "情结（荣格）", "depth": "深入"}],
        "preferences": {"known_interests": ["荣格"], "communication_style_hint": "喜欢结合自身经历讨论理论"},
        "health_sensitive_topics": ["焦虑症状"],
    }
    merge_profile_facts(profile, facts)

    assert profile["basic_info"]["nickname"] == "漂泊者"
    assert profile["basic_info"]["known_identity_tags"] == ["心理学爱好者"]  # 去重
    assert profile["sleep_pattern"]["avg_bedtime"] == "23:30"
    assert profile["sleep_pattern"]["last_updated"] != ""
    assert profile["discussed_theories"][0]["name"] == "情结（荣格）"
    # 敏感仅存 topic 标签，无具体内容
    assert profile["health_sensitive"] == [
        {"topic": "焦虑症状", "first_mentioned": profile["health_sensitive"][0]["first_mentioned"]}
    ]

    # 二次 merge：theory 更新 discussed_at 而非重复条目；irregular_days 去重
    facts2 = {
        "sleep_pattern": {"irregular_days": ["2026-08-30", "2026-09-01"]},
        "discussed_theories": [{"name": "情结（荣格）", "depth": "深入"}],
        "health_sensitive_topics": ["焦虑症状"],
    }
    merge_profile_facts(profile, facts2)
    assert len(profile["discussed_theories"]) == 1
    assert profile["sleep_pattern"]["irregular_days"].count("2026-08-30") == 1
    assert len(profile["health_sensitive"]) == 1


def test_profile_store_persistence_and_forget_field(tmp_path):
    s = make_settings(tmp_path)
    store = UserProfileStore(s)
    store.merge_and_save("chat:alice", {"basic_info": {"nickname": "漂泊者"}})

    # 新实例读取
    store2 = UserProfileStore(s)
    p = store2.load("chat:alice")
    assert p["basic_info"]["nickname"] == "漂泊者"
    assert "sleep_pattern" in p  # 缺省字段自动补齐

    # 字段遗忘
    assert store2.forget_field("chat:alice", "basic_info.nickname") is True
    p2 = store2.load("chat:alice")
    assert p2["basic_info"]["nickname"] == ""
    assert store2.forget_field("chat:alice", "nonexistent.path") is False


# ==================== MemoryManager（M-01 / M-06 / M-08 / M-09）====================

GOOD_EXTRACT = json.dumps({
    "episode": {"summary": "用户讨论了荣格的情结理论并联系了自己的梦境", "topics": ["荣格", "情结", "梦境"], "emotion_tag": "好奇"},
    "profile_facts": {
        "discussed_theories": [{"name": "情结（荣格）", "depth": "中等"}],
        "preferences": {"known_interests": ["荣格"]},
    },
}, ensure_ascii=False)

SENSITIVE_EXTRACT = json.dumps({
    "episode": {"summary": "用户提到近期情绪困扰", "topics": ["情绪"], "emotion_tag": "低落"},
    "profile_facts": {"health_sensitive_topics": ["焦虑症状"]},
}, ensure_ascii=False)

NULL_EPISODE_EXTRACT = json.dumps({
    "episode": None,
    "profile_facts": {},
}, ensure_ascii=False)


def test_manager_write_read_roundtrip(tmp_path):
    """M-01/M-08/M-09：写 → 提取异步完成 → 读命中；工作记忆同步更新。"""
    s = make_settings(tmp_path)
    llm = FakeLLM([GOOD_EXTRACT])
    mgr = MemoryManager(llm, FakeEmbedding(), s)
    uid = "chat:alice"

    async def run():
        await mgr.write("情结理论能讲讲吗", "当然，情结是荣格理论的核心概念之一……", uid)
        # M-08：write 返回即完成同步部分（工作记忆已更新），异步提取并行进行
        ctx_immediate = await mgr.read("", uid)
        assert len(ctx_immediate.working_window) == 2
        assert ctx_immediate.episodic_hits == []  # 提取可能尚未完成
        await drain_tasks()  # 等待异步提取任务结束
        return await mgr.read("情结理论", uid)

    ctx = asyncio.run(run())
    assert len(ctx.episodic_hits) == 1
    assert "情结" in ctx.episodic_hits[0]["summary"]
    assert ctx.user_profile["discussed_theories"][0]["name"] == "情结（荣格）"
    # 提取只调用了一次 LLM（情景+画像合并提取，节省配额）
    assert len(llm.calls) == 1
    assert llm.calls[0]["kwargs"].get("json_mode") is True


def test_manager_sensitive_desensitization(tmp_path):
    """M-06：敏感对话 → 脱敏摘要入库（无原文细节）+ health_sensitive 仅标签 + 一键遗忘。"""
    s = make_settings(tmp_path)
    llm = FakeLLM([SENSITIVE_EXTRACT])
    em = EpisodicMemory(s, embedding=FakeEmbedding())
    mgr = MemoryManager(llm, FakeEmbedding(), s)
    # 注入同一个 em，便于检查遗忘结果
    mgr._episodic = em
    uid = "chat:alice"

    async def run():
        await mgr.write(
            "我最近焦虑得睡不着，总是心悸……",
            "听到你这样我很担心……如果持续的话，寻求专业帮助会更好。",
            uid, is_sensitive=True,
        )
        await drain_tasks()
        return await mgr.read("情绪", uid)

    ctx = asyncio.run(run())
    # 提取 prompt 收到 is_sensitive=true 标记
    sent = llm.calls[0]["messages"][1]["content"]
    assert "is_sensitive = true" in sent
    # 情景摘要是脱敏版本
    assert len(ctx.episodic_hits) == 1
    assert ctx.episodic_hits[0]["summary"] == "用户提到近期情绪困扰"
    # 画像仅存 topic 标签
    assert ctx.user_profile["health_sensitive"][0]["topic"] == "焦虑症状"
    assert "心悸" not in json.dumps(ctx.user_profile, ensure_ascii=False)

    # 一键遗忘敏感内容
    result = asyncio.run(mgr.forget(uid, "all_sensitive"))
    assert result["removed"] == 1
    ctx2 = asyncio.run(mgr.read("情绪", uid))
    assert ctx2.episodic_hits == []
    assert ctx2.user_profile["health_sensitive"] == []


def test_extract_json_tolerates_fences():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('{"b": [1,2]}') == {"b": [1, 2]}
    assert _extract_json("不是JSON") is None
    assert _extract_json("") is None


def test_forget_scope_validation(tmp_path):
    s = make_settings(tmp_path)
    mgr = MemoryManager(FakeLLM([]), FakeEmbedding(), s)
    with pytest.raises(ValueError):
        asyncio.run(mgr.forget("chat:alice", "episodic"))
    with pytest.raises(ValueError):
        asyncio.run(mgr.forget("chat:alice", "unknown_scope"))


# ==================== Orchestrator 集成（记忆注入 persona）====================

def test_persona_injects_memory_blocks():
    from app.core.persona import Persona

    persona = Persona()
    profile = default_profile("chat:alice")
    profile["basic_info"]["nickname"] = "漂泊者"
    profile["sleep_pattern"]["avg_bedtime"] = "23:30"
    profile["health_sensitive"].append({"topic": "焦虑症状", "first_mentioned": "2026-09-01"})
    hits = [{"summary": "用户之前讨论过荣格的情结理论", "topics": ["荣格"]}]

    prompt = persona.build_system_prompt(
        scene="daily", user_profile=profile, episodic_context=hits,
    )
    assert "【用户画像" in prompt
    assert "漂泊者" in prompt
    assert "23:30" in prompt
    assert "焦虑症状" in prompt
    assert "【既往对话片段" in prompt
    assert "情结理论" in prompt

    # 空记忆不注入
    prompt_empty = persona.build_system_prompt(scene="daily")
    assert "【用户画像" not in prompt_empty
    assert "【既往对话片段" not in prompt_empty


def test_orchestrator_uses_memory_manager(tmp_path):
    """编排器集成：MemoryRead 前置 + MemoryWrite 异步后置（用 fake 全链路，不走真实 LLM）。"""
    from unittest.mock import AsyncMock, patch

    from app.core.intent import IntentParser
    from app.core.models import UnifiedRequest
    from app.core.orchestrator import Orchestrator
    from app.core.persona import Persona

    s = make_settings(tmp_path)
    intent_json = '{"intent": "chat", "scene": "daily", "requires_knowledge": false, "domain_filter": null, "is_sensitive": false, "slots": {}}'
    main_reply = "你好呀……漂泊者今天也来了呢。……"
    llm = FakeLLM([intent_json, main_reply, GOOD_EXTRACT])
    mgr = MemoryManager(llm, FakeEmbedding(), s)
    persona = Persona()
    orchestrator = Orchestrator(
        llm_client=llm, persona=persona,
        intent_parser=IntentParser(llm), memory_manager=mgr,
    )

    async def run():
        # 先预置一条情景记忆，验证 MemoryRead 注入
        await mgr._episodic.add(user_id="local:t", summary="用户之前讨论过情结理论", topics=["情结"])
        req = UnifiedRequest(user_id="local:t", chat_id="local", open_id="t", text="你好呀")
        resp = await orchestrator.handle(req)
        await drain_tasks()
        return resp

    resp = asyncio.run(run())
    assert resp.text == main_reply
    # 主回复的 messages 中 system prompt 应含预置情景片段
    main_call = llm.calls[1]["messages"]  # 0=intent, 1=主回复
    assert "情结理论" in main_call[0]["content"]
    # 异步写入后工作记忆含本轮
    window = asyncio.run(mgr._working.read("local:t"))
    roles = [m["role"] for m in window]
    assert roles[-2:] == ["user", "assistant"]


# ==================== 真实 API 冒烟（M-01/M-05/M-09，配额不足时跳过）====================

def test_real_api_smoke(tmp_path):
    """真实 LLM 提取 + 真实 Embedding 检索的一轮闭环（ARK 未配置则跳过）。"""
    import os

    if not os.environ.get("ARK_KEY") or not os.environ.get("ARK_BASE_URL"):
        pytest.skip("ARK 环境变量未配置，跳过真实 API 冒烟")

    from app.llm.client import LLMClient
    from app.llm.embedding import EmbeddingClient

    s = make_settings(tmp_path)
    llm = LLMClient(s)
    emb = EmbeddingClient(s)
    mgr = MemoryManager(llm, emb, s)
    uid = f"smoke:{int(time.time())}"

    async def run():
        await mgr.write(
            "我最近总梦到大海，荣格说海是潜意识的象征吗？",
            "海在荣格的意象里确实常与潜意识相连……你的梦或许在提示一些沉在水下的内容。",
            uid,
        )
        # 等待异步提取（真实 API：提取 1 次调用 + 摘要 embedding 1 次；与 RAG 索引共享限流）
        for _ in range(90):
            await asyncio.sleep(1)
            ctx = await mgr.read("大海的梦", uid)
            if ctx.episodic_hits:
                return ctx
        return await mgr.read("大海的梦", uid)

    ctx = asyncio.run(run())
    assert ctx.episodic_hits, "情景记忆未命中（真实 API 提取或向量化失败）"
    assert "梦" in ctx.episodic_hits[0]["summary"] or "海" in ctx.episodic_hits[0]["summary"]
    print(f"[smoke] episodic hit: {ctx.episodic_hits[0]}")
