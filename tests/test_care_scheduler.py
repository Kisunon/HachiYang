"""Phase 6 关怀调度测试（零 API 配额）。

覆盖（对齐 PRD §6 / A-06 与 IMPLEMENTATION_DESIGN Phase 6）：
- 6.1 trigger：早晚安起居识别（含第二人称祈使/单字"早"误伤排除）、
  sleep_pattern 更新（EMA 跨午夜平均、irregular_days、last_state、旧画像补字段）；
- 6.2 scheduler：就寝/起床窗口判定、已道早晚安不重复提醒、活跃判定、
  单日上限、失败可重试、dry-run 不消费提醒、到期提醒投递、多用户隔离；
- 6.3 composer：模板保底、同模板不连续复用、LLM 增强与失败降级、提醒文案；
- 6.4 推送：sender 抽象（sent / failed / dry_run）+ care_events 落库；
- 编排器集成：晚安/早安消息触发起居指引注入与画像记录。

全部使用 FakeLLM / FakeSender 与临时 SQLite，不调用真实 API。
运行：python -m pytest tests/test_care_scheduler.py -v
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.care.composer import CareComposer
from app.care.scheduler import CareScheduler, in_window, parse_window
from app.care.trigger import CareTrigger, SleepEvent, detect_sleep_event
from app.config import Settings
from app.core.command import save_reminder
from app.core.models import UnifiedRequest
from app.db.connection import db_conn, init_db

TZ = timezone(timedelta(hours=8))


def _dt(y: int, m: int, d: int, h: int, mi: int = 0) -> datetime:
    return datetime(y, m, d, h, mi, tzinfo=TZ)


# ==================== 测试基础设施 ====================

def make_settings(tmp_path: Path, **overrides) -> Settings:
    overrides.setdefault("care_llm_compose", False)  # 默认零配额；LLM 路径用例单独开启
    s = Settings(
        sqlite_path=str(tmp_path / "test.db"),
        chroma_dir=str(tmp_path / "chroma"),
        **overrides,
    )
    init_db(s)
    return s


class FakeLLM:
    def __init__(self, response: str = "晚安，漂泊者，好梦。", raise_exc: Exception | None = None):
        self.response = response
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    async def chat(self, messages, **kwargs) -> str:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.response


class FakeSender:
    """推送桩：记录调用，可编排返回值/异常。"""

    def __init__(self, result: bool | Exception = True):
        self.sent: list[tuple[str, str]] = []
        self.result = result

    async def __call__(self, chat_id: str, text: str) -> bool:
        self.sent.append((chat_id, text))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class SmartLLM:
    """编排器测试用 LLM 桩：主生成（非 json_mode）→ reply；记忆提取 → extract；
    意图解析（json_mode，关键词快速路径未命中时）→ intent JSON。"""

    def __init__(self, reply: str, intent_json: str, extract_json: str):
        self.reply = reply
        self.intent_json = intent_json
        self.extract_json = extract_json
        self.calls: list[dict] = []

    async def chat(self, messages, **kwargs) -> str:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if not kwargs.get("json_mode"):
            return self.reply
        blob = json.dumps(messages, ensure_ascii=False)
        if "episode" in blob or "画像" in blob or "情景" in blob:
            return self.extract_json
        return self.intent_json


def upsert_session(s: Settings, user_id: str, chat_id: str, at: datetime) -> None:
    epoch = int(at.timestamp())
    with db_conn(s) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO chat_sessions (user_id, chat_id, open_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, chat_id, "ou_test", epoch, epoch),
        )


def make_scheduler(s: Settings, sender=None, llm=None) -> CareScheduler:
    return CareScheduler(
        s, composer=CareComposer(s, llm=llm), sender=sender,
    )


def care_events(s: Settings) -> list[dict]:
    with db_conn(s) as conn:
        rows = conn.execute("SELECT * FROM care_events ORDER BY id").fetchall()
    return [dict(r) for r in rows]


async def drain_tasks():
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


# ==================== 6.1 起居识别（纯函数）====================

def test_detect_sleep_phrases():
    for text in ("晚安", "我睡了", "准备睡了", "我先睡啦", "去睡觉了，明天聊"):
        evt = detect_sleep_event(text, now=_dt(2026, 9, 6, 23, 30))
        assert evt is not None and evt.kind == "sleep", text


def test_detect_wake_phrases():
    for text in ("早安", "早上好呀", "起床了", "我醒了", "早，今天开会"):
        evt = detect_sleep_event(text, now=_dt(2026, 9, 6, 7, 30))
        assert evt is not None and evt.kind == "wake", text


def test_detect_non_events():
    # 晚上好是问候；早点睡/你也休息是第二人称祈使；单字"早"需搭配语气/标点
    for text in ("晚上好", "早点睡吧", "你也早点休息", "太早了", "早知道就不去了", "今天天气不错"):
        assert detect_sleep_event(text) is None, text


# ==================== 6.1 CareTrigger 画像更新 ====================

def test_trigger_records_sleep(tmp_path):
    s = make_settings(tmp_path)
    trigger = CareTrigger(s)
    sp = asyncio.run(trigger.record("oc_1:ou_1", SleepEvent("sleep", _dt(2026, 9, 6, 23, 30))))
    assert sp["avg_bedtime"] == "23:30"
    assert sp["last_state"] == "asleep"
    assert sp["last_state_date"] == "2026-09-06"
    assert sp["last_event_at"] == "23:30"
    # 落库可回读
    again = trigger.profile_store.load("oc_1:ou_1")["sleep_pattern"]
    assert again["last_state"] == "asleep"


def test_trigger_ema_bedtime_across_midnight(tmp_path):
    s = make_settings(tmp_path)
    trigger = CareTrigger(s)
    asyncio.run(trigger.record("u", SleepEvent("sleep", _dt(2026, 9, 5, 23, 30))))
    sp = asyncio.run(trigger.record("u", SleepEvent("sleep", _dt(2026, 9, 7, 0, 20))))
    # 轴（origin=18）：23:30=330，00:20=380 → EMA 0.6*330+0.4*380=350 → 23:50
    assert sp["avg_bedtime"] == "23:50"


def test_trigger_irregular_late_bedtime(tmp_path):
    s = make_settings(tmp_path)
    trigger = CareTrigger(s)
    sp = asyncio.run(trigger.record("u", SleepEvent("sleep", _dt(2026, 9, 6, 1, 30))))
    assert sp["last_state"] == "asleep"
    assert "2026-09-06" in sp["irregular_days"]


def test_trigger_wake_updates_and_irregular(tmp_path):
    s = make_settings(tmp_path)
    trigger = CareTrigger(s)
    sp = asyncio.run(trigger.record("u", SleepEvent("wake", _dt(2026, 9, 6, 7, 0))))
    assert sp["avg_wake_time"] == "07:00" and sp["last_state"] == "awake"
    # 04:30 起床早于合理区间（轴 origin=3：04:30=90 < 120）→ irregular
    sp2 = asyncio.run(trigger.record("u", SleepEvent("wake", _dt(2026, 9, 7, 4, 30))))
    assert "2026-09-07" in sp2["irregular_days"]


def test_trigger_old_profile_fills_defaults(tmp_path):
    """旧画像 sleep_pattern 缺 last_state 等字段时自动补齐，不报错。"""
    s = make_settings(tmp_path)
    store = CareTrigger(s).profile_store
    store.save("u", {"sleep_pattern": {"avg_bedtime": "23:00"}})  # 旧结构
    trigger = CareTrigger(s, profile_store=store)
    sp = asyncio.run(trigger.record("u", SleepEvent("wake", _dt(2026, 9, 6, 8, 0))))
    assert sp["last_state"] == "awake" and sp["avg_wake_time"] == "08:00"
    assert sp["avg_bedtime"] == "23:00"  # 既有数据保留


# ==================== 6.3 Composer 文案 ====================

def test_composer_template_fallback(tmp_path):
    s = make_settings(tmp_path)
    composer = CareComposer(s, llm=None)
    c = asyncio.run(composer.compose("bedtime", {}))
    assert c.source == "template" and c.template_id.startswith("bedtime-")
    assert ("漂泊者" in c.text or "哈基星" in c.text)


def test_composer_avoids_last_template(tmp_path):
    s = make_settings(tmp_path)
    composer = CareComposer(s, llm=None)
    c1 = asyncio.run(composer.compose("wake", {}))
    c2 = asyncio.run(composer.compose("wake", {}, avoid_template_id=c1.template_id))
    assert c2.template_id != c1.template_id


def test_composer_llm_enhancement(tmp_path):
    s = make_settings(tmp_path, care_llm_compose=True)
    llm = FakeLLM("夜深了，愿风替你盖好被子。")
    composer = CareComposer(s, llm=llm)
    c = asyncio.run(composer.compose("bedtime", {"sleep_pattern": {"avg_bedtime": "23:30"}}))
    assert c.source == "llm" and c.text == "夜深了，愿风替你盖好被子。"
    assert c.template_id.startswith("bedtime-")  # 基底模板仍记录
    assert llm.calls and "23:30" in llm.calls[0]["messages"][1]["content"]


def test_composer_llm_failure_falls_back(tmp_path):
    s = make_settings(tmp_path, care_llm_compose=True)
    composer = CareComposer(s, llm=FakeLLM(raise_exc=RuntimeError("429 quota")))
    c = asyncio.run(composer.compose("wake", {}))
    assert c.source == "template" and c.text


def test_compose_reminder_contains_content(tmp_path):
    s = make_settings(tmp_path)
    c = CareComposer(s).compose_reminder("晚上十点吃药")
    assert "晚上十点吃药" in c.text and c.template_id.startswith("reminder-")


# ==================== 窗口工具 ====================

def test_window_parse_and_in_window():
    start, end = parse_window("22:30-23:59")
    assert (start, end) == (1350, 1439)
    assert in_window(23 * 60, start, end) and not in_window(22 * 60, start, end)
    # 跨午夜窗口
    s2, e2 = parse_window("23:00-01:00")
    assert in_window(30, s2, e2) and in_window(23 * 60 + 30, s2, e2)
    assert not in_window(12 * 60, s2, e2)


# ==================== 6.2/6.4 Scheduler 主动关怀 ====================

def test_bedtime_tick_pushes_once(tmp_path):
    s = make_settings(tmp_path)
    sender = FakeSender()
    sch = make_scheduler(s, sender=sender)
    now = _dt(2026, 9, 6, 23, 0)
    upsert_session(s, "oc_1:ou_1", "oc_1", now)

    recs = asyncio.run(sch.tick(now))
    assert len(recs) == 1 and recs[0]["care_type"] == "bedtime"
    assert len(sender.sent) == 1 and sender.sent[0][0] == "oc_1"
    ev = care_events(s)
    assert len(ev) == 1 and ev[0]["status"] == "sent"

    # 同一窗口重复扫描不重推
    recs2 = asyncio.run(sch.tick(now + timedelta(minutes=10)))
    assert recs2 == [] and len(sender.sent) == 1


def test_bedtime_skipped_when_said_goodnight(tmp_path):
    s = make_settings(tmp_path)
    trigger = CareTrigger(s)
    asyncio.run(trigger.record("oc_1:ou_1", SleepEvent("sleep", _dt(2026, 9, 6, 22, 45))))
    sender = FakeSender()
    sch = make_scheduler(s, sender=sender)
    now = _dt(2026, 9, 6, 23, 0)
    upsert_session(s, "oc_1:ou_1", "oc_1", now)

    assert asyncio.run(sch.tick(now)) == []
    assert sender.sent == []


def test_bedtime_skipped_when_inactive_today(tmp_path):
    s = make_settings(tmp_path)
    sender = FakeSender()
    sch = make_scheduler(s, sender=sender)
    now = _dt(2026, 9, 6, 23, 0)
    # 最近互动在昨天中午（就寝关怀要求今天有互动）
    upsert_session(s, "oc_1:ou_1", "oc_1", _dt(2026, 9, 5, 12, 0))
    assert asyncio.run(sch.tick(now)) == []
    assert sender.sent == []


def test_wake_tick_pushes_and_skips_after_greeting(tmp_path):
    s = make_settings(tmp_path)
    sender = FakeSender()
    sch = make_scheduler(s, sender=sender)
    now = _dt(2026, 9, 6, 7, 30)
    # 起床问候放宽到近两天互动（昨天晚上聊过）
    upsert_session(s, "oc_1:ou_1", "oc_1", _dt(2026, 9, 5, 22, 0))

    recs = asyncio.run(sch.tick(now))
    assert len(recs) == 1 and recs[0]["care_type"] == "wake"
    assert len(sender.sent) == 1

    # 用户随后道早安 → 同窗口再扫描不重推
    asyncio.run(CareTrigger(s).record("oc_1:ou_1", SleepEvent("wake", _dt(2026, 9, 6, 7, 45))))
    assert asyncio.run(sch.tick(now + timedelta(minutes=20))) == []
    assert len(sender.sent) == 1


def test_outside_window_no_push(tmp_path):
    s = make_settings(tmp_path)
    sender = FakeSender()
    sch = make_scheduler(s, sender=sender)
    now = _dt(2026, 9, 6, 12, 0)
    upsert_session(s, "oc_1:ou_1", "oc_1", now)
    assert asyncio.run(sch.tick(now)) == []
    assert sender.sent == []


def test_daily_cap_blocks_extra_care(tmp_path):
    s = make_settings(tmp_path, care_daily_max=1)
    sender = FakeSender()
    sch = make_scheduler(s, sender=sender)
    now = _dt(2026, 9, 6, 23, 0)
    upsert_session(s, "oc_1:ou_1", "oc_1", now)
    # 今日已成功推送过 1 次起床关怀（达上限）
    with db_conn(s) as conn:
        conn.execute(
            "INSERT INTO care_events (user_id, care_type, template_id, content, status, created_at) "
            "VALUES (?, 'wake', 'wake-1', 'x', 'sent', ?)",
            ("oc_1:ou_1", int(_dt(2026, 9, 6, 7, 0).timestamp())),
        )
    assert asyncio.run(sch.tick(now)) == []
    assert sender.sent == []


def test_failed_push_is_retriable(tmp_path):
    s = make_settings(tmp_path)
    sender = FakeSender(result=False)
    sch = make_scheduler(s, sender=sender)
    now = _dt(2026, 9, 6, 23, 0)
    upsert_session(s, "oc_1:ou_1", "oc_1", now)

    recs = asyncio.run(sch.tick(now))
    assert recs[0]["status"] == "failed"
    assert len(sender.sent) == 1  # 第一次尝试已调用推送（返回失败）
    # failed 不计入今日成功数 → 渠道恢复后下轮可重推
    sender.result = True
    recs2 = asyncio.run(sch.tick(now + timedelta(minutes=5)))
    assert len(recs2) == 1 and recs2[0]["status"] == "sent"
    assert len(sender.sent) == 2  # 失败重试 + 成功送达各一次
    statuses = [e["status"] for e in care_events(s)]
    assert statuses == ["failed", "sent"]


def test_dry_run_without_sender(tmp_path):
    s = make_settings(tmp_path)
    sch = make_scheduler(s, sender=None)  # 无推送渠道
    now = _dt(2026, 9, 6, 23, 0)
    upsert_session(s, "oc_1:ou_1", "oc_1", now)
    # dry-run 主动关怀落 dry_run 事件
    recs = asyncio.run(sch.tick(now))
    assert recs[0]["status"] == "dry_run"
    assert care_events(s)[0]["status"] == "dry_run"

    # dry-run 不消费到期提醒（等真实渠道就绪再投递）
    save_reminder("oc_1:ou_1", "吃药", now - timedelta(minutes=5), "提醒我吃药", s)
    asyncio.run(sch.tick(now))
    with db_conn(s) as conn:
        status = conn.execute("SELECT status FROM reminders").fetchone()[0]
    assert status == "pending"


def test_reminder_due_consumed_and_future_kept(tmp_path):
    s = make_settings(tmp_path)
    sender = FakeSender()
    sch = make_scheduler(s, sender=sender)
    now = _dt(2026, 9, 6, 22, 0)
    upsert_session(s, "oc_1:ou_1", "oc_1", now)
    save_reminder("oc_1:ou_1", "给妈妈打电话", now - timedelta(minutes=10), "提醒我给妈妈打电话", s)
    save_reminder("oc_1:ou_1", "明天开会", now + timedelta(hours=12), "提醒我明天开会", s)

    recs = asyncio.run(sch.tick(now))
    due = [r for r in recs if r["care_type"] == "reminder_due"]
    assert len(due) == 1 and "给妈妈打电话" in due[0]["content"]
    assert any("给妈妈打电话" in t for _cid, t in sender.sent)

    with db_conn(s) as conn:
        rows = conn.execute("SELECT id, status FROM reminders ORDER BY id").fetchall()
    statuses = {r["id"]: r["status"] for r in rows}
    assert list(statuses.values()) == ["done", "pending"]


def test_multi_user_isolation(tmp_path):
    s = make_settings(tmp_path)
    sender = FakeSender()
    sch = make_scheduler(s, sender=sender)
    now = _dt(2026, 9, 6, 23, 0)
    upsert_session(s, "oc_1:ou_a", "oc_1", now)
    upsert_session(s, "oc_2:ou_b", "oc_2", now)
    # 用户 A 已道晚安
    asyncio.run(CareTrigger(s).record("oc_1:ou_a", SleepEvent("sleep", _dt(2026, 9, 6, 22, 50))))

    recs = asyncio.run(sch.tick(now))
    assert len(recs) == 1
    assert recs[0]["user_id"] == "oc_2:ou_b"
    assert sender.sent[0][0] == "oc_2"


# ==================== 编排器集成（6.1 接入主链路）====================

def test_orchestrator_goodnight_records_profile(tmp_path):
    """晚安消息：起居指引注入 System Prompt + 画像 sleep_pattern 异步落库。"""
    s = make_settings(tmp_path)
    chat_intent = json.dumps({
        "intent": "chat", "scene": "intimate", "requires_knowledge": False,
        "domain_filter": None, "is_sensitive": False, "slots": {},
    }, ensure_ascii=False)
    extract = json.dumps({
        "episode": {"summary": "用户道晚安准备休息", "topics": ["起居"], "emotion_tag": "平静"},
        "profile_facts": {},
    }, ensure_ascii=False)

    from app.core.intent import IntentParser
    from app.core.orchestrator import Orchestrator
    from app.core.persona import Persona
    from app.memory.manager import MemoryManager

    class _FakeEmbedding:
        async def embed_query(self, text):
            return [0.0]

        async def embed_texts(self, texts):
            return [[0.0] for _ in texts]

    llm = SmartLLM("晚安，漂泊者，愿好梦。", chat_intent, extract)
    trigger = CareTrigger(s)
    mgr = MemoryManager(llm, _FakeEmbedding(), s)
    orch = Orchestrator(
        llm_client=llm, persona=Persona(), intent_parser=IntentParser(llm),
        memory_manager=mgr, care_trigger=trigger,
    )

    async def run():
        req = UnifiedRequest(user_id="local:t", chat_id="local", open_id="t", text="晚安，我先睡了")
        resp = await orch.handle(req)
        await drain_tasks()
        return resp

    resp = asyncio.run(run())
    assert "晚安" in resp.text
    # 起居回应指引注入了某条 LLM 请求的 system prompt
    systems = [c["messages"][0]["content"] for c in llm.calls]
    assert any("起居回应指引" in sys_p for sys_p in systems)
    # 画像已记录就寝状态
    sp = trigger.profile_store.load("local:t")["sleep_pattern"]
    assert sp["last_state"] == "asleep" and sp["last_event_at"]


def test_orchestrator_morning_greeting_block(tmp_path):
    """早安消息同样注入起居指引。"""
    s = make_settings(tmp_path)
    chat_intent = json.dumps({
        "intent": "chat", "scene": "daily", "requires_knowledge": False,
        "domain_filter": None, "is_sensitive": False, "slots": {},
    }, ensure_ascii=False)
    extract = json.dumps({"episode": {"summary": "用户早安", "topics": [], "emotion_tag": "愉悦"},
                          "profile_facts": {}}, ensure_ascii=False)

    from app.core.intent import IntentParser
    from app.core.orchestrator import Orchestrator
    from app.core.persona import Persona
    from app.memory.manager import MemoryManager

    class _FakeEmbedding:
        async def embed_query(self, text):
            return [0.0]

        async def embed_texts(self, texts):
            return [[0.0] for _ in texts]

    llm = SmartLLM("早呀，漂泊者，新的一天慢慢来。", chat_intent, extract)
    trigger = CareTrigger(s)
    mgr = MemoryManager(llm, _FakeEmbedding(), s)
    orch = Orchestrator(
        llm_client=llm, persona=Persona(), intent_parser=IntentParser(llm),
        memory_manager=mgr, care_trigger=trigger,
    )

    async def run():
        req = UnifiedRequest(user_id="local:t2", chat_id="local", open_id="t2", text="早安！")
        await orch.handle(req)
        await drain_tasks()

    asyncio.run(run())
    systems = [c["messages"][0]["content"] for c in llm.calls]
    assert any("起居回应指引" in sys_p and "早安" in sys_p for sys_p in systems)
    assert trigger.profile_store.load("local:t2")["sleep_pattern"]["last_state"] == "awake"
