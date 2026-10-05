"""指令执行基础模块（Phase 5.5）。

对齐 IMPLEMENTATION_DESIGN Phase 5 任务 5.5 与 PRD A-05：
- 明确指令可执行：提醒（登记 reminders 表，Phase 6 关怀调度器消费）、
  记录（内容随记忆系统自动落库）、查询（转 RAG 知识链路）；
- 模糊指令主动确认：缺时间/缺内容/意图不明时输出追问指引；
- 不可执行礼貌说明：超出能力边界的第三方操作（订票/转账/购物等）。

设计：规则化实现（零 LLM 配额），输出 CommandOutcome.guidance 供编排器注入
System Prompt，最终确认/追问话术由 persona 主生成（保持角色化）。
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from app.config import Settings, get_settings
from app.db.connection import db_conn

logger = logging.getLogger(__name__)


# ==================== 结果结构 ====================

@dataclass
class CommandOutcome:
    """指令执行结果（编排器据此注入策略块）。"""

    action: str = "unknown"        # reminder / record / query / unknown
    status: str = "clarify"        # executed / clarify / infeasible
    missing: list[str] = field(default_factory=list)   # clarify 时缺失要素
    detail: str = ""               # executed 时的要素摘要（时间/内容）
    guidance: str = ""             # 注入 System Prompt 的指引文本

    @property
    def is_query(self) -> bool:
        return self.action == "query"


# ==================== 规则表 ====================

# 不可执行的第三方操作（能力边界，礼貌说明）
_INFEASIBLE_PATTERNS = [
    r"订[机票票餐]", r"买.{0,6}(机票|票|东西|礼物)", r"抢票", r"下单",
    r"转账", r"付款", r"支付", r"发(邮件|红包|快递)", r"寄.{0,4}东西",
    r"打电话给", r"预约挂号", r"帮我.{0,4}(订|购|买|寄|转)",
]
_INFEASIBLE_REGEXES = [re.compile(p) for p in _INFEASIBLE_PATTERNS]

# 动作识别
_REMINDER_TRIGGER = re.compile(r"(提醒我|设个提醒|定时提醒|记得提醒|叫我)")
_RECORD_TRIGGER = re.compile(r"(记录一下|帮我记|记一下|记住|记录|帮我记住)")
_QUERY_TRIGGER = re.compile(r"(帮我查|查一下|查询|搜一下|查查)")

# 时间表达（相对解析到下一个发生时刻）
_TIME_PATTERNS = [
    re.compile(r"(?P<day>明[天早晚]|后天)?\s*(?P<period>凌晨|早上|早晨|上午|中午|下午|傍晚|晚上|夜里)?\s*(?P<hour>\d{1,2})[点时:：]\s*(?:(?P<minute>\d{1,2})分|(?P<half>半|一刻|三刻))?"),
    re.compile(r"(?P<day>明[天早晚]|后天)?\s*(?P<period>凌晨|早上|早晨|上午|中午|下午|傍晚|晚上|夜里)\s*(?P<hour>\d{1,2})\s*点?\s*(?:(?P<minute>\d{1,2})分|(?P<half>半|一刻|三刻))?"),
]
# 时段语义说明见 _parse_period_hour（归一化逻辑集中于此）


def _parse_period_hour(period: str | None, hour: int) -> int:
    """时段 + 小时归一化到 24 小时（「晚上10点」→ 22，「凌晨2点」→ 2）。"""
    if not period:
        return hour
    if period in ("凌晨", "早上", "早晨", "上午"):
        return hour % 24
    if period == "中午":
        if hour <= 2:
            return 12 + hour  # 中午1点 → 13
        return hour if hour >= 12 else 12
    # 下午 / 傍晚 / 晚上 / 夜里
    return (12 + hour) % 24 if hour <= 12 else hour


def parse_reminder_time(raw: str, now: datetime | None = None) -> datetime | None:
    """把中文时间表达解析为下一个发生时刻；解析失败返回 None。

    支持：明[天]/后天 + 时段（凌晨/早上/上午/中午/下午/晚上/夜里）+ X点[半/X分]。
    """
    if not raw:
        return None
    now = now or datetime.now()
    for pattern in _TIME_PATTERNS:
        m = pattern.search(raw)
        if not m:
            continue
        day = m.group("day") or ""
        period = m.group("period")
        try:
            hour = int(m.group("hour"))
        except (TypeError, ValueError):
            continue
        if not (0 <= hour <= 23):
            continue
        minute_txt = m.group("minute")
        half = m.group("half")
        minute = 0
        if minute_txt:
            minute = int(minute_txt)
            if not (0 <= minute <= 59):
                continue
        elif half == "半":
            minute = 30
        elif half == "一刻":
            minute = 15
        elif half == "三刻":
            minute = 45

        target = now.replace(
            hour=_parse_period_hour(period, hour), minute=minute, second=0, microsecond=0,
        )
        if "明" in day:
            target += timedelta(days=1 if day == "明天" or day == "明早" or day == "明晚" else 2)
        elif "后" in day:
            target += timedelta(days=2)
        elif target <= now:
            target += timedelta(days=1)  # 已过时刻 → 下一天同时刻
        return target
    return None


def _extract_content(text: str, trigger: re.Pattern) -> str:
    """去掉触发短语后的剩余文本作为内容。"""
    content = trigger.sub("", text, count=1)
    content = content.strip(" ，,。：:？?！!、")
    return content


def _strip_time_text(content: str, raw_time: str) -> str:
    """从内容中剔除时间表达（时间属于提醒要素，不属于提醒内容）。"""
    for p in _TIME_PATTERNS:
        content = p.sub("", content)
    if raw_time:
        content = content.replace(raw_time, "")
    return content.strip(" ，,。：:？?！!、")


# ==================== reminders 持久化（Phase 6 调度器消费）====================

def save_reminder(
    user_id: str,
    content: str,
    remind_at: datetime | None,
    raw_text: str,
    settings: Settings | None = None,
) -> int:
    """登记一条待提醒（status=pending），返回自增 id。"""
    settings = settings or get_settings()
    with db_conn(settings) as conn:
        cur = conn.execute(
            "INSERT INTO reminders (user_id, content, remind_at, raw_time_text, raw_text, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, 'pending', ?)",
            (
                user_id,
                content,
                remind_at.isoformat(timespec="minutes") if remind_at else None,
                "",
                raw_text,
                int(time.time()),
            ),
        )
        return int(cur.lastrowid or 0)


def list_pending_reminders(settings: Settings | None = None) -> list[dict]:
    """列出全部待提醒（Phase 6 调度器按 remind_at 消费）。"""
    settings = settings or get_settings()
    with db_conn(settings) as conn:
        rows = conn.execute(
            "SELECT * FROM reminders WHERE status='pending' ORDER BY remind_at"
        ).fetchall()
    return [dict(r) for r in rows]


# ==================== CommandExecutor ====================

class CommandExecutor:
    """指令执行器（规则化，零 LLM 配额）。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def handle(self, user_text: str, slots: dict, user_id: str) -> CommandOutcome:
        """解析并执行指令，返回 CommandOutcome（不抛异常，失败降级为 clarify）。"""
        text = (user_text or "").strip()
        try:
            # 1) 不可执行：第三方现实操作
            if any(r.search(text) for r in _INFEASIBLE_REGEXES):
                return self._infeasible()

            # 2) 提醒
            if _REMINDER_TRIGGER.search(text):
                return self._handle_reminder(text, slots, user_id)

            # 3) 记录
            if _RECORD_TRIGGER.search(text):
                return self._handle_record(text)

            # 4) 查询：转知识链路（编排器据 action=query 走 RAG + 置信度）
            if _QUERY_TRIGGER.search(text):
                return self._handle_query()

            # 5) 意图不明（如「帮我把这个整理一下」）
            return self._clarify(action="unknown", missing=["具体想让我做什么"])
        except Exception:  # noqa: BLE001
            logger.exception("[Command] 指令处理异常，降级为追问确认")
            return self._clarify(action="unknown", missing=["具体想让我做什么"])

    # ---- 各动作处理 ----

    def _handle_reminder(self, text: str, slots: dict, user_id: str) -> CommandOutcome:
        raw_time = (slots.get("reminder_time") or "").strip()
        content = _extract_content(text, _REMINDER_TRIGGER)
        content = _strip_time_text(content, raw_time)

        missing: list[str] = []
        # 优先从全文解析（保留时段语义，如「晚上10点」）；slots 仅作兜底
        remind_at = parse_reminder_time(text) or parse_reminder_time(raw_time)
        if remind_at is None:
            missing.append("提醒时间（几点？）")
        if not content:
            missing.append("提醒内容")
        if missing:
            return self._clarify(action="reminder", missing=missing)

        reminder_id = save_reminder(user_id, content, remind_at, text, self._settings)
        logger.info(
            "[Command] 提醒已登记 #%d：user=%s at=%s content=%s",
            reminder_id, user_id, remind_at, content[:30],
        )
        time_str = remind_at.strftime("%m月%d日 %H:%M") if remind_at else ""
        return CommandOutcome(
            action="reminder",
            status="executed",
            detail=f"提醒时间：{time_str}；提醒内容：{content}",
            guidance=(
                "【指令执行指引】用户请求设置提醒，已成功登记："
                f"{time_str}，提醒「{content}」。"
                "请以角色身份自然确认（一两句即可，可带一点温柔叮嘱，"
                "比如到时会叫 Ta），不要复述本指引，不要假装还在操作什么。"
            ),
        )

    def _handle_record(self, text: str) -> CommandOutcome:
        content = _extract_content(text, _RECORD_TRIGGER)
        if not content:
            return self._clarify(action="record", missing=["想记录的内容"])
        # 内容随本轮对话写入记忆系统（工作记忆 + 情景/画像提取），无需额外落库
        return CommandOutcome(
            action="record",
            status="executed",
            detail=f"记录内容：{content}",
            guidance=(
                f"【指令执行指引】用户请求记录：「{content}」。"
                "这条内容已经进入你的记忆。请以角色身份自然确认记下了要点"
                "（一两句即可，可加一句简短回应），不要复述本指引。"
            ),
        )

    def _handle_query(self) -> CommandOutcome:
        return CommandOutcome(
            action="query",
            status="executed",
            detail="查询类指令，转入知识库检索链路",
            guidance=(
                "【指令执行指引】用户希望你帮忙查证后回答。"
                "请依据下方知识参考作答并遵循置信度措辞要求（如有）。"
            ),
        )

    def _infeasible(self) -> CommandOutcome:
        return CommandOutcome(
            action="unknown",
            status="infeasible",
            guidance=(
                "【能力边界指引】用户请求的事情超出你的能力范围（你无法在现实世界中"
                "代为订票、购物、转账、打电话等）。请以角色身份温和、诚实地说明做不到，"
                "并主动提供力所能及的替代（比如帮你查资料、记下这件事、或者陪 Ta 聊聊），"
                "不要道歉过度，不要假装已执行。"
            ),
        )

    def _clarify(self, action: str, missing: list[str]) -> CommandOutcome:
        detail = "、".join(missing)
        return CommandOutcome(
            action=action,
            status="clarify",
            missing=missing,
            guidance=(
                f"【指令澄清指引】用户的指令信息不完整（缺少：{detail}）。"
                "请以角色身份自然地追问缺失的要素（一次只问一个），"
                "不要假装已经完成指令。"
            ),
        )
