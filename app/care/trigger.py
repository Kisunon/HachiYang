"""起居识别（Phase 6.1）。

对齐 IMPLEMENTATION_DESIGN §6.10 / PRD §6.1：
- 监听早晚安关键词/语义（"晚安""起床了""准备睡了"等），结合 Asia/Shanghai
  时区与合理作息区间判定起居状态；
- 更新用户画像 sleep_pattern（avg_bedtime / avg_wake_time 指数滑动平均、
  irregular_days 不规律日期、last_state 当日起居状态）；
- 画像由记忆系统承接（MEMORY_SYSTEM_DESIGN §6），本模块只写 sleep_pattern
  子字段，不重复存储作息；chat_sessions.updated_at 与 care_events 由调度器使用。

设计：detect_sleep_event 为无副作用纯函数（主链路每轮调用，零 IO）；
CareTrigger.record 做画像持久化（编排器在异步记忆写入后调用，避免画像竞争）。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

try:  # Windows 无 tzdata 时回退固定 +8（Asia/Shanghai 无夏令时，恒为 UTC+8）
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from app.config import Settings, get_settings
from app.memory.profile import UserProfileStore

logger = logging.getLogger(__name__)

# ==================== 时区工具 ====================

_SHANGHAI_FIXED = timezone(timedelta(hours=8))


def care_tz(settings: Settings | None = None) -> timezone | "ZoneInfo":  # type: ignore[name-defined]
    """关怀调度时区（Asia/Shanghai；zoneinfo 不可用时回退 UTC+8 固定偏移）。"""
    settings = settings or get_settings()
    if ZoneInfo is not None:
        try:
            return ZoneInfo(settings.care_timezone)
        except Exception as e:  # noqa: BLE001
            logger.warning("[Care] 时区 %r 解析失败，回退 UTC+8：%s", settings.care_timezone, e)
    return _SHANGHAI_FIXED


def care_now(settings: Settings | None = None) -> datetime:
    """当前关怀时区时间（aware）。"""
    return datetime.now(care_tz(settings))


# ==================== 起居事件识别 ====================

@dataclass
class SleepEvent:
    """一次起居事件。kind: 'sleep'（就寝）/ 'wake'（起床）。"""

    kind: str
    at: datetime


# 就寝：第一人称表述 + 晚安问候（"早点睡吧/你也早点休息"是对 Agent 说的，不计）
_SLEEP_PATTERNS = [
    re.compile(r"晚安"),
    re.compile(r"(我|俺|人家|先|准备|打算|要|去|快|这就|马上)?[^。！？!?，,]{0,6}(睡了|睡啦|睡觉了|去睡|要睡|歇息|就寝|休息了|关灯)"),
    re.compile(r"(我|俺|人家)?先睡"),
]
# 起床：早安问候 + 第一人称醒/起表述（"太早了/早说过"不误伤）
_WAKE_PATTERNS = [
    re.compile(r"(早安|早上好|早啊|早呀|早～|早~|早，|早,|早。)"),
    re.compile(r"(起床了|起来了|醒了|睡醒|醒啦|起了|起床啦)"),
]
# 第二人称祈使（用户在催 Agent 睡/醒），命中则排除
_SECOND_PERSON = re.compile(r"(你|您|秧秧|玄翎|哈基|宝贝|乖乖).{0,8}(睡|休息|醒|起)")


def detect_sleep_event(text: str, now: datetime | None = None) -> SleepEvent | None:
    """从用户文本识别起居事件；非起居表述返回 None（纯函数，零 IO）。

    注意：单字"早"不触发（"太早了""早知道"等误伤），必须是早安问候或
    明确的起床/睡醒表述；"早点睡""你也早点休息"为第二人称祈使，不计。
    """
    if not text:
        return None
    now = now or care_now()
    snippet = text.strip()[:60]
    if _SECOND_PERSON.search(snippet):
        return None
    if any(p.search(snippet) for p in _SLEEP_PATTERNS):
        return SleepEvent(kind="sleep", at=now)
    if any(p.search(snippet) for p in _WAKE_PATTERNS):
        return SleepEvent(kind="wake", at=now)
    return None


# ==================== 作息时间平均（跨午夜）====================

def _time_to_axis(hhmm: str, origin: int) -> int | None:
    """'HH:MM' → 以 origin 点为零点的分钟轴（处理跨午夜）。

    就寝 origin=18（18:00=0 … 次日 17:59=1439）；起床 origin=3（03:00=0 … 次日 02:59）。
    """
    m = re.match(r"^(\d{1,2}):(\d{2})$", (hhmm or "").strip())
    if not m:
        return None
    minutes = int(m.group(1)) * 60 + int(m.group(2))
    axis = minutes - origin * 60
    if axis < 0:
        axis += 1440
    return axis


def _axis_to_time(axis: int, origin: int) -> str:
    minutes = (origin * 60 + axis) % 1440
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _ema_time(old_hhmm: str, new_hhmm: str, origin: int, alpha: float = 0.4) -> str:
    """作息时间指数滑动平均（跨午夜轴上做 EMA）；无历史值时直接取新值。"""
    new_axis = _time_to_axis(new_hhmm, origin)
    if new_axis is None:
        return old_hhmm or new_hhmm
    old_axis = _time_to_axis(old_hhmm, origin) if old_hhmm else None
    if old_axis is None:
        return new_hhmm
    merged = old_axis * (1 - alpha) + new_axis * alpha
    return _axis_to_time(int(round(merged)), origin)


# 合理作息区间（轴外记 irregular_days）
_BEDTIME_ORIGIN = 18       # 就寝轴：18:00 起
# 21:00 = 3*60=180；次日 00:30 = (24-18)*60+30 = 390
_BEDTIME_REGULAR = (180, 390)
_WAKE_ORIGIN = 3           # 起床轴：03:00 起
# 05:00 = 2*60=120；10:00 = 7*60=420
_WAKE_REGULAR = (120, 420)


# ==================== CareTrigger ====================

class CareTrigger:
    """起居事件 → 画像 sleep_pattern 持久化。"""

    def __init__(
        self,
        settings: Settings | None = None,
        profile_store: UserProfileStore | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.profile_store = profile_store or UserProfileStore(self._settings)
        self._profiles = self.profile_store  # 内部别名

    async def record(self, user_id: str, event: SleepEvent) -> dict:
        """记录一次起居事件到画像 sleep_pattern，返回更新后的 sleep_pattern。

        只更新 sleep_pattern 子字段（load → 局部修改 → save），不触碰画像其他部分；
        任何异常降级为日志告警，不阻塞主链路。
        """
        try:
            profile = self._profiles.load(user_id)
            sp = profile.setdefault("sleep_pattern", {})
            # 旧画像补齐字段
            for k, v in (
                ("avg_bedtime", ""), ("avg_wake_time", ""), ("irregular_days", []),
                ("last_updated", ""), ("last_state", ""), ("last_state_date", ""),
                ("last_event_at", ""),
            ):
                sp.setdefault(k, v)

            local_date = event.at.strftime("%Y-%m-%d")
            hhmm = event.at.strftime("%H:%M")
            irregular = False

            if event.kind == "sleep":
                sp["avg_bedtime"] = _ema_time(sp.get("avg_bedtime", ""), hhmm, _BEDTIME_ORIGIN)
                sp["last_state"] = "asleep"
                axis = _time_to_axis(hhmm, _BEDTIME_ORIGIN)
                if axis is not None and not (_BEDTIME_REGULAR[0] <= axis <= _BEDTIME_REGULAR[1]):
                    irregular = True
            elif event.kind == "wake":
                sp["avg_wake_time"] = _ema_time(sp.get("avg_wake_time", ""), hhmm, _WAKE_ORIGIN)
                sp["last_state"] = "awake"
                axis = _time_to_axis(hhmm, _WAKE_ORIGIN)
                if axis is not None and not (_WAKE_REGULAR[0] <= axis <= _WAKE_REGULAR[1]):
                    irregular = True
            else:  # pragma: no cover
                logger.warning("[CareTrigger] 未知起居事件类型：%s", event.kind)
                return sp

            sp["last_state_date"] = local_date
            sp["last_event_at"] = hhmm
            sp["last_updated"] = local_date
            if irregular:
                days = sp.setdefault("irregular_days", [])
                if local_date not in days:
                    days.append(local_date)
                del days[:-30]

            self._profiles.save(user_id, profile)
            logger.info(
                "[CareTrigger] 起居事件已记录 user=%s kind=%s at=%s state=%s",
                user_id, event.kind, hhmm, sp["last_state"],
            )
            return sp
        except Exception as e:  # noqa: BLE001
            logger.warning("[CareTrigger] 起居事件记录失败（不阻塞对话）：%s", e)
            return {}


# ==================== 编排器注入用指引 ====================

def build_sleep_greeting_guidance(event: SleepEvent) -> str:
    """用户道早晚安时注入 System Prompt 的回应指引（保持角色化，不重复提醒）。"""
    if event.kind == "sleep":
        return (
            "【起居回应指引】用户刚道晚安（准备休息）。请以角色身份温柔回应，"
            "祝漂泊者一夜好眠，可带一句简短的诗意叮嘱（风声、夜色、好梦之类），"
            "一两句即可，不要再主动开启新话题，不要说教式地讲健康知识。"
        )
    return (
        "【起居回应指引】用户刚道早安（刚起床）。请以角色身份温柔回应新的一天，"
        "可带一句轻快的问候或简短祝福，一两句即可，自然温暖，不要机械地汇报时间或天气。"
    )
