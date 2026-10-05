"""关怀定时调度（Phase 6.2 + 6.4）。

对齐 IMPLEMENTATION_DESIGN §6.10 / §7.3 / PRD §6.2-6.3：
- asyncio 定时循环，按 care_bedtime_window / care_wake_window（Asia/Shanghai）
  扫描所有 chat_sessions，对未识别起居状态的用户触发主动关怀；
- 自然性约束：单用户单日主动关怀 ≤ care_daily_max（默认 2）；同模板不连续复用；
  已道晚安/早安的用户当日不再提醒；
- 用户主动设置的提醒（reminders 表，Phase 5 登记）到期一并消费投递，
  不计入主动关怀上限；
- 推送渠道抽象：sender(chat_id, text) -> True 成功 / False 失败 / None 无渠道（dry-run）。
  飞书未配置（本地调试）时 dry-run 只记录日志，不消费提醒、不伪造送达。
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime
from typing import Awaitable, Callable

from app.care.composer import CareComposer, ComposedCare
from app.care.trigger import care_now, care_tz
from app.config import Settings, get_settings
from app.core.command import list_pending_reminders
from app.db.connection import db_conn
from app.memory.profile import UserProfileStore

logger = logging.getLogger(__name__)

# sender：返回 True=已送达 / False=发送失败 / None=无推送渠道（dry-run）
CareSender = Callable[[str, str], Awaitable[bool | None]]

_ACTIVE_TYPES = ("bedtime", "wake")
_SENT_STATUSES = ("sent", "dry_run")


# ==================== 纯函数判定（可独立测试）====================

def parse_window(spec: str) -> tuple[int, int]:
    """'22:30-23:59' → (1350, 1439) 分钟数；非法规格回退全天不触发。"""
    m = re.match(r"^\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*$", spec or "")
    if not m:
        logger.warning("[CareScheduler] 时段配置非法：%r，该窗口不触发", spec)
        return (-1, -1)
    h1, m1, h2, m2 = (int(g) for g in m.groups())
    return h1 * 60 + m1, h2 * 60 + m2


def in_window(now_minutes: int, start: int, end: int) -> bool:
    """分钟数是否落在窗口内；支持跨午夜窗口（start > end）。"""
    if start < 0 or end < 0:
        return False
    if start <= end:
        return start <= now_minutes <= end
    return now_minutes >= start or now_minutes <= end


def local_midnight_epoch(now: datetime) -> int:
    """本地时区今天 0 点的 UTC epoch（care_events.created_at 按日统计用）。"""
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(midnight.timestamp())


# ==================== CareScheduler ====================

class CareScheduler:
    """关怀调度器（asyncio 定时循环 + 主动推送）。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        composer: CareComposer | None = None,
        profile_store: UserProfileStore | None = None,
        sender: CareSender | None = None,
        llm=None,
    ) -> None:
        self._settings = settings or get_settings()
        self._composer = composer or CareComposer(self._settings, llm=llm)
        self._profiles = profile_store or UserProfileStore(self._settings)
        self._sender = sender
        self._task: asyncio.Task | None = None
        self._next_scan_at: float | None = None
        self._bedtime_window = parse_window(self._settings.care_bedtime_window)
        self._wake_window = parse_window(self._settings.care_wake_window)
        logger.info(
            "[CareScheduler] 初始化（bedtime=%s wake=%s 日上限=%d 间隔=%ds llm_compose=%s）",
            self._settings.care_bedtime_window, self._settings.care_wake_window,
            self._settings.care_daily_max, self._settings.care_scan_interval,
            self._settings.care_llm_compose,
        )

    # ==================== 生命周期 ====================

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._loop())
        logger.info("[CareScheduler] 定时循环已启动")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            logger.info("[CareScheduler] 定时循环已停止")

    async def _loop(self) -> None:
        try:
            while True:
                try:
                    await self.tick()
                except Exception as e:  # noqa: BLE001
                    logger.exception("[CareScheduler] 扫描轮异常（下轮继续）：%s", e)
                self._next_scan_at = time.time() + self._settings.care_scan_interval
                await asyncio.sleep(self._settings.care_scan_interval)
        except asyncio.CancelledError:
            raise

    # ==================== 核心扫描（注入 now 可测试）====================

    async def tick(self, now: datetime | None = None) -> list[dict]:
        """执行一轮扫描：到期提醒投递 + 窗口内主动关怀。返回本轮推送记录。"""
        now = now or care_now(self._settings)
        records: list[dict] = []

        # 1) 到期提醒（用户主动设置，不计日上限；dry-run 不消费）
        records.extend(await self._process_due_reminders(now))

        # 2) 窗口判定（就寝 / 起床，两窗口不重叠）
        now_minutes = now.hour * 60 + now.minute
        if in_window(now_minutes, *self._bedtime_window):
            care_type = "bedtime"
        elif in_window(now_minutes, *self._wake_window):
            care_type = "wake"
        else:
            return records

        # 3) 扫描会话，逐用户判定
        midnight = local_midnight_epoch(now)
        for sess in self._list_chat_sessions():
            try:
                rec = await self._maybe_care_user(sess, care_type, now, midnight)
                if rec is not None:
                    records.append(rec)
            except Exception as e:  # noqa: BLE001
                logger.warning("[CareScheduler] 用户 %s 关怀处理失败（跳过）：%s", sess.get("user_id"), e)
        return records

    # ==================== 到期提醒消费 ====================

    async def _process_due_reminders(self, now: datetime) -> list[dict]:
        records: list[dict] = []
        tz = care_tz(self._settings)
        for rem in list_pending_reminders(self._settings):
            try:
                due = self._parse_remind_at(rem.get("remind_at"), tz)
                if due is None or due > now:
                    continue
                chat_id = self._chat_id_of(rem["user_id"])
                if not chat_id:
                    logger.warning("[CareScheduler] 提醒 #%s 无 chat_sessions 记录，跳过投递", rem.get("id"))
                    continue
                composed = self._composer.compose_reminder(rem["content"])
                status = await self._deliver(chat_id, rem["user_id"], "reminder_due", composed)
                if status is None:
                    # dry-run：不消费提醒，等真实渠道就绪后投递
                    logger.info("[CareScheduler] dry-run：提醒 #%s 暂不消费：%s", rem.get("id"), rem["content"][:30])
                    continue
                if status == "sent":
                    self._mark_reminder_done(int(rem["id"]))
                records.append({
                    "user_id": rem["user_id"], "care_type": "reminder_due",
                    "template_id": composed.template_id, "status": status,
                    "content": composed.text,
                })
            except Exception as e:  # noqa: BLE001
                logger.warning("[CareScheduler] 提醒 #%s 处理失败：%s", rem.get("id"), e)
        return records

    @staticmethod
    def _parse_remind_at(raw: str | None, tz) -> datetime | None:
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        return dt

    def _mark_reminder_done(self, reminder_id: int) -> None:
        with db_conn(self._settings) as conn:
            conn.execute("UPDATE reminders SET status='done' WHERE id=? AND status='pending'", (reminder_id,))

    # ==================== 主动关怀单用户判定 ====================

    async def _maybe_care_user(
        self, sess: dict, care_type: str, now: datetime, midnight_epoch: int
    ) -> dict | None:
        user_id = sess["user_id"]
        chat_id = sess.get("chat_id") or ""
        if not chat_id:
            return None

        # 活跃判定：就寝关怀要求今天有互动；起床问候放宽到近两天（唤醒性质）
        active_since = midnight_epoch if care_type == "bedtime" else midnight_epoch - 86400
        if int(sess.get("updated_at") or 0) < active_since:
            return None

        profile = self._profiles.load(user_id)
        sp = profile.get("sleep_pattern") or {}

        # 已识别起居状态（今天已道晚安/早安）→ 不重复提醒
        today = now.strftime("%Y-%m-%d")
        wanted_state = "asleep" if care_type == "bedtime" else "awake"
        if sp.get("last_state_date") == today and sp.get("last_state") == wanted_state:
            return None

        # 今日该类型已成功推送过 → 不重复
        if self._count_today_care(user_id, midnight_epoch, types=(care_type,)) > 0:
            return None
        # 单日主动关怀上限（bedtime+wake；提醒不计）
        if self._count_today_care(user_id, midnight_epoch, types=_ACTIVE_TYPES) >= self._settings.care_daily_max:
            logger.info("[CareScheduler] 用户 %s 今日主动关怀已达上限 %d，跳过", user_id, self._settings.care_daily_max)
            return None

        avoid = self._last_template_id(user_id, care_type)
        composed = await self._composer.compose(care_type, profile, avoid_template_id=avoid)
        status = await self._deliver(chat_id, user_id, care_type, composed)
        return {
            "user_id": user_id, "care_type": care_type,
            "template_id": composed.template_id, "status": status,
            "content": composed.text, "source": composed.source,
        }

    async def _deliver(self, chat_id: str, user_id: str, care_type: str, composed: ComposedCare) -> str:
        """推送并落 care_events；返回 'sent' / 'failed' / 'dry_run'。"""
        if self._sender is None:
            logger.info("[CareScheduler] dry-run（无推送渠道）：%s → %s", care_type, composed.text[:40])
            self._insert_event(user_id, care_type, composed, "dry_run")
            return "dry_run"
        try:
            ok = await self._sender(chat_id, composed.text)
        except Exception as e:  # noqa: BLE001
            logger.warning("[CareScheduler] 推送异常 user=%s：%s", user_id, e)
            ok = False
        status = "sent" if ok else "failed"
        self._insert_event(user_id, care_type, composed, status)
        if not ok:
            logger.warning("[CareScheduler] 推送失败 user=%s type=%s（下轮可重试）", user_id, care_type)
        return status

    # ==================== DB 辅助 ====================

    def _list_chat_sessions(self) -> list[dict]:
        with db_conn(self._settings) as conn:
            rows = conn.execute(
                "SELECT user_id, chat_id, open_id, updated_at FROM chat_sessions"
            ).fetchall()
        return [dict(r) for r in rows]

    def _chat_id_of(self, user_id: str) -> str | None:
        with db_conn(self._settings) as conn:
            row = conn.execute(
                "SELECT chat_id FROM chat_sessions WHERE user_id=?", (user_id,)
            ).fetchone()
        return row["chat_id"] if row else None

    def _count_today_care(self, user_id: str, since_epoch: int, *, types: tuple[str, ...]) -> int:
        """今日已成功送达（sent/dry_run）的关怀数；failed 不计（允许重试）。"""
        placeholders = ",".join("?" for _ in types)
        statuses = ",".join("?" for _ in _SENT_STATUSES)
        with db_conn(self._settings) as conn:
            row = conn.execute(
                f"SELECT COUNT(*) FROM care_events "
                f"WHERE user_id=? AND care_type IN ({placeholders}) "
                f"AND status IN ({statuses}) AND created_at >= ?",
                (user_id, *types, *_SENT_STATUSES, since_epoch),
            ).fetchone()
        return int(row[0])

    def _last_template_id(self, user_id: str, care_type: str) -> str | None:
        with db_conn(self._settings) as conn:
            row = conn.execute(
                "SELECT template_id FROM care_events "
                "WHERE user_id=? AND care_type=? AND template_id IS NOT NULL "
                "ORDER BY created_at DESC, id DESC LIMIT 1",
                (user_id, care_type),
            ).fetchone()
        return row["template_id"] if row else None

    def _insert_event(self, user_id: str, care_type: str, composed: ComposedCare, status: str) -> None:
        with db_conn(self._settings) as conn:
            conn.execute(
                "INSERT INTO care_events (user_id, care_type, template_id, content, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, care_type, composed.template_id, composed.text, status, int(time.time())),
            )

    # ==================== 调试台状态（Phase 7 监测面板用）====================

    def status(self) -> dict:
        with db_conn(self._settings) as conn:
            total = conn.execute("SELECT COUNT(*) FROM care_events").fetchone()[0]
            recent = conn.execute(
                "SELECT user_id, care_type, template_id, status, created_at "
                "FROM care_events ORDER BY id DESC LIMIT 10"
            ).fetchall()
        return {
            "running": self._task is not None and not self._task.done(),
            "next_scan_at": self._next_scan_at,
            "events_total": total,
            "recent_events": [dict(r) for r in recent],
            "bedtime_window": self._settings.care_bedtime_window,
            "wake_window": self._settings.care_wake_window,
            "daily_max": self._settings.care_daily_max,
        }
