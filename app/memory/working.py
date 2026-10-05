"""工作记忆（Working Memory）—— Phase 4 任务 4.1。

对齐 docs/tech_docs/MEMORY_SYSTEM_DESIGN.md §2.1：
- 滑动窗口 N 轮（默认 12 轮 = 24 条 user/assistant 消息），超出窗口最老的消息丢弃；
- 进程内 deque + SQLite working_memory 表持久化（防进程重启丢失，懒加载恢复）；
- 存储完整 user/assistant turn 序列，不做摘要；
- 每条消息同步读写，无延迟；窗口只保留最近 2N 条，SQLite 不无界增长。

接口对齐 Phase 2 SimpleWorkingMemory（read/write），替换零改动。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque

from app.config import Settings, get_settings
from app.db.connection import db_conn

logger = logging.getLogger(__name__)

# 默认窗口轮数（config memory_window_size）；1 轮 = 1 条 user + 1 条 assistant
_DEFAULT_WINDOW_SIZE = 12


class WorkingMemory:
    """进程内工作记忆：滑动窗口 + SQLite 持久化 + 懒加载。

    - 窗口容量按"轮"计（window_size=12 → 最多保留最近 12 轮 = 24 条消息）；
    - append 时最老的自动溢出（丢弃；情景记忆归档由 MemoryManager 在写入时异步完成）；
    - SQLite working_memory 表只保留每用户最近 2N 条（防无界增长）；
    - 进程重启后首次 read/write 时从 SQLite 懒加载恢复窗口。
    """

    def __init__(self, window_size: int = _DEFAULT_WINDOW_SIZE, settings: Settings | None = None) -> None:
        self._window_size = max(1, window_size)
        self._max_messages = self._window_size * 2  # 1 轮 = user + assistant 两条
        self._settings = settings or get_settings()
        self._buffers: dict[str, deque[dict]] = {}
        self._loaded: set[str] = set()  # 已从 SQLite 懒加载过的用户
        self._lock = asyncio.Lock()
        logger.info(
            "[WorkingMemory] 初始化完成：window_size=%d 轮（%d 条消息）",
            self._window_size, self._max_messages,
        )

    # ==================== 对外接口 ====================

    async def read(self, user_id: str) -> list[dict]:
        """读取该用户的工作记忆窗口（[{'role', 'content'}]，按时间升序）。"""
        async with self._lock:
            await self._ensure_loaded(user_id)
            return list(self._buffers.get(user_id, ()))

    async def write(self, user_id: str, role: str, content: str) -> None:
        """追加一条消息到工作记忆（同步持久化，失败不阻塞）。"""
        async with self._lock:
            await self._ensure_loaded(user_id)
            buffer = self._buffers.setdefault(
                user_id, deque(maxlen=self._max_messages)
            )
            buffer.append({"role": role, "content": content})
        # 持久化放在锁外（SQLite 慢操作不阻塞其他用户；失败仅告警）
        try:
            self._persist(user_id, role, content)
        except Exception as e:  # noqa: BLE001
            logger.warning("[WorkingMemory] SQLite 持久化失败：%s", e)

    async def clear(self, user_id: str) -> None:
        """清空该用户的工作记忆（内存 + SQLite）。"""
        async with self._lock:
            self._buffers.pop(user_id, None)
            self._loaded.discard(user_id)
        try:
            with db_conn(self._settings) as conn:
                conn.execute("DELETE FROM working_memory WHERE user_id = ?", (user_id,))
        except Exception as e:  # noqa: BLE001
            logger.warning("[WorkingMemory] SQLite 清理失败：%s", e)

    # ==================== 内部实现 ====================

    async def _ensure_loaded(self, user_id: str) -> None:
        """进程重启后首次访问时，从 SQLite 恢复该用户窗口（幂等）。"""
        if user_id in self._loaded:
            return
        self._loaded.add(user_id)
        if user_id in self._buffers:
            return
        try:
            rows = self._load_recent(user_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("[WorkingMemory] SQLite 懒加载失败（按空窗口继续）：%s", e)
            return
        if rows:
            self._buffers[user_id] = deque(
                ({"role": r["role"], "content": r["content"]} for r in rows),
                maxlen=self._max_messages,
            )
            logger.info("[WorkingMemory] %s 从 SQLite 恢复 %d 条消息", user_id, len(rows))

    def _load_recent(self, user_id: str) -> list:
        """从 SQLite 读取该用户最近 2N 条消息（turn_index 升序）。"""
        with db_conn(self._settings) as conn:
            rows = conn.execute(
                """
                SELECT role, content FROM working_memory
                WHERE user_id = ?
                ORDER BY turn_index DESC
                LIMIT ?
                """,
                (user_id, self._max_messages),
            ).fetchall()
        rows.reverse()  # 存储是 DESC 取最近，恢复成时间升序
        return rows

    def _persist(self, user_id: str, role: str, content: str) -> None:
        """同步写入 SQLite，并裁剪窗口外旧记录（只保留最近 2N 条）。"""
        now = int(time.time())
        with db_conn(self._settings) as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(turn_index), -1) FROM working_memory WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            next_idx = (row[0] if row else -1) + 1
            conn.execute(
                """
                INSERT INTO working_memory (user_id, turn_index, role, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, next_idx, role, content, now),
            )
            # 裁剪：只保留最近 2N 条（N 轮），避免表无界增长
            conn.execute(
                """
                DELETE FROM working_memory
                WHERE user_id = ?
                  AND turn_index NOT IN (
                    SELECT turn_index FROM working_memory
                    WHERE user_id = ?
                    ORDER BY turn_index DESC
                    LIMIT ?
                  )
                """,
                (user_id, user_id, self._max_messages),
            )
