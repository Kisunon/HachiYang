"""Memory Manager —— Phase 4 任务 4.4：三层记忆统一编排。

对齐 docs/tech_docs/MEMORY_SYSTEM_DESIGN.md §3 / §4：
- MemoryRead：对话开始前调用（意图解析之前），返回 MemoryContext
  （working_window + episodic_hits + user_profile）；
- MemoryWrite：LLM 回应后**异步**调用（不阻塞主链路，M-08）：
  1) 同步更新工作记忆；
  2) 异步：单次 LLM 调用同时提取情景记忆摘要与画像事实（节省 API 配额），
     摘要向量化后写入情景记忆，事实 merge 进用户画像；
- MemoryForget：用户主动遗忘（episodic / profile / all_sensitive）；
- 周期清理：每周清理 decay < 0.1 的情景记忆（start_decay_task）。

多用户隔离：全部接口以 user_id（"chat_id:open_id" 分区键）强制隔离（M-02）。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable

from app.config import Settings, get_settings
from app.core.models import MemoryContext
from app.llm.prompts import build_memory_extract_messages
from app.memory.episodic import EpisodicMemory
from app.memory.profile import UserProfileStore
from app.memory.working import WorkingMemory

logger = logging.getLogger(__name__)

# 情景记忆检索 Top-K（设计建议 K=3~5）
_EPISODIC_TOP_K = 5
# 衰减清理周期（秒）：每周
_DECAY_CLEANUP_INTERVAL = 7 * 86400

# 链路追踪回调签名：trace(name, input, output, notes)
TraceFn = Callable[[str, str, str, str], None]


def _extract_json(text: str) -> dict | None:
    """解析 LLM 返回的 JSON（容忍 ```json 围栏）。"""
    text = text.strip()
    if text.startswith("```"):
        # 去掉 ```json / ``` 围栏
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


class MemoryManager:
    """三层记忆统一编排器（对上层只暴露 Read / Write / Forget）。"""

    def __init__(
        self,
        llm_client: Any,
        embedding: Any,
        settings: Settings | None = None,
        *,
        window_size: int | None = None,
    ) -> None:
        self._llm = llm_client
        self._embedding = embedding
        self._settings = settings or get_settings()
        self._working = WorkingMemory(
            window_size=window_size or self._settings.memory_window_size,
            settings=self._settings,
        )
        self._episodic = EpisodicMemory(self._settings, embedding=embedding)
        self._profile = UserProfileStore(self._settings)
        self._decay_task: asyncio.Task | None = None
        logger.info("[MemoryManager] 初始化完成（Phase 4 三层记忆）")

    # ==================== MemoryRead（意图解析前调用）====================

    async def read(self, query: str, user_id: str, *, trace: TraceFn | None = None) -> MemoryContext:
        """读取记忆上下文：工作窗口 + 情景记忆 Top-K + 用户画像。

        任何子层失败都降级为空值，绝不阻塞主链路。

        trace: 可选链路追踪回调，记录工作窗口/情景记忆/用户画像的完整内容。
        """
        window = await self._working.read(user_id)
        # 链路追踪：工作记忆窗口（完整内容）
        if trace:
            win_lines = [
                f"[{i}] {t.get('role', '')}: {t.get('content', '')}"
                for i, t in enumerate(window, 1)
            ]
            trace(
                "memory/working_window",
                input=f"user_id={user_id}",
                output=f"共 {len(window)} 条工作记忆：\n" + ("\n".join(win_lines) if win_lines else "（空）"),
                notes=f"window_size={self._working._window_size}",
            )

        episodic_hits: list[dict] = []
        if query:
            try:
                episodic_hits = await self._episodic.search(query, user_id, k=_EPISODIC_TOP_K)
            except Exception as e:  # noqa: BLE001
                logger.warning("[MemoryManager] 情景记忆检索失败（降级为空）：%s", e)
        # 链路追踪：情景记忆检索（完整命中内容）
        if trace:
            epi_lines = []
            for i, h in enumerate(episodic_hits, 1):
                epi_lines.append(
                    f"[{i}] episode_id={h.get('episode_id', '')} "
                    f"relevance={h.get('relevance', '')}\n"
                    f"    topics={h.get('topics', [])} emotion={h.get('emotion_tag', '')}\n"
                    f"    summary: {h.get('summary', '')}"
                )
            trace(
                "memory/episodic_search",
                input=f"query={query}\nuser_id={user_id}\nTop-K={_EPISODIC_TOP_K}",
                output=f"共 {len(episodic_hits)} 条情景记忆命中：\n"
                       + ("\n\n".join(epi_lines) if epi_lines else "（无命中）"),
                notes="基于 query 向量语义检索 + 时间衰减加权（embedding，非 LLM）",
            )

        try:
            profile = self._profile.load(user_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("[MemoryManager] 用户画像加载失败（降级为空）：%s", e)
            profile = {}
        # 链路追踪：用户画像（完整内容）
        if trace:
            trace(
                "memory/user_profile",
                input=f"user_id={user_id}",
                output=json.dumps(profile, ensure_ascii=False, indent=2) if profile else "（空画像）",
                notes="用户画像（从 SQLite 加载，非 LLM）",
            )

        return MemoryContext(
            working_window=window,
            episodic_hits=episodic_hits,
            user_profile=profile,
        )

    # ==================== MemoryWrite（回应生成后异步调用）====================

    async def write(
        self,
        user_text: str,
        assistant_text: str,
        user_id: str,
        *,
        is_sensitive: bool = False,
    ) -> None:
        """写入一轮对话：同步更新工作记忆 + 异步提取情景/画像（M-08 不阻塞）。"""
        # 1) 工作记忆同步写入
        await self._working.write(user_id, "user", user_text)
        await self._working.write(user_id, "assistant", assistant_text)

        # 2) 异步提取（不阻塞调用方）
        asyncio.create_task(
            self._async_extract(user_text, assistant_text, user_id, is_sensitive)
        )

    async def _async_extract(
        self, user_text: str, assistant_text: str, user_id: str, is_sensitive: bool
    ) -> None:
        """LLM 一次性提取情景摘要 + 画像事实，分别落库（失败仅告警）。"""
        try:
            window = await self._working.read(user_id)
            # 排除本轮（窗口里已含本轮），只把历史作为上下文
            history = window[:-2] if len(window) >= 2 else []
            messages = build_memory_extract_messages(
                user_text, assistant_text,
                recent_window=history, is_sensitive=is_sensitive,
            )
            raw = await self._llm.chat(messages, json_mode=True, temperature=0.2, timeout=20.0)
            data = _extract_json(raw)
            if data is None:
                logger.warning("[MemoryManager] 记忆提取 JSON 解析失败，本轮跳过：%s", raw[:120])
                return
        except Exception as e:  # noqa: BLE001
            logger.warning("[MemoryManager] 记忆提取 LLM 调用失败（本轮跳过）：%s", e)
            return

        # 情景记忆
        episode = data.get("episode")
        if isinstance(episode, dict) and episode.get("summary"):
            try:
                await self._episodic.add(
                    user_id=user_id,
                    summary=str(episode["summary"]),
                    topics=[str(t) for t in (episode.get("topics") or [])][:8],
                    emotion_tag=episode.get("emotion_tag"),
                    is_sensitive=is_sensitive,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("[MemoryManager] 情景记忆写入失败：%s", e)

        # 用户画像
        facts = data.get("profile_facts")
        if isinstance(facts, dict) and facts:
            try:
                self._profile.merge_and_save(user_id, facts)
            except Exception as e:  # noqa: BLE001
                logger.warning("[MemoryManager] 用户画像 merge 失败：%s", e)

    # ==================== MemoryForget（用户主动遗忘）====================

    async def forget(self, user_id: str, scope: str, target: str | None = None) -> dict:
        """遗忘接口（对齐 MEMORY_SYSTEM_DESIGN §3.3）。

        - scope="episodic"：target 为 episode_id；
        - scope="profile"：target 为画像字段路径（如 "sleep_pattern"）；
        - scope="all_sensitive"：清除所有敏感情景记忆 + health_sensitive 标签。
        """
        result: dict[str, Any] = {"scope": scope, "user_id": user_id, "removed": 0}

        if scope == "episodic":
            if not target:
                raise ValueError('scope="episodic" 需要提供 episode_id')
            self._episodic.forget_episode(target)
            result["removed"] = 1
        elif scope == "profile":
            if not target:
                raise ValueError('scope="profile" 需要提供字段路径（如 sleep_pattern）')
            result["removed"] = 1 if self._profile.forget_field(user_id, target) else 0
        elif scope == "all_sensitive":
            result["removed"] = self._episodic.forget_all_sensitive(user_id)
            try:
                profile = self._profile.load(user_id)
                profile["health_sensitive"] = []
                self._profile.save(user_id, profile)
            except Exception as e:  # noqa: BLE001
                logger.warning("[MemoryManager] health_sensitive 清理失败：%s", e)
        else:
            raise ValueError(f"未知遗忘 scope：{scope}（应为 episodic / profile / all_sensitive）")

        logger.info("[MemoryManager] 遗忘完成：%s", result)
        return result

    # ==================== 周期清理（decay < 0.1）====================

    def start_decay_task(self) -> None:
        """启动每周衰减清理任务（在应用 lifespan 中调用）。"""
        if self._decay_task is not None and not self._decay_task.done():
            return
        self._decay_task = asyncio.create_task(self._decay_loop())

    async def stop_decay_task(self) -> None:
        if self._decay_task is not None:
            self._decay_task.cancel()
            try:
                await self._decay_task
            except asyncio.CancelledError:
                pass
            self._decay_task = None

    async def _decay_loop(self) -> None:
        """每周清理 decay_score < 0.1 的情景记忆（M-04）。"""
        try:
            while True:
                await asyncio.sleep(_DECAY_CLEANUP_INTERVAL)
                try:
                    self._episodic.cleanup_decayed(threshold=0.1)
                except Exception as e:  # noqa: BLE001
                    logger.warning("[MemoryManager] 衰减清理异常：%s", e)
        except asyncio.CancelledError:
            raise

    # ==================== 调试台状态支持 ====================

    def status(self) -> dict:
        """记忆系统状态摘要（调试台状态监测用）。"""
        try:
            with db_conn(self._settings) as conn:
                episode_count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
                profile_count = conn.execute("SELECT COUNT(*) FROM user_profiles").fetchone()[0]
        except Exception as e:  # noqa: BLE001
            logger.warning("[MemoryManager] 状态查询失败：%s", e)
            episode_count = profile_count = -1
        return {
            "working_users": len(self._working._buffers),
            "window_size": self._working._window_size,
            "episodes_total": episode_count,
            "profiles_total": profile_count,
            "decay_task_running": self._decay_task is not None and not self._decay_task.done(),
        }
