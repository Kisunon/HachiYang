"""情景记忆（Episodic Memory）—— Phase 4 任务 4.2。

对齐 docs/tech_docs/MEMORY_SYSTEM_DESIGN.md §2.2 / §3.3：
- 存储：SQLite episodes 表（元数据）+ ChromaDB episodic_memory Collection（向量）；
- 一个 Episode = 一轮（或少数几轮）对话的 LLM 提取摘要（非原文）；
- 向量化复用 doubao-embedding-vision（M-09）；
- 检索：余弦相似度 Top-K + 时间衰减加权（final = cosine × decay）；
- 衰减：decay(t) = exp(-λ·Δt)，命中重置为 1.0；定期清理 decay < 0.1（每周）；
- 多用户隔离：检索强制 user_id 过滤（M-02）；
- 敏感对话：LLM 提取时脱敏，条目带 is_sensitive 标记，支持一键遗忘（M-06/M-07）。

LLM 提取本身由 MemoryManager 统一调用（一次调用同时产出情景与画像事实），
本模块只负责存储、检索、衰减与遗忘。
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import chromadb.api.models.Collection as ChromaCollection

from app.config import Settings, get_settings
from app.db.connection import db_conn, get_chroma_client

logger = logging.getLogger(__name__)

# ChromaDB episodic_memory Collection 名称（connection.py 已定义）
COLLECTION_NAME = "episodic_memory"


@dataclass
class Episode:
    """情景记忆条目（对齐 MEMORY_SYSTEM_DESIGN §2.2 Schema）。"""

    episode_id: str
    user_id: str
    timestamp: int
    summary: str
    topics: list[str] = field(default_factory=list)
    emotion_tag: str | None = None
    source_window: list[str] = field(default_factory=list)
    decay_score: float = 1.0
    last_hit_at: int | None = None
    is_sensitive: bool = False


def decay_score_at(decay_score: float, last_ref_ts: int, *, lam: float, now_ts: int | None = None) -> float:
    """按设计公式计算当前衰减分：decay(t) = last_hit_score × exp(-λ·Δt)。

    Δt 以天为单位。λ=0.005 时约 139 天衰减至 0.5（设计文档注释"14 天"
    与公式参数不一致，此处以公式与 config 默认值为准——优先级"降低"即可满足 M-04）。
    """
    now_ts = now_ts if now_ts is not None else int(time.time())
    dt_days = max(0.0, (now_ts - last_ref_ts)) / 86400.0
    return decay_score * math.exp(-lam * dt_days)


class EpisodicMemory:
    """情景记忆存储与检索。"""

    def __init__(self, settings: Settings | None = None, embedding: Any = None) -> None:
        self._settings = settings or get_settings()
        self._embedding = embedding  # app.llm.embedding.EmbeddingClient（测试可注入 fake）
        self._col: ChromaCollection | None = None
        self._ensure_schema()

    # ==================== Schema 迁移 ====================

    def _ensure_schema(self) -> None:
        """episodes 表增加 is_sensitive 列（幂等迁移：schema.sql 的 CREATE IF NOT EXISTS
        不会给已存在的表加列）。"""
        try:
            with db_conn(self._settings) as conn:
                conn.execute(
                    "ALTER TABLE episodes ADD COLUMN is_sensitive INTEGER NOT NULL DEFAULT 0"
                )
                logger.info("[EpisodicMemory] episodes 表迁移：新增 is_sensitive 列")
        except Exception:
            pass  # 列已存在或表结构一致

    def _get_collection(self) -> ChromaCollection:
        if self._col is None:
            client = get_chroma_client(self._settings)
            self._col = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine", "purpose": "情景记忆摘要"},
            )
        return self._col

    # ==================== 写入 ====================

    async def add(
        self,
        *,
        user_id: str,
        summary: str,
        topics: list[str] | None = None,
        emotion_tag: str | None = None,
        source_window: list[str] | None = None,
        is_sensitive: bool = False,
        timestamp: int | None = None,
    ) -> Episode:
        """写入一条情景记忆：SQLite 元数据 + ChromaDB 向量（embedding of summary）。"""
        episode = Episode(
            episode_id=uuid.uuid4().hex,
            user_id=user_id,
            timestamp=timestamp or int(time.time()),
            summary=summary,
            topics=topics or [],
            emotion_tag=emotion_tag,
            source_window=source_window or [],
            is_sensitive=is_sensitive,
        )

        with db_conn(self._settings) as conn:
            conn.execute(
                """
                INSERT INTO episodes
                    (episode_id, user_id, timestamp, summary, topics, emotion_tag,
                     source_window, decay_score, last_hit_at, is_sensitive)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode.episode_id,
                    episode.user_id,
                    episode.timestamp,
                    episode.summary,
                    json.dumps(episode.topics, ensure_ascii=False),
                    episode.emotion_tag,
                    json.dumps(episode.source_window, ensure_ascii=False),
                    episode.decay_score,
                    episode.last_hit_at,
                    1 if episode.is_sensitive else 0,
                ),
            )

        # 向量写入（失败不阻塞元数据：检索退化由 query 的向量失败路径兜底）
        try:
            col = self._get_collection()
            vector = await self._embed_text(summary)
            col.upsert(
                ids=[episode.episode_id],
                embeddings=[vector],
                documents=[episode.summary],
                metadatas=[{
                    "user_id": episode.user_id,
                    "timestamp": episode.timestamp,
                    "topics": json.dumps(episode.topics, ensure_ascii=False),
                    "emotion_tag": episode.emotion_tag or "",
                    "is_sensitive": 1 if episode.is_sensitive else 0,
                }],
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("[EpisodicMemory] 向量写入失败（元数据已存，检索暂不可命中）：%s", e)

        logger.info("[EpisodicMemory] 写入 episode user=%s sensitive=%s：%s",
                    user_id, is_sensitive, summary[:40])
        return episode

    # ==================== 检索 ====================

    async def search(self, query: str, user_id: str, *, k: int = 5) -> list[dict]:
        """按当前查询做语义检索，返回衰减加权 Top-K（强制 user_id 隔离）。

        返回 [{"summary", "topics", "emotion_tag", "relevance", "timestamp", "episode_id"}]。
        """
        try:
            col = self._get_collection()
            query_vector = await self._embed_text(query)
            res = col.query(
                query_embeddings=[query_vector],
                n_results=min(k, max(1, col.count())),
                where={"user_id": user_id},
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("[EpisodicMemory] 向量检索失败（返回空）：%s", e)
            return []

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        if not ids:
            return []

        lam = self._settings.memory_decay_lambda
        now_ts = int(time.time())
        hits: list[dict] = []
        for eid, doc, meta, dist in zip(ids, docs, metas, dists):
            cosine = max(0.0, 1.0 - float(dist))  # cosine space: distance = 1 - cos
            # SQLite 行可能已被遗忘（向量为残留）：行缺失则跳过
            row = self._get_episode_row(eid)
            if row is None:
                self._delete_vectors([eid])
                continue
            last_ref = row["last_hit_at"] or row["timestamp"]
            decay = decay_score_at(row["decay_score"], last_ref, lam=lam, now_ts=now_ts)
            if cosine <= 0.0:
                # 正交（完全不相关）向量也占 Top-K 名额（Chroma 无条件返回 k 条），跳过
                continue
            hits.append({
                "episode_id": eid,
                "summary": row["summary"],
                "topics": json.loads(row["topics"] or "[]"),
                "emotion_tag": row["emotion_tag"],
                "timestamp": row["timestamp"],
                "relevance": round(cosine * decay, 4),
            })
            # 命中重置：decay_score=1.0、last_hit_at=now（被反复讨论的事件不被遗忘）
            self._reset_decay(eid)

        hits.sort(key=lambda h: h["relevance"], reverse=True)
        hits = hits[:k]
        logger.info("[EpisodicMemory] 检索 user=%s query=%r 命中 %d 条", user_id, query[:30], len(hits))
        return hits

    # ==================== 遗忘 / 清理 ====================

    def forget_episode(self, episode_id: str) -> None:
        """删除指定情景记忆（M-07）。"""
        with db_conn(self._settings) as conn:
            conn.execute("DELETE FROM episodes WHERE episode_id = ?", (episode_id,))
        self._delete_vectors([episode_id])
        logger.info("[EpisodicMemory] 已遗忘 episode=%s", episode_id)

    def forget_all_sensitive(self, user_id: str) -> int:
        """一键遗忘该用户所有敏感情景记忆（M-06/M-07，scope=all_sensitive）。"""
        with db_conn(self._settings) as conn:
            rows = conn.execute(
                "SELECT episode_id FROM episodes WHERE user_id = ? AND is_sensitive = 1",
                (user_id,),
            ).fetchall()
            ids = [r["episode_id"] for r in rows]
            if ids:
                conn.executemany(
                    "DELETE FROM episodes WHERE episode_id = ?",
                    [(i,) for i in ids],
                )
        self._delete_vectors(ids)
        logger.info("[EpisodicMemory] 已遗忘 user=%s 的 %d 条敏感情景", user_id, len(ids))
        return len(ids)

    def cleanup_decayed(self, threshold: float = 0.1) -> int:
        """清理衰减至阈值以下的条目（设计建议每周执行，decay < 0.1 清除）。"""
        lam = self._settings.memory_decay_lambda
        now_ts = int(time.time())
        removed = 0
        with db_conn(self._settings) as conn:
            rows = conn.execute("SELECT episode_id, decay_score, timestamp, last_hit_at FROM episodes").fetchall()
            stale = [
                r["episode_id"] for r in rows
                if decay_score_at(r["decay_score"], r["last_hit_at"] or r["timestamp"], lam=lam, now_ts=now_ts) < threshold
            ]
            if stale:
                conn.executemany(
                    "DELETE FROM episodes WHERE episode_id = ?",
                    [(i,) for i in stale],
                )
                removed = len(stale)
        if removed:
            self._delete_vectors(stale)
        logger.info("[EpisodicMemory] 周期清理：移除 %d 条衰减条目", removed)
        return removed

    # ==================== 内部工具 ====================

    async def _embed_text(self, text: str) -> list[float]:
        if self._embedding is None:
            raise RuntimeError("EpisodicMemory 未配置 Embedding 客户端")
        return await self._embedding.embed_query(text)

    def _get_episode_row(self, episode_id: str) -> dict | None:
        with db_conn(self._settings) as conn:
            row = conn.execute(
                "SELECT episode_id, user_id, timestamp, summary, topics, emotion_tag, "
                "decay_score, last_hit_at, is_sensitive FROM episodes WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()
        return dict(row) if row else None

    def _reset_decay(self, episode_id: str) -> None:
        try:
            with db_conn(self._settings) as conn:
                conn.execute(
                    "UPDATE episodes SET decay_score = 1.0, last_hit_at = ? WHERE episode_id = ?",
                    (int(time.time()), episode_id),
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("[EpisodicMemory] decay 重置失败 episode=%s：%s", episode_id, e)

    def _delete_vectors(self, episode_ids: list[str]) -> None:
        if not episode_ids:
            return
        try:
            self._get_collection().delete(ids=episode_ids)
        except Exception as e:  # noqa: BLE001
            logger.warning("[EpisodicMemory] 向量删除失败（%d 条）：%s", len(episode_ids), e)
