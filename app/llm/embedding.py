"""Embedding 客户端：doubao-embedding-vision 向量化入口。

对齐 docs/IMPLEMENTATION_DESIGN.md §6.2。
- 与 LLM 共用 ARK_BASE_URL + ARK_KEY，仅模型名不同（PRD §7.2 共享 Endpoint）；
- 批量向量化自动分批，避免单次请求文本过多；
- 瞬时错误指数退避重试。
"""

from __future__ import annotations

import asyncio
import logging

import openai

from app.config import Settings, get_settings
from app.llm.client import LLMConfigError, _RETRYABLE_ERRORS

logger = logging.getLogger(__name__)

# 单批最大文本数（保守取值；OpenAI 兼容 Embedding 接口单次批量不宜过大）
_DEFAULT_BATCH_SIZE = 10

# ==================== 进程级自适应限速器 ====================
# ARK 端点对账户级突发较敏感（AccountRateLimitExceeded）：
# 并发 + 重试风暴会互相挤压令牌桶，实测"串行 + 2s 间隔"可稳定 ~30 req/min。
# 策略：请求前等到下一个允许时刻；429 时放大间隔；连续成功时收敛到最小间隔。

_PACE_MIN_GAP = 1.0    # 最小请求间隔（秒）
_PACE_MAX_GAP = 30.0   # 429 后的最大间隔（秒）
_PACE_GROW = 1.6       # 429 间隔放大系数
_PACE_SHRINK = 0.95    # 连续成功间隔收敛系数

_pace_lock = asyncio.Lock()
_pace_next = 0.0   # 下一个允许请求的时刻（loop time）
_pace_gap = 1.0    # 当前请求间隔（秒）


async def _pace_wait() -> None:
    """请求前调用：等到下一个允许时刻并占位。"""
    global _pace_next
    async with _pace_lock:
        loop = asyncio.get_running_loop()
        now = loop.time()
        if now < _pace_next:
            await asyncio.sleep(_pace_next - now)
            now = loop.time()
        _pace_next = now + _pace_gap


def _pace_on_ok() -> None:
    """请求成功：间隔缓慢收敛。"""
    global _pace_gap
    _pace_gap = max(_pace_gap * _PACE_SHRINK, _PACE_MIN_GAP)


def _pace_on_rate_limit() -> None:
    """请求 429：间隔放大（由重试循环在下一次请求前生效）。"""
    global _pace_gap
    _pace_gap = min(_pace_gap * _PACE_GROW, _PACE_MAX_GAP)


class EmbeddingClient:
    """doubao-embedding-vision 调用封装（OpenAI 兼容 embeddings 端点）。"""

    def __init__(self, settings: Settings | None = None, batch_size: int = _DEFAULT_BATCH_SIZE) -> None:
        self.settings = settings or get_settings()
        self._model = self.settings.embedding_model
        self._batch_size = batch_size
        self._client: openai.AsyncOpenAI | None = None

    def _get_client(self) -> openai.AsyncOpenAI:
        if not self.settings.ark_key or not self.settings.ark_base_url:
            raise LLMConfigError(
                "ARK_KEY / ARK_BASE_URL 未配置，无法调用 Embedding。"
                "请在环境变量或 .env 中配置（参考 .env.example）。"
            )
        if self._client is None:
            self._client = openai.AsyncOpenAI(
                api_key=self.settings.ark_key,
                base_url=self.settings.ark_base_url,
            )
        return self._client

    async def _embed_batch(self, texts: list[str], *, max_retries: int = 5) -> list[list[float]]:
        """调用一次 embeddings 接口（进程级节奏控制 + 重试）。"""
        client = self._get_client()
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                await _pace_wait()
                resp = await client.embeddings.create(model=self._model, input=texts)
                # 按 index 排序保证顺序与输入一致
                data = sorted(resp.data, key=lambda d: d.index)
                _pace_on_ok()
                return [item.embedding for item in data]
            except _RETRYABLE_ERRORS as e:
                last_exc = e
                if isinstance(e, openai.RateLimitError):
                    _pace_on_rate_limit()
                if attempt < max_retries:
                    backoff = min(2 ** attempt, 30)
                    logger.warning(
                        "Embedding 调用瞬时错误（第 %d/%d 次），%.0fs 后重试：%s",
                        attempt, max_retries, backoff, e,
                    )
                    await asyncio.sleep(backoff)
                else:
                    logger.error("Embedding 调用重试 %d 次后仍失败：%s", max_retries, e)
            except openai.BadRequestError as e:
                logger.error("Embedding 请求参数错误（不重试）：%s", e)
                raise
            except openai.AuthenticationError as e:
                logger.error("ARK_KEY 鉴权失败，请检查 ARK_KEY/ARK_BASE_URL：%s", e)
                raise
        assert last_exc is not None
        raise last_exc

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量文本向量化（自动分批，顺序与输入一致）。"""
        if not texts:
            return []
        results: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors = await self._embed_batch(batch)
            results.extend(vectors)
            logger.debug("Embedding 批次完成：%d/%d 条", min(start + self._batch_size, len(texts)), len(texts))
        return results

    async def embed_query(self, text: str) -> list[float]:
        """单条查询向量化。"""
        vectors = await self.embed_texts([text])
        return vectors[0]
