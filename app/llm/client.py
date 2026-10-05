"""LLM 客户端：deepseek-v4-pro 统一调用入口。

对齐 docs/IMPLEMENTATION_DESIGN.md §6.2。
- 经火山方舟 OpenAI 兼容端点调用；
- 支持 JSON 输出模式（意图/置信度/Rerank 等结构化场景）；
- 瞬时错误（限流/连接/超时/5xx）指数退避重试，最多 3 次；
  参数错误/鉴权错误不重试，直接抛出便于定位。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import openai

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# 可重试的瞬时错误类型
_RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


class LLMConfigError(RuntimeError):
    """ARK_KEY / ARK_BASE_URL 未配置时抛出（启动阶段已告警，此处防止静默失败）。"""


class LLMClient:
    """deepseek-v4-pro 调用封装（OpenAI 兼容）。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model = self.settings.llm_model
        self._client: openai.AsyncOpenAI | None = None

    def _get_client(self) -> openai.AsyncOpenAI:
        if not self.settings.ark_key or not self.settings.ark_base_url:
            raise LLMConfigError(
                "ARK_KEY / ARK_BASE_URL 未配置，无法调用 LLM。"
                "请在环境变量或 .env 中配置（参考 .env.example）。"
            )
        if self._client is None:
            self._client = openai.AsyncOpenAI(
                api_key=self.settings.ark_key,
                base_url=self.settings.ark_base_url,
            )
        return self._client

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_mode: bool = False,
        timeout: float = 10.0,
        max_retries: int = 3,
        sdk_max_retries: int | None = None,
    ) -> str:
        """调用 chat completions，返回助手消息文本。

        Args:
            messages: OpenAI 风格消息列表 [{"role": ..., "content": ...}]
            temperature: 采样温度
            max_tokens: 最大生成 token 数
            json_mode: 是否强制 JSON 输出（response_format=json_object）
            timeout: 单次请求超时（秒）
            max_retries: 瞬时错误最大重试次数（指数退避，本方法外层）
            sdk_max_retries: openai SDK 内部重试次数；None 时用 SDK 默认（2）。
                批处理脚本可传 0 以完全接管重试决策。
        """
        client = self._get_client()
        # SDK 内部重试次数是 client 级参数（create() 不接受），用 with_options 派生
        # 浅拷贝客户端（共享连接池），不影响其他调用方的默认重试设置。
        if sdk_max_retries is not None:
            client = client.with_options(max_retries=sdk_max_retries)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "timeout": timeout,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = await client.chat.completions.create(**kwargs)
                choice = resp.choices[0]
                content = choice.message.content
                if content is None:
                    raise RuntimeError("LLM 返回内容为空（choices[0].message.content is None）")
                if not content.strip():
                    # 推理模型在 max_tokens 被思维链耗尽时常见：finish_reason=length 且 content 为空串
                    logger.warning(
                        "LLM 返回内容为空串（finish_reason=%s, model=%s）；"
                        "若频繁出现请调大 max_tokens",
                        getattr(choice, "finish_reason", "?"), self._model,
                    )
                return content.strip()
            except _RETRYABLE_ERRORS as e:
                last_exc = e
                if attempt < max_retries:
                    backoff = min(2 ** (attempt - 1), 8)  # 1s, 2s, 4s...
                    logger.warning(
                        "LLM 调用瞬时错误（第 %d/%d 次），%.0fs 后重试：%s",
                        attempt, max_retries, backoff, e,
                    )
                    await asyncio.sleep(backoff)
                else:
                    logger.error("LLM 调用重试 %d 次后仍失败：%s", max_retries, e)
            except openai.BadRequestError as e:
                # 参数/模型问题不重试
                logger.error("LLM 请求参数错误（不重试）：%s", e)
                raise
            except openai.AuthenticationError as e:
                logger.error("ARK_KEY 鉴权失败，请检查 ARK_KEY/ARK_BASE_URL：%s", e)
                raise
        assert last_exc is not None
        raise last_exc
