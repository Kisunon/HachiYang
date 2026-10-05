"""飞书 SDK 封装：消息发送（SDK 自动管理 tenant_access_token）。

不自己缓存 token——lark-oapi 内部已经处理好 tenant_access_token 的获取/续期，
我们只需要用 ``lark.Client.builder()`` 构造一次 API client，然后调用高层接口。

暴露两个发送路径：
- send_text_message：主动推送（关怀调度用，直接 chat_id）
- reply_text_message：回复用户消息（Phase 1 内主要走这条，避免在群里 @ 错人）

两个路径都按 FeishuMessage 里的分片数逐条发送，失败指数退避重试（最多 3 次）。
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import lark_oapi as lark
from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody
from lark_oapi.api.im.v1 import ReplyMessageRequest, ReplyMessageRequestBody

from app.config import Settings, get_settings
from app.channels.feishu.adapter import FeishuMessage

if TYPE_CHECKING:
    from lark_oapi.client import Client  # noqa: F401

logger = logging.getLogger(__name__)

# 瞬时错误重试上限
_MAX_RETRIES = 3


class FeishuClient:
    """飞书消息发送客户端（封装 lark-oapi API Client）。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._api_client: "lark.Client" | None = None

    def _get_client(self) -> "lark.Client":
        if not self.settings.feishu_app_id or not self.settings.feishu_app_secret:
            raise RuntimeError(
                "FEISHU_APP_ID / FEISHU_APP_SECRET 未配置，无法调用飞书接口。"
                "请在环境变量或 .env 中配置。"
            )
        if self._api_client is None:
            self._api_client = (
                lark.Client.builder()
                .app_id(self.settings.feishu_app_id)
                .app_secret(self.settings.feishu_app_secret)
                .log_level(lark.LogLevel.WARNING)
                .build()
            )
            logger.info("[FeishuClient] lark.Client 已构造")
        return self._api_client

    # ==================== 发送 ====================

    async def send_text_message(
        self,
        chat_id: str,
        *,
        msg_type: str = "text",
        content: str,
    ) -> bool:
        """主动推送：按 chat_id 发送（关怀调度用）。"""
        body = CreateMessageRequestBody.builder().chat_id(chat_id).msg_type(msg_type).content(content).build()
        req = CreateMessageRequest.builder().request_body(body).build()
        return await self._call_with_retry(
            lambda: self._get_client().im.v1.message.create(req),
            label=f"send_text chat={chat_id}",
        )

    async def reply_text_message(
        self,
        message_id: str,
        *,
        msg_type: str = "text",
        content: str,
    ) -> bool:
        """回复某条消息：消息会挂在原消息的 thread 下，群聊里不会产生额外干扰。"""
        body = ReplyMessageRequestBody.builder().msg_type(msg_type).content(content).build()
        req = ReplyMessageRequest.builder().message_id(message_id).request_body(body).build()
        return await self._call_with_retry(
            lambda: self._get_client().im.v1.message.reply(req),
            label=f"reply_text msg={message_id}",
        )

    async def send_feishu_message(
        self,
        msg: FeishuMessage,
        *,
        prefer_reply: bool = True,
    ) -> bool:
        """按 FeishuMessage 内部分片自动发送，优先用 reply 回到原消息线程。"""
        ok = True
        for i, content in enumerate(msg.contents):
            if prefer_reply and msg.reply_to_message_id:
                sent = await self.reply_text_message(
                    msg.reply_to_message_id,
                    msg_type=msg.msg_type,
                    content=content,
                )
            else:
                sent = await self.send_text_message(
                    msg.chat_id,
                    msg_type=msg.msg_type,
                    content=content,
                )
            if not sent:
                ok = False
                break
            if i < len(msg.contents) - 1:
                await asyncio.sleep(0.2)  # 分片之间轻微间隔，避免飞书限流
        return ok

    # ==================== 内部：重试 ====================

    async def _call_with_retry(self, fn, *, label: str) -> bool:
        """同步调用 SDK 方法（lark-oapi 1.7.x 的 .create/.reply 是同步的），失败重试。"""
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = fn()
                # lark-oapi 的同步方法返回带 code/msg/data 的对象
                code = getattr(resp, "code", None)
                if code == 0:
                    return True
                msg = getattr(resp, "msg", "")
                logger.warning(
                    "[FeishuClient] API 返回非零 code=%s msg=%s (%s)，第 %d/%d 次",
                    code, msg, label, attempt, _MAX_RETRIES,
                )
                last_exc = RuntimeError(f"feishu code={code} msg={msg}")
            except Exception as e:  # noqa: BLE001
                last_exc = e
                logger.warning(
                    "[FeishuClient] 调用异常 (%s) 第 %d/%d 次：%s",
                    label, attempt, _MAX_RETRIES, e,
                )
            if attempt < _MAX_RETRIES:
                backoff = min(2 ** (attempt - 1), 4)
                await asyncio.sleep(backoff)
        logger.error("[FeishuClient] %s 连续 %d 次失败", label, _MAX_RETRIES)
        return False
