"""飞书适配器（继承 ChannelAdapter，协调 handler / adapter / client 三层）。

与 adapter.py（纯函数转换）分离，避免循环导入。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

import lark_oapi as lark

from app.channels.base import ChannelAdapter, Orchestrator
from app.channels.feishu import adapter as msg_adapter
from app.channels.feishu.client import FeishuClient
from app.channels.feishu.handler import EventDeduplicator, dispatch_event
from app.config import Settings, get_settings
from app.core.models import UnifiedRequest, UnifiedResponse

if TYPE_CHECKING:
    from lark_oapi.event.dispatcher_handler import EventDispatcherHandler  # noqa: F401

logger = logging.getLogger(__name__)


class FeishuAdapter(ChannelAdapter):
    """飞书渠道适配器（进程内单例）。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.dedup = EventDeduplicator(ttl_seconds=600)
        self.client = FeishuClient(self.settings)
        self._dispatcher: "EventDispatcherHandler | None" = None
        self._ws_client: lark.ws.Client | None = None
        self._orchestrator: Orchestrator | None = None
        self._loop: asyncio.AbstractEventLoop | None = None  # 主循环引用，线程间安全投递
        # 并发控制：每个 user_id 一个 semaphore，防止同用户消息并发改工作记忆
        self._user_semaphores: dict[str, asyncio.Semaphore] = {}
        self._global_lock = asyncio.Lock()

    # ==================== 启动 / 退出 ====================

    async def start(self, orchestrator: Orchestrator) -> None:
        # **关键**：捕获主事件循环引用，SDK WebSocket 内部线程回调时通过它安全投递任务
        self._loop = asyncio.get_running_loop()
        self._orchestrator = orchestrator
        self._dispatcher = self._build_dispatcher()

        if self.settings.feishu_mode == "websocket":
            await self._start_websocket()
        else:
            logger.info("[FeishuAdapter] webhook 模式：等待 FastAPI 路由 feed dispatcher")

    async def stop(self) -> None:
        if self._ws_client is not None:
            logger.info("[FeishuAdapter] 关闭 WebSocket 连接")
            self._ws_client = None
        self._loop = None

    # ==================== 消息入口 ====================

    async def handle_raw_event(self, raw_event: Any) -> None:
        extracted = await dispatch_event(raw_event, self.dedup)
        if extracted is None:
            return
        await self._handle_extracted(extracted)

    def _build_dispatcher(self) -> "EventDispatcherHandler":
        def _on_message_receive(event, context):
            """SDK 回调（运行在 lark-oapi 内部线程，必须线程安全地投递到主循环）。"""
            assert self._loop is not None, "start() 必须先于 SDK 回调被调用"
            # call_soon_threadsafe 是 Python 标准库中从非 asyncio 线程投递任务的唯一正确做法
            self._loop.call_soon_threadsafe(
                asyncio.ensure_future,
                self._sdk_callback(event),
            )

        builder = lark.EventDispatcherHandler.builder(
            self.settings.feishu_verification_token, ""
        ).register_p2_im_message_receive_v1(_on_message_receive)
        return builder.build()

    async def _sdk_callback(self, event: Any) -> None:
        extracted = await dispatch_event(event, self.dedup)
        if extracted is None:
            return
        await self._handle_extracted(extracted)

    # ==================== 主链路 ====================

    async def _handle_extracted(self, extracted) -> None:
        user_id = msg_adapter.build_user_id(extracted.chat_id, extracted.open_id)
        async with await self._get_user_semaphore(user_id):
            msg_adapter.upsert_chat_session(extracted)

            req = msg_adapter.extracted_to_request(extracted)
            logger.info("[FeishuAdapter] UnifiedRequest: user_id=%s text=%s", req.user_id, req.text[:60])

            assert self._orchestrator is not None, "orchestrator 未设置"
            t0 = time.monotonic()
            try:
                resp = await self._orchestrator(req)
            except Exception as e:  # noqa: BLE001
                logger.exception("[FeishuAdapter] 编排器抛异常")
                resp = UnifiedResponse(user_id=req.user_id, text=f"（内部错误：{type(e).__name__}）")
            latency = (time.monotonic() - t0) * 1000
            logger.info("[FeishuAdapter] 编排耗时 %.0f ms user_id=%s", latency, req.user_id)

            feishu_msg = msg_adapter.response_to_feishu_messages(resp, reply_to_message_id=extracted.message_id)
            sent = await self.client.send_feishu_message(feishu_msg, prefer_reply=True)
            if not sent:
                logger.warning(
                    "[FeishuAdapter] 消息发送失败：chat_id=%s message_id=%s",
                    extracted.chat_id, extracted.message_id,
                )

    async def _get_user_semaphore(self, user_id: str) -> asyncio.Semaphore:
        async with self._global_lock:
            if user_id not in self._user_semaphores:
                self._user_semaphores[user_id] = asyncio.Semaphore(1)
        return self._user_semaphores[user_id]

    # ==================== 主动推送 ====================

    async def send_response(self, resp: UnifiedResponse, *, reply_in_thread: bool = False) -> bool:
        feishu_msg = msg_adapter.response_to_feishu_messages(resp)
        return await self.client.send_feishu_message(feishu_msg)

    # ==================== WebSocket ====================

    async def _start_websocket(self) -> None:
        self._ws_client = lark.ws.Client(
            app_id=self.settings.feishu_app_id,
            app_secret=self.settings.feishu_app_secret,
            event_handler=self._dispatcher,
            log_level=lark.LogLevel.INFO,
        )
        logger.info("[FeishuAdapter] 启动 WebSocket 长连（app_id=%s）", self.settings.feishu_app_id)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._ws_client.start)
