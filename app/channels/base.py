"""渠道适配层抽象基类。

对齐 docs/IMPLEMENTATION_DESIGN.md §3.1 / §6.3：
- 渠道层的职责是**传输层隔离**：把飞书等不同 IM 渠道的收发逻辑封在适配层内部，
  向上只暴露统一的 handle_request / send_response 接口；
- 飞书内部再拆成 adapter（消息格式转换）、handler（事件分发/去重）、client（SDK 封装）
  三层，共用本抽象基类定义的契约。
"""

from __future__ import annotations

import abc
import logging
from typing import Awaitable, Callable

from app.core.models import UnifiedRequest, UnifiedResponse

logger = logging.getLogger(__name__)

# 编排器签名：接收统一请求、返回统一回应
Orchestrator = Callable[[UnifiedRequest], Awaitable[UnifiedResponse]]


class ChannelAdapter(abc.ABC):
    """渠道适配器抽象基类。

    子类实现：
    - start(orchestrator)：启动渠道监听（WebSocket 长连或 Webhook HTTP 回调）
    - stop()：优雅退出（释放连接/关闭端口）
    - send_response(resp)：主动下发消息（回应/关怀推送）
    """

    @abc.abstractmethod
    async def start(self, orchestrator: Orchestrator) -> None:
        """启动渠道，将收到的用户消息交给 orchestrator 处理。"""

    @abc.abstractmethod
    async def stop(self) -> None:
        """优雅退出。"""

    @abc.abstractmethod
    async def send_response(self, resp: UnifiedResponse, *, reply_in_thread: bool = False) -> bool:
        """主动下发消息（回应 / 关怀推送）。

        Returns:
            是否成功；飞书 SDK 可能抛异常，调用方需要自行决定是否降级。
        """
