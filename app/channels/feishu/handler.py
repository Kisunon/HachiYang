"""飞书事件处理层（传输层透明）。

职责：事件去重（event_id + 内存 TTL）、字段安全提取、忽略自发言/系统消息、过滤非文本消息。

纯函数 + 一个轻量的 EventDeduplicator 类，可离线单元测试。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ==================== 数据类 ====================

@dataclass(frozen=True)
class ExtractedEvent:
    """从飞书事件中提取的统一字段。"""
    event_id: str
    message_id: str
    chat_id: str
    open_id: str
    chat_type: str        # "p2p" | "group"
    message_type: str     # "text" | "image" | "system" ...
    content_json: str     # 原始 content JSON 字符串（飞书文本消息内容是 {"text":"..."} JSON）
    mentions: list[dict]   # 群聊中的 @ 提及列表


# ==================== 事件去重 ====================

class EventDeduplicator:
    """基于 event_id 的事件去重（内存 + TTL，单实例够用）。

    - 用 OrderedDict 按时间顺序存储收到的 event_id，天然 FIFO；
    - 每次 add 时先清理过期项（O(过期数)，远小于总数）；
    - 超量时主动截断最老的，避免内存泄漏。
    """

    def __init__(self, ttl_seconds: int = 600, max_size: int = 10000) -> None:
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._store: OrderedDict[str, float] = OrderedDict()  # event_id -> timestamp
        self._lock = asyncio.Lock()

    async def seen(self, event_id: str) -> bool:
        """如果 event_id 已在 TTL 窗口内返回 True（重复），否则返回 False（首次）。"""
        async with self._lock:
            now = time.monotonic()
            # 清理过期项
            while self._store:
                oldest_ts = next(iter(self._store.values()))
                if now - oldest_ts > self._ttl:
                    self._store.popitem(last=False)
                else:
                    break
            # 检查
            if event_id in self._store:
                return True
            self._store[event_id] = now
            # 超量截断
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)
            return False

    def size(self) -> int:
        return len(self._store)


# ==================== 事件提取（纯函数） ====================

def _getattr(obj: Any, name: str, default: Any = None) -> Any:
    """安全 getattr：obj 可能是 Lark SDK 的 Builder/Model 实例，也可能是 dict。"""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def extract_event_fields(event: Any) -> ExtractedEvent | None:
    """从飞书消息接收事件对象中提取统一字段。

    入参 ``event`` 是 lark-oapi SDK 反序列化后的 P2ImMessageReceiveV1 事件对象
    （也兼容手动传入的 dict，便于 Webhook 模式共用）。

    Returns:
        提取成功返回 ExtractedEvent；自发言/忽略类型/字段缺失返回 None。
    """
    # 兼容两种输入：SDK 对象（带 data 属性）或 dict（Webhook 解密后的 JSON）
    if hasattr(event, "data"):
        data = event.data
    elif isinstance(event, dict):
        data = event.get("event") or event.get("data")
    else:
        logger.warning("未知事件类型：%s", type(event))
        return None

    message = _getattr(data, "message")
    sender = _getattr(data, "sender")
    if message is None or sender is None:
        logger.debug("事件缺 message 或 sender，跳过")
        return None

    chat_id = _getattr(message, "chat_id")
    message_id = _getattr(message, "message_id")
    chat_type = _getattr(message, "chat_type", "p2p")
    message_type = _getattr(message, "message_type", "text")
    content_json = _getattr(message, "content", "")
    mentions = _getattr(message, "mentions") or []

    # sender 相关
    sender_type = _getattr(sender, "sender_type", "user")
    sender_id = _getattr(sender, "sender_id")
    open_id = _getattr(sender_id, "open_id") if sender_id else None

    # 忽略自己发出的消息（sender_type == "app"）
    if sender_type == "app":
        logger.debug("忽略自发言（sender_type=app）")
        return None

    if not chat_id or not open_id or not message_id:
        logger.warning("事件缺关键字段：chat_id=%s open_id=%s message_id=%s", chat_id, open_id, message_id)
        return None

    # 只处理文本消息（Phase 1 范围）
    if message_type != "text":
        logger.info("跳过非文本消息：message_type=%s chat_id=%s", message_type, chat_id)
        return None

    # event_id：从 SDK 对象 header 或 dict header 中取
    event_id = None
    if hasattr(event, "header"):
        event_id = _getattr(event.header, "event_id")
    elif isinstance(event, dict):
        event_id = (event.get("header") or {}).get("event_id")
    event_id = event_id or message_id  # 兜底

    return ExtractedEvent(
        event_id=str(event_id),
        message_id=str(message_id),
        chat_id=str(chat_id),
        open_id=str(open_id),
        chat_type=str(chat_type),
        message_type=str(message_type),
        content_json=str(content_json),
        mentions=list(mentions) if mentions else [],
    )


def parse_text_content(content_json: str) -> str:
    """从飞书 text 消息的 content JSON 中提取纯文本，去掉 @机器人 标记。

    飞书文本消息格式：{"text": "@_user_1 你好"}，其中 @_user_N 是 mentions 占位符。
    这里简单把 @_user_N 去掉，返回干净文本；复杂的 mention 解析由调用方按需处理。
    """
    try:
        obj = json.loads(content_json)
    except (json.JSONDecodeError, TypeError):
        logger.debug("content 不是合法 JSON：%s", content_json[:80])
        return content_json.strip()

    text = obj.get("text", "")
    # 去掉 @_user_N 占位符（飞书 mentions 的 key 格式）
    import re
    text = re.sub(r"@_user_\d+\s*", "", text).strip()
    return text


def is_group_chat(evt: ExtractedEvent) -> bool:
    return evt.chat_type == "group"


def is_at_bot(evt: ExtractedEvent) -> bool:
    """群聊中是否 @ 了机器人。

    SDK 的 mentions 列表在 dict/object 两种输入下格式可能不同，做兼容处理。
    """
    if not is_group_chat(evt):
        return True  # p2p 单聊始终触发
    for m in evt.mentions:
        if isinstance(m, dict):
            # SDK 内部 mention 对象通常带有 id.open_id；但也可能只有 key（@_user_N）
            # 机器人在 mentions 中会带有 "mention" 字段或 open_id 指向机器人 open_id；
            # 简化处理：只要有任何 mention 就算用户 @ 了什么，我们在 adapter 里再做精确判断
            if m.get("key") or m.get("id") or m.get("mention"):
                return True
        else:
            # SDK 对象，用 getattr
            if _getattr(m, "key") or _getattr(m, "id"):
                return True
    return False


# ==================== 高层入口 ====================

async def dispatch_event(
    event: Any,
    dedup: EventDeduplicator,
) -> ExtractedEvent | None:
    """事件处理管线：去重 → 字段提取 → 群聊 @ 判断。

    WebSocket/Webhook 两种模式都走这一条。
    返回 None 表示该事件已被过滤（重复 / 自发言 / 非文本 / 群聊未 @）。
    """
    extracted = extract_event_fields(event)
    if extracted is None:
        return None

    if await dedup.seen(extracted.event_id):
        logger.info("事件去重：event_id=%s message_id=%s", extracted.event_id, extracted.message_id)
        return None

    if not is_at_bot(extracted):
        logger.info("群聊未 @ 机器人，跳过：chat_id=%s message_id=%s", extracted.chat_id, extracted.message_id)
        return None

    logger.info(
        "事件提取成功：event_id=%s chat_type=%s chat_id=%s message_id=%s open_id=%s",
        extracted.event_id, extracted.chat_type, extracted.chat_id, extracted.message_id, extracted.open_id,
    )
    return extracted
