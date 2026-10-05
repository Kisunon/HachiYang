"""飞书消息格式转换：handler.ExtractedEvent ↔ core.UnifiedRequest / UnifiedResponse。

职责：
1. 把 handler 层提取出的 ExtractedEvent 转成统一请求；
2. 持久化 chat_sessions（后续关怀调度需要 chat_id）；
3. 把统一回应转成飞书文本 / Markdown 消息的 content JSON；
4. 长文本自动分段（飞书单条上限 ≈ 2000 汉字）。

本文件**只含纯函数**（除了内部 logging / JSON），没有 FeishuAdapter 类（见 channel.py）。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

from app.core.models import UnifiedRequest, UnifiedResponse
from app.db.connection import db_conn
from app.channels.feishu.handler import ExtractedEvent, is_group_chat, parse_text_content

logger = logging.getLogger(__name__)

# 飞书单条消息文本上限（保守 1500 汉字 / 4500 字符）
_TEXT_CHUNK_SIZE = 1500


# ==================== ExtractedEvent → UnifiedRequest ====================

def build_user_id(chat_id: str, open_id: str) -> str:
    """分区键（M-02）：chat_id + open_id 拼接，多用户隔离。"""
    return f"{chat_id}:{open_id}"


def extracted_to_request(evt: ExtractedEvent) -> UnifiedRequest:
    """ExtractedEvent → UnifiedRequest（同步，纯函数）。"""
    text = parse_text_content(evt.content_json)
    return UnifiedRequest(
        user_id=build_user_id(evt.chat_id, evt.open_id),
        chat_id=evt.chat_id,
        open_id=evt.open_id,
        text=text,
        is_group=is_group_chat(evt),
    )


def upsert_chat_session(evt: ExtractedEvent) -> None:
    """把 (chat_id, open_id, user_id) 写入 chat_sessions（UPSERT）。

    同步 SQLite，失败仅记 warning，不阻塞主链路。
    """
    user_id = build_user_id(evt.chat_id, evt.open_id)
    now = int(time.time())
    try:
        with db_conn() as conn:
            conn.execute(
                """
                INSERT INTO chat_sessions (user_id, chat_id, open_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (user_id, evt.chat_id, evt.open_id, now, now),
            )
    except Exception as e:
        logger.warning("chat_sessions upsert 失败：%s", e)


# ==================== UnifiedResponse → 飞书 content JSON ====================

def build_text_content(text: str) -> str:
    """飞书 text 消息的 content JSON。"""
    return json.dumps({"text": text}, ensure_ascii=False)


def build_post_content(text: str, title: str = "HachiYang") -> str:
    """飞书 post（富文本）消息的 content JSON，支持基础格式。"""
    post = {
        "zh_cn": {
            "title": title,
            "content": [[{"tag": "text", "text": text}]],
        }
    }
    return json.dumps({"post": post}, ensure_ascii=False)


def chunk_text(text: str, size: int = _TEXT_CHUNK_SIZE) -> list[str]:
    """按字符数分段，中文硬切（无 mid-word 概念）。"""
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    for start in range(0, len(text), size):
        chunks.append(text[start : start + size])
    return chunks


@dataclass
class FeishuMessage:
    """飞书待发送消息（自动分片后的待发送载荷）。"""
    chat_id: str
    msg_type: str          # "text" | "post"
    contents: list[str]    # 每条是飞书 content JSON 字符串
    reply_to_message_id: str | None = None


def response_to_feishu_messages(
    resp: UnifiedResponse,
    *,
    reply_to_message_id: str | None = None,
    prefer_post: bool = True,
) -> FeishuMessage:
    """把编排器 UnifiedResponse 转成飞书消息（自动分段 + 格式选择）。

    - 短文本直接发一条；长文本按 _TEXT_CHUNK_SIZE 自动分段；
    - prefer_post=True 时用 post（富文本），否则纯 text。
    - chat_id 从 user_id（f"{chat_id}:{open_id}"）里解出来。
    """
    chat_id, _ = resp.user_id.split(":", 1) if ":" in resp.user_id else (resp.user_id, "")
    chunks = chunk_text(resp.text)
    msg_type = "post" if prefer_post else "text"
    builder = build_post_content if prefer_post else build_text_content
    contents = [builder(c) for c in chunks]
    return FeishuMessage(
        chat_id=chat_id,
        msg_type=msg_type,
        contents=contents,
        reply_to_message_id=reply_to_message_id,
    )
