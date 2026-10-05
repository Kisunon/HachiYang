"""本地调试台（Phase 2.5 基础版）。

仅绑定 127.0.0.1，不对外暴露。功能（对齐 IMPLEMENTATION_DESIGN §6.11）：
- GET  /debug          → 静态页面（聊天面板 + 日志面板）
- POST /debug/chat      → 模拟对话（接入编排器 handle_message）
- GET  /debug/status    → 汇总状态 JSON（编排器/工作记忆就绪情况）
- GET  /debug/events   → SSE 实时日志流（root logger 推送）

说明：RAG/记忆/关怀监测面板随 Phase 3/4/6 完成后逐步接入（见 Phase 7.1）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.core.models import UnifiedRequest, UnifiedResponse
from app.web.rag_inspect import create_rag_inspect_router

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_INDEX_HTML = _STATIC_DIR / "index.html"

# 默认模拟 user_id（与真实飞书用户隔离）
_DEFAULT_DEBUG_USER_ID = "local:debug"
_DEFAULT_CHAT_ID = "local"
_DEFAULT_OPEN_ID = "debug"

# SSE 心跳间隔（秒），防止代理/浏览器因长时间无数据断开
_SSE_HEARTBEAT_INTERVAL = 15.0


# ==================== SSE 日志广播器 ====================

class _SSELogBroker:
    """进程内日志广播：logging.Handler → 多 SSE 客户端。

    - 每个 SSE 客户端订阅时创建一个 asyncio.Queue；
    - Handler.emit 把日志记录推入所有活跃队列（非阻塞，慢客户端丢包）；
    - 保留最近 N 条历史日志，新连接的客户端可立即看到上下文。
    """

    def __init__(self, history_maxlen: int = 200) -> None:
        self._subscribers: set[asyncio.Queue[dict]] = set()
        self._lock = asyncio.Lock()
        self._history: deque[dict] = deque(maxlen=history_maxlen)

    async def subscribe(self) -> asyncio.Queue[dict]:
        q: asyncio.Queue[dict] = asyncio.Queue(maxsize=500)
        async with self._lock:
            self._subscribers.add(q)
            # 先把历史日志塞进队列，新客户端立即看到上下文
            for item in list(self._history):
                try:
                    q.put_nowait(item)
                except asyncio.QueueFull:
                    break
        return q

    async def unsubscribe(self, q: asyncio.Queue[dict]) -> None:
        async with self._lock:
            self._subscribers.discard(q)

    def publish(self, record_dict: dict) -> None:
        """同步 publish（从 logging.Handler 调用，无 await）。"""
        self._history.append(record_dict)
        # 复制一份避免遍历过程中订阅集合变化
        for q in list(self._subscribers):
            try:
                q.put_nowait(record_dict)
            except asyncio.QueueFull:
                # 慢客户端队列满则丢弃，不阻塞其他订阅者
                pass


_broker = _SSELogBroker()


class _QueueLogHandler(logging.Handler):
    """把日志记录格式化后推送到 SSE broker。"""

    def __init__(self, broker: _SSELogBroker) -> None:
        super().__init__(level=logging.INFO)
        self._broker = broker

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._broker.publish({
                "ts": record.created,
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            })
        except Exception:  # noqa: BLE001
            self.handleError(record)


# 全局是否已注入 handler（避免重复注入）
_handler_installed = False


def install_log_broker(root_level: int = logging.INFO) -> None:
    """把 SSE handler 注入到 root logger。main.py 启动时调用一次。

    - root_level：handler 自身的日志级别阈值（默认 INFO）。
    - 不修改 root logger 已有 handler，仅追加。
    """
    global _handler_installed
    if _handler_installed:
        return
    handler = _QueueLogHandler(_broker)
    handler.setLevel(root_level)
    logging.getLogger().addHandler(handler)
    _handler_installed = True
    logger.info("[debug-console] SSE 日志流已注入 root logger")


# ==================== 请求模型 ====================

class ChatRequestBody(BaseModel):
    text: str
    user_id: str | None = None


# ==================== 本地访问校验 ====================

def _ensure_localhost(request: Request) -> None:
    """仅允许 127.0.0.1 / ::1 / localhost 访问调试台。

    "testclient" 是 ASGI TestClient 的默认 client.host（FastAPI/Pytest 内置），
    仅在测试进程内出现，真实 HTTP 请求绝不会有该 host，故等效 localhost。
    """
    client = request.client
    if client is None:
        raise HTTPException(status_code=403, detail="无法识别客户端地址")
    host = client.host
    if host not in ("127.0.0.1", "::1", "localhost", "testclient"):
        raise HTTPException(status_code=403, detail="调试台仅限本地访问（127.0.0.1）")


# ==================== 路由工厂 ====================

def create_debug_router(orchestrator_getter: Callable[[], Any]) -> APIRouter:
    """创建调试台路由。

    orchestrator_getter：无参 callable，返回当前 Orchestrator 实例（或 None）。
    用 getter 而非直接传实例，是因为 main.py lifespan 中才构造 orchestrator。
    """
    router = APIRouter(prefix="/debug", tags=["debug"])

    # Phase 3.5：知识库索引人工检查（只读 /debug/rag/*，路由自身 prefix=/rag）
    router.include_router(create_rag_inspect_router())

    def _get_orchestrator() -> Any:
        orch = orchestrator_getter()
        if orch is None:
            raise HTTPException(
                status_code=503,
                detail="Orchestrator 未就绪（启动中或配置缺失）",
            )
        return orch

    # ---- 静态页面 ----
    @router.get("", response_class=FileResponse)
    @router.get("/", response_class=FileResponse)
    async def debug_page(request: Request) -> FileResponse:
        _ensure_localhost(request)
        if not _INDEX_HTML.exists():
            raise HTTPException(status_code=404, detail="index.html 不存在")
        return FileResponse(_INDEX_HTML)

    # ---- 模拟对话 ----
    @router.post("/chat")
    async def debug_chat(body: ChatRequestBody, request: Request) -> JSONResponse:
        _ensure_localhost(request)
        orch = _get_orchestrator()

        text = body.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="text 不能为空")

        user_id = (body.user_id or _DEFAULT_DEBUG_USER_ID).strip() or _DEFAULT_DEBUG_USER_ID
        req = UnifiedRequest(
            user_id=user_id,
            chat_id=_DEFAULT_CHAT_ID,
            open_id=_DEFAULT_OPEN_ID,
            text=text,
            is_group=False,
        )

        try:
            resp: UnifiedResponse = await orch.handle(req, collect_trace=True)
        except Exception as e:  # noqa: BLE001
            logger.exception("[debug/chat] 编排器调用失败")
            raise HTTPException(status_code=500, detail=f"编排器错误：{e}") from e

        return JSONResponse(content={
            "user_id": resp.user_id,
            "text": resp.text,
            "trace": resp.trace,
        })

    # ---- 汇总状态 ----
    @router.get("/status")
    async def debug_status(request: Request) -> JSONResponse:
        _ensure_localhost(request)
        orch = _get_orchestrator()

        # Phase 4：orch._memory 为 MemoryManager（三层记忆）；兼容旧版工作记忆
        mem = getattr(orch, "_memory", None)
        buffer_info: dict[str, Any] = {}
        if mem is not None and hasattr(mem, "status"):
            try:
                buffer_info = mem.status()
            except Exception:  # noqa: BLE001
                buffer_info = {}
        elif mem is not None:
            buffers = getattr(mem, "_buffers", None)
            if isinstance(buffers, dict):
                buffer_info = {
                    "sessions": len(buffers),
                    "window_size": getattr(mem, "_window_size", None),
                    "total_turns": sum(len(d) for d in buffers.values() if hasattr(d, "__len__")),
                }

        persona = getattr(orch, "_persona", None)
        persona_name = None
        soul = getattr(persona, "soul", None) if persona is not None else None
        if soul is not None:
            persona_name = getattr(soul, "name", None)

        llm = getattr(orch, "_llm", None)
        llm_model = getattr(llm, "_model", None) if llm is not None else None

        return JSONResponse(content={
            "orchestrator_ready": True,
            "llm_model": llm_model,
            "persona": persona_name,
            "working_memory": buffer_info,
            "note": "记忆监测面板已接入（Phase 4）；知识库检查见 /debug 页面「知识库检查」视图（Phase 3.5）；关怀监测面板随 Phase 6 接入",
        })

    # ---- SSE 日志流 ----
    @router.get("/events")
    async def debug_events(request: Request) -> StreamingResponse:
        _ensure_localhost(request)
        q = await _broker.subscribe()

        async def event_stream():
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        item = await asyncio.wait_for(q.get(), timeout=_SSE_HEARTBEAT_INTERVAL)
                        yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                    except asyncio.TimeoutError:
                        # 心跳保活，防止代理/浏览器断开
                        yield ": heartbeat\n\n"
            finally:
                await _broker.unsubscribe(q)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return router
