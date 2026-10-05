"""应用入口（Phase 4 版本）。

职责：
1. 启动校验 + 存储初始化（Phase 0）；
2. 构造 LLMClient / EmbeddingClient / Persona / IntentParser / MemoryManager / Orchestrator
   （Phase 2 主链路 + Phase 4 三层记忆）；
3. FastAPI 应用：健康检查 + Webhook 事件路由 + 本地调试台；
4. FeishuAdapter 按 feishu_mode 启动，注入 Orchestrator；
5. lifespan 上下文管理：衰减清理任务启停 + 优雅退出。
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# 直接运行本文件（python app/main.py）时，脚本目录 app/ 会成为 sys.path[0]，
# 导致 `from app import ...` 找不到包；这里把项目根目录加入 sys.path 修复。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app import __version__
from app.care.composer import CareComposer
from app.care.scheduler import CareScheduler
from app.care.trigger import CareTrigger
from app.channels.feishu.channel import FeishuAdapter
from app.config import check_startup, get_settings
from app.core.command import CommandExecutor
from app.core.intent import IntentParser
from app.core.orchestrator import Orchestrator
from app.core.persona import Persona
from app.db.connection import init_infrastructure
from app.llm.client import LLMClient
from app.llm.embedding import EmbeddingClient
from app.memory.manager import MemoryManager
from app.rag.retriever import RAGRetrieve
from app.rag.stores import RAGStore
from app.web.server import create_debug_router, install_log_broker

logger = logging.getLogger(__name__)

# ==================== 应用级全局对象（FastAPI lifespan 内注入）====================

feishu_adapter: FeishuAdapter | None = None
_orchestrator: Orchestrator | None = None
_memory_manager: MemoryManager | None = None
_care_scheduler: CareScheduler | None = None


async def _push_care_message(chat_id: str, text: str) -> bool | None:
    """关怀调度推送渠道：飞书 send_as_bot；飞书未配置时返回 None（scheduler 走 dry-run）。"""
    if feishu_adapter is None:
        return None
    from app.channels.feishu.adapter import build_text_content
    return await feishu_adapter.client.send_text_message(
        chat_id, content=build_text_content(text)
    )


def _build_components(settings) -> tuple[Orchestrator, CareScheduler]:
    """构造 Phase 2 + Phase 4 + Phase 5 + Phase 6 全部组件并注入 Orchestrator。"""
    llm_client = LLMClient(settings)
    embedding_client = EmbeddingClient(settings)
    persona = Persona()  # 从 character/yangyang_xuanling_soul.json 加载
    intent_parser = IntentParser(llm_client)
    memory_manager = MemoryManager(
        llm_client, embedding_client, settings,
        window_size=settings.memory_window_size,
    )
    rag_retriever = RAGRetrieve(
        embedding=embedding_client, llm=llm_client,
        store=RAGStore(settings), settings=settings,
    )
    command_executor = CommandExecutor(settings)
    care_trigger = CareTrigger(settings)
    care_composer = CareComposer(settings, llm=llm_client)
    care_scheduler = CareScheduler(
        settings,
        composer=care_composer,
        profile_store=care_trigger._profiles,  # 复用同一画像存储实例
        sender=_push_care_message,
    )
    orchestrator = Orchestrator(
        llm_client=llm_client,
        persona=persona,
        intent_parser=intent_parser,
        memory_manager=memory_manager,
        rag_retriever=rag_retriever,
        command_executor=command_executor,
        care_trigger=care_trigger,
    )
    logger.info(
        "[main] 组件构造完成：LLMClient + EmbeddingClient + Persona(%s) + IntentParser "
        "+ MemoryManager(n=%d) + RAGRetrieve + CommandExecutor + CareTrigger/CareScheduler",
        persona.soul.name, settings.memory_window_size,
    )
    return orchestrator, care_scheduler


# ==================== FastAPI 应用 ====================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动 / 退出生命周期。"""
    settings = get_settings()

    # 1) 启动校验 + 存储初始化（这步不依赖飞书配置）
    missing = check_startup(settings)
    init_infrastructure(settings)

    # 1.5) 注入 SSE 日志 broker（调试台实时日志流；幂等，仅注入一次）
    install_log_broker()

    # 2) 构造 Phase 2 + Phase 4 + Phase 5 + Phase 6 全部组件（始终需要，IntentParser/Polish 等都依赖 LLM）
    global _orchestrator, _memory_manager, _care_scheduler
    _orchestrator, _care_scheduler = _build_components(settings)
    _memory_manager = getattr(_orchestrator, "_memory", None)

    # 2.5) 启动记忆系统周期任务（每周清理 decay < 0.1 的情景记忆）
    if _memory_manager is not None:
        _memory_manager.start_decay_task()

    # 2.6) 启动关怀调度定时循环（飞书未配置时 sender 走 dry-run，仅记录日志）
    _care_scheduler.start()

    # 3) 飞书未配置时只启动 API 服务，调试台可用
    if len(missing) >= 3:  # 飞书 3 项都缺：APP_ID/SECRET/VERIFICATION_TOKEN
        logger.warning(
            "[main] 飞书配置未就绪（%d/%d 缺），本次启动仅提供 API 服务，不启动消息渠道。",
            len(missing), 5,
        )
    else:
        # 4) 构造 FeishuAdapter 并注入 Orchestrator
        global feishu_adapter
        feishu_adapter = FeishuAdapter(settings)
        asyncio.create_task(feishu_adapter.start(_orchestrator))
        logger.info("[main] FeishuAdapter 已启动（mode=%s）", settings.feishu_mode)

    try:
        yield
    finally:
        if _memory_manager is not None:
            await _memory_manager.stop_decay_task()
        if feishu_adapter is not None:
            await feishu_adapter.stop()
        logger.info("[main] 应用优雅退出")


app = FastAPI(
    title="HachiYang",
    description="秧秧·玄翎陪伴型 Agent（Phase 0 骨架 + Phase 1 飞书收发 + Phase 2.5 调试台 + Phase 3 RAG + Phase 4 记忆系统）",
    version=__version__,
    lifespan=lifespan,
)

# 调试台路由：仅 127.0.0.1 可访问（路由层校验），Orchestrator 通过 getter 延迟获取
app.include_router(create_debug_router(lambda: _orchestrator))


# ==================== 路由 ====================

@app.get("/health")
async def health() -> dict[str, Any]:
    """健康检查（部署探针 + 调试用）。"""
    settings = get_settings()
    return {
        "service": "HachiYang",
        "version": __version__,
        "feishu_mode": settings.feishu_mode,
        "websocket_started": feishu_adapter is not None and settings.feishu_mode == "websocket",
    }


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "HachiYang",
        "version": __version__,
        "hint": "访问 /health 查看健康状态；飞书 Webhook 事件订阅 POST 到 /feishu/event",
    }


@app.post("/feishu/event")
async def feishu_event(request: Request) -> JSONResponse:
    """飞书 Webhook 事件订阅端点（生产用）。

    处理：
    1) 解密（如果 FEISHU_ENCRYPT_KEY 配置了）；
    2) URL 校验挑战（type=url_verification 时返回 challenge）；
    3) 其他事件 feed 给 FeishuAdapter。
    """
    body = await request.json()

    # 1) 加密解密：飞书启用事件订阅加密时，body 里是 {encrypt: "..."}
    #    lark-oapi 1.7.x 提供 EventDispatcherHandler.do_without_validation(decrypted_dict)
    #    内部有 LarkCrypto 工具类解密；如果没配置 encrypt_key，这里直接用 body
    encrypt = body.get("encrypt") if isinstance(body, dict) else None
    if encrypt and get_settings().feishu_encrypt_key:
        body = _decrypt_feishu_event(encrypt, get_settings().feishu_encrypt_key)

    # 2) URL 校验挑战
    if isinstance(body, dict) and body.get("type") == "url_verification":
        challenge = body.get("challenge", "")
        logger.info("[feishu/event] URL 校验挑战：challenge=%s", challenge)
        return JSONResponse(content={"challenge": challenge})

    # 3) feed FeishuAdapter（异步处理，立即返回 200 避免飞书重试）
    if feishu_adapter is None:
        raise HTTPException(status_code=503, detail="FeishuAdapter 未就绪（飞书配置缺失）")

    asyncio.create_task(feishu_adapter.handle_raw_event(body))
    return JSONResponse(content={"code": 0, "msg": "ok"})


# ==================== 加密解密辅助 ====================

def _decrypt_feishu_event(encrypt_str: str, encrypt_key: str) -> dict:
    """飞书事件订阅加密解密。

    飞书开放平台支持 AES-256-CBC 或 AEAD_AES_256_GCM 两种加密，
    lark-oapi 1.7.x 内部有 LarkCrypto 类；如果找不到公开 API，退化为
    简单的占位实现（仅 log，不阻断）——实际生产时再完善。
    """
    # 尝试用 lark-oapi 内部的 LarkCrypto
    try:
        from lark_oapi.core.crypto import LarkCrypto  # type: ignore
        crypto = LarkCrypto()
        decrypted = crypto.decrypt(encrypt_str, encrypt_key)
        return json.loads(decrypted)
    except Exception as e:  # noqa: BLE001
        logger.warning("飞书事件解密失败，原样传递（SDK crypto 未导出或密钥问题）：%s", e)
        try:
            return json.loads(encrypt_str)
        except Exception:
            raise HTTPException(status_code=400, detail="encrypt 字段无法解密") from e


# ==================== 启动入口 ====================

def main() -> None:
    """python -m app.main 或直接运行本文件时启动 uvicorn。"""
    import uvicorn

    settings = get_settings()
    # 飞书 SDK 日志级别
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # 配置 lark_oapi 内部 logger 到 WARNING，避免刷屏
    logging.getLogger("lark_oapi").setLevel(logging.WARNING)

    logger.info("=" * 60)
    logger.info("HachiYang v%s 启动中...", __version__)
    logger.info("=" * 60)
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8080,
        log_level="info",
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
