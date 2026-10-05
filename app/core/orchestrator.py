"""Phase 2 对话编排器 + Phase 4 记忆系统 + Phase 5 置信度与对话策略。

Orchestrator 主链路（对齐 IMPLEMENTATION_DESIGN §6.4 / PRD §5.1）：
1. MemoryManager.read（意图解析之前：工作窗口 + 情景记忆 + 用户画像）
2. 意图解析（LLM JSON + 关键词兜底）
3. 指令执行基础处理（Phase 5.5：command 类 → CommandExecutor；查询类指令转知识链路）
4. RAG 检索 + 置信度评估（Phase 5.2：discussion/qa/查询指令走完整链路，量化映射高/中/低）
5. 策略块拼装（Phase 5.3/5.4：讨论/问答/敏感/指令指引）→ Persona 构建 System Prompt
6. LLM 生成回应 + 风格兜底改写（polish）
7. MemoryManager.write 异步后置（工作记忆同步 + 情景/画像异步提取，不阻塞返回）
"""

from __future__ import annotations

import asyncio
import logging
import time

from app.core.command import CommandExecutor, CommandOutcome
from app.core.confidence import assess
from app.core.intent import IntentParser
from app.core.models import (
    ConfidenceSignals,
    IntentType,
    RAGResult,
    UnifiedRequest,
    UnifiedResponse,
)
from app.core.persona import Persona
from app.core.strategy import (
    build_command_block,
    build_discussion_block,
    build_qa_block,
    build_sensitive_block,
)
from app.care.trigger import SleepEvent, build_sleep_greeting_guidance, detect_sleep_event
from app.llm.prompts import build_intent_messages
from app.memory.manager import MemoryManager

logger = logging.getLogger(__name__)


def _messages_to_prompt_text(messages: list[dict]) -> str:
    """把 OpenAI 风格 messages 列表渲染为可读的提示词文本（供调试追踪展示）。"""
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content", "")
        lines.append(f"【{role}】\n{content}")
    return "\n\n".join(lines)


class Orchestrator:
    """对话编排器（Phase 5 版本：RAG + 置信度 + 对话策略 + 指令执行）。"""

    def __init__(
        self,
        llm_client,
        persona: Persona,
        intent_parser: IntentParser,
        memory_manager: MemoryManager,
        rag_retriever=None,
        command_executor: CommandExecutor | None = None,
        care_trigger=None,
    ) -> None:
        self._llm = llm_client
        self._persona = persona
        self._intent = intent_parser
        self._memory = memory_manager
        self._rag = rag_retriever          # RAGRetrieve | None（None 时知识链路降级跳过）
        self._command = command_executor   # CommandExecutor | None
        self._care_trigger = care_trigger  # CareTrigger | None（None 时起居识别只注入指引不落画像）
        logger.info(
            "[Orchestrator] 初始化完成（Phase 6：rag=%s, command=%s, care=%s）",
            "on" if rag_retriever is not None else "off",
            "on" if command_executor is not None else "off",
            "on" if care_trigger is not None else "off",
        )

    async def handle(self, req: UnifiedRequest, *, collect_trace: bool = False) -> UnifiedResponse:
        """处理一个用户请求，返回角色化回应。

        collect_trace=True 时，把各环节（记忆/意图/指令/RAG/提示词拼装/LLM/改写）的
        输入与组装好的提示词、输出采集到 UnifiedResponse.trace，供本地调试台人工审核。
        """
        user_id = req.user_id
        user_text = req.text
        t0 = time.monotonic()
        trace: list[dict] = []

        def _trace(name: str, input: str = "", output: str = "", notes: str = "") -> None:
            if collect_trace:
                trace.append({
                    "name": name,
                    "input": input,
                    "output": output,
                    "notes": notes,
                })

        # 1) MemoryRead（意图解析之前，对齐 MEMORY_SYSTEM_DESIGN §4.1 ①）
        memory = await self._memory.read(user_text, user_id, trace=_trace)
        window = memory.working_window
        if memory.episodic_hits or memory.user_profile:
            logger.info(
                "[Orchestrator] 记忆上下文：episodic=%d 条, profile=%s",
                len(memory.episodic_hits),
                "有" if memory.user_profile else "空",
            )
        # 注：memory 子环节（working_window / episodic_search / user_profile）
        # 已通过 trace 回调直接写入 trace 列表，此处不再单独记录 memory_read。

        # 2) 意图解析（intent_parse 子环节通过 trace 回调记录提示词+LLM原始输出）
        intent = await self._intent.parse(user_text, working_window=window, trace=_trace)
        logger.info(
            "[Orchestrator] 意图解析：intent=%s scene=%s requires_knowledge=%s sensitive=%s",
            intent.intent.value, intent.scene, intent.requires_knowledge, intent.is_sensitive,
        )

        # 2.5) 起居识别（Phase 6.1）：早晚安关键词检测（纯函数，零 IO）；
        #      画像 sleep_pattern 更新放到异步记忆写入之后，避免与画像提取竞争
        sleep_evt = detect_sleep_event(user_text)
        if sleep_evt is not None:
            logger.info("[Orchestrator] 起居事件识别：kind=%s", sleep_evt.kind)

        # 3) 指令执行基础处理（Phase 5.5）：查询类指令转入知识链路
        command_outcome: CommandOutcome | None = None
        requires_knowledge = intent.requires_knowledge
        if intent.intent == IntentType.COMMAND and self._command is not None:
            command_outcome = self._command.handle(user_text, intent.slots, user_id)
            logger.info(
                "[Orchestrator] 指令执行：action=%s status=%s missing=%s",
                command_outcome.action, command_outcome.status, command_outcome.missing,
            )
            if command_outcome.is_query:
                requires_knowledge = True
            _trace(
                "command_exec",
                input=f"text={user_text}; slots={intent.slots}",
                output=(
                    f"action={command_outcome.action}; status={command_outcome.status}; "
                    f"is_query={command_outcome.is_query}; missing={command_outcome.missing}"
                ),
            )

        # 4) RAG 检索 + 置信度评估（Phase 5.2：讨论/问答/查询指令走完整链路）
        rag_result: RAGResult | None = None
        confidence = None
        if requires_knowledge:
            rag_result = await self._retrieve(user_text, window, intent.domain_filter, trace=_trace)
            confidence = assess(rag_result.confidence_signals)
            logger.info(
                "[Orchestrator] 置信度评估：%s（hits=%d max_score=%.2f）",
                confidence.level.value,
                rag_result.confidence_signals.total_hits,
                rag_result.confidence_signals.max_rerank_score,
            )
            # 注：RAG 子环节（term_lookup / query_rewrite / 三路召回 / merge / rerank / final_candidates）
            # 已通过 trace 回调直接写入 trace 列表，此处仅补充置信度评估结论。
            _trace(
                "rag/confidence_assess",
                input=f"signals={rag_result.confidence_signals.model_dump()}",
                output=f"confidence={confidence.level.value}",
                notes="基于召回信号的置信度评估（high/mid/low）",
            )

        # 5) 策略块拼装（Phase 5.3/5.4 + Phase 6.1 起居回应）+ Persona 构建 System Prompt
        strategy_blocks = self._build_strategy_blocks(intent, command_outcome, sleep_evt)
        system_prompt = self._persona.build_system_prompt(
            scene=intent.scene,
            user_profile=memory.user_profile or None,
            episodic_context=memory.episodic_hits or None,
            rag_result=rag_result,
            confidence=confidence,
            strategy_blocks=strategy_blocks,
        )

        messages = self._persona.build_messages(
            system_prompt,
            user_text,
            working_window=window,
        )
        _trace(
            "system_prompt",
            input=(
                f"scene={intent.scene}; strategy_blocks={len(strategy_blocks)}; "
                f"rag={'有' if rag_result else '无'}; confidence={confidence.level.value if confidence else 'N/A'}"
            ),
            output=system_prompt,
            notes="Persona.build_system_prompt 拼装结果",
        )
        _trace(
            "llm_messages",
            input="system_prompt + working_window + user_text",
            output=_messages_to_prompt_text(messages),
            notes=f"共 {len(messages)} 条消息（含 system、历史窗口、当前用户）",
        )

        # 6) LLM 生成
        try:
            raw_reply = await self._llm.chat(
                messages,
                temperature=0.8,  # 角色回复可更高创意
                timeout=15.0,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("[Orchestrator] LLM 调用失败")
            reply = "（我的风……似乎暂时静了一下……漂泊者，请稍等片刻好吗？……）"
            _trace("llm_generate", input="（见 llm_messages）", output=f"LLM 调用失败：{e}", notes="降级兜底回复")
        else:
            _trace("llm_generate", input="（见 llm_messages）", output=raw_reply, notes=f"temperature=0.8, timeout=15.0")
            # 7) 风格兜底改写（polish 子环节通过 trace 回调记录每轮提示词+输出）
            reply = await self._persona.polish(raw_reply, self._llm, scene=intent.scene, trace=_trace)

        latency = (time.monotonic() - t0) * 1000
        logger.info(
            "[Orchestrator] 完成：user_id=%s latency=%.0fms reply_len=%d",
            user_id, latency, len(reply),
        )

        # 8) MemoryWrite 异步后置（不阻塞返回，M-08）；其后记录起居事件到画像
        asyncio.create_task(
            self._async_write_memory(user_id, user_text, reply, intent.is_sensitive, sleep_evt)
        )

        resp = UnifiedResponse(user_id=user_id, text=reply)
        if collect_trace:
            resp.trace = trace
        return resp

    # ---- Phase 5 内部步骤 ----

    async def _retrieve(
        self,
        query: str,
        window: list[dict],
        domain_filter: list[str] | None,
        *,
        trace=None,
    ) -> RAGResult:
        """RAG 检索；任何失败降级为空结果（→ 低置信度措辞，不阻塞对话）。"""
        if self._rag is None:
            logger.warning("[Orchestrator] requires_knowledge 但未注入 RAGRetrieve，跳过检索")
            return RAGResult(candidates=[], confidence_signals=ConfidenceSignals())
        try:
            return await self._rag.retrieve(
                query, context=window, domain_filter=domain_filter, trace=trace,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("[Orchestrator] RAG 检索失败，降级为低置信度回应：%s", e)
            return RAGResult(candidates=[], confidence_signals=ConfidenceSignals())

    def _build_strategy_blocks(
        self, intent, command_outcome: CommandOutcome | None, sleep_evt: SleepEvent | None
    ) -> list[str]:
        """按意图拼装策略块（对齐 PRD §5.3/5.4：敏感优先，其次意图策略/指令指引）。"""
        blocks: list[str] = []
        if intent.is_sensitive:
            blocks.append(build_sensitive_block())
        if intent.intent == IntentType.DISCUSSION:
            blocks.append(build_discussion_block())
        elif intent.intent == IntentType.QA:
            blocks.append(build_qa_block())
        elif intent.intent == IntentType.COMMAND and command_outcome is not None:
            blocks.append(build_command_block(command_outcome))
        if sleep_evt is not None:
            blocks.append(build_sleep_greeting_guidance(sleep_evt))
        return blocks

    async def _async_write_memory(
        self,
        user_id: str,
        user_text: str,
        reply: str,
        is_sensitive: bool,
        sleep_evt: SleepEvent | None = None,
    ) -> None:
        """异步记忆写入：工作记忆 + 情景/画像提取；记忆写入完成后再记录起居事件。

        起居画像更新放在 MemoryManager.write 之后：两者都是 load→局部 merge→save，
        串行执行可避免 sleep_pattern 字段互相覆盖（MemoryManager 内部画像提取为
        fire-and-forget 任务，故此处再让出一轮事件循环等待其落库）。
        """
        try:
            await self._memory.write(
                user_text, reply, user_id, is_sensitive=is_sensitive
            )
            if sleep_evt is not None and self._care_trigger is not None:
                await asyncio.sleep(1.0)  # 等待 MemoryManager 画像提取任务落库
                await self._care_trigger.record(user_id, sleep_evt)
        except Exception as e:  # noqa: BLE001
            logger.exception("[Orchestrator] 异步记忆/起居写入失败：%s", e)

    # ==================== 兼容 ChannelAdapter Orchestrator 签名 ====================

    async def __call__(self, req: UnifiedRequest) -> UnifiedResponse:
        """FeishuAdapter 期待 Orchestrator 签名为 Callable[[UnifiedRequest], Awaitable[UnifiedResponse]]。"""
        return await self.handle(req)
