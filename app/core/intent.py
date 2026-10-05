"""意图解析模块（Phase 2 版本，纯 LLM + 关键词兜底）。

职责（对齐 IMPLEMENTATION_DESIGN §6.5）：
- 对用户输入进行五类意图分类（discussion/qa/chat/command/care）；
- 识别场景 scene（daily/academic/sadness/intimate/battle）；
- 判断是否需要知识库检索、领域过滤、是否敏感话题；
- 解析指令类的 slots 参数。

设计：
- **主路径**：LLM JSON 模式，prompt 集中在 llm/prompts.py；
- **关键词兜底**：快速识别早晚安、帮我/提醒我等高频模式，省掉一次 LLM 调用；
- **降级兜底**：LLM 超时/解析失败时返回 IntentResult(intent=chat, scene=daily)，不阻塞对话；
- 调用方传入 working_window 最近 3 轮补足指代。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

from app.core.models import IntentResult, IntentType
from app.llm.prompts import build_intent_messages

logger = logging.getLogger(__name__)


# ==================== 关键词规则（快速路径）====================

# care 触发词：早晚安 / 作息相关
_CARE_PATTERNS = [
    r"^(早|早安|早上好|晚安|晚上好|我起床了|起床了|我睡了|睡了|我要睡|困了)$",
    r"^[早晚][，,]?$",
    r"(早安|晚安|该睡|该起床|睡觉了|起床啦)",
]
_CARE_REGEXES = [re.compile(p) for p in _CARE_PATTERNS]

# command 触发词：明确指令
_COMMAND_PATTERNS = [
    r"^(帮我|提醒我|帮我记|帮我查|查一下|查询|记录一下|记录|记住|设个提醒|定时提醒)",
    r"(提醒我|帮我.*做|查一下|查询.*怎么|记录下来)",
]
_COMMAND_REGEXES = [re.compile(p) for p in _COMMAND_PATTERNS]

# sadness 触发词
_SADNESS_KEYWORDS = [
    "难过", "伤心", "抑郁", "焦虑", "痛苦", "无助", "崩溃", "委屈",
    "不开心", "想哭", "想哭", "好烦", "好累", "压力大", "绝望", "孤独",
]

# sensitive 触发词
_SENSITIVE_KEYWORDS = [
    "自杀", "自残", "抑郁发作", "重度抑郁", "抑郁症", "PTSD", "创伤",
    "我不想活", "活不下去", "结束自己",
]

# domain 触发词（对应 PRD 知识库三大领域）
_DOMAIN_KEYWORDS = {
    "freud": ["弗洛伊德", "佛洛伊德", "Freud", "潜意识", "压抑", "防御机制", "梦的解析", "强迫性重复"],
    "jung": ["荣格", "Jung", "情结", "原型", "自性化", "集体潜意识"],
    "philosophy": ["哲学", "苏格拉底", "柏拉图", "亚里士多德", "尼采", "海德格尔", "存在主义"],
}

# academic 场景触发词
_ACADEMIC_INDICATORS = [
    "理论", "定义", "概念", "研究", "学派", "学说",
]

# intimate 触发词
_INTIMATE_KEYWORDS = [
    "睡不着", "失眠", "深夜", "睡不着觉", "陪我", "陪我聊", "好想你",
]

# battle 触发词
_BATTLE_KEYWORDS = [
    "剑", "保护", "战斗", "守护", "危险", "敌人", "残象",
]


def _match_any(regexes: list[re.Pattern], text: str) -> bool:
    return any(r.search(text) for r in regexes)


def _contains_any(keywords: list[str], text: str) -> bool:
    return any(kw in text for kw in keywords)


def _keyword_heuristic(text: str) -> IntentResult | None:
    """纯关键词规则快速路径，命中时省掉一次 LLM 调用。

    返回 None 表示关键词没覆盖到，交给 LLM。
    """
    stripped = text.strip()

    # 1) care：最高优先级（早晚安是非常高频的触发）
    if _match_any(_CARE_REGEXES, stripped):
        return IntentResult(
            intent=IntentType.CARE,
            scene="daily",
            requires_knowledge=False,
            is_sensitive=False,
        )

    # 2) command
    if _match_any(_COMMAND_REGEXES, stripped):
        slots: dict[str, Any] = {}
        # 简单抽取 reminder_time
        time_match = re.search(r"(\d{1,2}[点时分半])", stripped)
        if time_match:
            slots["reminder_time"] = time_match.group(1)
        return IntentResult(
            intent=IntentType.COMMAND,
            scene="daily",
            requires_knowledge=False,
            is_sensitive=False,
            slots=slots,
        )

    # 3) sensitive（必须尽早识别）
    is_sensitive = _contains_any(_SENSITIVE_KEYWORDS, stripped)
    if is_sensitive:
        return IntentResult(
            intent=IntentType.CHAT,  # 敏感话题先归为 chat，让编排器在敏感话题处理模块介入
            scene="sadness",
            requires_knowledge=False,
            is_sensitive=True,
        )

    return None


# ==================== LLM 解析 + JSON 提取 ====================

def _extract_json(raw: str) -> dict | None:
    """从 LLM 返回中提取 JSON 对象（处理可能的 ```json 包裹 / 额外文字）。"""
    text = raw.strip()
    # 去掉可能的 markdown fence
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    # 找第一个 { 到最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _parse_llm_output(obj: dict) -> IntentResult | None:
    """把 LLM 返回的 dict 映射到 IntentResult；字段缺省则回退到默认值。"""
    intent_val = (obj.get("intent") or "chat").lower()
    # 兼容 IntentType 枚举
    try:
        intent = IntentType(intent_val)
    except ValueError:
        intent = IntentType.CHAT

    scene = (obj.get("scene") or "daily").lower()
    valid_scenes = {"daily", "academic", "sadness", "intimate", "battle"}
    if scene not in valid_scenes:
        scene = "daily"

    requires_knowledge = bool(obj.get("requires_knowledge", False))

    domain_filter = obj.get("domain_filter")
    if not isinstance(domain_filter, list):
        domain_filter = None
    else:
        # 清理非法值
        domain_filter = [d.lower() for d in domain_filter if isinstance(d, str)]
        if not domain_filter:
            domain_filter = None

    is_sensitive = bool(obj.get("is_sensitive", False))

    slots = obj.get("slots")
    if not isinstance(slots, dict):
        slots = {}

    return IntentResult(
        intent=intent,
        scene=scene,
        requires_knowledge=requires_knowledge,
        domain_filter=domain_filter,
        is_sensitive=is_sensitive,
        slots=slots,
    )


# ==================== 场景/领域后处理（关键词补漏）====================

def _post_enrichment(result: IntentResult, text: str) -> IntentResult:
    """LLM 返回后用关键词补漏（domain_filter、scene、sensitive）。

    LLM 偶尔漏判领域/敏感/场景，关键词规则是便宜的保险。
    """
    # domain_filter 补漏
    if result.domain_filter is None:
        detected: list[str] = []
        for domain, kws in _DOMAIN_KEYWORDS.items():
            if _contains_any(kws, text):
                detected.append(domain)
        if detected:
            result.domain_filter = detected
            # 同时提升 requires_knowledge（discussion/qa 本来就会 true，chat 也可能有知识性讨论）
            if result.intent in (IntentType.DISCUSSION, IntentType.QA, IntentType.CHAT):
                result.requires_knowledge = True

    # sensitive 补漏
    if not result.is_sensitive:
        if _contains_any(_SENSITIVE_KEYWORDS, text):
            result.is_sensitive = True

    # scene 补漏
    # sadness：LLM 可能判 daily，但关键词命中
    if result.scene == "daily" and _contains_any(_SADNESS_KEYWORDS, text):
        result.scene = "sadness"
    # intimate
    elif result.scene == "daily" and _contains_any(_INTIMATE_KEYWORDS, text):
        result.scene = "intimate"
    # battle
    elif result.scene == "daily" and _contains_any(_BATTLE_KEYWORDS, text):
        result.scene = "battle"
    # academic：需要有 domain 或术语
    elif result.scene == "daily" and _contains_any(_ACADEMIC_INDICATORS, text):
        result.scene = "academic"

    # 意图与 scene 的合理性修正：
    # care + 非 daily → 强制 daily
    if result.intent == IntentType.CARE and result.scene != "daily":
        result.scene = "daily"
    # command + 非 daily → 强制 daily
    if result.intent == IntentType.COMMAND and result.scene != "daily":
        result.scene = "daily"

    # qa/discussion + 无 requires_knowledge 但有 domain → 补 true
    if result.intent in (IntentType.QA, IntentType.DISCUSSION) and not result.requires_knowledge:
        result.requires_knowledge = True

    return result


# ==================== IntentParser 类 ====================

class IntentParser:
    """意图解析器（Phase 2 版本，LLM + 关键词双路径）。"""

    def __init__(self, llm_client: Any) -> None:
        self._llm = llm_client

    async def parse(
        self,
        text: str,
        *,
        working_window: list[dict] | None = None,
        trace: Callable[[str, str, str, str], None] | None = None,
    ) -> IntentResult:
        """解析用户输入，返回 IntentResult。

        流程：关键词快速路径 → LLM 路径 → 后处理补漏 → 失败降级。

        trace: 可选链路追踪回调，记录意图解析的提示词与 LLM 原始输出。
        """
        text = text.strip()
        if not text:
            return IntentResult(intent=IntentType.CHAT, scene="daily")

        # 1) 关键词快速路径（省掉一次 LLM）
        keyword_result = _keyword_heuristic(text)
        if keyword_result is not None:
            logger.debug("[IntentParser] 关键词命中：%s", keyword_result.intent.value)
            if trace:
                trace(
                    "intent_parse",
                    input=text,
                    output=f"（关键词快速路径，未调用 LLM）intent={keyword_result.intent.value}",
                    notes="关键词启发式命中，跳过 LLM",
                )
            # 关键词路径也要做后处理补漏（domain_filter 等）
            return _post_enrichment(keyword_result, text)

        # 2) LLM 路径
        try:
            messages = build_intent_messages(text, working_window)
            intent_prompt = "\n\n".join(
                f"【{m.get('role', '')}】\n{m.get('content', '')}" for m in messages
            )
            raw = await self._llm.chat(
                messages,
                temperature=0.1,  # 意图识别要确定性
                max_tokens=300,
                json_mode=True,
                timeout=10.0,
            )
            # 链路追踪：记录意图解析提示词 + LLM 原始输出
            if trace:
                trace(
                    "intent_parse",
                    input=text,
                    output=f"LLM 原始输出：\n{raw}",
                    notes=intent_prompt,
                )
            obj = _extract_json(raw)
            if obj is None:
                logger.warning("[IntentParser] LLM 输出解析失败：%s", raw[:80])
                # 降级但先走后处理
                fallback = IntentResult(intent=IntentType.CHAT, scene="daily")
                return _post_enrichment(fallback, text)

            parsed = _parse_llm_output(obj)
            if parsed is None:
                fallback = IntentResult(intent=IntentType.CHAT, scene="daily")
                return _post_enrichment(fallback, text)

            return _post_enrichment(parsed, text)

        except Exception as e:  # noqa: BLE001
            logger.warning("[IntentParser] LLM 调用失败（降级为 chat+daily）：%s", e)
            if trace:
                trace(
                    "intent_parse",
                    input=text,
                    output=f"（LLM 调用失败：{e}，降级为 chat+daily）",
                    notes="意图解析 LLM 调用异常",
                )
            fallback = IntentResult(intent=IntentType.CHAT, scene="daily")
            return _post_enrichment(fallback, text)
