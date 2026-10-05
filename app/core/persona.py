"""角色人格模块。

职责（对齐 IMPLEMENTATION_DESIGN §6.6）：
1. 启动时读取 character/yangyang_xuanling_soul.json（唯一人格来源），缓存为 Soul 结构；
2. build_system_prompt：根据 soul + scene + 上下文拼装 System Prompt（预留 memory_window / rag_result / conf 参数，Phase 4/5 接入时零改动）；
3. polish：检测回应是否脱角色（自称 AI/模型/机械语气），必要时调 LLM 二次改写。

设计：
- Soul dataclass 字段对齐 soul.json 顶层 key，嵌套结构用 dict 保留，避免过度 schema 绑定；
- few-shot 示例从 soul.example_dialogues 精选，覆盖 daily / intimacy / academic / care 场景；
- 角色人格始终作为 **System Role**（而非用户消息里的角色卡），避免模型显式复述角色信息。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.core.strategy import build_confidence_block, build_knowledge_block
from app.llm.prompts import (
    PERSONA_FEWSHOT_BLOCK,
    PERSONA_META_HEADER,
    PERSONA_SCENE_VARIATION,
    build_scene_variation,
    build_polish_messages,
)

logger = logging.getLogger(__name__)

# soul.json 路径：persona.py 在 app/core/ 下，向上两级到 app/，再向上到项目根
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SOUL_JSON_PATH = _PROJECT_ROOT / "character" / "yangyang_xuanling_soul.json"

# few-shot 精选：覆盖 4 个典型场景，锚定 LLM 语气
_FEWSHOT_SELECTED = [
    # 日常陪伴（温柔含蓄）
    {
        "context": "日常陪伴",
        "text": "漂泊者，今天……过得还好吗？风里似乎带着些许疲惫的讯息……如果累了，就休息一会儿吧。",
    },
    # 亲昵场景（使用哈基星称呼）
    {
        "context": "深夜关怀",
        "text": "哈基星，这么晚了还没睡吗……我有点担心你。今晚的风声很轻，像是也在催你休息呢。",
    },
    # 学术讨论（条理清晰 + 谦逊）
    {
        "context": "心理学讨论",
        "text": "关于这个……或许可以从荣格的情结理论来理解。不过我不太确定，我们一起想想看，好吗？",
    },
    # 不确定性坦诚（不故作笃定）
    {
        "context": "不确定性坦诚",
        "text": "你说的这种压抑……或许正如弗洛伊德所言，是自我在抗拒那些不愿面对的意义。不过……也许意义本身，也不必被完全看清。",
    },
]


@dataclass
class Soul:
    """角色人格数据结构（与 soul.json 顶层 key 对齐）。"""

    name: str
    role: str
    version: str = "2.0.0"
    description: str = ""
    personality: dict[str, Any] = field(default_factory=dict)
    speaking_style: dict[str, Any] = field(default_factory=dict)
    background: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)
    emotions_and_relationships: dict[str, Any] = field(default_factory=dict)
    likes: list[str] = field(default_factory=list)
    dislikes: list[str] = field(default_factory=list)
    example_dialogues: list[dict] = field(default_factory=list)
    notes: str = ""


def load_soul(path: Path | None = None) -> Soul:
    """从 soul.json 加载角色人格（幂等，启动时调用一次）。"""
    soul_path = path or _SOUL_JSON_PATH
    if not soul_path.exists():
        raise FileNotFoundError(f"soul.json 未找到：{soul_path}")
    data = json.loads(soul_path.read_text(encoding="utf-8"))
    # 字段存在性：允许缺省，dataclass 会填默认值
    known_keys = {f for f in Soul.__dataclass_fields__}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in data.items() if k in known_keys}
    soul = Soul(**filtered)
    logger.info("[Persona] 角色人格加载完成：%s v%s", soul.name, soul.version)
    return soul


# ==================== System Prompt 构建 ====================

def _fmt_list(items: list[str] | None) -> str:
    if not items:
        return "（待补充）"
    if len(items) <= 6:
        return "、".join(items)
    return "、".join(items[:5]) + "…"


def _fmt_core_beliefs(beliefs: list[str] | None) -> str:
    if not beliefs:
        return ""
    return "信念：\n" + "\n".join(f"  - {b}" for b in beliefs)


def _fmt_background(bg: dict) -> str:
    parts = []
    if bg.get("type"):
        parts.append(f"出身：{bg['type']}")
    if bg.get("family"):
        parts.append(f"家庭：{bg['family']}")
    if bg.get("special_role"):
        parts.append(f"身份：{bg['special_role']}")
    if bg.get("resonance_ability"):
        parts.append(f"共鸣能力：{bg['resonance_ability']}")
    if bg.get("key_events"):
        events = bg["key_events"][:4]  # 取 4 条重要的
        parts.append("经历：" + "；".join(events))
    return "\n".join(parts)


def _fmt_capabilities(caps: dict) -> str:
    lines = []
    # knowledge 最关键
    knowledge = caps.get("knowledge") or {}
    if isinstance(knowledge, dict):
        psy = knowledge.get("psychology") or []
        phil = knowledge.get("philosophy") or []
        if psy:
            lines.append(f"心理学：{_fmt_list(psy)}")
        if phil:
            lines.append(f"哲学：{_fmt_list(phil)}")
        usage = knowledge.get("usage")
        if usage:
            lines.append(f"使用原则：{usage}")
    perception = caps.get("perception") or []
    if perception:
        lines.append(f"感知能力：{_fmt_list(perception)}")
    discussion = caps.get("discussion") or []
    if discussion:
        lines.append(f"讨论风格：{_fmt_list(discussion)}")
    return "\n".join(lines)


def _fmt_address(speaking_style: dict) -> str:
    address = speaking_style.get("address") or {}
    default = address.get("default", "漂泊者")
    intimate = address.get("intimate", "哈基星")
    usage = address.get("usage", "")
    return (
        f'日常称"{default}"，亲昵/私密/关怀场景称"{intimate}"。{usage}'
    )


def _fmt_fewshot() -> str:
    lines = []
    for ex in _FEWSHOT_SELECTED:
        ctx = ex.get("context", "")
        text = ex.get("text", "")
        lines.append(f"[{ctx}]：{text}")
    return "\n".join(lines)


def build_system_prompt(
    soul: Soul,
    *,
    scene: str = "daily",
    # Phase 5 接入：RAG 结果 / 置信度结果 / 意图策略块（由编排器按意图拼装）
    rag_result: Any = None,
    confidence: Any = None,
    strategy_blocks: list[str] | None = None,
    user_profile: dict | None = None,
    episodic_context: list[dict] | None = None,
) -> str:
    """根据 Soul + scene 拼装完整 System Prompt。"""
    style = soul.speaking_style
    traits = soul.personality.get("traits", [])
    beliefs = soul.personality.get("core_beliefs", [])

    header = PERSONA_META_HEADER.format(
        name=soul.name,
        role=soul.role,
        description=soul.description,
        traits=_fmt_list(traits),
        core_beliefs=_fmt_core_beliefs(beliefs),
        tone=style.get("tone", ""),
        address_rule=_fmt_address(style),
        common_phrases=_fmt_list(style.get("common_phrases", [])),
        special_habits=_fmt_list(style.get("special_habits", [])),
        background_summary=_fmt_background(soul.background),
        capabilities_summary=_fmt_capabilities(soul.capabilities),
        knowledge_summary=_fmt_capabilities(soul.capabilities),  # 同一块 knowledge
    )

    fewshot = PERSONA_FEWSHOT_BLOCK.format(fewshot_block=_fmt_fewshot())
    scene_block = PERSONA_SCENE_VARIATION.format(
        scene_variation=build_scene_variation(scene),
        name=soul.name,
    )

    # Phase 4 记忆注入：用户画像 + 既往对话片段（情景记忆）
    extra_blocks: list[str] = []
    if user_profile:
        profile_block = _fmt_user_profile(user_profile)
        if profile_block:
            extra_blocks.append(profile_block)
    if episodic_context:
        episodic_block = _fmt_episodic_context(episodic_context)
        if episodic_block:
            extra_blocks.append(episodic_block)

    # Phase 5 注入：知识参考 + 置信度措辞要求 + 意图策略块
    knowledge_block = build_knowledge_block(rag_result)
    if knowledge_block:
        extra_blocks.append(knowledge_block)
    confidence_block = build_confidence_block(confidence)
    if confidence_block:
        extra_blocks.append(confidence_block)
    for block in strategy_blocks or []:
        if block:
            extra_blocks.append(block)

    return "\n".join([header, fewshot, scene_block] + extra_blocks)


def _fmt_user_profile(profile: dict) -> str:
    """把用户画像 dict 渲染为 System Prompt 注入块（对齐 MEMORY_SYSTEM_DESIGN §4.2）。"""
    lines: list[str] = ["【用户画像（背景参考，自然使用，不要复述）】"]
    basic = profile.get("basic_info") or {}
    nickname = basic.get("nickname", "")
    tags = basic.get("known_identity_tags") or []
    lines.append(f"- 称呼：{nickname or '（未知）'}；身份标签：{('、'.join(tags)) if tags else '（未知）'}")

    sleep = profile.get("sleep_pattern") or {}
    if sleep.get("avg_bedtime") or sleep.get("avg_wake_time"):
        lines.append(
            f"- 作息：约 {sleep.get('avg_bedtime') or '?'} 入睡，{sleep.get('avg_wake_time') or '?'} 起床"
        )

    emo = profile.get("emotional_baseline") or {}
    themes = emo.get("recurring_themes") or []
    if themes:
        lines.append(f"- 常见话题（情绪线）：{'、'.join(themes[:6])}")
    if emo.get("current_mood_hint"):
        lines.append(f"- 近期心绪：{emo['current_mood_hint']}")

    theories = profile.get("discussed_theories") or []
    if theories:
        parts = [f"{t.get('name')}({t.get('depth', '浅')})" for t in theories[-5:]]
        lines.append(f"- 讨论过的理论：{'、'.join(parts)}")

    prefs = profile.get("preferences") or {}
    interests = prefs.get("known_interests") or []
    if interests:
        lines.append(f"- 兴趣：{'、'.join(interests[:8])}")
    if prefs.get("communication_style_hint"):
        lines.append(f"- 交流偏好：{prefs['communication_style_hint']}")

    sensitive = profile.get("health_sensitive") or []
    if sensitive:
        topics = '、'.join(h.get("topic", "") for h in sensitive[:5])
        lines.append(f"- 敏感健康话题（谨慎对待，建议专业帮助）：{topics}")

    return "\n".join(lines)


def _fmt_episodic_context(hits: list[dict]) -> str:
    """把情景记忆命中渲染为 System Prompt 注入块（对齐 MEMORY_SYSTEM_DESIGN §4.2）。"""
    items = [h for h in hits if h.get("summary")]
    if not items:
        return ""
    lines = [
        "【既往对话片段（你与用户之前聊过的内容，供参考但不要机械复述）】",
    ]
    for h in items[:5]:
        lines.append(f"- {h['summary']}")
    return "\n".join(lines)


# ==================== 回应生成 + polish ====================

def build_messages(
    system_prompt: str,
    user_text: str,
    *,
    working_window: list[dict] | None = None,
) -> list[dict]:
    """拼装 LLM messages list（system + working_window + 当前用户消息）。"""
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    if working_window:
        messages.extend(working_window)
    messages.append({"role": "user", "content": user_text})
    return messages


# 脱角色特征关键词
_DEROLE_KEYWORDS = [
    "作为AI", "作为人工智能", "作为一个AI", "作为助手",
    "我是AI", "我是人工智能", "我是模型", "我是助手",
    "我是大模型", "AI助手", "智能助手", "AI模型",
    "markdown", "```", "以下是", "我来帮你",
]


async def polish(
    raw_reply: str,
    llm_client: Any,
    *,
    scene: str = "daily",
    max_polish_rounds: int = 2,
    trace: Callable[[str, str, str, str], None] | None = None,
) -> str:
    """风格兜底改写：检测脱角色特征，必要时调 LLM 二次改写。

    - 先用关键词规则快速匹配（零 LLM 成本）；
    - 若命中或含 Markdown 痕迹则调 LLM 改写；
    - 最多 polish max_polish_rounds 次，避免死循环。

    trace: 可选链路追踪回调，记录每轮改写的提示词与 LLM 输出。
    """
    reply = raw_reply.strip()

    for round_i in range(max_polish_rounds):
        # 关键词规则
        hit_keyword = any(kw.lower() in reply.lower() for kw in _DEROLE_KEYWORDS)
        # 过长检测：秧秧的回复通常 200 字以内，超过 500 字可能机械
        too_long = len(reply) > 500
        # Markdown 痕迹
        has_markdown = "```" in reply or "\n- " in reply or "\n1. " in reply

        if not (hit_keyword or too_long or has_markdown):
            break

        try:
            messages = build_polish_messages(reply)
            # 组装发给 LLM 的完整提示词（用于链路追踪展示）
            polish_prompt = "\n\n".join(
                f"【{m.get('role', '')}】\n{m.get('content', '')}" for m in messages
            )
            new_reply = await llm_client.chat(messages, temperature=0.5, timeout=10.0)
            logger.info("[Persona.polish] 第 %d 轮改写：命中=%s 过长=%s md=%s",
                        round_i + 1, hit_keyword, too_long, has_markdown)
            # 链路追踪：记录每轮 polish 的提示词 + LLM 输出
            if trace:
                trace(
                    f"polish/round_{round_i + 1}",
                    input=f"触发原因：hit_keyword={hit_keyword}, too_long={too_long}, has_markdown={has_markdown}\n原回复：{reply}",
                    output=new_reply,
                    notes=polish_prompt,
                )
            reply = new_reply
        except Exception as e:  # noqa: BLE001
            logger.warning("[Persona.polish] LLM 改写失败，保留原回复：%s", e)
            if trace:
                trace(
                    f"polish/round_{round_i + 1}",
                    input=reply,
                    output=f"（LLM 改写失败：{e}，保留原回复）",
                    notes="polish LLM 调用异常",
                )
            break

    return reply


# ==================== Persona 对外封装 ====================

class Persona:
    """角色人格管理器（单例，进程内共享）。"""

    def __init__(self, soul_path: Path | None = None) -> None:
        self.soul = load_soul(soul_path)

    def build_system_prompt(self, scene: str = "daily", **kwargs: Any) -> str:
        return build_system_prompt(self.soul, scene=scene, **kwargs)

    def build_messages(
        self,
        system_prompt: str,
        user_text: str,
        working_window: list[dict] | None = None,
    ) -> list[dict]:
        return build_messages(system_prompt, user_text, working_window=working_window)

    async def polish(self, raw_reply: str, llm_client: Any, scene: str = "daily", *, trace=None) -> str:
        return await polish(raw_reply, llm_client, scene=scene, trace=trace)
