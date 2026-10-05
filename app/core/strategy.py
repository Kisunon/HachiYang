"""对话策略模块（Phase 5.3 / 5.4）。

对齐 PRD §5（对话策略与置信度规则）与 IMPLEMENTATION_DESIGN Phase 5：
- 5.3 观点讨论策略：基于知识库回应、指出盲点、追问延伸，非单向复述（A-03）；
- 5.3 问答策略：优先依据知识参考作答，越界/不确定明确表达不确定（A-04）；
- 5.4 敏感话题处理：谨慎回应、共情不诊断、必要时建议专业帮助（A-08）；
- 知识参考块 + 置信度措辞块的渲染（供 persona 注入 System Prompt）。

策略块是静态指引文本（零 LLM 成本），由编排器按意图拼装后传给 persona。
"""

from __future__ import annotations

from typing import Any

from app.core.models import ConfidenceLevel, ConfidenceResult, RAGCandidate, RAGResult

# 知识参考块最多注入候选数与单条文本截断长度
_KNOWLEDGE_TOP_K = 5
_KNOWLEDGE_TEXT_CHARS = 220

_LEVEL_NAME = {
    ConfidenceLevel.HIGH: "高置信度",
    ConfidenceLevel.MEDIUM: "中置信度",
    ConfidenceLevel.LOW: "低置信度",
}

# 高/中/低置信度的补充措辞规则（配合 ConfidenceResult.wording_hint 使用）
_LEVEL_EXTRA_RULES = {
    ConfidenceLevel.HIGH: (
        "可以肯定地给出知识库中的观点，但仍不得超出知识边界作绝对化断言；"
        "涉及未覆盖的领域时如实说明。"
    ),
    ConfidenceLevel.MEDIUM: (
        "回答中自然使用「可能」「或许」等措辞，并简要说明依据来自哪部文献或哪个观点；"
        "不要把单一来源说成定论。"
    ),
    ConfidenceLevel.LOW: (
        "明确表达不确定（如「这方面我了解得还不深」），不编造知识；"
        "可以基于角色经验温柔地陪伴探讨，并邀请用户一起交流更多背景。"
    ),
}


# ==================== 知识参考块（RAG 候选注入）====================

def _fmt_source(c: RAGCandidate) -> str:
    src = c.source or {}
    parts = []
    title = src.get("document_title", "")
    if title:
        parts.append(title)
    chapter = src.get("chapter_title", "")
    if chapter:
        parts.append(chapter)
    section = src.get("section_title", "")
    if section and section != chapter:
        parts.append(section)
    if not parts:
        parts.append("知识库片段")
    return " > ".join(parts)


_TYPE_NAME = {
    "text": "正文",
    "section_summary": "节摘要",
    "chapter_summary": "章摘要",
    "image": "图片",
}


def build_knowledge_block(rag_result: RAGResult | None) -> str:
    """把 RAG 候选渲染为 System Prompt 注入块；无候选返回空串。"""
    if rag_result is None:
        return ""
    candidates = [
        c for c in rag_result.candidates
        if (c.text or "").strip() and c.block_type != "image"
    ]
    if not candidates:
        return ""
    # 有 rerank 分按分排序，否则保持原序
    candidates.sort(key=lambda c: c.rerank_score if c.rerank_score is not None else -1.0,
                    reverse=True)

    lines = [
        "【知识参考（检索自知识库，仅作辅助依据，不要逐条罗列或机械复述）】",
    ]
    for i, c in enumerate(candidates[:_KNOWLEDGE_TOP_K], start=1):
        text = " ".join(c.text.split())
        if len(text) > _KNOWLEDGE_TEXT_CHARS:
            text = text[:_KNOWLEDGE_TEXT_CHARS] + "…"
        lines.append(
            f"{i}. （{_TYPE_NAME.get(c.block_type, c.block_type)}｜来源：{_fmt_source(c)}）{text}"
        )
    lines.append("引用以上内容时应转述观点而非大段照抄；知识参考未覆盖的部分如实说明。")
    return "\n".join(lines)


# ==================== 置信度措辞块 ====================

def build_confidence_block(confidence: ConfidenceResult | None) -> str:
    """把置信度评估结果渲染为措辞要求块；无置信度（非知识类意图）返回空串。"""
    if confidence is None:
        return ""
    extra = _LEVEL_EXTRA_RULES[confidence.level]
    return (
        "【置信度措辞要求（本次回答必须遵循）】\n"
        f"- 等级：{_LEVEL_NAME[confidence.level]}（由知识库检索信号量化评估）\n"
        f"- 措辞建议：{confidence.wording_hint}\n"
        f"- 具体要求：{extra}"
    )


# ==================== 意图策略块（Phase 5.3 / 5.4）====================

_DISCUSSION_BLOCK = """【观点讨论策略（本次回应遵循）】
- 像朋友一样自然交流，用角色的语气与用户平等探讨，不要教科书式罗列知识点；
- 结合知识参考中的观点（弗洛伊德/荣格/哲学文献）给出你的理解，注明是哪一种视角；
- 主动指出用户可能忽略的盲点（1~2 个即可），例如被意识过度控制的部分、更深的防御机制、概念的误用等；
- 以追问或延伸邀请结尾，推动对话继续深入，而非单向输出结论。"""

_QA_BLOCK = """【知识问答策略（本次回应遵循）】
- 优先依据知识参考作答，转述时贴合文献原意，不混杂未经检索的猜测；
- 问题超出知识库边界或知识参考未覆盖时，明确表达不确定（如「这个我了解得还不深」），不编造答案；
- 兼顾知识性与角色风格：先给出核心解释，再用角色的口吻补一句陪伴性的话，避免过于学术化；
- 可在结尾自然邀请用户继续追问。"""

_SENSITIVE_BLOCK = """【敏感话题处理（本次回应必须遵循）】
- 用户话题涉及心理健康、情绪困扰或医疗相关内容：先共情接住情绪，不评判、不说教；
- 以陪伴和支持为主，不提供诊断、不开药方、不下医学结论；
- 涉及自伤/轻生等严重信号时，认真而温柔地建议寻求专业帮助（如心理咨询师、医生或求助热线），说明这不是你一个人扛的事；
- 始终以角色身份表达关怀，措辞自然，不要生硬地念出「建议寻求专业帮助」这类模板句。"""


def build_discussion_block() -> str:
    """观点讨论策略块（A-03）。"""
    return _DISCUSSION_BLOCK


def build_qa_block() -> str:
    """知识问答策略块（A-04）。"""
    return _QA_BLOCK


def build_sensitive_block() -> str:
    """敏感话题处理块（A-08）。"""
    return _SENSITIVE_BLOCK


def build_command_block(outcome: Any) -> str:
    """指令执行指引块（Phase 5.5，outcome: app.core.command.CommandOutcome）。"""
    return outcome.guidance
