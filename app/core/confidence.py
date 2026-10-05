"""置信度评估模块（Phase 5.1）。

对齐 IMPLEMENTATION_DESIGN §6.7 / PRD §5.2 / RAG_SYSTEM_DESIGN §5.7：
- 输入 RAG 检索输出的结构化 ConfidenceSignals，**量化映射**至高/中/低置信度，
  不依赖 LLM 自我评估；
- 输出 ConfidenceResult（等级 + 措辞建议），供 persona 注入 System Prompt；
- 非知识类意图（chat/care/command）不调用本模块，走默认角色化回应。

映射规则（PRD §5.2）：
- 低置信度  ↔ total_hits = 0 或 max_rerank_score < 0.6；
- 高置信度  ↔ total_hits ≥ 3 且 max_rerank_score ≥ 0.85 且 coverage_estimate = "high"
              且 section_coverage ≥ 2；domain_filter 未启用时还要求 source_diversity ≥ 2
              （domain_filter 启用时该条件降级，避免单一领域过滤误伤）；
- 中置信度  ↔ 命中但不满足全部高置信度条件。
"""

from __future__ import annotations

from app.core.models import ConfidenceLevel, ConfidenceResult, ConfidenceSignals

# ---- 映射阈值（集中定义，便于验收调参）----
_LOW_SCORE = 0.6        # max_rerank_score 低于此值 → 低置信度
_HIGH_SCORE = 0.85      # 高置信度要求的最低 rerank 分
_MIN_HIGH_HITS = 3      # 高置信度要求的最低命中数
_MIN_HIGH_SECTIONS = 2  # 高置信度要求的最低节覆盖数
_MIN_HIGH_DIVERSITY = 2  # 高置信度要求的最低来源多样性（未启用 domain_filter 时）

_WORDING_LOW = "必须使用「可能」「或许」等不确定措辞，主动表达不确定，必要时邀请用户进一步探讨"
_WORDING_MEDIUM = "使用「可能」「或许」等不确定措辞，并简要说明依据"
_WORDING_HIGH = "可用肯定语气作答，但不得超出知识边界作绝对化断言"


def assess(signals: ConfidenceSignals) -> ConfidenceResult:
    """把 RAG 置信度信号量化映射为置信度等级 + 措辞建议。"""
    s = signals

    # 低置信度：无命中或关联度过低
    if s.total_hits == 0 or s.max_rerank_score < _LOW_SCORE:
        return ConfidenceResult(level=ConfidenceLevel.LOW, wording_hint=_WORDING_LOW)

    # 高置信度条件（domain_filter 启用时 source_diversity 条件降级）
    high_ok = (
        s.total_hits >= _MIN_HIGH_HITS
        and s.max_rerank_score >= _HIGH_SCORE
        and s.coverage_estimate == "high"
        and s.section_coverage >= _MIN_HIGH_SECTIONS
    )
    if not s.domain_filter_applied:
        high_ok = high_ok and s.source_diversity >= _MIN_HIGH_DIVERSITY

    if high_ok:
        return ConfidenceResult(level=ConfidenceLevel.HIGH, wording_hint=_WORDING_HIGH)
    return ConfidenceResult(level=ConfidenceLevel.MEDIUM, wording_hint=_WORDING_MEDIUM)
