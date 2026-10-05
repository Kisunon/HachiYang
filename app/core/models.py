"""统一数据结构。

对齐 docs/IMPLEMENTATION_DESIGN.md §5.3：
- 渠道层统一请求/回应（UnifiedRequest / UnifiedResponse）
- RAG 候选与检索结果（RAGCandidate / ConfidenceSignals / RAGResult，对齐 RAG_SYSTEM_DESIGN §5.7）
- 记忆上下文（MemoryContext，对齐 MEMORY_SYSTEM_DESIGN §3.1）
- 置信度结果（ConfidenceLevel / ConfidenceResult）

这些结构是渠道层、编排器、RAG、记忆之间的契约，Phase 0 先定义，
后续各阶段模块直接引用，避免接口漂移。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ==================== 渠道层统一契约 ====================

class UnifiedRequest(BaseModel):
    """渠道层统一用户请求（飞书 / 本地调试台均转换为此结构）。"""

    user_id: str = Field(description='分区键，格式 "chat_id:open_id"；本地调试台为 "local:xxx"')
    chat_id: str = Field(description="飞书 chat_id（主动推送用）；本地调试台可为 local")
    open_id: str = Field(description="飞书用户 open_id；本地调试台可为 debug")
    text: str = Field(description="用户消息文本")
    is_group: bool = Field(default=False, description="是否群聊（群聊需 @ 触发）")


class UnifiedResponse(BaseModel):
    """渠道层统一回应。"""

    user_id: str
    text: str = Field(description="最终角色化文本")
    trace: list[dict] | None = Field(
        default=None,
        description="调试链路追踪（仅本地调试台 collect_trace=True 时填充，渠道层恒为 None）",
    )


# ==================== RAG 检索相关 ====================

class RAGCandidate(BaseModel):
    """单条 RAG 检索候选（段落 / 章节摘要 / 图片块）。"""

    block_id: str
    block_type: str = Field(description="text / image / section_summary")
    text: str
    source: dict = Field(
        default_factory=dict,
        description="document_title / chapter_title / section_title / page_range / language / domain",
    )
    vector_score: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None
    hit_count: int = 1
    retrieval_source: str = Field(description="vector / keyword / parent")
    is_term_hit: bool = False


class ConfidenceSignals(BaseModel):
    """RAG 检索置信度信号（对齐 RAG_SYSTEM_DESIGN §5.7）。"""

    total_hits: int = Field(default=0, description="融合后候选总数")
    double_hits: int = Field(default=0, description="向量+关键词双命中数")
    max_rerank_score: float = Field(default=0.0)
    avg_rerank_score: float = Field(default=0.0)
    source_diversity: int = Field(default=0, description="命中来源文档数")
    section_coverage: int = Field(default=0, description="命中覆盖的章节数")
    coverage_estimate: str = Field(default="low", description="high / medium / low")
    languages_hit: list[str] = Field(default_factory=list)
    domain_filter_applied: list[str] | None = Field(
        default=None, description="本次检索应用的领域软过滤；None 表示未过滤"
    )
    term_hits_count: int = Field(default=0, description="术语表命中数")


class RAGResult(BaseModel):
    """RAG 检索管线输出。"""

    candidates: list[RAGCandidate] = Field(default_factory=list)
    confidence_signals: ConfidenceSignals = Field(default_factory=ConfidenceSignals)


# ==================== 记忆相关 ====================

class MemoryContext(BaseModel):
    """记忆读取结果（对齐 MEMORY_SYSTEM_DESIGN §3.1）。"""

    working_window: list[dict] = Field(
        default_factory=list, description="工作记忆窗口 [{role, content}]，最近 N 轮"
    )
    episodic_hits: list[dict] = Field(
        default_factory=list, description="情景记忆命中 [{summary, topics, relevance, ...}]"
    )
    user_profile: dict = Field(default_factory=dict, description="用户画像（含 sleep_pattern）")


# ==================== 置信度评估 ====================

class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ConfidenceResult(BaseModel):
    """置信度评估结果 + 措辞建议。"""

    level: ConfidenceLevel
    wording_hint: str = Field(description='措辞建议，如「必须使用"可能/或许"」')


# ==================== 意图解析（Phase 2 使用，契约先行）====================

class IntentType(str, Enum):
    DISCUSSION = "discussion"  # 观点讨论
    QA = "qa"                  # 知识问答
    CHAT = "chat"              # 日常闲聊
    COMMAND = "command"        # 指令（提醒/记录/查询）
    CARE = "care"              # 早晚安/关怀


class IntentResult(BaseModel):
    """意图解析输出（JSON 模式），对齐 IMPLEMENTATION_DESIGN §6.5。"""

    intent: IntentType
    scene: str = Field(default="daily", description="daily / academic / sadness / intimate / battle")
    requires_knowledge: bool = Field(default=False)
    domain_filter: list[str] | None = Field(default=None, description='如 ["jung"] / ["freud"]')
    is_sensitive: bool = Field(default=False, description="心理健康/医疗等敏感话题")
    slots: dict = Field(default_factory=dict, description='指令类参数，如 {"reminder_time": "..."}')
