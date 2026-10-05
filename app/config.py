"""配置模块：环境变量加载与启动校验。

对齐 docs/IMPLEMENTATION_DESIGN.md §6.1 与 docs/PRD.md 附录 C。

设计要点：
- 所有密钥/配置经环境变量注入，不写入代码仓库。
- **密钥安全约定**：``ARK_KEY`` 等密钥只从系统「用户环境变量」（同名变量）读取，
  禁止写入 .env 明文。pydantic-settings 加载优先级为 系统环境变量 > .env 文件，
  故系统环境变量中配置后自动生效，.env 仅承载非密钥配置。
- 必填项缺失时**启动告警**（A-11）而非静默失败/直接崩溃：
  字段均给默认值，由 ``check_startup()`` 统一汇总缺失项并输出 WARNING 日志，
  使本地调试台、存储初始化等不依赖密钥的能力仍可启动。
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# 项目根目录（app/ 的上一级），用于把相对路径锚定到项目根
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """全局配置。环境变量名与字段名大小写不敏感匹配（如 ARK_KEY -> ark_key）。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- 火山方舟（语言模型与 Embedding 共享 Endpoint，仅模型名不同）----
    ark_key: str = Field(default="", description="火山方舟 API Key（OpenAI 兼容端点；仅从系统环境变量读取，禁止写入 .env）")
    ark_base_url: str = Field(default="", description="火山方舟 API Base URL（LLM/Embedding 共用）")
    llm_model: str = Field(default="deepseek-v4-pro", description="语言模型名")
    embedding_model: str = Field(default="doubao-embedding-vision", description="Embedding 模型名")

    # ---- 飞书 ----
    feishu_app_id: str = Field(default="", description="飞书机器人 App ID")
    feishu_app_secret: str = Field(default="", description="飞书机器人 App Secret")
    feishu_encrypt_key: str | None = Field(default=None, description="飞书事件加密密钥（启用加密时必填）")
    feishu_verification_token: str = Field(default="", description="飞书事件验证 Token")
    feishu_mode: str = Field(default="websocket", description="接入模式：websocket（开发）| webhook（生产）")

    # ---- RAG ----
    rag_enable_image: bool = Field(default=False, description="RAG 图片模态开关（默认关闭）")
    rag_enable_term: bool = Field(default=False, description="RAG 术语表开关（默认关闭，本期弃置留壳旁路）")
    rag_data_dir: str = Field(default="knowledge_base", description="知识库源文件目录（阶段 0）")
    rag_extracted_dir: str = Field(default="knowledge_extracted", description="阶段 1 提取产物目录（extracted.md/.json）")
    rag_chunks_dir: str = Field(default="knowledge_chunks", description="阶段 2 切片产物目录（chunks.json）")
    section_expand_threshold: float = Field(default=0.4, description="分级链式展开：章摘要展开的节摘要二次门槛相似度")

    # ---- 记忆 ----
    memory_window_size: int = Field(default=12, description="工作记忆滑动窗口轮数（N=12）")
    memory_decay_lambda: float = Field(default=0.005, description="情景记忆衰减系数")

    # ---- 关怀调度 ----
    care_timezone: str = Field(default="Asia/Shanghai", description="关怀调度时区")
    care_bedtime_window: str = Field(default="22:30-23:59", description="就寝关怀时段")
    care_wake_window: str = Field(default="06:00-09:00", description="起床关怀时段")
    care_daily_max: int = Field(default=2, description="单用户单日主动关怀上限（就寝/起床；用户主动设置的提醒不计）")
    care_scan_interval: int = Field(default=300, description="关怀调度扫描间隔（秒）")
    care_llm_compose: bool = Field(default=True, description="关怀文案是否经 LLM 角色化生成（失败自动降级模板）")

    # ---- 存储路径 ----
    chroma_dir: str = Field(default=".data/chroma", description="ChromaDB 持久化目录")
    sqlite_path: str = Field(default=".data/hachiyang.db", description="SQLite 数据库文件路径")

    @field_validator("feishu_mode")
    @classmethod
    def _validate_feishu_mode(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("websocket", "webhook"):
            logger.warning("FEISHU_MODE=%r 非法，回退为 'websocket'（应为 websocket | webhook）", v)
            return "websocket"
        return v

    # ---- 路径便捷属性（相对路径锚定到项目根）----
    @property
    def chroma_path(self) -> Path:
        p = Path(self.chroma_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def sqlite_file(self) -> Path:
        p = Path(self.sqlite_path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def knowledge_base_path(self) -> Path:
        p = Path(self.rag_data_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def knowledge_extracted_path(self) -> Path:
        """阶段 1 产物目录（提取稿 extracted.md/.json）。"""
        p = Path(self.rag_extracted_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def knowledge_chunks_path(self) -> Path:
        """阶段 2 产物目录（切片 chunks.json）。"""
        p = Path(self.rag_chunks_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def data_dir(self) -> Path:
        """SQLite/Chroma 共同的运行时数据根目录。"""
        return self.sqlite_file.parent

    def missing_required(self) -> list[str]:
        """返回缺失的必填配置项（人类可读），供启动告警（A-11）。"""
        missing: list[str] = []
        if not self.ark_key:
            missing.append("ARK_KEY（火山方舟 API Key，LLM/Embedding 调用必需）")
        if not self.ark_base_url:
            missing.append("ARK_BASE_URL（火山方舟 API Base URL）")
        if not self.feishu_app_id:
            missing.append("FEISHU_APP_ID（飞书机器人 App ID）")
        if not self.feishu_app_secret:
            missing.append("FEISHU_APP_SECRET（飞书机器人 App Secret）")
        if not self.feishu_verification_token:
            missing.append("FEISHU_VERIFICATION_TOKEN（飞书事件验证 Token）")
        if self.feishu_mode == "webhook" and not self.feishu_encrypt_key:
            # webhook 生产模式若启用事件加密则需要 encrypt key；此处仅提示
            logger.info("webhook 模式下未配置 FEISHU_ENCRYPT_KEY；若事件订阅启用了加密请补充。")
        return missing

    def ensure_dirs(self) -> None:
        """确保运行时数据目录存在。"""
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.sqlite_file.parent.mkdir(parents=True, exist_ok=True)


# 单例：进程内共享同一份配置
_settings: Settings | None = None


def get_settings() -> Settings:
    """获取全局 Settings 单例。"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def check_startup(settings: Settings | None = None) -> list[str]:
    """启动校验：缺失必填配置时输出明确告警（A-11），返回缺失项列表。

    - 不抛异常、不静默失败：告警后继续启动，便于不依赖密钥的能力（调试台/存储）可用；
    - 配置完整时输出 INFO 确认。
    """
    settings = settings or get_settings()
    settings.ensure_dirs()
    missing = settings.missing_required()
    if missing:
        logger.warning(
            "=" * 60
            + "\n[启动告警] 以下必填配置缺失，相关功能将不可用（A-11）：\n  - "
            + "\n  - ".join(missing)
            + "\n请在环境变量或 .env 文件中配置（参考 .env.example 与 docs/PRD.md 附录 C）。\n"
            + "=" * 60
        )
    else:
        logger.info("[启动校验] 必填配置齐全：ARK / 飞书配置均已注入。")
    logger.info(
        "[运行配置] feishu_mode=%s | llm=%s | embedding=%s | rag_image=%s | db=%s",
        settings.feishu_mode,
        settings.llm_model,
        settings.embedding_model,
        settings.rag_enable_image,
        settings.sqlite_file,
    )
    return missing
