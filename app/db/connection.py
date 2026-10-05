"""存储基础设施：SQLite 连接管理 + ChromaDB 初始化。

对齐 docs/IMPLEMENTATION_DESIGN.md §4/§5.2 与 Phase 0 任务 0.4：
- SQLite（sqlite3 标准库）：会话/工作记忆/情景记忆元数据/画像/RAG 元数据 + FTS5 全文索引；
- ChromaDB（persistent client）：向量存储，两个 Collection ——
  * ``rag_knowledge``：知识库文献（段落/章节摘要/图片块）
  * ``episodic_memory``：情景记忆摘要
  两者与 SQLite 共同位于 ``.data/``（已在 .gitignore）。

注意：ChromaDB 写入/查询时由调用方显式传入 doubao-embedding-vision 向量，
不使用 ChromaDB 默认的本地 Embedding 函数。
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

# v1.6 三阶段流水线新增列（旧库幂等迁移：表已存在时 CREATE TABLE IF NOT EXISTS 不会加列）
# 格式：(表名, 列名, 列定义)
_COLUMN_MIGRATIONS: list[tuple[str, str, str]] = [
    ("rag_documents", "chapter_summary_count", "INTEGER DEFAULT 0"),
    ("rag_documents", "volume_count", "INTEGER DEFAULT 1"),
    ("rag_documents", "volumes_json", "TEXT"),
    ("rag_blocks", "volume_title", "TEXT"),
    ("rag_blocks", "volume_index", "INTEGER"),
    ("rag_blocks", "subsection_title", "TEXT"),
    ("rag_blocks", "parent_chapter_summary_id", "TEXT"),
    ("rag_blocks", "source_anchor_in_md", "TEXT"),
    ("rag_blocks", "extracted_md_path", "TEXT"),
]


def _apply_column_migrations(conn: sqlite3.Connection) -> list[str]:
    """对旧库幂等执行 ALTER TABLE ADD COLUMN（新列已存在则跳过）。"""
    applied: list[str] = []
    for table, column, ddl in _COLUMN_MIGRATIONS:
        cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if not cols:
            continue  # 表尚不存在（schema.sql 执行后会按新定义创建）
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
            applied.append(f"{table}.{column}")
    return applied

# ChromaDB Collection 定义（名称 -> 用途），对齐 §5.2
CHROMA_COLLECTIONS: dict[str, str] = {
    "rag_knowledge": "知识库文献（段落/章节摘要/图片块）",
    "episodic_memory": "情景记忆摘要",
}


# ==================== SQLite ====================

def connect_db(settings: Settings | None = None) -> sqlite3.Connection:
    """新建一个 SQLite 连接（WAL 模式 + Row 工厂 + 外键开启）。

    调用方负责关闭；推荐使用 :func:`db_conn` 上下文管理器。
    """
    settings = settings or get_settings()
    settings.ensure_dirs()
    conn = sqlite3.connect(
        str(settings.sqlite_file),
        timeout=30.0,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=30000;")
    return conn


@contextmanager
def db_conn(settings: Settings | None = None) -> Iterator[sqlite3.Connection]:
    """SQLite 连接上下文管理器：正常提交，异常回滚，最终关闭。"""
    conn = connect_db(settings)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(settings: Settings | None = None) -> Path:
    """执行 schema.sql 建表（幂等），返回数据库文件路径。"""
    settings = settings or get_settings()
    settings.ensure_dirs()
    ddl = _SCHEMA_PATH.read_text(encoding="utf-8")
    with db_conn(settings) as conn:
        conn.executescript(ddl)
        migrated = _apply_column_migrations(conn)
    if migrated:
        logger.info("[SQLite] 旧库列迁移完成：%s", ", ".join(migrated))
    logger.info("[SQLite] 初始化完成：%s", settings.sqlite_file)
    return settings.sqlite_file


def list_tables(settings: Settings | None = None) -> list[str]:
    """列出当前数据库中的表/虚表（验收与调试台状态监测用）。"""
    with db_conn(settings) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name"
        ).fetchall()
    return [r["name"] for r in rows]


# ==================== ChromaDB ====================

def get_chroma_client(settings: Settings | None = None) -> ClientAPI:
    """获取 ChromaDB PersistentClient（数据落盘到 chroma_dir）。"""
    settings = settings or get_settings()
    settings.ensure_dirs()
    return chromadb.PersistentClient(path=str(settings.chroma_path))


def init_vector_store(settings: Settings | None = None) -> dict[str, Collection]:
    """创建/获取两个 Collection（余弦距离），返回 {名称: Collection}。

    幂等：已存在则直接获取。调用方在 add/query 时必须显式传入 Embedding 向量。
    """
    settings = settings or get_settings()
    client = get_chroma_client(settings)
    collections: dict[str, Collection] = {}
    for name, purpose in CHROMA_COLLECTIONS.items():
        col = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine", "purpose": purpose},
        )
        collections[name] = col
        logger.info("[ChromaDB] Collection 就绪：%s（%s），现有 %d 条", name, purpose, col.count())
    return collections


# ==================== 统一初始化 ====================

def init_infrastructure(settings: Settings | None = None) -> dict:
    """初始化全部存储基础设施（SQLite + ChromaDB），返回状态摘要。

    供 app 启动流程与本地调试台调用；不依赖 ARK/飞书配置，缺失密钥也可执行。
    """
    settings = settings or get_settings()
    db_path = init_db(settings)
    collections = init_vector_store(settings)
    summary = {
        "sqlite_path": str(db_path),
        "tables": list_tables(settings),
        "chroma_dir": str(settings.chroma_path),
        "collections": {name: col.count() for name, col in collections.items()},
    }
    logger.info("[基础设施] 存储初始化完成：%s", summary)
    return summary
