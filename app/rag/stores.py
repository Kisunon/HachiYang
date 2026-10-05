"""RAG 存储封装：ChromaDB（rag_knowledge）+ SQLite（rag_blocks / rag_fts / rag_terms / rag_documents）。

对齐 RAG_SYSTEM_DESIGN §7：
- 向量 Collection `rag_knowledge`：text / section_summary 共存，block_type 区分；
- FTS5：block_id + text + document_title + chapter_title + domain，BM25 精确术语召回；
- rag_blocks：结构化元数据 + parent_block_ids（Parent Retrieval）；
- rag_terms：术语表精确映射，供 TermLookup 毫秒级查询；
- rag_documents：文档索引状态（增量索引用）。

调用方显式传入 doubao-embedding-vision 向量，不使用 ChromaDB 默认 Embedding。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterator

from app.config import Settings, get_settings
from app.db.connection import db_conn, get_chroma_client

logger = logging.getLogger(__name__)


# ==================== 工具 ====================

def _meta_json(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False) if v is not None else "[]"


def _load_json(s: str | None, default: Any) -> Any:
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        return default


# ==================== RAGStore ====================

class RAGStore:
    """RAG 统一存储层（向量 + 关系 + 全文）。

    线程安全：SQLite 用独立连接上下文；ChromaDB PersistentClient 自身线程安全。
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._chroma = None
        self._collection = None
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """旧库幂等列迁移（v1.6 三阶段流水线新增列），避免未走 init_db 时写入失败。"""
        from app.db.connection import _apply_column_migrations, db_conn as _db_conn

        try:
            with _db_conn(self.settings) as conn:
                _apply_column_migrations(conn)
        except Exception as e:  # noqa: BLE001
            logger.warning("[RAGStore] schema 迁移检查失败：%s", e)

    @property
    def term_feature_enabled(self) -> bool:
        """术语表功能开关（v1.6 弃置留壳，默认 False）。"""
        return bool(getattr(self.settings, "rag_enable_term", False))

    # ---- ChromaDB ----

    def _get_collection(self):
        if self._collection is None:
            client = get_chroma_client(self.settings)
            self._collection = client.get_or_create_collection(
                name="rag_knowledge",
                metadata={"hnsw:space": "cosine", "purpose": "知识库文献"},
            )
        return self._collection

    def add_vectors(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """批量写入向量（幂等：ChromaDB upsert 语义）。"""
        if not ids:
            return
        col = self._get_collection()
        # ChromaDB metadata 值必须 str/int/float/bool，扁平化
        flat_metas = []
        for m in metadatas:
            fm = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in m.items()}
            flat_metas.append(fm)
        col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=flat_metas)

    def query_vector(
        self,
        embedding: list[float],
        n: int = 50,
        where: dict | None = None,
    ) -> list[dict]:
        """向量检索 Top-N，返回 [{id, document, metadata, distance}]。"""
        col = self._get_collection()
        res = col.query(query_embeddings=[embedding], n_results=n, where=where)
        out = []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for i, bid in enumerate(ids):
            # ChromaDB 余弦距离 → 相似度（1-distance）
            dist = dists[i] if i < len(dists) else 1.0
            out.append({
                "block_id": bid,
                "text": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "vector_score": max(0.0, 1.0 - dist),
            })
        return out

    # ---- rag_documents ----

    def upsert_document(
        self,
        document_id: str,
        file_path: str,
        content_hash: str,
        domain: str,
        language: str | None = None,
        status: str = "pending",
        block_count: int = 0,
        section_summary_count: int = 0,
        term_count: int = 0,
        chapter_summary_count: int = 0,
        volume_count: int = 1,
        volumes: list[dict] | None = None,
    ) -> None:
        now = int(time.time())
        with db_conn(self.settings) as conn:
            conn.execute(
                """
                INSERT INTO rag_documents
                    (document_id, file_path, content_hash, domain, language, status,
                     block_count, section_summary_count, chapter_summary_count, term_count,
                     volume_count, volumes_json, indexed_at, retry_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(document_id) DO UPDATE SET
                    file_path=excluded.file_path,
                    content_hash=excluded.content_hash,
                    domain=excluded.domain,
                    language=excluded.language,
                    status=excluded.status,
                    block_count=excluded.block_count,
                    section_summary_count=excluded.section_summary_count,
                    chapter_summary_count=excluded.chapter_summary_count,
                    term_count=excluded.term_count,
                    volume_count=excluded.volume_count,
                    volumes_json=excluded.volumes_json,
                    indexed_at=CASE WHEN excluded.status='done' THEN ? ELSE indexed_at END
                """,
                (document_id, file_path, content_hash, domain, language, status,
                 block_count, section_summary_count, chapter_summary_count, term_count,
                 volume_count, _meta_json(volumes or []), now, now),
            )

    def get_document(self, document_id: str) -> dict | None:
        with db_conn(self.settings) as conn:
            row = conn.execute(
                "SELECT * FROM rag_documents WHERE document_id=?", (document_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_document_by_path(self, file_path: str) -> dict | None:
        with db_conn(self.settings) as conn:
            row = conn.execute(
                "SELECT * FROM rag_documents WHERE file_path=?", (file_path,)
            ).fetchone()
        return dict(row) if row else None

    def list_pending_documents(self) -> list[dict]:
        with db_conn(self.settings) as conn:
            rows = conn.execute(
                "SELECT * FROM rag_documents WHERE status IN ('pending','failed') AND retry_count < 3"
            ).fetchall()
        return [dict(r) for r in rows]

    def set_document_status(self, document_id: str, status: str,
                            block_count: int = 0, section_summary_count: int = 0,
                            term_count: int = 0, chapter_summary_count: int = 0) -> None:
        now = int(time.time())
        with db_conn(self.settings) as conn:
            conn.execute(
                """UPDATE rag_documents SET status=?, block_count=?, section_summary_count=?,
                   chapter_summary_count=?, term_count=?, indexed_at=? WHERE document_id=?""",
                (status, block_count, section_summary_count, chapter_summary_count,
                 term_count, now, document_id),
            )

    def delete_document(self, document_id: str) -> None:
        """删除文档及其全部块（向量 + SQLite + FTS）。"""
        with db_conn(self.settings) as conn:
            block_ids = [r["block_id"] for r in conn.execute(
                "SELECT block_id FROM rag_blocks WHERE document_id=?", (document_id,)
            ).fetchall()]
            conn.execute("DELETE FROM rag_blocks WHERE document_id=?", (document_id,))
            # FTS5 删除（按 block_id）
            if block_ids:
                placeholders = ",".join("?" * len(block_ids))
                conn.execute(f"DELETE FROM rag_fts WHERE block_id IN ({placeholders})", block_ids)
            conn.execute("DELETE FROM rag_documents WHERE document_id=?", (document_id,))
        # 向量库删除
        if block_ids:
            try:
                self._get_collection().delete(ids=block_ids)
            except Exception as e:  # noqa: BLE001
                logger.warning("[RAGStore] 向量删除失败 document=%s: %s", document_id, e)

    def get_block_id_set(self, document_id: str) -> set[str]:
        """文档已入库的 block_id 集合（断点续跑时跳过已写块）。"""
        with db_conn(self.settings) as conn:
            rows = conn.execute(
                "SELECT block_id FROM rag_blocks WHERE document_id=?", (document_id,)
            ).fetchall()
        return {r["block_id"] for r in rows}

    def delete_summary_blocks(self, document_id: str) -> int:
        """删除文档的全部两级摘要块（section_summary / chapter_summary：SQLite + FTS + 向量）。

        断点续跑时正文块复用、摘要块每次重新生成（secs-/chs- id 带随机 uuid），
        故重跑前先清除旧摘要，避免残留。返回删除块数。
        """
        with db_conn(self.settings) as conn:
            rows = conn.execute(
                "SELECT block_id FROM rag_blocks WHERE document_id=? "
                "AND block_type IN ('section_summary','chapter_summary')",
                (document_id,),
            ).fetchall()
            ids = [r["block_id"] for r in rows]
            if not ids:
                return 0
            placeholders = ",".join("?" * len(ids))
            conn.execute(f"DELETE FROM rag_blocks WHERE block_id IN ({placeholders})", ids)
            conn.execute(f"DELETE FROM rag_fts WHERE block_id IN ({placeholders})", ids)
        try:
            self._get_collection().delete(ids=ids)
        except Exception as e:  # noqa: BLE001
            logger.warning("[RAGStore] 摘要向量删除失败 document=%s: %s", document_id, e)
        logger.info("[RAGStore] 文档 %s 清除旧摘要块 %d 个（续跑重生成）", document_id, len(ids))
        return len(ids)

    # ---- rag_blocks ----

    def add_blocks(self, blocks: list) -> None:
        """批量写入 ParagraphBlock 元数据（dataclass，v1.6 含 6 级溯源列）。"""
        if not blocks:
            return
        now = int(time.time())
        rows = []
        for b in blocks:
            rows.append((
                b.block_id, b.document_id, b.domain, b.language,
                getattr(b, "volume_title", "") or "",
                getattr(b, "volume_index", 1) or 1,
                b.chapter_title, b.section_title,
                getattr(b, "subsection_title", "") or "",
                b.page_range,
                b.block_index_in_section, b.char_count, b.block_type,
                1 if b.has_structure else 0,
                _meta_json(b.parent_block_ids),
                getattr(b, "parent_chapter_summary_id", None) or None,
                getattr(b, "source_anchor_in_md", "") or "",
                getattr(b, "extracted_md_path", "") or "",
                b.text,
            ))
        with db_conn(self.settings) as conn:
            conn.executemany(
                """
                INSERT INTO rag_blocks
                    (block_id, document_id, domain, language,
                     volume_title, volume_index,
                     chapter_title, section_title, subsection_title, page_range,
                     block_index_in_section, char_count, block_type, has_structure,
                     parent_block_ids, parent_chapter_summary_id,
                     source_anchor_in_md, extracted_md_path, text)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(block_id) DO UPDATE SET text=excluded.text,
                    char_count=excluded.char_count,
                    parent_block_ids=excluded.parent_block_ids,
                    parent_chapter_summary_id=excluded.parent_chapter_summary_id
                """,
                rows,
            )
        logger.debug("[RAGStore] 写入 %d 个 rag_blocks", len(blocks))

    def get_block(self, block_id: str) -> dict | None:
        with db_conn(self.settings) as conn:
            row = conn.execute(
                "SELECT * FROM rag_blocks WHERE block_id=?", (block_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_blocks_by_ids(self, block_ids: list[str]) -> dict[str, dict]:
        if not block_ids:
            return {}
        placeholders = ",".join("?" * len(block_ids))
        with db_conn(self.settings) as conn:
            rows = conn.execute(
                f"SELECT * FROM rag_blocks WHERE block_id IN ({placeholders})", block_ids
            ).fetchall()
        return {r["block_id"]: dict(r) for r in rows}

    def get_blocks_by_document(self, document_id: str, block_type: str | None = None) -> list[dict]:
        with db_conn(self.settings) as conn:
            if block_type:
                rows = conn.execute(
                    "SELECT * FROM rag_blocks WHERE document_id=? AND block_type=?",
                    (document_id, block_type),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM rag_blocks WHERE document_id=?", (document_id,)
                ).fetchall()
        return [dict(r) for r in rows]

    # ---- rag_fts (FTS5) ----

    def add_fts_rows(self, rows: list[dict]) -> None:
        """rows: [{block_id, text, document_title, chapter_title, domain}]。"""
        if not rows:
            return
        with db_conn(self.settings) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO rag_fts (block_id, text, document_title, chapter_title, domain)
                VALUES (?, ?, ?, ?, ?)
                """,
                [(r["block_id"], r["text"], r.get("document_title", ""),
                  r.get("chapter_title", ""), r.get("domain", "")) for r in rows],
            )

    def query_fts(self, query: str, n: int = 50, domains: list[str] | None = None) -> list[dict]:
        """BM25 全文检索 Top-N。返回 [{block_id, text, document_title, chapter_title, domain, bm25_score}]。"""
        if not query.strip():
            return []
        with db_conn(self.settings) as conn:
            try:
                if domains:
                    placeholders = ",".join("?" * len(domains))
                    sql = (
                        f"SELECT block_id, text, document_title, chapter_title, domain, "
                        f"bm25(rag_fts) AS score FROM rag_fts "
                        f"WHERE rag_fts MATCH ? AND domain IN ({placeholders}) "
                        f"ORDER BY score LIMIT ?"
                    )
                    rows = conn.execute(sql, (query, *domains, n)).fetchall()
                else:
                    sql = (
                        "SELECT block_id, text, document_title, chapter_title, domain, "
                        "bm25(rag_fts) AS score FROM rag_fts "
                        "WHERE rag_fts MATCH ? ORDER BY score LIMIT ?"
                    )
                    rows = conn.execute(sql, (query, n)).fetchall()
            except sqlite3.OperationalError as e:
                logger.warning("[RAGStore] FTS5 查询失败（%s），query=%r", e, query[:50])
                return []
        # bm25() 分数越小越相关（负值），取绝对值规范化
        results = []
        for r in rows:
            d = dict(r)
            d["bm25_score"] = abs(r["score"]) if r["score"] is not None else 0.0
            results.append(d)
        return results

    # ---- rag_terms ----

    def upsert_term(self, term: dict) -> None:
        """term: {term_id, canonical_term, synonyms, language_pairs, domain, definition_location, occurrence_block_ids, related_terms}。"""
        with db_conn(self.settings) as conn:
            conn.execute(
                """
                INSERT INTO rag_terms
                    (term_id, canonical_term, synonyms, language_pairs, domain,
                     definition_location, occurrence_block_ids, related_terms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(term_id) DO UPDATE SET
                    canonical_term=excluded.canonical_term,
                    synonyms=excluded.synonyms,
                    language_pairs=excluded.language_pairs,
                    domain=excluded.domain,
                    definition_location=excluded.definition_location,
                    occurrence_block_ids=excluded.occurrence_block_ids,
                    related_terms=excluded.related_terms
                """,
                (
                    term["term_id"], term["canonical_term"],
                    _meta_json(term.get("synonyms")),
                    _meta_json(term.get("language_pairs")),
                    term.get("domain", ""),
                    _meta_json(term.get("definition_location")),
                    _meta_json(term.get("occurrence_block_ids")),
                    _meta_json(term.get("related_terms")),
                ),
            )

    def query_terms(self, query: str, match_mode: str = "prefix") -> list[dict]:
        """术语表查询。返回 [{term_id, canonical_term, ...}]。弃置期间返回空（§3.6.1 旁路）。"""
        if not self.term_feature_enabled:
            return []
        query = query.strip()
        if not query:
            return []
        with db_conn(self.settings) as conn:
            if match_mode == "exact":
                rows = conn.execute(
                    "SELECT * FROM rag_terms WHERE canonical_term = ? COLLATE NOCASE", (query,)
                ).fetchall()
            else:
                # prefix：LIKE 前缀匹配 + canonical_term 或 synonyms 含查询词
                like = f"%{query}%"
                rows = conn.execute(
                    "SELECT * FROM rag_terms WHERE canonical_term LIKE ? OR synonyms LIKE ? COLLATE NOCASE",
                    (like, like),
                ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["synonyms"] = _load_json(r["synonyms"], [])
            d["language_pairs"] = _load_json(r["language_pairs"], {})
            d["definition_location"] = _load_json(r["definition_location"], {})
            d["occurrence_block_ids"] = _load_json(r["occurrence_block_ids"], [])
            d["related_terms"] = _load_json(r["related_terms"], [])
            out.append(d)
        return out

    # ---- 调试台：知识库人工检查（只读，Phase 3.5，对齐 §6.11.1）----

    @staticmethod
    def _domain_from_dir(path: Path) -> str:
        parent = path.parent.name.lower()
        return {"freud": "freud", "jung": "jung", "philosophy": "philosophy"}.get(parent, "philosophy")

    def list_documents_overview(self, kb_path: Path | str | None = None) -> list[dict]:
        """文档总览：rag_documents 状态 + knowledge_base/ 磁盘文件外连接。

        磁盘上有但 DB 无记录的文件标记 status='not_indexed'；
        DB 有记录但文件已消失的标记 on_disk=False。
        附带实际 SQLite 切片数（含索引中途状态）与 ChromaDB 向量数（覆盖率）。
        """
        with db_conn(self.settings) as conn:
            rows = conn.execute(
                """SELECT document_id, file_path, content_hash, domain, language, status,
                          block_count, section_summary_count, chapter_summary_count,
                          term_count, volume_count, volumes_json, indexed_at, retry_count
                   FROM rag_documents ORDER BY domain, file_path"""
            ).fetchall()
            block_stats = {
                r["document_id"]: {"blocks": r["n"], "sec_s": r["s"], "ch_s": r["c"]}
                for r in conn.execute(
                    """SELECT document_id, COUNT(*) AS n,
                              SUM(CASE WHEN block_type='section_summary' THEN 1 ELSE 0 END) AS s,
                              SUM(CASE WHEN block_type='chapter_summary' THEN 1 ELSE 0 END) AS c
                       FROM rag_blocks GROUP BY document_id"""
                ).fetchall()
            }
        db_docs = [dict(r) for r in rows]

        base = Path(kb_path) if kb_path else self.settings.knowledge_base_path
        disk_files: list[Path] = []
        if base.exists():
            disk_files = sorted(
                p for p in base.rglob("*")
                if p.is_file() and p.suffix.lower() in (".epub", ".pdf")
            )

        def _augment(d: dict, stats: dict | None) -> dict:
            stats = stats or {"blocks": 0, "sec_s": 0, "ch_s": 0}
            d["sqlite_block_count"] = stats["blocks"]
            d["sqlite_summary_count"] = (stats["sec_s"] or 0) + (stats["ch_s"] or 0)
            d["sqlite_section_summary_count"] = stats["sec_s"] or 0
            d["sqlite_chapter_summary_count"] = stats["ch_s"] or 0
            d["volumes"] = _load_json(d.get("volumes_json"), [])
            d["term_count"] = d.get("term_count", 0) if self.term_feature_enabled else 0
            d["term_feature_enabled"] = self.term_feature_enabled
            return d

        out: list[dict] = []
        indexed_paths: set[str] = set()
        for p in disk_files:
            pstr = str(p)
            match = next((d for d in db_docs if d["file_path"] == pstr), None)
            try:
                size_mb = round(p.stat().st_size / 1024 / 1024, 2)
            except OSError:
                size_mb = None
            if match:
                indexed_paths.add(pstr)
                d = match
                stats = block_stats.get(d["document_id"], {"blocks": 0, "sec_s": 0, "ch_s": 0})
                out.append({
                    **_augment(d, stats),
                    "file_name": p.name,
                    "on_disk": True,
                    "size_mb": size_mb,
                    "vector_count": self._count_vectors_safe(d["document_id"]),
                })
            else:
                out.append({
                    "document_id": None,
                    "file_path": pstr,
                    "content_hash": None,
                    "domain": self._domain_from_dir(p),
                    "language": None,
                    "status": "not_indexed",
                    "block_count": 0,
                    "section_summary_count": 0,
                    "chapter_summary_count": 0,
                    "term_count": 0,
                    "volume_count": 0,
                    "volumes": [],
                    "term_feature_enabled": self.term_feature_enabled,
                    "indexed_at": None,
                    "retry_count": 0,
                    "file_name": p.name,
                    "on_disk": True,
                    "size_mb": size_mb,
                    "sqlite_block_count": 0,
                    "sqlite_summary_count": 0,
                    "sqlite_section_summary_count": 0,
                    "sqlite_chapter_summary_count": 0,
                    "vector_count": 0,
                })

        # DB 有记录但磁盘文件已删除（阶段 3 文档 file_path 指向源 EPUB；源被移除时在此列出）
        for d in db_docs:
            if d["file_path"] in indexed_paths:
                continue
            stats = block_stats.get(d["document_id"], {"blocks": 0, "sec_s": 0, "ch_s": 0})
            out.append({
                **_augment(d, stats),
                "file_name": Path(d["file_path"]).name if d.get("file_path") else d["document_id"],
                "on_disk": bool(d.get("file_path")) and Path(d["file_path"]).exists(),
                "size_mb": None,
                "vector_count": self._count_vectors_safe(d["document_id"]),
            })
        return out

    def _count_vectors_safe(self, document_id: str) -> int:
        """ChromaDB 中该文档的向量数（仅取 id，不取向量/文档）；失败返回 -1。"""
        try:
            res = self._get_collection().get(
                where={"document_id": document_id}, include=[]
            )
            return len(res.get("ids") or [])
        except Exception as e:  # noqa: BLE001
            logger.warning("[RAGStore] 向量计数失败 document=%s: %s", document_id, e)
            return -1

    def list_chapters(self, document_id: str) -> list[dict]:
        """章节列表（按卷次 + 入库顺序）：卷名 + 章名 + 切片数 + 两级摘要数。"""
        with db_conn(self.settings) as conn:
            rows = conn.execute(
                """SELECT COALESCE(volume_title, '') AS volume_title,
                          COALESCE(volume_index, 1) AS volume_index,
                          COALESCE(chapter_title, '') AS chapter_title,
                          COUNT(*) AS block_count,
                          SUM(CASE WHEN block_type='section_summary' THEN 1 ELSE 0 END) AS section_summary_count,
                          SUM(CASE WHEN block_type='chapter_summary' THEN 1 ELSE 0 END) AS chapter_summary_count
                   FROM rag_blocks WHERE document_id=?
                   GROUP BY volume_index, chapter_title
                   ORDER BY MIN(volume_index), MIN(rowid)""",
                (document_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def count_blocks(
        self, document_id: str, chapter: str | None = None, block_type: str | None = None
    ) -> int:
        sql = "SELECT COUNT(*) FROM rag_blocks WHERE document_id=?"
        params: list[Any] = [document_id]
        if chapter is not None:
            sql += " AND COALESCE(chapter_title,'')=?"
            params.append(chapter)
        if block_type:
            sql += " AND block_type=?"
            params.append(block_type)
        with db_conn(self.settings) as conn:
            return conn.execute(sql, params).fetchone()[0]

    def paginate_blocks(
        self, document_id: str, chapter: str | None = None, block_type: str | None = None,
        limit: int = 50, offset: int = 0,
    ) -> list[dict]:
        """切片分页（文档顺序）。返回完整行（调用方自行截断 text）。"""
        sql = (
            "SELECT * FROM rag_blocks WHERE document_id=?"
        )
        params: list[Any] = [document_id]
        if chapter is not None:
            sql += " AND COALESCE(chapter_title,'')=?"
            params.append(chapter)
        if block_type:
            sql += " AND block_type=?"
            params.append(block_type)
        sql += " ORDER BY COALESCE(volume_index,1), rowid LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with db_conn(self.settings) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def find_parent_summary(self, document_id: str, block_id: str) -> dict | None:
        """反查普通段落所属的章节摘要（Parent Retrieval 反向链接）。"""
        with db_conn(self.settings) as conn:
            rows = conn.execute(
                """SELECT block_id, chapter_title, section_title, parent_block_ids
                   FROM rag_blocks WHERE document_id=? AND block_type='section_summary'""",
                (document_id,),
            ).fetchall()
        for r in rows:
            ids = _load_json(r["parent_block_ids"], [])
            if block_id in ids:
                return {
                    "block_id": r["block_id"],
                    "chapter_title": r["chapter_title"],
                    "section_title": r["section_title"],
                }
        return None

    # ---- 术语（检查用；v1.6 弃置留壳：rag_enable_term=False 时全部返回空）----

    def count_terms(self, query: str = "", domain: str | None = None) -> int:
        if not self.term_feature_enabled:
            return 0
        where, params = self._terms_where(query, domain)
        with db_conn(self.settings) as conn:
            return conn.execute(f"SELECT COUNT(*) FROM rag_terms{where}", params).fetchone()[0]

    def paginate_terms(
        self, query: str = "", domain: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[dict]:
        if not self.term_feature_enabled:
            return []
        where, params = self._terms_where(query, domain)
        sql = f"SELECT * FROM rag_terms{where} ORDER BY canonical_term LIMIT ? OFFSET ?"
        with db_conn(self.settings) as conn:
            rows = conn.execute(sql, [*params, limit, offset]).fetchall()
        return [self._term_row_to_dict(r) for r in rows]

    def get_term(self, term_id: str) -> dict | None:
        if not self.term_feature_enabled:
            return None
        with db_conn(self.settings) as conn:
            row = conn.execute(
                "SELECT * FROM rag_terms WHERE term_id=?", (term_id,)
            ).fetchone()
        return self._term_row_to_dict(row) if row else None

    @staticmethod
    def _terms_where(query: str, domain: str | None) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        q = (query or "").strip()
        if q:
            like = f"%{q}%"
            clauses.append("(canonical_term LIKE ? OR synonyms LIKE ? COLLATE NOCASE)")
            params.extend([like, like])
        if domain:
            clauses.append("domain=?")
            params.append(domain)
        return ((" WHERE " + " AND ".join(clauses)) if clauses else "", params)

    @staticmethod
    def _term_row_to_dict(row: Any) -> dict:
        d = dict(row)
        d["synonyms"] = _load_json(d.get("synonyms"), [])
        d["language_pairs"] = _load_json(d.get("language_pairs"), {})
        d["definition_location"] = _load_json(d.get("definition_location"), {})
        d["occurrence_block_ids"] = _load_json(d.get("occurrence_block_ids"), [])
        d["related_terms"] = _load_json(d.get("related_terms"), [])
        return d

    # ---- 统计 ----

    def health_check(self) -> dict:
        """健康检查（对齐 RAG §9.3）。"""
        with db_conn(self.settings) as conn:
            doc_count = conn.execute("SELECT COUNT(*) FROM rag_documents WHERE status='done'").fetchone()[0]
            total_blocks = conn.execute("SELECT COUNT(*) FROM rag_blocks").fetchone()[0]
            section_summary_count = conn.execute(
                "SELECT COUNT(*) FROM rag_blocks WHERE block_type='section_summary'"
            ).fetchone()[0]
            chapter_summary_count = conn.execute(
                "SELECT COUNT(*) FROM rag_blocks WHERE block_type='chapter_summary'"
            ).fetchone()[0]
            raw_term_count = conn.execute("SELECT COUNT(*) FROM rag_terms").fetchone()[0]
            pending = conn.execute(
                "SELECT COUNT(*) FROM rag_documents WHERE status IN ('pending','failed')"
            ).fetchone()[0]
            last_row = conn.execute(
                "SELECT indexed_at FROM rag_documents WHERE indexed_at IS NOT NULL ORDER BY indexed_at DESC LIMIT 1"
            ).fetchone()
        vec_ok = True
        try:
            self._get_collection().count()
        except Exception as e:  # noqa: BLE001
            vec_ok = False
            logger.warning("[RAGStore] 向量库健康检查失败：%s", e)
        term_enabled = self.term_feature_enabled
        return {
            "document_count": doc_count,
            "total_blocks": total_blocks,
            "section_summary_count": section_summary_count,
            "chapter_summary_count": chapter_summary_count,
            "term_count": raw_term_count if term_enabled else 0,
            "term_disabled": not term_enabled,
            "last_indexed": last_row["indexed_at"] if last_row else None,
            "pending_documents": pending,
            "vector_store_ok": vec_ok,
            "fts_index_ok": True,
            "terms_table_ok": True,
        }
