"""RAG 增量索引编排器 + 健康检查。

对齐 RAG_SYSTEM_DESIGN §3.4 / §9（v1.6 三阶段流水线）：
- 旧管线：扫描 knowledge_base/ → 解析 → 分块 → Embedding → 写库（index_document/index_all）；
- 阶段 3（indexer_from_chunks）：扫描 knowledge_chunks/**/*.chunks.json，同一本书的各卷
  合并为一个 rag_documents 记录；block_id 沿用 chunks.json 的 uuid（跨步骤一致）；
  Embedding 批量写 ChromaDB + rag_blocks + rag_fts；两级摘要（节→章，§3.5）生成后入库并
  回填 chunks.json；容错状态 indexed_but_summary_pending / summary_partial；
- 术语表（§3.6）本期弃置：rag_enable_term=False 时 indexer 跳过 term_builder。

对外接口：
  RAGIndex().index_document(file_path, enable_image=False) -> IndexStatus   # 旧 EPUB/PDF 管线
  RAGIndex().index_all() -> 扫描全库增量索引
  indexer_from_chunks(...) / RAGIndex().index_from_chunks(...)              # 阶段 3
  RAGHealthCheck() -> HealthStatus
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path

from app.config import Settings, get_settings
from app.llm.client import LLMClient
from app.llm.embedding import EmbeddingClient
from app.llm.quota import run_with_quota_wait
from app.rag.chunker import Document, ParagraphBlock, chunk_document
from app.rag.stores import RAGStore

logger = logging.getLogger(__name__)

# Embedding 单次写入 ChromaDB 的批次（保守取值，与 EmbeddingClient 分批一致）
_EMBED_WRITE_BATCH = 10
# Embedding 请求并发数：ARK 端点账户级限流敏感，节奏控制见 embedding._pace_wait；
# 并发 1 + 自适应间隔实测可稳定 ~30 请求/分钟
_EMBED_CONCURRENCY = 1


def _content_hash(file_path: str) -> str:
    """文件内容 MD5（大文件分块读取）。"""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):  # 1MB
            h.update(chunk)
    return h.hexdigest()


def _parse_by_extension(file_path: str) -> Document:
    """按扩展名分发解析器。"""
    ext = Path(file_path).suffix.lower()
    if ext == ".epub":
        from app.rag.parsers.epub_parser import parse_epub
        return parse_epub(file_path)
    if ext == ".pdf":
        from app.rag.parsers.pdf_parser import parse_pdf
        return parse_pdf(file_path)
    raise ValueError(f"不支持的文件格式：{ext}（{file_path}）")


class IndexStatus:
    """单文档索引结果。"""

    def __init__(self, document_id: str, file_path: str, success: bool,
                 block_count: int = 0, section_summary_count: int = 0,
                 term_count: int = 0, content_hash: str = "",
                 error: str | None = None) -> None:
        self.document_id = document_id
        self.file_path = file_path
        self.success = success
        self.block_count = block_count
        self.section_summary_count = section_summary_count
        self.term_count = term_count
        self.content_hash = content_hash
        self.error = error

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "file_path": self.file_path,
            "success": self.success,
            "block_count": self.block_count,
            "section_summary_count": self.section_summary_count,
            "term_count": self.term_count,
            "content_hash": self.content_hash,
            "error": self.error,
        }


class RAGIndex:
    """增量索引编排器。"""

    def __init__(
        self,
        embedding: EmbeddingClient | None = None,
        llm: LLMClient | None = None,
        store: RAGStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._embedding = embedding or EmbeddingClient(self.settings)
        self._llm = llm
        self._store = store or RAGStore(self.settings)
        self._summarizer = None
        if self._llm is not None:
            from app.rag.summarizer import SectionSummarizer
            self._summarizer = SectionSummarizer(self._llm)
        # 术语表构建器：第一阶段（索引页/术语表解析）不依赖 LLM，无 LLM 时仍可构建；
        # 第二阶段（LLM 从段落补充术语）在 TermBuilder 内部自动跳过
        from app.rag.term_builder import TermBuilder
        self._term_builder = TermBuilder(self._llm, self._store)

    # ---- 单文档索引 ----

    async def index_document(self, file_path: str, enable_image: bool = False) -> IndexStatus:
        """索引单个文档（增量：hash 未变则跳过）。"""
        path = Path(file_path)
        if not path.exists():
            return IndexStatus("", file_path, False, error="文件不存在")

        content_hash = _content_hash(str(path))

        # 增量检查
        existing = self._store.get_document_by_path(str(path))
        if existing and existing.get("content_hash") == content_hash and existing.get("status") == "done":
            logger.info("[indexer] 跳过未变更文档：%s", path.name)
            return IndexStatus(
                existing["document_id"], str(path), True,
                block_count=existing.get("block_count", 0),
                section_summary_count=existing.get("section_summary_count", 0),
                term_count=existing.get("term_count", 0),
                content_hash=content_hash,
            )

        # 变更则先删除旧索引
        if existing:
            logger.info("[indexer] 文档已变更，重建索引：%s", path.name)
            self._store.delete_document(existing["document_id"])

        # 1) 解析 + 分块
        try:
            doc = _parse_by_extension(str(path))
            doc.content_hash = content_hash
        except Exception as e:  # noqa: BLE001
            logger.exception("[indexer] 解析失败：%s", path.name)
            self._store.upsert_document(
                doc.document_id if False else f"doc_{path.stem[:12]}",
                str(path), content_hash, _domain_from_path(str(path)),
                status="failed",
            )
            return IndexStatus("", str(path), False, content_hash=content_hash, error=str(e))

        blocks = chunk_document(doc)
        if not blocks:
            logger.warning("[indexer] 分块为空，跳过：%s", path.name)
            self._store.upsert_document(
                doc.document_id, str(path), content_hash, doc.domain,
                doc.language, status="done",
            )
            return IndexStatus(doc.document_id, str(path), True, content_hash=content_hash)

        # 登记文档（indexing 中）
        self._store.upsert_document(
            doc.document_id, str(path), content_hash, doc.domain,
            doc.language, status="indexing",
        )

        try:
            # 2) 段落 Embedding + 写入 ChromaDB + rag_blocks + rag_fts
            failed_batches = await self._embed_and_store_blocks(blocks, doc.title, log_label=path.name)

            # 3) 章节摘要生成（可选，LLM 失败则降级）
            summary_count = 0
            if self._summarizer is not None:
                try:
                    summaries = await self._summarizer.build_section_summaries(
                        blocks, doc.document_id, doc.domain, doc.language,
                    )
                    if summaries:
                        failed_batches += await self._embed_and_store_blocks(
                            summaries, doc.title, log_label=path.name + "(摘要)",
                        )
                        summary_count = len(summaries)
                except Exception as e:  # noqa: BLE001
                    logger.warning("[indexer] 章节摘要生成失败，降级为仅段落索引：%s", e)

            # Embedding 批次失败（重试后仍失败，如配额耗尽）：标记 failed，待下次重试重建
            if failed_batches > 0:
                logger.error(
                    "[indexer] %s 有 %d 个 Embedding 批次失败，标记为 failed 待重试",
                    path.name, failed_batches,
                )
                self._store.set_document_status(doc.document_id, "failed")
                return IndexStatus(
                    doc.document_id, str(path), False,
                    block_count=len(blocks),
                    section_summary_count=summary_count,
                    content_hash=content_hash,
                    error=f"{failed_batches} 个 Embedding 批次失败（可能配额耗尽/限流）",
                )

            # 4) 术语表构建（第一阶段：PDF 索引页正则 + EPUB 结构化条目；第二阶段：LLM 补充）
            term_count = 0
            try:
                terms = await self._term_builder.build_all(
                    doc.index_text, doc.domain, blocks,
                    index_entries=doc.index_entries,
                )
                term_count = len(terms)
            except Exception as e:  # noqa: BLE001
                logger.warning("[indexer] 术语表构建失败：%s", e)

            # 5) 更新状态为 done
            self._store.set_document_status(
                doc.document_id, "done",
                block_count=len(blocks),
                section_summary_count=summary_count,
                term_count=term_count,
            )
            logger.info(
                "[indexer] %s 索引完成：blocks=%d summaries=%d terms=%d",
                path.name, len(blocks), summary_count, term_count,
            )
            return IndexStatus(
                doc.document_id, str(path), True,
                block_count=len(blocks),
                section_summary_count=summary_count,
                term_count=term_count,
                content_hash=content_hash,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("[indexer] 索引失败：%s", path.name)
            self._store.set_document_status(doc.document_id, "failed")
            return IndexStatus(doc.document_id, str(path), False, content_hash=content_hash, error=str(e))

    async def _embed_and_store_blocks(
        self, blocks: list[ParagraphBlock], document_title: str, log_label: str = ""
    ) -> int:
        """批量向量化 + 写入向量库 + SQLite。

        断点续跑：已存在 block_id（SQLite rag_blocks）跳过——不重复写 FTS、不重复
        Embedding，因此中断前已完成的条目不会重复消耗配额，失败批次在重跑时必然重试
        （无遗漏）。

        Returns:
            失败批次数（0 表示全部成功；>0 时调用方应将文档标记为 failed 以便重试）。
        """
        store = self._store
        label = log_label or document_title
        if not blocks:
            return 0

        document_id = blocks[0].document_id
        existing_ids = store.get_block_id_set(document_id)
        new_blocks = [b for b in blocks if b.block_id not in existing_ids]
        skipped = len(blocks) - len(new_blocks)
        if skipped:
            logger.info(
                "[indexer] %s 断点续跑：%d 个块已入库跳过，%d 个块待向量化",
                label, skipped, len(new_blocks),
            )
        if not new_blocks:
            return 0

        # SQLite 元数据（全量 upsert，幂等）+ FTS5（仅新块）
        store.add_blocks(blocks)
        store.add_fts_rows([
            {
                "block_id": b.block_id,
                "text": b.text,
                "document_title": document_title,
                "chapter_title": b.chapter_title or "",
                "domain": b.domain,
            }
            for b in new_blocks
        ])

        # 分批并发 Embedding + 写 ChromaDB
        batches = [
            new_blocks[start : start + _EMBED_WRITE_BATCH]
            for start in range(0, len(new_blocks), _EMBED_WRITE_BATCH)
        ]
        sem = asyncio.Semaphore(_EMBED_CONCURRENCY)
        failed_batches = 0
        done_count = 0

        async def _one(batch: list[ParagraphBlock], batch_no: int) -> None:
            nonlocal failed_batches, done_count
            async with sem:
                async def _do():
                    vectors = await self._embedding.embed_texts([b.text for b in batch])
                    store.add_vectors(
                        ids=[b.block_id for b in batch],
                        embeddings=vectors,
                        documents=[b.text for b in batch],
                        metadatas=[b.to_metadata() for b in batch],
                    )

                try:
                    # 配额耗尽 → 等 2 小时后重试本批次（条目不跳过）；其他异常才计失败
                    await run_with_quota_wait(
                        _do, what=f"Embedding《{label}》第 {batch_no}/{len(batches)} 批"
                    )
                except Exception as e:  # noqa: BLE001
                    failed_batches += 1
                    logger.warning(
                        "[indexer] Embedding 批次失败（已重试）：%s（累计失败 %d 批）",
                        e, failed_batches,
                    )
                    return
                done_count += 1
                if done_count % 50 == 0:
                    logger.info(
                        "[indexer] %s 向量进度：%d/%d 批",
                        label, done_count, len(batches),
                    )

        await asyncio.gather(*(_one(b, i + 1) for i, b in enumerate(batches)))
        return failed_batches

    # ---- 全库扫描 ----

    async def index_all(self, base_dir: str | None = None, enable_image: bool = False) -> list[IndexStatus]:
        """扫描 knowledge_base/ 增量索引全部文档。"""
        kb_path = Path(base_dir) if base_dir else self.settings.knowledge_base_path
        if not kb_path.exists():
            logger.warning("[indexer] 知识库目录不存在：%s", kb_path)
            return []

        files = sorted(
            p for p in kb_path.rglob("*")
            if p.is_file() and p.suffix.lower() in (".epub", ".pdf")
        )
        logger.info("[indexer] 扫描到 %d 个文档待检查", len(files))

        results: list[IndexStatus] = []
        for f in files:
            status = await self.index_document(str(f), enable_image=enable_image)
            results.append(status)
        done = sum(1 for r in results if r.success)
        logger.info("[indexer] 全库索引完成：%d/%d 成功", done, len(results))
        return results


    # ==================== 阶段 3：chunks.json → Embedding 入库（v1.6 三阶段流水线） ====================

    async def index_from_chunks(
        self,
        chunks_root_dir: str | Path | None = None,
        only_file: str | Path | None = None,
        enable_summaries: bool = True,
    ) -> list[dict]:
        """阶段 3：扫描 chunks.json → 合并同书各卷 → Embedding + 两级摘要 → 入库 + 回填。

        Args:
            chunks_root_dir: knowledge_chunks 根目录（默认 settings.knowledge_chunks_path）。
            only_file: 仅索引指定 chunks.json 所属的那本书（整书重建，含该书全部卷）。
            enable_summaries: False 时跳过两级摘要（零 LLM 配额），状态记
                indexed_but_summary_pending；Embedding 仍消耗 API。

        Returns:
            每本书一个结果 dict（document_id / book_title / 各计数 / status / error）。
        """
        from app.config import PROJECT_ROOT

        root = Path(chunks_root_dir) if chunks_root_dir else self.settings.knowledge_chunks_path
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        if not root.exists():
            raise FileNotFoundError(f"chunks 目录不存在：{root}")

        groups = _load_chunks_book_groups(root, only_file)
        if not groups:
            logger.warning("[indexer] 阶段 3：未找到可索引的 chunks.json（%s）", root)
            return []

        results: list[dict] = []
        for (domain, book_title), volumes in sorted(groups.items()):
            try:
                result = await self._index_one_book_from_chunks(
                    domain, book_title, volumes, enable_summaries=enable_summaries,
                )
            except Exception as e:  # noqa: BLE001
                logger.exception("[indexer] 阶段 3 索引失败：《%s》", book_title)
                result = {
                    "document_id": _doc_id_for_book(domain, book_title),
                    "book_title": book_title,
                    "domain": domain,
                    "status": "failed",
                    "error": str(e),
                }
            results.append(result)
        return results

    async def _index_one_book_from_chunks(
        self,
        domain: str,
        book_title: str,
        volumes: list[tuple[Path, dict]],
        enable_summaries: bool,
    ) -> dict:
        """单本书的阶段 3 入库（volumes: [(chunks_path, payload), ...]，按卷次排序）。"""
        document_id = _doc_id_for_book(domain, book_title)
        volumes = sorted(volumes, key=lambda vp: vp[1].get("source_ref", {}).get("volume_index", 1))
        doc_hash = _book_content_hash(volumes)

        text_blocks: list[ParagraphBlock] = []
        volumes_meta: list[dict] = []
        for path, payload in volumes:
            ref = payload.get("source_ref", {})
            vi = int(ref.get("volume_index", 1) or 1)
            vblocks = _blocks_from_chunks(payload, document_id)
            text_blocks.extend(vblocks)
            volumes_meta.append({
                "volume_index": vi,
                "volume_title": ref.get("volume_title", ""),
                "chunks_path": str(path.resolve()),
                "content_hash": ref.get("content_hash", ""),
                "block_count": len(vblocks),
            })

        result: dict = {
            "document_id": document_id,
            "book_title": book_title,
            "domain": domain,
            "volume_count": len(volumes),
            "text_block_count": len(text_blocks),
            "section_summary_count": 0,
            "chapter_summary_count": 0,
            "status": "failed",
            "error": None,
            "skipped": False,
            "resumed": False,
        }
        if not text_blocks:
            result["status"] = "failed"
            result["error"] = "chunks.json 中无 text 块"
            return result

        # 增量：同 hash 且 done 则跳过
        existing = self._store.get_document(document_id)
        resumed = False
        if existing and existing.get("content_hash") == doc_hash and existing.get("status") == "done":
            logger.info("[indexer] 阶段 3 跳过未变更书籍：《%s》", book_title)
            result["status"] = existing.get("status", "done")
            result["skipped"] = True
            result["section_summary_count"] = existing.get("section_summary_count", 0)
            result["chapter_summary_count"] = existing.get("chapter_summary_count", 0)
            return result
        if existing and existing.get("content_hash") != doc_hash:
            # 内容变更：整书重建
            logger.info("[indexer] 阶段 3《%s》内容变更，整书重建", book_title)
            self._store.delete_document(document_id)
            existing = None
        elif existing:
            # 同 hash 但中断/部分完成（failed / indexing / summary_partial 等）：断点续跑——
            # 正文块与向量复用（_embed_and_store_blocks 按 block_id 跳过），
            # 摘要块（secs-/chs- 随机 id）清除后重新生成，保证不重不漏。
            resumed = True
            removed = self._store.delete_summary_blocks(document_id)
            logger.info(
                "[indexer] 阶段 3《%s》断点续跑（上次状态 %s）：复用正文，清除旧摘要 %d 块",
                book_title, existing.get("status"), removed,
            )
        result["resumed"] = resumed

        source_file = _resolve_source_file(
            [v["content_hash"] for v in volumes_meta if v["content_hash"]],
            self.settings,
        ) or str(volumes_meta[0]["chunks_path"])

        self._store.upsert_document(
            document_id, source_file, doc_hash, domain,
            language=text_blocks[0].language or "zh",
            status="indexing",
            volume_count=len(volumes_meta),
            volumes=volumes_meta,
        )

        # 1) 正文 Embedding + 入库
        failed_batches = await self._embed_and_store_blocks(
            text_blocks, book_title, log_label=f"《{book_title}》正文",
        )
        if failed_batches > 0:
            logger.error("[indexer] 《%s》正文 %d 个 Embedding 批次失败，标记 failed",
                         book_title, failed_batches)
            self._store.set_document_status(document_id, "failed")
            result["error"] = f"{failed_batches} 个正文 Embedding 批次失败（限流/配额）"
            return result

        # 2) 两级摘要（节 → 章）；术语表受 rag_enable_term 守卫，本期跳过
        two_level = None
        if enable_summaries and self._summarizer is not None:
            try:
                two_level = await self._summarizer.build_two_level_summaries(text_blocks, document_id)
            except Exception as e:  # noqa: BLE001
                logger.warning("[indexer] 《%s》两级摘要整体失败，降级 pending：%s", book_title, e)
        elif enable_summaries and self._summarizer is None:
            logger.warning("[indexer] 未配置 LLM，《%s》跳过摘要生成", book_title)

        summary_failed_batches = 0
        if two_level is not None and two_level.has_any_summary:
            summary_blocks = two_level.all_blocks
            summary_failed_batches = await self._embed_and_store_blocks(
                summary_blocks, book_title, log_label=f"《{book_title}》摘要",
            )
            # 回填 chunks.json（摘要块按卷次归属，重跑时先剔除旧摘要块）
            _write_summaries_back(volumes, summary_blocks, book_title)

        sec_count = len(two_level.section_summaries) if two_level else 0
        ch_count = len(two_level.chapter_summaries) if two_level else 0
        result["section_summary_count"] = sec_count
        result["chapter_summary_count"] = ch_count

        # 3) 状态判定（§3.5 容错）
        if two_level is None or not two_level.has_any_summary:
            status = "indexed_but_summary_pending"
        elif two_level.is_partial or summary_failed_batches > 0:
            status = "summary_partial"
        else:
            status = "done"

        self._store.set_document_status(
            document_id, status,
            block_count=len(text_blocks),
            section_summary_count=sec_count,
            chapter_summary_count=ch_count,
            term_count=0,
        )
        result["status"] = status
        if two_level is not None:
            result["failed_sections"] = two_level.failed_sections
            result["failed_chapters"] = two_level.failed_chapters
            result["total_sections"] = two_level.total_sections
            result["total_chapters"] = two_level.total_chapters
        logger.info(
            "[indexer] 阶段 3《%s》完成：%d 卷 / %d 正文块 / 节摘要 %d / 章摘要 %d → %s",
            book_title, len(volumes_meta), len(text_blocks), sec_count, ch_count, status,
        )
        return result


# ---- 阶段 3 模块级辅助 ----

def _doc_id_for_book(domain: str, book_title: str) -> str:
    """由 (domain, book_title) 派生稳定 document_id（跨重跑一致）。"""
    h = hashlib.md5(f"{domain}\x00{book_title}".encode("utf-8")).hexdigest()[:12]
    return f"doc_{domain}_{h}"


def _book_content_hash(volumes: list[tuple[Path, dict]]) -> str:
    """整书内容 hash：任一卷 chunks 变化即失效（卷次 + 卷内容 hash + 块数）。"""
    parts = []
    for path, payload in volumes:
        ref = payload.get("source_ref", {})
        stats = payload.get("stats", {})
        parts.append("|".join([
            str(ref.get("volume_index", "")),
            str(ref.get("content_hash", "")),
            str(stats.get("total_blocks", "")),
            path.name,
        ]))
    return hashlib.md5("\n".join(sorted(parts)).encode("utf-8")).hexdigest()


def _load_chunks_book_groups(
    root: Path, only_file: str | Path | None = None
) -> dict[tuple[str, str], list[tuple[Path, dict]]]:
    """扫描 chunks 根目录，按 (domain, book_title) 分组。only_file 时仅保留该书。"""
    files = sorted(root.rglob("*.chunks.json"))
    target_key: tuple[str, str] | None = None
    if only_file:
        of = Path(only_file)
        if not of.is_absolute():
            of = (Path.cwd() / of).resolve()
        payload = json.loads(of.read_text(encoding="utf-8"))
        ref = payload.get("source_ref", {})
        target_key = (ref.get("domain", ""), ref.get("book_title", ""))

    groups: dict[tuple[str, str], list[tuple[Path, dict]]] = {}
    for f in files:
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            logger.warning("[indexer] chunks.json 读取失败，跳过 %s：%s", f.name, e)
            continue
        ref = payload.get("source_ref", {})
        key = (ref.get("domain", "philosophy"), ref.get("book_title", f.stem))
        if target_key is not None and key != target_key:
            continue
        groups.setdefault(key, []).append((f, payload))
    return groups


def _blocks_from_chunks(payload: dict, document_id: str) -> list[ParagraphBlock]:
    """把单卷 chunks.json 的 text 块转换为 ParagraphBlock（block_id 沿用 uuid）。"""
    ref = payload.get("source_ref", {})
    domain = ref.get("domain", "philosophy")
    extracted_md = ref.get("extracted_md", "")
    blocks: list[ParagraphBlock] = []
    for b in payload.get("blocks", []):
        if b.get("block_type", "text") != "text":
            continue  # 回填的摘要块不参与正文重建（摘要每次重跑重新生成）
        pr = b.get("parent_ref", {})
        blocks.append(ParagraphBlock(
            block_id=b["block_id"],
            document_id=document_id,
            domain=domain,
            language=b.get("language", "zh"),
            chapter_title=pr.get("chapter_title", ""),
            section_title=pr.get("section_title", ""),
            page_range=None,
            block_index_in_section=b.get("block_index_in_section", 0),
            char_count=b.get("char_count", len(b.get("text", ""))),
            block_type="text",
            text=b.get("text", ""),
            has_structure=True,
            volume_title=pr.get("volume_title", ref.get("volume_title", "")),
            volume_index=int(pr.get("volume_index", ref.get("volume_index", 1)) or 1),
            subsection_title=pr.get("subsection_title", ""),
            source_anchor_in_md=b.get("source_anchor_in_md", ""),
            extracted_md_path=extracted_md,
        ))
    return blocks


def _summary_block_to_chunk_dict(b: ParagraphBlock, book_title: str) -> dict:
    """摘要 ParagraphBlock → chunks.json 回填块格式。"""
    d = {
        "block_id": b.block_id,
        "block_type": b.block_type,
        "parent_ref": {
            "book_title": book_title,
            "volume_title": b.volume_title or "",
            "volume_index": b.volume_index or 1,
            "chapter_title": b.chapter_title or "",
            "section_title": b.section_title or "",
            "subsection_title": b.subsection_title or "",
        },
        "block_index_in_section": 0,
        "text": b.text,
        "char_count": b.char_count,
        "language": b.language,
        "source_anchor_in_md": b.source_anchor_in_md or "",
        "parent_block_ids": list(b.parent_block_ids or []),
    }
    if b.block_type == "section_summary" and b.parent_chapter_summary_id:
        d["parent_chapter_summary_id"] = b.parent_chapter_summary_id
    return d


def _write_summaries_back(
    volumes: list[tuple[Path, dict]], summary_blocks: list[ParagraphBlock], book_title: str
) -> None:
    """把摘要块回填到各卷 chunks.json（按 volume_index 归属），并更新 stats 计数。"""
    by_volume: dict[int, list[ParagraphBlock]] = {}
    for b in summary_blocks:
        by_volume.setdefault(b.volume_index or 1, []).append(b)

    for path, payload in volumes:
        vi = int(payload.get("source_ref", {}).get("volume_index", 1) or 1)
        mine = by_volume.get(vi, [])
        text_blocks = [b for b in payload.get("blocks", []) if b.get("block_type", "text") == "text"]
        payload["blocks"] = text_blocks + [
            _summary_block_to_chunk_dict(b, book_title) for b in mine
        ]
        stats = payload.setdefault("stats", {})
        stats["total_blocks"] = len(text_blocks)
        stats["total_section_summary_blocks"] = sum(
            1 for b in mine if b.block_type == "section_summary"
        )
        stats["total_chapter_summary_blocks"] = sum(
            1 for b in mine if b.block_type == "chapter_summary"
        )
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    logger.info("[indexer] 摘要已回填 %d 个 chunks.json", len(volumes))


def _resolve_source_file(content_hashes: list[str], settings: Settings) -> str | None:
    """在 knowledge_base/ 中按内容 hash 反查源 EPUB/PDF（找不到返回 None）。"""
    if not content_hashes:
        return None
    wanted = set(content_hashes)
    base = settings.knowledge_base_path
    if not base.exists():
        return None
    for p in sorted(base.rglob("*")):
        if not (p.is_file() and p.suffix.lower() in (".epub", ".pdf")):
            continue
        try:
            if _content_hash(str(p)) in wanted:
                return str(p.resolve())
        except OSError:
            continue
    return None


async def indexer_from_chunks(
    chunks_root_dir: str | Path | None = None,
    only_file: str | Path | None = None,
    enable_summaries: bool = True,
    embedding: EmbeddingClient | None = None,
    llm: LLMClient | None = None,
    store: RAGStore | None = None,
    settings: Settings | None = None,
) -> list[dict]:
    """阶段 3 模块级入口（对齐 RAG_SYSTEM_DESIGN indexer_from_chunks）。"""
    settings = settings or get_settings()
    index = RAGIndex(embedding=embedding, llm=llm, store=store, settings=settings)
    return await index.index_from_chunks(
        chunks_root_dir=chunks_root_dir,
        only_file=only_file,
        enable_summaries=enable_summaries,
    )


def _domain_from_path(file_path: str) -> str:
    parent = Path(file_path).parent.name.lower()
    mapping = {"freud": "freud", "jung": "jung", "philosophy": "philosophy"}
    return mapping.get(parent, "philosophy")


# ==================== 健康检查 ====================

def RAGHealthCheck(store: RAGStore | None = None) -> dict:
    """RAG 健康检查（对齐 §9.3）。"""
    store = store or RAGStore()
    return store.health_check()
