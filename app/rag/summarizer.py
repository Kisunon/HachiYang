"""章节摘要生成器（多粒度检索之一）。

对齐 RAG_SYSTEM_DESIGN §3.5（v1.6 两级摘要）：
- 对每个 Section 生成 <200 字节摘要（section_summary），parent_block_ids 指向该节全部正文块；
- 对每个 Chapter 生成 <400 字章摘要（chapter_summary），parent_block_ids 指向该章全部节摘要
  （分级链式展开：章摘要命中只展开节摘要，见 §5.5b）；
- 生成顺序强制「节 → 章」：节摘要并发小批量，章摘要串行（输入为节标题+节摘要，约 1.5k 字符）；
- Section 文本 < 300 字跳过；个别节摘要失败 → [缺失] 占位，章摘要照常生成；
  全部节摘要失败 → 章摘要一并跳过（调用方标记 summary_partial）；
- LLM 失败时降级为"仅段落索引"，不阻塞正文入库。
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from app.llm.client import LLMClient
from app.llm.prompts import (
    build_chapter_summary_messages,
    build_section_summary_messages,
)
from app.llm.quota import run_with_quota_wait
from app.rag.chunker import ParagraphBlock

logger = logging.getLogger(__name__)

# 跳过摘要的最低字数阈值
_MIN_SECTION_CHARS = 300
# 并发摘要生成上限（避免 LLM 限流）
_SUMMARY_CONCURRENCY = 4
# 章摘要长度校验上限（prompt 要求 <400 字，留容差）
_CHAPTER_SUMMARY_MAX_CHARS = 450


class TwoLevelSummaryResult:
    """两级摘要生成结果。"""

    def __init__(self) -> None:
        self.section_summaries: list[ParagraphBlock] = []
        self.chapter_summaries: list[ParagraphBlock] = []
        self.failed_sections = 0       # 节摘要失败/缺失的节数
        self.failed_chapters = 0       # 章摘要跳过/失败的章数
        self.total_sections = 0
        self.total_chapters = 0

    @property
    def all_blocks(self) -> list[ParagraphBlock]:
        return self.chapter_summaries + self.section_summaries

    @property
    def is_partial(self) -> bool:
        """有节/章摘要缺失（部分失败）。"""
        return self.failed_sections > 0 or self.failed_chapters > 0

    @property
    def has_any_summary(self) -> bool:
        return bool(self.section_summaries) or bool(self.chapter_summaries)


class SectionSummarizer:
    """章节摘要生成器（调用 deepseek-v4-pro）。"""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def summarize_section(
        self, section_title: str, paragraphs_text: str
    ) -> str | None:
        """生成单个 Section 的摘要。返回 None 表示跳过/失败。"""
        if len(paragraphs_text) < _MIN_SECTION_CHARS:
            return None
        try:
            # 配额耗尽 → 等 2 小时后重试同一节（条目不跳过、不产生 [缺失] 占位）
            raw = await run_with_quota_wait(
                lambda: self._llm.chat(
                    build_section_summary_messages(section_title, paragraphs_text),
                    temperature=0.3,
                    # deepseek-v4-pro 为推理模型：max_tokens 需同时容纳思维链与正文，
                    # 否则思维链耗尽预算后 content 返回空串（finish_reason=length）
                    max_tokens=8192,
                    timeout=60.0,
                ),
                what=f"节摘要《{section_title[:40]}》",
            )
            summary = raw.strip()
            # 简单校验：非空且不太长
            if not summary or len(summary) > 300:
                logger.warning("[summarizer] 摘要异常（len=%d），跳过", len(summary))
                return None
            return summary
        except Exception as e:  # noqa: BLE001
            logger.warning("[summarizer] 摘要生成失败（%s），跳过", e)
            return None

    async def build_section_summaries(
        self,
        blocks: list[ParagraphBlock],
        document_id: str,
        domain: str,
        language: str = "zh",
    ) -> list[ParagraphBlock]:
        """为全部 Section 生成摘要 block。

        blocks: 同一文档的 text 类型 ParagraphBlock 列表。
        返回 section_summary 类型 ParagraphBlock 列表（parent_block_ids 已填充）。
        """
        from app.rag.chunker import group_blocks_by_section

        groups = group_blocks_by_section(blocks)
        sem = asyncio.Semaphore(_SUMMARY_CONCURRENCY)

        async def _one(key: str, section_blocks: list[ParagraphBlock]) -> ParagraphBlock | None:
            async with sem:
                chapter_title = section_blocks[0].chapter_title
                section_title = section_blocks[0].section_title
                page_range = section_blocks[0].page_range
                full_text = "\n".join(b.text for b in section_blocks)
                summary = await self.summarize_section(section_title, full_text)
                if summary is None:
                    return None
                return ParagraphBlock(
                    block_id=uuid.uuid4().hex,
                    document_id=document_id,
                    domain=domain,
                    language=language,
                    chapter_title=chapter_title,
                    section_title=section_title,
                    page_range=page_range,
                    block_index_in_section=0,
                    char_count=len(summary),
                    block_type="section_summary",
                    text=summary,
                    has_structure=True,
                    parent_block_ids=[b.block_id for b in section_blocks],
                    parent_section_id=section_blocks[0].parent_section_id,
                )

        tasks = [_one(k, v) for k, v in groups.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        summaries: list[ParagraphBlock] = []
        for r in results:
            if isinstance(r, ParagraphBlock):
                summaries.append(r)
            elif isinstance(r, Exception):
                logger.warning("[summarizer] 摘要任务异常：%s", r)

        logger.info(
            "[summarizer] %s 生成 %d 个章节摘要（共 %d 个 Section）",
            document_id, len(summaries), len(groups),
        )
        return summaries

    # ---- v1.6 两级摘要（节 → 章，§3.5）----

    async def summarize_chapter(
        self, chapter_title: str, section_summaries: list[dict]
    ) -> str | None:
        """生成单个 Chapter 的章摘要（输入为节标题 + 节摘要）。返回 None 表示失败。"""
        try:
            # 配额耗尽 → 等 2 小时后重试同一章（条目不跳过）
            raw = await run_with_quota_wait(
                lambda: self._llm.chat(
                    build_chapter_summary_messages(chapter_title, section_summaries),
                    temperature=0.3,
                    # 推理模型需为思维链预留充足 token（节摘要同因）
                    max_tokens=8192,
                    timeout=60.0,
                ),
                what=f"章摘要《{chapter_title[:40]}》",
            )
            summary = raw.strip()
            if not summary or len(summary) > _CHAPTER_SUMMARY_MAX_CHARS:
                logger.warning("[summarizer] 章摘要异常（len=%d），跳过", len(summary))
                return None
            return summary
        except Exception as e:  # noqa: BLE001
            logger.warning("[summarizer] 章摘要生成失败（%s），跳过", e)
            return None

    async def build_two_level_summaries(
        self, blocks: list[ParagraphBlock], document_id: str
    ) -> TwoLevelSummaryResult:
        """两级摘要编排（§3.5）：节摘要并发 → 章摘要串行 → 回填 parent_chapter_summary_id。

        blocks: 同一文档全部 text 类型 ParagraphBlock（含 volume/chapter/section 溯源）。
        摘要块与所属章节共享 document_id/domain/language/volume 元数据。
        """
        result = TwoLevelSummaryResult()

        # 分组：章 = (volume_index, chapter_title)；节 = (volume_index, section_title)
        chapters: dict[tuple, dict] = {}
        chapter_order: list[tuple] = []
        for b in blocks:
            ch_key = (b.volume_index, b.chapter_title or "")
            sec_key = (b.volume_index, b.section_title or "")
            if ch_key not in chapters:
                chapters[ch_key] = {"title": b.chapter_title or "", "volume_index": b.volume_index,
                                    "volume_title": b.volume_title, "sections": {}}
                chapter_order.append(ch_key)
            ch = chapters[ch_key]
            ch["sections"].setdefault(sec_key, []).append(b)

        result.total_chapters = len(chapter_order)
        result.total_sections = sum(len(c["sections"]) for c in chapters.values())
        sem = asyncio.Semaphore(_SUMMARY_CONCURRENCY)

        async def _summarize_section(sec_blocks: list[ParagraphBlock]) -> ParagraphBlock | None:
            async with sem:
                first = sec_blocks[0]
                full_text = "\n".join(b.text for b in sec_blocks)
                if len(full_text) < _MIN_SECTION_CHARS:
                    return None  # 过短节跳过（不计失败）
                summary = await self.summarize_section(
                    first.section_title or first.chapter_title, full_text,
                )
                if summary is None:
                    return None
                return ParagraphBlock(
                    block_id="secs-" + uuid.uuid4().hex,
                    document_id=document_id,
                    domain=first.domain,
                    language=first.language,
                    chapter_title=first.chapter_title,
                    section_title=first.section_title,
                    page_range=None,
                    block_index_in_section=0,
                    char_count=len(summary),
                    block_type="section_summary",
                    text=summary,
                    has_structure=True,
                    parent_block_ids=[b.block_id for b in sec_blocks],
                    volume_title=first.volume_title,
                    volume_index=first.volume_index,
                    subsection_title="",
                    source_anchor_in_md="",
                    extracted_md_path=first.extracted_md_path,
                )

        # 第一级：节摘要（全部章的节并发，信号量限流）
        sec_tasks: dict[tuple, list[tuple[str, asyncio.Task]]] = {}
        for ch_key in chapter_order:
            ch = chapters[ch_key]
            task_entries: list[tuple[str, asyncio.Task]] = []
            for sec_key, sec_blocks in ch["sections"].items():
                sec_title = sec_blocks[0].section_title or ""
                task_entries.append((sec_title, asyncio.create_task(_summarize_section(sec_blocks))))
            sec_tasks[ch_key] = task_entries

        # 第二级：章摘要串行（等待该章节摘要完成后生成）
        for ch_key in chapter_order:
            ch = chapters[ch_key]
            entries = sec_tasks[ch_key]
            sec_summaries_meta: list[dict] = []
            sec_blocks_built: list[ParagraphBlock] = []
            eligible = 0
            for sec_title, task in entries:
                sec_blocks = ch["sections"][(ch_key[0], sec_title)]
                full_len = sum(len(b.text) for b in sec_blocks)
                if full_len >= _MIN_SECTION_CHARS:
                    eligible += 1
                built = await task
                if built is not None:
                    sec_blocks_built.append(built)
                    sec_summaries_meta.append({"title": sec_title, "summary": built.text})
                else:
                    sec_summaries_meta.append({"title": sec_title, "summary": None})

            result.failed_sections += max(0, eligible - len(sec_blocks_built))
            result.section_summaries.extend(sec_blocks_built)

            if not sec_blocks_built:
                # 全部节摘要缺失 → 章摘要一并跳过（§3.5 容错）
                result.failed_chapters += 1
                continue

            sample = next(iter(ch["sections"].values()))[0]
            ch_summary = await self.summarize_chapter(ch["title"], sec_summaries_meta)
            if ch_summary is None:
                result.failed_chapters += 1
                continue
            chs_block = ParagraphBlock(
                block_id="chs-" + uuid.uuid4().hex,
                document_id=document_id,
                domain=sample.domain,
                language=sample.language,
                chapter_title=ch["title"],
                section_title="",
                page_range=None,
                block_index_in_section=0,
                char_count=len(ch_summary),
                block_type="chapter_summary",
                text=ch_summary,
                has_structure=True,
                parent_block_ids=[b.block_id for b in sec_blocks_built],
                volume_title=ch["volume_title"],
                volume_index=ch["volume_index"],
                subsection_title="",
                source_anchor_in_md="",
                extracted_md_path=sample.extracted_md_path,
            )
            # 回填节摘要 → 章摘要反向指针
            for sb in sec_blocks_built:
                sb.parent_chapter_summary_id = chs_block.block_id
            result.chapter_summaries.append(chs_block)

        logger.info(
            "[summarizer] %s 两级摘要完成：节摘要 %d/%d（失败 %d），章摘要 %d/%d（失败 %d）",
            document_id,
            len(result.section_summaries), result.total_sections, result.failed_sections,
            len(result.chapter_summaries), result.total_chapters, result.failed_chapters,
        )
        return result
