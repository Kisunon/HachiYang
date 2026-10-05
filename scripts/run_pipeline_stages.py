"""三阶段流水线入口（v1.6）。

阶段 1（提取落盘）：knowledge_base/ → knowledge_extracted/<域>/<卷名>.extracted.md/.json（零 API）
阶段 2（切片落盘）：knowledge_extracted/ → knowledge_chunks/<域>/<卷名>.chunks.json（零 API）
阶段 3（Embedding 入库）：knowledge_chunks/ → ChromaDB + SQLite + rag_fts + 两级摘要
        （Embedding 与摘要 LLM 消耗 API 配额；--no-summaries 可只耗 Embedding）

用法：
  python scripts/run_pipeline_stages.py --stage extract        # 仅阶段 1
  python scripts/run_pipeline_stages.py --stage chunk          # 仅阶段 2
  python scripts/run_pipeline_stages.py --stage index          # 阶段 3：全部 chunks 入库
  python scripts/run_pipeline_stages.py --stage index --file knowledge_chunks/freud/卷01-爱情心理学.chunks.json
  python scripts/run_pipeline_stages.py --stage index --no-summaries   # 仅 Embedding 入库，不生成摘要
  python scripts/run_pipeline_stages.py --stage all            # 阶段 1 + 阶段 2（不含阶段 3）
  python scripts/run_pipeline_stages.py --stage all --only 弗洛伊德   # 仅处理文件名含关键词的书
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("ebooklib").setLevel(logging.WARNING)


def run_extract(only: str | None) -> None:
    from app.config import get_settings
    from app.rag.extractor import run_extract_stage

    settings = get_settings()
    report = run_extract_stage(settings=settings)

    print("\n" + "=" * 70)
    print("阶段 1（提取落盘）完成")
    print("=" * 70)
    for book in report.books:
        if only and only not in Path(book.file_path).name:
            continue
        if book.error:
            print(f"  FAIL {Path(book.file_path).name}: {book.error}")
            continue
        print(f"  {'多卷' if book.multi_volume else '单卷'} {Path(book.file_path).name}")
        for v in book.volumes:
            print(
                f"    - {v.volume_title}: {v.chapter_count} 章 / {v.section_count} 节 / "
                f"{v.subsection_count} 小节 / {v.paragraph_count} 段 / {v.total_chars} 字"
                f"（剔除 front {v.skipped_front_sections} / back {v.skipped_back_sections}）"
            )
            print(f"        → {Path(v.md_path).relative_to(Path.cwd()) if Path(v.md_path).is_relative_to(Path.cwd()) else v.md_path}")
    print(f"\n合计：{len(report.books)} 书 / {report.total_volumes} 卷 / {report.total_paragraphs} 段落")


def run_chunk() -> None:
    from app.rag.chunker import run_chunk_stage

    report = run_chunk_stage()
    print("\n" + "=" * 70)
    print("阶段 2（切片落盘）完成")
    print("=" * 70)
    for v in report.volumes:
        if v.error:
            print(f"  FAIL {v.volume_title}: {v.error}")
        else:
            print(
                f"  OK {v.volume_title}: {v.block_count} blocks / {v.section_count} 节 / "
                f"avg {v.avg_chars:.0f} 字 → {Path(v.chunks_path).name}"
            )
    print(f"\n合计：{len(report.volumes)} 卷 / {report.total_blocks} blocks")


def run_index(only_file: str | None, no_summaries: bool, chunks_dir: str | None = None) -> None:
    from app.config import get_settings
    from app.llm.embedding import EmbeddingClient
    from app.llm.client import LLMClient
    from app.rag.indexer import indexer_from_chunks

    settings = get_settings()
    if not settings.ark_key:
        print("错误：缺少 ARK_KEY 环境变量（Embedding 调用必需）。请在系统环境变量配置后重试。")
        sys.exit(2)

    embedding = EmbeddingClient(settings)
    llm = None
    if not no_summaries:
        llm = LLMClient(settings)
        print(f"两级摘要已启用（LLM={settings.llm_model}）；--no-summaries 可跳过摘要仅入库。")
    else:
        print("已指定 --no-summaries：仅 Embedding 入库，文档状态将为 indexed_but_summary_pending。")

    results = asyncio.run(indexer_from_chunks(
        chunks_root_dir=chunks_dir,
        only_file=only_file,
        enable_summaries=not no_summaries,
        embedding=embedding,
        llm=llm,
        settings=settings,
    ))

    print("\n" + "=" * 70)
    print("阶段 3（Embedding + 两级摘要入库）完成")
    print("=" * 70)
    for r in results:
        if r.get("skipped"):
            print(f"  SKIP 《{r['book_title']}》：内容未变更（status={r['status']}）")
            continue
        flag = "OK  " if r["status"] in ("done", "summary_partial", "indexed_but_summary_pending") else "FAIL"
        print(
            f"  {flag} 《{r['book_title']}》[{r['domain']}]：{r.get('volume_count', 0)} 卷 / "
            f"{r.get('text_block_count', 0)} 正文块 / 节摘要 {r.get('section_summary_count', 0)} / "
            f"章摘要 {r.get('chapter_summary_count', 0)} → {r['status']}"
        )
        if r.get("failed_sections") or r.get("failed_chapters"):
            print(
                f"        摘要缺失：节 {r.get('failed_sections', 0)}/{r.get('total_sections', 0)}，"
                f"章 {r.get('failed_chapters', 0)}/{r.get('total_chapters', 0)}"
            )
        if r.get("error"):
            print(f"        error: {r['error']}")
    if not results:
        print("  （无 chunks.json 被处理）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 三阶段流水线：阶段 1 提取 / 阶段 2 切片（零 API）/ 阶段 3 入库（耗 API）")
    parser.add_argument("--stage", choices=["extract", "chunk", "index", "all"], default="all")
    parser.add_argument("--only", default=None, help="仅处理文件名包含该关键词的书（阶段 1）")
    parser.add_argument("--file", default=None, help="阶段 3：仅索引指定 chunks.json 所属的书（整书重建）")
    parser.add_argument("--chunks-dir", default=None, help="阶段 3：自定义 chunks 根目录（默认 knowledge_chunks/；单卷调试可用临时目录）")
    parser.add_argument("--no-summaries", action="store_true", help="阶段 3：跳过两级摘要，仅 Embedding 入库")
    args = parser.parse_args()

    if args.stage in ("extract", "all"):
        run_extract(args.only)
    if args.stage in ("chunk", "all"):
        run_chunk()
    if args.stage == "index":
        run_index(args.file, args.no_summaries, args.chunks_dir)
