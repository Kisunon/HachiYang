"""全库重索引入口（运维脚本）。

1) 清空旧索引：rag_documents / rag_blocks / rag_fts / rag_terms / ChromaDB rag_knowledge；
2) 扫描 knowledge_base/ 全量重建索引（RAGIndex.index_all）；
3) 输出健康检查与各文档索引统计。

用法：
  python scripts/reindex_all.py            # 清空 + 全量索引
  python scripts/reindex_all.py --clear-only   # 仅清空
  python scripts/reindex_all.py --resume       # 不清空，续跑未完成文档（中断后使用）
  python scripts/reindex_all.py --no-llm       # 跳过章节摘要（不调用 LLM）
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def clear_all() -> None:
    from app.config import get_settings
    from app.db.connection import db_conn, init_infrastructure
    from app.rag.stores import RAGStore

    init_infrastructure()
    settings = get_settings()
    store = RAGStore(settings)

    with db_conn(settings) as conn:
        doc_ids = [r["document_id"] for r in conn.execute(
            "SELECT document_id FROM rag_documents"
        ).fetchall()]
        term_count = conn.execute("SELECT COUNT(*) FROM rag_terms").fetchone()[0]

    print(f"[clear] 待删除文档数: {len(doc_ids)}, 术语数: {term_count}")
    for doc_id in doc_ids:
        store.delete_document(doc_id)
        print(f"[clear] 已删除文档: {doc_id}")

    with db_conn(settings) as conn:
        conn.execute("DELETE FROM rag_terms")

    health = store.health_check()
    print(f"[clear] 清空后健康检查: {health}")


async def reindex(enable_llm: bool) -> None:
    from app.config import get_settings
    from app.db.connection import init_infrastructure
    from app.rag.indexer import RAGIndex

    init_infrastructure()
    settings = get_settings()

    llm = None
    if enable_llm:
        from app.llm.client import LLMClient
        llm = LLMClient(settings)

    indexer = RAGIndex(llm=llm, settings=settings)
    t0 = time.time()
    results = await indexer.index_all()
    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print(f"全库索引完成，耗时 {elapsed/60:.1f} 分钟")
    print("=" * 70)
    for r in results:
        if r.success:
            print(
                f"  OK  {Path(r.file_path).name}: blocks={r.block_count} "
                f"summaries={r.section_summary_count} terms={r.term_count}"
            )
        else:
            print(f"  FAIL {Path(r.file_path).name}: {r.error}")

    from app.rag.indexer import RAGHealthCheck
    print("\n健康检查:")
    for k, v in RAGHealthCheck().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-only", action="store_true", help="仅清空旧索引")
    parser.add_argument("--resume", action="store_true", help="不清空，续跑未完成文档")
    parser.add_argument("--no-llm", action="store_true", help="跳过章节摘要（不调用 LLM）")
    args = parser.parse_args()

    if not args.resume:
        clear_all()
    if not args.clear_only:
        asyncio.run(reindex(enable_llm=not args.no_llm))
