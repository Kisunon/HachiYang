"""阶段 3 全量自动构建（后台运行）：chunks.json → Embedding + 两级摘要 → 入库。

特性：
- 串行处理 knowledge_chunks/ 下所有书（同书多卷自动合并为一个文档）；
- **配额超限自动等待**：Embedding 批次 / 节摘要 / 章摘要调用遇到
  AccountQuotaExceeded 等配额错误时，等待 --wait-hours（默认 2）小时后重试**同一
  条目**，失败条目绝不跳过；等待期间每 5 分钟输出心跳日志；
- **断点续跑**：done 且 hash 一致的书自动跳过；中断/failed 的书复用已入库正文向量，
  仅补写缺失向量并重新生成摘要（block_id 稳定，报错条目必被重试，无遗漏）；
- 轮次机制：非配额原因失败的书在下一轮自动重试，连续 --max-rounds 轮仍失败则
  非零退出并打印汇总；
- 优雅停止：在 .data/ 下创建 auto_index_stop.flag 文件，脚本会在当前书完成后退出。

用法（后台）：
  python scripts/auto_index_all.py                       # 全量（含两级摘要）
  python scripts/auto_index_all.py --no-summaries        # 仅 Embedding（零 LLM）
  python scripts/auto_index_all.py --wait-hours 2        # 配额等待时长（小时）
日志：.data/auto_index.log（追加）
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# 确保项目根在 sys.path（脚本可从任意 cwd 启动）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.llm.client import LLMClient  # noqa: E402
from app.llm.embedding import EmbeddingClient  # noqa: E402
from app.llm.quota import run_with_quota_wait  # noqa: E402
from app.rag.indexer import RAGIndex, _load_chunks_book_groups  # noqa: E402

logger = logging.getLogger("auto_index")

# 终态：到达这些状态的书不再重试（summary_partial 为内容原因导致的个别摘要缺失，属设计内）
_TERMINAL_OK = {"done", "summary_partial", "indexed_but_summary_pending"}
# 轮次间短暂停顿（秒），避免非配额失败时紧密空转
_ROUND_GAP_SECONDS = 30


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    logging.getLogger("ebooklib").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


async def run() -> int:
    parser = argparse.ArgumentParser(description="阶段 3 全量自动构建（配额超限自动等待 2 小时续跑）")
    parser.add_argument("--chunks-dir", default=None, help="chunks.json 根目录（默认 knowledge_chunks）")
    parser.add_argument("--no-summaries", action="store_true", help="跳过两级摘要（零 LLM 配额）")
    parser.add_argument("--wait-hours", type=float, default=2.0, help="配额超限时等待小时数（默认 2）")
    parser.add_argument("--max-rounds", type=int, default=5, help="非配额失败书目的最大重试轮次（默认 5）")
    args = parser.parse_args()

    settings = get_settings()
    log_path = settings.data_dir / "auto_index.log"
    setup_logging(log_path)
    stop_flag = settings.data_dir / "auto_index_stop.flag"

    wait_seconds = int(args.wait_hours * 3600)
    logger.info(
        "===== 阶段 3 全量自动构建启动：summaries=%s, 配额等待=%d 分钟, 日志=%s =====",
        "off" if args.no_summaries else "on", wait_seconds // 60, log_path,
    )

    embedding = EmbeddingClient(settings)
    llm = None if args.no_summaries else LLMClient(settings)
    if args.no_summaries:
        logger.info("已指定 --no-summaries：仅做 Embedding，不调用 LLM")
    index = RAGIndex(embedding=embedding, llm=llm, settings=settings)

    root = Path(args.chunks_dir) if args.chunks_dir else settings.knowledge_chunks_path
    if not root.is_absolute():
        root = _PROJECT_ROOT / root
    if not root.exists():
        logger.error("chunks 目录不存在：%s", root)
        return 2

    round_no = 0
    while True:
        round_no += 1
        groups = _load_chunks_book_groups(root)
        if not groups:
            logger.warning("未发现任何 *.chunks.json（%s），退出", root)
            return 2
        logger.info("----- 第 %d 轮：共 %d 部书 -----", round_no, len(groups))

        round_failed: list[dict] = []
        round_ok: list[dict] = []
        for (domain, book_title), volumes in sorted(groups.items()):
            if stop_flag.exists():
                logger.warning("检测到停止标志 %s，当前书处理完成后退出", stop_flag)
                _print_summary(round_ok, round_failed, partial=True)
                return 3

            first_volume = str(sorted(
                volumes, key=lambda vp: vp[1].get("source_ref", {}).get("volume_index", 1)
            )[0][0])
            logger.info(
                ">>> 开始处理《%s》（domain=%s，%d 卷）", book_title, domain, len(volumes),
            )
            t0 = time.time()

            def _factory():
                return index.index_from_chunks(
                    chunks_root_dir=root,
                    only_file=first_volume,
                    enable_summaries=not args.no_summaries,
                )

            try:
                # 外层再加一道配额等待（防御内部未覆盖的配额逃逸点）
                results = await run_with_quota_wait(
                    _factory, what=f"整书《{book_title}》", wait_seconds=wait_seconds,
                )
                result = results[0] if results else {
                    "book_title": book_title, "status": "failed", "error": "无结果返回",
                }
            except Exception as e:  # noqa: BLE001
                logger.exception("《%s》处理异常（非配额错误）", book_title)
                result = {"book_title": book_title, "status": "failed", "error": str(e)}

            elapsed = time.time() - t0
            status = result.get("status", "failed")
            logger.info(
                "<<< 《%s》完成：status=%s, 卷=%s, 正文块=%s, 节摘要=%s, 章摘要=%s, 耗时 %.0f 秒%s%s",
                book_title, status,
                result.get("volume_count", len(volumes)),
                result.get("text_block_count", "-"),
                result.get("section_summary_count", "-"),
                result.get("chapter_summary_count", "-"),
                elapsed,
                f"，跳过={result.get('skipped')}" if result.get("skipped") else "",
                f"，续跑={result.get('resumed')}" if result.get("resumed") else "",
            )
            if result.get("error"):
                logger.warning("《%s》错误信息：%s", book_title, str(result["error"])[:300])

            if status in _TERMINAL_OK or result.get("skipped"):
                round_ok.append(result)
            else:
                round_failed.append(result)

        logger.info(
            "----- 第 %d 轮结束：成功/终态 %d 部，待重试 %d 部 -----",
            round_no, len(round_ok), len(round_failed),
        )

        if not round_failed:
            _print_summary(round_ok, round_failed, partial=False)
            logger.info("===== 全部书目已达终态，自动构建完成 =====")
            return 0
        if round_no >= args.max_rounds:
            _print_summary(round_ok, round_failed, partial=False)
            logger.error(
                "===== 达到最大轮次 %d，仍有 %d 部书未达终态，退出码 1 =====",
                round_no, len(round_failed),
            )
            return 1

        logger.info("等待 %d 秒后开始下一轮重试失败书目…", _ROUND_GAP_SECONDS)
        await asyncio.sleep(_ROUND_GAP_SECONDS)


def _print_summary(ok: list[dict], failed: list[dict], *, partial: bool) -> None:
    logger.info("===== 构建汇总%s =====", "（停止标志中断）" if partial else "")
    for r in ok:
        logger.info(
            "  [OK] 《%s》 status=%s 正文=%s 节摘要=%s 章摘要=%s%s",
            r.get("book_title"), r.get("status"),
            r.get("text_block_count", "-"), r.get("section_summary_count", "-"),
            r.get("chapter_summary_count", "-"),
            "（跳过）" if r.get("skipped") else "",
        )
    for r in failed:
        logger.info(
            "  [FAIL] 《%s》 status=%s error=%s",
            r.get("book_title"), r.get("status"), str(r.get("error"))[:200],
        )


if __name__ == "__main__":
    try:
        rc = asyncio.run(run())
    except KeyboardInterrupt:
        logger.warning("收到 Ctrl+C，退出（已完成条目不受影响，重跑自动续跑）")
        rc = 130
    sys.exit(rc)
