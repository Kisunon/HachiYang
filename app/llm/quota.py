"""配额超限识别与长等待重试（阶段 3 全量自动构建支撑）。

背景：经 OpenAI 兼容端点（火山方舟 ARK_BASE_URL）调用 Embedding / LLM 时，
客户端短退避重试（EmbeddingClient 5 次 / LLMClient 3 次）只能消化「请求过频」
（AccountRateLimitExceeded），对「配额耗尽」（AccountQuotaExceeded / quota
exceeded / 余额不足）无效——此时应等待配额窗口恢复（默认 2 小时）后重试
**同一条目**，绝不跳过失败条目。

用法：
    from app.llm.quota import is_quota_exceeded, run_with_quota_wait

    async def _do():
        return await client.embeddings.create(...)

    await run_with_quota_wait(_do, what="Embedding 批次《xxx》")
"""

from __future__ import annotations

import asyncio
import logging
import re

logger = logging.getLogger(__name__)

# 配额耗尽特征（等待窗口恢复后重试）
_QUOTA_PATTERN = re.compile(
    r"accountquotaexceeded|quota\s*exceeded|quota|配额|coding\s*plan|"
    r"usage\s*limit|5-hour|exceeded\s+the\s+.*usage|余额不足|欠费|insufficient\s+balance",
    re.IGNORECASE,
)
# 纯频控特征（短退避即可，不触发 2 小时等待）
_RATE_LIMIT_PATTERN = re.compile(
    r"ratelimit|rate\s*limit|too\s*frequent|请求过于频繁|频繁",
    re.IGNORECASE,
)
# 配额语义明确存在的关键词（用于频控误判兜底）
_QUOTA_KEYWORD = re.compile(r"quota|配额|余额|欠费|balance", re.IGNORECASE)


def is_quota_exceeded(exc: BaseException) -> bool:
    """判断异常是否为配额耗尽（需长等待），而非瞬时频控。

    识别顺序：优先解析 openai 风格异常体中的 error.code；再退化为异常文本匹配。
    AccountRateLimitExceeded（请求过频）→ False；AccountQuotaExceeded（配额耗尽）→ True。
    """
    code = ""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            code = str(err.get("code") or "")
    text = f"{code} {exc}"
    if not _QUOTA_PATTERN.search(text):
        return False
    # 命中频控且无任何配额语义关键词 → 视为瞬时频控（客户端短重试已覆盖）
    if _RATE_LIMIT_PATTERN.search(text) and not _QUOTA_KEYWORD.search(text):
        return False
    return True


async def run_with_quota_wait(
    coro_factory,
    *,
    what: str,
    wait_seconds: int = 7200,
    log_every: int = 300,
    sleep=None,
):
    """执行 coro_factory()；遇配额耗尽则等待 wait_seconds 后重试同一条目（无限轮次）。

    Args:
        coro_factory: 无参 async callable，每次重试重新调用（保证失败条目重放）。
        what: 日志中标识当前条目（书名/批次/节名），确保追踪不漏。
        wait_seconds: 配额等待时长（默认 2 小时）。
        log_every: 等待期间日志间隔（秒）。
        sleep: 睡眠函数（生产用 asyncio.sleep；测试可注入即时 fake）。

    非配额异常直接抛出，由上层容错逻辑处理。
    """
    sleeper = sleep or asyncio.sleep
    round_no = 0
    while True:
        try:
            return await coro_factory()
        except Exception as e:  # noqa: BLE001
            if not is_quota_exceeded(e):
                raise
            round_no += 1
            logger.warning(
                "[quota] %s 触发配额超限（第 %d 次等待）：%s；"
                "等待 %d 分钟后重试【该条目不会跳过】",
                what, round_no, str(e)[:200], wait_seconds // 60,
            )
            elapsed = 0
            while elapsed < wait_seconds:
                step = min(60, wait_seconds - elapsed)
                await sleeper(step)
                elapsed += step
                remaining = wait_seconds - elapsed
                if remaining <= 0 or remaining % log_every < 60:
                    logger.info(
                        "[quota] %s 配额等待中：已等待 %d 分钟，剩余 %d 分钟",
                        what, elapsed // 60, remaining // 60,
                    )
            logger.info("[quota] %s 等待结束，立即重试该条目", what)
