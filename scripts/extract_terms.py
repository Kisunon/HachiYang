"""术语提取（以小节为单位，OpenAI 兼容 LLM）：chunks.json → knowledge_terms/*.terms.json。

范围：弗洛伊德文集（freud，12 卷）与荣格文集（jung，典藏版）。哲学书不在范围内。

特性：
- 以**小节（section）**为组织与出处单位：按 parent_ref 聚合 block_type=text 的块，
  保持 block_index_in_section 顺序；
- 大节按块边界软切为 ~--batch-chars（默认 3500）字符的批次逐批调用 LLM，小节内
  合并去重；每批结果含 zh/en/aliases_zh/aliases_en/definition（中文简要解释 ≤80 字）；
- **仅提取落盘，不做后续处理**：不写 rag_terms、不接入检索、不做跨卷全局合并，
  产物留待后续使用；
- **配额超限自动等待**：run_with_quota_wait（默认等 2 小时重放同批，
  失败条目绝不跳过）；串行调用 + 间隔，避开频控；
- **断点续跑**：每完成一个小节立即落盘，重跑时已 done 小节自动跳过；
- **优雅停止**：.data/extract_terms_stop.flag 存在时，当前小节完成后退出。

用法：
  python scripts/extract_terms.py --domain freud --only 卷01        # M1 单卷试跑
  python scripts/extract_terms.py --domain freud                    # 弗洛伊德 12 卷
  python scripts/extract_terms.py --domain jung                     # 荣格典藏版
  python scripts/extract_terms.py --domain all                      # 全部
  python scripts/extract_terms.py --domain freud --only 卷01 --dry-run   # 只打印批次不耗配额
日志：.data/extract_terms.log（追加）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path

# 确保项目根在 sys.path（脚本可从任意 cwd 启动）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.llm.client import LLMClient  # noqa: E402
from app.llm.quota import is_quota_exceeded, run_with_quota_wait  # noqa: E402

logger = logging.getLogger("extract_terms")

# ==================== 提示词 ====================

# 每批 LLM 返回术语数上限（提示词软约束）。召回优先：宁多勿漏，length 截断由空响应重试+拆半兜底
_TERMS_CAP = 25

TERM_SYSTEM_PROMPT = """你是精神分析与分析心理学的术语提取专家。从给定的中文心理学文献段落（弗洛伊德文集或荣格文集）中提取专业术语，严格输出 JSON 对象（不要输出任何其他文字），格式：

```json
{
  "terms": [
    {
      "zh": "压抑",
      "en": "Repression",
      "aliases_zh": ["潜抑"],
      "aliases_en": [],
      "definition": "弗洛伊德提出的核心防御机制：将不为意识所接受的冲动、观念逐出意识并压入潜意识，使之不能直接进入意识。"
    }
  ]
}
```

规则：
1. 只提取心理学/精神分析/分析心理学的专业术语：概念名、心理机制名、术语化的现象或病症名、理论专名；忽略人名、书名、普通词汇与日常用语。
2. zh：术语中文规范名（以该文献通行译法为准）。
3. en：该术语标准英译（如 Repression、Libido、Complex、Self、Synchronicity、Oedipus complex）；无法确定时输出空字符串，禁止编造。
4. aliases_zh：常见异译、别名、旧译或混用名（如「俄狄浦斯情结」的别名「恋母情结」；「潜意识」常见混用「无意识」）；无则空数组。
5. aliases_en：英文异名、缩写或德文原词的英文转写；无则空数组。
6. definition：一句话简要解释，**统一用中文**，不超过 80 个汉字，结合该术语在本文献中的含义，不照抄原文长句。
7. 每批最多提取 __TERMS_CAP__ 个最核心、最明确的术语，宁缺毋滥；同一术语只列一次。"""

TERM_SYSTEM_PROMPT = TERM_SYSTEM_PROMPT.replace("__TERMS_CAP__", str(_TERMS_CAP))


def build_messages(ref: dict, batch_text: str) -> list[dict]:
    header = (
        f"【出处】《{ref.get('book_title', '')}》{ref.get('volume_title', '')}"
        f" · {ref.get('chapter_title', '')} · {ref.get('section_title', '')}"
    )
    return [
        {"role": "system", "content": TERM_SYSTEM_PROMPT},
        {"role": "user", "content": f"{header}\n【文本】\n{batch_text}"},
    ]


# ==================== JSON 容错解析 ====================

def parse_terms_array(raw: str) -> list[dict]:
    """从 LLM 输出解析术语 JSON 数组（容忍 ```json 围栏与杂文本）。"""
    if not raw:
        return []
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict):
            # json_object 模式下模型可能包装为 {"terms": [...]} / {"data": [...]}
            for v in data.values():
                if isinstance(v, list):
                    return [d for d in v if isinstance(d, dict)]
    except Exception:  # noqa: BLE001
        pass
    # 兜底：逐个抽取 {...} 对象
    items: list[dict] = []
    for m in re.finditer(r"\{[^{}]*\}", text, flags=re.S):
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict):
                items.append(obj)
        except Exception:  # noqa: BLE001
            continue
    return items


def normalize_term(item: dict) -> dict | None:
    """字段规范化；无中文规范名的条目丢弃。"""
    zh = str(item.get("zh") or "").strip(" ，,。；;、")
    if not zh or len(zh) > 60:
        return None

    def _as_str_list(v) -> list[str]:
        if isinstance(v, str):
            v = [v] if v.strip() else []
        elif not isinstance(v, list):
            v = []
        out = []
        for x in v:
            s = str(x).strip(" ，,。；;、")
            if s and len(s) <= 60 and s not in out:
                out.append(s)
        return out

    return {
        "zh": zh,
        "en": str(item.get("en") or "").strip()[:80],
        "aliases_zh": _as_str_list(item.get("aliases_zh")),
        "aliases_en": _as_str_list(item.get("aliases_en")),
        "definition": str(item.get("definition") or "").strip()[:200],
    }


def merge_section_terms(batch_results: list[list[dict]]) -> list[dict]:
    """小节内跨批次合并：按 zh（去空格小写）去重，别名/英译取并集，定义优先取非空。"""
    merged: dict[str, dict] = {}
    order: list[str] = []
    for batch in batch_results:
        for raw in batch:
            t = normalize_term(raw)
            if t is None:
                continue
            key = re.sub(r"\s+", "", t["zh"]).lower()
            if key not in merged:
                merged[key] = t
                order.append(key)
                continue
            cur = merged[key]
            if not cur["en"] and t["en"]:
                cur["en"] = t["en"]
            if not cur["definition"] and t["definition"]:
                cur["definition"] = t["definition"]
            for field in ("aliases_zh", "aliases_en"):
                for a in t[field]:
                    if a not in cur[field]:
                        cur[field].append(a)
    return [merged[k] for k in order]


# ==================== 小节聚合与分批 ====================

def group_sections(blocks: list[dict]) -> list[dict]:
    """按 parent_ref 聚合 text 块为有序小节列表。

    返回 [{ref, blocks:[block...], char_count}]，顺序按首次出现。
    """
    sections: list[dict] = []
    index: dict[tuple, dict] = {}
    for b in blocks:
        if b.get("block_type") != "text":
            continue
        pr = b.get("parent_ref") or {}
        key = (
            pr.get("book_title", ""), pr.get("volume_title", ""),
            pr.get("chapter_title", ""), pr.get("section_title", ""),
            pr.get("subsection_title", ""),
        )
        if key not in index:
            sec = {"ref": dict(pr), "blocks": [], "char_count": 0}
            index[key] = sec
            sections.append(sec)
        index[key]["blocks"].append(b)
        index[key]["char_count"] += len(b.get("text", ""))
    # 块内按 block_index_in_section 稳定排序
    for sec in sections:
        sec["blocks"].sort(key=lambda b: b.get("block_index_in_section", 0))
    return sections


def split_batches(section: dict, batch_chars: int) -> list[list[dict]]:
    """按块边界把小节软切为 ~batch_chars 的批次，返回每批的块列表（单块超长则一块一批）。"""
    batches: list[list[dict]] = []
    buf: list[dict] = []
    buf_len = 0
    for b in section["blocks"]:
        text = b.get("text", "").strip()
        if not text:
            continue
        if buf and buf_len + len(text) > batch_chars:
            batches.append(buf)
            buf, buf_len = [], 0
        buf.append(b)
        buf_len += len(text)
        if buf_len >= batch_chars:
            batches.append(buf)
            buf, buf_len = [], 0
    if buf:
        batches.append(buf)
    return batches


def join_blocks(blocks: list[dict]) -> str:
    return "\n\n".join(b.get("text", "").strip() for b in blocks if b.get("text", "").strip())


def is_sensitive_content(exc: BaseException) -> bool:
    """端点内容审核（SensitiveContentDetected）：确定性 400，重试/等待均无效，需拆分跳过。"""
    body = getattr(exc, "body", None)
    code = ""
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            code = str(err.get("code") or "")
    elif isinstance(body, str):
        code = body
    # 兜底：部分 SDK 版本 body 非 dict，错误码在异常字符串中
    return "SensitiveContentDetected" in f"{code} {exc}"


# 非配额瞬时错误（超时/网络）退避序列（秒）
_BACKOFF = [20, 40, 60]
# 频控（429 AccountRateLimit）触发后的全局冷却秒数
_RATE_LIMIT_COOLDOWN = 60
# 输入超过该字符数却返回 0 术语，视为可疑空响应（finish_reason=length 截断），需重试/拆半
_EMPTY_MIN_CHARS = 400
# 单任务瞬时错误最大尝试次数（超时不原地重试到死，超预算则拆半减负载）
_MAX_ATTEMPTS = 3
# 并发硬上限（Lite 套餐保守值，AIMD 在此之内自适应）
_CONCURRENCY_CAP = 6

import dataclasses
import time
from collections import deque

import openai


class _EmptyResponse(Exception):
    """输入较长却返回 0 术语（推理模型思维链耗尽，finish_reason=length）。"""

    def __init__(self, chars: int):
        self.chars = chars


# ==================== 自适应并发限流（AIMD：加性增、乘性减） ====================

class AdaptiveLimiter:
    """有效并发 = limit（动态）；active 为在飞请求数。频控/超时窗口触发乘性减，
    连续成功触发加性增（小步进，不超过 cap）。"""

    def __init__(self, init: int, cap: int = _CONCURRENCY_CAP):
        self.cap = cap
        self.limit = max(1, min(init, cap))
        self.active = 0
        self._cond = asyncio.Condition()

    async def acquire(self) -> None:
        async with self._cond:
            while self.active >= self.limit:
                await self._cond.wait()
            self.active += 1

    async def release(self) -> None:
        async with self._cond:
            self.active -= 1
            self._cond.notify_all()

    async def decrease(self, reason: str) -> None:
        async with self._cond:
            new = max(1, self.limit // 2)
            if new != self.limit:
                logger.warning("[aimd] 并发 %d→%d（%s）", self.limit, new, reason)
                self.limit = new
            self._cond.notify_all()

    async def maybe_increase(self, stats: "RunStats") -> None:
        async with self._cond:
            if (
                self.limit < self.cap
                and stats.consecutive_success >= 20
                and time.monotonic() - stats.last_increase_ts > 120
            ):
                self.limit += 1
                stats.last_increase_ts = time.monotonic()
                logger.info("[aimd] 并发提升至 %d（连续成功 %d）", self.limit, stats.consecutive_success)
                self._cond.notify_all()


# ==================== 配额门（5h 窗口耗尽时全队统一等待，单协调者） ====================

class QuotaGate:
    """配额门：5h 窗口耗尽时，首个触发者成为协调者启动一个等待 task；
    其余触发者与 worker 都挂在 gate() 上等待放行，等待只发生一次。"""

    def __init__(self, wait_seconds: int, sleep=None):
        self.wait_seconds = wait_seconds
        self._open = asyncio.Event()
        self._open.set()
        self._lock = asyncio.Lock()
        self._wait_task: asyncio.Task | None = None
        self._sleep = sleep or asyncio.sleep

    async def gate(self) -> None:
        await self._open.wait()

    async def _do_wait(self, what: str, exc: Exception) -> None:
        logger.warning(
            "[quota] %s 触发配额超限，全队暂停等待 %d 分钟：%s",
            what, self.wait_seconds // 60, str(exc)[:150],
        )
        try:
            elapsed = 0
            while elapsed < self.wait_seconds:
                await self._sleep(min(60, self.wait_seconds - elapsed))
                elapsed += 60
                if elapsed % 300 < 60:
                    logger.info(
                        "[quota] 全队配额等待中：已等待 %d 分钟，剩余 %d 分钟",
                        elapsed // 60, (self.wait_seconds - elapsed) // 60,
                    )
        finally:
            self._wait_task = None
            self._open.set()
            logger.info("[quota] 配额等待结束，全队恢复")

    async def trigger_and_wait(self, what: str, exc: Exception) -> None:
        """首个触发者创建协调等待 task（单例）；并发触发者直接等门放行，
        不会因「等待结束后才拿到锁」而重复等待一整轮。"""
        async with self._lock:
            if self._open.is_set() and self._wait_task is None:
                self._open.clear()
                self._wait_task = asyncio.create_task(self._do_wait(what, exc))
        await self._open.wait()


# ==================== 任务与小节状态 ====================

@dataclasses.dataclass
class BatchTask:
    vol_label: str
    sec_idx: int
    sec_total: int
    sec_key: str
    slot: str
    blocks: list
    ref: dict
    label: str
    attempt: int = 0          # 瞬时错误尝试计数
    empty_retries: int = 0    # 空响应重试计数


class SectionState:
    def __init__(self, sec: dict, si: int, total: int, vol_label: str):
        self.key = section_key(sec["ref"])
        self.ref = sec["ref"]
        self.char_count = sec["char_count"]
        self.sec = sec
        self.si = si
        self.total = total
        self.vol_label = vol_label
        self.results: dict[str, list] = {}
        self.skipped = 0
        self.leaf_total = 0
        self.leaf_done = 0
        self.failed = False
        self.fatal_reason: str | None = None
        self.done_event = asyncio.Event()
        self.terms: list = []


@dataclasses.dataclass
class RunStats:
    http_calls: int = 0
    retry_calls: int = 0
    completed_leaves: int = 0
    consecutive_success: int = 0
    last_increase_ts: float = 0.0
    recent_done: deque = dataclasses.field(default_factory=lambda: deque())
    timeout_hits: deque = dataclasses.field(default_factory=lambda: deque())


@dataclasses.dataclass
class RunCtx:
    llm: object
    limiter: AdaptiveLimiter
    quota: QuotaGate
    stats: RunStats
    stop_flag: Path
    timeout: float = 240.0
    cooldown_until: float = 0.0
    states: dict = dataclasses.field(default_factory=dict)


def _slot_order(slot: str) -> tuple:
    return tuple(int(x) for x in slot.split("."))


def _maybe_finish_state(state: SectionState) -> None:
    """叶任务全部完成（且小节未失败）时触发收尾事件。拆半净增叶不会误触发。"""
    if not state.failed and not state.done_event.is_set() and state.leaf_done >= state.leaf_total:
        state.done_event.set()


def _task_llm_chat(llm, blocks: list, ref: dict, timeout: float):
    """单次 HTTP 调用（client 外层 max_retries=1、SDK 内部 max_retries=0，
    重试决策完全由调度层掌握）。返回原始文本。"""
    text = join_blocks(blocks)
    messages = build_messages(ref, text)
    return llm.chat(
        messages, json_mode=True, temperature=0.2, max_tokens=8192,
        timeout=timeout, max_retries=1, sdk_max_retries=0,
    )


async def _split_or_finish(task: BatchTask, state: SectionState, queue: asyncio.Queue,
                           ctx: RunCtx, what: str, reason: str,
                           *, skip_single: bool = False) -> None:
    """拆半入队；单块则按原因跳过/接受 0/标记失败。"""
    if len(task.blocks) > 1:
        mid = len(task.blocks) // 2
        state.leaf_total += 1  # 1 叶 → 2 叶，净增 1
        logger.warning("%s：%s，拆分为 %d+%d 块分别重试", what, reason, mid, len(task.blocks) - mid)
        for slot, sub in ((task.slot + ".0", task.blocks[:mid]), (task.slot + ".1", task.blocks[mid:])):
            await queue.put(dataclasses.replace(
                task, slot=slot, blocks=sub, attempt=0, empty_retries=0,
            ))
        return
    if skip_single:
        state.skipped += 1
        logger.warning("%s：单块触发%s，跳过该块（%d 字符）", what, reason, len(join_blocks(task.blocks)))
    else:
        logger.warning("%s：单块%s，接受 0 术语（%d 字符）", what, reason, len(join_blocks(task.blocks)))
    state.results[task.slot] = []
    state.leaf_done += 1
    ctx.stats.completed_leaves += 1
    _maybe_finish_state(state)


async def _handle_error(exc: Exception, task: BatchTask, state: SectionState,
                        queue: asyncio.Queue, ctx: RunCtx, what: str) -> None:
    stats = ctx.stats
    stats.consecutive_success = 0

    # 1) 配额耗尽（5h 窗口）：全队统一等待后重放，不消耗尝试预算
    if is_quota_exceeded(exc):
        await ctx.quota.trigger_and_wait(what, exc)
        await queue.put(task)
        return

    # 2) 频控（429 AccountRateLimitExceeded）：乘性减并发 + 60s 冷却后重放
    if isinstance(exc, openai.RateLimitError):
        stats.retry_calls += 1
        await ctx.limiter.decrease("频控 429 AccountRateLimit")
        ctx.cooldown_until = max(ctx.cooldown_until, time.monotonic() + _RATE_LIMIT_COOLDOWN)
        logger.warning("%s：频控（%s），降并发并冷却 60s 后重放", what, str(exc)[:100])
        await queue.put(task)
        return

    # 3) 内容审核（确定性 400）：拆半；单块跳过
    if is_sensitive_content(exc):
        await _split_or_finish(task, state, queue, ctx, what, "内容审核", skip_single=True)
        return

    # 4) 瞬时错误（超时/连接/5xx）：有限退避重试，超预算则拆半减负载（超时不做无收益的原地死重试）
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError)):
        stats.retry_calls += 1
        if isinstance(exc, openai.APITimeoutError):
            now = time.monotonic()
            stats.timeout_hits.append(now)
            while stats.timeout_hits and now - stats.timeout_hits[0] > 60:
                stats.timeout_hits.popleft()
            if len(stats.timeout_hits) >= 3:
                await ctx.limiter.decrease(f"60s 内 {len(stats.timeout_hits)} 次超时")
                stats.timeout_hits.clear()
        task.attempt += 1
        if task.attempt < _MAX_ATTEMPTS:
            wait = _BACKOFF[min(task.attempt - 1, len(_BACKOFF) - 1)]
            logger.warning(
                "%s：瞬时错误（%s），%ds 后重试（第 %d/%d 次）",
                what, str(exc)[:100], wait, task.attempt, _MAX_ATTEMPTS,
            )
            await asyncio.sleep(wait)
            await queue.put(task)
        else:
            await _split_or_finish(task, state, queue, ctx, what, "重试预算耗尽")
        return

    # 5) 其他 400（参数/格式问题）：不重试，本小节标记失败待下次重跑，不影响其余小节
    if isinstance(exc, openai.BadRequestError):
        logger.error("%s：参数错误（不重试），本小节标记失败、不落盘：%s", what, str(exc)[:150])
        state.failed = True
        state.fatal_reason = str(exc)[:200]
        state.done_event.set()
        return

    # 6) 未知异常：按瞬时错误对待，超预算标记失败
    task.attempt += 1
    if task.attempt < _MAX_ATTEMPTS:
        logger.warning("%s：未知异常（%s），%ds 后重试", what, str(exc)[:100], _BACKOFF[0])
        await asyncio.sleep(_BACKOFF[0])
        await queue.put(task)
    else:
        logger.error("%s：未知异常重试耗尽，本小节标记失败：%s", what, str(exc)[:150])
        state.failed = True
        state.fatal_reason = str(exc)[:200]
        state.done_event.set()


def _fail_all_pending(ctx: RunCtx, reason: str) -> None:
    """停止标志触发：未完成小节统一标记失败（不落盘，下次断点重跑）。"""
    for s in ctx.states.values():
        if not s.done_event.is_set():
            s.failed = True
            s.fatal_reason = reason
            s.done_event.set()


async def worker(queue: asyncio.Queue, ctx: RunCtx, wid: int) -> None:
    while True:
        task = await queue.get()
        try:
            if ctx.stop_flag.exists():
                _fail_all_pending(ctx, "停止标志")
                return
            state = ctx.states.get(task.sec_key)
            if state is None or state.failed:
                continue  # 小节已失败，丢弃残余任务

            await ctx.quota.gate()
            await ctx.limiter.acquire()
            try:
                now = time.monotonic()
                if now < ctx.cooldown_until:
                    await asyncio.sleep(ctx.cooldown_until - now)

                what = f"{task.label} [slot {task.slot}]"
                ctx.stats.http_calls += 1
                try:
                    raw = await _task_llm_chat(ctx.llm, task.blocks, task.ref, ctx.timeout)
                except Exception as e:  # noqa: BLE001
                    await _handle_error(e, task, state, queue, ctx, what)
                else:
                    items = parse_terms_array(raw)
                    if not items and len(join_blocks(task.blocks)) > _EMPTY_MIN_CHARS:
                        raise _EmptyResponse(len(join_blocks(task.blocks)))
                    # 成功叶
                    ctx.stats.consecutive_success += 1
                    ctx.stats.completed_leaves += 1
                    ctx.stats.recent_done.append(time.monotonic())
                    state.results[task.slot] = items
                    state.leaf_done += 1
                    _maybe_finish_state(state)
                    logger.info(
                        "%s：%d 字符 → %d 个术语%s",
                        what, sum(len(b.get("text", "")) for b in task.blocks), len(items),
                        f"（并发 {ctx.limiter.active}/{ctx.limiter.limit}）",
                    )
                    await ctx.limiter.maybe_increase(ctx.stats)
            finally:
                await ctx.limiter.release()

            # 空响应重试/拆半（在 limiter 外做，避免占着并发槽 sleep）
            # 注：_EmptyResponse 在 try 内 raise 会被 finally 释放后传播——改为在此处理
        except _EmptyResponse as e:
            stats = ctx.stats
            stats.consecutive_success = 0
            if task.empty_retries < 2:
                task.empty_retries += 1
                wait = _BACKOFF[min(task.empty_retries - 1, len(_BACKOFF) - 1)]
                logger.warning(
                    "%s：%d 字符输入却 0 术语（疑似 length 截断），%ds 后重试（第 %d 次）",
                    task.label, e.chars, wait, task.empty_retries,
                )
                stats.retry_calls += 1
                await asyncio.sleep(wait)
                await queue.put(task)
            else:
                await _split_or_finish(task, state, queue, ctx, task.label, "空响应重试仍为空")
        finally:
            queue.task_done()


def section_key(ref: dict) -> str:
    return "||".join([
        str(ref.get("chapter_title", "")),
        str(ref.get("section_title", "")),
        str(ref.get("subsection_title", "")),
    ])


# ==================== 产物读写 ====================

def load_terms_file(path: Path) -> dict:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "sections" in data:
                return data
        except Exception as e:  # noqa: BLE001
            logger.warning("产物文件损坏，将重建：%s（%s）", path, e)
    return {}


def done_section_keys(data: dict) -> set[str]:
    return {
        section_key(s.get("ref", {}))
        for s in data.get("sections", [])
        if s.get("status") == "done"
    }


def save_terms_file(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# ==================== 主流程 ====================

async def extract_volume(
    chunks_path: Path,
    out_path: Path,
    domain: str,
    ctx: RunCtx,
    *,
    batch_chars: int,
    dry_run: bool,
) -> dict:
    """处理单个 chunks.json（一卷）：批次级并发 + AIMD 限流 + 配额门。返回统计。"""
    data = json.loads(chunks_path.read_text(encoding="utf-8"))
    blocks = data.get("blocks", [])
    source_ref = data.get("source_ref", {})
    sections = group_sections(blocks)
    vol_label = source_ref.get("volume_title") or chunks_path.stem
    logger.info(
        "《%s》：%d 个 text 块，%d 个小节，%.1f 万字符 → %s",
        vol_label,
        sum(1 for b in blocks if b.get("block_type") == "text"),
        len(sections),
        sum(s["char_count"] for s in sections) / 10000,
        out_path.name,
    )

    out = load_terms_file(out_path)
    if not out:
        try:
            rel_chunks = str(chunks_path.relative_to(_PROJECT_ROOT))
        except ValueError:
            rel_chunks = str(chunks_path)
        out = {
            "schema_version": 1,
            "domain": domain,
            "source_ref": {**source_ref, "chunks_path": rel_chunks},
            "sections": [],
            "stats": {},
        }
    out_sections = out["sections"]
    # 需重跑的已落盘节：
    #  (a) 疑似空响应：done 但 0 术语且内容较长（多为 finish_reason=length 截断）；
    #  (b) 上限版本过旧：terms_cap < 当前 _TERMS_CAP（字段缺失视为当前上限，保留）。
    def _needs_rerun(s: dict) -> bool:
        if s.get("status") != "done":
            return False
        cap_ok = s.get("terms_cap", _TERMS_CAP) >= _TERMS_CAP
        empty_susp = (not s.get("terms")) and (s.get("char_count") or 0) > _EMPTY_MIN_CHARS
        return empty_susp or not cap_ok

    suspicious = [s for s in out_sections if _needs_rerun(s)]
    if suspicious:
        susp_keys = {section_key(s.get("ref", {})) for s in suspicious}
        out["sections"] = [s for s in out_sections if section_key(s.get("ref", {})) not in susp_keys]
        out_sections = out["sections"]
        logger.info(
            "《%s》检测到 %d 个需重跑小节（空响应/上限过旧），将重新提取：%s",
            chunks_path.stem, len(suspicious),
            "; ".join(str((s.get("ref") or {}).get("section_title", ""))[:24] for s in suspicious),
        )
    already = done_section_keys(out)

    n_done = sum(1 for s in out_sections if s.get("status") == "done")
    n_terms = sum(len(s.get("terms") or []) for s in out_sections if s.get("status") == "done")

    pending = [
        (si, sec) for si, sec in enumerate(sections, start=1)
        if section_key(sec["ref"]) not in already
    ]

    if dry_run:
        for si, sec in pending:
            batches = split_batches(sec, batch_chars)
            logger.info(
                "[dry-run] %s 第 %d/%d 节：%d 批，共 %d 字符",
                vol_label, si, len(sections), len(batches), sec["char_count"],
            )
        return {"sections": len(sections), "done": n_done, "terms": n_terms, "batches": 0}

    if not pending:
        logger.info("《%s》全部 %d 小节已完成，跳过", chunks_path.stem, n_done)
        return {"sections": len(sections), "done": n_done, "terms": n_terms, "batches": 0}

    # ---- 建小节状态与初始批次任务 ----
    ctx.states = {}
    queue: asyncio.Queue = asyncio.Queue()
    for si, sec in pending:
        if ctx.stop_flag.exists():
            break
        state = SectionState(sec, si, len(sections), vol_label)
        batches = split_batches(sec, batch_chars)
        state.leaf_total = len(batches)
        ctx.states[state.key] = state
        for bi, bl in enumerate(batches, start=1):
            label = (
                f"术语提取《{vol_label}》"
                f"{sec['ref'].get('chapter_title', '')}/{sec['ref'].get('section_title', '')}"
                f" 第 {si}/{len(sections)} 节 第 {bi}/{len(batches)} 批"
            )
            queue.put_nowait(BatchTask(
                vol_label=vol_label, sec_idx=si, sec_total=len(sections),
                sec_key=state.key, slot=str(bi), blocks=bl, ref=sec["ref"], label=label,
            ))

    save_lock = asyncio.Lock()
    failed_sections: list[str] = []

    async def finalize_one(state: SectionState) -> None:
        nonlocal n_done, n_terms
        await state.done_event.wait()
        if state.failed:
            failed_sections.append(f"第{state.si}节({state.fatal_reason})")
            logger.error(
                "《%s》第 %d/%d 节 失败，不落盘（下次重跑续接）：%s",
                vol_label, state.si, state.total, state.fatal_reason,
            )
            return
        ordered = [state.results[s] for s in sorted(state.results, key=_slot_order)]
        terms = merge_section_terms(ordered)
        state.terms = terms
        async with save_lock:
            out_sections.append({
                "ref": state.ref,
                "status": "done",
                "terms_cap": _TERMS_CAP,
                "char_count": state.char_count,
                "batch_count": state.leaf_total,
                "skipped_blocks": state.skipped,
                "terms": terms,
            })
            n_done += 1
            n_terms += len(terms)
            out["stats"] = {
                "section_total": len(sections),
                "section_done": n_done,
                "term_total": n_terms,
                "batch_total": ctx.stats.completed_leaves,
            }
            save_terms_file(out_path, out)
            logger.info(
                "《%s》第 %d/%d 节 完成：本节 %d 个术语（累计 %d）",
                vol_label, state.si, state.total, len(terms), n_terms,
            )

    async def progress() -> None:
        while True:
            await asyncio.sleep(30)
            now = time.monotonic()
            while ctx.stats.recent_done and now - ctx.stats.recent_done[0] > 300:
                ctx.stats.recent_done.popleft()
            done_secs = sum(1 for s in ctx.states.values() if s.done_event.is_set())
            cooling = "冷却中" if now < ctx.cooldown_until else "正常"
            logger.info(
                "[进度] 《%s》 节 %d/%d | 队列剩余 %d | 并发 %d/%d | 近5分钟完成 %d 叶"
                " | 累计HTTP %d（重试 %d）| 连续成功 %d | %s",
                vol_label, done_secs, len(ctx.states), queue.qsize(),
                ctx.limiter.active, ctx.limiter.limit, len(ctx.stats.recent_done),
                ctx.stats.http_calls, ctx.stats.retry_calls,
                ctx.stats.consecutive_success, cooling,
            )
            if done_secs >= len(ctx.states):
                return

    workers = [asyncio.create_task(worker(queue, ctx, wid)) for wid in range(ctx.limiter.cap)]
    finalizers = [asyncio.create_task(finalize_one(s)) for s in ctx.states.values()]
    prog = asyncio.create_task(progress())

    await asyncio.gather(*finalizers)
    prog.cancel()
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)

    out["stats"] = {
        "section_total": len(sections),
        "section_done": n_done,
        "term_total": n_terms,
        "batch_total": ctx.stats.completed_leaves,
    }
    save_terms_file(out_path, out)
    if failed_sections:
        logger.warning(
            "《%s》%d 个小节失败（重跑同一命令自动续接）：%s",
            chunks_path.stem, len(failed_sections), "; ".join(failed_sections),
        )
    return {
        "sections": len(sections), "done": n_done, "terms": n_terms,
        "batches": ctx.stats.completed_leaves, "failed_sections": len(failed_sections),
    }


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
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


async def run() -> int:
    parser = argparse.ArgumentParser(description="以小节为单位提取弗洛伊德/荣格文集术语（中英对照+别名+中文简释）")
    parser.add_argument("--domain", choices=["freud", "jung", "all"], default="all")
    parser.add_argument("--only", default=None, help="仅处理文件名包含该子串的卷（如 卷01）")
    parser.add_argument("--chunks-dir", default=None, help="chunks.json 根目录（默认 knowledge_chunks）")
    parser.add_argument("--out-dir", default=None, help="产物目录（默认 knowledge_terms）")
    parser.add_argument("--batch-chars", type=int, default=3500, help="每批输入字符上限（默认 3500）")
    parser.add_argument("--wait-hours", type=float, default=2.0, help="配额超限等待小时数（默认 2）")
    parser.add_argument("--concurrency", type=int, default=3,
                        help=f"初始并发数（AIMD 自适应，硬上限 {_CONCURRENCY_CAP}；1=串行回退）")
    parser.add_argument("--timeout", type=float, default=240.0, help="单次请求超时秒数（默认 240）")
    parser.add_argument("--dry-run", action="store_true", help="只打印批次计划，不调用 LLM")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.data_dir / "extract_terms.log")
    stop_flag = settings.data_dir / "extract_terms_stop.flag"

    chunks_root = Path(args.chunks_dir) if args.chunks_dir else settings.knowledge_chunks_path
    if not chunks_root.is_absolute():
        chunks_root = _PROJECT_ROOT / chunks_root
    out_root = Path(args.out_dir) if args.out_dir else _PROJECT_ROOT / "knowledge_terms"
    if not out_root.is_absolute():
        out_root = _PROJECT_ROOT / out_root

    domains = ["freud", "jung"] if args.domain == "all" else [args.domain]
    targets: list[tuple[str, Path]] = []
    for dom in domains:
        dom_dir = chunks_root / dom
        if not dom_dir.exists():
            logger.warning("目录不存在，跳过：%s", dom_dir)
            continue
        for p in sorted(dom_dir.glob("*.chunks.json")):
            if args.only and args.only not in p.name:
                continue
            targets.append((dom, p))

    if not targets:
        logger.error("未匹配到任何 chunks.json（domain=%s only=%s）", args.domain, args.only)
        return 2

    logger.info(
        "===== 术语提取启动：%d 个卷，batch=%d 字符，并发=%d（上限 %d），timeout=%ds，"
        "配额等待=%d 分钟，dry_run=%s =====",
        len(targets), args.batch_chars, args.concurrency, _CONCURRENCY_CAP,
        int(args.timeout), int(args.wait_hours * 60), args.dry_run,
    )

    llm = None if args.dry_run else LLMClient(settings)
    wait_seconds = int(args.wait_hours * 3600)
    # 全局并发上下文（卷间共享 AIMD 状态与配额门）
    ctx = RunCtx(
        llm=llm,
        limiter=AdaptiveLimiter(args.concurrency),
        quota=QuotaGate(wait_seconds),
        stats=RunStats(),
        stop_flag=stop_flag,
        timeout=args.timeout,
    )
    grand = {"volumes": 0, "sections": 0, "done": 0, "terms": 0, "batches": 0}
    failed_volumes: list[str] = []

    for dom, chunks_path in targets:
        if stop_flag.exists():
            logger.warning("停止标志存在，剩余卷留待下次续跑")
            break
        out_path = out_root / dom / f"{chunks_path.stem}.terms.json"
        try:
            stat = await extract_volume(
                chunks_path, out_path, dom, ctx,
                batch_chars=args.batch_chars, dry_run=args.dry_run,
            )
        except Exception as e:  # noqa: BLE001
            # 单卷失败不中断其余卷；该卷小节 checkpoint 已逐节落盘，重跑自动续接
            logger.exception("卷处理失败 %s：%s（跳过该卷，断点已保存，重跑可续）", chunks_path.name, e)
            failed_volumes.append(chunks_path.name)
            continue
        grand["volumes"] += 1
        for k in ("sections", "done", "terms", "batches"):
            grand[k] += stat[k]
        logger.info(
            "《%s》完成：小节 %d/%d，术语 %d，批次 %d",
            chunks_path.stem, stat["done"], stat["sections"], stat["terms"], stat["batches"],
        )

    if failed_volumes:
        logger.warning("===== %d 个卷未完成（重跑同一命令即可断点续跑）：%s =====",
                       len(failed_volumes), ", ".join(failed_volumes))
    logger.info(
        "===== 本轮结束：成功 %d 卷，小节 %d，术语累计 %d，LLM 批次 %d =====",
        grand["volumes"], grand["done"], grand["terms"], grand["batches"],
    )
    return 1 if failed_volumes else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run()) or 0)
