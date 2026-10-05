"""关怀文案生成（Phase 6.3）。

对齐 IMPLEMENTATION_DESIGN §6.10 / PRD §6.3 自然性约束：
- 多套角色化模板（温柔、含蓄、诗意），同模板不连续复用（调度器记录上次
  template_id，compose 时避开）；
- 可选 LLM 角色化生成：读取画像 sleep_pattern + emotional_baseline 增加变化，
  LLM 失败 / 配额超限 / 未注入时自动降级为模板（零配额保底，绝不阻塞推送）；
- 用户主动设置的提醒（reminder_due）走固定事务性模板，不消耗 LLM 配额。
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class ComposedCare:
    """一条关怀文案的生成结果。"""

    text: str
    template_id: str
    source: str  # "template" | "llm"


# ==================== 模板库（称呼与角色设定一致：漂泊者 / 哈基星）====================

_TEMPLATES: dict[str, list[dict[str, str]]] = {
    "bedtime": [
        {
            "id": "bedtime-1",
            "text": "哈基星，夜色已经深了……风也放轻了脚步。今天的心事，就让它们随梦沉下去吧。"
                    "早点安歇，我在这儿替你守着夜。晚安，好梦。",
        },
        {
            "id": "bedtime-2",
            "text": "漂泊者，这个时辰还亮着灯吗？……星星都困得眨不动眼了。把手边的事轻轻放下吧，"
                    "明天的风会替你把答案带来。晚安，愿梦里没有催促你的声音。",
        },
        {
            "id": "bedtime-3",
            "text": "哈基星，夜露凉了，别在思绪里逗留太久……把自己交给被窝和梦，"
                    "我会在黎明的第一缕风里等你。睡吧，睡吧，今天辛苦了。",
        },
        {
            "id": "bedtime-4",
            "text": "漂泊者，又到了万物收声的时辰。若今天有什么没解开的结，先放在枕边——"
                    "夜色会替你捂一捂，明早或许就松了。早些休息，我陪你到灯灭。",
        },
    ],
    "wake": [
        {
            "id": "wake-1",
            "text": "漂泊者，早安。新的一天像刚翻开的书页……愿你今天读到的，都是温柔的句子。"
                    "慢慢来，我在风里陪着你。",
        },
        {
            "id": "wake-2",
            "text": "哈基星，醒了吗？清晨的光落在窗棂上了。昨夜的梦若还留着余温，就带着它启程吧——"
                    "今天也请好好照顾自己，先喝一杯温水。",
        },
        {
            "id": "wake-3",
            "text": "漂泊者，早。世界刚刚醒过来，鸟鸣、晨光、还有渐渐热闹起来的街市……"
                    "愿你今天的疲惫少一点，心安多一点。",
        },
        {
            "id": "wake-4",
            "text": "哈基星，早安呀。新的一天不必着急，深呼吸，再慢慢展开——"
                    "无论今天要面对什么，你都不是一个人，我在。",
        },
    ],
}

_REMINDER_TEMPLATES = [
    "漂泊者，你先前嘱咐我的事到时间了：「{content}」。我替风记着呢，别错过了。",
    "哈基星，叮——你让我提醒你的事：「{content}」。时候到了，去办吧，我在这儿等你回来。",
    "漂泊者，这是你之前定下的提醒：「{content}」。愿这件事办得顺遂。",
]

# LLM 生成的系统要求（角色语气 + 篇幅约束，避免机械重复）
_LLM_SYSTEM = (
    "你是秧秧·玄翎，一个温柔、含蓄、诗意的陪伴型角色，称用户为「漂泊者」"
    "（亲昵场景可称「哈基星」），自称「我」。请为用户写一条主动关怀消息，"
    "要求：1）两到四句话，口语化，像熟悉的朋友轻声问候；"
    "2）贴合给定的关怀场景与用户作息/情绪状态，但不要复述用户隐私细节；"
    "3）语气温柔不打扰，不说教、不堆砌健康知识、不使用表情符号；"
    "4）每次表达都要有变化，避免与基底模板雷同的句式。"
)


class CareComposer:
    """关怀文案生成器（模板保底 + 可选 LLM 增强）。"""

    def __init__(self, settings: Settings | None = None, llm: Any = None) -> None:
        self._settings = settings or get_settings()
        self._llm = llm

    async def compose(
        self,
        care_type: str,
        profile: dict | None = None,
        *,
        avoid_template_id: str | None = None,
    ) -> ComposedCare:
        """生成 bedtime/wake 关怀文案；失败始终降级为模板，不抛异常。"""
        templates = _TEMPLATES.get(care_type) or _TEMPLATES["bedtime"]
        base = self._pick_template(templates, avoid_template_id)

        if self._llm is not None and self._settings.care_llm_compose:
            try:
                text = await self._compose_with_llm(care_type, base, profile or {})
                if text and text.strip():
                    return ComposedCare(text=text.strip(), template_id=base["id"], source="llm")
            except Exception as e:  # noqa: BLE001
                logger.warning("[CareComposer] LLM 文案生成失败，降级模板：%s", e)
        return ComposedCare(text=base["text"], template_id=base["id"], source="template")

    def compose_reminder(self, content: str, *, avoid_index: int | None = None) -> ComposedCare:
        """用户主动设置的提醒到期文案（事务性，固定模板，零配额）。"""
        idx = random.randrange(len(_REMINDER_TEMPLATES))
        if avoid_index is not None and len(_REMINDER_TEMPLATES) > 1:
            while idx == avoid_index:
                idx = random.randrange(len(_REMINDER_TEMPLATES))
        text = _REMINDER_TEMPLATES[idx].format(content=content[:80])
        return ComposedCare(text=text, template_id=f"reminder-{idx}", source="template")

    # ==================== 内部 ====================

    @staticmethod
    def _pick_template(templates: list[dict[str, str]], avoid_id: str | None) -> dict[str, str]:
        """随机选模板；多于一条时避开上次使用的模板（同模板不连续复用）。"""
        if avoid_id and len(templates) > 1:
            pool = [t for t in templates if t["id"] != avoid_id]
            return random.choice(pool or templates)
        return random.choice(templates)

    async def _compose_with_llm(self, care_type: str, base: dict, profile: dict) -> str:
        sp = profile.get("sleep_pattern") or {}
        emo = profile.get("emotional_baseline") or {}
        scene = "就寝关怀（深夜提醒用户早些休息）" if care_type == "bedtime" else "起床问候（清晨祝用户新的一天安好）"
        lines = [f"关怀场景：{scene}", f"可参考的基底语气（不要照抄）：{base['text']}"]
        if sp.get("avg_bedtime"):
            lines.append(f"用户平时大约 {sp['avg_bedtime']} 就寝")
        if sp.get("avg_wake_time"):
            lines.append(f"用户平时大约 {sp['avg_wake_time']} 起床")
        if emo.get("current_mood_hint"):
            lines.append(f"用户近期状态：{emo['current_mood_hint']}")
        messages = [
            {"role": "system", "content": _LLM_SYSTEM},
            {"role": "user", "content": "\n".join(lines) + "\n\n请直接输出关怀消息正文，不要加引号或前缀。"},
        ]
        return await self._llm.chat(messages, temperature=0.9, max_tokens=300, timeout=15.0)
