"""用户画像 / 语义记忆（User Profile / Semantic Memory）—— Phase 4 任务 4.3。

对齐 docs/tech_docs/MEMORY_SYSTEM_DESIGN.md §2.3 / §5.2 / §6：
- 结构化事实条目（非对话片段），跨会话长期持久；
- SQLite user_profiles 表存完整画像 JSON；
- MemoryManager 写入路径异步调用 LLM 抽取事实（health_sensitive 只存 topic 标签），
  本模块负责 merge 规则与持久化；
- sleep_pattern 字段直接供 Phase 6 关怀调度模块读取（M-05）。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.config import Settings, get_settings
from app.db.connection import db_conn

logger = logging.getLogger(__name__)


def default_profile(user_id: str) -> dict:
    """新建用户画像骨架（对齐 MEMORY_SYSTEM_DESIGN §2.3 Schema）。"""
    return {
        "user_id": user_id,
        "basic_info": {
            "nickname": "",
            "known_identity_tags": [],
        },
        "sleep_pattern": {
            "avg_bedtime": "",
            "avg_wake_time": "",
            "irregular_days": [],
            "last_updated": "",
            # Phase 6 关怀调度：最近一次起居事件（避免已道早晚安后重复提醒）
            "last_state": "",          # asleep / awake
            "last_state_date": "",     # 本地日期 YYYY-MM-DD
            "last_event_at": "",       # 本地时刻 HH:MM
        },
        "emotional_baseline": {
            "recurring_themes": [],
            "current_mood_hint": "",
        },
        "discussed_theories": [],
        "preferences": {
            "known_interests": [],
            "communication_style_hint": "",
        },
        "health_sensitive": [],
    }


def merge_profile_facts(profile: dict, facts: dict) -> dict:
    """把 LLM 抽取的画像事实 merge 进现有画像（原地更新并返回）。

    Merge 规则：
    - basic_info：非空字段覆盖，标签数组去重合并；
    - sleep_pattern：非空时间覆盖均值；irregular_days 去重追加（上限 30 条）；
      last_updated 更新为今天；
    - emotional_baseline：recurring_themes 去重合并；current_mood_hint 覆盖；
    - discussed_theories：按 name 去重，已存在则更新 discussed_at/depth；
    - preferences：known_interests 去重合并；communication_style_hint 非空覆盖；
    - health_sensitive：仅 topic 标签（脱敏约束），按 topic 去重，记录首次提及。
    """
    if not facts:
        return profile
    today = time.strftime("%Y-%m-%d")

    basic = facts.get("basic_info") or {}
    if basic.get("nickname"):
        profile["basic_info"]["nickname"] = str(basic["nickname"])[:40]
    tags = basic.get("known_identity_tags") or []
    if tags:
        merged = profile["basic_info"]["known_identity_tags"]
        for t in tags:
            if t not in merged:
                merged.append(str(t)[:30])

    sleep = facts.get("sleep_pattern") or {}
    sp = profile["sleep_pattern"]
    changed = False
    if sleep.get("avg_bedtime"):
        sp["avg_bedtime"] = str(sleep["avg_bedtime"])[:8]
        changed = True
    if sleep.get("avg_wake_time"):
        sp["avg_wake_time"] = str(sleep["avg_wake_time"])[:8]
        changed = True
    days = sleep.get("irregular_days") or []
    if days:
        merged_days = sp.setdefault("irregular_days", [])
        for d in days:
            if d not in merged_days:
                merged_days.append(str(d)[:10])
        del merged_days[:-30]  # 只保留最近 30 条
        changed = True
    if changed:
        sp["last_updated"] = today

    emo = facts.get("emotional_baseline") or {}
    themes = emo.get("recurring_themes") or []
    if themes:
        merged_themes = profile["emotional_baseline"].setdefault("recurring_themes", [])
        for t in themes:
            if t not in merged_themes:
                merged_themes.append(str(t)[:30])
        del merged_themes[:-20]
    if emo.get("current_mood_hint"):
        profile["emotional_baseline"]["current_mood_hint"] = str(emo["current_mood_hint"])[:60]

    theories = facts.get("discussed_theories") or []
    existing = {t.get("name"): t for t in profile["discussed_theories"]}
    for th in theories:
        name = str(th.get("name", ""))[:50]
        if not name:
            continue
        if name in existing:
            existing[name]["discussed_at"] = today
            if th.get("depth"):
                existing[name]["depth"] = str(th["depth"])[:10]
        else:
            entry = {
                "name": name,
                "discussed_at": today,
                "depth": str(th.get("depth", "浅"))[:10],
            }
            profile["discussed_theories"].append(entry)
            existing[name] = entry
    del profile["discussed_theories"][:-50]

    prefs = facts.get("preferences") or {}
    interests = prefs.get("known_interests") or []
    if interests:
        merged_int = profile["preferences"].setdefault("known_interests", [])
        for i in interests:
            if i not in merged_int:
                merged_int.append(str(i)[:30])
        del merged_int[:-30]
    if prefs.get("communication_style_hint"):
        profile["preferences"]["communication_style_hint"] = str(prefs["communication_style_hint"])[:80]

    # 敏感话题：只存 topic 标签（隐私约束，禁止具体内容入库）
    sensitive_topics = facts.get("health_sensitive_topics") or []
    existing_topics = {h.get("topic") for h in profile["health_sensitive"]}
    for topic in sensitive_topics:
        t = str(topic)[:30]
        if t and t not in existing_topics:
            profile["health_sensitive"].append({"topic": t, "first_mentioned": today})
            existing_topics.add(t)
    del profile["health_sensitive"][:-30]

    return profile


class UserProfileStore:
    """用户画像存储（SQLite user_profiles 表，JSON 整体存取）。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def load(self, user_id: str) -> dict:
        """加载画像；不存在则创建默认骨架（不立即落库，首次 merge 时持久化）。"""
        with db_conn(self._settings) as conn:
            row = conn.execute(
                "SELECT profile_json FROM user_profiles WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if row:
            try:
                profile = json.loads(row["profile_json"])
                if isinstance(profile, dict):
                    return self._fill_defaults(profile, user_id)
            except Exception as e:  # noqa: BLE001
                logger.warning("[UserProfile] %s 画像 JSON 损坏，重建默认画像：%s", user_id, e)
        return default_profile(user_id)

    def save(self, user_id: str, profile: dict) -> None:
        """全量覆盖保存画像。"""
        profile["user_id"] = user_id
        now = int(time.time())
        with db_conn(self._settings) as conn:
            conn.execute(
                """
                INSERT INTO user_profiles (user_id, profile_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    updated_at = excluded.updated_at
                """,
                (user_id, json.dumps(profile, ensure_ascii=False), now),
            )
        logger.info("[UserProfile] 画像已保存 user=%s", user_id)

    def merge_and_save(self, user_id: str, facts: dict) -> dict:
        """merge LLM 抽取的画像事实并持久化，返回更新后的画像。"""
        profile = self.load(user_id)
        merge_profile_facts(profile, facts or {})
        self.save(user_id, profile)
        return profile

    def forget_field(self, user_id: str, field_path: str) -> bool:
        """按字段路径遗忘画像内容（scope=profile）。

        field_path 形如 "sleep_pattern"、"emotional_baseline.current_mood_hint"、
        "health_sensitive"。字段存在则重置为默认骨架值并持久化。
        """
        profile = self.load(user_id)
        default = default_profile(user_id)
        keys = [k for k in field_path.split(".") if k]

        target: Any = profile
        target_default: Any = default
        ok = True
        for key in keys:
            if not isinstance(target, dict) or key not in target:
                ok = False
                break
            if not isinstance(target_default, dict) or key not in target_default:
                # 动态键（如 discussed_theories 的条目）：父层处理
                break
            target = target[key]
            target_default = target_default[key]

        if not ok or not keys:
            logger.warning("[UserProfile] forget_field 路径无效：%s", field_path)
            return False

        parent = profile
        for key in keys[:-1]:
            parent = parent[key]
        parent[keys[-1]] = target_default if not isinstance(target_default, (dict, list)) else json.loads(
            json.dumps(target_default, ensure_ascii=False)
        )
        self.save(user_id, profile)
        logger.info("[UserProfile] 已遗忘字段 user=%s path=%s", user_id, field_path)
        return True

    # ==================== 内部工具 ====================

    @staticmethod
    def _fill_defaults(profile: dict, user_id: str) -> dict:
        """旧版本画像补齐新增字段（避免 KeyError）。"""
        base = default_profile(user_id)
        for key, value in base.items():
            if key not in profile:
                profile[key] = value
            elif isinstance(value, dict) and isinstance(profile[key], dict):
                for k2, v2 in value.items():
                    profile[key].setdefault(k2, v2)
        return profile
