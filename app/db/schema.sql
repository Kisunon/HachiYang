-- HachiYang SQLite 建表 DDL
-- 对齐 docs/IMPLEMENTATION_DESIGN.md §5.1
-- 幂等：全部使用 IF NOT EXISTS，可重复执行（增量/重启安全）

-- 会话表：记录每个用户的飞书会话
CREATE TABLE IF NOT EXISTS chat_sessions (
    user_id    TEXT PRIMARY KEY,          -- 格式 "chat_id:user_id"
    chat_id    TEXT NOT NULL,             -- 飞书 chat_id（主动推送用）
    open_id    TEXT NOT NULL,             -- 飞书 user open_id
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

-- 工作记忆表：会话内对话历史的持久化备份（防进程重启丢失）
CREATE TABLE IF NOT EXISTS working_memory (
    user_id    TEXT NOT NULL,
    turn_index INTEGER NOT NULL,          -- 会话内递增序号
    role       TEXT NOT NULL,             -- 'user' | 'assistant'
    content    TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (user_id, turn_index)
);

-- 情景记忆表：元数据（向量存 ChromaDB episodic_memory）
CREATE TABLE IF NOT EXISTS episodes (
    episode_id   TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    timestamp    INTEGER NOT NULL,
    summary      TEXT NOT NULL,
    topics       TEXT NOT NULL,           -- JSON 数组
    emotion_tag  TEXT,
    source_window TEXT NOT NULL,          -- JSON 数组（turn_id）
    decay_score  REAL NOT NULL DEFAULT 1.0,
    last_hit_at  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id);

-- 用户画像表：完整画像 JSON
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id      TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    updated_at   INTEGER NOT NULL
);

-- RAG 文档索引状态表
CREATE TABLE IF NOT EXISTS rag_documents (
    document_id          TEXT PRIMARY KEY,
    file_path            TEXT NOT NULL,
    content_hash         TEXT NOT NULL,
    domain               TEXT NOT NULL,   -- freud / jung / philosophy
    language             TEXT,
    status               TEXT NOT NULL,   -- pending / indexing / done / failed / indexed_but_summary_pending / summary_partial
    block_count          INTEGER DEFAULT 0,
    section_summary_count INTEGER DEFAULT 0,
    chapter_summary_count INTEGER DEFAULT 0,
    term_count           INTEGER DEFAULT 0,
    volume_count         INTEGER DEFAULT 1,  -- v1.6：合订本按卷合并后的卷数
    volumes_json         TEXT,              -- v1.6：卷列表 [{volume_index, volume_title, chunks_path}]
    indexed_at           INTEGER,
    retry_count          INTEGER DEFAULT 0
);

-- RAG 段落块元数据表
CREATE TABLE IF NOT EXISTS rag_blocks (
    block_id             TEXT PRIMARY KEY,
    document_id          TEXT NOT NULL,
    domain               TEXT NOT NULL,
    language             TEXT,
    volume_title         TEXT,             -- v1.6 6 级溯源：卷名
    volume_index         INTEGER,          -- v1.6：卷次
    chapter_title        TEXT,
    section_title        TEXT,
    subsection_title     TEXT,             -- v1.6 6 级溯源：小节（多级路径拼接）
    page_range           TEXT,
    block_index_in_section INTEGER,
    char_count           INTEGER,
    block_type           TEXT NOT NULL,    -- text / image / section_summary / chapter_summary
    has_structure        INTEGER DEFAULT 1,
    parent_block_ids     TEXT,             -- JSON 数组（摘要块：链式展开目标）
    parent_chapter_summary_id TEXT,        -- v1.6：节摘要 → 所属章摘要（检查页跳转）
    source_anchor_in_md  TEXT,             -- v1.6：提取稿锚点 #ch-N-sec-N-ss-N
    extracted_md_path    TEXT,             -- v1.6：提取稿文件路径（人工核对跳转）
    text                 TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rag_blocks_doc ON rag_blocks(document_id);
CREATE INDEX IF NOT EXISTS idx_rag_blocks_type ON rag_blocks(block_type);
CREATE INDEX IF NOT EXISTS idx_rag_blocks_vol ON rag_blocks(document_id, volume_index);

-- RAG 术语表
CREATE TABLE IF NOT EXISTS rag_terms (
    term_id              TEXT PRIMARY KEY,
    canonical_term       TEXT NOT NULL,
    synonyms             TEXT,            -- JSON 数组
    language_pairs       TEXT,            -- JSON 对象
    domain               TEXT,
    definition_location  TEXT,            -- JSON 对象
    occurrence_block_ids TEXT,            -- JSON 数组
    related_terms        TEXT             -- JSON 数组
);
CREATE INDEX IF NOT EXISTS idx_rag_terms_canonical ON rag_terms(canonical_term);

-- RAG 全文检索索引（关键词检索，BM25）
CREATE VIRTUAL TABLE IF NOT EXISTS rag_fts USING fts5(
    block_id, text, document_title, chapter_title, domain,
    tokenize='unicode61'
);

-- 待提醒表（Phase 5 指令执行登记；Phase 6 关怀调度器按 remind_at 消费投递）
CREATE TABLE IF NOT EXISTS reminders (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT NOT NULL,
    content       TEXT NOT NULL,
    remind_at     TEXT,              -- ISO 时刻（尽力解析；NULL 表示模糊时间待确认）
    raw_time_text TEXT,
    raw_text      TEXT,
    status        TEXT NOT NULL DEFAULT 'pending',  -- pending / done / cancelled
    created_at    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reminders_status ON reminders(status, remind_at);

-- 关怀推送记录表（Phase 6：单日频次控制 / 模板不连续复用 / 调试台触发记录）
CREATE TABLE IF NOT EXISTS care_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    care_type   TEXT NOT NULL,          -- bedtime / wake / reminder_due
    template_id TEXT,                   -- 文案模板 id（LLM 生成时为生成时基底模板）
    content     TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'sent',  -- sent / failed / dry_run（无推送渠道）
    created_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_care_events_user ON care_events(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_care_events_type ON care_events(care_type, created_at);
