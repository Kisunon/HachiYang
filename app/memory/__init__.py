"""Phase 4 记忆层 —— 三层记忆 + Memory Manager 统一编排。

对齐 docs/tech_docs/MEMORY_SYSTEM_DESIGN.md：
- working.WorkingMemory：工作记忆（滑动窗口 + SQLite 持久化 + 懒加载）
- episodic.EpisodicMemory：情景记忆（LLM 提取摘要 + 向量化 + 衰减检索）
- profile.UserProfileStore：用户画像/语义记忆（事实 merge + sleep_pattern）
- manager.MemoryManager：Read / Write / Forget 统一编排接口
"""
