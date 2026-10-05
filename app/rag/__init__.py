"""RAG 检索模块（Phase 3）。

对齐 docs/tech_docs/RAG_SYSTEM_DESIGN.md：
- 文档解析与层级分块（parsers + chunker）
- 章节摘要（summarizer）+ 术语表（term_builder）
- 存储（stores：ChromaDB + SQLite FTS5）
- 增量索引（indexer）+ 健康检查
- 混合检索管线（retriever + reranker + term_lookup）
"""
