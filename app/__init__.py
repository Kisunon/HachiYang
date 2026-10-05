"""HachiYang 陪伴型 Agent 应用包。

模块划分见 docs/IMPLEMENTATION_DESIGN.md §3.1：
- config：环境变量加载与启动校验
- llm：deepseek-v4-pro / doubao-embedding-vision 客户端
- db：SQLite + ChromaDB 基础设施
- core：对话编排、意图、人格、置信度、统一数据结构
- channels：渠道适配（飞书）
- rag / memory / care / web：后续阶段实现
"""

__version__ = "0.1.0"
