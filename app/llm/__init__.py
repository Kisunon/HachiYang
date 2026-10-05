"""llm：火山方舟模型客户端。

- LLMClient：deepseek-v4-pro（chat 统一入口，意图/回应/摘要/Rerank/Rewrite 共用）
- EmbeddingClient：doubao-embedding-vision（文本/查询向量化）

两者共用 ARK_BASE_URL + ARK_KEY，仅模型名不同（PRD §7.2 共享 Endpoint）。
"""

from app.llm.client import LLMClient
from app.llm.embedding import EmbeddingClient

__all__ = ["LLMClient", "EmbeddingClient"]
