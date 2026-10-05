"""飞书渠道适配子包。

结构（对齐 docs/IMPLEMENTATION_DESIGN.md §3.1 / §6.3）：
- handler.py      事件接收层（WebSocket/Webhook 共用）：解密校验、事件去重、提取 chat_id + open_id + 文本
- adapter.py      消息格式转换：lark event ↔ UnifiedRequest / UnifiedResponse
- client.py       SDK 封装：tenant_access_token 由 lark-oapi 内部管理，此处仅暴露高层收发 API

设计要点：
- **传输层透明**：handler 只处理事件级别的接收/去重/解密，不关心 WebSocket vs Webhook；
  WebSocket 模式下 SDK 的 EventDispatcherHandler 直接喂给 handler；
  Webhook 模式下 FastAPI 路由先做签名校验（FEISHU_VERIFICATION_TOKEN）+ 加密解密（FEISHU_ENCRYPT_KEY），
  再把解密后的 event body 喂给 handler。
- **会话隔离**：以 ``user_id = f"{chat_id}:{open_id}"`` 为分区键（M-02）。
"""
