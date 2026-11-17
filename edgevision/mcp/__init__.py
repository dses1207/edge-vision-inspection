"""MCP server：讓 agent 用自然語言操作訓練平台。

它是控制 API 的另一個客戶端（跟前端 /ui、命令列工具平起平坐），
不碰資料庫、不碰 GPU，所以權限、稽核、佇列規則自動沿用。

    python -m edgevision.mcp.login     登入（一次可用 30 天）
    python -m edgevision.mcp.server    啟動 MCP server（stdio，由 MCP client 拉起）
"""
