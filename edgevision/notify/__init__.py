"""通知系統（Notifier）：日報、健康巡檢、NG 通知。

三條 loop 在同一個 process。通道走 Apprise：現在是 Teams，加寄信只改 .env。

  sender.py   投遞：Card → Apprise → 任何通道（Teams / Email / …），tag 路由
  daily.py    ① 每日 08:00 一則日報 —— 同時是 dead-man switch（沒收到 = notifier 掛了）
  health.py   ② 每 60 秒巡檢 /healthz，狀態機只在【變化】時通知
  ng.py       ③ NG 通知（第二階段，未實作）
"""
