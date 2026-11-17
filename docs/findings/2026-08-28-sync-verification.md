# 實測記錄：影像與偵測結果的幀級同步

> 日期：2026-08-28
> 環境：go2rtc 1.9.14、Chromium 151、cam156 (720p H.264)

---

## 結論

**主方案（rVFC + rtpTimestamp 直接配對）不成立。**

兩條路徑的 RTP timestamp 各自從獨立的基準起算，且偏移會抖動。

```
前端（WebRTC）  ：7,366,500 起算
推論端（RTSP）  ：13,303,603 起算
偏移中位數       ：5,946,140
偏移抖動         ：200 ms（標準差 59 ms）
共同值           ：0 個
```

**根因**：go2rtc 對每個 consumer 開獨立 session，RTP timestamp 依 RTSP/WebRTC
規範各自從自身起點計時。這不是缺陷，是協定的正常行為。

---

## 已驗證與排除的方案

### ① 直接用 rtpTimestamp 配對 —— 不成立

步進模式一致（4500/4500/9000 循環），證明**同源**，
但起始基準不同且偏移抖動 200ms。

### ② 固定偏移校準 —— 準確度不足

```
偏移全距 200ms，標準差 59ms
→ 框會在 ±100ms 範圍飄
→ 輸送帶場景下物件已移動數公分
```

不符合驗收標準（移動物體偏差 < 10px）。

### ③ rVFC 的 captureTime —— 不可用

`requestVideoFrameCallback` metadata 各欄位實測：

| 欄位 | 可用 | 備註 |
|---|---|---|
| `rtpTimestamp` | 是 | 但基準與推論端不同（見上） |
| `receiveTime` | 是 | 本機接收時刻，非擷取時刻 |
| `mediaTime` | 是 | 播放時間軸 |
| `presentationTime` | 是 | |
| `expectedDisplayTime` | 是 | |
| `processingDuration` | 是 | |
| **`captureTime`** | **** | **不可用** |

### ④ 用 getStats 的 RTCP SR 自建映射 —— 不可行

go2rtc **確實有送** RTCP Sender Report：

```
remote-outbound-rtp:
  remoteTimestamp: 1787888365523.28   ← 持續更新
  reportsSent: 1 → 3 → 4 → 6 → 7 → 9
```

但 `RTCRemoteOutboundRtpStreamStats` **不暴露 SR 內的 RTP timestamp 欄位**
（完整欄位已 dump 確認），因此無法建立 RTP牆鐘的換算基準。

---

## 剩餘方案

| 方案 | 準確度 | 代價 | 評估 |
|---|---|---|---|
| **影像內嵌 frame id** | 完美 | 需重新編碼 | 首選備案 |
| 伺服器端燒錄框 | 完美 | 重新編碼 + 框無法關閉/篩選 | 保底 |
| 內容雜湊配對 | ? | 前端每幀計算 | 未驗證 |

### 為何內嵌 frame id 優於伺服器端燒錄

兩者都需要重新編碼，成本相同。但內嵌 frame id 保留了：
- 前端可自由開關框線、篩選類別
- 框的樣式可調整而不需重新編碼
- 結果資料本來就要傳給下游，等於順便用

---

## 附：目前系統的行為

MVP 目前仍以 rtpTimestamp 配對，因此：
- 精確配對命中率 0%
- 全部走「沿用上一筆結果」路徑（MAX_AGE 400ms 內）
- 畫面有框但位置有約 100ms 偏移

此狀態足以驗證管線串接，但不符合生產要求。
