"""PTZ 的常數（舊系統的 PTZ 服務；數字都是那邊現場調出來的，改之前先看註解）。"""

from __future__ import annotations

import os


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


DEFAULT_ONVIF_PORT = int(os.environ.get("PTZ_ONVIF_PORT", "2020"))     # Tapo 的 ONVIF 埠

# 連續移動看門狗：送出 move 後這麼久沒有新指令就強制停（前端長按每 1.5 秒 keepalive）
MOVE_WATCHDOG_SEC = _f("PTZ_MOVE_WATCHDOG_SEC", 2.5)

# 本批 TP-Link 的 ONVIF 怪癖：ContinuousMove 的 pan 符號跟 RelativeMove 相反。取負讓「正 pan = 向右」各端點一致
CONTINUOUS_PAN_SIGN = int(os.environ.get("PTZ_CONTINUOUS_PAN_SIGN", "-1"))

# 自動歸位：每 10 分鐘把有歸位點的相機強定位回去
SCHEDULE_INTERVAL_SEC = _f("PTZ_SCHEDULE_INTERVAL_SEC", 600)

# 失準偵測 + 自動回正
DRIFT_CHECK_INTERVAL = _f("PTZ_DRIFT_CHECK_INTERVAL", 90)   # 每台平均每隔幾秒檢查一次（分攤在整個週期）
DRIFT_WORKERS = int(os.environ.get("PTZ_DRIFT_WORKERS", "8"))   # 同時檢查幾台
DRIFT_MAX_RECOVERIES = 2              # 同時最多幾台在回正（避免校準風暴）
DRIFT_SHIFT_PX = 14                   # 256x144 上位移超過此值視為失準
DRIFT_CONSECUTIVE = 3                 # 連續幾次失準才自動回正
DRIFT_CONTROL_QUIET = 60              # 最近這麼久內有人操作 → 不檢查、不歸位（秒）
DRIFT_ATHOME_TOL = 0.03               # 座標與基準差在此內才算「自認在家」
DRIFT_RECOVER_WAIT = 40               # calibrateMotor 後等掃描秒數
DRIFT_ALIGN_GAIN_X = 570.0            # px / 單位 pan（視覺對齊的相對移動增益）
DRIFT_ALIGN_GAIN_Y = 200.0            # px / 單位 tilt
DRIFT_ALIGN_MAX_ITER = 4
DRIFT_ALIGN_OK_PX = 8                 # 位移小於此值視為對齊完成
# 信心低於此 → 相位相關峰值不可信，不移動（免得在雜訊上發散）。低紋理相機（反光場景）可往上調
DRIFT_ALIGN_MIN_CONF = _f("PTZ_DRIFT_ALIGN_MIN_CONF", 12.0)
# 回正後回寫基準座標的信心門檻：擋掉低信心鎖錯位置 → 錨點污染（相機越校越歪）
DRIFT_REF_UPDATE_MIN_CONF = _f("PTZ_DRIFT_REF_UPDATE_MIN_CONF", 30.0)

# 抓畫面：向 go2rtc 要截圖，不另開 RTSP（Tapo 同時連線數少，多開一條會擠掉推論那條）
GO2RTC_API = os.environ.get("GO2RTC_API", "http://go2rtc:1984/go2rtc").rstrip("/")
REF_BUCKET = "raw"
REF_PREFIX = "ptz/refs"
