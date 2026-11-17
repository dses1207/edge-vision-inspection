"""偵測 / 判定的資料契約 —— 唯一來源是 ai-infer 的 edgevision-contracts 套件，這裡只轉出。

⚠️ 不要在這裡定義或修改任何欄位。

    以前這裡是 ai-infer 那份的手抄副本，抄漏過兩次，兩次都是【靜默資料遺失】：
      - keypoints（pose 結果）：event-processor 解析時被 Pydantic 丟掉
      - FrameResult.station（一線多站的站別）：同上，DB 分不出 NG 是哪一站判的
    現在 runner（產生端）與 event-processor（消費端）用同一份程式碼。

要加欄位：改 ai-infer/libs/contracts → 升版號 → 發佈到 Gitea 套件庫（ai-infer CI 自動）
        → 這個 repo 的 requirements.txt 升 edgevision-contracts 版號。
"""
from edgevision_contracts.detection import *  # noqa: F401,F403
from edgevision_contracts.detection import (  # noqa: F401
    SCHEMA_VERSION,
    BBox,
    CoordSystem,
    Detection,
    FrameResult,
    ImageMeta,
    LetterboxParams,
    Verdict,
    VerdictStatus,
)
