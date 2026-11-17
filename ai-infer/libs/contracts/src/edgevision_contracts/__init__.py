"""資料契約。推論輸出、判定結果、落地、前端都依賴這份 schema；改動前先想下游有誰。"""
from edgevision_contracts.detection import (
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

__all__ = [
    "SCHEMA_VERSION", "BBox", "CoordSystem", "Detection", "FrameResult",
    "ImageMeta", "LetterboxParams", "Verdict", "VerdictStatus",
]
