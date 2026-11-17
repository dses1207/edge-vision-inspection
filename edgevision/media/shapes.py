"""標注形狀：驗證、與 YOLO 標籤互轉。

座標一律 0–1 正規化（相對原圖寬高）—— 跟 YOLO 標籤同一套，換解析度不用重算。
  rect      points = [[x1, y1], [x2, y2]]（左上、右下）
  polygon   points = [[x, y], ...]（≥3 點）
  keypoints bbox = [[x1, y1], [x2, y2]]，points = [[x, y, v], ...]（v：0 不存在 / 1 被遮 / 2 可見）

每個 shape 記來源：人工標的 {"kind": "human"}；AI 標的 {"kind": "model", "engine", "model", "version", "conf"}。
每個 shape 可帶 id（字串）：審核時靠它判斷框是「原樣 / 修改 / 新增 / 刪除」（edgevision/media/review.py），前端修改框要保留原 id。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

SHAPE_TYPES = ("rect", "polygon", "keypoints")
TASK_SHAPE = {"detect": "rect", "segment": "polygon", "pose": "keypoints"}


class ShapeSource(BaseModel):
    kind: Literal["human", "model"] = "human"
    engine: str | None = None          # registry | yolo_world | sam
    model: str | None = None
    version: int | None = None
    conf: float | None = None


class Shape(BaseModel):
    id: str | None = Field(None, max_length=64)
    type: Literal["rect", "polygon", "keypoints"]
    cls: str = Field(alias="class")
    points: list[list[float]]
    bbox: list[list[float]] | None = None
    source: ShapeSource = Field(default_factory=ShapeSource)

    model_config = {"populate_by_name": True}

    @field_validator("points", "bbox")
    @classmethod
    def _in_range(cls, v):
        if v is None:
            return v
        for p in v:
            if len(p) < 2 or not all(-0.001 <= c <= 1.001 for c in p[:2]):
                raise ValueError(f"座標要 0–1 正規化：{p}")
        return v

    @model_validator(mode="after")
    def _arity(self):
        n = len(self.points)
        if self.type == "rect" and n != 2:
            raise ValueError("rect 要兩點（左上、右下）")
        if self.type == "polygon" and n < 3:
            raise ValueError("polygon 至少 3 點")
        if self.type == "keypoints" and (self.bbox is None or len(self.bbox) != 2):
            raise ValueError("keypoints 要有 bbox（左上、右下）")
        return self

    def dump(self) -> dict:
        return self.model_dump(by_alias=True, exclude_none=True)


def validate_shapes(shapes: list[dict], class_names: list[str]) -> list[dict]:
    """驗證並正規化；類別必須在圖片集的類別清單裡。"""
    out = []
    for i, s in enumerate(shapes):
        sh = Shape.model_validate(s)
        if class_names and sh.cls not in class_names:
            raise ValueError(f"第 {i} 個 shape 的類別 {sh.cls!r} 不在圖片集類別 {class_names} 裡")
        out.append(sh.dump())
    return out


def _clip(v: float) -> float:
    return min(1.0, max(0.0, float(v)))


def _rect_of(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


LABEL_TASKS = ("detect", "segment", "pose")


def conversion_counts(shapes: list[dict], class_names: list[str], task: str) -> dict[str, int]:
    """匯出成 task 格式時，形狀會怎麼變（給前端提醒用）。只算類別在清單裡的。"""
    out = {"polygon_to_box": 0, "keypoints_to_box": 0, "box_to_polygon": 0, "dropped": 0}
    for s in shapes:
        if s.get("class") not in class_names:
            continue
        t = s["type"]
        if task == "detect":
            if t == "polygon":
                out["polygon_to_box"] += 1
            elif t == "keypoints":
                out["keypoints_to_box"] += 1
        elif task == "segment":
            if t == "rect":
                out["box_to_polygon"] += 1
            elif t != "polygon":
                out["dropped"] += 1
        elif t != "keypoints":
            out["dropped"] += 1
    return out


def to_yolo_lines(shapes: list[dict], class_names: list[str], task: str) -> list[str]:
    """shapes → YOLO 標籤行。task 決定輸出格式（detect 時 polygon 取外接框；segment 時 rect 轉四點）。"""
    idx = {n: i for i, n in enumerate(class_names)}
    lines = []
    for s in shapes:
        c = idx.get(s["class"])
        if c is None:
            continue
        if task == "detect":
            x1, y1, x2, y2 = _rect_of(s.get("bbox") or s["points"]) if s["type"] == "keypoints" else _rect_of(s["points"])
            lines.append(f"{c} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
        elif task == "segment":
            if s["type"] == "rect":
                x1, y1, x2, y2 = s["points"][0] + s["points"][1]
                pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            elif s["type"] == "polygon":
                pts = s["points"]
            else:
                continue
            lines.append(f"{c} " + " ".join(f"{_clip(x):.6f} {_clip(y):.6f}" for x, y in (p[:2] for p in pts)))
        elif task == "pose":
            if s["type"] != "keypoints":
                continue
            x1, y1, x2, y2 = _rect_of(s["bbox"])
            kp = " ".join(f"{_clip(p[0]):.6f} {_clip(p[1]):.6f} {int(p[2]) if len(p) > 2 else 2}" for p in s["points"])
            lines.append(f"{c} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f} {kp}")
    return lines


def from_yolo_lines(lines: list[str], class_names: list[str], task: str,
                    kpt_shape: list[int] | None = None) -> list[dict]:
    """YOLO 標籤行 → shapes（匯入帶標注的 zip 用）。依每行欄位數判斷框 / 多邊形 / keypoints。"""
    out = []
    nk = (kpt_shape or [0, 3])
    for ln in lines:
        parts = ln.split()
        if not parts:
            continue
        c = int(float(parts[0]))
        if c >= len(class_names):
            raise ValueError(f"類別編號 {c} 超出類別清單（{len(class_names)} 類）")
        vals = [float(v) for v in parts[1:]]
        name = class_names[c]
        if task == "pose" and nk[0] and len(vals) == 4 + nk[0] * nk[1]:
            cx, cy, w, h = vals[:4]
            kp = vals[4:]
            pts = [kp[i:i + nk[1]] + ([2] if nk[1] == 2 else []) for i in range(0, len(kp), nk[1])]
            out.append({"type": "keypoints", "class": name,
                        "bbox": [[_clip(cx - w / 2), _clip(cy - h / 2)], [_clip(cx + w / 2), _clip(cy + h / 2)]],
                        "points": [[_clip(p[0]), _clip(p[1]), p[2]] for p in pts], "source": {"kind": "human"}})
        elif len(vals) == 4:
            cx, cy, w, h = vals
            out.append({"type": "rect", "class": name,
                        "points": [[_clip(cx - w / 2), _clip(cy - h / 2)], [_clip(cx + w / 2), _clip(cy + h / 2)]],
                        "source": {"kind": "human"}})
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            out.append({"type": "polygon", "class": name,
                        "points": [[_clip(vals[i]), _clip(vals[i + 1])] for i in range(0, len(vals), 2)],
                        "source": {"kind": "human"}})
        else:
            raise ValueError(f"看不懂的標籤行：{ln[:80]}")
    return out


def merge_by_class(existing: list[dict], new: list[dict], class_filter: list[str] | None) -> list[dict]:
    """自動標注寫回：class_filter 有給 → 只換這些類別，其他保留；沒給 → 整張換掉。"""
    if not class_filter:
        return list(new)
    keep = [s for s in existing if s.get("class") not in class_filter]
    return keep + [s for s in new if s.get("class") in class_filter]
