"""轉檔一致性比對（純邏輯，不需 GPU）。

同一批影像，PyTorch 的偵測結果 vs ONNX / TensorRT（透過 Triton）的偵測結果：
    ・同類別、IoU ≥ iou_match 視為配對
    ・配對比例 ≥ min_match_ratio
    ・配對框的信心度差 ≤ conf_tolerance
三者都過才算一致。這是「轉檔後精度損失沒人發現」的防線。

偵測表示法（兩邊統一）：{"class_id": int, "conf": float, "bbox": [x1, y1, x2, y2]}，原圖像素座標。
"""

from __future__ import annotations

from dataclasses import dataclass, field


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass(slots=True)
class ParityResult:
    passed: bool
    n_reference: int
    n_candidate: int
    n_matched: int
    match_ratio: float
    max_conf_diff: float
    mean_iou: float
    per_image: list[dict] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "passed": self.passed, "n_reference": self.n_reference, "n_candidate": self.n_candidate,
            "n_matched": self.n_matched, "match_ratio": round(self.match_ratio, 4),
            "max_conf_diff": round(self.max_conf_diff, 4), "mean_iou": round(self.mean_iou, 4),
            "reason": self.reason, "per_image": self.per_image,
        }


def match_detections(ref: list[dict], cand: list[dict], *, iou_match: float) -> list[tuple[int, int, float]]:
    """貪婪配對：依 IoU 高→低，同類別、未被用過的才配。回傳 (ref_idx, cand_idx, iou)。"""
    pairs = []
    for i, r in enumerate(ref):
        for j, c in enumerate(cand):
            if r["class_id"] != c["class_id"]:
                continue
            v = iou(r["bbox"], c["bbox"])
            if v >= iou_match:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    used_r, used_c, out = set(), set(), []
    for v, i, j in pairs:
        if i in used_r or j in used_c:
            continue
        used_r.add(i); used_c.add(j)
        out.append((i, j, v))
    return out


def compare(
    reference: dict[str, list[dict]], candidate: dict[str, list[dict]], *,
    iou_match: float = 0.8, min_match_ratio: float = 0.9, conf_tolerance: float = 0.05,
) -> ParityResult:
    """reference / candidate：{影像名: [偵測...]}。"""
    n_ref = n_cand = n_matched = 0
    max_conf = 0.0
    ious: list[float] = []
    per_image = []
    for name, ref in reference.items():
        cand = candidate.get(name, [])
        pairs = match_detections(ref, cand, iou_match=iou_match)
        conf_diffs = [abs(ref[i]["conf"] - cand[j]["conf"]) for i, j, _ in pairs]
        n_ref += len(ref); n_cand += len(cand); n_matched += len(pairs)
        ious.extend(v for _, _, v in pairs)
        if conf_diffs:
            max_conf = max(max_conf, max(conf_diffs))
        per_image.append({"image": name, "ref": len(ref), "cand": len(cand), "matched": len(pairs),
                          "max_conf_diff": round(max(conf_diffs), 4) if conf_diffs else 0.0})

    # 兩邊都空：沒有可比對的框，視為通過但註記
    if n_ref == 0 and n_cand == 0:
        return ParityResult(True, 0, 0, 0, 1.0, 0.0, 0.0, per_image, "兩邊皆無偵測，無法比對（視為通過）")

    denom = max(n_ref, n_cand)
    ratio = n_matched / denom if denom else 1.0
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    reasons = []
    if ratio < min_match_ratio:
        reasons.append(f"配對比例 {ratio:.3f} < {min_match_ratio}")
    if max_conf > conf_tolerance:
        reasons.append(f"信心度差 {max_conf:.3f} > {conf_tolerance}")
    return ParityResult(not reasons, n_ref, n_cand, n_matched, ratio, max_conf, mean_iou,
                        per_image, "; ".join(reasons) or "一致")
