"""把 YOLO-seg 的 mask 合成塞進 ONNX：Triton 直接輸出每個偵測的 160×160 mask。

⭐ 為什麼

    seg 模型原本輸出 32 張 160×160 的 prototypes（3.28 MB/幀），mask 由 runner 在
    CPU 算「係數 × prototypes」。正式機實測
    7 路就要搬 463 MB/s，Triton 的 CPU 有 71~78% 花在把這些搬回主機再走 gRPC ——
    那是整台機器的單一最大瓶頸，不是 GPU 推論。

    合成放進模型後 Triton 只輸出「每個偵測一張 mask」：
      7 路傳輸 463 → 182 MB/s（省 61%），Triton CPU 預估 71~78% → 45~50%。

⭐ 怎麼做

    Ultralytics `nms=True` 匯出已把 NMS 放進 graph：output0 變成 (B, 300, 38) ——
    每列 [x1,y1,x2,y2, score, class, 32 個係數]，padding 到 300 列（score=0）。
    但 prototypes 還是整張輸出。這裡在 graph 尾端接一小段：

        coeffs = output0[:, :, 6:38]                 (B, 300, 32)
        proto  = output1.reshape(B, 32, 25600)      (B, 32, 25600)
        masks  = sigmoid(coeffs @ proto)            (B, 300, 25600) → (B, 300, 160, 160)

    公式與 ultralytics.utils.ops.process_mask 完全相同（它也是 masks_in @ protos.view(c,-1)）。
    然後把 output1 從 graph 輸出拿掉 —— 不然還是會搬。

⚠️ 不裁切、不二值化、保留 sigmoid

    官方 process_mask 會 crop 到 bbox 再 gt(0)。這裡刻意不做：
    後處理的 ROI 路徑（yolo_seg._roi_mask）要的是「框附近帶 margin 的 float 機率」，
    它自己會放大、閾值、找輪廓。在 graph 裡先裁掉就沒有 margin 可用，邊界會差幾個像素。
    面積語意「原圖解析度前景像素數、洞不算」由後處理維持，跟舊格式一樣。

⚠️ 300 × 25600 × 4 bytes = 30.7 MB/幀？

    不是。Triton 輸出的是 dense tensor 沒錯，但 nvinferserver 走 gRPC 時
    ⋯⋯ 其實就是會搬 30 MB。所以 max_det 不能留 300。
    一幀最多十來個物件時，max_det=16 綽綽有餘：16 × 0.1 MB = 1.6 MB/幀，
    比原本 4.56 MB 還小。max_det 是匯出參數，在這裡強制檢查。

用法
    python -m edgevision.training.seg_fuse in.onnx out.onnx --imgsz 640   # 接子圖
    python -m edgevision.training.seg_fuse in.onnx out.onnx --check # 接完用 onnxruntime 跟原始算法比一次
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

N_MASK = 32
MASKS_OUTPUT = "masks"          # 新輸出的名字；runner 的 OUTPUT_LAYERS 與 config.pbtxt 要對上
MAX_DET_LIMIT = 64              # 超過就是忘了設 max_det，傳輸量會爆


def fuse(src: Path, dst: Path, *, imgsz: int) -> dict:
    """接子圖。回傳 {"max_det": int, "proto_hw": int, "outputs": [...]} 供上層記 metadata。

    imgsz 必給：dynamic=True 匯出的 graph 連輸入 H/W 都是符號維度，
    prototype 尺寸（imgsz/4）從 graph 裡讀不到，只有呼叫端知道。
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(src))
    g = model.graph
    outs = {o.name: o for o in g.output}
    if "output0" not in outs or "output1" not in outs:
        raise SystemExit(f"不是 YOLO-seg 匯出（輸出有 {sorted(outs)}，要 output0 + output1）")

    def dims(vi):
        return [d.dim_value if d.dim_value else -1 for d in vi.type.tensor_type.shape.dim]

    o0, o1 = dims(outs["output0"]), dims(outs["output1"])
    # nms=True 的 output0 是 (B, max_det, 4+2+32)；沒 nms 的是 (B, 38, 8400) —— 不能接
    if len(o0) != 3 or o0[2] != 4 + 2 + N_MASK:
        raise SystemExit(f"output0 shape {o0} 不是 nms=True 匯出的 (B, max_det, 38)。"
                         f"要先用 nms=True 匯出（見 export.py 的 ExportSpec.fuse_masks）")
    max_det = o0[1]
    if max_det <= 0 or max_det > MAX_DET_LIMIT:
        raise SystemExit(f"max_det={max_det} 超過 {MAX_DET_LIMIT}：每偵測一張 160×160 mask，"
                         f"300 列就是 30 MB/幀，比原本還多。匯出時設 max_det=16")
    _, c, mh, mw = o1
    if c != N_MASK:
        raise SystemExit(f"output1 通道數 {c} ≠ {N_MASK}")
    # prototype 尺寸 = imgsz / 4（訓練時 mask_ratio=4；後處理 _build_masks 也這樣假設）。
    # dynamic=True 匯出時 graph 裡的 H/W 是符號維度讀不到，所以以 imgsz 為準；
    # graph 若有具體值就必須一致，不一致代表 imgsz 給錯。
    want = imgsz // 4
    if (mh > 0 and mh != want) or (mw > 0 and mw != want):
        raise SystemExit(f"output1 尺寸 {mh}×{mw} ≠ imgsz/4={want}：imgsz={imgsz} 給錯了")
    mh = mw = want

    # ── 子圖 ──────────────────────────────────────────────
    # coeffs = output0[:, :, 6:38]
    g.initializer.extend([
        numpy_helper.from_array(np.array([6], np.int64), "fuse_start"),
        numpy_helper.from_array(np.array([6 + N_MASK], np.int64), "fuse_end"),
        numpy_helper.from_array(np.array([2], np.int64), "fuse_axis"),
        # proto (B, 32, H, W) → (B, 32, H*W)；0 = 沿用該維
        numpy_helper.from_array(np.array([0, N_MASK, -1], np.int64), "fuse_proto_shape"),
        # (B, max_det, H*W) → (B, max_det, H, W)
        numpy_helper.from_array(np.array([0, max_det, mh, mw], np.int64), "fuse_mask_shape"),
    ])
    g.node.extend([
        helper.make_node("Slice", ["output0", "fuse_start", "fuse_end", "fuse_axis"], ["fuse_coeffs"], name="fuse/slice"),
        helper.make_node("Reshape", ["output1", "fuse_proto_shape"], ["fuse_proto_flat"], name="fuse/reshape_proto"),
        helper.make_node("MatMul", ["fuse_coeffs", "fuse_proto_flat"], ["fuse_logits"], name="fuse/matmul"),
        helper.make_node("Sigmoid", ["fuse_logits"], ["fuse_prob"], name="fuse/sigmoid"),
        helper.make_node("Reshape", ["fuse_prob", "fuse_mask_shape"], [MASKS_OUTPUT], name="fuse/reshape_masks"),
    ])

    # 輸出：留 output0、加 masks、拿掉 output1（不拿掉就白做了）
    batch_dim = outs["output0"].type.tensor_type.shape.dim[0]
    masks_vi = helper.make_tensor_value_info(MASKS_OUTPUT, TensorProto.FLOAT, [None, max_det, mh, mw])
    masks_vi.type.tensor_type.shape.dim[0].CopyFrom(batch_dim)     # 保留 dynamic batch 的名字
    g.output.remove(outs["output1"])
    g.output.append(masks_vi)

    onnx.checker.check_model(model)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst))
    info = {"max_det": max_det, "proto_hw": mh, "outputs": [o.name for o in g.output]}
    logger.info("已接 mask 合成：%s → %s（max_det=%d，輸出 %s）", src.name, dst.name, max_det, info["outputs"])
    return info


def check(src: Path, dst: Path, *, n: int = 2) -> float:
    """接完驗證：同一張隨機輸入，融合模型的 masks 要等於「原模型 output0/output1 自己算」。回最大絕對誤差。"""
    import onnxruntime as ort

    a = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(n):
        x = rng.random((1, 3, 640, 640), dtype=np.float32)
        o0, o1 = a.run(None, {"images": x})
        f0, fm = b.run(None, {"images": x})
        assert np.array_equal(o0, f0), "融合後 output0 變了 —— 不該動它"
        coeffs = o0[0, :, 6:6 + N_MASK]
        ref = 1.0 / (1.0 + np.exp(-(coeffs @ o1[0].reshape(N_MASK, -1))))
        ref = ref.reshape(-1, o1.shape[2], o1.shape[3])
        worst = max(worst, float(np.abs(ref - fm[0]).max()))
    return worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--imgsz", type=int, default=640, help="匯出時的 imgsz（prototype = imgsz/4）")
    ap.add_argument("--check", action="store_true", help="接完用 onnxruntime 跟原始算法比一次")
    args = ap.parse_args()
    logging.basicConfig(level="INFO", format="%(levelname)-7s %(message)s")
    info = fuse(args.src, args.dst, imgsz=args.imgsz)
    print(info)
    if args.check:
        err = check(args.src, args.dst)
        print(f"最大絕對誤差 {err:.2e}", "✓" if err < 1e-4 else "✗ 超過 1e-4")
        return 0 if err < 1e-4 else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
