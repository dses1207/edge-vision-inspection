import numpy as np
from edgevision_postprocess import YoloPosePostprocessor, compute_letterbox, make_postprocessor

CLASSES = ["box", "mat", "part"]
N_KPT, N_ANCHORS = 4, 8400
FEATURES = 4 + len(CLASSES) + N_KPT * 3     # 19


def make_out0():
    out0 = np.zeros((FEATURES, N_ANCHORS), np.float32)
    # anchor 0：pad，框 (320,320,200,120)，四角 keypoint 在 letterbox 座標
    out0[:4, 0] = (320, 320, 200, 120)
    out0[5, 0] = 0.9                                   # pad
    kp = [(220, 260, 0.95), (420, 260, 0.9), (420, 380, 0.85), (220, 380, 0.8)]
    for i, (x, y, c) in enumerate(kp):
        out0[7 + i * 3: 7 + i * 3 + 3, 0] = (x, y, c)
    # anchor 1：part，無 keypoint 意義（自動補的角），低分類別其餘
    out0[:4, 1] = (100, 100, 40, 40)
    out0[6, 1] = 0.8                                   # part
    return out0


def test_pose_keypoints_mapped_to_original():
    pp = YoloPosePostprocessor(CLASSES, n_kpt=N_KPT)
    lb = compute_letterbox(1280, 720, 640)            # scale 0.5, pad_y 140
    dets = pp(make_out0(), None, lb, 1280, 720)
    by = {d.class_name: d for d in dets}
    assert set(by) == {"mat", "part"}
    pad = by["mat"]
    assert len(pad.keypoints) == 4
    x, y, c = pad.keypoints[0]
    assert abs(x - 440) < 1e-6 and abs(y - 240) < 1e-6 and abs(c - 0.95) < 1e-6   # (220-0)/0.5, (260-140)/0.5
    assert by["part"].keypoints is not None and len(by["part"].keypoints) == 4


def test_factory_by_task():
    assert isinstance(make_postprocessor("pose", CLASSES), YoloPosePostprocessor)
    assert make_postprocessor("detect", CLASSES).n_kpt == 0
    assert type(make_postprocessor("segment", ["board", "hand"])).__name__ == "YoloSegPostprocessor"
