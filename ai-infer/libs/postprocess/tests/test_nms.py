"""NMS 要每類別各自做（Ultralytics 預設）：不同類別重疊不互砍。"""
import numpy as np
from edgevision_postprocess import YoloPosePostprocessor, compute_letterbox
from edgevision_postprocess.nms import nms_keep


def test_overlapping_boxes_of_different_classes_both_survive():
    boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105]], float)
    scores = np.array([0.9, 0.8])
    assert sorted(nms_keep(boxes, scores, np.array([0, 1]), 0.5, 0.45).tolist()) == [0, 1]   # hand 疊 board
    assert nms_keep(boxes, scores, np.array([0, 0]), 0.5, 0.45).tolist() == [0]              # 同類別才互砍
    assert nms_keep(boxes, scores, np.array([0, 1]), 0.5, 0.45, agnostic=True).tolist() == [0]


def test_detect_postprocessor_keeps_hand_on_board():
    pp = YoloPosePostprocessor(["board", "hand"], n_kpt=0)
    out0 = np.zeros((4 + 2, 8400), np.float32)
    out0[:4, 0] = (320, 320, 200, 120); out0[4, 0] = 0.95          # board
    out0[:4, 1] = (325, 322, 190, 110); out0[5, 1] = 0.90          # hand，幾乎同一個位置
    dets = pp(out0, None, compute_letterbox(1280, 720, 640), 1280, 720)
    assert sorted(d.class_name for d in dets) == ["board", "hand"]
