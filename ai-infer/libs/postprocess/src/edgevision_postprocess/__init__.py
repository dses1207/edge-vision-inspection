from edgevision_postprocess.letterbox import compute_letterbox, compute_letterbox_roi
from edgevision_postprocess.render import draw_detections
from edgevision_postprocess.yolo_pose import YoloPosePostprocessor
from edgevision_postprocess.yolo_seg import YoloSegPostprocessor

__all__ = ["YoloPosePostprocessor", "YoloSegPostprocessor", "compute_letterbox",
    "compute_letterbox_roi", "draw_detections"]


def make_postprocessor(task: str, class_names: list[str], **kw):
    """依 station.yaml 的 model.task 挑後處理：segment → YOLO-seg；pose → YOLO-pose；detect → 純框。"""
    if task == "segment":
        return YoloSegPostprocessor(class_names, **kw)
    if task == "pose":
        return YoloPosePostprocessor(class_names, **kw)
    if task == "detect":
        return YoloPosePostprocessor(class_names, n_kpt=0, **kw)
    raise ValueError(f"不支援的 task: {task!r}")
