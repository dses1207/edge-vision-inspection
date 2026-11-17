"""runner 調和器：線路表 → 該有哪些 runner、哪些要建 / 換 / 收。"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgevision.orchestrator.runner_reconciler import (  # noqa: E402
    ANN_HASH, ANN_VERSION, Reconciler, desired_runners, parse_runner_images, plan, runner_deployment,
)

CAM = [{"id": "c1", "enabled": True}]


def _lines(*items):
    metas = [{"id": i, "enabled": e} for i, e, _ in items]
    details = {i: {"config_version": v, "model": {"name": "m"}, "cameras": CAM} for i, _, v in items}
    return metas, details


def _desired(metas, details, image="img:1"):
    return desired_runners(metas, details, image=image, gpu_claim="g", namespace="edgevision")


def test_version_lives_on_pod_template_so_k8s_restarts():
    d = runner_deployment("alpha_a", "seg_alpha", image="img", gpu_claim="g", namespace="edgevision", config_version=7)
    assert d["spec"]["template"]["metadata"]["annotations"][ANN_VERSION] == "7"
    env = {e["name"] for e in d["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert "ENCODER" not in env                         # 預設交給 runner 自己判斷


def test_plan_create_update_delete_noop():
    metas, details = _lines(("a", True, 1), ("b", True, 1), ("c", False, 1))
    want = _desired(metas, details)
    assert set(want) == {"runner-a", "runner-b"}        # 停用的 c 不要
    h = {n: d["metadata"]["annotations"][ANN_HASH] for n, d in want.items()}
    existing = {"runner-a": h["runner-a"], "runner-c": "x"}
    kinds = {(a.kind, a.name) for a in plan(want, existing)}
    assert kinds == {("create", "runner-b"), ("delete", "runner-c")}   # a 沒變不動


def test_version_or_image_change_means_update():
    metas, details = _lines(("a", True, 1))
    old = _desired(metas, details)["runner-a"]["metadata"]["annotations"][ANN_HASH]
    _, details2 = _lines(("a", True, 2))
    assert [x.kind for x in plan(_desired(metas, details2), {"runner-a": old})] == ["update"]
    assert [x.kind for x in plan(_desired(metas, details, image="img:2"), {"runner-a": old})] == ["update"]


def test_per_line_image_only_restarts_that_line():
    """image.<線路 id> 只換那條線；其他線的 spec-hash 不變 → 不重啟。"""
    metas, details = _lines(("a", True, 1), ("b", True, 1))
    base = {n: d["metadata"]["annotations"][ANN_HASH] for n, d in _desired(metas, details).items()}
    default, per_line = parse_runner_images({"image": "img:1", "image.b": "img:2", "image.c": " "})
    assert (default, per_line) == ("img:1", {"b": "img:2"})
    want = desired_runners(metas, details, image=default, gpu_claim="g", namespace="edgevision", line_images=per_line)
    assert want["runner-b"]["spec"]["template"]["spec"]["containers"][0]["image"] == "img:2"
    assert [(x.kind, x.name) for x in plan(want, base)] == [("update", "runner-b")]
    with pytest.raises(RuntimeError):
        parse_runner_images({"image.b": "img:2"})                  # 沒有預設：整輪放棄，不亂動


def test_line_without_model_or_cameras_is_skipped():
    metas = [{"id": "a", "enabled": True}, {"id": "b", "enabled": True}]
    details = {"a": {"config_version": 1, "model": None, "cameras": CAM},
               "b": {"config_version": 1, "model": {"name": "m"}, "cameras": []}}
    assert _desired(metas, details) == {}


def test_unreadable_source_does_nothing():
    class Core:
        def read_namespaced_config_map(self, *a):
            raise RuntimeError("API 掛了")

    class Apps:
        def list_namespaced_deployment(self, *a, **k):
            raise AssertionError("不該走到這裡")

        def delete_namespaced_deployment(self, *a):
            raise AssertionError("讀不到設定時絕不能收 runner")

    r = Reconciler(config_api="http://x", namespace="edgevision", gpu_claim="g", apps_api=Apps(), core_api=Core())
    assert r.once() == []


def test_model_promote_restarts_runner_and_unknown_version_is_held():
    from edgevision.orchestrator.runner_reconciler import ANN_MODELS, model_versions
    metas = [{"id": "a", "enabled": True}, {"id": "b", "enabled": True}]
    v1 = {"a": {"config_version": 1, "model": {"name": "m", "version": 1}, "cameras": CAM,
                "extra_stations": [{"station": "ppe_check", "model": {"name": "ppe", "version": 1}}]},
          "b": {"config_version": 1, "model": {"name": "n", "version": 4}, "cameras": CAM}}
    assert model_versions(v1["a"]) == "m:1,ppe:1"
    want = desired_runners(metas, v1, image="img", gpu_claim="g", namespace="edgevision", hold=set())
    assert want["runner-a"]["spec"]["template"]["metadata"]["annotations"][ANN_MODELS] == "m:1,ppe:1"
    existing = {n: d["metadata"]["annotations"][ANN_HASH] for n, d in want.items()}

    # ppe promote 到 v2 → 只有用到它的 a 要換（類別、imgsz 以模型為準，runner 啟動時才讀）
    v2 = {**v1, "a": {**v1["a"], "extra_stations": [{"station": "ppe_check", "model": {"name": "ppe", "version": 2}}]}}
    hold: set[str] = set()
    acts = plan(desired_runners(metas, v2, image="img", gpu_claim="g", namespace="edgevision", hold=hold), existing,
                frozenset(hold))
    assert [(x.kind, x.name) for x in acts] == [("update", "runner-a")]

    # Registry 一時查不到 b 的版本 → b 原樣不動（不換、不收），a 照常
    v3 = {**v1, "b": {**v1["b"], "model": {"name": "n"}}}
    hold = set()
    acts = plan(desired_runners(metas, v3, image="img", gpu_claim="g", namespace="edgevision", hold=hold), existing,
                frozenset(hold))
    assert hold == {"runner-b"} and acts == []
    assert plan({}, {"runner-b": "x"}, frozenset({"runner-b"})) == []
