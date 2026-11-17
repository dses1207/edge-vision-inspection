"""ClearML 實作的模型 Registry。

分工：
    ・檔案      → MinIO models/<name>/v<N>/...（由本模組用 ObjectStore 上傳）
    ・身分/血緣 → ClearML Model（metadata 欄位存完整 ModelMetadata JSON）
    ・在線狀態  → ClearML tag（production / candidate / rejected）

為什麼不用 ClearML 的 OutputModel 自動上傳：
    自動上傳走 ClearML SDK 的 S3 設定，路徑由它決定；我們要的是
    可預測的 models/<name>/v<N>/ 分層（部署 Controller、備份、人眼都靠它）。
    所以檔案自己傳，ClearML 只登錄 URI（InputModel.import_model，官方確認）。

版本號：ClearML 模型沒有整數版本，我們自己編（metadata.version），
        查最大值 + 1。同名同版不允許重複註冊。

連線：全走環境變數 CLEARML_API_HOST / CLEARML_WEB_HOST / CLEARML_FILES_HOST /
      CLEARML_API_ACCESS_KEY / CLEARML_API_SECRET_KEY（Compose 與 K8s 同一份程式）。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from libs.contracts import ModelMetadata
from libs.contracts.model_meta import TAG_CANDIDATE, TAG_PRODUCTION, TAG_REJECTED

from edgevision.registry.base import (
    ARTIFACT_FILENAMES,
    MODELS_BUCKET,
    ResolvedModel,
    VersionInfo,
    artifact_key,
    artifact_uri,
    serving_uri,
)

logger = logging.getLogger(__name__)

META_KEY = "metadata_json"      # 完整 ModelMetadata JSON
VERSION_KEY = "version"         # 整數版本（字串存）
MANAGED_TAGS = {TAG_PRODUCTION, TAG_CANDIDATE, TAG_REJECTED}


def _add_tag(model, tag: str) -> None:
    """ClearML 的 Model（query 回來的）沒有 add_tags，只有 tags setter；OutputModel 才有 add_tags。"""
    if hasattr(model, "add_tags"):
        model.add_tags([tag])
    else:
        model.tags = sorted(set(model.tags or []) | {tag})


def _remove_tag(model, tag: str) -> None:
    if hasattr(model, "remove_tags"):
        model.remove_tags([tag])
    else:
        model.tags = [t for t in (model.tags or []) if t != tag]


class ClearMLRegistry:
    def __init__(self, *, objects, project: str | None = None) -> None:
        try:
            from clearml import Model  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("需要 clearml：pip install clearml") from exc
        self.objects = objects
        self.project = project or os.environ.get("CLEARML_PROJECT", "edgevision")

    # ── 查詢 ─────────────────────────────────────────────

    def _query(self, model_name: str | None = None, *, tags: list[str] | None = None,
               max_results: int = 500):
        from clearml import Model

        kw = dict(project_name=self.project, only_published=False,
                  include_archived=False, max_results=max_results)
        if model_name:
            kw["model_name"] = model_name
        if tags:
            kw["tags"] = tags
        try:
            return list(Model.query_models(**kw) or [])
        except Exception as exc:  # 專案不存在時 ClearML 會丟例外，視為空
            logger.debug("query_models 空結果：%s", exc)
            return []

    @staticmethod
    def _meta_of(model) -> ModelMetadata | None:
        raw = model.get_metadata(META_KEY)
        if not raw:
            return None
        try:
            return ModelMetadata.model_validate_json(raw)
        except Exception as exc:
            logger.warning("模型 %s 的 metadata 解析失敗：%s", model.id, exc)
            return None

    def _find(self, model_name: str, version: int):
        # ClearML 的 model_name 是模糊比對，這裡再用 metadata 精確過濾
        for m in self._query(model_name):
            meta = self._meta_of(m)
            if meta and meta.model_name == model_name and meta.version == version:
                return m, meta
        return None, None

    def _to_resolved(self, model, meta: ModelMetadata) -> ResolvedModel:
        return ResolvedModel(
            model_name=meta.model_name, version=meta.version,
            source_uri=serving_uri(meta) if (meta.artifacts.tensorrt or meta.artifacts.onnx)
            else (meta.artifacts.pytorch or ""),
            meta=meta, registry_id=model.id,
        )

    def next_version(self, model_name: str) -> int:
        versions = [v.version for v in self.list_versions(model_name)]
        return (max(versions) + 1) if versions else 1

    def list_models(self) -> list[str]:
        names: set[str] = set()
        for m in self._query():
            meta = self._meta_of(m)
            if meta:
                names.add(meta.model_name)
        return sorted(names)

    def list_versions(self, model_name: str) -> list[VersionInfo]:
        out: list[VersionInfo] = []
        for m in self._query(model_name):
            meta = self._meta_of(m)
            if not meta or meta.model_name != model_name:
                continue
            out.append(VersionInfo(
                version=meta.version, tags=list(m.tags or []), registry_id=m.id,
                created_at=meta.created_at, metrics=meta.metrics,
            ))
        return sorted(out, key=lambda v: v.version)

    def get(self, model_name: str, version: int) -> ResolvedModel | None:
        m, meta = self._find(model_name, version)
        return self._to_resolved(m, meta) if m else None

    def resolve(self, model_name: str, *, tag: str = TAG_PRODUCTION) -> ResolvedModel | None:
        candidates = []
        for m in self._query(model_name, tags=[tag]):
            meta = self._meta_of(m)
            if meta and meta.model_name == model_name and tag in (m.tags or []):
                candidates.append((meta.version, m, meta))
        if not candidates:
            return None
        # 同一 tag 理論上只在一版；若殘留多版，取最新版並警告
        candidates.sort(key=lambda c: c[0], reverse=True)
        if len(candidates) > 1:
            logger.warning("%s 有 %d 個版本帶 tag=%s，取 v%d",
                           model_name, len(candidates), tag, candidates[0][0])
        _, m, meta = candidates[0]
        return self._to_resolved(m, meta)

    # ── 寫入 ─────────────────────────────────────────────

    def _upload(self, meta: ModelMetadata, files: dict[str, Path]) -> dict[str, str]:
        uris: dict[str, str] = {}
        for kind, path in files.items():
            if kind not in ARTIFACT_FILENAMES:
                raise ValueError(f"未知的 artifact 種類：{kind}")
            path = Path(path)
            key = artifact_key(meta.model_name, meta.version, kind)
            with path.open("rb") as fh:
                self.objects.put(bucket=MODELS_BUCKET, key=key, data=fh,
                                 length=path.stat().st_size)
            uris[kind] = artifact_uri(meta.model_name, meta.version, kind)
            logger.info("已上傳 %s → %s", path.name, uris[kind])
        return uris

    def register(self, meta: ModelMetadata, files: dict[str, Path], *,
                 tags: list[str] | None = None) -> ResolvedModel:
        from clearml import InputModel

        if self._find(meta.model_name, meta.version)[0] is not None:
            raise ValueError(f"{meta.model_name} v{meta.version} 已存在，不可重複註冊")

        uris = self._upload(meta, files)
        for kind, uri in uris.items():
            setattr(meta.artifacts, kind, uri)

        # metadata.json 也放進 MinIO（人眼可讀、備份可還原）
        meta_bytes = meta.model_dump_json(indent=2).encode()
        import io
        key = artifact_key(meta.model_name, meta.version, "metadata")
        self.objects.put(bucket=MODELS_BUCKET, key=key, data=io.BytesIO(meta_bytes),
                         length=len(meta_bytes), content_type="application/json")
        meta.artifacts.metadata = artifact_uri(meta.model_name, meta.version, "metadata")

        # ClearML 登錄：weights_url 指向主要檔（優先 pytorch，供 fine-tune 起手）
        weights_url = meta.artifacts.pytorch or serving_uri(meta)
        model = InputModel.import_model(
            weights_url=weights_url,
            name=meta.model_name,
            project=self.project,
            label_enumeration={n: i for i, n in enumerate(meta.class_names)},
            framework="PyTorch",
            tags=[f"v{meta.version}", *(tags or [])],
            comment=meta.notes or f"{meta.model_name} v{meta.version}",
        )
        self._write_meta(model, meta)
        model.publish()
        logger.info("已註冊 %s v%d（ClearML id=%s，tags=%s）",
                    meta.model_name, meta.version, model.id, tags)
        return ResolvedModel(model_name=meta.model_name, version=meta.version,
                             source_uri=weights_url, meta=meta, registry_id=model.id)

    def _write_meta(self, model, meta: ModelMetadata) -> None:
        model.set_all_metadata({
            META_KEY: {"type": "str", "value": meta.model_dump_json()},
            VERSION_KEY: {"type": "int", "value": str(meta.version)},
            "dataset_version": {"type": "str", "value": meta.lineage.dataset_version},
            "map50_box": {"type": "float", "value": str(meta.metrics.get("map50_box", ""))},
        })

    def update_artifacts(self, model_name: str, version: int, **uris: str) -> ResolvedModel:
        m, meta = self._find(model_name, version)
        if m is None:
            raise KeyError(f"{model_name} v{version} 不存在")
        for kind, uri in uris.items():
            setattr(meta.artifacts, kind, uri)
        self._write_meta(m, meta)
        return self._to_resolved(m, meta)

    def update_metrics(self, model_name: str, version: int, metrics: dict) -> ResolvedModel:
        m, meta = self._find(model_name, version)
        if m is None:
            raise KeyError(f"{model_name} v{version} 不存在")
        meta.metrics.update(metrics)
        self._write_meta(m, meta)
        return self._to_resolved(m, meta)

    def set_tag(self, model_name: str, version: int, tag: str, *, exclusive: bool = True) -> None:
        """加 tag。exclusive=True 時同名 tag 從該模型其他版本移除。"""
        target, _ = self._find(model_name, version)
        if target is None:
            raise KeyError(f"{model_name} v{version} 不存在")
        if exclusive:
            for m in self._query(model_name, tags=[tag]):
                meta = self._meta_of(m)
                if meta and meta.model_name == model_name and m.id != target.id and tag in (m.tags or []):
                    _remove_tag(m, tag)
        if tag not in (target.tags or []):
            _add_tag(target, tag)

    def remove_tag(self, model_name: str, version: int, tag: str) -> None:
        m, _ = self._find(model_name, version)
        if m is not None and tag in (m.tags or []):
            _remove_tag(m, tag)

    def rename(self, model_name: str, version: int, new_name: str, new_version: int | None = None,
               *, note: str = "") -> ResolvedModel:
        """把一個版本搬成另一個名字（Registry 以名稱+版本為身分，所以是「複製成新的、舊的收掉」）。

        ① MinIO 伺服器端複製 <name>/v<N>/ 的檔案到新路徑  ② 用同一份 metadata 登錄新身分，tag 沿用
        ③ 新的查得到才封存舊的 ClearML 紀錄（可 unarchive 救回）並刪舊路徑的檔案。
        現役（production）版本不准改：Triton 正在用那個名字載入。
        """
        from clearml import InputModel

        old, meta = self._find(model_name, version)
        if old is None:
            raise KeyError(f"{model_name} v{version} 不存在")
        if TAG_PRODUCTION in (old.tags or []):
            raise ValueError(f"{model_name} v{version} 是現役版本，不能改名")
        new_version = new_version or self.next_version(new_name)
        if self._find(new_name, new_version)[0] is not None:
            raise ValueError(f"{new_name} v{new_version} 已存在")
        tags = [t for t in (old.tags or []) if not (t.startswith("v") and t[1:].isdigit())]

        for kind in ARTIFACT_FILENAMES:
            uri = getattr(meta.artifacts, kind)
            if kind == "metadata" or not uri:
                continue
            _, src_key = self.objects.parse_uri(uri)
            self.objects.copy(src_bucket=MODELS_BUCKET, src_key=src_key, bucket=MODELS_BUCKET,
                              key=artifact_key(new_name, new_version, kind))
            setattr(meta.artifacts, kind, artifact_uri(new_name, new_version, kind))

        meta.notes = "\n".join(x for x in (meta.notes, f"由 {model_name} v{version} 改名而來。{note}".strip()) if x)
        meta.model_name, meta.version = new_name, new_version
        import io
        meta.artifacts.metadata = artifact_uri(new_name, new_version, "metadata")
        meta_bytes = meta.model_dump_json(indent=2).encode()
        self.objects.put(bucket=MODELS_BUCKET, key=artifact_key(new_name, new_version, "metadata"),
                         data=io.BytesIO(meta_bytes), length=len(meta_bytes), content_type="application/json")

        model = InputModel.import_model(
            weights_url=meta.artifacts.pytorch or serving_uri(meta), name=new_name, project=self.project,
            label_enumeration={n: i for i, n in enumerate(meta.class_names)}, framework="PyTorch",
            tags=[f"v{new_version}", *tags], comment=meta.notes,
        )
        self._write_meta(model, meta)
        model.publish()
        if self.get(new_name, new_version) is None:
            raise RuntimeError(f"{new_name} v{new_version} 登錄後查不到，舊版本保留不動")

        old.archive()
        n = self.objects.remove_prefix(bucket=MODELS_BUCKET, prefix=f"{model_name}/v{version}/")
        logger.info("%s v%d → %s v%d（ClearML id=%s；舊紀錄已封存，刪 %d 個檔）",
                    model_name, version, new_name, new_version, model.id, n)
        return self._to_resolved(model, meta)

    def promote(self, model_name: str, version: int, *, tag: str = TAG_PRODUCTION) -> ResolvedModel:
        resolved = self.get(model_name, version)
        if resolved is None:
            raise KeyError(f"{model_name} v{version} 不存在")
        if tag == TAG_PRODUCTION:
            serving_uri(resolved.meta)   # 沒有可服務檔案就不准上線
        self.set_tag(model_name, version, tag, exclusive=True)
        if tag == TAG_PRODUCTION:
            self.remove_tag(model_name, version, TAG_CANDIDATE)
        logger.info("%s：tag=%s → v%d", model_name, tag, version)
        return self.get(model_name, version)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def describe(resolved: ResolvedModel) -> dict:
    """給 API 回傳用的摘要。"""
    m = resolved.meta
    return {
        "model_name": m.model_name, "version": m.version, "registry_id": resolved.registry_id,
        "source_uri": resolved.source_uri, "created_at": m.created_at,
        "lineage": m.lineage.model_dump(), "metrics": m.metrics,
        "artifacts": m.artifacts.model_dump(), "class_names": m.class_names,
        # 模型自己的屬性：線路設定不存，設定 API 讀這兩個附給 runner（edgevision/api/model_info.py）
        "task": m.task, "imgsz": m.imgsz,
    }


__all__ = ["ClearMLRegistry", "describe", "now_iso", "json"]
