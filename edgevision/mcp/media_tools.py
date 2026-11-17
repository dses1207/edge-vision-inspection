"""圖片庫的 MCP 工具（stdio 版 server.py 與遠端版 remote.py 共用一份定義）。

給 agent 的是「搬資料、跑預標、匯出」這些批次動作，以及審核（輪次、看一張、送出），
讓 agent 能先做第一輪粗審；最後判斷仍由人在前端做。
"""

from __future__ import annotations

from typing import Any, Callable


def _select(target: int, roi, ignore_regions, dense_interval_s: float) -> dict:
    d: dict[str, Any] = {"target": target, "dense_interval_s": dense_interval_s}
    if roi is not None:
        d["roi"] = roi
    if ignore_regions is not None:
        d["ignore_regions"] = ignore_regions
    return d


def register(mcp, call: Callable, safe: Callable, RO, WRITE, DESTRUCTIVE=None) -> None:
    DESTRUCTIVE = DESTRUCTIVE or WRITE
    @mcp.tool(title="列出圖片集", annotations=RO)
    @safe
    def list_media_collections(include_archived: bool = False, org: str | None = None) -> list[dict]:
        """圖片庫裡的圖片集：分類、類別、張數、已標完張數（labeled + background）。
        org：只列這個分類（空字串 = 只列未分類的）；有哪些分類用 list_media_orgs。"""
        params: dict[str, Any] = {"include_archived": include_archived}
        if org is not None:
            params["org"] = org
        return call("GET", "/api/media/collections", params=params)

    @mcp.tool(title="圖片集分類", annotations=RO)
    @safe
    def list_media_orgs(include_archived: bool = False) -> list[dict]:
        """用過的分類與各有幾個圖片集（org = null 是未分類）。"""
        return call("GET", "/api/media/collections/orgs", params={"include_archived": include_archived})

    @mcp.tool(title="圖片集詳情", annotations=RO)
    @safe
    def get_media_collection(collection_id: int) -> dict:
        """單一圖片集：類別、各狀態張數（unlabeled 未標 / auto AI 預標未審 / labeled 人工標完 / background 確認無物件）。"""
        return call("GET", f"/api/media/collections/{collection_id}")

    @mcp.tool(title="列出圖片集的圖", annotations=RO)
    @safe
    def list_media_items(collection_id: int, status: list[str] | None = None, after_id: int = 0,
                         limit: int = 100, batch_ids: list[int] | None = None) -> list[dict]:
        """圖片集裡的圖（分頁：用上一頁最後一個 id 當 after_id）。每張帶 batch_id（屬於哪一批）。

        Args:
            status: 只列這些狀態（unlabeled / auto / labeled / background）
            batch_ids: 只列這些批次的圖
        """
        params: dict[str, Any] = {"after_id": after_id, "limit": limit}
        if status:
            params["status"] = status
        if batch_ids:
            params["batch_id"] = batch_ids
        return call("GET", f"/api/media/collections/{collection_id}/items", params=params)

    @mcp.tool(title="建立圖片集", annotations=WRITE)
    @safe
    def create_media_collection(name: str, class_names: list[str], description: str = "",
                                kpt_shape: list[int] | None = None, line_id: str | None = None,
                                camera_id: str | None = None, org: str | None = None) -> dict:
        """建一個圖片集（一組圖 + 一套類別）。類別之後只能往後加，不能改順序。
        圖片集不綁輸出格式：框、多邊形、關鍵點都能標，匯出時（export_media_collection 的 task）才選要哪一種。

        Args:
            kpt_shape: 要標關鍵點時給 [點數, 3]，例如 [4, 3]；其他不用給
            org: 分類（自由文字，例「倉庫 A」；用過的見 list_media_orgs）
        """
        body = {"name": name, "class_names": class_names, "description": description,
                "kpt_shape": kpt_shape, "line_id": line_id, "camera_id": camera_id, "org": org}
        return call("POST", "/api/media/collections", json_body={k: v for k, v in body.items() if v is not None})

    @mcp.tool(title="修改圖片集", annotations=WRITE)
    @safe
    def update_media_collection(collection_id: int, name: str | None = None, description: str | None = None,
                                class_names: list[str] | None = None, archived: bool | None = None,
                                line_id: str | None = None, camera_id: str | None = None,
                                org: str | None = None) -> dict:
        """改圖片集名稱 / 說明 / 類別 / 分類 / 封存（archived=true 後預設清單不列）。

        類別只能【往後加】，不能改既有順序或刪除（已標的類別編號會全錯）。
        org 給空字串 = 清掉分類。
        """
        body = {"name": name, "description": description, "class_names": class_names, "archived": archived,
                "line_id": line_id, "camera_id": camera_id, "org": org}
        return call("PATCH", f"/api/media/collections/{collection_id}",
                    json_body={k: v for k, v in body.items() if v is not None})

    # ── 批次（每次把圖放進圖片集 = 一批；預標 / 建輪次 / 匯出都能按批次篩）──

    @mcp.tool(title="列出批次", annotations=RO)
    @safe
    def list_media_batches(collection_id: int) -> list[dict]:
        """圖片集的所有批次：名稱、來源（upload / zip / video / nas / copy＋參數）、匯入任務、
        各狀態張數（n_unlabeled 未標 / n_auto AI 預標未確認 / n_done 已確認）。"""
        return call("GET", f"/api/media/collections/{collection_id}/batches")

    @mcp.tool(title="改批次名稱", annotations=WRITE)
    @safe
    def update_media_batch(batch_id: int, name: str) -> dict:
        """改批次名稱（例「第二批」）。批次 = 一次匯入放進圖片集的那群圖；用 list_media_batches 查 batch_id。"""
        return call("PATCH", f"/api/media/batches/{batch_id}", json_body={"name": name})

    @mcp.tool(title="刪除批次", annotations=DESTRUCTIVE)
    @safe
    def delete_media_batch(batch_id: int) -> dict:
        """刪整個批次：這批的圖（含標注）從圖片集移除、批次刪掉；圖片庫原圖不動。
        匯入還在跑、或是進行中標注專案的初始批次 → 擋下。只拿掉幾張用 remove_media_items。"""
        return call("DELETE", f"/api/media/batches/{batch_id}")

    @mcp.tool(title="複製圖到別的圖片集", annotations=WRITE)
    @safe
    def copy_media_items(collection_id: int, item_ids: list[int], target_collection_id: int,
                         with_labels: bool = False, batch_name: str | None = None,
                         batch_id: int | None = None) -> dict:
        """把 collection_id 的這些圖複製到 target_collection_id（只多一列關聯，不複製檔案），在目標成為一個新批次
        （或用 batch_id 加進目標圖片集的既有批次）。
        目標已經有的圖會跳過（回傳 added = 實際新增張數、batch_id）。

        Args:
            item_ids: 來源圖片集的圖（用 list_media_items 取；整個圖片集就把全部 id 給進來）
            with_labels: 連已確認的標注一起帶過去（只帶目標圖片集有的類別）
            batch_name: 新批次名稱；省略 = 「複製 <時間>」
            batch_id: 加進目標圖片集的這個既有批次（list_media_batches 查）；跟 batch_name 擇一
        """
        body = {"item_ids": item_ids, "target_collection_id": target_collection_id, "with_labels": with_labels,
                "batch_name": batch_name, "batch_id": batch_id}
        return call("POST", f"/api/media/collections/{collection_id}/copy",
                    json_body={k: v for k, v in body.items() if v is not None})

    @mcp.tool(title="從圖片集移除圖", annotations=DESTRUCTIVE)
    @safe
    def remove_media_items(collection_id: int, item_ids: list[int]) -> dict:
        """從圖片集移除這些圖（只刪關聯與這個圖片集裡的標注；圖片庫的原圖不動）。"""
        return call("POST", f"/api/media/collections/{collection_id}/items/remove", json_body={"item_ids": item_ids})

    @mcp.tool(title="取得圖片庫上傳網址", annotations=WRITE)
    @safe
    def create_media_upload_url(filename: str) -> dict:
        """拿臨時上傳網址（2 小時），把 zip（圖片或 YOLO 資料集）或影片直傳物件儲存。

        流程：① 本工具 ② 使用者執行 `curl -T <檔案> '<upload_url>'` ③ 用 source_uri 呼叫 import_media
        （抽幀進圖片集）；影片只想存進影片庫、不抽幀 → ③ 改呼叫 upload_media_video。
        """
        return call("POST", "/api/media/upload-url", params={"filename": filename})

    @mcp.tool(title="匯入圖片庫", annotations=WRITE)
    @safe
    def import_media(collection_id: int, source_uri: str, kind: str = "zip", with_labels: bool = False,
                     strategy: str = "diverse", target_frames: int = 2000, roi: list[float] | None = None,
                     ignore_regions: list[list[float]] | None = None, dense_interval_s: float = 1.0,
                     interval_s: float = 1.0, max_frames: int = 2000, camera_id: str | None = None,
                     batch_name: str | None = None, batch_id: int | None = None) -> dict:
        """把已上傳的檔案匯入圖片集（背景任務，回 task_id 與 batch_id；用 get_task 看進度）。
        這次匯入 = 一個新批次，或用 batch_id 加進既有批次。

        Args:
            kind: zip（圖片，可含 YOLO 標注）/ video（影片切幀）。
                要某次 NG 事件的畫面：用 import_from_recording（NAS 乾淨錄影）—— 存證影片有燒錄的框，不能當訓練資料
            with_labels: zip 內含 labels/ + classes.txt（或 data.yaml）→ 匯入成「人工標完」
            strategy: 影片怎麼挑：diverse（預設；去重、有變動優先、分群挑 target_frames 張，不用任何模型）
                / interval（每 interval_s 秒一張，上限 max_frames）
            roi: 只看這一塊 [x1,y1,x2,y2]（0–1），例如作業台；省略＝整張
            ignore_regions: 比對時塗掉的區域；省略＝左上角時間戳
            batch_name: 批次名稱（例「第二批」）；省略 = 「zip 匯入 / 影片 <時間>」
            batch_id: 加進這個既有批次（list_media_batches 查）；跟 batch_name 擇一
        """
        body = {"source_uri": source_uri, "kind": kind, "with_labels": with_labels, "strategy": strategy,
                "select": _select(target_frames, roi, ignore_regions, dense_interval_s),
                "interval_s": interval_s, "max_frames": max_frames, "camera_id": camera_id, "batch_name": batch_name,
                "batch_id": batch_id}
        return call("POST", f"/api/media/collections/{collection_id}/import",
                    json_body={k: v for k, v in body.items() if v is not None})

    @mcp.tool(title="從錄影擷取圖片", annotations=WRITE)
    @safe
    def import_from_recording(collection_id: int, camera_id: str, start: str, end: str,
                              strategy: str = "diverse", target_frames: int = 2000, roi: list[float] | None = None,
                              ignore_regions: list[list[float]] | None = None, dense_interval_s: float = 1.0,
                              interval_s: float = 2.0, max_frames: int = 2000, batch_name: str | None = None,
                              batch_id: int | None = None) -> dict:
        """從 NAS 24h 錄影挑一段時間的畫面進圖片集（背景任務，回 task_id 與 batch_id）。
        這次挑的圖 = 一個新批次，或用 batch_id 加進既有批次。

        預設 strategy=diverse（不用任何模型）：每 dense_interval_s 秒看一張 → 去掉幾乎一樣的 →
        有變動的優先 → 分群挑 target_frames 張代表。結果的 import.select 有每一步剩幾張、各時段張數；
        每張圖的 select_info 記它代表多少張相似畫面（group_size）、有沒有在動。

        Args:
            camera_id: 相機編號（如 siteb12）
            start / end: 本地時間 ISO，如 2026-10-13T08:00:00
            strategy: diverse（預設）/ interval（每 interval_s 秒一張，上限 max_frames）
            target_frames: diverse 最後要幾張
            roi: 只看這一塊 [x1,y1,x2,y2]（0–1），例如作業台；走道有人經過就不算變動
            ignore_regions: 比對時塗掉的區域；省略＝左上角時間戳（不遮的話每張都「不一樣」）
            dense_interval_s: 密集看的間隔（秒）
            batch_name: 批次名稱；省略 = 「NAS 錄影 <時間>」
            batch_id: 加進這個既有批次（list_media_batches 查）；跟 batch_name 擇一
        """
        body = {"camera_id": camera_id, "start": start, "end": end, "strategy": strategy,
                "select": _select(target_frames, roi, ignore_regions, dense_interval_s),
                "interval_s": interval_s, "max_frames": max_frames, "batch_name": batch_name, "batch_id": batch_id}
        return call("POST", f"/api/media/collections/{collection_id}/from-recording",
                    json_body={k: v for k, v in body.items() if v is not None})

    @mcp.tool(title="自動標注", annotations=WRITE)
    @safe
    def autolabel_collection(collection_id: int, engine: str, model_name: str | None = None,
                             model_version: int | None = None, prompts: dict[str, str] | None = None,
                             class_map: dict[str, str] | None = None, conf: float = 0.25,
                             class_filter: list[str] | None = None, item_ids: list[int] | None = None,
                             skip_item_ids: list[int] | None = None, status: list[str] | None = None,
                             overwrite_human: bool = False, batch_ids: list[int] | None = None) -> dict:
        """批次 AI 預標（GPU 背景任務，跟訓練共用排隊）。結果是 auto（要人確認後才會匯出成訓練資料）。

        Args:
            engine: registry（Registry 模型，預設現役版）/ yolo_world（提示詞 → 框）/ sam（現有框 → 多邊形）
            prompts: yolo_world 用：{"提示詞": "圖片集類別"}
            class_map: registry 用：模型類別名 → 圖片集類別名（同名可省略）
            class_filter: 只換這些類別，其他保留（例如只把 board 的框換成多邊形）
            skip_item_ids: 跳過這些圖（例如種子圖）
            overwrite_human: 是否改寫人看過的圖；預設不改
            batch_ids: 只跑這些批次的圖（例如剛匯入的新一批）
        """
        body = {"engine": engine, "model_name": model_name, "model_version": model_version, "prompts": prompts,
                "class_map": class_map, "conf": conf, "class_filter": class_filter, "item_ids": item_ids,
                "skip_item_ids": skip_item_ids, "status": status, "overwrite_human": overwrite_human,
                "batch_ids": batch_ids}
        return call("POST", f"/api/media/collections/{collection_id}/autolabel",
                    json_body={k: v for k, v in body.items() if v is not None})

    @mcp.tool(title="圖片集匯出成資料集", annotations=WRITE)
    @safe
    def export_media_collection(collection_id: int, repo: str, message: str, val_pct: int = 20,
                                batch_ids: list[int] | None = None, task: str | None = None) -> dict:
        """把人看過的圖（labeled + background）匯出成 LakeFS 資料集版本，可直接拿去訓練（背景任務）。

        AI 預標（auto）沒人確認的不會匯出。完成後 get_task 的 result.export.dataset_ref 就是資料集版本。

        Args:
            repo: 資料集名稱（小寫英數與 -，3–63 字，不能有底線、大寫）
            message: 版本說明（來源、用途）
            val_pct: 驗證集比例；100 = 全部當驗證（黃金測試集）
            batch_ids: 只匯出這些批次（每批一個快照）；省略 = 整個圖片集
            task: 標注輸出格式，要跟訓練的模型一致：detect 框（多邊形取外接框）/ segment 多邊形（框轉四點）/
                pose 關鍵點；省略 = detect。結果的 export.converted 列出轉換了幾個形狀
        """
        if not message.strip():
            raise ValueError("message 必填：說明這份資料的來源與用途")
        body = {"repo": repo, "message": message, "val_pct": val_pct, "batch_ids": batch_ids, "task": task}
        return call("POST", f"/api/media/collections/{collection_id}/export",
                    json_body={k: v for k, v in body.items() if v is not None})

    @mcp.tool(title="圖片集的匯出紀錄", annotations=RO)
    @safe
    def list_media_exports(collection_id: int) -> list[dict]:
        """這個圖片集匯出過的 LakeFS 版本：repo、dataset_ref、張數、匯出了哪些批次、狀態、誰、何時。"""
        return call("GET", f"/api/media/collections/{collection_id}/exports")

    # ── 審核────────────────────────

    @mcp.tool(title="審核設定", annotations=RO)
    @safe
    def get_review_config(collection_id: int) -> dict:
        """圖片集的審核設定（全域預設 ← 這個圖片集的覆蓋）：每類門檻 / 能否批次通過 / 少見類別、一輪幾張、抽查比例…

        override 欄位 = 這個圖片集存在資料庫裡、跟預設不同的部分。
        """
        return call("GET", f"/api/media/collections/{collection_id}/review-config")

    @mcp.tool(title="修改審核設定", annotations=WRITE)
    @safe
    def set_review_config(collection_id: int, patch: dict) -> dict:
        """部分更新圖片集的審核設定覆蓋（只寫要改的欄位；值給 null = 回到預設）。驗證不過整筆不寫。

        例：{"round": {"size": 300}, "classes": {"tap": {"rare": true, "bulk_accept": false}},
             "swap_pairs": [["glove", "tap"]]}
        """
        return call("PATCH", f"/api/media/collections/{collection_id}/review-config", json_body=patch)

    @mcp.tool(title="建立審核輪次", annotations=WRITE)
    @safe
    def create_review_round(collection_id: int, name: str | None = None, size: int | None = None,
                            blind_ratio: float | None = None, batch_ids: list[int] | None = None) -> dict:
        """建一輪審核：依優先分數（少見類別 / 低信心 / 沒偵測到 / 代表性）挑 size 張，抽 blind_ratio 當抽查圖。

        抽查圖的預標會被清空（原預測留在輪次裡，之後拿來量模型準不準），審核時要從頭標。

        Args:
            size: 幾張；省略 = 設定的 round.size
            blind_ratio: 抽查比例；省略 = 設定的 round.blind_ratio
            batch_ids: 只從這些批次挑；省略 = 整個圖片集
        """
        body = {"name": name, "size": size, "blind_ratio": blind_ratio, "batch_ids": batch_ids}
        return call("POST", f"/api/media/collections/{collection_id}/rounds",
                    json_body={k: v for k, v in body.items() if v is not None})

    @mcp.tool(title="列出審核輪次", annotations=RO)
    @safe
    def list_review_rounds(collection_id: int) -> list[dict]:
        """圖片集的審核輪次：狀態、進度（待審 / 通過 / 背景 / 有疑問）、各原因張數、審核者平均秒數。"""
        return call("GET", f"/api/media/collections/{collection_id}/rounds")

    @mcp.tool(title="審核輪次詳情", annotations=RO)
    @safe
    def get_review_round(round_id: int) -> dict:
        """單一輪次：進度（待審 / 通過 / 背景 / 有疑問）、各原因張數、抽查張數、審核者平均秒數、建立當下的完整設定。"""
        return call("GET", f"/api/media/rounds/{round_id}")

    @mcp.tool(title="輪次裡的圖", annotations=RO)
    @safe
    def list_round_items(round_id: int, state: str | None = None, reason: str | None = None,
                         class_name: str | None = None, blind: bool | None = None, order: str = "priority",
                         offset: int = 0, limit: int = 100) -> dict:
        """輪次裡的圖（縮圖網址、框、原因、狀態）。抽查圖審完前不給框。

        Args:
            state: pending（待審）/ done / doubt（有疑問）
            reason: rare / low_conf / empty / diverse
            class_name: 只列含這個類別框的圖
            order: priority（審核順序）/ time / conf（信心最低的先）
        """
        params = {"state": state, "reason": reason, "class": class_name, "blind": blind, "order": order,
                  "offset": offset, "limit": limit}
        return call("GET", f"/api/media/rounds/{round_id}/items",
                    params={k: v for k, v in params.items() if v is not None})

    @mcp.tool(title="下一張待審", annotations=WRITE)
    @safe
    def next_review_item(round_id: int, after_id: int | None = None) -> dict:
        """依優先順序拿下一張待審的圖，並鎖給自己（別人拿不到）。item 為 null = 這輪審完了。

        回傳的 item.image_url 是原圖臨時網址；item.review.blind = true 表示抽查圖（沒有預標，要從頭標）。
        看完用 review_media_item 送出；不審了要放掉鎖就用 doubt 或等鎖逾時。
        """
        params = {"after_id": after_id} if after_id is not None else None
        return call("GET", f"/api/media/rounds/{round_id}/next", params=params)

    @mcp.tool(title="單張圖", annotations=RO)
    @safe
    def get_media_item(item_id: int) -> dict:
        """一張圖：原圖臨時網址、框（含 id 與信心）、狀態、revision、鎖、所在審核輪次。"""
        return call("GET", f"/api/media/items/{item_id}")

    @mcp.tool(title="一張圖的標注歷史", annotations=RO)
    @safe
    def get_media_item_history(item_id: int, limit: int = 20) -> list[dict]:
        """這張圖每一次寫入（匯入 / AI 預標 / 人送出 / 審核）的框、狀態、誰、哪個任務，最新的在前。"""
        return call("GET", f"/api/media/items/{item_id}/history", params={"limit": limit})

    @mcp.tool(title="圖片集標注進度", annotations=RO)
    @safe
    def get_collection_progress(collection_id: int) -> dict:
        """各狀態張數（unlabeled / auto / labeled / background …）。"""
        return call("GET", f"/api/media/collections/{collection_id}/progress")

    @mcp.tool(title="批次詳情", annotations=RO)
    @safe
    def get_media_batch(batch_id: int) -> dict:
        """一批（每次把圖放進圖片集算一批）的名稱、來源、張數、建立者與時間。改名用 update_media_batch。"""
        return call("GET", f"/api/media/batches/{batch_id}")

    @mcp.tool(title="續標下一張", annotations=WRITE)
    @safe
    def next_collection_item(collection_id: int, status: list[str] | None = None,
                             after_id: int | None = None) -> dict:
        """不經審核輪次、直接在圖片集裡續標：從上次停的地方找下一張並領取（上鎖）。
        status 預設 unlabeled + auto；item 為 null = 沒有了。"""
        params: dict[str, Any] = {"status": status, "after_id": after_id}
        return call("GET", f"/api/media/collections/{collection_id}/next",
                    params={k: v for k, v in params.items() if v is not None})

    @mcp.tool(title="鎖定 / 放開一張圖", annotations=WRITE)
    @safe
    def lock_media_item(item_id: int, release: bool = False, seconds: int | None = None) -> dict:
        """標注前先鎖，別人（前端）就不會同時改；release=True 放開。seconds：鎖多久（預設照伺服器設定）。"""
        if release:
            return call("DELETE", f"/api/media/items/{item_id}/lock")
        return call("POST", f"/api/media/items/{item_id}/lock",
                    params={"seconds": seconds} if seconds else None)

    @mcp.tool(title="存標注草稿", annotations=WRITE)
    @safe
    def save_media_draft(item_id: int, revision: int, shapes: list[dict]) -> dict:
        """暫存框（不改狀態、不留歷史）。revision 不符回 409。座標格式同 submit_media_annotation。"""
        return call("PUT", f"/api/media/items/{item_id}/draft", json_body={"shapes": shapes, "revision": revision})

    @mcp.tool(title="送出標注", annotations=WRITE)
    @safe
    def submit_media_annotation(item_id: int, revision: int, shapes: list[dict], status: str = "labeled") -> dict:
        """直接把框寫進一張圖（不經審核輪次）：人工標注、批次修正用。

        Args:
            revision: get_media_item 拿到的 revision（被別人改過會 409）
            shapes: 整張圖的框（會整份取代）。座標 0–1：
                {"type": "rect", "class": "Load_Board", "points": [[x1, y1], [x2, y2]]}
            status: labeled（有標註）/ background（確認沒有任何物件，shapes 給空）
        """
        return call("POST", f"/api/media/items/{item_id}/submit",
                    json_body={"shapes": shapes, "revision": revision, "status": status})

    @mcp.tool(title="單張即時預測", annotations=RO)
    @safe
    def predict_media_item(item_id: int, model_name: str, conf: float = 0.25,
                           class_map: dict[str, str] | None = None) -> dict:
        """用 Triton 上的現役模型預測一張圖，只回框、不寫入。class_map：模型類別 → 圖片集類別。"""
        body: dict[str, Any] = {"model_name": model_name, "conf": conf}
        if class_map:
            body["class_map"] = class_map
        return call("POST", f"/api/media/items/{item_id}/predict", json_body=body)

    @mcp.tool(title="送出審核", annotations=WRITE)
    @safe
    def review_media_item(item_id: int, revision: int, action: str, shapes: list[dict] | None = None,
                          note: str | None = None, seconds_spent: float | None = None) -> dict:
        """審核送出一張圖。

        Args:
            revision: 拿到圖時的 revision（被別人改過會 409）
            action: submit（確認，→ labeled，會進訓練集）/ background（確認沒有任何物件）/
                doubt（拿不定，不改圖、記備註、放掉鎖，留給人判斷）
            shapes: 要留下的框（action=submit）。修改的框保留原 id；新增的框不用給 id。
                座標 0–1：{"type": "rect", "class": "tap", "points": [[x1, y1], [x2, y2]], "id": "..."}
            note: action=doubt 必填，寫清楚哪裡拿不定
        """
        body = {"revision": revision, "action": action, "shapes": shapes or [], "note": note,
                "seconds_spent": seconds_spent}
        return call("POST", f"/api/media/items/{item_id}/review",
                    json_body={k: v for k, v in body.items() if v is not None})

    @mcp.tool(title="關閉審核輪次", annotations=WRITE)
    @safe
    def close_review_round(round_id: int) -> dict:
        """關閉輪次（之後只能看）。沒審完的圖回到可選池，下一輪還會被挑。"""
        return call("POST", f"/api/media/rounds/{round_id}/close")

    # ── 標注專案────────

    @mcp.tool(title="列出標注專案", annotations=RO)
    @safe
    def list_campaigns() -> list[dict]:
        """標注專案（照配方一輪一輪訓練的專線）：每個專案的狀態、進行到哪一步、下一輪是第幾代、圖片集、初始模型。"""
        return call("GET", "/api/media/campaigns")

    @mcp.tool(title="標注專案詳情", annotations=RO)
    @safe
    def get_campaign(campaign_id: int) -> dict:
        """專案詳情：狀態（idle 閒置 / running 進行中 / failed 失敗 / closed 結案）與進行到哪一步（stage）、
        圖片集的所有批次（哪些是初始、最新批次快照）、配方、每一代的批次快照 / 配方 / 訓練快照 / 模型版本 / 指標 / 預標任務、
        最新一代的模型（latest_model）、最近事件、完整設定。
        """
        return call("GET", f"/api/media/campaigns/{campaign_id}")

    @mcp.tool(title="建立標注專案", annotations=WRITE)
    @safe
    def create_campaign(name: str, initial_model: dict, model_name: str, base_model: str,
                        collection_id: int | None = None, initial_batch_ids: list[int] | None = None,
                        line_id: str | None = None, camera_id: str | None = None, golden_ref: str | None = None,
                        golden_collection_id: int | None = None, config: dict | None = None,
                        recipe: dict | None = None) -> dict:
        """建標注專案（全部是引用：圖片庫的圖片集與批次、Registry 的模型、LakeFS 的測試集）。建好是閒置，不會自己跑。

        專案只管訓練＋預標：run_campaign = 匯出 → 混合（依配方）→ 訓練 → 用新模型預標還沒人審的圖；
        prelabel_campaign = 只用最新一代的模型預標。加批次用圖片庫的匯入（import_media / import_from_recording /
        copy_media_items），審核用圖片集審核，上線用 promote_model，專案都不管。
        每一代都從 base_model 重新訓練、註冊成 model_name 的新 candidate 版本（不會自動上線）。

        Args:
            collection_id: 圖片集（初始批次與之後每一批都在這）。不給 → 用 initial_batch_ids 所在的圖片集；
                都沒給 → 自動建一個空的（名稱＝專案名、類別從初始模型帶）。一個圖片集只給一個專案用
            line_id / camera_id: 自動建圖片集時記在圖片集上
            initial_model: 初始模型 {"name": "siteb_bravo_seed", "version": 1}（第 0 代）
            initial_batch_ids: 圖片集裡哪幾個批次是初始人工標注（list_media_batches 查）
            model_name: 每代註冊的模型名，建議用產線最後要用的名字（例 siteb_bravo）
            base_model: 例 s3://models/_base/yolo26m.pt
            golden_ref: 測試集（LakeFS 快照），可之後用 update_campaign 設；每一代訓練都在它上面評估
            golden_collection_id: 測試集圖片集（只是參照，前端顯示標注進度）
            config: 覆蓋預設（dataset、prelabel.conf、train、export、iteration_gates）
            recipe: 混合配方 {"batches": {"<batch_id>": {"repeat": 0.5}}, "total": …, "seed": 0}；沒寫的批次全部取一次
        """
        body = {"name": name, "collection_id": collection_id, "initial_model": initial_model,
                "initial_batch_ids": initial_batch_ids, "line_id": line_id, "camera_id": camera_id,
                "model_name": model_name, "base_model": base_model,
                "golden_ref": golden_ref, "golden_collection_id": golden_collection_id, "config": config,
                "recipe": recipe}
        return call("POST", "/api/media/campaigns", json_body={k: v for k, v in body.items() if v is not None})

    @mcp.tool(title="修改標注專案", annotations=WRITE)
    @safe
    def update_campaign(campaign_id: int, golden_ref: str | None = None, clear_golden: bool = False,
                        golden_collection_id: int | None = None, clear_golden_collection: bool = False,
                        config: dict | None = None, recipe: dict | None = None,
                        initial_batch_ids: list[int] | None = None, initial_model: dict | None = None) -> dict:
        """改專案：換測試集、調混合配方（下一次 run 生效）、部分更新設定覆蓋（值給 null = 回到預設）、
        更正初始批次 / 初始模型。

        配方例：{"batches": {"4": {"repeat": 0.5}, "7": {"choose": 300}}} —— 批次 4 抽一半、批次 7 取 300 張，
        其他批次全部取一次；{"batches": {"4": null}} 拿掉批次 4 的設定。
        """
        body = {"golden_ref": golden_ref, "clear_golden": clear_golden, "golden_collection_id": golden_collection_id,
                "clear_golden_collection": clear_golden_collection, "config": config, "recipe": recipe,
                "initial_batch_ids": initial_batch_ids, "initial_model": initial_model}
        return call("PATCH", f"/api/media/campaigns/{campaign_id}",
                    json_body={k: v for k, v in body.items() if v is not None})

    @mcp.tool(title="標注專案：跑下一輪", annotations=WRITE)
    @safe
    def run_campaign(campaign_id: int) -> dict:
        """跑下一輪（專案閒置時）：有改過的批次各自匯出快照 → 依配方混成訓練快照 → 訓練（GPU，約半小時）→
        用新模型預標圖片集裡還沒人審的圖 → 回閒置，記成一代。進度看 get_campaign 的 state / stage / step。
        """
        return call("POST", f"/api/media/campaigns/{campaign_id}/run")

    @mcp.tool(title="標注專案：只預標", annotations=WRITE)
    @safe
    def prelabel_campaign(campaign_id: int) -> dict:
        """只預標（專案閒置時）：用最新一代的模型（get_campaign 的 latest_model）預標圖片集裡還沒人審的圖。
        加了新批次、還不想訓練時用；預標過但還沒審的圖也會被重新預標。
        """
        return call("POST", f"/api/media/campaigns/{campaign_id}/prelabel")

    @mcp.tool(title="標注專案：重試", annotations=WRITE)
    @safe
    def retry_campaign(campaign_id: int) -> dict:
        """專案失敗（state=failed，看 last_error）時，修好原因後從失敗的那一步重跑，不從頭。"""
        return call("POST", f"/api/media/campaigns/{campaign_id}/retry")

    @mcp.tool(title="標注專案：結案 / 接回", annotations=WRITE)
    @safe
    def close_campaign(campaign_id: int, reopen: bool = False) -> dict:
        """結案（reopen=false）：它的圖片集釋放出來，別的專案才能拿去用；圖片集、批次、快照、模型、每一代紀錄都留著。
        跑到一半不能結案。reopen=true 接回（回到閒置）；圖片集這期間被別的專案拿走就不能接回。
        """
        if reopen:
            return call("POST", f"/api/media/campaigns/{campaign_id}/reopen")
        return call("POST", f"/api/media/campaigns/{campaign_id}/close")

    # ── 告警回流（edgevision/media/alerts.py）──────────────────

    @mcp.tool(title="告警待判清單", annotations=RO)
    @safe
    def list_alert_queue(day: str | None = None, line_id: str | None = None, camera_id: str | None = None,
                         station: str | None = None, per_station: int | None = None) -> dict:
        """某天（YYYY-MM-DD，預設今天）的線上 NG 告警事件，每站抽 per_station 個（同一天重複查是同一批；判過的一律列出）。
        每個事件有 event_id、證據影片網址、當下的框、判定、回流結果。用 review_alert 判。"""
        q = {"day": day, "line_id": line_id, "camera_id": camera_id, "station": station, "per_station": per_station}
        return call("GET", "/api/media/alerts/queue", params={k: v for k, v in q.items() if v is not None})

    @mcp.tool(title="判告警", annotations=WRITE)
    @safe
    def review_alert(event_id: int, result: str | None) -> dict:
        """判一個告警事件：true_positive（真異常）/ false_positive（誤報）/ null（取消）。
        判過的會在每天的回流放進該模型標注專案的圖片集（NAS 原始幀＋當下的框當標注）；已回流的不能改。"""
        return call("POST", f"/api/media/alerts/{event_id}/review", json_body={"result": result})

    @mcp.tool(title="執行告警回流", annotations=WRITE)
    @safe
    def run_alert_feedback(lookback_days: int | None = None, dry_run: bool = False) -> dict:
        """把最近 N 天判過、還沒回流的告警送進各模型標注專案的圖片集（每個圖片集一個「告警回流 <日期>」批次＋匯入任務）。
        每天 CronJob 會自動跑；dry_run=true 只看會怎麼分。"""
        body = {"lookback_days": lookback_days, "dry_run": dry_run}
        return call("POST", "/api/media/alerts/feedback/run", json_body={k: v for k, v in body.items() if v is not None})

    @mcp.tool(title="告警回流紀錄", annotations=RO)
    @safe
    def list_alert_feedback(status: str | None = None, task_id: str | None = None, limit: int = 200) -> list[dict]:
        """每個事件回流的結果：added（放進哪個圖片集、哪張圖、labeled/auto）、skipped / failed（原因）、
        align（怎麼對到 NAS 那一幀：ok、score、ts_offset、why）。"""
        q = {"status": status, "task_id": task_id, "limit": limit}
        return call("GET", "/api/media/alerts/feedback", params={k: v for k, v in q.items() if v is not None})

    # ── 影片測試（edgevision/media/video_test.py）──────────────────

    @mcp.tool(title="列出影片庫", annotations=RO)
    @safe
    def list_media_videos(camera_id: str | None = None, kind: str | None = None, tag: str | None = None,
                          q: str | None = None, limit: int = 100) -> list[dict]:
        """影片庫：上傳的影片（upload_video）、從 NAS 錄影剪的片段（recording_clip）、抽圖用過的 NAS 錄影檔（nas_recording）。
        影片測試用它的 id 當 source_id。q = 名稱關鍵字。"""
        p = {"camera_id": camera_id, "kind": kind, "tag": tag, "q": q, "limit": limit}
        return call("GET", "/api/media/videos", params={k: v for k, v in p.items() if v})

    @mcp.tool(title="影片詳情", annotations=RO)
    @safe
    def get_media_video(video_id: int) -> dict:
        """影片庫的一筆，含被哪些圖片集 / 批次用過、跑過哪些影片測試。"""
        return call("GET", f"/api/media/videos/{video_id}")

    @mcp.tool(title="修改影片", annotations=WRITE)
    @safe
    def update_media_video(video_id: int, name: str | None = None, tags: list[str] | None = None,
                           note: str | None = None) -> dict:
        """改影片庫一筆的名稱、標籤（整組換掉；可用來標「黃金測試影片」等）、備註。只送要改的欄位。"""
        body = {"name": name, "tags": tags, "note": note}
        return call("PATCH", f"/api/media/videos/{video_id}", json_body={k: v for k, v in body.items() if v is not None})

    @mcp.tool(title="刪除影片", annotations=DESTRUCTIVE)
    @safe
    def delete_media_video(video_id: int) -> dict:
        """從影片庫刪除。從它抽出的圖都還在；MinIO 上的影片檔沒別人用就一起刪；NAS 錄影原檔不會刪。"""
        return call("DELETE", f"/api/media/videos/{video_id}")

    # ── NAS 24h 錄影（edgevision/media/recordings.py）──────────────

    @mcp.tool(title="NAS 錄影總覽", annotations=RO)
    @safe
    def list_recordings() -> dict:
        """NAS 用量＋有錄影的相機（段數、大小、最後寫入距今幾秒、在不在錄影名單）。"""
        return call("GET", "/api/media/recordings")

    @mcp.tool(title="某台相機的錄影", annotations=RO)
    @safe
    def get_recordings(camera_id: str, day: str | None = None) -> list[dict] | dict:
        """day 不給 = 有錄影的日子清單；給 YYYY-MM-DD = 那天每一段的起訖、狀態（ok / writing / broken）、缺口、播放網址。"""
        if day:
            return call("GET", f"/api/media/recordings/{camera_id}", params={"day": day})
        return call("GET", f"/api/media/recordings/{camera_id}/days")

    @mcp.tool(title="上傳影片到影片庫", annotations=WRITE)
    @safe
    def upload_media_video(source_uri: str, name: str | None = None, tags: list[str] | None = None,
                           note: str | None = None, camera_id: str | None = None) -> dict:
        """影片只存進影片庫（不抽幀、不需要圖片集），之後可拿去跑影片測試。source_uri 來自 create_media_upload_url
        （影片要先 curl -T 傳上去）。背景任務，回 task_id；完成後 get_task 的 result.video.video_id。
        同一支影片已在影片庫 → 不重複新增，result.video.duplicate = true。name 不給 = 原檔名。"""
        body = {"source_uri": source_uri, "name": name, "tags": tags or [], "note": note, "camera_id": camera_id}
        return call("POST", "/api/media/videos", json_body={k: v for k, v in body.items() if v is not None})

    @mcp.tool(title="剪錄影片段", annotations=WRITE)
    @safe
    def create_recording_clip(camera_id: str, start: str, end: str, name: str | None = None,
                              tags: list[str] | None = None, note: str | None = None) -> dict:
        """從 NAS 錄影剪一段（本地時間 ISO）存進影片庫（背景任務，回 task_id；完成後 get_task 的 result.clip.video_id）。
        不重新編碼，起點對齊到前一個關鍵幀；一次最多 60 分鐘；剪下的不會自動刪。"""
        body = {"start": start, "end": end, "name": name, "tags": tags or [], "note": note}
        return call("POST", f"/api/media/recordings/{camera_id}/clips",
                    json_body={k: v for k, v in body.items() if v is not None})

    @mcp.tool(title="錄影名單", annotations=RO)
    @safe
    def get_recording_roster() -> list[dict]:
        """所有相機要不要 24h 錄到 NAS，與名單上的每台最近 24 小時實際寫了多少 GB。"""
        return call("GET", "/api/media/recordings/roster")

    @mcp.tool(title="開關錄影", annotations=WRITE)
    @safe
    def set_recording_roster(camera_id: str, enabled: bool, note: str | None = None) -> dict:
        """開 / 關這台的 24h 錄影（nas-recorder 5 分鐘內跟上）。⚠️ 原始畫質一台一天約 10–25 GB，開之前先看 NAS 剩多少。"""
        body = {"enabled": enabled, "note": note}
        return call("PUT", f"/api/media/recordings/roster/{camera_id}",
                    json_body={k: v for k, v in body.items() if v is not None})

    # ── 圖片集 / 圖片管理（edgevision/media/collection_ops.py）──────────────

    @mcp.tool(title="刪除圖片集", annotations=DESTRUCTIVE)
    @safe
    def delete_media_collection(collection_id: int, confirm: str) -> dict:
        """刪圖片集（裡面的圖、標注、批次、輪次一起刪；圖片庫原圖與已匯出的快照不動）。confirm = 圖片集名稱。
        被標注專案用著 → 擋下。只是不想看到請用 update_media_collection(archived=true)。"""
        return call("DELETE", f"/api/media/collections/{collection_id}", params={"confirm": confirm})

    @mcp.tool(title="類別改名 / 刪除", annotations=DESTRUCTIVE)
    @safe
    def change_collection_classes(collection_id: int, rename: dict | None = None,
                                  delete: list[str] | None = None) -> dict:
        """rename = {"舊名": "新名"}：所有框、審核設定、專案退步判斷一起改。delete = 類別清單：那些框全部拿掉。
        順序不能調（快照與模型依順序編號）。已匯出的快照、訓練好的模型仍是舊名稱。"""
        body = {"rename": rename, "delete": delete}
        return call("PATCH", f"/api/media/collections/{collection_id}/classes",
                    json_body={k: v for k, v in body.items() if v is not None})

    @mcp.tool(title="找圖", annotations=RO)
    @safe
    def search_media_items(q: str | None = None, sha256: str | None = None, tag: str | None = None,
                           collection_id: int | None = None, limit: int = 100) -> list[dict]:
        """跨圖片集找圖：檔名關鍵字、sha256（前 8 碼以上）、標籤。"""
        p = {"q": q, "sha256": sha256, "tag": tag, "collection_id": collection_id, "limit": limit}
        return call("GET", "/api/media/items/search", params={k: v for k, v in p.items() if v is not None})

    @mcp.tool(title="圖片標籤 / 備註", annotations=WRITE)
    @safe
    def set_media_item_meta(item_id: int, tags: list[str] | None = None, note: str | None = None,
                            clear_note: bool = False) -> dict:
        """圖片的標籤（整組換掉）與備註；跟標注無關，不算修訂、不進匯出。"""
        body = {"tags": tags, "note": note, "clear_note": clear_note}
        return call("PATCH", f"/api/media/items/{item_id}/meta",
                    json_body={k: v for k, v in body.items() if v is not None})

    @mcp.tool(title="打包下載圖片集", annotations=WRITE)
    @safe
    def download_media_collection(collection_id: int, format: str, statuses: list[str] | None = None,
                                  batch_ids: list[int] | None = None, val_pct: int = 20) -> dict:
        """打包成 zip（format：yolo 框 / yolo-seg 多邊形 / yolo-pose 關鍵點 / coco / voc；背景任務）。statuses 預設只含人確認過的（labeled、background），
        加 auto = 含 AI 預標。完成後 get_media_download 拿網址。"""
        body = {"format": format, "statuses": statuses, "batch_ids": batch_ids, "val_pct": val_pct}
        return call("POST", f"/api/media/collections/{collection_id}/download",
                    json_body={k: v for k, v in body.items() if v is not None})

    @mcp.tool(title="下載進度", annotations=RO)
    @safe
    def get_media_download(task_id: str) -> dict:
        """打包任務（圖片集或資料集版本）的狀態與下載網址（2 小時有效）。"""
        return call("GET", f"/api/media/downloads/{task_id}")

    # ── 資料集版本（LakeFS）──────────────────────────────

    @mcp.tool(title="資料集版本清單", annotations=RO)
    @safe
    def list_dataset_versions(repo: str, limit: int = 50, after: str | None = None) -> dict:
        """一個資料集 repo 的全部版本（新到舊，分頁；next 不是 null 就還有）。list_datasets 只附最近 10 版。"""
        p = {"limit": limit, "after": after}
        return call("GET", f"/api/datasets/{repo}/versions", params={k: v for k, v in p.items() if v is not None})

    @mcp.tool(title="比較兩個資料集版本", annotations=RO)
    @safe
    def diff_dataset_versions(repo: str, from_version: str, to_version: str, limit: int = 200) -> dict:
        """兩版差在哪：新增 / 刪除 / 改變的檔案數（整體、依目錄、只算圖片），加前 limit 筆明細。"""
        return call("GET", f"/api/datasets/{repo}/diff", params={"from": from_version, "to": to_version, "limit": limit})

    @mcp.tool(title="打包下載資料集版本", annotations=WRITE)
    @safe
    def download_dataset_version(repo: str, version_id: str) -> dict:
        """資料集版本原樣打包成 zip（背景任務）；完成後 get_media_download 拿網址。"""
        return call("POST", f"/api/datasets/{repo}/versions/{version_id}/download")

    @mcp.tool(title="刪除資料集", annotations=DESTRUCTIVE)
    @safe
    def delete_dataset(repo: str, confirm: str) -> dict:
        """刪整個資料集 repo（LakeFS 版本不可變，不能只刪一版）。confirm = repo 名稱。
        有訓練任務（模型的來源）或標注專案指向它就擋下。"""
        return call("DELETE", f"/api/datasets/{repo}", params={"confirm": confirm})

    @mcp.tool(title="影片測試", annotations=WRITE)
    @safe
    def create_video_test(line_id: str, source_id: int | None = None, model_name: str | None = None,
                          model_version: int | None = None, camera_id: str | None = None,
                          start_s: float | None = None, end_s: float | None = None,
                          start: str | None = None, end: str | None = None,
                          station: str | None = None, station_params: dict | None = None,
                          mode: str = "logic") -> dict:
        """拿一段影片，用指定的模型版本 ＋ 某條線的判定邏輯跑一遍（跟線上同一個 runner），存下打框影片與 NG 清單。

        GPU 背景任務（跟訓練共用排隊），回 task_id。mode=logic 照影片實際速度播，10 分鐘的影片大約跑 10 分鐘（影片最長 10 分鐘，長的請切段）；
        mode=model 純模型測試，能跑多快跑多快、不出 NG，改出各類別框數 / 出現幀數 / 平均信心值。
        同一段影片換不同模型版本各跑一次，結果各自保留，可以並排比。

        Args:
            line_id: 用哪條線的判定邏輯與參數（例 siteb_echo）
            source_id: 圖片庫的影片（list_media_videos 的 id）；或改給 camera_id + start + end 直接拿 NAS 錄影
            model_name / model_version: 主站模型；不給 = 這條線現在的模型、現役版
            camera_id: 用哪台相機的現場參數（ROI 等）；NAS 錄影時也是錄影的相機
            start_s / end_s: source_id 時只測其中一段（秒）
            start / end: NAS 錄影的本地時間 ISO，如 2026-10-16T08:00:00
            station: 換成別的判定邏輯（站別名，例 siteb_bravo_v2）；新邏輯上線前先拿錄影試。不給 = 這條線現在的
            station_params: 換站別時的現場參數（thresholds / roi_regions）；不給 = 站別預設
            mode: logic（預設，邏輯＋模型，出 NG）或 model（純模型，比兩版模型看得到什麼、哪裡漏框；不需要 ROI、不載入判定邏輯）
        """
        body = {"line_id": line_id, "source_id": source_id, "model_name": model_name, "model_version": model_version,
                "camera_id": camera_id, "start_s": start_s, "end_s": end_s, "start": start, "end": end,
                "station": station, "station_params": station_params, "mode": mode}
        return call("POST", "/api/media/video-tests", json_body={k: v for k, v in body.items() if v is not None})

    @mcp.tool(title="影片測試紀錄", annotations=RO)
    @safe
    def list_video_tests(line_id: str | None = None, model_name: str | None = None,
                         source_id: int | None = None, limit: int = 50) -> list[dict]:
        """歷次影片測試（新的在前）：每次用的模式、線路、模型與版本、NG 次數與原因、倍速。可依線路 / 模型 / 影片篩。"""
        q = {"line_id": line_id, "model_name": model_name, "source_id": source_id, "limit": limit}
        return call("GET", "/api/media/video-tests", params={k: v for k, v in q.items() if v is not None})

    @mcp.tool(title="影片測試結果", annotations=RO)
    @safe
    def get_video_test(task_id: str, ng_limit: int = 200) -> dict:
        """一次影片測試的結果：NG 清單（t = 影片第幾秒；mode=model 沒有）、各類別統計 class_stats（框數 / 出現幀數 / 平均信心值）、
        打框影片 video_url 與每幀結果 results_url（2 小時有效）。"""
        return call("GET", f"/api/media/video-tests/{task_id}", params={"ng_limit": ng_limit})
