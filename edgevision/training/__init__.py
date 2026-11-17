"""訓練層。

每一步都是獨立、冪等的函式，也是獨立 CLI（python -m edgevision.training.pipeline <step>）。
GPU 相依套件（ultralytics / torch）一律在函式內延遲匯入，
讓契約、設定、報告這些純邏輯在無 GPU 的開發機也能測。
"""
