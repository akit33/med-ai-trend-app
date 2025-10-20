# app/main.py（バックエンド API ルーター）

from fastapi import FastAPI, HTTPException
import requests
import os
from dotenv import load_dotenv # 環境変数のロード

# ==============================================================================
# I. 初期設定とエンドポイント定義
# ==============================================================================

# Fast API のインスタンス化 (バックエンド API ルーター)
app = FastAPI(
    title="Medical AI Trend Analysis API",
    description="Streamlit からのリクエストを受け取り、予測ワーカーに転送します。"
)

# 環境変数のロード
load_dotenv(dotenv_path='../.env') # ルートディレクトリの.env ファイルを指定

# 予測ワーカー (Flask) の URL
# 環境変数が見つからない場合、デフォルト値 (http://127.0.0.1:5002/predict) を使用
WORKER_URL = os.getenv("WORKER_URL", "http://127.0.0.1:5002/predict") 

# ==============================================================================
# II. ルーターエンドポイント (Prediction Router)
# ==============================================================================

@app.post("/predict")
def predict(keywords: dict):
    """
    ユーザーからの予測リクエストを受け取り、Flask 予測ワーカーに転送します。
    リクエストが集中した場合、FastAPI が応答を待つ間、他のリクエストを捌くことが可能です。
    """
    try:
        # ワーカーへの POST リクエストを送信 (30秒のタイムアウトを設定し、応答性を確保)
        resp = requests.post(WORKER_URL, json=keywords, timeout=30)
        resp.raise_for_status() # HTTP ステータスコードが 4xx または 5xx の場合に例外を発生させる
        
        # ワーカーからのレスポンスをそのままクライアント (Streamlit) に返却
        return resp.json()
        
    except requests.exceptions.RequestException as e:
        # サービス利用不可 (503) エラーのハンドリング
        # ワーカーがダウンしている、またはタイムアウトした場合にクライアントに通知
        raise HTTPException(status_code=503, detail=f"予測ワーカーへの接続に失敗しました: {e}. ワーカー (pt_worker.py) が起動しているか確認してください。")

# ==============================================================================
# III. ヘルスチェック (オプション)
# ==============================================================================

@app.get("/health")
def health_check():
    """ サービスが稼働していることを確認するためのヘルスチェックエンドポイント。 """
    return {"status": "ok", "service": "FastAPI Router"}