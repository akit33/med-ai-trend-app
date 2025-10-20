#!/bin/bash
# start_app.sh

# 仮想環境のアクティベート (Python 3.11 を使用)
source venv/bin/activate

echo "========================================="
echo "  [1/3] PyTorch Prediction Worker (Flask/Gunicorn) 起動中..."
echo "  ワーカー数: 1 (初回デバッグと安定性を最優先)"
echo "========================================="
# Flask ワーカーを Gunicorn で起動。ワーカー数を 1 に設定
gunicorn app.pt_worker:app -w 1 -b 127.0.0.1:5002 &
WORKER_PID=$!
sleep 3 # ワーカーの起動を待つ

echo "========================================="
echo "  [2/3] FastAPI Router (Uvicorn) 起動中..."
echo "========================================="
# FastAPI ルーターを Uvicorn で起動。ホットリロードを有効に
uvicorn app.main:app --host 127.0.0.1 --port 8000 &
ROUTER_PID=$!
sleep 2 # ルーターの起動を待つ

echo "========================================="
echo "  [3/3] Streamlit Frontend 起動中..."
echo "========================================="
# Streamlit UI を起動。Python のパッケージ検索パスにルートディレクトリを追加
PYTHONPATH="." streamlit run app/ui.py 

# Streamlit 終了時に、バックグラウンドで起動していたワーカーとルーターを停止
kill $WORKER_PID
kill $ROUTER_PID