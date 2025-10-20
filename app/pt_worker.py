# app/pt_worker.py

from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import joblib
import numpy as np
import json
import os
from dotenv import load_dotenv

# ==============================================================================
# I. 環境設定と初期化
# ==============================================================================

load_dotenv(dotenv_path='../.env')  # ルートの.envをロード
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "../pubmed_articles_details.csv")
MODEL_PATH = os.path.join(BASE_DIR, "../model/lstm_model_state.pth")
SCALER_PATH = os.path.join(BASE_DIR, "../model/scaler.pkl")
HYPERPARAMS_PATH = os.path.join(BASE_DIR, "../model/hyperparams.json")

print("=== デバッグ情報 ===")
print("Current working dir:", os.getcwd())
print("MODEL_PATH:", os.path.abspath(MODEL_PATH))
print("Exists:", os.path.exists(MODEL_PATH))
print("===================")

BASE_MODEL = None
SCALER = None
GLOBAL_ARTICLE_DATA = None  # CSVデータのインメモリキャッシュ
GLOBAL_HYPERPARAMS = None   # hyperparams.json の内容を保持

# ==============================================================================
# II. LSTM モデル定義
# ==============================================================================

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout_rate=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out

# ==============================================================================
# III. 初期ロード処理
# ==============================================================================

def load_global_assets():
    global BASE_MODEL, SCALER, GLOBAL_ARTICLE_DATA, GLOBAL_HYPERPARAMS

    # --- 1. モデルとスケーラーのロード ---
    try:
        with open(HYPERPARAMS_PATH, "r") as f:
            hyperparams = json.load(f)
        GLOBAL_HYPERPARAMS = hyperparams  # グローバルに保存

        input_dim = 1
        output_dim = 1
        hidden_dim = hyperparams["hidden_dim"]
        num_layers = 1
        dropout_rate = 0.2
        
        BASE_MODEL = LSTMModel(input_dim, hidden_dim, output_dim, num_layers, dropout_rate)
        BASE_MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
        BASE_MODEL.eval()
        
        SCALER = joblib.load(SCALER_PATH)
        print("✅ モデルとスケーラーが正常にロードされました。")
        print(" 使用ハイパーパラメータ:", hyperparams)

    except FileNotFoundError:
        print("❌ エラー: モデルアセットが見つかりません。")
        BASE_MODEL = None
        SCALER = None
        return
    except Exception as e:
        print(f"❌ モデルまたはスケーラーのロードに失敗しました: {e}")
        BASE_MODEL = None
        SCALER = None
        return

    # --- 2. CSV データロードと前処理 ---
    try:
        data_df = pd.read_csv(DATA_PATH, on_bad_lines='skip')

        # 数値列を安全に変換（CSVの列に合わせて変更）
        numeric_cols = ['文献数', 'keyword_count']
        for col in numeric_cols:
            if col in data_df.columns:
                data_df[col] = pd.to_numeric(data_df[col], errors='coerce').fillna(0)

        # 検索用テキスト列の作成（文字列列だけ結合）
        text_cols = ['Title', 'Abstract']
        for col in text_cols:
            if col not in data_df.columns:
                data_df[col] = ''
        data_df['search_text'] = data_df[text_cols].fillna('').agg(' '.join, axis=1)

        # 日付列の処理（列名に合わせて変更）
        if 'Date' in data_df.columns:
            data_df['Date'] = pd.to_datetime(data_df['Date'], errors='coerce')
            data_df = data_df.dropna(subset=['Date'])

        GLOBAL_ARTICLE_DATA = data_df
        print(f"✅ 全文献データ ({len(data_df)}件) をインメモリキャッシュにロードしました。")

    except FileNotFoundError:
        print(f"❌ エラー: データファイル '{DATA_PATH}' が見つかりません。")
        GLOBAL_ARTICLE_DATA = None
    except Exception as e:
        print(f"❌ データロード中の予期せぬエラー: {e}")
        GLOBAL_ARTICLE_DATA = None

load_global_assets()

# ==============================================================================
# IV. 予測エンドポイント
# ==============================================================================

# Fast API ルーター（main.py）からリクエストを受け取るエンドポイント
@app.route("/predict", methods=["POST"])
def predict():
    if BASE_MODEL is None or GLOBAL_ARTICLE_DATA is None:
        return jsonify({"error": "サーバーが初期化されていません (モデル/データ欠損)"}), 503

    req = request.json
    keywords = [k for k in req.get("keywords",) if k]

    if not keywords:
        return jsonify({"error": "少なくとも 1 つのキーワードを指定してください"}), 400

    data_df = GLOBAL_ARTICLE_DATA.copy() # キャッシュデータを変更しないようコピーを使用

    data_df['YearMonth'] = pd.to_datetime(data_df['YearMonth'], errors='coerce')
    data_df = data_df.dropna(subset=['YearMonth'])

    # --- モデルハイパーパラメータを取得 ---
    input_dim = 1
    output_dim = 1
    hidden_dim = GLOBAL_HYPERPARAMS["hidden_dim"]
    num_layers = 1
    dropout_rate = 0.2
    look_back = 12

    # ----------------------------------------------------
    # ステップ 1: 全文献数の予測 (ベースモデルの利用)
    # ----------------------------------------------------
    
    all_monthly_counts_series = data_df.groupby(data_df['YearMonth']).size()
    all_past_start_date = data_df['YearMonth'].min()
    all_past_end_date = data_df['YearMonth'].max()
    
    # 期間の欠損値を 0 で補完
    all_full_date_range = pd.date_range(start=all_past_start_date, end=all_past_end_date, freq='MS')
    all_monthly_counts_full = all_monthly_counts_series.reindex(all_full_date_range, fill_value=0)
    all_monthly_counts = all_monthly_counts_full.values
    
    # データをスケーリングし、予測の入力シーケンスを準備
    scaled_all = SCALER.transform(all_monthly_counts.reshape(-1, 1)).flatten()

    predictions_all_scaled = []
    look_back = 12
    current_input_all = torch.tensor(scaled_all[-look_back:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    
    # 今後 12 期間の予測を実行 (再帰的予測)
    for _ in range(12):
        with torch.no_grad():
            pred = BASE_MODEL(current_input_all).item()
            pred = max(0, pred) # 予測値が負にならないようにクリップ
            predictions_all_scaled.append(pred)
            
            # 次の予測のために、入力シーケンスを更新
            current_input_all = torch.roll(current_input_all, -1, 1)
            current_input_all[:, -1, :] = pred
    
    # スケーリングを元に戻し、結果をリスト化
    all_predictions_unscaled = SCALER.inverse_transform(np.array(predictions_all_scaled).reshape(-1,1)).flatten().tolist()
    
    # ----------------------------------------------------
    # ステップ 2: キーワード文献のフィルタリング
    # ----------------------------------------------------

    def contains_all_keywords(text):
        return all(k.lower() in text.lower() for k in keywords)

    # 検索テキスト全体を対象にフィルタリング
    df_filtered = data_df[data_df['search_text'].apply(contains_all_keywords)].copy()
    
    # 時系列化の処理
    keyword_past_start_date = data_df['YearMonth'].min() 
    keyword_past_end_date = data_df['YearMonth'].max() 

    # フィルタリング後の月次件数を集計
    monthly_counts_series = df_filtered.groupby(df_filtered['YearMonth']).size()
    full_date_range = pd.date_range(start=keyword_past_start_date, end=keyword_past_end_date, freq='MS')
    monthly_counts_full = monthly_counts_series.reindex(full_date_range, fill_value=0)
    monthly_counts = monthly_counts_full.values

    # ----------------------------------------------------
    # ステップ 3: ファインチューニングと予測
    # ----------------------------------------------------
    
    # ベースモデルをコピーしてファインチューニングモデルを作成
    finetune_model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers, dropout_rate)
    finetune_model.load_state_dict(BASE_MODEL.state_dict())

    # ファインチューニング層の設定 (LSTMの最初の層と最終層のみ更新)
    for name, param in finetune_model.named_parameters():
        if ('lstm.weight_hh_l0' in name or 'lstm.weight_ih_l0' in name or 'fc' in name):
            param.requires_grad = True # 更新を許可
        else:
            param.requires_grad = False # 重みを固定

    finetune_data = monthly_counts.reshape(-1, 1).astype(np.float32)
    
    # 注意: スケーラーは全体データで fit されているものをそのまま利用
    scaled_finetune_data = SCALER.transform(finetune_data).flatten()

    X_finetune, y_finetune = [], [],
    if len(scaled_finetune_data) >= look_back + 1:
        # ファインチューニング用データセットの作成
        for i in range(len(scaled_finetune_data) - look_back):
            X_finetune.append(scaled_finetune_data[i:i+look_back])
            y_finetune.append(scaled_finetune_data[i+look_back])
        
        if X_finetune:
            X_finetune_tensor = torch.tensor(np.array(X_finetune), dtype=torch.float32).unsqueeze(-1)
            y_finetune_tensor = torch.tensor(np.array(y_finetune), dtype=torch.float32).unsqueeze(-1)

            # ファインチューニングの実行
            criterion = nn.MSELoss()
            optimizer = optim.Adam(finetune_model.parameters(), lr=0.001)
            finetune_epochs = 100
            
            finetune_model.train()
            # 訓練ステップ (リクエスト処理中のオンデマンド訓練)
            for epoch in range(finetune_epochs): 
                optimizer.zero_grad()
                outputs = finetune_model(X_finetune_tensor)
                loss = criterion(outputs, y_finetune_tensor)
                loss.backward()
                optimizer.step()

    # ----------------------------------------------------
    # ステップ 4: キーワード予測とスケーリング調整
    # ----------------------------------------------------

    finetune_model.eval()
    predictions_finetune = []

    if scaled_finetune_data.size > 0 and X_finetune:
        # 最新のデータポイントを入力として使用
        current_input_finetune = torch.tensor(scaled_finetune_data[-look_back:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        for _ in range(12):
            with torch.no_grad():
                pred = finetune_model(current_input_finetune).item()
                pred = max(0, pred)
                predictions_finetune.append(pred)
                current_input_finetune = torch.roll(current_input_finetune, -1, 1)
                current_input_finetune[:, -1, :] = pred

        finetune_predictions_unscaled = SCALER.inverse_transform(np.array(predictions_finetune).reshape(-1,1)).flatten().tolist()
    else:
        # ファインチューニングデータが不足している場合は予測をゼロにする
        finetune_predictions_unscaled = [0] * 12

    # --- ヒューリスティックなスケーリング調整 ---
    # 注: 予測精度が低い場合の見た目の安定化のために行われる調整。
    if len(monthly_counts) > 0:
        past_avg = np.mean(monthly_counts)
        pred_avg = np.mean(finetune_predictions_unscaled)
        scaling_factor = past_avg / pred_avg if pred_avg > 0 else 0
    else:
        scaling_factor = 0
        
    adjusted_predictions = [p * scaling_factor for p in finetune_predictions_unscaled]
    
    # ----------------------------------------------------
    # ステップ 5: 結果の返却
    # ----------------------------------------------------

    return jsonify({
        "keywords": keywords,
        "extracted_articles_count": len(df_filtered),
        "all_past_literature_counts": all_monthly_counts.tolist(),
        "all_predicted_literature_counts": [round(p) for p in all_predictions_unscaled],
        "all_past_start_date": all_past_start_date.strftime("%Y-%m-%d"),
        "all_past_end_date": all_past_end_date.strftime("%Y-%m-%d"),
        "keyword_past_literature_counts": monthly_counts.tolist(),
        "keyword_predicted_literature_counts": adjusted_predictions,
        "keyword_past_start_date": keyword_past_start_date.strftime("%Y-%m-%d"),
        "keyword_past_end_date": keyword_past_end_date.strftime("%Y-%m-%d"),
    })

# 注意: プロダクション環境では、以下の開発用サーバー起動コードは使用せず、
# Gunicorn/Uvicorn サーバーを通じて pt_worker:app を起動します。
# if __name__ == "__main__":
#     app.run(port=5002)