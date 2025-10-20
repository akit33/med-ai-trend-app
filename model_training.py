# model_training.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os
import joblib
import json
from dotenv import load_dotenv

# GPU が使える場合は使用（Colab環境での訓練を想定）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

load_dotenv()

# ==============================================================================
# I. LSTM モデル定義
# ==============================================================================

class LSTMModel(nn.Module):
    """
    時系列予測用の LSTM モデル構造。
    訓練と予測ワーカー（pt_worker.py）で共通の定義を使用します。
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 1, dropout_rate: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # batch_first=True: 入力テンソルが (batch, sequence, feature) 形式であることを指定
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 隠れ状態 h0 とセル状態 c0 をゼロで初期化
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        
        # LSTM 層の実行
        # out: 全てのタイムステップの出力, _: 最終的な隠れ状態とセル状態
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        
        # 最後のタイムステップの出力のみを全結合層に入力
        out = self.fc(out[:, -1, :])
        return out

# ==============================================================================
# II. データ前処理
# ==============================================================================

def preprocess_for_base_model(file_path: str) -> np.ndarray:
    """
    論文詳細 CSV から月次の総文献数を集計し、時系列データとして整形します。
    """
    df = pd.read_csv(file_path, on_bad_lines='skip')
    
    # 年月を結合し、タイムスタンプに変換 (月の最初の日を設定)
    df = pd.to_datetime(df.astype(str) + '-' + df['Month'].astype(str) + '-01')
    
    # PeriodIndex を使用して月次の文献数を集計 (時系列データ化)
    # ここで NaN や N/A の行は既に除外されている前提
    monthly_counts = df.groupby(pd.PeriodIndex(df, freq='M')).size().reset_index(name='count')
    monthly_counts = monthly_counts.dt.to_timestamp()
    
    # モデル入力用の NumPy 配列 (形状: N x 1) として返す
    return monthly_counts[['count']].values  

def create_dataset(dataset: np.ndarray, look_back: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """
    時系列データを LSTM 入力形式（シーケンス）に変換します。
    look_back 期間のデータで次のタイムステップを予測する入出力ペアを作成。
    """
    X, y =,
    for i in range(len(dataset) - look_back):
        # X: 入力シーケンス (過去 look_back 期間)
        X.append(dataset[i:i+look_back])
        # y: ターゲット (次の期間の値)
        y.append(dataset[i+look_back])
    return np.array(X), np.array(y)

# ==============================================================================
# III. モデル評価と訓練
# ==============================================================================

def evaluate_and_find_best_hyperparameters(data: np.ndarray, look_back: int = 12) -> dict | None:
    """
    ローリングウィンドウ評価に基づき、LSTM の最適ハイパーパラメータをグリッドサーチで探索します。
    （注: 時系列データの評価では、ローリングウィンドウ評価はデータリーク防止の観点から推奨される手法です。）
    """
    param_grid = {
        'epochs': ,
        'lr': [0.001, 0.0005],
        'hidden_dim': 
    }
    
    best_avg_rmse = float('inf')
    best_params = {}
    
    # グリッドサーチの実行
    for epochs in param_grid['epochs']:
        for lr in param_grid['lr']:
            for hidden_dim in param_grid['hidden_dim']:
                print(f"\n--- 評価中: epochs={epochs}, lr={lr}, hidden_dim={hidden_dim} ---")
                all_rmses =
                
                # ローリングウィンドウ評価 (look_back の期間を訓練に、次の 1 期間をテストに使用)
                # 注: この実装では、look_back 期間のみでモデルを学習させるため、データが少ないと過学習リスクが高まります。
                for i in range(look_back, len(data)):
                    train_slice = data[i-look_back:i]
                    test_slice = data[i:i+1]
                    
                    # 各ウィンドウでスケーラーを再フィット（データリーク防止）
                    scaler = MinMaxScaler()
                    scaled_train = scaler.fit_transform(train_slice)
                    
                    # 3 次元テンソル化: (バッチサイズ=1, シーケンス長=look_back, 特徴量=1)
                    X_train_tensor = torch.from_numpy(scaled_train.reshape(1, look_back, 1)).float()
                    y_train_tensor = torch.from_numpy(scaled_train[-1:].reshape(1, 1)).float()

                    # モデル定義と訓練
                    model = LSTMModel(input_dim=1, hidden_dim=hidden_dim, output_dim=1).float()
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                    model.train()
                    for epoch in range epochs: # epochs: 訓練回数はハイパーパラメータ
                        # バッチが 1 のため DataLoader はシンプルに使用
                        optimizer.zero_grad()
                        output = model(X_train_tensor)
                        loss = criterion(output, y_train_tensor)
                        loss.backward()
                        optimizer.step()

                    # 予測と評価
                    model.eval()
                    with torch.no_grad():
                        pred_scaled = model(X_train_tensor).numpy()
                        # スケーリングを元に戻し、RMSE を計算（モデルの性能指標）
                        pred_unscaled = scaler.inverse_transform(pred_scaled)
                        rmse = np.sqrt(mean_squared_error(test_slice, pred_unscaled))
                        all_rmses.append(rmse)

                avg_rmse = np.mean(all_rmses)
                print(f"  平均 RMSE: {avg_rmse:.4f}")

                if avg_rmse < best_avg_rmse:
                    best_avg_rmse = avg_rmse
                    best_params = {'epochs': epochs, 'lr': lr, 'hidden_dim': hidden_dim}
                    print(f" : 最適なハイパーパラメータを更新しました。")

    print(f"\n: 最適なハイパーパラメータ: {best_params}, RMSE={best_avg_rmse:.4f}")
    return best_params if best_params else None

def train_final_model(data: np.ndarray, best_params: dict, look_back: int = 12, save_dir: str = "model"):
    """
    最適なハイパーパラメータを使用して全データでモデルを再訓練し、モデルアセットを保存します。
    （注: モデルアセットは GitHub には含めず、外部ストレージで管理されます。）
    """
    print("\n--- 最終モデル訓練 ---")
    
    # 最終モデル訓練では、全データに対してスケーラーをフィット
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    X_np, y_np = create_dataset(scaled_data, look_back)
    
    # テンソルへの変換と形状調整
    X_tensor = torch.from_numpy(X_np).float().unsqueeze(-1)  # (サンプル数, look_back, 1)
    y_tensor = torch.from_numpy(y_np).float().unsqueeze(-1)  # (サンプル数, 1, 1)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tensor, y_tensor),
        batch_size=1, # バッチサイズは 1
        shuffle=False
    )
    
    # 最適なハイパーパラメータでモデルを初期化
    model = LSTMModel(input_dim=1, hidden_dim=best_params['hidden_dim'], output_dim=1).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(best_params['epochs']):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

    # --- モデルアセットの保存 ---
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. モデルの重み (pth)
    torch.save(model.state_dict(), os.path.join(save_dir, "lstm_model_state.pth"))
    # 2. スケーラー (pkl)
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
    # 3. ハイパーパラメータ (json)
    with open(os.path.join(save_dir, "hyperparams.json"), "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"✅ 最終モデル・スケーラー・ハイパーパラメータを '{save_dir}/' に保存しました。")
    return model, scaler

# ==============================================================================
# IV. 実行エントリポイント
# ==============================================================================

if __name__ == '__main__':
    # 注意: ここに Colab 環境におけるデータパスを設定
    DATA_PATH = 'pubmed_articles_details.csv' 
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ エラー: 必要なデータファイル '{DATA_PATH}' が見つかりません。先にデータ取得を実行してください。")
    else:
        # 1. データ準備
        data = preprocess_for_base_model(DATA_PATH)
        
        # 2. ハイパーパラメータ探索
        best_params = evaluate_and_find_best_hyperparameters(data, look_back=12)
        
        if best_params:
            # 3. 最終モデル訓練と保存
            train_final_model(data, best_params, look_back=12, save_dir="model")
            print("\n最終モデル構築完了。モデルアセットを外部ストレージにアップロードしてください。")
        else:
            print("\n最適なハイパーパラメータが見つかりませんでした。モデル構築を中止します。")