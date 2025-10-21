# 医療AI文献トレンド分析アプリ

## 目的
過去のPubMed文献データを分析し、医療×AI分野の文献トレンドや注目トピックを可視化。  
プロダクトやサービス開発のヒントを得ることを目的としています。

> 注：PubMedは米国国立医学図書館（NLM）が運営する医学・生物学分野の学術文献データベースです。

## 概要
- PubMed APIで過去5年間の医療AI関連文献数を取得
- LSTMモデルで翌年の文献数を予測・可視化
- StreamlitでUIを構築、FastAPI経由でFlaskにモデル推論を依頼
- TF-IDFとMeSHを用いたトレンドワード分析機能も計画中

## 主な機能
- キーワード（最大3つ）入力による文献数推移グラフ表示
- LSTMによる翌年の文献数予測
- 【開発中】トレンドワード解析（TF-IDF/MeSH）

## 使用技術
Python、Streamlit、FastAPI、Flask、PyTorch（LSTM）、PubMed E-utilities API、scikit-learn（TF-IDF）

---

## データ取得と前処理
- **データソース**：NCBI E-utilities APIでPubMed文献を検索
- **工夫**
  - APIの取得件数制限やレート制限を回避するため、期間を半年または3ヶ月単位に分割
  - リクエスト間に遅延 (`time.sleep`) を設けレート制限違反を防止
- **取得情報**：PMID、タイトル、抄録、MeSH Term、掲載年月

---

## モデルと予測
- **ベースモデル**：LSTMで全文献数の時系列を学習
- **ファインチューニング**：ユーザー入力キーワードに合致する文献時系列に基づき、モデルの一部（LSTM層と最終FC層）をオンデマンドで調整

---

## セットアップ（ローカル）
1. 仮想環境作成
2. `requirements.txt`で必要ライブラリをインストール
3. `.env`にPubMed APIキー等を設定
4. データ取得〜ベースモデル学習を実施
   - `data_acquisition.py` → `model_training.py` の順に実行
5. `start_app.sh`でアプリ起動（ローカルホスト）

### Colabでの実行手順
1. 必要パッケージをインストール
2. `data_acquisition.py` と `model_training.py` のコードをセルにコピー
3. コード内で NCBI_API_KEY に取得済みキーを直接入力
4. コードを順に実行
5. 出力ファイルをローカルにコピー
   - CSV：プロジェクトのルート直下
   - モデル：model/ フォルダ直下

---

## アーキテクチャ
Streamlit UI → FastAPI（APIルーター）→ Flask（LSTM推論）
PubMed API → データ取得

---

## フォルダ構成例
```text
med_ai_trend_app/
├─ app/
│   ├─ ui.py           # Streamlit UI
│   ├─ main.py         # FastAPIエンドポイント
│   ├─ pt_worker.py    # Flask推論ワーカー
│   └─ mesh_trend.py   # トレンドワード解析
├─ data_acquisition.py # PubMedデータ取得
├─ model_training.py   # LSTMモデル学習
├─ model/
│   ├─ lstm_model_state.pth
│   ├─ scaler.pkl
│   └─ hyperparams.json
├─ pubmed_articles_details.csv
├─ requirements.txt
├─ start_app.sh
└─ .env
```

---

## 現在の課題と今後の展開
- 過去5年間、月次約60点のデータで予測するため、モデル予測が平坦化する場合あり
- 今後の改善案：
  - データ取得範囲拡大（期間延長や関連用語追加）
  - より多様なモデル（統計モデルやTransformer系）の検討
  - トレンドワード解析や国・疾患別傾向分析





