# app/ui.py (Streamlit UI - フロントエンド)

import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from app.mesh_trend import get_top_terms 
import os
from dotenv import load_dotenv

# ==============================================================================
# I. 初期設定とパス定義
# ==============================================================================

load_dotenv(dotenv_path='../.env') # 環境変数をロード

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "../pubmed_articles_details.csv")

# FastAPI の URL を環境変数から取得 (デフォルト値はローカルホスト)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

# ====  サイドバー: ナビゲーション  ====
st.sidebar.title("医療 AI 文献トレンド分析アプリ")

# セッションステートにモードを保存
if "mode" not in st.session_state:
    st.session_state["mode"] = "文献数予測"

# ページ切り替えボタン
if st.sidebar.button("📈 文献数予測 (時系列予測)"):
    st.session_state["mode"] = "文献数予測"
if st.sidebar.button("🔍 トレンドワード (クラスタリング/TF-IDF)"):
    st.session_state["mode"] = "トレンドワード"
    
mode = st.session_state["mode"]

# ==============================================================================
# II. 文献数予測モード (時系列予測の結果表示)
# ==============================================================================

if mode == "文献数予測":
    st.header("📈 医療 AI 文献数予測")
    st.markdown("任意のキーワードを入力し、ベースモデルをファインチューニングした**特定のトレンド**と**医療 AI 全体**の文献数の推移を比較します。")
    
    # キーワード入力を 1 行に並べる (UXの向上)
    st.write("任意のキーワードを最大 3 つ入力してください")
    col1, col2, col3 = st.columns(3)
    with col1:
        keyword1 = st.text_input("キーワード 1", label_visibility="collapsed", placeholder="キーワード 1")
    with col2:
        keyword2 = st.text_input("キーワード 2", label_visibility="collapsed", placeholder="キーワード 2")
    with col3:
        keyword3 = st.text_input("キーワード 3", label_visibility="collapsed", placeholder="キーワード 3")
        
    keywords = [k for k in [keyword1, keyword2, keyword3] if k]

    if st.button("予測実行"):
        if not keywords:
            st.warning("🚨 少なくとも 1 つのキーワードを入力してください。")
        else:
            with st.spinner("⏳ 予測モデルをファインチューニング中..."):
                try:
                    # FastAPI ルーターにリクエスト送信
                    response = requests.post(API_URL, json={"keywords": keywords})
                    response.raise_for_status() # 4xx/5xx エラーを捕捉
                    result = response.json()
                    
                    st.success("✅ 予測が完了しました！")
                    
                    # ---  データの抽出  ---
                    all_past_counts = result["all_past_literature_counts"]
                    all_predicted_counts = result["all_predicted_literature_counts"]
                    all_past_end_date_str = result["all_past_end_date"]
                    keyword_past_counts = result["keyword_past_literature_counts"]
                    keyword_predicted_counts = result["keyword_predicted_literature_counts"]
                    keyword_past_end_date_str = result["keyword_past_end_date"]
                    
                    # ---  日付範囲の生成 (可視化のために必須)  ---
                    # 予測期間の日付範囲を正確に作成
                    def generate_dates(past_end_date_str, past_counts, predicted_counts):
                        past_dates = pd.date_range(start=pd.to_datetime(past_end_date_str) - pd.DateOffset(months=len(past_counts)-1), end=past_end_date_str, freq="MS")
                        predicted_start_date = pd.to_datetime(past_end_date_str) + pd.DateOffset(months=1)
                        predicted_dates = pd.date_range(start=predicted_start_date, periods=len(predicted_counts), freq="MS")
                        return past_dates, predicted_dates

                    keyword_past_dates, keyword_predicted_dates = generate_dates(
                        keyword_past_end_date_str, keyword_past_counts, keyword_predicted_counts)
                    all_past_dates, all_predicted_dates = generate_dates(
                        all_past_end_date_str, all_past_counts, all_predicted_counts)
                    
                    # --- 1 つの Plotly Figure に統合  ---
                    st.markdown("<h4 style='margin-bottom:5px;'>文献掲載数の推移 (実績と予測)</h4>", unsafe_allow_html=True)
                    fig = go.Figure()
                    
                    # キーワード系列 (実績 - 実線 / 予測 - 破線)
                    fig.add_trace(go.Scatter(
                        x=keyword_past_dates, y=keyword_past_counts,
                        mode='lines+markers', name=f'キーワード: {", ".join(keywords)} (実績)', line=dict(color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        x=keyword_predicted_dates, y=keyword_predicted_counts,
                        mode='lines+markers', name='キーワード (予測)', line=dict(color='red', dash='dash') # 予測は破線で明確化
                    ))
                    
                    # 全文献系列 (ベースライン)
                    fig.add_trace(go.Scatter(
                        x=all_past_dates, y=all_past_counts,
                        mode='lines', name='医療 AI 文献全体 (実績)', line=dict(color='gray')
                    ))
                    fig.add_trace(go.Scatter(
                        x=all_predicted_dates, y=all_predicted_counts,
                        mode='lines', name='医療 AI 文献全体 (予測)', line=dict(color='orange', dash='dot') # 予測は点線で明確化
                    ))

                    fig.update_layout(
                        xaxis_title='年月',
                        yaxis_title='掲載文献数',
                        template='plotly_white',
                        legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0)'),
                        margin=dict(t=30, l=50, r=30, b=50)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # ---  サマリーと予測値のテーブル  ---
                    extracted_articles_count = result["extracted_articles_count"]
                    st.markdown(
                        f"<h4 style='margin-bottom:10px;'>キーワードに合致する文献の過去掲載数: {extracted_articles_count} 件 </h4>",
                        unsafe_allow_html=True
                    )
                    
                    df_pred = pd.DataFrame([keyword_predicted_counts],
                                            columns=[d.strftime("%Y-%m") for d in keyword_predicted_dates],
                                            index=["掲載数"])
                    df_pred = df_pred.round(0).astype(int)
                    
                    st.markdown("キーワード文献の予測値（月ごと）", unsafe_allow_html=True)
                    st.dataframe(df_pred.T) # 縦向きに変換して Streamlit の標準テーブルで表示
                    
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ API接続エラーが発生しました。バックエンドサービス (FastAPI/pt_worker) が起動しているか確認してください: {e}")
                except Exception as e:
                    st.error(f"❌ データの処理中に予期せぬエラーが発生しました: {e}")

# ==============================================================================
# III. トレンドワードモード (分析結果の表示)
# ==============================================================================

elif mode == "トレンドワード":
    st.header("🔍 トレンドワード解析")
    st.info("MeSH 用語の使用率の伸び、およびタイトル・抄録の TF-IDF 頻出語を解析します。")
    
    if not os.path.exists(CSV_PATH):
        st.error(f"🚨 データファイルが見つかりません: {CSV_PATH}。データ取得ステップを完了してください。")
    else:
        # ====  パラメータ設定  ====
        # 表示数をユーザーが変更可能にする
        top_n = st.slider("表示する上位語数", 10, 50, 20, step=5)
        last_n = st.slider("比較する月数 (直近 N ヶ月 vs その前 N ヶ月)", 6, 24, 12, step=6)
        
        # ====  表示ボタン  ====
        if st.button("解析実行"):
            with st.spinner("⏳ 大規模テキスト分析とクラスタリングを実行中..."):
                try:
                    # ルートディレクトリの CSV パスを渡す
                    growth, clusters, usage, keyword_growth = get_top_terms(
                        CSV_PATH, top_n=top_n, last_n=last_n
                    )
                    
                    st.success("✅ 解析が完了しました！")

                    # 1. MeSH 用語の伸び率ランキング
                    st.subheader("1. 伸び率ランキング (MeSH Terms)")
                    st.markdown("直近期間の使用率の**絶対増加量**が大きい順にランキングしました。")
                    
                    # 必要な列のみを整理して表示
                    growth_display = growth.head(top_n).reset_index().rename(
                        columns={'index': 'MeSH ターム', 'abs_increase': '絶対増加量', 'rel_increase': '相対増加率', 'slope': '線形トレンド傾き'}
                    )
                    st.dataframe(growth_display.style.format(precision=4))
                    
                    # 2. クラスタリング結果
                    st.subheader(f"2. MeSH 用語の KMeans クラスタリング (クラスター数: {len(clusters['cluster'].unique())})")
                    st.markdown("用語の使用率パターンを時系列で分類しました。")
                    
                    # クラスタリング結果と平均伸び率を結合して表示
                    clustered_growth = pd.merge(growth, clusters, on='term').reset_index(drop=True)
                    clustered_growth_display = clustered_growth[['term', 'cluster', 'abs_increase', 'rel_increase']].sort_values('cluster')
                    st.dataframe(clustered_growth_display.rename(columns={'term': 'MeSH ターム', 'cluster': 'クラスターID'}))
                    
                    # 3. TF-IDF 頻出語ランキング
                    st.subheader("3. タイトル・抄録 TF-IDF 頻出語ランキング")
                    st.markdown("MeSH 用語リストとクロス参照し、医療 AI 関連度の高い頻出語を抽出しています。")
                    
                    keyword_growth_display = keyword_growth.head(top_n).reset_index().rename(
                        columns={'index': 'キーワード', 'abs_increase': '絶対増加量', 'recent_mean': '直近平均使用率'}
                    )
                    st.dataframe(keyword_growth_display.style.format(precision=4))

                except Exception as e:
                    st.error(f"❌ 解析中にエラーが発生しました: {e}")