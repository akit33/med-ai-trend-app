# app/mesh_trend.py (トレンド分析ロジック)

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from collections import Counter

# ==============================================================================
# I. 設定値とストップワード
# ==============================================================================

# NLTK ダウンロードチェック (一度実行すれば次回はスキップされます)
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
    
# 英語の基本ストップワードに加え、医療文献で高頻度に出現する一般単語を除去
# これにより、より専門的なトレンド語を抽出できます。
EN_STOPWORDS = set(stopwords.words("english")) | {
    "study", "clinical", "model", "patients", "methods", "results", "conclusions"
}

# ==============================================================================
# II. データロードと MeSH 出現率の計算
# ==============================================================================

def load_and_preprocess(csv_path: str) -> tuple:
    """
    CSV を読み込み、月ごとの MeSH 用語の出現率を計算します。
    
    Args:
        csv_path (str): ルートディレクトリにあるデータファイルパス (e.g., '../pubmed_articles_details.csv')
    
    Returns:
        tuple: 元のデータフレームと MeSH 使用率の時系列データ
    """
    df = pd.read_csv(csv_path)
    df = df.fillna("")
    
    # MeSH タームをリスト化
    df["MeSH_list"] = df["MeSH Terms"].apply(
        lambda x: [t.strip() for t in x.split(";") if t.strip()]
    )
    # 時系列処理に備え、YearMonth をタイムスタンプに変換
    df["YearMonth"] = pd.to_datetime(df["YearMonth"]) 
    
    # 月ごとの総文献数を計算 (分母)
    total_docs = df.groupby("YearMonth").count()
    
    # 用語ごとの出現率を計算し、時系列データ (usage) を作成
    all_terms = set(sum(df["MeSH_list"].tolist(), []))
    usage = pd.DataFrame(0.0, index=sorted(all_terms), columns=sorted(total_docs.index))
    
    for ym, group in df.groupby("YearMonth"):
        counts = Counter(t for terms in group["MeSH_list"] for t in terms)
        for t, c in counts.items():
            # 出現率 = (用語の出現回数) / (その月の総文献数)
            usage.loc[t, ym] = c / total_docs.loc[ym]
            
    return df, usage

# ==============================================================================
# III. MeSH 用語の伸び率計算
# ==============================================================================

def compute_growth_metrics(usage_rate: pd.DataFrame, last_n: int = 12) -> pd.DataFrame:
    """
    MeSH 用語の時系列データに対し、絶対増加量、相対増加率、線形トレンドの傾きを計算します。
    直近 last_n ヶ月とその前の last_n ヶ月を比較します。
    """
    cols = sorted(usage_rate.columns)
    
    # データ期間が短い場合の調整
    if len(cols) < last_n * 2:
        last_n = len(cols) // 2
        if last_n == 0:
            return pd.DataFrame() # データが少なすぎる場合は空のDataFrameを返す

    recent = cols[-last_n:]
    prev = cols[-2 * last_n : -last_n]
    
    # 平均使用率の計算
    recent_mean = usage_rate[recent].mean(axis=1)
    prev_mean = usage_rate[prev].mean(axis=1)
    
    # 増加指標
    abs_increase = recent_mean - prev_mean
    # 相対増加率: ゼロ除算を避ける処理を導入
    rel_increase = (abs_increase / prev_mean.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )
    
    # 線形回帰による傾き (トレンド) の計算
    slopes = []
    X = np.arange(len(cols)).reshape(-1, 1) # 時系列インデックス
    for term in usage_rate.index:
        y = usage_rate.loc[term, cols].values
        if np.all(y == 0):
            slopes.append(0.0) # 全期間ゼロの場合は傾きもゼロ
        else:
            # 線形回帰モデルを適用
            lr = LinearRegression().fit(X, y)
            slopes.append(lr.coef_)
            
    out = pd.DataFrame(
        {
            "term": usage_rate.index,
            "recent_mean": recent_mean,
            "prev_mean": prev_mean,
            "abs_increase": abs_increase,
            "rel_increase": rel_increase,
            "slope": slopes,
        }
    )
    # 絶対増加量に基づきランキング
    return out.sort_values("abs_increase", ascending=False)

# ==============================================================================
# IV. クラスタリングと TF-IDF
# ==============================================================================

def cluster_terms(usage: pd.DataFrame, n_clusters: int = 6) -> pd.DataFrame:
    """
    MeSH 用語の時系列使用率パターンに基づき、KMeans でクラスタリングを実行します。
    これにより、成長パターンが類似した用語グループを特定します。
    """
    if len(usage) < n_clusters:
        n_clusters = len(usage)
        
    scaler = StandardScaler()
    X = scaler.fit_transform(usage.fillna(0).T) # 時系列を特徴量として標準化
    
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    labels = km.fit_predict(X)
    
    # クラスタリングは行 (時系列) ではなく列 (用語) に対して行う
    return pd.DataFrame({"term": usage.index, "cluster": labels})


def compute_keyword_growth_tfidf(df: pd.DataFrame, last_n: int = 12) -> pd.DataFrame:
    """
    タイトル＋抄録の TF-IDF ベース頻出語ランキングを作成し、MeSH 用語でフィルタリングします。
    """
    # タイトル＋抄録を結合して分析対象テキストを作成
    df["text"] = df["Title"].fillna("") + " " + df["Abstract"].fillna("")
    df["YearMonth"] = pd.to_datetime(df["YearMonth"])
    corpus = df["text"].tolist()
    
    # TF-IDF ベクトル化 (1-2 gram、ストップワード除去)
    vectorizer = TfidfVectorizer(
        stop_words=list(EN_STOPWORDS),
        ngram_range=(1, 2),
        max_features=500 # 上位 500 語に限定
    )
    X = vectorizer.fit_transform(corpus)
    terms = np.array(vectorizer.get_feature_names_out())
    
    # MeSH リストとのクロス参照
    mesh_terms = set(sum(df["MeSH_list"].tolist(), []))
    # TF-IDF 頻出語のうち、MeSH 用語リストと関連性の高い用語のみを抽出
    filtered_terms = [t for t in terms if any(mt.lower() in t.lower() for mt in mesh_terms)]
    
    # 月ごとの出現率を再計算
    docs_by_month = df.groupby("YearMonth")["text"].apply(list)
    usage = pd.DataFrame(0.0, index=filtered_terms, columns=sorted(docs_by_month.index))
    
    for ym, texts in docs_by_month.items():
        total_docs = len(texts)
        for term in filtered_terms:
            # 出現回数を数える
            count = sum(1 for text in texts if term in text.lower())
            usage.loc[term, ym] = count / total_docs
            
    # MeSH 成長指標と同じロジックで伸び率を計算
    cols = sorted(usage.columns)
    if len(cols) < last_n * 2:
        last_n = len(cols) // 2
        if last_n == 0:
            return pd.DataFrame()
    
    recent = cols[-last_n:]
    prev = cols[-2 * last_n : -last_n]
    recent_mean = usage[recent].mean(axis=1)
    prev_mean = usage[prev].mean(axis=1)
    abs_increase = recent_mean - prev_mean
    
    keyword_growth = pd.DataFrame({
        "recent_mean": recent_mean,
        "prev_mean": prev_mean,
        "abs_increase": abs_increase
    }).sort_values("abs_increase", ascending=False)
    
    return keyword_growth

# ==============================================================================
# V. メインエントリポイント (Streamlit からの呼び出し)
# ==============================================================================

def get_top_terms(csv_path: str, top_n: int = 20, last_n: int = 12):
    """
    トレンド分析機能の統合関数。Streamlit (ui.py) から呼び出されます。
    """
    df, usage = load_and_preprocess(csv_path)
    
    # 伸び率指標、クラスタリング、TF-IDF 成長指標をそれぞれ計算
    growth = compute_growth_metrics(usage, last_n=last_n)
    clusters = cluster_terms(usage.T) # 時系列をクラスタリングするために転置
    keyword_growth = compute_keyword_growth_tfidf(df, last_n=last_n)
    
    # 使用率時系列データ (usage) はデバッグや詳細分析用に含める
    return growth, clusters, usage, keyword_growth