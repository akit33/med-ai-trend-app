import requests
import pandas as pd
import time
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from xml.etree import ElementTree as ET
import os
from dotenv import load_dotenv

# ==============================================================================
# I. 設定値と定数の定義 (NCBI E-utilities API 関連)
# ==============================================================================

load_dotenv() #.env ファイルから環境変数をロード

# API_KEY を環境変数から取得
API_KEY = os.getenv("NCBI_API_KEY", "DEFAULT_IF_NOT_FOUND") 
# もし環境変数が見つからない場合に備えて、デフォルト値を設定することも可能

# PMID 検索エンドポイント
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
# 詳細情報取得エンドポイント
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# 論文のスコープを定義する複合検索クエリ
# 医療 AI 関連用語（AI, ML, DL, NLP）と医学関連用語（Medicine, Diagnosis, Genomics, Digital Healthなど）を MeSH/Title/Abstractで組み合わせて検索範囲を限定
SEARCH_TERM = '("Artificial Intelligence" OR "Machine Learning" OR "Deep Learning" OR "Natural Language Processing" OR "Computer Vision" OR "Neural Networks") AND ("Medicine" OR "Medical Informatics" OR "Diagnosis" OR "Personalized Medicine" OR "Medical Imaging" OR "Genomics" OR "Preventive Medicine" OR "Health Promotion" OR "Digital Health")'

RETMAX = 10000  # 一度のリクエストで取得する最大件数
PMID_CHUNK_SIZE = 200  # 詳細情報取得時 (efetch) の推奨チャンクサイズ
DELAY_SECONDS = 0.3  # NCBI のレート制限遵守のための遅延時間 (0.2秒～0.5秒推奨)

# ==============================================================================
# II. PMID 取得関数 (esearch)
# ==============================================================================

def get_pmids(start_date: date, end_date: date) -> tuple[int | None, list[str] | None]:
    """
    指定された期間の PubMed ID (PMID) を取得します。
    API のレート制限やエラーに対応するための処理を含みます。
    """
    mindate_str = start_date.strftime("%Y/%m/%d")
    maxdate_str = end_date.strftime("%Y/%m/%d")

    params = {
        "db": "pubmed",
        "term": SEARCH_TERM,
        "datetype": "pdat",  # Publication Date (PMID登録日) でフィルタリング
        "mindate": mindate_str,
        "maxdate": maxdate_str,
        "retmax": RETMAX,
        "retstart": 0,
        "retmode": "json",
        "api_key": API_KEY
    }

    print(f"  > 検索期間: {mindate_str} to {maxdate_str}")
    try:
        response = requests.get(ESEARCH_URL, params=params)
        response.raise_for_status()  # HTTP エラー (4xx, 5xx) を検出
        data = response.json()
        result = data.get("esearchresult", {})

        count = int(result.get("count", 0))
        pmids = result.get("idlist",)

        # データエンジニアリング的なロバスト性のチェック
        if count >= RETMAX:
            print(f" : 取得件数 ({count}) が RETMAX に達しました。取得漏れを防ぐため、期間の再分割を検討してください。")

        return count, pmids

    except requests.exceptions.RequestException as e:
        # ネットワークエラーや API タイムアウト時のエラーハンドリング
        print(f" : リクエストに失敗しました: {e}")
        return None, None

# ==============================================================================
# III. 詳細情報取得関数 (efetch)
# ==============================================================================

def fetch_article_details(pmid_list: list[str], output_csv_path: str = "pubmed_articles_details.csv"):
    """
    PMID リストに基づき、論文の詳細情報（タイトル、抄録、MeSH タームなど）を取得し、CSVに保存します。
    NCBI の推奨に基づき、PMID をチャンクに分割して処理します。
    """
    all_article_data =

    # PMID リストをチャンクに分割して処理
    for i in range(0, len(pmid_list), PMID_CHUNK_SIZE):
        chunk = pmid_list
        pmids_str = ",".join(map(str, chunk))

        # API リクエストのパラメータ
        params = {
            "db": "pubmed",
            "retmode": "xml",
            "id": pmids_str,
            "api_key": API_KEY
        }

        print(f"  > 詳細取得中: {i+1} to {i + len(chunk)} 件目...")
        try:
            # 大量のデータを送信するため POST メソッドを使用
            response = requests.post(EFETCH_URL, data=params, timeout=10)
            response.raise_for_status()

            root = ET.fromstring(response.content)

            # 各論文の情報を XML から抽出
            for article in root.findall(".//PubmedArticle"):
                # PMID
                pmid_node = article.find(".//MedlineCitation/PMID")
                pmid = pmid_node.text if pmid_node is not None else "N/A"
                # タイトル
                title_node = article.find(".//ArticleTitle")
                title = title_node.text if title_node is not None else "N/A"
                
                # 抄録（複数の AbstractText タグに対応）
                abstract_node = article.find(".//Abstract")
                abstract_parts = if abstract_node is not None else
                abstract = " ".join(abstract_parts)
                
                # MeSH ターム
                mesh_terms =
                mesh_list_node = article.find(".//MeshHeadingList")
                if mesh_list_node is not None:
                    for mesh_heading in mesh_list_node.findall("MeshHeading"):
                        descriptor_name = mesh_heading.find("DescriptorName")
                        if descriptor_name is not None:
                            mesh_terms.append(descriptor_name.text)
                mesh_terms_str = "; ".join(mesh_terms) if mesh_terms else "N/A"

                # 出版日 (PubMed に登録された日付を取得: PubStatus='pubmed')
                pub_date_node = article.find(".//PubmedData/History/PubMedPubDate")
                year, month = "N/A", "N/A"
                if pub_date_node is not None:
                    year_node = pub_date_node.find("Year")
                    month_node = pub_date_node.find("Month")
                    year = year_node.text if year_node is not None else "N/A"
                    month = month_node.text if month_node is not None else "N/A"
                
                all_article_data.append({
                    "PMID": pmid,
                    "Title": title,
                    "Abstract": abstract,
                    "MeSH Terms": mesh_terms_str,
                    "Year": year,
                    "Month": month
                })

        except requests.exceptions.RequestException as e:
            print(f" : 詳細取得リクエストに失敗しました: {e}. 現在のチャンクで処理を中断します。")
            break
        except ET.ParseError as e:
            print(f" : XMLパースに失敗しました: {e}. API から無効なデータが返された可能性があります。")
            break

        # 次のリクエストまでの待機 (NCBI レート制限対策)
        time.sleep(DELAY_SECONDS)

    # データを DataFrame に変換し、不要な行を削除して保存
    df_articles = pd.DataFrame(all_article_data)
    # 年月情報のない行は、時系列データとして利用できないため除外
    df_articles_clean = df_articles.dropna(subset=)
    df_articles_clean = df_articles_clean.isin(['N/A']) & ~df_articles_clean['Month'].isin(['N/A'])]
    
    # Year/Month の結合 (予測ワーカーでの利用のため)
    df_articles_clean = df_articles_clean.astype(str) + '-' + df_articles_clean['Month'].astype(str)
    
    df_articles_clean.to_csv(output_csv_path, index=False)
    print(f"\n: 論文詳細 {len(df_articles_clean)} 件を {output_csv_path} に保存しました。")


# ==============================================================================
# IV. メイン実行ロジック
# ==============================================================================

def run_data_acquisition():
    """
    データ取得のメインフローを実行します。
    """
    all_pmids =
    
    # ---  期間 1: 2020/01/01 から 2023/12/31 を半年ごとに処理（ボリュームが大きい期間） ---
    # データ量の増加傾向を考慮し、期間を分割して処理することで RETMAX 超過を防ぐ
    print("--- ステップ 1: 2020-2023 年の PMID を半年ごとに取得 ---")
    current_date = date(2020, 1, 1)
    end_of_period = date(2023, 12, 31)
    
    while current_date <= end_of_period:
        next_date = current_date + relativedelta(months=6) - timedelta(days=1)
        if next_date > end_of_period:
            next_date = end_of_period
        
        count, pmids = get_pmids(current_date, next_date)
        if pmids is not None:
            all_pmids.extend(pmids)
            print(f"  - 取得済み PMID 合計: {len(all_pmids)}")
        
        current_date = next_date + timedelta(days=1)
        time.sleep(DELAY_SECONDS)


    # ---  期間 2: 2024 年 1 月 1 日 以降を 3 ヶ月ごとに処理（より細かく分割） ---
    # データ量が急増する可能性に対応するため、より短い期間で分割
    print("\n--- ステップ 2: 2024 年以降の PMID を 3 ヶ月ごとに取得 ---")
    current_date = date(2024, 1, 1)
    # この日付は実行時に合わせて調整してください
    end_of_data = date(2025, 8, 28) 
    
    while current_date <= end_of_data:
        next_date = current_date + relativedelta(months=3) - timedelta(days=1)
        if next_date > end_of_data:
            next_date = end_of_data
        
        count, pmids = get_pmids(current_date, next_date)
        if pmids is not None:
            all_pmids.extend(pmids)
            print(f"  - 取得済み PMID 合計: {len(all_pmids)}")
        
        current_date = next_date + timedelta(days=1)
        time.sleep(DELAY_SECONDS)

    # 重複の削除と PMID の保存
    unique_pmids = sorted(list(set(all_pmids)))
    print(f"\n--- ステップ 3: 論文詳細の取得準備 ---")
    print(f"収集されたユニークな PMID 総数: {len(unique_pmids)}")
    
    # 中間ファイルとして PMID リストを保存 (データセットのバージョン管理に利用可能)
    pmid_list_csv = "all_pmids_reprocessed.csv"
    df_pmids = pd.DataFrame(unique_pmids, columns=)
    df_pmids.to_csv(pmid_list_csv, index=False)
    print(f"中間ファイル {pmid_list_csv} に PMID リストを保存しました。")

    # 論文詳細の取得を実行
    fetch_article_details(unique_pmids, output_csv_path="pubmed_articles_details.csv")
    
if __name__ == "__main__":
    run_data_acquisition()