import argparse
import base64
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Union

import google.generativeai as genai
import pandas as pd
import PyPDF2
import requests
from bs4 import BeautifulSoup

MODEL_NAMES = ["gemini-1.5-pro", "gemini-1.5-flash"]


def fetch_content_and_images(url: str) -> Dict[str, Union[str, List[Dict[str, str]]]]:
    """
    Fetch content and images from the specified URL. Supports PDF files as well.

    Args:
        url (str): The URL to fetch content from

    Returns:
        Dict[str, Union[str, List[str]]]: A dictionary containing text content and a list of image URLs

    Raises:
        requests.RequestException: If an error occurs during the request
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        if url.lower().endswith(".pdf"):
            # Process PDF file
            pdf_file = BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text()
            images = []  # Image extraction from PDF is complex, so we skip it here
        else:
            # Process HTML
            soup = BeautifulSoup(response.content, "html.parser")
            text_content = soup.get_text(strip=True)
            images = []
            for img in soup.find_all("img"):
                if "src" in img.attrs:
                    src = img["src"]
                    if src.startswith("data:image"):
                        images.append(
                            {
                                "mime_type": src.split(";")[0].split(":")[1],
                                "data": src.split(",")[1],
                            }
                        )
                    else:
                        if url.startswith("https://ar5iv.labs.arxiv.org/"):
                            full_url = "https://ar5iv.labs.arxiv.org/" + src
                        else:
                            full_url = src

                        try:
                            response = requests.get(full_url, timeout=10)
                            response.raise_for_status()
                            img_data = base64.b64encode(response.content).decode(
                                "utf-8"
                            )
                            extension = full_url.split(".")[-1]
                            mime_type = response.headers.get(
                                "Content-Type", f"image/{extension}"
                            )
                            images.append({"mime_type": mime_type, "data": img_data})
                        except requests.RequestException as e:
                            print(f"Error fetching image from {full_url}: {e}")

        return {"text": text_content, "images": images}
    except requests.RequestException as e:
        print(f"Error fetching content from {url}: {e}")
        return {"text": "", "images": []}


def analyze_content_and_images(
    content: str, images: List[Dict[str, str]], model_name: str = "gemini-1.5-pro"
) -> Dict[str, Union[str, List[str]]]:
    """
    Analyze content and images using the Gemini API.

    Args:
        content (str): The text content to analyze
        image_urls (List[str]): A list of image URLs to analyze

    Returns:
        Dict[str, Union[str, List[str]]]: A dictionary containing the analysis results
    """
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(model_name)

    prompt = """
    以下の論文/記事の内容と関連する図表を解析し、以下の形式で日本語で要約してください。
    後でパースするために、各項目についてのみ以下のように必ず<<<>>>で囲んで出力してください。

    <<<1. 論文/記事のタイトル>>>
    <<<2. 概要>>>
    最大5行で出力すること
    <<<3. 技術的なラベル>>>
    カンマ区切りで最大10個まで出力すること
    特に、レコメンドシステムに関するラベルがあれば優先して出力すること
        例: ABtest, Coldstart, Diversity, Serendipity, Debias, Multi-list interface
    <<<4. コードリンク>>>
    存在する場合はリンクを出力すること
    <<<5. 詳細な内容 (markdown形式)>>>
    特に、該当する内容がある場合は以下を含めること
    - リサーチクエスチョン
    - 技術的側面
        - 各手法の基本的な概念や原理 
        - 手法の特徴や利点、適用範囲
        - 手法の選択理由や研究における役割 
        - 手法の具体的な適用方法や実装方法 
        - 手法に関連する重要な数式やアルゴリズム 
        - 手法のパラメータ設定や学習プロセス 
    - 研究の結果
        - 研究の目的に沿った結果
        - 手法の有効性を示す定量的・定性的な結果
        - 研究の新規性や重要性を裏付ける結果 
        - リサーチクエスチョンに対する回答
    - 結果の解釈や考察の妥当性や限界
        - 著者の解釈や主張を裏付けるエビデンスの強さ 
        - 結果の解釈における仮定や前提条件 
        - 結果の解釈が持つ限界や対象となる範囲 
        - 今後の研究方向や課題
    - 関連研究のリストアップ
        - 論文名・著者・所属・発表年: 引用箇所（例: 緒言, 方法, 結果, 考察）
    - 特にどういう問題・タスクを解いているか
    - 図表の説明と重要性

    また、以下に留意する
    - 重要な部分やキーワードは太文字で表現する
    - 読み手にわかりやすい文章構成を心がけ、段落構成を適切に行い、論理的な流れを意識する
    - 専門用語には説明を加える
    - 数式についてはnotationを必ず記載する
    - 図表や数式のレンダリングを活用する
    - 数式はLaTeX形式で出力する。出力する際には$$数式$$という形式を使用する
    - アルゴリズムがある場合はPythonコードで出力し、コードブロックとして表示する
    - 複数の手法を組み合わせて使用している場合、以下の観点を踏まえて説明する
        - 手法間の関連性や相互作用 
        - 類似の手法との比較 
        - 本研究で使用した手法の優位性や特徴 

    コンテンツ:
    {content}
    """
    image_parts = [
        {"mime_type": image["mime_type"], "data": base64.b64decode(image["data"])}
        for image in images
    ]
    response = model.generate_content([prompt.format(content=content)] + image_parts)

    # parse response
    items = {}
    patterns = {
        "title": r"<<<1\.\s*(.*?)(?=<<<2\.|$)",
        "summary": r"<<<2\.\s*(.*?)(?=<<<3\.|$)",
        "labels": r"<<<3\.\s*(.*?)(?=<<<4\.|$)",
        "code_link": r"<<<4\.\s*(.*?)(?=<<<5\.|$)",
        "description": r"<<<5\.\s*(.*?)(?=$)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, response.text, re.DOTALL)
        if match:
            items[key] = match.group(1).strip().replace("<<<", "").replace(">>>", "")

    if "labels" in items:
        items["labels"] = [label.strip() for label in items["labels"].split(",")]

    if "code_link" in items:
        if not items["code_link"].lower().startswith("http"):
            items["code_link"] = None

    return items


def main(urls: List[str]) -> None:
    """
    Main function to analyze content from given URLs.

    Args:
        urls (List[str]): List of URLs to analyze
    """
    output_path = Path("output.csv")
    scraped_urls = (
        pd.read_csv(output_path)["url"].tolist() if output_path.exists() else []
    )

    for url in urls:
        if url in scraped_urls:
            continue
        print(f"Fetching content from {url}")
        content_and_images = fetch_content_and_images(url)
        if content_and_images["text"]:
            for model_name in MODEL_NAMES:
                try:
                    result = analyze_content_and_images(
                        content_and_images["text"],
                        content_and_images["images"],
                        model_name=model_name,
                    )
                    result["url"] = url
                    result["model_name"] = model_name
                    pd.DataFrame([result]).to_csv(
                        output_path,
                        mode="a",
                        header=not output_path.exists(),
                        index=False,
                    )
                    break
                except Exception as e:
                    print(f"Error with model {model_name}: {e}")
                    continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze content from URLs")
    parser.add_argument(
        "--urls", nargs="+", help="Space-separated list of URLs to analyze"
    )
    args = parser.parse_args()
    main(args.urls)
