import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import os
import pandas as pd

XLS_PATH = "data/savedrecs.xls"
RAW_PAPER_DIR = "data/raw_papers/"
BASE_SCI_HUB_URL = "https://sci-hub.se/"


def load_excel(file_path: str) -> pd.DataFrame:
    """Load the xls file and return a DataFrame with DOI and citation counts for columns."""
    df = pd.read_excel(file_path)
    doi_df = df[["DOI", "Times Cited, All Databases"]].copy()
    doi_df = doi_df.dropna(subset=["DOI"])
    doi_df = doi_df.drop_duplicates(subset=["DOI"]).reset_index(drop=True)
    return doi_df


def download_pdfs(doi_df: pd.DataFrame, output_dir: str):
    """
    Download PDFs for the given DOIs using Sci-Hub.
    Args:
        doi_df (pd.DataFrame): DataFrame containing DOIs and their citation counts.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    missed_df = pd.DataFrame(columns=["DOI", "Times Cited, All Databases"])

    for i in tqdm(range(len(doi_df))):
        doi = doi_df.loc[i, "DOI"]
        # trace missing PDFs for papers cited more than 100
        # download manually or find another way
        times_cited = int(doi_df.loc[i, "Times Cited, All Databases"])

        filename = f"{output_dir}{doi.replace('/', '_')}.pdf"

        if os.path.exists(filename):
            continue

        try:
            page_url = BASE_SCI_HUB_URL + doi
            response = requests.get(page_url, headers=headers, timeout=30)
            response.raise_for_status()

            # find real PDF link
            soup = BeautifulSoup(response.content, "html.parser")
            pdf_element = soup.find("iframe", {"id": "pdf"}) or soup.find("embed")

            if not pdf_element and times_cited > 100:
                missed_df = pd.concat(
                    [
                        missed_df,
                        pd.DataFrame(
                            [
                                {
                                    "DOI": doi,
                                    "Times Cited, All Databases": times_cited,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
                continue

            pdf_url = pdf_element["src"]

            if pdf_url.startswith("//"):
                pdf_url = "https:" + pdf_url

            pdf_response = requests.get(pdf_url, headers=headers, timeout=60)
            pdf_response.raise_for_status()

            with open(filename, "wb") as f:
                f.write(pdf_response.content)
            print(f"Downloaded PDF for DOI {doi} as {filename.removeprefix('../data/raw_papers/')}")

        except requests.exceptions.RequestException as e:
            print(f"Failed to process DOI {doi}: {e}")
            continue

    print(f"\nDownload failed for {len(missed_df)} papers.")


if __name__ == "__main__":
    doi_df = load_excel(XLS_PATH)
    os.makedirs(RAW_PAPER_DIR, exist_ok=True)
    download_pdfs(doi_df, RAW_PAPER_DIR)
