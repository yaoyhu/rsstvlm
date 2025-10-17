import os
import glob
import requests
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup

"""
Web of Science QUERY, Thanks to zr1997@mail.ustc.edu.cn

TS = (
    (
        Satellite OR "remote sensing" OR spaceborne OR "spectral retrieval"
        OR TROPOMI OR OMI OR MODIS OR VIIRS OR "Sentinel-5P" OR GOSAT OR "OCO-2" OR TEMPO OR GEMS
    )
    AND
    (
        "Climate change" OR pollution OR "air quality" OR "human health" OR emission OR inventory
        OR "global warming" OR "atmospheric trace gases" OR "polluting gases" OR "greenhouse gases"
        OR flux
    )
    AND
    (
        "Nitrogen dioxide" OR NO2 OR "Sulfur dioxide" OR SO2 OR Ozone OR O3 OR Formaldehyde OR HCHO
        OR Glyoxal OR CHOCHO OR "Carbon dioxide" OR CO2 OR "Carbon monoxide" OR CO
        OR "Nitrous oxide" OR N2O OR "Water vapor" OR H2O
    )
)
AND PY=(2015-2025)
AND SU=("Environmental Sciences" OR "Meteorology & Atmospheric Sciences" OR "Remote Sensing") 
"""

WOS_PATH = "/exports/yaoyhu/rsstvlm/data/wos/"
RAW_PAPER_DIR = "/satellite/d3/yaoyhu/rsstvlm/raw_papers/"
BASE_SCI_HUB_URL = "https://sci-hub.se/"


def sanitize_doi(doi: str) -> str:
    """Convert a DOI into a filesystem-friendly string."""
    return doi.replace("/", "_")


def list_downloaded_dois(output_dir: str) -> set[str]:
    """
    Help function to return the set of downloaded DOIs (sanitized), for emergency resume.
    """
    if not os.path.isdir(output_dir):
        return set()

    downloaded: set[str] = set()
    for entry in os.listdir(output_dir):
        if not entry.lower().endswith(".pdf"):
            continue

        stem = entry[:-4]
        parts = stem.split("_", 1)
        if len(parts) != 2:
            continue
        downloaded.add(parts[1])

    return downloaded


def load_excel(file_path: str) -> pd.DataFrame:
    """Load the xls file and return a DataFrame with DOI and citation counts for columns."""
    df = pd.read_excel(file_path)
    doi_df = df[["DOI", "Times Cited, All Databases"]].copy()
    doi_df = doi_df.dropna(subset=["DOI"])
    doi_df = doi_df.drop_duplicates(subset=["DOI"]).reset_index(drop=True)
    return doi_df


def load_excels(excel_path: str) -> pd.DataFrame:
    """Load and concat all WOS XLS exports into a single deduplicated DataFrame."""
    if not excel_path or not os.path.isdir(excel_path):
        raise ValueError(f"Excel directory {excel_path} does not exist.")

    pattern = os.path.join(excel_path, "savedrecs-*.xls")
    wos_xls = sorted(glob.glob(pattern))

    if not wos_xls:
        raise FileNotFoundError(f"No export files found under {pattern}.")

    merged_df = pd.concat((load_excel(xls_path) for xls_path in wos_xls), ignore_index=True)
    # remove duplicates DOI in different files, keep the max citation count
    merged_df = merged_df.groupby("DOI", as_index=False)["Times Cited, All Databases"].max()
    return merged_df.sort_values(by="Times Cited, All Databases", ascending=False).reset_index(drop=True)


def download_pdfs(doi_df: pd.DataFrame, output_dir: str):
    """
    Download PDFs by the given DOIs using Sci-Hub.
    Args:
        doi_df (pd.DataFrame): DataFrame containing DOIs and their citation counts.
        output_dir (str): Directory to save downloaded PDFs.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    downloaded_dois = list_downloaded_dois(output_dir)

    if downloaded_dois:
        sanitized_series = doi_df["DOI"].map(sanitize_doi)
        downloaded_mask = sanitized_series.isin(downloaded_dois)
        skipped = int(downloaded_mask.sum())
        if skipped:
            print(f"Skipping {skipped} DOIs already on disk.")
            doi_df = doi_df.loc[~downloaded_mask].reset_index(drop=True)

    iterable = doi_df.itertuples(index=False, name=None)
    for doi, times_cited in tqdm(
        iterable,
        total=len(doi_df),
        desc="Processing DOIs",
        unit="paper",
    ):
        times_cited = int(times_cited)
        sanitized_doi = sanitize_doi(doi)

        # Construct filename
        filename = os.path.join(output_dir, f"{times_cited}_{sanitized_doi}.pdf")

        if sanitized_doi in downloaded_dois or os.path.exists(filename):
            continue

        try:
            page_url = BASE_SCI_HUB_URL + doi
            response = requests.get(page_url, headers=headers, timeout=30)
            response.raise_for_status()

            # find real PDF link
            # Written by Copilot, I know nothing about HTML parsing
            soup = BeautifulSoup(response.content, "html.parser")
            pdf_element = soup.find("iframe", {"id": "pdf"}) or soup.find("iframe") or soup.find("embed")

            pdf_url: str | None = None
            if pdf_element:
                pdf_url = pdf_element.get("src") or pdf_element.get("data") or pdf_element.get("href")

            if not pdf_url:
                anchor = soup.select_one("a[href$='.pdf']")
                if anchor:
                    pdf_url = anchor.get("href")

            if not pdf_url:
                print(f"Could not locate PDF link for DOI {doi} (cited {times_cited}).")
                continue

            if pdf_url.startswith("//"):
                pdf_url = "https:" + pdf_url

            if pdf_url.startswith("/"):
                pdf_url = BASE_SCI_HUB_URL.rstrip("/") + pdf_url

            pdf_response = requests.get(pdf_url, headers=headers, timeout=60)
            pdf_response.raise_for_status()

            with open(filename, "wb") as f:
                f.write(pdf_response.content)
            print(f"Downloaded PDF for DOI {doi} as {os.path.basename(filename)}")
            downloaded_dois.add(sanitized_doi)

        except requests.exceptions.RequestException as e:
            print(f"Failed to process DOI {doi}: {e}")
            continue
        except Exception as e:  # pragma: no cover - defensive
            print(f"Unexpected error while processing DOI {doi}: {e}")
            continue


if __name__ == "__main__":
    os.makedirs(RAW_PAPER_DIR, exist_ok=True)
    doi_df = load_excels(WOS_PATH)
    download_pdfs(doi_df, RAW_PAPER_DIR)
