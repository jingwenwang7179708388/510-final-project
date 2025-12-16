"""
clean_data.py

Clean and preprocess scraped BBC news data.

Input:
- data/raw/metadata.csv
- data/raw/html/*.html (optional; we mainly use metadata.csv for speed)

Output:
- data/processed/articles_clean.csv
- data/processed/articles_clean.jsonl

Usage (from project root):
    python src/clean_data.py

Cleaning steps:
1) Remove non-article / landing pages (e.g., "NewsNews", very short text).
2) Standardize datetime to ISO and also create a date column (YYYY-MM-DD).
3) Filter by event time window.
4) Drop duplicates by URL.
5) Keep balanced samples per section (cap per section).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


# -----------------------------
# Configuration
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_META_PATH = RAW_DIR / "metadata.csv"
OUT_CSV_PATH = PROCESSED_DIR / "articles_clean.csv"
OUT_JSONL_PATH = PROCESSED_DIR / "articles_clean.jsonl"

# Event window (adjust if needed)
EVENT_START = "2024-12-01"
EVENT_END   = "2025-12-15"



# Text quality thresholds
MIN_HEADLINE_CHARS = 8
MIN_BODY_WORDS = 60  # helps remove paywall/landing pages

# Balance control
MAX_PER_SECTION = 120  # cap per section so analysis can take ~100 later


# -----------------------------
# Helpers
# -----------------------------

def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse published_at into datetime and derive published_date.
    Rows with unparsable dates become NaT and can be filtered.
    """
    df = df.copy()
    df["published_dt"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df["published_date"] = df["published_dt"].dt.date.astype("string")
    return df


def basic_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove obvious non-article rows and low-quality rows.
    """
    df = df.copy()

    # headline quality
    df["headline"] = df["headline"].fillna("").astype(str).str.strip()
    df["body_preview"] = df["body_preview"].fillna("").astype(str).str.strip()

    # Remove "NewsNews" and similar generic titles
    bad_headlines = {"newsnews", "news", "bbc news"}
    df = df[~df["headline"].str.lower().isin(bad_headlines)]

    # Remove very short headline/body
    df = df[df["headline"].str.len() >= MIN_HEADLINE_CHARS]
    df["body_words"] = df["body_preview"].str.split().map(len)
    df = df[df["body_words"] >= MIN_BODY_WORDS]

    # Drop rows missing critical fields
    df = df.dropna(subset=["url", "section"])

    # Drop rows where body_preview looks like navigation / very generic
    df = df[~df["body_preview"].str.contains(r"\b(?:Home|Skip to content|BBC Homepage|News)\b", case=False, regex=True)]

    return df


def filter_event_window(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    df = df.copy()

    # Split into (has date) and (missing date)
    df_has = df.dropna(subset=["published_dt"]).copy()
    df_miss = df[df["published_dt"].isna()].copy()

    start = pd.to_datetime(start_date, utc=True)
    end = pd.to_datetime(end_date, utc=True) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    df_has = df_has[df_has["published_dt"].between(start, end)]

    # Keep missing-date rows (they can be used for section-level comparisons)
    return pd.concat([df_has, df_miss], ignore_index=True)




def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicates by URL (keep first).
    """
    df = df.copy()
    df = df.drop_duplicates(subset=["url"], keep="first")
    return df


def cap_per_section(df: pd.DataFrame, max_per_section: int) -> pd.DataFrame:
    """
    Cap number of articles per section to improve balance.
    Strategy: keep most recent articles first.
    """
    df = df.copy()
    df = df.sort_values(by=["section", "published_dt"], ascending=[True, False])

    capped = (
        df.groupby("section", group_keys=False)
        .head(max_per_section)
        .reset_index(drop=True)
    )
    return capped


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns needed downstream.
    """
    cols = [
        "url",
        "section",
        "published_at",
        "published_date",
        "headline",
        "body_preview",
        "raw_html_path",
    ]
    # Some columns might not exist depending on earlier scripts; keep what exists
    existing = [c for c in cols if c in df.columns]
    return df[existing].copy()


def main() -> None:
    ensure_dirs()

    if not RAW_META_PATH.exists():
        raise FileNotFoundError(f"Missing raw metadata file: {RAW_META_PATH}")

    df = pd.read_csv(RAW_META_PATH)
    print(f"[LOAD] Raw metadata rows: {len(df)}")

    df = parse_dates(df)
    df = basic_filters(df)
    print(f"[STEP] After basic filters: {len(df)}")

    df = filter_event_window(df, EVENT_START, EVENT_END)
    print(f"[STEP] After event window {EVENT_START} to {EVENT_END}: {len(df)}")

    df = deduplicate(df)
    print(f"[STEP] After deduplication: {len(df)}")

    df = cap_per_section(df, MAX_PER_SECTION)
    print("[STEP] After capping per section:")
    print(df["section"].value_counts())

    out_df = select_output_columns(df)

    out_df.to_csv(OUT_CSV_PATH, index=False, encoding="utf-8")
    out_df.to_json(OUT_JSONL_PATH, orient="records", lines=True, force_ascii=False)

    print(f"\n[WRITE] Clean CSV:   {OUT_CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"[WRITE] Clean JSONL: {OUT_JSONL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"[DONE] Final cleaned rows: {len(out_df)}")


if __name__ == "__main__":
    main()
