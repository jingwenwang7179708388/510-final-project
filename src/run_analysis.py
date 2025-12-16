"""
run_analysis.py

Compute VADER sentiment for BBC news headlines vs body text and summarize results.

Input:
- data/processed/articles_clean.csv

Output:
- data/processed/articles_with_sentiment.csv
- results/summary_section.csv
- results/summary_time.csv

Usage (from project root):
    python src/run_analysis.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

# We use NLTK's VADER implementation (downloads vader_lexicon automatically if missing)
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


@dataclass(frozen=True)
class Config:
    # Event window = election aftermath
    start_date: str = "2024-12-01"
    end_date: str = "2025-12-15"


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_CLEAN = PROCESSED_DIR / "articles_clean.csv"

OUTPUT_WITH_SENTIMENT = PROCESSED_DIR / "articles_with_sentiment.csv"
OUTPUT_SECTION_SUMMARY = RESULTS_DIR / "summary_section.csv"
OUTPUT_TIME_SUMMARY = RESULTS_DIR / "summary_time.csv"


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def init_vader() -> SentimentIntensityAnalyzer:
    """
    Initialize NLTK VADER sentiment analyzer.
    """
    try:
        # Ensure lexicon exists
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")
    return SentimentIntensityAnalyzer()


def vader_compound(sia: SentimentIntensityAnalyzer, text: str) -> float:
    """
    Return VADER compound score in [-1, 1].
    """
    if text is None:
        text = ""
    text = str(text).strip()
    if not text:
        return 0.0
    return float(sia.polarity_scores(text)["compound"])


def label_sentiment(compound: float) -> str:
    """
    Standard VADER thresholds:
      compound >= 0.05 -> positive
      compound <= -0.05 -> negative
      else -> neutral
    """
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def load_clean() -> pd.DataFrame:
    if not INPUT_CLEAN.exists():
        raise FileNotFoundError(f"Missing cleaned file: {INPUT_CLEAN}")

    df = pd.read_csv(INPUT_CLEAN)
    print(f"[LOAD] Cleaned rows: {len(df)}")
    return df


def coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize dates. We prefer published_date if available; fall back to published_at.
    Adds:
      - published_dt (datetime64)
      - date (YYYY-MM-DD string)
    """
    # Prefer published_date (your clean_data already created it)
    if "published_date" in df.columns:
        dt = pd.to_datetime(df["published_date"], errors="coerce")
    elif "published_at" in df.columns:
        dt = pd.to_datetime(df["published_at"], errors="coerce")
    else:
        raise ValueError("No published_date or published_at column found.")

    df = df.copy()
    df["published_dt"] = dt
    df = df.dropna(subset=["published_dt"])

    # Convert to date string (YYYY-MM-DD)
    df["date"] = df["published_dt"].dt.strftime("%Y-%m-%d")
    return df


def apply_event_window(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Keep only rows in the election aftermath window.
    """
    start = pd.to_datetime(cfg.start_date)
    end = pd.to_datetime(cfg.end_date)

    df = df.copy()
    df = df[(df["published_dt"] >= start) & (df["published_dt"] <= end)]
    return df


def compute_sentiment(df: pd.DataFrame, sia: SentimentIntensityAnalyzer) -> pd.DataFrame:
    """
    Compute headline vs body sentiment.
    Adds:
      - headline_compound, headline_label
      - body_compound, body_label
      - headline_minus_body
    """
    df = df.copy()

    # Safety: ensure text columns exist
    if "headline" not in df.columns:
        df["headline"] = ""
    if "body_preview" not in df.columns:
        df["body_preview"] = ""

    df["headline"] = df["headline"].fillna("").astype(str)
    df["body_preview"] = df["body_preview"].fillna("").astype(str)

    df["headline_compound"] = df["headline"].apply(lambda x: vader_compound(sia, x))
    df["body_compound"] = df["body_preview"].apply(lambda x: vader_compound(sia, x))

    df["headline_label"] = df["headline_compound"].apply(label_sentiment)
    df["body_label"] = df["body_compound"].apply(label_sentiment)

    df["headline_minus_body"] = df["headline_compound"] - df["body_compound"]

    return df


def write_outputs(df: pd.DataFrame) -> None:
    df.to_csv(OUTPUT_WITH_SENTIMENT, index=False)
    print(f"[WRITE] {OUTPUT_WITH_SENTIMENT.relative_to(PROJECT_ROOT)}")


def summarize_by_section(df: pd.DataFrame) -> pd.DataFrame:
    """
    Section-level summary for proposal questions:
      - avg headline sentiment
      - avg body sentiment
      - delta headline-body
      - counts
    """
    if "section" not in df.columns:
        raise ValueError("Missing column: section")

    sec = (
        df.groupby("section")
        .agg(
            n_articles=("url", "count"),
            mean_headline=("headline_compound", "mean"),
            mean_body=("body_compound", "mean"),
            mean_delta=("headline_minus_body", "mean"),
            std_headline=("headline_compound", "std"),
            std_body=("body_compound", "std"),
        )
        .reset_index()
        .sort_values("n_articles", ascending=False)
    )
    sec.to_csv(OUTPUT_SECTION_SUMMARY, index=False)
    print(f"[WRITE] {OUTPUT_SECTION_SUMMARY.relative_to(PROJECT_ROOT)}")
    return sec


def summarize_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Time-series summary (daily) within each section:
      - mean headline, mean body, mean delta
      - n articles
    """
    t = (
        df.groupby(["date", "section"])
        .agg(
            n_articles=("url", "count"),
            mean_headline=("headline_compound", "mean"),
            mean_body=("body_compound", "mean"),
            mean_delta=("headline_minus_body", "mean"),
        )
        .reset_index()
        .sort_values(["section", "date"])
    )
    t.to_csv(OUTPUT_TIME_SUMMARY, index=False)
    print(f"[WRITE] {OUTPUT_TIME_SUMMARY.relative_to(PROJECT_ROOT)}")
    return t


def print_quick_checks(df: pd.DataFrame) -> None:
    print("\n[CHECK] Headline sentiment label distribution (overall):")
    print(df["headline_label"].value_counts(dropna=False))

    print("\n[CHECK] Body sentiment label distribution (overall):")
    print(df["body_label"].value_counts(dropna=False))

    # how many sections, how many rows
    if "section" in df.columns:
        print("\n[CHECK] Articles per section:")
        print(df["section"].value_counts())


def main() -> None:
    cfg = Config()
    ensure_dirs()

    df = load_clean()
    df = coerce_dates(df)

    # Event window: election aftermath
    df = apply_event_window(df, cfg)
    print(f"[STEP] After event window {cfg.start_date} to {cfg.end_date}: {len(df)}")

    sia = init_vader()
    df = compute_sentiment(df, sia)

    # Save enriched dataset
    write_outputs(df)

    # Summaries used for visualization + report
    summarize_by_section(df)
    summarize_by_time(df)

    print_quick_checks(df)
    print("\n[DONE] Analysis completed.")


if __name__ == "__main__":
    main()
