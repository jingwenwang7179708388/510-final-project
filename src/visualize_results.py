"""
visualize_results.py

Create visualizations for BBC news sentiment analysis (headline vs body).

Input:
- data/processed/articles_with_sentiment.csv
- results/summary_section.csv
- results/summary_time.csv

Output (PNG files in results/):
- fig1_articles_per_section.png
- fig2_headline_vs_body_by_section.png
- fig3_headline_minus_body_distribution.png
- fig4_label_proportions_headline_vs_body.png
- fig5_time_series_headline_vs_body_rolling7d.png

Usage (from project root):
    python src/visualize_results.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_ARTICLES = PROCESSED_DIR / "articles_with_sentiment.csv"
INPUT_SECTION_SUMMARY = RESULTS_DIR / "summary_section.csv"
INPUT_TIME_SUMMARY = RESULTS_DIR / "summary_time.csv"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(filename: str) -> None:
    out_path = RESULTS_DIR / filename
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[SAVE] {out_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    ensure_dirs()

    if not INPUT_ARTICLES.exists():
        raise FileNotFoundError(f"Missing: {INPUT_ARTICLES}")
    if not INPUT_SECTION_SUMMARY.exists():
        raise FileNotFoundError(f"Missing: {INPUT_SECTION_SUMMARY}")
    if not INPUT_TIME_SUMMARY.exists():
        raise FileNotFoundError(f"Missing: {INPUT_TIME_SUMMARY}")

    df = pd.read_csv(INPUT_ARTICLES)
    sec = pd.read_csv(INPUT_SECTION_SUMMARY)
    t = pd.read_csv(INPUT_TIME_SUMMARY)

    # -----------------------------
    # Fig 1: Number of articles per section
    # -----------------------------
    counts = df["section"].value_counts().sort_index()
    plt.figure()
    counts.plot(kind="bar")
    plt.title("Number of Articles per Section (Election Aftermath Window)")
    plt.xlabel("Section")
    plt.ylabel("Count")
    save_fig("fig1_articles_per_section.png")

    # -----------------------------
    # Fig 2: Headline vs Body mean sentiment by section (grouped bars)
    # -----------------------------
    # sec must contain: section, mean_headline, mean_body
    sec_sorted = sec.sort_values("n_articles", ascending=False)

    x = range(len(sec_sorted))
    plt.figure()
    plt.bar([i - 0.2 for i in x], sec_sorted["mean_headline"], width=0.4, label="Headline")
    plt.bar([i + 0.2 for i in x], sec_sorted["mean_body"], width=0.4, label="Body")

    plt.title("Average VADER Compound Score: Headline vs Body (by Section)")
    plt.xlabel("Section")
    plt.ylabel("Mean compound score")
    plt.xticks(list(x), sec_sorted["section"])
    plt.legend()
    save_fig("fig2_headline_vs_body_by_section.png")

    # -----------------------------
    # Fig 3: Distribution of headline-minus-body sentiment
    # -----------------------------
    plt.figure()
    df["headline_minus_body"].hist(bins=30)
    plt.title("Distribution of (Headline - Body) VADER Compound Scores")
    plt.xlabel("headline_minus_body")
    plt.ylabel("Frequency")
    save_fig("fig3_headline_minus_body_distribution.png")

    # -----------------------------
    # Fig 4: Label proportions (headline vs body) by section (stacked bars)
    # -----------------------------
    # headline label proportions
    headline_counts = (
        df.groupby(["section", "headline_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=sorted(df["section"].unique()))
    )
    headline_props = headline_counts.div(headline_counts.sum(axis=1), axis=0)

    # body label proportions
    body_counts = (
        df.groupby(["section", "body_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=sorted(df["section"].unique()))
    )
    body_props = body_counts.div(body_counts.sum(axis=1), axis=0)

    # Plot in two stacked bar charts (headline on top, body on bottom)
    plt.figure(figsize=(9, 7))
    ax1 = plt.subplot(2, 1, 1)
    headline_props.plot(kind="bar", stacked=True, ax=ax1)
    ax1.set_title("Headline Sentiment Label Proportions by Section")
    ax1.set_xlabel("")
    ax1.set_ylabel("Proportion")
    ax1.legend(title="Label", bbox_to_anchor=(1.02, 1), loc="upper left")

    ax2 = plt.subplot(2, 1, 2)
    body_props.plot(kind="bar", stacked=True, ax=ax2)
    ax2.set_title("Body Sentiment Label Proportions by Section")
    ax2.set_xlabel("Section")
    ax2.set_ylabel("Proportion")
    ax2.legend(title="Label", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig4_label_proportions_headline_vs_body.png", dpi=200)
    plt.close()
    print("[SAVE] results/fig4_label_proportions_headline_vs_body.png")

    # -----------------------------
    # Fig 5: Time series (7-day rolling) of headline vs body sentiment
    # -----------------------------
    # t must contain: date, section, mean_headline, mean_body
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t = t.dropna(subset=["date"])
    t = t.sort_values(["section", "date"])

    # 7-day rolling within each section
    t["headline_roll7"] = (
        t.groupby("section")["mean_headline"]
        .rolling(window=7, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )
    t["body_roll7"] = (
        t.groupby("section")["mean_body"]
        .rolling(window=7, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    plt.figure(figsize=(10, 6))
    for section, g in t.groupby("section"):
        plt.plot(g["date"], g["headline_roll7"], label=f"{section} (headline)")
        plt.plot(g["date"], g["body_roll7"], linestyle="--", label=f"{section} (body)")

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)

    plt.title("Election Aftermath: 7-day Rolling Sentiment Over Time (Headline vs Body)")
    plt.xlabel("Date")
    plt.ylabel("7-day rolling mean compound score")
    plt.legend(ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig("fig5_time_series_headline_vs_body_rolling7d.png")

    print("[DONE] Visualizations created in results/.")


if __name__ == "__main__":
    main()
