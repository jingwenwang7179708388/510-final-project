"""
DSCI 510 Final Project - Data Collection

This script scrapes BBC News articles from selected sections and saves:
1) raw HTML files in data/raw/
2) a metadata CSV in data/raw/articles_metadata.csv

Run:
    python src/get_data.py
"""

from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


# -----------------------------
# Configuration
# -----------------------------
BASE_URL = "https://www.bbc.com"

SECTIONS: Dict[str, str] = {
    "world": f"{BASE_URL}/news/world",
    "business": f"{BASE_URL}/news/business",
    "technology": f"{BASE_URL}/news/technology",
}

ARTICLES_PER_SECTION = 50          # target per section (you can change)
MAX_PAGES_PER_SECTION = 60         # safety cap to avoid infinite paging
SLEEP_BETWEEN_REQUESTS = 1.0       # polite delay between requests (seconds)
REQUEST_TIMEOUT = 15               # seconds

MIN_WORDS = 80                     # skip very short articles

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

HEADERS = {"User-Agent": USER_AGENT}

# Project paths (repo root is parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

METADATA_CSV = RAW_DIR / "articles_metadata.csv"


# -----------------------------
# Data containers
# -----------------------------
@dataclass
class ArticleRecord:
    """Structured record for one BBC article."""
    article_id: str
    url: str
    section: str
    title: str
    publish_date: str  # ISO format if possible, else raw string
    body_text: str
    word_count: int
    raw_html_path: str  # relative path from project root


# -----------------------------
# Helpers: networking
# -----------------------------
def fetch_html(url: str) -> Optional[str]:
    """
    Fetch HTML for a URL.

    Args:
        url (str): Target URL.

    Returns:
        Optional[str]: HTML string if success, else None.
    """
    try:
        print(f"[GET] {url}")
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        return None


# -----------------------------
# Helpers: URL filtering
# -----------------------------
def is_article_url(url: str) -> bool:
    """
    Return True only if the URL looks like a real BBC article page.

    Accept only:
        1) https://www.bbc.com/news/articles/<id>
        2) https://www.bbc.com/news/<slug>-<digits>

    Reject:
        /live/, /av/, /video/, /tv/, /sounds/, /topics/, etc.
        section landing pages like /news/world, /news/us-canada, /news/war-in-ukraine
    """
    if not url.startswith(f"{BASE_URL}/news"):
        return False

    bad_patterns = ["/live/", "/av/", "/video/", "/tv/", "/sounds/", "/topics/", "/in_pictures", "/special/"]
    if any(p in url for p in bad_patterns):
        return False

    path = urlparse(url).path  # e.g. "/news/articles/c62v7n9wzkyo"

    # (1) /news/articles/<id>
    if path.startswith("/news/articles/"):
        return True

    # (2) /news/<slug>-<digits>  (must end with digits)
    # Example: /news/world-europe-12345678
    if re.match(r"^/news/[^/]+-\d+$", path):
        return True

    return False


def extract_article_links_from_section_page(html: str) -> List[str]:
    """
    Extract candidate article URLs from a BBC section landing page HTML.

    Args:
        html (str): HTML of section page.

    Returns:
        List[str]: List of absolute URLs (filtered to real article patterns).
    """
    soup = BeautifulSoup(html, "html.parser")
    urls: Set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()

        # Normalize to absolute
        full_url = urljoin(BASE_URL, href)

        if is_article_url(full_url):
            urls.add(full_url)

    return sorted(urls)


# -----------------------------
# Helpers: parsing article HTML
# -----------------------------
def parse_publish_date(soup: BeautifulSoup) -> str:
    """
    Try to parse publish date from BBC article HTML.

    Returns:
        str: ISO string if parseable, else empty string.
    """
    # Common pattern: <time datetime="2025-12-14T...Z">
    time_tag = soup.find("time")
    if time_tag and time_tag.has_attr("datetime"):
        dt_raw = time_tag["datetime"].strip()
        # keep as ISO if already ISO-ish
        return dt_raw

    return ""


def parse_title(soup: BeautifulSoup) -> str:
    """
    Parse title from BBC article HTML.

    Returns:
        str: Title text.
    """
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(" ", strip=True)
        if title:
            return title
    return ""


def parse_body_text(soup: BeautifulSoup) -> str:
    """
    Parse the main body text from BBC article HTML.

    Strategy:
        - BBC articles typically have multiple <p> tags in the article body.
        - We collect paragraph text and join.

    Returns:
        str: Article body text (joined).
    """
    paragraphs: List[str] = []

    # Try to prioritize paragraphs inside <main> if present
    main = soup.find("main")
    scope = main if main else soup

    for p in scope.find_all("p"):
        txt = p.get_text(" ", strip=True)
        if not txt:
            continue
        # Filter out very short boilerplate fragments
        if len(txt) < 20:
            continue
        paragraphs.append(txt)

    # De-duplicate consecutive duplicates (sometimes repeated in page)
    cleaned: List[str] = []
    for t in paragraphs:
        if not cleaned or cleaned[-1] != t:
            cleaned.append(t)

    return "\n".join(cleaned).strip()


def derive_article_id(url: str) -> str:
    """
    Derive a stable article_id from URL.

    For /news/articles/<id> -> use <id>
    For /news/<slug>-<digits> -> use <digits>
    """
    path = urlparse(url).path

    if path.startswith("/news/articles/"):
        return path.split("/news/articles/")[-1].strip("/")

    m = re.match(r"^/news/[^/]+-(\d+)$", path)
    if m:
        return m.group(1)

    # Fallback: last segment
    last = path.strip("/").split("/")[-1]
    return last or "article"


def save_raw_html(html: str, section: str, article_id: str) -> Path:
    """
    Save raw HTML to data/raw/.

    Returns:
        Path: absolute file path saved.
    """
    filename = f"{section}_{article_id}.html"
    out_path = RAW_DIR / filename
    out_path.write_text(html, encoding="utf-8")
    return out_path


# -----------------------------
# Metadata CSV handling
# -----------------------------
def ensure_metadata_header(path: Path) -> None:
    """
    Ensure metadata CSV exists and has header row.
    """
    if path.exists():
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "article_id",
            "url",
            "section",
            "title",
            "publish_date",
            "body_text",
            "word_count",
            "raw_html_path",
        ])


def append_metadata_row(path: Path, rec: ArticleRecord) -> None:
    """
    Append one record to the metadata CSV.
    """
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            rec.article_id,
            rec.url,
            rec.section,
            rec.title,
            rec.publish_date,
            rec.body_text,
            rec.word_count,
            rec.raw_html_path,
        ])


# -----------------------------
# Main scraping logic
# -----------------------------
def scrape_section(section: str, start_url: str, target_count: int) -> int:
    """
    Scrape one BBC section until target_count valid articles are collected.

    Args:
        section (str): section name, e.g. 'world'
        start_url (str): section landing page URL
        target_count (int): desired number of valid articles

    Returns:
        int: number of collected valid articles
    """
    print(f"\n===== Scraping section: {section} =====")
    collected = 0
    seen_urls: Set[str] = set()
    no_new_url_pages = 0

    for page in range(0, MAX_PAGES_PER_SECTION):
        if page == 0:
            url = start_url
        else:
            url = f"{start_url}?page={page}"

        html = fetch_html(url)
        if html is None:
            print(f"[WARN] Failed to load section page {url}, stopping section.")
            break

        candidate_urls = extract_article_links_from_section_page(html)
        print(f"[INFO] Page {page}: Found {len(candidate_urls)} candidate article URLs")

        # Filter new URLs only
        new_urls = [u for u in candidate_urls if u not in seen_urls]
        if not new_urls:
            no_new_url_pages += 1
        else:
            no_new_url_pages = 0

        if no_new_url_pages >= 3:
            print("[INFO] No new URLs for 3 pages, stopping section.")
            break

        for article_url in new_urls:
            if collected >= target_count:
                break

            seen_urls.add(article_url)

            article_html = fetch_html(article_url)
            if article_html is None:
                continue

            soup = BeautifulSoup(article_html, "html.parser")

            title = parse_title(soup)
            pub_date = parse_publish_date(soup)
            body = parse_body_text(soup)

            # Basic validity checks
            word_count = len(body.split())
            if not title or word_count < MIN_WORDS:
                if word_count < MIN_WORDS:
                    print(f"[INFO] Skipping very short article ({word_count} words): {article_url}")
                else:
                    print(f"[INFO] Skipping missing-title article: {article_url}")
                continue

            article_id = derive_article_id(article_url)

            raw_path = save_raw_html(article_html, section, article_id)
            raw_rel = str(raw_path.relative_to(PROJECT_ROOT))

            rec = ArticleRecord(
                article_id=article_id,
                url=article_url,
                section=section,
                title=title,
                publish_date=pub_date,
                body_text=body,
                word_count=word_count,
                raw_html_path=raw_rel,
            )

            append_metadata_row(METADATA_CSV, rec)
            collected += 1
            print(f"[OK] {section}: collected {collected}/{target_count}")

            time.sleep(SLEEP_BETWEEN_REQUESTS)

        # polite delay between section pages
        time.sleep(SLEEP_BETWEEN_REQUESTS)

        if collected >= target_count:
            break

    print(f"[DONE] Section {section}: collected {collected} valid articles.")
    return collected


def run() -> None:
    """
    Run the full data collection pipeline for all sections.
    """
    ensure_metadata_header(METADATA_CSV)

    total = 0
    for section, url in SECTIONS.items():
        total += scrape_section(section, url, ARTICLES_PER_SECTION)

    print(f"\n===== DONE: collected {total} total valid articles across sections =====")
    print(f"[INFO] Metadata CSV: {METADATA_CSV}")
    print(f"[INFO] Raw HTML dir: {RAW_DIR}")


if __name__ == "__main__":
    run()
