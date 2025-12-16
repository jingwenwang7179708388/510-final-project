"""
get_data.py

Download (scrape) BBC News articles and save:
1) Raw HTML files to: data/raw/html/
2) Metadata CSV to:  data/raw/metadata.csv

This script is designed for the DSCI 510 final project repository structure.

Usage (from project root):
    python src/get_data.py

Notes:
- We collect articles from BBC section landing pages (World, Business, Technology).
- We attempt to parse publication date from <time datetime="..."> when available.
- We store headline text and a short body preview (first few paragraphs).
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser


# -----------------------------
# Configuration
# -----------------------------

USER_AGENT: str = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

REQUEST_TIMEOUT: int = 15
SLEEP_SECONDS: float = 1.0  # be polite to the website
MAX_ARTICLES_PER_SECTION: int = 120  # target ~100 each after filtering
MIN_BODY_WORDS: int = 60  # filter out nav pages / very short content

SECTIONS: Dict[str, str] = {
    "world": "https://www.bbc.com/news/world",
    "business": "https://www.bbc.com/news/business",
    "technology": "https://www.bbc.com/news/technology",
}

# Event-specific keywords used for BBC search expansion
SECTION_KEYWORDS = {
    "world": ["election", "Trump", "Biden", "Ukraine", "Israel", "China", "Russia", "war"],
    "business": ["inflation", "Fed", "interest rates", "stocks", "markets", "tariffs", "oil", "jobs", "economy", "earnings"],
    "technology": ["AI", "OpenAI", "Google", "Apple", "Microsoft", "chip", "semiconductor", "quantum", "cyber", "TikTok"],
}


PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
RAW_HTML_DIR: Path = RAW_DIR / "html"
METADATA_PATH: Path = RAW_DIR / "metadata.csv"


# -----------------------------
# Data Structures
# -----------------------------

@dataclass(frozen=True)
class ArticleRecord:
    url: str
    section: str
    published_at: Optional[str]  # ISO 8601 string if available
    headline: str
    body_preview: str
    raw_html_path: str


# -----------------------------
# Helpers
# -----------------------------

def ensure_dirs() -> None:
    """Create required output directories if they do not exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)


def normalize_bbc_url(url: str) -> str:
    """
    Normalize BBC URLs:
    - Convert relative URLs to absolute (bbc.com)
    - Convert bbc.co.uk to bbc.com (so filters work consistently)
    - Strip fragments and query params
    """
    url = url.strip()
    if url.startswith("/"):
        url = "https://www.bbc.com" + url

    # unify domain
    url = url.replace("https://www.bbc.co.uk", "https://www.bbc.com")
    url = url.replace("http://www.bbc.co.uk", "https://www.bbc.com")
    url = url.replace("http://www.bbc.com", "https://www.bbc.com")

    # Remove fragments
    url = url.split("#", 1)[0]
    # Remove query params
    url = re.sub(r"\?.*$", "", url)

    return url



def is_probably_article_url(url: str) -> bool:
    """
    Strict article URL filter:
    Accept only real article patterns such as:
    - https://www.bbc.com/news/articles/<id>
    - https://www.bbc.com/news/<section>/<cxxxxxxxxxx>  (article id usually starts with 'c')
    Reject section/region landing pages like /news/world/africa or /news/world/asia
    """
    if not url.startswith("https://www.bbc.com/news"):
        return False

    bad_keywords = ["/videos/", "/live/", "/topics/", "/av/", "/in_pictures", "/special/", "/resources/"]
    if any(k in url for k in bad_keywords):
        return False

    # Pattern 1: /news/articles/<id>
    if re.search(r"^https://www\.bbc\.com/news/articles/[a-z0-9]+$", url):
        return True

    # Pattern 2: /news/<something>/<article_id> where article_id often starts with 'c'
    if re.search(r"^https://www\.bbc\.com/news/[^/]+/c[a-z0-9]{8,}$", url):
        return True

    return False



def safe_filename_from_url(url: str) -> str:
    """Create a stable filename for a URL using a short hash."""
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return f"{h}.html"


def fetch_html(url: str) -> str:
    """Download HTML from a URL and return text. Raises for bad status codes."""
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def extract_links_from_section(section_url: str) -> List[str]:
    """
    Fetch a section landing page and extract candidate article links.
    Returns a de-duplicated list of normalized URLs.
    """
    html = fetch_html(section_url)
    soup = BeautifulSoup(html, "html.parser")

    urls: List[str] = []
    for a in soup.find_all("a", href=True):
        href = normalize_bbc_url(a["href"])
        if is_probably_article_url(href):
            urls.append(href)

    # Deduplicate while preserving order
    seen: Set[str] = set()
    unique_urls: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique_urls.append(u)
    return unique_urls

def extract_links_from_bbc_search(query: str, max_pages: int = 10) -> List[str]:
    """
    Use BBC search pages to collect more article URLs.
    BBC search URL pattern (can change, but usually works):
        https://www.bbc.co.uk/search?q=<query>&page=<n>
    """
    base = "https://www.bbc.co.uk/search"
    collected: List[str] = []
    seen: Set[str] = set()

    for page in range(1, max_pages + 1):
        params = {"q": query, "page": page}
        print(f"[SEARCH] q='{query}' page={page}")
        html = requests.get(base, params=params, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT).text
        soup = BeautifulSoup(html, "html.parser")

        # search results usually contain <a href="..."> to news pages
        for a in soup.find_all("a", href=True):
            href = normalize_bbc_url(a["href"])
            if is_probably_article_url(href) and href not in seen:
                seen.add(href)
                collected.append(href)

        time.sleep(SLEEP_SECONDS)

    return collected


def parse_article_page(html: str) -> Tuple[str, Optional[str], str]:
    """
    Parse an article HTML page and return:
    (headline, published_at_iso, body_preview)

    We try multiple strategies since site HTML can vary.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Headline
    headline = ""
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        headline = h1.get_text(strip=True)

    # Publication date
    published_at_iso: Optional[str] = None
    time_tag = soup.find("time")
    if time_tag and time_tag.has_attr("datetime"):
        dt_raw = str(time_tag["datetime"]).strip()
        try:
            published_at_iso = date_parser.parse(dt_raw).isoformat()
        except Exception:
            published_at_iso = None

    # Body text: collect paragraphs inside <article> when possible
    body_texts: List[str] = []
    article_tag = soup.find("article")
    if article_tag:
        ps = article_tag.find_all("p")
    else:
        ps = soup.find_all("p")

    for p in ps:
        txt = p.get_text(" ", strip=True)
        # Filter out empty / boilerplate bits
        if txt and len(txt.split()) >= 5:
            body_texts.append(txt)

    # Keep only the first few paragraphs (preview)
    body_preview = " ".join(body_texts[:5]).strip()

    return headline, published_at_iso, body_preview


def save_raw_html(url: str, html: str) -> Path:
    """Save raw HTML to data/raw/html/<hash>.html and return the path."""
    filename = safe_filename_from_url(url)
    path = RAW_HTML_DIR / filename
    path.write_text(html, encoding="utf-8")
    return path


def looks_like_section_landing(headline: str, body_preview: str) -> bool:
    """
    Some BBC URLs that pass heuristics can still be section/topic landing pages.
    If headline/body looks uninformative, we skip.
    """
    if not headline:
        return True
    if headline.lower() in {"news", "bbc news", "world", "business", "technology"}:
        return True
    if len(body_preview.split()) < MIN_BODY_WORDS:
        return True
    return False


def collect_section_articles(section_name: str, section_url: str) -> List[ArticleRecord]:
    """
    Collect articles for a single section:
    - fetch section page
    - extract candidate URLs
    - fetch each article page until reaching MAX_ARTICLES_PER_SECTION
    """
    print(f"\n===== Scraping section: {section_name} =====")

    candidate_urls = extract_links_from_section(section_url)

    # Expand using BBC search (to reach enough articles)
    # Combine section name + event keywords to bias results into relevant sections/topics
    search_urls: List[str] = []
    for kw in SECTION_KEYWORDS.get(section_name, []):
        q = f"{section_name} {kw}"
        search_urls.extend(extract_links_from_bbc_search(q, max_pages=20))

    # Merge and deduplicate (preserve order)
    merged: List[str] = []
    seen_merge: Set[str] = set()
    for u in candidate_urls + search_urls:
        if u not in seen_merge:
            seen_merge.add(u)
            merged.append(u)

    candidate_urls = merged
    print(f"[INFO] Expanded candidate URLs (section+search): {len(candidate_urls)}")
    print(f"[INFO] Found {len(candidate_urls)} candidate article URLs")

    records: List[ArticleRecord] = []
    visited: Set[str] = set()

    for url in candidate_urls:
        if url in visited:
            continue
        visited.add(url)

        if len(records) >= MAX_ARTICLES_PER_SECTION:
            break

        try:
            print(f"[GET] {url}")
            html = fetch_html(url)
            headline, published_at_iso, body_preview = parse_article_page(html)

            if looks_like_section_landing(headline, body_preview):
                print(f"[SKIP] Not an article / too short: {headline[:50]}")
                continue

            raw_path = save_raw_html(url, html)

            rec = ArticleRecord(
                url=url,
                section=section_name,
                published_at=published_at_iso,
                headline=headline,
                body_preview=body_preview,
                raw_html_path=str(raw_path.relative_to(PROJECT_ROOT)),
            )
            records.append(rec)
            print(f"[OK] Saved: {headline[:60]}")

            time.sleep(SLEEP_SECONDS)

        except requests.HTTPError as e:
            print(f"[ERROR] HTTP error for {url}: {e}")
        except requests.RequestException as e:
            print(f"[ERROR] Request error for {url}: {e}")
        except Exception as e:
            print(f"[ERROR] Unexpected error for {url}: {e}")

    print(f"[DONE] {section_name}: collected {len(records)} articles")
    return records


def write_metadata_csv(records: List[ArticleRecord]) -> None:
    """Write metadata records to data/raw/metadata.csv."""
    rows = []
    for r in records:
        rows.append(
            {
                "url": r.url,
                "section": r.section,
                "published_at": r.published_at,
                "headline": r.headline,
                "body_preview": r.body_preview,
                "raw_html_path": r.raw_html_path,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(METADATA_PATH, index=False, encoding="utf-8")
    print(f"\n[WRITE] Metadata CSV saved to: {METADATA_PATH.relative_to(PROJECT_ROOT)}")
    print(f"[INFO] Total rows: {len(df)}")


def main() -> None:
    """Main entry point for scraping all sections and saving outputs."""
    ensure_dirs()

    all_records: List[ArticleRecord] = []
    for section_name, section_url in SECTIONS.items():
        recs = collect_section_articles(section_name, section_url)
        all_records.extend(recs)

    write_metadata_csv(all_records)
    print("\n✅ Scraping completed successfully.")


if __name__ == "__main__":
    main()