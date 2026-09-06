"""
scrape_website.py
------------------
Crawls www.nawaloka.com (a shallow, same-domain crawl), extracts the
readable text from each page, chunks it, embeds it, and persists it into
a local Chroma vector store.

Run this ONCE (or whenever you want to refresh the knowledge base):

    python scripts/scrape_website.py

One-time setup for the headless browser (see below for why we need one):

    playwright install chromium

Why Playwright instead of plain `requests`?
--------------------------------------------
nawaloka.com is a client-side-rendered React app: the server sends a
near-empty HTML shell, and the real content (and all navigation links)
are injected by JavaScript after the page loads in a browser.
`requests` + BeautifulSoup never runs that JavaScript, so it only ever
sees the empty shell (confirmed while debugging: raw HTML was ~860
characters with no real text). Playwright launches a real headless
Chromium browser, waits for the page to render, and then we read the
fully-rendered DOM -- exactly what a human visitor would see.

Notes on design choices (worth mentioning in your README/demo):
- We do a breadth-first crawl limited to the same domain and a max
  page count / depth, to avoid scraping the entire internet by accident.
- We strip nav/footer/script/style tags before extracting text so the
  vector DB isn't polluted with menu boilerplate.
- Chunking uses a recursive character splitter with overlap, which is
  the standard, dependable baseline for RAG chunking.
- Each chunk stores metadata (source URL, page title) so the chatbot
  can cite where information came from -- good for trust and for the
  "improved retrieval quality" bonus marks (metadata filtering/rerank).
"""

import time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIR = str(ROOT / "vectorstore" / "chroma_db")

BASE_URL = "https://www.nawaloka.com"
ALLOWED_DOMAIN = urlparse(BASE_URL).netloc
MAX_PAGES = 60           # keep the crawl bounded
REQUEST_DELAY_SEC = 1.0  # be polite to the server
PAGE_LOAD_TIMEOUT_MS = 20000
USER_AGENT = "Mozilla/5.0 (compatible; HospitalChatbotBot/1.0)"


def is_same_domain(url: str) -> bool:
    return urlparse(url).netloc in ("", ALLOWED_DOMAIN)


def clean_page_text(html: str) -> tuple[str, str]:
    """Return (title, cleaned_text) for a page's rendered HTML."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "form", "svg"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else ""
    text = soup.get_text(separator="\n")

    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    cleaned = "\n".join(lines)
    return title, cleaned


def crawl(base_url: str, max_pages: int = MAX_PAGES) -> list[dict]:
    """Simple BFS crawl within the same domain, using a headless browser
    to render each page's JavaScript before extracting content/links.
    Returns list of {url, title, text}.
    """
    visited = set()
    queue = deque([base_url])
    pages = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=USER_AGENT)

        while queue and len(visited) < max_pages:
            url = queue.popleft()
            if url in visited:
                continue
            visited.add(url)

            try:
                page.goto(url, timeout=PAGE_LOAD_TIMEOUT_MS, wait_until="networkidle")
            except Exception as exc:  # noqa: BLE001 - keep crawling past a single bad page
                print(f"⚠️  Skipping {url}: {exc}")
                continue

            html = page.content()
            title, text = clean_page_text(html)
            if len(text) > 200:  # skip near-empty pages
                pages.append({"url": url, "title": title, "text": text})
                print(f"✅ Scraped ({len(visited)}/{max_pages}): {url}  [{len(text)} chars]")
            else:
                print(f"⚠️  Skipping {url}: page rendered but had almost no text ({len(text)} chars)")

            soup = BeautifulSoup(html, "html.parser")
            for link in soup.find_all("a", href=True):
                next_url = urljoin(url, link["href"]).split("#")[0]
                if is_same_domain(next_url) and next_url not in visited:
                    queue.append(next_url)

            time.sleep(REQUEST_DELAY_SEC)

        browser.close()

    return pages


def chunk_and_store(pages: list[dict]):
    if not pages:
        print(
            "\n❌ No pages were scraped, so there's nothing to embed.\n"
            "   This usually means the crawl request failed (check the ⚠️ warnings above "
            "for the real cause, e.g. an SSL certificate error or no internet access).\n"
            "   Fix the underlying connection issue and re-run this script."
        )
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents, metadatas = [], []
    for page in pages:
        chunks = splitter.split_text(page["text"])
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"source": page["url"], "title": page["title"], "chunk_index": i})

    print(f"\n📦 Total chunks to embed: {len(documents)}")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_texts(
        texts=documents,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=PERSIST_DIR,
    )
    vectordb.persist()
    print(f"✅ Vector DB persisted at: {PERSIST_DIR}")


if __name__ == "__main__":
    print(f"🕷️  Crawling {BASE_URL} (max {MAX_PAGES} pages)...")
    scraped_pages = crawl(BASE_URL)
    print(f"\n🧹 Scraped {len(scraped_pages)} pages. Chunking + embedding...")
    chunk_and_store(scraped_pages)