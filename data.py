"""
Extract (title, article) pairs from cc_news.

This is the closest open-source approximation to Exa's training signal:
  query    = news article title (descriptive headline written to attract readers)
  positive = article body text (the actual content the title describes)

This mirrors how Exa trains: someone shares a link with a descriptive title,
and the model learns to predict the linked document from that description.
"""

import os
import random
from datasets import load_dataset, Dataset
from tqdm import tqdm


MIN_QUERY_LEN = 30
MIN_TEXT_LEN  = 200
MAX_TEXT_LEN  = 2000  # first ~2000 chars of article body

# Description is ad copy / spam if it contains these
AD_MARKERS = [
    "click here", "sign up", "subscribe now", "free trial", "buy now",
    "limited time", "unleash your", "become a ", "create stunning",
    "100%", "best price",
]

# Patterns that indicate a login wall / paywalled content
LOGIN_MARKERS = [
    "login", "log in", "sign in", "subscribe", "paid subscriber",
    "premium web access", "create an account", "register to read",
]

# Patterns that indicate a noisy title
TITLE_NOISE = [
    r"[-–]\s*\w+\.\w+",                        # domain suffix: "story - wave3.com"
    r"\|\s*\w+\.\w+",                           # pipe + domain: "story | bbc.com"
    r"[-–]\s+\w[\w\s]+\d+\s+News\b",           # "story - WAFB 9 News Baton Rouge"
    r"\(VIDEO\)|\(PHOTOS?\)|\(WATCH\)|\(UPDATE\)",  # media tags
    r"^\[",                                     # starts with bracket
]

import re as _re
TITLE_NOISE_RE = _re.compile("|".join(TITLE_NOISE), _re.IGNORECASE)


def clean(text: str) -> str:
    return " ".join(text.split())


def strip_domain_suffix(title: str) -> str:
    """Remove trailing '- site.com - Section Name' from titles."""
    return _re.sub(r"\s*[-|–].*\.\w{2,4}.*$", "", title).strip()


def best_query(title: str, description: str) -> str:
    """
    Prefer description over title as the query — it's more natural and
    reads like how a person describes an article they want to find.
    Fall back to title if description is absent, too short, or ad copy.
    """
    desc = clean(description or "")
    if (
        len(desc) >= MIN_QUERY_LEN
        and desc.lower() != title.lower()             # not a duplicate of title
        and not any(m in desc.lower() for m in AD_MARKERS)
    ):
        return desc
    return title


def is_valid(query: str, text: str) -> bool:
    if not query or not text:
        return False
    if len(query) < MIN_QUERY_LEN:
        return False
    if len(text) < MIN_TEXT_LEN:
        return False
    if query.strip().isdigit():
        return False
    # Drop paywalled / login-walled articles
    text_preview = text.lower()[:300]
    if any(marker in text_preview for marker in LOGIN_MARKERS):
        return False
    # Drop titles with noisy patterns (domain names, media tags)
    if TITLE_NOISE_RE.search(query):
        return False
    # Drop pipe separators
    if "|" in query:
        return False
    return True


def load_pairs(max_pairs: int = 50_000, seed: int = 42) -> Dataset:
    """
    Load cc_news and return a HuggingFace Dataset with columns:
      query    — article headline
      positive — article body text
      domain   — source domain (e.g. bbc.com)
    """
    print("Loading cc_news (streaming)...")
    ds = load_dataset(
        "cc_news",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    pairs = []
    scanned = 0

    for row in tqdm(ds, desc="Scanning articles"):
        scanned += 1
        title = clean(row.get("title", "") or "")
        desc  = clean(row.get("description", "") or "")
        text  = clean(row.get("text",  "") or "")
        query = strip_domain_suffix(best_query(title, desc))

        if not is_valid(query, text):
            continue

        pairs.append({
            "query":    query,
            "positive": text[:MAX_TEXT_LEN],
            "domain":   row.get("domain", ""),
        })

        if len(pairs) >= max_pairs:
            break

    random.seed(seed)
    random.shuffle(pairs)

    print(f"\nScanned {scanned:,} articles → {len(pairs):,} valid pairs")

    if pairs:
        avg_q = sum(len(p["query"])    for p in pairs) / len(pairs)
        avg_p = sum(len(p["positive"]) for p in pairs) / len(pairs)
        print(f"Avg query length:    {avg_q:.0f} chars")
        print(f"Avg positive length: {avg_p:.0f} chars")

    return Dataset.from_list(pairs)


if __name__ == "__main__":
    dataset = load_pairs(max_pairs=50_000)
    os.makedirs("data", exist_ok=True)
    dataset.save_to_disk("data/cc_news_pairs")
    print(f"\nSaved {len(dataset):,} pairs to data/cc_news_pairs")
