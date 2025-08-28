"""
search_engine.py
----------------
A small, dependency-light search engine over Coventry Pure publications.

Design goals:
- Keep it simple, fast, and readable.
- Be robust to messy JSON (authors can be str/list/dict, links can be under different keys).
- Optional NLTK stemming/stopwords; degrade gracefully if NLTK is unavailable.
- Support small fielded filters in queries: author:..., year:YYYY or year:YYYY..YYYY
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

# ---------------- Optional NLTK ----------------
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    _HAVE_NLTK = True
except Exception:
    _HAVE_NLTK = False

# ---------------- Scikit-learn ----------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# IO
# ============================================================

def load_publications(
    filepath_primary: str = "data/publications.json",
    filepath_fallback: str = "data/publications_detailed.json"
) -> List[Dict[str, Any]]:
    """
    Load publications from JSON. Use a fallback file if the primary is missing.
    Returns a list of dicts (possibly empty).
    """
    try:
        with open(filepath_primary, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        with open(filepath_fallback, "r", encoding="utf-8") as f:
            data = json.load(f)
    return data or []


# ============================================================
# NLTK helpers (optional)
# ============================================================

def _ensure_nltk() -> bool:
    """
    Try to make sure NLTK resources exist; quietly degrade if not.
    """
    if not _HAVE_NLTK:
        return False
    try:
        _ = stopwords.words("english")
        nltk.word_tokenize("ok")
        return True
    except Exception:
        try:
            nltk.download("stopwords", quiet=True)
            nltk.download("punkt", quiet=True)
            nltk.word_tokenize("ok")
            return True
        except Exception:
            return False


_HAVE_DATA = _ensure_nltk()
STEM = PorterStemmer() if (_HAVE_NLTK and _HAVE_DATA) else None
STOP = set(stopwords.words("english")) if (_HAVE_NLTK and _HAVE_DATA) else {
    # small, pragmatic fallback
    "a","an","the","and","or","but","if","then","else","of","to","in","for","on","with","by","from","as","at","is",
    "are","was","were","be","been","being","it","this","that","these","those","i","you","he","she","we","they","them",
    "about","into","over","under","up","down","out","so","than","too","very"
}


# ============================================================
# Text normalization
# ============================================================

def _tokenize_basic(text: str) -> List[str]:
    """Lowercase, strip punctuation/numbers lightly, split on whitespace, drop stopwords."""
    if not text:
        return []
    text = text.lower()
    # keep digits because some titles contain years/ids; drop most punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    toks = text.split()
    return [t for t in toks if t and t not in STOP and len(t) > 1]

def preprocess_text(text: str) -> str:
    """
    Convert raw text into a preprocessed token string for vectorization.
    - lowercasing
    - punctuation stripping
    - optional Porter stemming
    """
    toks = _tokenize_basic(text)
    if STEM:
        toks = [STEM.stem(t) for t in toks]
    return " ".join(toks)

def _authors_as_names(auth: Any) -> List[str]:
    """
    Normalize authors to a list of names (strings) for indexing.
    We DO NOT mutate the original structure in the stored record.
    """
    if not auth:
        return []
    if isinstance(auth, str):
        s = auth.strip()
        return [s] if s else []
    if isinstance(auth, dict):
        n = str(auth.get("name", "")).strip()
        return [n] if n else []
    if isinstance(auth, list):
        out: List[str] = []
        for a in auth:
            if isinstance(a, dict):
                n = str(a.get("name", "")).strip()
                if n:
                    out.append(n)
            elif isinstance(a, str):
                s = a.strip()
                if s:
                    out.append(s)
        return out
    return []


# ============================================================
# Record normalization
# ============================================================

@dataclass
class PubRecord:
    """
    Normalized publication record used by the engine.
    We keep the original fields where possible and add our normalized ones.
    """
    raw: Dict[str, Any]
    title: str
    abstract: str
    authors_field: Any        # keep raw structure (list/dict/str)
    authors_index: List[str]  # names only, for indexing
    date: str
    link: str                 # best-effort primary link
    page_url: str             # safe fallback (often equals link)

def _pick_link(d: Dict[str, Any]) -> Tuple[str, str]:
    """Return (link, page_url) with sane fallbacks."""
    link = (
        d.get("link")
        or d.get("page_url")
        or d.get("url")
        or d.get("pageUrl")
        or ""
    )
    page_url = d.get("page_url") or link
    return str(link), str(page_url)

def _normalize_record(d: Dict[str, Any]) -> PubRecord:
    title = str(d.get("title") or "").strip()
    abstract = str(d.get("abstract") or d.get("summary") or "").strip()
    date = str(d.get("date") or d.get("published_date") or d.get("year") or "").strip()
    authors_field = d.get("authors", [])
    authors_idx = _authors_as_names(authors_field)
    link, page_url = _pick_link(d)
    return PubRecord(
        raw=d,
        title=title,
        abstract=abstract,
        authors_field=authors_field,
        authors_index=authors_idx,
        date=date,
        link=link,
        page_url=page_url
    )


# ============================================================
# Query parsing (author/year filters)
# ============================================================

_AUTHOR_RE = re.compile(r'author:\s*"(.*?)"|author:\s*([^\s]+)', re.IGNORECASE)
_YEAR_RE   = re.compile(r'year:\s*(\d{4})(?:\.\.(\d{4}))?', re.IGNORECASE)

@dataclass
class QueryFilters:
    author: Optional[str] = None   # substring match on author names (case-insensitive)
    year_from: Optional[int] = None
    year_to: Optional[int] = None

def _parse_filters(q: str) -> Tuple[str, QueryFilters]:
    """
    Extract simple fielded filters from the query string:
      - author:"Full Name" or author:lastname
      - year:YYYY or year:YYYY..YYYY
    Returns (cleaned_query, filters).
    """
    filters = QueryFilters()

    # author
    def _author_sub(m: re.Match) -> str:
        a = m.group(1) or m.group(2) or ""
        filters.author = a.strip().lower()
        return " "

    q = _AUTHOR_RE.sub(_author_sub, q)

    # year / year range
    def _year_sub(m: re.Match) -> str:
        y1, y2 = m.group(1), m.group(2)
        if y1 and not y2:
            filters.year_from = filters.year_to = int(y1)
        elif y1 and y2:
            y1i, y2i = int(y1), int(y2)
            filters.year_from = min(y1i, y2i)
            filters.year_to   = max(y1i, y2i)
        return " "

    q = _YEAR_RE.sub(_year_sub, q)
    return q.strip(), filters


# ============================================================
# Engine
# ============================================================

class SearchEngine:
    """
    Tiny in-memory search engine:
    - Build: normalize records → build preprocessed text → TF-IDF matrix
    - Query: optional filters → cosine similarity → top-k
    """

    def __init__(self, publications: List[Dict[str, Any]]):
        # Normalize input records (robust to varying JSON shapes)
        self.records: List[PubRecord] = [_normalize_record(p) for p in publications]

        # Build searchable blobs (title + authors + abstract), preprocessed once.
        blobs: List[str] = []
        for r in self.records:
            title = preprocess_text(r.title)
            authors_txt = preprocess_text(" ".join(r.authors_index))
            abstract = preprocess_text(r.abstract)
            blobs.append(f"{title} {authors_txt} {abstract}".strip())

        # Vectorizer consumes *already tokenized strings* → fast, no extra work at query time.
        # We set tokenizer=str.split to avoid re-tokenization heuristics and keep results stable.
        self.vectorizer = TfidfVectorizer(
            tokenizer=str.split,    # use our preprocessed tokens
            preprocessor=None,
            lowercase=False,
            ngram_range=(1, 2),     # light n-grams help on short titles
            min_df=1,
            sublinear_tf=True,
            norm="l2"
        )
        self.tfidf = self.vectorizer.fit_transform(blobs)

    # ------------- helpers -------------

    @staticmethod
    def _parse_year_from_record(r: PubRecord) -> Optional[int]:
        """
        Try to extract a 4-digit year from the record's date field or raw fields.
        """
        # Prefer explicit int-like fields if present
        for key in ("year", "published_year"):
            v = r.raw.get(key)
            if isinstance(v, int):
                return v
            if isinstance(v, str) and re.fullmatch(r"\d{4}", v.strip()):
                return int(v)
        # Generic scan on date string (e.g., "2023-06-01", "June 2022", "2021")
        text = r.date or ""
        m = re.search(r"\b(19|20)\d{2}\b", text)
        return int(m.group(0)) if m else None

    def _passes_filters(self, r: PubRecord, f: QueryFilters) -> bool:
        # author substring (case-insensitive) over names we indexed
        if f.author:
            joined = " ".join(r.authors_index).lower()
            if f.author not in joined:
                return False

        # year range
        if f.year_from or f.year_to:
            y = self._parse_year_from_record(r)
            if y is None:
                return False
            if f.year_from and y < f.year_from:
                return False
            if f.year_to and y > f.year_to:
                return False
        return True

    # ------------- public API -------------

    def search(self, query: str, k: int = 50) -> List[Dict[str, Any]]:
        """
        Search the collection.
        - Supports author/year filters via fielded syntax inside `query`.
        - Returns up to k results, each a small dict of safe fields.

        Example filters:
            "inflation targeting author:woodford year:2018..2024"
            "bank competition author:\"Jane Smith\" year:2022"
        """
        if not query or not query.strip():
            return []

        # Parse and strip filters out of the lexical query
        cleaned_q, filters = _parse_filters(query)

        # If the cleaned query collapses to empty, keep a tiny token to avoid an empty vector
        cleaned_q = cleaned_q if cleaned_q else "*"

        q_vec = self.vectorizer.transform([preprocess_text(cleaned_q)])
        sims = cosine_similarity(q_vec, self.tfidf).flatten()

        # Take a small pool (top 5k or corpus size) before applying filters to keep quality high.
        # For small corpora, slicing is cheap; for larger corpora, adjust the pool down.
        pool = min(len(self.records), max(k * 5, k))
        top_idx = sims.argsort()[-pool:][::-1]

        results: List[Dict[str, Any]] = []
        for i in top_idx:
            score = float(sims[i])
            if score < 0.01:   # ignore near-noise hits
                continue

            rec = self.records[i]
            if not self._passes_filters(rec, filters):
                continue

            # Build a stable, UI-friendly payload
            year = self._parse_year_from_record(rec)
            payload = {
                "title": rec.title,
                "link": rec.link or rec.page_url or "",
                "authors": rec.authors_field,   # preserve raw authors (names+profile URLs if present)
                "date": rec.date or (str(year) if year else ""),
                "abstract": rec.abstract,
                "score": round(score, 3),
            }
            results.append(payload)
            if len(results) >= k:
                break

        # Optional: tie-break by recency on equal scores (stable small boost)
        results.sort(key=lambda x: (x["score"], _safe_year(x.get("date"))), reverse=True)
        return results


# ============================================================
# Small utility
# ============================================================

def _safe_year(s: Any) -> int:
    """
    Extract a year integer from various date string shapes for sorting only.
    """
    if isinstance(s, int):
        return s
    if isinstance(s, str):
        m = re.search(r"\b(19|20)\d{2}\b", s)
        return int(m.group(0)) if m else -1
    return -1


# ============================================================
# Script helper (optional usage)
# ============================================================

if __name__ == "__main__":
    # Quick manual test:
    pubs = load_publications()
    engine = SearchEngine(pubs)
    q = 'inflation author:"john" year:2019..2025'
    for r in engine.search(q, k=10):
        print(r["score"], r["date"], r["title"])
