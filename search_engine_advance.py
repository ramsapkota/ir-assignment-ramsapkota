# search_engine_advance.py
# Advanced search: Multi-field BM25 + Cross-Encoder reranker with fusion & robust scoring
from __future__ import annotations

import math
import re
import time
import hashlib
from collections import Counter, defaultdict, OrderedDict
from typing import Any, Dict, List, Optional, Tuple

# -------- Optional NLTK (tokenization/stopwords/stemming); degrades gracefully --------
try:
    import nltk
    from nltk.corpus import stopwords, wordnet as wn
    from nltk.stem import PorterStemmer

    _HAVE_NLTK = True
except Exception:
    _HAVE_NLTK = False

# -------- Optional Cross-Encoder (Sentence-Transformers) for reranking --------
try:
    from sentence_transformers import CrossEncoder  # type: ignore
    _HAVE_SBERT = True
except Exception:
    _HAVE_SBERT = False


# ============================== NLTK bootstrap ==============================
def _ensure_nltk():
    if not _HAVE_NLTK:
        return
    try:
        # Quick smoke test; download if missing
        _ = stopwords.words("english")
        nltk.word_tokenize("ok")
        wn.synsets("search")
    except Exception:
        nltk.download("stopwords", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("wordnet", quiet=True)


if _HAVE_NLTK:
    _ensure_nltk()
    STOP = set(stopwords.words("english"))
    STEM = PorterStemmer()
else:
    STOP = {
        "a", "an", "the", "and", "or", "in", "of", "for", "to",
        "with", "on", "at", "by", "from", "as", "is", "are", "was", "were"
    }
    STEM = None

TOKEN_RE = re.compile(r"[a-z0-9]+")

def _normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokens(s: str) -> List[str]:
    if not s:
        return []
    s = _normalize_text(s)
    toks = TOKEN_RE.findall(s)
    if STEM:
        return [STEM.stem(t) for t in toks if t not in STOP and len(t) > 1]
    return [t for t in toks if t not in STOP and len(t) > 1]

def _authors_as_objects(v: Any) -> List[Dict[str, Optional[str]]]:
    out: List[Dict[str, Optional[str]]] = []
    if not v:
        return out
    if isinstance(v, str):
        n = v.strip()
        if n:
            out.append({"name": n, "profile": None})
    elif isinstance(v, dict):
        n = str(v.get("name", "")).strip()
        p = str(v.get("profile", "") or "").strip() or None
        if n:
            out.append({"name": n, "profile": p})
    elif isinstance(v, list):
        for a in v:
            if isinstance(a, dict):
                n = str(a.get("name", "")).strip()
                p = str(a.get("profile", "") or "").strip() or None
                if n:
                    out.append({"name": n, "profile": p})
            elif isinstance(a, str):
                n = a.strip()
                if n:
                    out.append({"name": n, "profile": None})
    return out

def _extract_year(date_or_year: Any) -> Optional[int]:
    if not date_or_year:
        return None
    s = str(date_or_year)
    m = re.search(r"\b(19|20)\d{2}\b", s)
    if not m:
        return None
    y = int(m.group(0))
    return y if 1900 <= y <= 2100 else None


# ============================== Fielded query parsing ==============================
QUOTE_RE = re.compile(r'"([^"]+)"')
YEAR_RANGE_RE = re.compile(r"\byear:(\d{4})(?:\.\.(\d{4}))?\b", re.I)
AUTHOR_RE = re.compile(r'\bauthor:(".*?"|\S+)\b', re.I)
DOI_RE = re.compile(r'\bdoi:([^\s"]+)\b', re.I)

class ParsedQuery(Tuple):
    def __new__(cls, free_text: str, phrases: List[str],
                author_terms: List[str], year_from: Optional[int],
                year_to: Optional[int], doi: Optional[str]):
        return tuple.__new__(cls, (free_text, phrases, author_terms, year_from, year_to, doi))

    @property
    def free_text(self) -> str: return self[0]
    @property
    def phrases(self) -> List[str]: return self[1]
    @property
    def author_terms(self) -> List[str]: return self[2]
    @property
    def year_from(self) -> Optional[int]: return self[3]
    @property
    def year_to(self) -> Optional[int]: return self[4]
    @property
    def doi(self) -> Optional[str]: return self[5]

def parse_query(q: str) -> ParsedQuery:
    q = " ".join((q or "").split())
    phrases = [m.group(1) for m in QUOTE_RE.finditer(q)]

    author_terms: List[str] = []
    for m in AUTHOR_RE.finditer(q):
        term = m.group(1).strip('"').strip()
        if term:
            author_terms.append(term)
    q = AUTHOR_RE.sub(" ", q)

    doi = None
    mdoi = DOI_RE.search(q)
    if mdoi:
        doi = mdoi.group(1).strip().lower()
        q = DOI_RE.sub(" ", q)

    year_from = year_to = None
    mrange = YEAR_RANGE_RE.search(q)
    if mrange:
        year_from = int(mrange.group(1))
        year_to = int(mrange.group(2) or mrange.group(1))
        q = YEAR_RANGE_RE.sub(" ", q)

    free_text = QUOTE_RE.sub(" ", q).strip()
    return ParsedQuery(free_text, phrases, author_terms, year_from, year_to, doi)


# ============================== BM25 per-field index ==============================
class _BM25Field:
    def __init__(self, docs_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs_tokens)
        self.doc_len = [len(toks) for toks in docs_tokens]
        self.avgdl = sum(self.doc_len) / max(1, self.N)
        self.tf: List[Counter] = [Counter(toks) for toks in docs_tokens]
        df = Counter(term for c in self.tf for term in c.keys())
        self.idf = {term: math.log((self.N - dfi + 0.5) / (dfi + 0.5) + 1.0) for term, dfi in df.items()}
        self.inv = defaultdict(list)
        for i, c in enumerate(self.tf):
            for term, f in c.items():
                self.inv[term].append((i, f))

    def score_docs_for_terms(self, q_terms: List[Tuple[str, float]]) -> Dict[int, float]:
        scores: Dict[int, float] = defaultdict(float)
        if not self.N:
            return scores
        for term, weight in q_terms:
            idf = self.idf.get(term, 0.0)
            if idf <= 0:
                continue
            postings = self.inv.get(term)
            if not postings:
                continue
            for doc_id, tf in postings:
                dl = self.doc_len[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s = idf * (tf * (self.k1 + 1)) / max(denom, 1e-6)
                scores[doc_id] += weight * s
        return scores


# ============================== Tiny LRU for reranker scores ==============================
class _LRU:
    def __init__(self, maxsize: int = 4096):
        self.maxsize = maxsize
        self.d: OrderedDict[str, float] = OrderedDict()

    def get(self, key: str) -> Optional[float]:
        val = self.d.get(key)
        if val is not None:
            self.d.move_to_end(key)
        return val

    def put(self, key: str, val: float):
        self.d[key] = val
        self.d.move_to_end(key)
        if len(self.d) > self.maxsize:
            self.d.popitem(last=False)

    @staticmethod
    def key(query: str, text: str) -> str:
        return hashlib.sha1((query + "||" + text).encode("utf-8")).hexdigest()


# ============================== Advanced Engine ==============================
class AdvancedSearch:
    """
    Advanced engine:
      1) Multi-field BM25 retrieval (title/authors/abstract with weights)
      2) Cross-Encoder reranking (optional)
      3) Score fusion + priors/bonuses:
         - normalized BM25 + normalized CE
         - recency prior (logistic)
         - title/phrase/author/DOI bonuses (capped)
    """

    # field weights for BM25 combination
    W_TITLE = 2.8
    W_AUTH = 1.6
    W_ABS = 1.0

    # fusion (tuneable)
    ALPHA_BM25 = 0.55
    BETA_CE = 0.45  # used if CE available

    # bonuses (caps)
    BONUS_TITLE_EXACT = 0.18
    BONUS_TITLE_SUBSTR = 0.10
    BONUS_PHRASE = 0.14
    BONUS_AUTHOR = 0.10
    BONUS_DOI = 0.30
    BONUS_RECENCY_WEIGHT = 0.20  # recency prior scaled by this

    def __init__(
        self,
        publications: List[Dict[str, Any]],
        enable_rerank: bool = True,
        rerank_top_k: int = 75,
        synonym_expansion: bool = True,
    ):
        # ---- Prep records ----
        self.records: List[Dict[str, Any]] = []
        title_docs: List[List[str]] = []
        author_docs: List[List[str]] = []
        abstr_docs: List[List[str]] = []

        for p in publications:
            title = p.get("title", "") or ""
            abstract = p.get("abstract", "") or ""
            authors_obj = _authors_as_objects(p.get("authors"))

            rec = {
                **p,
                "title": title,
                "abstract": abstract,
                "authors": authors_obj,
                "year": _extract_year(p.get("date") or p.get("published_date")),
                "_title_norm": _normalize_text(title),
                "_authors_norm": _normalize_text(" ".join(a["name"] for a in authors_obj)),
                "_abstract_norm": _normalize_text(abstract),
                "doi": str(p.get("doi") or p.get("DOI") or "").lower(),
                "link": p.get("link") or p.get("page_url") or p.get("url") or "",
            }
            self.records.append(rec)
            title_docs.append(_tokens(title))
            author_docs.append(_tokens(" ".join(a["name"] for a in authors_obj)))
            abstr_docs.append(_tokens(abstract))

        # ---- BM25 indexes ----
        self.idx_title = _BM25Field(title_docs)
        self.idx_auth = _BM25Field(author_docs)
        self.idx_abs = _BM25Field(abstr_docs)

        # ---- Cross-Encoder reranker ----
        self.enable_rerank = enable_rerank and _HAVE_SBERT
        self.rerank_top_k = int(max(1, rerank_top_k))
        self._ce = None
        if self.enable_rerank:
            try:
                self._ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except Exception as e:
                print(f"[warn] Cross-Encoder load failed: {e}. Rerank disabled.")
                self.enable_rerank = False

        self._cache = _LRU(4096)
        self._synonym_expansion = bool(synonym_expansion and _HAVE_NLTK)

        # for recency prior
        self.current_year = time.gmtime().tm_year

    # ------------------ helpers ------------------
    def _expand_query_terms(self, q: str) -> List[Tuple[str, float]]:
        """Stemmed tokens + (optional) up to 2 WordNet synonyms per token @ lower weight."""
        base = _tokens(q)
        if not self._synonym_expansion:
            return [(t, 1.0) for t in base]
        out: List[Tuple[str, float]] = []
        seen = set()
        for t in base:
            if not t or t in seen:
                continue
            out.append((t, 1.0)); seen.add(t)
            # careful with WordNet explosions
            syns = set()
            try:
                for syn in wn.synsets(t):
                    for lem in syn.lemmas():
                        w = lem.name().replace("_", " ").lower()
                        if STEM:
                            w = STEM.stem(w)
                        if w != t and w.isalpha():
                            syns.add(w)
            except Exception:
                pass
            for s in list(syns)[:2]:
                if s not in seen:
                    out.append((s, 0.6)); seen.add(s)
        return out

    def _get_filtered_indices(self, author, year, year_from, year_to, doi) -> set[int]:
        """Apply hard filters over normalized strings / parsed years."""
        all_idx = set(range(len(self.records)))
        if not any([author, year, year_from, year_to, doi]):
            return all_idx

        keep = all_idx
        if author:
            a_norm = _normalize_text(str(author))
            keep &= {i for i, r in enumerate(self.records) if a_norm and a_norm in r["_authors_norm"]}
        if year is not None:
            keep &= {i for i, r in enumerate(self.records) if r.get("year") == int(year)}
        if year_from is not None:
            yf = int(year_from)
            keep &= {i for i, r in enumerate(self.records) if r.get("year") and r["year"] >= yf}
        if year_to is not None:
            yt = int(year_to)
            keep &= {i for i, r in enumerate(self.records) if r.get("year") and r["year"] <= yt}
        if doi:
            d = str(doi).lower()
            keep &= {i for i, r in enumerate(self.records) if d and r["doi"] and d in r["doi"]}
        return keep

    @staticmethod
    def _norm(arr: List[float]) -> List[float]:
        """Robust min-max normalization to [0,1]."""
        if not arr:
            return []
        lo, hi = min(arr), max(arr)
        if hi - lo < 1e-9:
            return [0.0 for _ in arr]
        return [(x - lo) / (hi - lo) for x in arr]

    def _recency_prior(self, year: Optional[int]) -> float:
        """Smooth 0..1 prior; center ~ (current_year - 3). Missing year → ~0.5."""
        if year is None:
            return 0.5
        pivot = self.current_year - 3
        return 1.0 / (1.0 + math.exp(-float(year - pivot) / 2.0))

    # ------------------ public search ------------------
    # --- replace ONLY the search(...) method inside AdvancedSearch ---

    def search(
            self,
            q: str,
            *,
            author=None,
            year=None,
            year_from=None,
            year_to=None,
            k: Optional[int] = None,  # kept for backward-compat; ignored when page/page_size used
            page: int = 1,
            page_size: int = 50
    ) -> Dict[str, Any]:
        """
        Server-side pagination with exact total count.
        Returns: {"results": List[dict], "total_results": int}
        """
        q = (q or "").strip()
        pq = parse_query(q)

        # Merge inline filters with kwargs (kwargs win if provided)
        author_q = author if author is not None else (" ".join(pq.author_terms) if pq.author_terms else None)
        year_from_q = year_from if year_from is not None else pq.year_from
        year_to_q = year_to if year_to is not None else pq.year_to
        year_q = int(year) if year is not None else None
        doi_q = pq.doi

        has_any_constraint = any([q, author_q, year_q, year_from_q, year_to_q, doi_q])
        if not has_any_constraint:
            return {"results": [], "total_results": 0}

        # -------- Stage 1: fielded BM25 retrieval --------
        q_lex = " ".join(part for part in [pq.free_text] + pq.phrases + pq.author_terms if part)
        q_terms = self._expand_query_terms(q_lex)

        s_title = self.idx_title.score_docs_for_terms(q_terms)
        s_auth = self.idx_auth.score_docs_for_terms(q_terms)
        s_abs = self.idx_abs.score_docs_for_terms(q_terms)

        combined = defaultdict(float)
        for i, v in s_title.items(): combined[i] += self.W_TITLE * v
        for i, v in s_auth.items():  combined[i] += self.W_AUTH * v
        for i, v in s_abs.items():   combined[i] += self.W_ABS * v

        # Apply filters & keep positives
        allowed = self._get_filtered_indices(author_q, year_q, year_from_q, year_to_q, doi_q)
        items = [(i, float(s)) for i, s in combined.items() if i in allowed and s > 0.0]
        if not items:
            return {"results": [], "total_results": 0}

        # Sort by BM25 prior (desc) and compute exact total
        items.sort(key=lambda x: x[1], reverse=True)
        total_results = len(items)

        # Figure out slice range for current page
        page = max(1, int(page))
        page_size = max(1, int(page_size))
        start = (page - 1) * page_size
        end = min(start + page_size, total_results)
        if start >= total_results:
            return {"results": [], "total_results": total_results}

        # Rerank enough to cover this page
        # Pool should at least cover 'end' (the last item on this page), and be >= rerank_top_k
        pool_limit = max(self.rerank_top_k, end)
        candidates = items[:pool_limit]

        # Normalize BM25
        bm25_scores = [s for _, s in candidates]
        bm25_norm = self._norm(bm25_scores)

        # -------- Stage 2: Cross-Encoder reranking (optional) --------
        ce_norm: List[float] = [0.0] * len(candidates)
        if self.enable_rerank and self._ce is not None and pq.free_text:
            pairs_text: List[str] = []
            missing_idx: List[int] = []
            for idx, (doc_id, _) in enumerate(candidates):
                doc = self.records[doc_id]
                text = f"{doc['title']}. {doc['abstract']}".strip()
                key = _LRU.key(pq.free_text, text)
                cached = self._cache.get(key)
                if cached is None:
                    pairs_text.append(text)
                    missing_idx.append(idx)
                else:
                    ce_norm[idx] = cached

            if missing_idx:
                try:
                    pairs = [[pq.free_text, t] for t in pairs_text]
                    ce_raw = self._ce.predict(pairs, show_progress_bar=False)
                    ce_batch_norm = self._norm([float(s) for s in ce_raw])
                    for j, nscore in zip(missing_idx, ce_batch_norm):
                        ce_norm[j] = nscore
                        self._cache.put(_LRU.key(pq.free_text, pairs_text[missing_idx.index(j)]), nscore)
                except Exception as e:
                    print(f"[warn] rerank failed: {e}")

        # Bonuses / priors
        q_title = _normalize_text(pq.free_text)
        phrase_list = [p.lower() for p in pq.phrases]

        fused_scores: List[Tuple[int, float]] = []
        for rank_idx, (doc_id, _) in enumerate(candidates):
            r = self.records[doc_id]
            bonus = 0.0

            if q_title:
                if r["_title_norm"] == q_title:
                    bonus += self.BONUS_TITLE_EXACT
                elif q_title in r["_title_norm"]:
                    bonus += self.BONUS_TITLE_SUBSTR

            for ph in phrase_list:
                if ph and (ph in r["_title_norm"] or ph in r["_abstract_norm"]):
                    bonus += min(self.BONUS_PHRASE, 0.06 + 0.02 * len(ph.split()))

            if author_q:
                a_norm = _normalize_text(str(author_q))
                if a_norm and a_norm in r["_authors_norm"]:
                    bonus += self.BONUS_AUTHOR

            if doi_q and r["doi"] and doi_q in r["doi"]:
                bonus += self.BONUS_DOI

            recency = self._recency_prior(r.get("year"))
            bonus += self.BONUS_RECENCY_WEIGHT * recency

            fused = (
                    self.ALPHA_BM25 * bm25_norm[rank_idx]
                    + (self.BETA_CE * ce_norm[rank_idx] if self.enable_rerank else 0.0)
                    + bonus
            )
            fused_scores.append((doc_id, fused))

        # Final sort by fused score
        fused_scores.sort(key=lambda x: x[1], reverse=True)

        # Slice just the current page
        page_slice = fused_scores[start:end]

        # Build output
        def _highlight(text: str, terms: List[str]) -> str:
            if not text or not terms:
                return text or ""
            out = text
            for t in sorted(set(terms), key=len, reverse=True):
                if not t:
                    continue
                try:
                    out = re.sub(f"(?i)({re.escape(t)})", r"<mark>\1</mark>", out)
                except re.error:
                    pass
            return out

        highlight_terms = []
        if pq.free_text:
            highlight_terms.extend([w for w in pq.free_text.split() if len(w) > 1])
        highlight_terms.extend(pq.phrases)

        results: List[Dict[str, Any]] = []
        for doc_id, score in page_slice:
            r = self.records[doc_id]
            results.append({
                "title": r.get("title", ""),
                "link": r.get("link", ""),
                "authors": r.get("authors", []),
                "date": r.get("date", "") or r.get("published_date", ""),
                "abstract": _highlight(r.get("abstract", ""), highlight_terms),
                "score": round(float(score), 3),
                "doi": r.get("doi", ""),
                "oa_url": r.get("oa_url", "") or r.get("pdf", "") or "",
            })

        return {"results": results, "total_results": total_results}
