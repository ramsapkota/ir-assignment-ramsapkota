# app/main.py
import sys, os, math, time, subprocess, traceback
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse,FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# --- make repo root importable for search modules ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from search_engine import SearchEngine, load_publications
from search_engine_advance import AdvancedSearch

# --- ensure the custom tokenizer class exists at unpickle time ---
# (your TfidfVectorizer(tokenizer=LemmaTokenizer(...)) is defined in train_classifier.py)
try:
    import train_classifier  # noqa: F401
except Exception:
    pass

# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML = ROOT / "frontend" / "index.html"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "news_clf.joblib"

# ---------------------------------------------------------------------
# Load publications & init search engines once
all_publications = load_publications()
tfidf_engine = SearchEngine(all_publications)
bm25_engine = AdvancedSearch(
    all_publications, enable_rerank=True, rerank_top_k=75, synonym_expansion=True
)

# ---------------------------------------------------------------------
app = FastAPI(
    title="Search & Classification by Ram Sapkota",
    description="Vertical search (TF-IDF & BM25) + news subject classification.",
    version="4.4.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Defensive: make sure required NLTK resources exist at runtime
def _ensure_nltk() -> Optional[str]:
    try:
        import nltk
        # try find; if missing, download headlessly
        for p in ("punkt", "wordnet", "omw-1.4", "stopwords"):
            try:
                nltk.data.find(f"corpora/{p}")
            except LookupError:
                nltk.download(p, quiet=True)
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        return None
    except Exception as e:
        return f"NLTK unavailable: {type(e).__name__}: {e}"

# ---------------------------------------------------------------------
# Classifier: lazy-loaded, with explicit error surfacing
_clf = None
_classes: Optional[List[str]] = None
_meta: Dict[str, Any] = {}
_last_load_error: Optional[str] = None  # keep last failure reason

def _stable_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x -= np.max(x)
    ex = np.exp(x)
    s = ex.sum()
    return ex / s if s and np.isfinite(s) else np.ones_like(x) / len(x)

def _scores_to_probs(scores: np.ndarray) -> np.ndarray:
    scores = np.atleast_2d(scores)
    if scores.shape[1] == 1:  # degenerate binary margin
        scores = np.concatenate([-scores, scores], axis=1)
    return np.apply_along_axis(_stable_softmax, 1, scores)

def _predict_with_confidence(clf, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(texts)
        preds = np.argmax(probs, axis=1)
        return preds, probs
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(texts)
        probs = _scores_to_probs(scores)
        preds = np.argmax(probs, axis=1)
        return preds, probs
    preds = clf.predict(texts)
    c = len(getattr(clf, "classes_", [])) or 3
    probs = np.full((len(preds), c), 1.0 / max(c, 1), dtype=float)
    return preds, probs

def _load_classifier():
    """Load the sklearn pipeline and class order; keep detailed reason if it fails."""
    global _clf, _classes, _meta, _last_load_error
    if _clf is not None:
        return

    _last_load_error = _ensure_nltk()  # load/download NLTK if needed
    if not MODEL_PATH.exists():
        _last_load_error = f"Model file not found at {MODEL_PATH}"
        return

    try:
        payload = joblib.load(MODEL_PATH)  # requires train_classifier import above
        _clf = payload.get("pipeline")
        _classes = list(payload.get("labels", []))
        _meta = payload.get("meta", {}) or {}
        if hasattr(_clf, "classes_"):
            _classes = list(_clf.classes_)
        if not _clf:
            raise RuntimeError("Loaded payload has no 'pipeline'")
        if not _classes:
            # fall back to classes_ or 3 generic classes
            _classes = list(getattr(_clf, "classes_", [])) or ["politics", "business", "health"]
    except Exception as e:
        _last_load_error = f"{type(e).__name__}: {e}\n" + traceback.format_exc()
        _clf = None
        _classes = None
        _meta = {}

# ---------------------------------------------------------------------
# UI + health
@app.get("/", response_class=HTMLResponse)
def read_root():
    return HTMLResponse(content=INDEX_HTML.read_text(encoding="utf-8"), status_code=200)

@app.get("/styles.css")
def get_styles():
    css_path = ROOT / "frontend" / "styles.css"
    if css_path.exists():
        return FileResponse(css_path, media_type="text/css")
    return JSONResponse(status_code=404, content={"error": "CSS file not found"})

@app.get("/script.js")
def get_script():
    js_path = ROOT / "frontend" / "script.js"
    if js_path.exists():
        return FileResponse(js_path, media_type="application/javascript")
    return JSONResponse(status_code=404, content={"error": "JS file not found"})

@app.get("/healthz")
def healthz():
    _load_classifier()
    return {
        "ok": True,
        "pub_count": len(all_publications),
        "model_loaded": _clf is not None,
        "model_error": _last_load_error,
        "model_algo": _meta.get("algo"),
        "model_classes": _classes,
    }

@app.get("/model_info")
def model_info():
    _load_classifier()
    if _clf is None:
        return JSONResponse(status_code=503, content={"ok": False, "error": _last_load_error or "No model"})
    return {"ok": True, "algo": _meta.get("algo"), "classes": _classes, "params": _meta.get("params", {})}

# ---------------------------------------------------------------------
# Publications & search
@app.get("/publications/")
def get_all_publications(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1),
    engine: str = Query("bm25"),
):
    start, end = (page - 1) * page_size, page * page_size
    total = len(all_publications)
    return {
        "page": page, "page_size": page_size,
        "total_pages": math.ceil(total / page_size) if page_size else 0,
        "total_publications": total,
        "publications": all_publications[start:end],
        "engine": engine.lower(),
    }

@app.get("/search/")
def search_publications(
    q: Optional[str] = Query(None, min_length=1),
    query: Optional[str] = Query(None, min_length=1),
    engine: str = Query("bm25"),
    author: Optional[str] = Query(None),
    year: Optional[int] = Query(None),
    year_from: Optional[int] = Query(None),
    year_to: Optional[int] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    text = (q or query or "").strip()
    if not text and not any([author, year, year_from, year_to]):
        return {
            "query": "", "engine": engine.lower(),
            "reranker_active": False, "results_count": 0,
            "total_results": 0, "total_pages": 0,
            "page": page, "page_size": page_size,
            "search_time_ms": 0, "results": [],
        }

    t0 = time.perf_counter()
    eng = (engine or "bm25").strip().lower()

    if eng == "tfidf":
        hits = tfidf_engine.search(text, k=10000)
        total = len(hits)
        start, end = (page - 1) * page_size, min(page * page_size, total)
        results = hits[start:end]
        rerank = False
    else:
        payload = bm25_engine.search(
            text, author=author, year=year, year_from=year_from, year_to=year_to,
            page=page, page_size=page_size,
        )
        results = payload["results"]
        total = int(payload["total_results"])
        rerank = bool(
            getattr(bm25_engine, "enable_rerank", False)
            and getattr(bm25_engine, "_ce", None) is not None
        )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "query": text, "engine": eng, "reranker_active": rerank,
        "results_count": len(results), "total_results": total,
        "total_pages": math.ceil(total / page_size) if total else 0,
        "page": page, "page_size": page_size,
        "search_time_ms": elapsed_ms, "results": results,
    }

# ---------------------------------------------------------------------
# Classification
class ClassifyRequest(BaseModel):
    text: str

@app.post("/classify")
def classify(req: ClassifyRequest):
    _load_classifier()
    if _clf is None:
        reason = _last_load_error or "Model not available. Train it first."
        return JSONResponse(status_code=503, content={"ok": False, "error": reason})

    text = (req.text or "").strip()
    if not text:
        return {"ok": False, "error": "Empty text.", "label": None, "proba": {}}

    try:
        preds, probs = _predict_with_confidence(_clf, [text])
        classes = _classes or [f"class_{i}" for i in range(probs.shape[1])]
        proba = {cls: float(p) for cls, p in zip(classes, probs[0])}
        return {"ok": True, "label": classes[int(preds[0])], "proba": proba}
    except Exception as e:
        err = f"classify failed: {type(e).__name__}: {e}"
        # include short traceback tail for debugging
        tb = traceback.format_exc().splitlines()[-3:]
        return JSONResponse(status_code=500, content={"ok": False, "error": err, "trace": tb})

# ---------------------------------------------------------------------
# Optional: retrain helper
@app.post("/retrain")
def retrain():
    try:
        cmd = [sys.executable, str(ROOT / "train_classifier.py"), "--model", "all", "--use_lemmatization", "--ngram_max", "3"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        # clear in-memory model so it reloads on next /classify
        global _clf, _classes, _meta, _last_load_error
        _clf = None; _classes = None; _meta = {}; _last_load_error = None
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
