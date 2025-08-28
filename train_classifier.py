# train_classifier.py
import os, glob, json, argparse, random, re
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import joblib

# ---------------- Config ----------------
DATA_DIR = Path("data/classification")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "news_clf.joblib"
SUMMARY_PATH = MODEL_DIR / "news_clf_summary.json"
CATEGORIES = ["politics", "business", "health"]
RANDOM_STATE = 42

# ---------------- Optional Lemmatization (NLTK) ----------------
# Lightweight, no external models beyond NLTK data.
# If you haven't downloaded NLTK resources yet:
#   pip install nltk==3.9
#   python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('stopwords')"
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    _NLTK_AVAILABLE = True
except Exception:
    _NLTK_AVAILABLE = False

_WORD_RE = re.compile(r"[A-Za-z]+")
if _NLTK_AVAILABLE:
    try:
        _STOP = set(stopwords.words("english"))
    except Exception:
        _STOP = set()
    _LEM = WordNetLemmatizer()

class LemmaTokenizer:
    """Regex tokenize -> lowercase -> stopword filter -> WordNet lemmatize."""
    def __init__(self, lowercase=True, remove_stops=True, keep_alpha=True):
        self.lowercase = lowercase
        self.remove_stops = remove_stops
        self.keep_alpha = keep_alpha

    def __call__(self, doc: str):
        if not _NLTK_AVAILABLE:
            # Fallback: basic alpha tokenization, no lemmatization
            text = doc.lower() if self.lowercase else doc
            return _WORD_RE.findall(text)

        text = doc.lower() if self.lowercase else doc
        toks = _WORD_RE.findall(text) if self.keep_alpha else nltk.word_tokenize(text)
        out = []
        for w in toks:
            if self.remove_stops and w in _STOP:
                continue
            lemma = _LEM.lemmatize(w)
            if lemma:
                out.append(lemma)
        return out

# ---------------- Data ----------------
def load_dataset() -> Tuple[List[str], List[str]]:
    texts, labels = [], []
    for cat in CATEGORIES:
        folder = DATA_DIR / cat
        if not folder.exists():
            continue
        for fp in glob.glob(str(folder / "*.txt")):
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read().strip()
                if txt:
                    texts.append(txt)
                    labels.append(cat)
            except Exception as e:
                print(f"[warn] failed reading {fp}: {e}")
    return texts, labels

# ---------------- Pipelines ----------------
def build_pipeline(
    algo: str,
    min_df: int = 2,
    max_features: int = 80_000,
    ngram_max: int = 2,
    use_lemmatization: bool = False,
    nb_alpha: float = 0.3,
    lr_C: float = 2.0,
    svm_C: float = 1.0,
) -> Pipeline:
    """
    algo: 'nb' | 'lr' | 'svm'
    """
    tokenizer = LemmaTokenizer() if use_lemmatization else None

    # IMPORTANT:
    # - If providing a custom tokenizer, do NOT also pass stop_words="english".
    # - token_pattern=None prevents scikit-learn from warning when custom tokenizer is used.
    vec_kwargs = dict(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        ngram_range=(1, ngram_max),
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    if tokenizer is not None:
        vec_kwargs.update(dict(tokenizer=tokenizer, token_pattern=None))
    else:
        vec_kwargs.update(dict(stop_words="english"))

    vec = TfidfVectorizer(**vec_kwargs)

    if algo == "nb":
        clf = MultinomialNB(alpha=nb_alpha)
    elif algo == "lr":
        clf = LogisticRegression(
            solver="saga",
            max_iter=2000,
            n_jobs=-1,
            C=lr_C,
            class_weight=None  # set to 'balanced' if classes are imbalanced
        )
    elif algo == "svm":
        clf = LinearSVC(C=svm_C)
    else:
        raise ValueError(f"Unknown algo: {algo}")

    return Pipeline([("tfidf", vec), ("clf", clf)])

# ---------------- Training & Eval ----------------
def evaluate_cv(pipe: Pipeline, X: List[str], y: List[str], folds: int = 5) -> Dict[str, float]:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    cv = cross_validate(
        pipe, X, y, cv=skf,
        scoring=["f1_macro", "accuracy"],
        n_jobs=-1, return_train_score=False
    )
    return {
        "cv_f1_macro_mean": float(np.mean(cv["test_f1_macro"])),
        "cv_f1_macro_std": float(np.std(cv["test_f1_macro"])),
        "cv_acc_mean": float(np.mean(cv["test_accuracy"])),
        "cv_acc_std": float(np.std(cv["test_accuracy"])),
    }

def heldout_report(pipe: Pipeline, X_te: List[str], y_te: List[str]) -> Dict:
    y_hat = pipe.predict(X_te)
    acc = accuracy_score(y_te, y_hat)
    f1m = f1_score(y_te, y_hat, average="macro")
    report = classification_report(y_te, y_hat, labels=CATEGORIES, zero_division=0, digits=4)
    cm = confusion_matrix(y_te, y_hat, labels=CATEGORIES).tolist()
    return {
        "test_accuracy": float(acc),
        "test_f1_macro": float(f1m),
        "classification_report": report,
        "confusion_matrix_labels": CATEGORIES,
        "confusion_matrix": cm
    }

def train_and_eval(
    algo: str,
    X_tr: List[str], y_tr: List[str],
    X_te: List[str], y_te: List[str],
    folds: int,
    **pipe_kwargs
):
    pipe = build_pipeline(algo, **pipe_kwargs)

    print(f"\n[cv] {algo.upper()} on training fold...")
    cv_stats = evaluate_cv(pipe, X_tr, y_tr, folds=folds)
    print(f"  F1-macro {cv_stats['cv_f1_macro_mean']:.4f} ± {cv_stats['cv_f1_macro_std']:.4f} | "
          f"ACC {cv_stats['cv_acc_mean']:.4f} ± {cv_stats['cv_acc_std']:.4f}")

    pipe.fit(X_tr, y_tr)
    held = heldout_report(pipe, X_te, y_te)
    print("\n[held-out] accuracy:", f"{held['test_accuracy']:.4f}")
    print("[held-out] macro-F1:", f"{held['test_f1_macro']:.4f}")
    print(held["classification_report"])
    print("Confusion matrix (rows=true, cols=pred):")
    for row, label in zip(held["confusion_matrix"], held["confusion_matrix_labels"]):
        print(f"  {label[:8]:>8s} | {row}")

    return pipe, cv_stats, held

def main(
    test_size: float,
    folds: int,
    model: str,          # 'nb' | 'lr' | 'svm' | 'all'
    min_df: int,
    max_features: int,
    ngram_max: int,
    use_lemmatization: bool,
    alpha: float,
    lr_C: float,
    svm_C: float,
):
    random.seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)

    X, y = load_dataset()
    n = len(X)
    assert n >= 100, f"Need at least 100 docs; found {n}"
    counts = {c: y.count(c) for c in CATEGORIES}
    print(f"[data] Loaded {n} documents across classes: {counts}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    print(f"[split] train={len(X_tr)} test={len(X_te)}  (test_size={test_size})")

    # Decide which models to run
    to_run = [model] if model in {"nb","lr","svm"} else ["nb","lr","svm"]

    results = {}
    best = {"name": None, "pipe": None, "cv": None, "held": None, "f1": -1}

    for algo in to_run:
        pipe, cv_stats, held = train_and_eval(
            algo, X_tr, y_tr, X_te, y_te, folds,
            min_df=min_df,
            max_features=max_features,
            ngram_max=ngram_max,
            use_lemmatization=use_lemmatization,
            nb_alpha=alpha,
            lr_C=lr_C,
            svm_C=svm_C,
        )
        f1m = held["test_f1_macro"]
        results[algo] = {"cv": cv_stats, "heldout": {"f1_macro": f1m, "accuracy": held["test_accuracy"]}}
        if f1m > best["f1"]:
            best.update({"name": algo, "pipe": pipe, "cv": cv_stats, "held": held, "f1": f1m})

    # Persist best pipeline + summary
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": best["pipe"],
        "labels": CATEGORIES,
        "meta": {
            "algo": best["name"],
            "params": {
                "min_df": min_df, "max_features": max_features, "ngram_max": ngram_max,
                "use_lemmatization": use_lemmatization,
                "nb_alpha": alpha, "lr_C": lr_C, "svm_C": svm_C
            },
            "cv": best["cv"],
            "heldout": {
                "test_accuracy": best["held"]["test_accuracy"],
                "test_f1_macro": best["held"]["test_f1_macro"],
            }
        }
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"\n[saved] {MODEL_PATH.resolve()}")

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "labels": CATEGORIES,
        "class_counts": counts,
        "split": {"train": len(X_tr), "test": len(X_te), "test_size": test_size},
        "best_model": best["name"],
        "per_model": results,
        "cv_best": best["cv"],
        "heldout": best["held"],
        "notes": {
            "vectorizer": "TF-IDF word ngrams",
            "lemmatization": "NLTK WordNet (if installed) in custom tokenizer"
        }
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {SUMMARY_PATH.resolve()}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--model", type=str, choices=["nb","lr","svm","all"], default="all",
                   help="Train a specific model or run all and pick best")
    p.add_argument("--min_df", type=int, default=5)
    p.add_argument("--max_features", type=int, default=120000)
    p.add_argument("--ngram_max", type=int, default=2, help="2 = uni+bi, 3 = uni+bi+tri")
    p.add_argument("--use_lemmatization", action="store_true", help="Enable NLTK lemmatization in tokenizer")
    # Model hypers
    p.add_argument("--alpha", type=float, default=0.3, help="MultinomialNB alpha")
    p.add_argument("--lr_C", type=float, default=2.0, help="LogReg regularization C (higher=less regularization)")
    p.add_argument("--svm_C", type=float, default=1.0, help="LinearSVC C")
    args = p.parse_args()

    main(
        test_size=args.test_size,
        folds=args.folds,
        model=args.model,
        min_df=args.min_df,
        max_features=args.max_features,
        ngram_max=args.ngram_max,
        use_lemmatization=args.use_lemmatization,
        alpha=args.alpha,
        lr_C=args.lr_C,
        svm_C=args.svm_C,
    )
