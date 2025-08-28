#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, time, re, unicodedata, difflib
from math import ceil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse, quote

# Selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Parallelism
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- Config ----------
PORTAL_ROOT = "https://pureportal.coventry.ac.uk"
PERSONS_PREFIX = "/en/persons/"
BASE_URL = f"{PORTAL_ROOT}/en/organisations/fbl-school-of-economics-finance-and-accounting/publications/"

# =========================== Chrome helpers ===========================
def build_chrome_options(headless: bool, legacy_headless: bool = False) -> Options:
    opts = Options()
    if headless:
        opts.add_argument("--headless" + ("" if legacy_headless else "=new"))
    opts.add_argument("--window-size=1366,900")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--lang=en-US")
    opts.add_argument("--disable-notifications")
    opts.add_argument("--no-first-run")
    opts.add_argument("--no-default-browser-check")
    opts.add_argument("--disable-extensions")
    opts.add_argument("--disable-popup-blocking")
    opts.add_argument("--disable-renderer-backgrounding")
    opts.add_argument("--disable-backgrounding-occluded-windows")
    opts.add_argument("--disable-features=CalculateNativeWinOcclusion,MojoVideoDecoder")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-logging", "enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.page_load_strategy = "eager"
    opts.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36")
    return opts

def make_driver(headless: bool, legacy_headless: bool = False) -> webdriver.Chrome:
    service = ChromeService(ChromeDriverManager().install(), log_output=os.devnull)
    driver = webdriver.Chrome(service=service, options=build_chrome_options(headless, legacy_headless))
    driver.set_page_load_timeout(40)
    try:
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        })
    except Exception:
        pass
    return driver

def accept_cookies_if_present(driver: webdriver.Chrome):
    try:
        btn = WebDriverWait(driver, 6).until(EC.presence_of_element_located((By.ID, "onetrust-accept-btn-handler")))
        driver.execute_script("arguments[0].click();", btn)
        time.sleep(0.2)
    except TimeoutException:
        pass
    except Exception:
        pass

# =========================== Utilities ===========================
FIRST_DIGIT = re.compile(r"\d")
NAME_PAIR = re.compile(r"[A-Z][A-Za-z'’\-]+,\s*(?:[A-Z](?:\.)?)(?:\s*[A-Z](?:\.)?)*", flags=re.UNICODE)
SPACE = re.compile(r"\s+")

def _uniq_str(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for x in seq:
        x = x.strip()
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

def _uniq_authors(objs: List[Dict[str, Optional[str]]]) -> List[Dict[str, Optional[str]]]:
    seen: set[Tuple[str, str]] = set()
    out: List[Dict[str, Optional[str]]] = []
    for o in objs:
        name = (o.get("name") or "").strip()
        profile = (o.get("profile") or "").strip()
        key = (name, profile)
        if name and key not in seen:
            seen.add(key)
            out.append({"name": name, "profile": profile or None})
    return out

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\w\s\-']", " ", s, flags=re.UNICODE).strip().lower()
    return SPACE.sub(" ", s)

def _is_person_profile_url(href: str) -> bool:
    """Accept only /en/persons/<slug> (reject directory / search / empty)."""
    if not href:
        return False
    try:
        u = urlparse(href)
    except Exception:
        return False
    if u.netloc and "coventry.ac.uk" not in u.netloc:
        return False
    path = (u.path or "").rstrip("/")
    if not path.startswith(PERSONS_PREFIX):
        return False
    slug = path[len(PERSONS_PREFIX):].strip("/")
    if not slug or slug.startswith("?"):
        return False
    return True

def _looks_like_person_name(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    bad = {"profiles", "persons", "people", "overview"}
    if t.lower() in bad:
        return False
    return ((" " in t) or ("," in t)) and sum(ch.isalpha() for ch in t) >= 4

# =========================== LISTING (Stage 1) ===========================
def scrape_listing_page(driver: webdriver.Chrome, page_idx: int) -> List[Dict]:
    url = f"{BASE_URL}?page={page_idx}"
    driver.get(url)
    accept_cookies_if_present(driver)
    try:
        WebDriverWait(driver, 15).until(
            lambda d: d.find_elements(By.CSS_SELECTOR, ".result-container h3.title a")
                      or "No results" in d.page_source
        )
    except TimeoutException:
        pass

    rows = []
    for c in driver.find_elements(By.CLASS_NAME, "result-container"):
        try:
            a = c.find_element(By.CSS_SELECTOR, "h3.title a")
            title = a.text.strip()
            link = a.get_attribute("href")
            if title and link:
                rows.append({"title": title, "link": link})
        except Exception:
            continue
    return rows

def gather_all_listing_links(max_pages: int, headless_listing: bool = False, legacy_headless: bool = False) -> List[Dict]:
    driver = make_driver(headless_listing, legacy_headless)
    try:
        driver.get(BASE_URL)
        accept_cookies_if_present(driver)
        all_rows: List[Dict] = []
        for i in range(max_pages):
            print(f"[LIST] Page {i+1}/{max_pages}")
            rows = scrape_listing_page(driver, i)
            if not rows:
                print(f"[LIST] Empty at page index {i}; stopping early.")
                break
            all_rows.extend(rows)
        uniq = {}
        for r in all_rows:
            uniq[r["link"]] = r
        return list(uniq.values())
    finally:
        try:
            driver.quit()
        except Exception:
            pass

# =========================== DETAIL (Stage 2) ===========================
def _maybe_expand_authors(driver: webdriver.Chrome):
    try:
        for b in driver.find_elements(
            By.XPATH,
            "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'show') or "
            "contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'more')]"
        )[:2]:
            try:
                driver.execute_script("arguments[0].scrollIntoView({block:'center'});", b)
                time.sleep(0.1)
                b.click()
                time.sleep(0.2)
            except Exception:
                continue
    except Exception:
        pass

def _authors_from_header_anchors(driver: webdriver.Chrome) -> List[Dict]:
    """
    Grab /en/persons/<slug> anchors ABOVE the tab bar (Overview etc.),
    reject directory link 'Profiles' and other nav chrome.
    """
    # Find Y threshold of tab bar
    tabs_y = None
    for xp in [
        "//a[normalize-space()='Overview']",
        "//nav[contains(@class,'tabbed-navigation')]",
        "//div[contains(@class,'navigation') and .//a[contains(.,'Overview')]]",
    ]:
        try:
            el = driver.find_element(By.XPATH, xp)
            tabs_y = el.location.get("y", None)
            if tabs_y:
                break
        except Exception:
            continue
    if tabs_y is None:
        tabs_y = 900  # conservative fallback

    candidates: List[Dict[str, Optional[str]]] = []
    seen = set()
    for a in driver.find_elements(By.CSS_SELECTOR, "a[href*='/en/persons/']"):
        try:
            y = a.location.get("y", 99999)
            if y >= tabs_y:
                continue
            href = (a.get_attribute("href") or "").strip()
            if not _is_person_profile_url(href):
                continue
            try:
                name = a.find_element(By.CSS_SELECTOR, "span").text.strip()
            except NoSuchElementException:
                name = (a.text or "").strip()
            if not _looks_like_person_name(name):
                continue
            key = (name, href)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({"name": name, "profile": urljoin(driver.current_url, href)})
        except Exception:
            continue

    return _uniq_authors(candidates)

def _get_meta_list(driver: webdriver.Chrome, names_or_props: List[str]) -> List[str]:
    vals = []
    for nm in names_or_props:
        for el in driver.find_elements(By.CSS_SELECTOR, f'meta[name="{nm}"], meta[property="{nm}"]'):
            c = (el.get_attribute("content") or "").strip()
            if c:
                vals.append(c)
    return _uniq_str(vals)

def _extract_authors_jsonld(driver: webdriver.Chrome) -> List[str]:
    import json as _json
    names = []
    for s in driver.find_elements(By.CSS_SELECTOR, 'script[type="application/ld+json"]'):
        txt = (s.get_attribute("textContent") or "").strip()
        if not txt:
            continue
        try:
            data = _json.loads(txt)
        except Exception:
            continue
        objs = data if isinstance(data, list) else [data]
        for obj in objs:
            auth = obj.get("author")
            if not auth:
                continue
            if isinstance(auth, list):
                for a in auth:
                    n = a.get("name") if isinstance(a, dict) else str(a)
                    if n: names.append(n)
            elif isinstance(auth, dict):
                n = auth.get("name")
                if n: names.append(n)
            elif isinstance(auth, str):
                names.append(auth)
    return _uniq_str(names)

def _authors_from_subtitle_simple(driver: webdriver.Chrome, title_text: str) -> List[str]:
    """
    Use the subtitle line containing authors + date:
    strip title, cut before first digit (date), parse 'Surname, Initials' pairs.
    """
    try:
        date_el = driver.find_element(By.CSS_SELECTOR, "span.date")
    except NoSuchElementException:
        return []
    try:
        subtitle = date_el.find_element(By.XPATH, "ancestor::*[contains(@class,'subtitle')][1]")
    except Exception:
        try:
            subtitle = date_el.find_element(By.XPATH, "..")
        except Exception:
            subtitle = None
    line = (subtitle.text if subtitle else "")
    if title_text and title_text in line:
        line = line.replace(title_text, "")
    line = " ".join(line.split()).strip()
    m = FIRST_DIGIT.search(line)
    pre_date = line[:m.start()].strip(" -—–·•,;|") if m else line
    pre_date = pre_date.replace(" & ", ", ").replace(" and ", ", ")
    pairs = NAME_PAIR.findall(pre_date)
    return _uniq_str(pairs)

def _wrap_names_as_objs(names: List[str]) -> List[Dict]:
    return _uniq_authors([{"name": n, "profile": None} for n in names])

def extract_detail_for_link(driver: webdriver.Chrome, link: str, title_hint: str) -> Dict:
    driver.get(link)
    accept_cookies_if_present(driver)
    try:
        WebDriverWait(driver, 18).until(EC.presence_of_element_located((By.CSS_SELECTOR, "h1")))
    except TimeoutException:
        pass

    # Title
    try:
        title = driver.find_element(By.CSS_SELECTOR, "h1").text.strip()
    except NoSuchElementException:
        title = title_hint or ""

    _maybe_expand_authors(driver)

    # AUTHORS: header anchors → subtitle → meta → JSON-LD
    author_objs: List[Dict[str, Optional[str]]] = _authors_from_header_anchors(driver)
    # Final safety: filter any stray nav anchors that slipped through
    author_objs = [
        a for a in author_objs
        if _looks_like_person_name(a.get("name","")) and _is_person_profile_url(a.get("profile",""))
    ]
    if not author_objs:
        names = _authors_from_subtitle_simple(driver, title)
        author_objs = _wrap_names_as_objs(names)
    if not author_objs:
        names = _get_meta_list(driver, ["citation_author", "dc.contributor", "dc.contributor.author"])
        author_objs = _wrap_names_as_objs(names)
    if not author_objs:
        names = _extract_authors_jsonld(driver)
        author_objs = _wrap_names_as_objs(names)

    # PUBLISHED DATE
    published_date = None
    for sel in ["span.date", "time[datetime]", "time"]:
        try:
            el = driver.find_element(By.CSS_SELECTOR, sel)
            published_date = el.get_attribute("datetime") or el.text.strip()
            if published_date:
                break
        except NoSuchElementException:
            continue
    if not published_date:
        metas = _get_meta_list(driver, ["citation_publication_date", "dc.date", "article:published_time"])
        if metas:
            published_date = metas[0]

    # ABSTRACT
    abstract_txt = None
    for sel in [
        "section#abstract .textblock", "section.abstract .textblock", "div.abstract .textblock",
        "div#abstract", "section#abstract", "div.textblock",
    ]:
        try:
            el = driver.find_element(By.CSS_SELECTOR, sel)
            txt = el.text.strip()
            if txt and len(txt) > 15:
                abstract_txt = txt; break
        except NoSuchElementException:
            continue
    if not abstract_txt:
        try:
            for h in driver.find_elements(By.CSS_SELECTOR, "h2, h3"):
                if "abstract" in h.text.strip().lower():
                    nxt = h.find_element(By.XPATH, "./following::*[self::div or self::p or self::section][1]")
                    txt = nxt.text.strip()
                    if txt:
                        abstract_txt = txt; break
        except Exception:
            pass

    return {
        "title": title,
        "link": link,
        "authors": _uniq_authors(author_objs),  # Coventry person URLs only; else profile=None
        "published_date": published_date,
        "abstract": abstract_txt or ""
    }

# =========================== Workers ===========================
def worker_detail_batch(batch: List[Dict], headless: bool, legacy_headless: bool) -> List[Dict]:
    driver = make_driver(headless=headless, legacy_headless=legacy_headless)
    out: List[Dict] = []
    try:
        for i, it in enumerate(batch, 1):
            try:
                rec = extract_detail_for_link(driver, it["link"], it.get("title",""))
                out.append(rec)
                if i % 5 == 0:
                    print(f"[WORKER] {i}/{len(batch)} parsed")
            except WebDriverException as e:
                print(f"[WORKER] ERR {it['link']}: {e}")
                continue
    finally:
        try:
            driver.quit()
        except Exception:
            pass
    return out

def chunk(items: List[Dict], n: int) -> List[List[Dict]]:
    if n <= 1:
        return [items]
    size = ceil(len(items) / n)
    return [items[i:i+size] for i in range(0, len(items), size)]

# =========================== Orchestrator ===========================
def main():
    ap = argparse.ArgumentParser(description="Coventry PurePortal scraper (listing → details, clean author links).")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--max-pages", type=int, default=50, help="Max listing pages to scan.")
    ap.add_argument("--workers", type=int, default=8, help="Parallel headless browsers for detail pages.")
    ap.add_argument("--listing-headless", action="store_true", help="Run listing headless.")
    ap.add_argument("--legacy-headless", action="store_true", help="Use legacy --headless.")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # -------- Stage 1: listing
    print(f"[STAGE 1] Collecting links (up to {args.max_pages} pages)…")
    listing = gather_all_listing_links(args.max_pages, headless_listing=args.listing_headless, legacy_headless=args.legacy_headless)
    if not listing:
        print("No publications found on listing pages.")
        return
    (outdir / "publications_links.json").write_text(json.dumps(listing, indent=2), encoding="utf-8")
    print(f"[STAGE 1] Collected {len(listing)} unique links.")

    # -------- Stage 2: details (parallel; no per-item sleep)
    print(f"[STAGE 2] Scraping details with {args.workers} headless workers…")
    batches = chunk(listing, max(1, args.workers))
    results: List[Dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(worker_detail_batch, batch, True, args.legacy_headless) for batch in batches]
        done = 0
        for fut in as_completed(futs):
            part = fut.result() or []
            results.extend(part)
            done += 1
            print(f"[STAGE 2] Completed {done}/{len(batches)} batches (+{len(part)} items)")

    # -------- Save (prefer detail results)
    by_link: Dict[str, Dict] = {}
    for it in listing:
        by_link[it["link"]] = {"title": it["title"], "link": it["link"]}
    for rec in results:
        by_link[rec["link"]] = rec

    final_rows = list(by_link.values())
    out_path = outdir / "publications.json"
    out_path.write_text(json.dumps(final_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] Saved {len(final_rows)} records → {out_path}")

if __name__ == "__main__":
    main()
