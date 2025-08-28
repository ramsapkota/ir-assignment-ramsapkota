/**
 * Information Retrieval Assignment - Compact Interface
 * Author: Ram Sapkota
 * Google-style search with academic publication results
 */

"use strict";

// =============================================================================
// UTILITIES & CONSTANTS
// =============================================================================

const $ = (id) => document.getElementById(id);
const $$ = (selector) => document.querySelectorAll(selector);

const formatTime = (ms) => {
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.max(0, Math.round(ms))}ms`;
};

const sanitizeHTML = (html) => {
  if (!html) return "";
  return html.replace(/<(?!\/?mark\b)[^>]*>/gi, "");
};

const getSafeDate = (result) => {
  return result.date || result.published_date || "";
};

const truncateText = (text, maxLength = 200) => {
  if (!text || text.length <= maxLength) return text;
  return text.substr(0, maxLength).replace(/\s+\S*$/, "") + "...";
};

// =============================================================================
// STATE MANAGEMENT
// =============================================================================

const AppState = {
  currentPage: 1,
  totalPages: 1,
  pageSize: 50,
  isSearchMode: false,
  lastQuery: "",
  classifierModel: null,

  ENGINE_KEY: "ir_engine_compact",
  ENGINES: {
    BM25: "bm25",
    TFIDF: "tfidf",
  },
};

// =============================================================================
// DOM ELEMENT REFERENCES
// =============================================================================

const Elements = {
  // Navigation
  tabButtons: $$(".nav-tab"),
  tabContents: $$(".tab-content"),

  // Search
  searchForm: $("search-form"),
  searchInput: $("search-input"),
  searchBox: null, // Will be set on init
  clearBtn: $("clear-btn"),
  resultsContainer: $("results-container"),
  resultsText: $("results-text"),
  timePill: $("time-pill"),
  paginationControls: $("pagination-controls"),

  // Engine controls
  btnBM25: $("btn-bm25"),
  btnTFIDF: $("btn-tfidf"),
  engineSelector: null, // Will be set on init

  // Classification
  clsInput: $("cls-input"),
  clsRun: $("cls-run"),
  clsClear: $("cls-clear"),
  clsStatus: $("cls-status"),
  clsResult: $("cls-result"),
  clsLabel: $("cls-label"),
  clsConf: $("cls-conf"),
  clsBars: $("cls-bars"),
  clsModelChip: $("cls-model-chip"),
  clsTimePill: $("cls-time-pill"),
};

// =============================================================================
// UI HELPERS
// =============================================================================

const UIHelpers = {
  showLoading(container, count = 6) {
    container.innerHTML = "";
    for (let i = 0; i < count; i++) {
      const skeleton = document.createElement("div");
      skeleton.className = "skeleton";
      skeleton.innerHTML = `
        <div class="skeleton-line"></div>
        <div class="skeleton-line"></div>
        <div class="skeleton-line"></div>
      `;
      container.appendChild(skeleton);
    }
  },

  showEmptyState(container, title, message, icon = "🔍") {
    container.innerHTML = `
      <div class="welcome-state">
        <div class="welcome-icon">${icon}</div>
        <h3>${title}</h3>
        <p>${message}</p>
      </div>
    `;
  },

  updateSearchBoxState() {
    const hasContent = Elements.searchInput.value.trim().length > 0;
    Elements.searchBox.classList.toggle("has-content", hasContent);
  },

  toggleEngineSelector(show) {
    Elements.engineSelector.classList.toggle("hidden", !show);
  },
};

// =============================================================================
// TAB MANAGEMENT
// =============================================================================

const TabManager = {
  init() {
    Elements.tabButtons.forEach((button) => {
      button.addEventListener("click", () => {
        this.activateTab(button.dataset.tab);
      });
    });
  },

  activateTab(tabId) {
    Elements.tabButtons.forEach((button) => {
      const isActive = button.dataset.tab === tabId;
      button.classList.toggle("active", isActive);
      button.setAttribute("aria-selected", String(isActive));
    });

    Elements.tabContents.forEach((content) => {
      const isActive = content.id === tabId;
      content.classList.toggle("active", isActive);
      content.classList.toggle("hidden", !isActive);
    });

    // Show/hide engine selector based on active tab
    UIHelpers.toggleEngineSelector(tabId === "search-engine" && AppState.isSearchMode);
  },
};

// =============================================================================
// ENGINE MANAGEMENT
// =============================================================================

const EngineManager = {
  init() {
    this.setupEventListeners();
    this.setEngine(this.getEngine(), false);
  },

  getEngine() {
    const stored = localStorage.getItem(AppState.ENGINE_KEY);
    return Object.values(AppState.ENGINES).includes(stored) ? stored : AppState.ENGINES.BM25; // Default to BM25
  },

  setEngine(engine, rerun = false) {
    localStorage.setItem(AppState.ENGINE_KEY, engine);

    Elements.btnBM25.classList.toggle("active", engine === AppState.ENGINES.BM25);
    Elements.btnTFIDF.classList.toggle("active", engine === AppState.ENGINES.TFIDF);

    if (rerun) {
      if (AppState.isSearchMode) {
        SearchManager.performSearch(AppState.lastQuery, 1);
      } else {
        SearchManager.fetchAllPublications(1);
      }
    }
  },

  setupEventListeners() {
    Elements.btnBM25.addEventListener("click", () => {
      this.setEngine(AppState.ENGINES.BM25, true);
    });

    Elements.btnTFIDF.addEventListener("click", () => {
      this.setEngine(AppState.ENGINES.TFIDF, true);
    });
  },
};

// =============================================================================
// SEARCH MANAGEMENT
// =============================================================================

const SearchManager = {
  init() {
    this.setupEventListeners();
    this.fetchAllPublications(1);
  },

  setupEventListeners() {
    // Search form submission
    Elements.searchForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const query = Elements.searchInput.value.trim();

      if (query) {
        this.performSearch(query, 1);
      } else {
        this.fetchAllPublications(1);
      }
    });

    // Clear search
    Elements.clearBtn.addEventListener("click", () => {
      Elements.searchInput.value = "";
      UIHelpers.updateSearchBoxState();
      this.fetchAllPublications(1);
    });

    // Input changes
    Elements.searchInput.addEventListener("input", () => {
      UIHelpers.updateSearchBoxState();
    });

    // Keyboard shortcuts
    Elements.searchInput.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        Elements.searchInput.value = "";
        UIHelpers.updateSearchBoxState();
        this.fetchAllPublications(1);
      }
    });
  },

  async fetchAllPublications(page = 1) {
    try {
      AppState.isSearchMode = false;
      AppState.currentPage = page;

      Elements.resultsText.textContent = "Loading...";
      Elements.timePill.classList.add("hidden");
      Elements.paginationControls.classList.add("hidden");
      UIHelpers.toggleEngineSelector(false);
      UIHelpers.showLoading(Elements.resultsContainer, 8);

      const engine = EngineManager.getEngine();
      const startTime = performance.now();

      const response = await fetch(`/publications/?page=${page}&page_size=${AppState.pageSize}&engine=${encodeURIComponent(engine)}`);
      const data = await response.json();
      const elapsedTime = performance.now() - startTime;

      if (!response.ok) {
        throw new Error(data.error || "Failed to fetch publications");
      }

      AppState.totalPages = data.total_pages || 1;

      Elements.resultsText.textContent = `${data.total_publications} publications`;
      Elements.timePill.textContent = formatTime(elapsedTime);
      Elements.timePill.classList.remove("hidden");

      this.renderResults(data.publications, false);
      this.renderPagination(false);
    } catch (error) {
      console.error("Error fetching publications:", error);
      Elements.resultsText.textContent = "Error loading publications";
      UIHelpers.showEmptyState(Elements.resultsContainer, "Error Loading Publications", "Please try again or check your connection.", "⚠️");
    }
  },

  async performSearch(query, page = 1) {
    try {
      AppState.isSearchMode = true;
      AppState.lastQuery = query;
      AppState.currentPage = page;

      Elements.resultsText.textContent = "Searching...";
      Elements.timePill.classList.add("hidden");
      Elements.paginationControls.classList.add("hidden");
      UIHelpers.toggleEngineSelector(true);
      UIHelpers.showLoading(Elements.resultsContainer, 6);

      const engine = EngineManager.getEngine();
      const params = new URLSearchParams({
        engine,
        page: String(page),
        page_size: String(AppState.pageSize),
        q: query,
      });

      const startTime = performance.now();
      const response = await fetch(`/search/?${params.toString()}`);
      const data = await response.json();
      const elapsedTime = data.search_time_ms ?? Math.round(performance.now() - startTime);

      if (!response.ok) {
        throw new Error(data.error || "Search failed");
      }

      AppState.totalPages = data.total_pages || 0;

      if (data.results && data.results.length > 0) {
        Elements.resultsText.textContent = `About ${data.total_results} results (${formatTime(elapsedTime)})`;
        this.renderResults(data.results, true);
        this.renderPagination(true);
      } else {
        Elements.resultsText.textContent = `No results found for "${query}"`;
        UIHelpers.showEmptyState(Elements.resultsContainer, "No Results Found", "Try different keywords or search terms", "🔍");
      }

      Elements.timePill.textContent = formatTime(elapsedTime);
      Elements.timePill.classList.remove("hidden");
    } catch (error) {
      console.error("Search error:", error);
      Elements.resultsText.textContent = `Search error for "${query}"`;
      UIHelpers.showEmptyState(Elements.resultsContainer, "Search Error", "Please try again or check your connection.", "⚠️");
    }
  },

  renderResults(results, showScore) {
    Elements.resultsContainer.innerHTML = "";

    results.forEach((result, index) => {
      const card = this.createResultCard(result, showScore, index);
      Elements.resultsContainer.appendChild(card);
    });
  },

  createResultCard(result, showScore, index) {
    const card = document.createElement("div");
    card.className = "result-card";
    card.style.animationDelay = `${Math.min(index * 0.05, 0.3)}s`;

    // Title
    const title = document.createElement("div");
    title.className = "result-title";

    const titleLink = document.createElement("a");
    titleLink.href = result.link || "#";
    titleLink.textContent = result.title || "Untitled Publication";
    titleLink.target = "_blank";
    titleLink.rel = "noopener noreferrer";
    title.appendChild(titleLink);

    // Meta information
    const meta = document.createElement("div");
    meta.className = "result-meta";

    const authors = this.formatAuthors(result.authors);
    const date = getSafeDate(result);

    if (authors && date) {
      meta.innerHTML = `${authors} - ${date}`;
    } else if (authors) {
      meta.innerHTML = authors;
    } else if (date) {
      meta.innerHTML = date;
    } else {
      meta.innerHTML = "Unknown source";
    }

    // Badges
    const badges = document.createElement("div");
    badges.className = "result-badges";

    if (result.oa_url) {
      const pdfBadge = document.createElement("a");
      pdfBadge.href = result.oa_url;
      pdfBadge.className = "badge pdf";
      pdfBadge.innerHTML = "PDF";
      pdfBadge.target = "_blank";
      pdfBadge.rel = "noopener noreferrer";
      badges.appendChild(pdfBadge);
    }

    if (result.doi) {
      const doi = String(result.doi).replace(/^doi:\s*/i, "");
      const doiBadge = document.createElement("a");
      doiBadge.href = `https://doi.org/${encodeURIComponent(doi)}`;
      doiBadge.className = "badge doi";
      doiBadge.innerHTML = "DOI";
      doiBadge.target = "_blank";
      doiBadge.rel = "noopener noreferrer";
      badges.appendChild(doiBadge);
    }

    // Abstract snippet
    const snippet = document.createElement("div");
    snippet.className = "result-snippet";
    const abstract = result.abstract || "";
    snippet.innerHTML = sanitizeHTML(truncateText(abstract, 200));

    // Score display (if applicable)
    const scoreDisplay = document.createElement("div");
    if (showScore && typeof result.score !== "undefined") {
      const score = Number(result.score);
      const percentage = Math.round((1 / (1 + Math.exp(-score / 2))) * 100);

      scoreDisplay.className = "score-display";
      scoreDisplay.innerHTML = `
        <span class="score-badge">Relevance: ${score.toFixed(2)}</span>
        <div class="score-bar">
          <div class="score-fill" style="width: ${percentage}%"></div>
        </div>
      `;
    }

    // Actions
    const actions = document.createElement("div");
    actions.className = "result-actions";

    if (abstract && abstract.length > 200) {
      const abstractToggle = document.createElement("span");
      abstractToggle.className = "result-action";
      abstractToggle.textContent = "Show abstract";
      abstractToggle.addEventListener("click", () => {
        if (snippet.dataset.expanded === "true") {
          snippet.innerHTML = sanitizeHTML(truncateText(abstract, 200));
          abstractToggle.textContent = "Show abstract";
          snippet.dataset.expanded = "false";
        } else {
          snippet.innerHTML = sanitizeHTML(abstract);
          abstractToggle.textContent = "Hide abstract";
          snippet.dataset.expanded = "true";
        }
      });
      actions.appendChild(abstractToggle);
    }

    // Assemble card
    card.appendChild(title);
    card.appendChild(meta);
    if (badges.children.length > 0) {
      card.appendChild(badges);
    }
    card.appendChild(snippet);
    if (scoreDisplay.innerHTML) {
      card.appendChild(scoreDisplay);
    }
    if (actions.children.length > 0) {
      card.appendChild(actions);
    }

    return card;
  },

  formatAuthors(authors) {
    if (!Array.isArray(authors)) return "";

    const authorNames = authors
      .map((author) => {
        return author?.name || author || "Unknown";
      })
      .slice(0, 3); // Show max 3 authors

    let result = authorNames.join(", ");
    if (authors.length > 3) {
      result += ` et al.`;
    }

    return result;
  },

  renderPagination(isSearch = false) {
    Elements.paginationControls.innerHTML = "";

    if (AppState.totalPages <= 1) {
      Elements.paginationControls.classList.add("hidden");
      return;
    }

    const prevBtn = document.createElement("button");
    prevBtn.textContent = "← Previous";
    prevBtn.disabled = AppState.currentPage <= 1;
    prevBtn.addEventListener("click", () => {
      if (AppState.currentPage > 1) {
        const newPage = AppState.currentPage - 1;
        if (isSearch) {
          this.performSearch(AppState.lastQuery, newPage);
        } else {
          this.fetchAllPublications(newPage);
        }
      }
    });

    const pageInfo = document.createElement("button");
    pageInfo.disabled = true;
    pageInfo.textContent = `Page ${AppState.currentPage} of ${AppState.totalPages}`;

    const nextBtn = document.createElement("button");
    nextBtn.textContent = "Next →";
    nextBtn.disabled = AppState.currentPage >= AppState.totalPages;
    nextBtn.addEventListener("click", () => {
      if (AppState.currentPage < AppState.totalPages) {
        const newPage = AppState.currentPage + 1;
        if (isSearch) {
          this.performSearch(AppState.lastQuery, newPage);
        } else {
          this.fetchAllPublications(newPage);
        }
      }
    });

    Elements.paginationControls.appendChild(prevBtn);
    Elements.paginationControls.appendChild(pageInfo);
    Elements.paginationControls.appendChild(nextBtn);
    Elements.paginationControls.classList.remove("hidden");
  },
};

// =============================================================================
// CLASSIFICATION MANAGEMENT
// =============================================================================

const ClassificationManager = {
  init() {
    this.setupEventListeners();
    this.refreshHealthCheck();
  },

  setupEventListeners() {
    Elements.clsRun.addEventListener("click", () => {
      this.classifyText();
    });

    Elements.clsClear.addEventListener("click", () => {
      this.resetUI();
    });

    Elements.clsInput.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        this.classifyText();
      }
    });
  },

  async refreshHealthCheck() {
    try {
      const response = await fetch("/healthz");
      const data = await response.json();

      AppState.classifierModel = data?.model_algo || null;

      if (AppState.classifierModel) {
        const modelName = this.getModelDisplayName(AppState.classifierModel);
        Elements.clsModelChip.textContent = `Model: ${modelName}`;
        Elements.clsModelChip.classList.remove("hidden");
      } else {
        Elements.clsModelChip.classList.add("hidden");
      }
    } catch (error) {
      console.warn("Health check failed:", error);
      Elements.clsModelChip.classList.add("hidden");
    }
  },

  getModelDisplayName(modelMeta) {
    const modelMap = {
      nb: "Naive Bayes",
      lr: "Logistic Regression",
      svm: "Linear SVM",
    };
    return modelMap[modelMeta] || String(modelMeta).toUpperCase();
  },

  resetUI() {
    Elements.clsInput.value = "";
    Elements.clsStatus.textContent = "Ready";
    Elements.clsLabel.textContent = "—";
    Elements.clsConf.textContent = "—";
    Elements.clsBars.innerHTML = "";
    Elements.clsResult.classList.add("hidden");
    Elements.clsTimePill.classList.add("hidden");
  },

  async classifyText() {
    const text = Elements.clsInput.value.trim();

    if (!text) {
      Elements.clsStatus.textContent = "Enter text";
      return;
    }

    Elements.clsStatus.textContent = "Classifying...";
    Elements.clsResult.classList.add("hidden");
    Elements.clsTimePill.classList.add("hidden");

    const startTime = performance.now();

    try {
      const response = await fetch("/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      const data = await response.json();
      const elapsedTime = performance.now() - startTime;

      if (!response.ok || !data.ok) {
        throw new Error(data.error || "Classification failed");
      }

      Elements.clsStatus.textContent = "Complete";

      const probabilities = data.proba || {};
      const sortedProbs = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
      const topPrediction = sortedProbs[0] || [data.label || "—", 0];

      Elements.clsLabel.textContent = (data.label || topPrediction[0] || "—").toUpperCase();
      Elements.clsConf.textContent = `${Math.round((topPrediction[1] || 0) * 100)}%`;

      this.renderProbabilityBars(probabilities);
      Elements.clsResult.classList.remove("hidden");

      Elements.clsTimePill.textContent = formatTime(elapsedTime);
      Elements.clsTimePill.classList.remove("hidden");
    } catch (error) {
      console.error("Classification error:", error);
      Elements.clsStatus.textContent = "Error";

      Elements.clsLabel.textContent = "ERROR";
      Elements.clsConf.textContent = "0%";
      Elements.clsBars.innerHTML = '<div style="color: var(--text-secondary); font-style: italic;">Classification failed. Please try again.</div>';
      Elements.clsResult.classList.remove("hidden");
    }
  },

  renderProbabilityBars(probabilities) {
    Elements.clsBars.innerHTML = "";

    const sortedEntries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

    sortedEntries.forEach(([category, probability]) => {
      const item = document.createElement("div");
      item.className = "probability-item";

      const header = document.createElement("div");
      header.className = "probability-header";

      const name = document.createElement("span");
      name.className = "probability-name";
      name.textContent = category.charAt(0).toUpperCase() + category.slice(1);

      const value = document.createElement("span");
      value.className = "probability-value";
      value.textContent = `${Math.round(probability * 100)}%`;

      header.appendChild(name);
      header.appendChild(value);

      const bar = document.createElement("div");
      bar.className = "probability-bar";

      const fill = document.createElement("div");
      fill.className = "probability-fill";
      fill.style.width = `${Math.round(probability * 100)}%`;

      bar.appendChild(fill);
      item.appendChild(header);
      item.appendChild(bar);
      Elements.clsBars.appendChild(item);
    });
  },
};

// =============================================================================
// APPLICATION INITIALIZATION
// =============================================================================

class CompactIRApp {
  constructor() {
    this.initialized = false;
  }

  async init() {
    if (this.initialized) return;

    try {
      if (document.readyState === "loading") {
        await new Promise((resolve) => {
          document.addEventListener("DOMContentLoaded", resolve);
        });
      }

      // Set element references that need DOM to be ready
      Elements.searchBox = Elements.searchInput.closest(".search-box");
      Elements.engineSelector = document.querySelector(".engine-selector");

      // Initialize modules
      TabManager.init();
      EngineManager.init();
      SearchManager.init();
      ClassificationManager.init();

      // Setup global handlers
      this.setupGlobalHandlers();

      this.initialized = true;
      console.info("Compact IR Application initialized");
    } catch (error) {
      console.error("Failed to initialize application:", error);
      this.showInitializationError();
    }
  }

  setupGlobalHandlers() {
    // Keyboard shortcuts
    document.addEventListener("keydown", (e) => {
      // Focus search on '/' key
      if (e.key === "/" && !["INPUT", "TEXTAREA"].includes(e.target.tagName)) {
        e.preventDefault();
        Elements.searchInput.focus();
      }
    });

    // Handle browser back/forward
    window.addEventListener("popstate", (e) => {
      if (e.state && e.state.query) {
        Elements.searchInput.value = e.state.query;
        SearchManager.performSearch(e.state.query, e.state.page || 1);
      }
    });
  }

  showInitializationError() {
    const errorHTML = `
      <div style="
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: white;
        padding: 2rem;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        max-width: 400px;
        z-index: 9999;
      ">
        <h2 style="color: #dc2626; margin-bottom: 1rem;">Initialization Error</h2>
        <p style="margin-bottom: 1rem; color: #6b7280;">
          Failed to initialize the application. Please refresh to try again.
        </p>
        <button onclick="window.location.reload()" style="
          background: #2563eb;
          color: white;
          border: none;
          padding: 0.5rem 1rem;
          border-radius: 4px;
          cursor: pointer;
        ">
          Refresh Page
        </button>
      </div>
    `;
    document.body.insertAdjacentHTML("beforeend", errorHTML);
  }
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

window.IRUtils = {
  formatTime,
  sanitizeHTML,
  getSafeDate,
  getAppState: () => ({ ...AppState }),
  refreshClassifier: () => ClassificationManager.refreshHealthCheck(),
  clearSearch: () => {
    Elements.searchInput.value = "";
    UIHelpers.updateSearchBoxState();
    SearchManager.fetchAllPublications(1);
  },
};

// =============================================================================
// APPLICATION STARTUP
// =============================================================================

const app = new CompactIRApp();
app.init().catch((error) => {
  console.error("Application startup failed:", error);
});

// Export for module systems
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    CompactIRApp,
    SearchManager,
    ClassificationManager,
    TabManager,
    EngineManager,
  };
}
