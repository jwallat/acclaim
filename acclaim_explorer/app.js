// Visual demo for acclaim batch outputs.
// Loads a run from the FastAPI backend, renders one example at a time.

const state = {
  runs: [],          // list of jsonl filenames available in outputs/alce_eval/
  currentRun: null,  // filename currently loaded
  records: [],
  summary: null,
  currentIdx: 0,
  highlightMode: "support",
  pinnedSource: null,       // element that initiated the pin (sentence or claim)
  pinnedSentence: null,     // .sentence element to highlight (or null)
  pinnedClaimIds: null,     // string[] of pinned claim ids (or null)
  pinnedDocCiteNums: null,  // string[] of pinned doc citation numbers (or null)
};

const els = {
  runSelect:      document.getElementById("run-select"),
  runBadge:       document.getElementById("run-badge"),
  select:         document.getElementById("example-select"),
  prevBtn:        document.getElementById("prev-btn"),
  nextBtn:        document.getElementById("next-btn"),
  themeBtn:       document.getElementById("theme-btn"),
  highlightToggle: document.getElementById("highlight-toggle"),
  questionText:   document.getElementById("question-text"),
  answerText:     document.getElementById("answer-text"),
  metricsTable:   document.getElementById("metrics-table").querySelector("tbody"),
  batchMetricsTable: document.getElementById("batch-metrics-table").querySelector("tbody"),
  claimsList:     document.getElementById("claims-list"),
  claimsCount:    document.getElementById("claims-count"),
  docsList:       document.getElementById("docs-list"),
  status:         document.getElementById("status"),
};

// ---------- Theme ----------

const THEME_KEY = "attr-demo-theme";

function initTheme() {
  const saved = localStorage.getItem(THEME_KEY);
  const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  applyTheme(saved || (prefersDark ? "dark" : "light"));
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  els.themeBtn.textContent = theme === "dark" ? "☀️" : "🌙";
  els.themeBtn.title = theme === "dark" ? "Switch to light mode" : "Switch to dark mode";
}

function toggleTheme() {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  applyTheme(next);
  localStorage.setItem(THEME_KEY, next);
}

// ---------- Bootstrapping ----------

async function init() {
  initTheme();
  setStatus("Discovering runs…");
  try {
    state.runs = await discoverRuns();
    populateRunSelect();
    bindEvents();
    if (!state.runs.length) {
      setStatus("No saved runs — use the query panel to ask a question.");
      return;
    }
    await loadRun(state.runs[0]);
  } catch (e) {
    console.error(e);
    setStatus(`Failed to load: ${e.message}`, true);
  } finally {
    window.dispatchEvent(new CustomEvent("acclaim:ready"));
  }
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} → HTTP ${res.status}`);
  return res.json();
}

async function discoverRuns() {
  const names = await fetchJson("/api/runs");
  return names.filter((n) => n.endsWith(".jsonl")).sort();
}

async function loadRun(filename) {
  setStatus(`Loading ${filename}…`);
  try {
    const data = await fetchJson(`/api/runs/${encodeURIComponent(filename)}`);
    state.currentRun = filename;
    state.records = data.items;
    state.summary = data; // { items, aggregate_metrics, metadata }
    state.currentIdx = 0;

    els.runSelect.value = filename;
    els.runBadge.textContent = `${data.items.length} examples`;
    populateSelect();
    renderBatchMetrics();

    if (!data.items.length) {
      setStatus(`No records in ${filename}.`, true);
      return;
    }
    render(0);
    setStatus("");
  } catch (e) {
    console.error(e);
    setStatus(`Failed to load ${filename}: ${e.message}`, true);
  }
}

function setStatus(msg, isError = false) {
  els.status.textContent = msg || "";
  els.status.classList.toggle("error", isError);
}

// ---------- UI wiring ----------

function populateRunSelect() {
  els.runSelect.innerHTML = "";
  state.runs.forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    els.runSelect.appendChild(opt);
  });
}

function populateSelect() {
  els.select.innerHTML = "";
  state.records.forEach((rec, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = `#${i} — ${truncate(rec.question, 60)}`;
    els.select.appendChild(opt);
  });
}

function bindEvents() {
  els.runSelect.addEventListener("change", (e) => loadRun(e.target.value));
  els.select.addEventListener("change", (e) => render(Number(e.target.value)));
  els.themeBtn.addEventListener("click", toggleTheme);
  els.prevBtn.addEventListener("click", () =>
    render(Math.max(0, state.currentIdx - 1))
  );
  els.nextBtn.addEventListener("click", () =>
    render(Math.min(state.records.length - 1, state.currentIdx + 1))
  );
  els.highlightToggle.addEventListener("change", () => {
    state.highlightMode = els.highlightToggle.checked ? "support" : "off";
    document.body.dataset.highlight = state.highlightMode;
  });
  document.body.dataset.highlight = state.highlightMode;
  document.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft")  els.prevBtn.click();
    if (e.key === "ArrowRight") els.nextBtn.click();
    if (e.key === "Escape" && state.pinnedSource) {
      clearPin();
      reorderClaims([]);
      reorderDocs([]);
    }
  });
}

// ---------- Rendering ----------

function render(idx) {
  state.currentIdx = idx;
  els.select.value = String(idx);
  els.prevBtn.disabled = idx === 0;
  els.nextBtn.disabled = idx === state.records.length - 1;

  // Switching examples rebuilds the DOM — drop any stale pin.
  state.pinnedSource = null;
  state.pinnedSentence = null;
  state.pinnedClaimIds = null;
  state.pinnedDocCiteNums = null;

  const rec = state.records[idx];
  els.questionText.textContent = (rec.question || "(no question)").trim();

  const citationToDoc = rec.citation_to_doc || {};
  const docIdToCiteNum = invertCitationMap(citationToDoc);

  renderAnswer(rec.sentences || [], rec.citation_markers || [], rec.claims || [], citationToDoc);
  renderMetrics(rec.metrics || {});
  renderClaims(rec.claims || [], docIdToCiteNum);
  renderDocs(rec.documents || [], docIdToCiteNum);
}

function renderAnswer(sentences, citationMarkers, claimResults, citationToDoc) {
  const sentClaims = mapClaimsToSentences(sentences, claimResults);

  els.answerText.innerHTML = sentences
    .map((sent, i) =>
      renderSentence(sent, sentClaims[i], claimResults, citationMarkers, citationToDoc)
    )
    .join("");

  // Citation markers scroll to the corresponding doc block.
  // stopPropagation so clicking [N] doesn't also pin the sentence.
  els.answerText.querySelectorAll("a.cite").forEach((a) => {
    a.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      const target = document.getElementById(a.getAttribute("href").slice(1));
      if (!target) return;
      target.scrollIntoView({ behavior: "smooth", block: "start" });
      target.classList.add("target");
      setTimeout(() => target.classList.remove("target"), 1500);
    });
  });

  // Hover sentence → highlight its claims (and vice-versa via renderClaims).
  // Click sentence → pin claims to the top until cleared.
  els.answerText.querySelectorAll(".sentence").forEach((s) => {
    const ids = parseClaimIds(s.dataset.claimIds);
    if (!ids.length) return;
    s.addEventListener("mouseenter", () => setClaimHover(ids, true));
    s.addEventListener("mouseleave", () => setClaimHover(ids, false));
    s.addEventListener("click", () => pinFromSentence(s));
  });
}

// ---------- Pinning ----------

function pinFromSentence(sentenceEl) {
  const claimIds = parseClaimIds(sentenceEl.dataset.claimIds);
  if (!claimIds.length) return;
  applyPin({
    source: sentenceEl,
    sentenceEl,
    claimIds,
    docCiteNums: citeNumsForSentence(sentenceEl),
    highlightTokens: false,
  });
}

function pinFromClaim(claimEl) {
  const claimId = claimEl.dataset.claimId;
  applyPin({
    source: claimEl,
    sentenceEl: findSentenceForClaim(claimId),
    claimIds: [claimId],
    docCiteNums: citeNumsForClaim(claimEl),
    highlightTokens: true,
  });
}

function applyPin({ source, sentenceEl, claimIds, docCiteNums, highlightTokens }) {
  const same = state.pinnedSource === source;
  clearPin();
  if (same) {
    reorderClaims([]);
    reorderDocs([]);
    return;
  }
  state.pinnedSource = source;
  state.pinnedSentence = sentenceEl;
  state.pinnedClaimIds = claimIds.map(String);
  state.pinnedDocCiteNums = docCiteNums.map(String);

  if (sentenceEl) sentenceEl.classList.add("sentence-pinned");
  state.pinnedClaimIds.forEach((id) => {
    document.getElementById(`claim-${id}`)?.classList.add("claim-pinned");
  });
  state.pinnedDocCiteNums.forEach((num) => {
    document.getElementById(`doc-${num}`)?.classList.add("doc-pinned");
  });

  reorderClaims(state.pinnedClaimIds);
  reorderDocs(state.pinnedDocCiteNums);
  if (highlightTokens) {
    highlightClaimTokensInDocs(state.pinnedClaimIds, state.pinnedDocCiteNums);
  }
}

function clearPin() {
  if (state.pinnedSentence) state.pinnedSentence.classList.remove("sentence-pinned");
  state.pinnedSource = null;
  state.pinnedSentence = null;
  state.pinnedClaimIds = null;
  state.pinnedDocCiteNums = null;
  els.claimsList.querySelectorAll(".claim-pinned").forEach((el) =>
    el.classList.remove("claim-pinned")
  );
  els.docsList.querySelectorAll(".doc-pinned").forEach((el) =>
    el.classList.remove("doc-pinned")
  );
  clearTokenHighlights();
}

function citeNumsForClaim(claimEl) {
  return (claimEl.dataset.citeNums || "").split(",").filter(Boolean);
}

function citeNumsForSentence(sentenceEl) {
  const ids = parseClaimIds(sentenceEl.dataset.claimIds);
  const set = new Set();
  ids.forEach((id) => {
    const c = document.getElementById(`claim-${id}`);
    if (c) citeNumsForClaim(c).forEach((n) => set.add(n));
  });
  return [...set];
}

function findSentenceForClaim(claimId) {
  return (
    [...els.answerText.querySelectorAll(".sentence")].find((s) =>
      parseClaimIds(s.dataset.claimIds).includes(String(claimId))
    ) || null
  );
}

function mapClaimsToSentences(sentences, claimResults) {
  // For each sentence, list of claim indices whose span falls inside it.
  // Claims with span=[-1,-1] (un-aligned) are skipped.
  const out = sentences.map(() => []);
  claimResults.forEach((c, idx) => {
    const span = c.claim && c.claim.span;
    if (!Array.isArray(span) || span.length !== 2) return;
    const [start, end] = span;
    if (start < 0 || end < 0) return;
    const si = sentences.findIndex((s) => start >= s.start && start < s.end);
    if (si !== -1) out[si].push(idx);
  });
  return out;
}

function renderSentence(sent, claimIdxs, claimResults, citationMarkers, citationToDoc) {
  const labels = new Set(
    claimIdxs.map((i) => (claimResults[i].support && claimResults[i].support.label) || "UNCLEAR")
  );
  const cls = aggregateLabelClass(labels);
  const relevantMarkers = citationMarkers.filter(
    (m) => m.start >= sent.start && m.end <= sent.end
  );
  const html = renderTextWithCitationLinks(sent.text, sent.start, relevantMarkers, citationToDoc);
  const ids = claimIdxs.join(",");
  return `<span class="sentence ${cls}" data-claim-ids="${ids}">${html}</span>`;
}

function aggregateLabelClass(labels) {
  if (labels.size === 0) return "";
  if (labels.size === 1) return [...labels][0].toLowerCase();
  // Any REFUTED present → severe (red/amber). Otherwise it's SUPPORTED +
  // UNCLEAR only → soft (green/amber).
  return labels.has("REFUTED") ? "mixed-negative" : "mixed-positive";
}

function parseClaimIds(s) {
  if (!s) return [];
  return s.split(",").map((x) => x.trim()).filter(Boolean);
}

function setClaimHover(claimIds, on) {
  if (state.pinnedSource) return;  // pin takes precedence
  claimIds.forEach((id) => {
    const el = document.getElementById(`claim-${id}`);
    if (el) el.classList.toggle("claim-hover", on);
  });
  reorderClaims(on ? claimIds : []);
  // Cascade: also highlight + reorder the docs cited by these claims.
  const citeNums = [];
  claimIds.forEach((id) => {
    const claimEl = document.getElementById(`claim-${id}`);
    if (claimEl) citeNumsForClaim(claimEl).forEach((n) => citeNums.push(n));
  });
  setDocsHover(citeNums, on);
}

function setDocsHover(citeNums, on) {
  const unique = [...new Set(citeNums.map(String))];
  unique.forEach((num) => {
    const el = document.getElementById(`doc-${num}`);
    if (el) el.classList.toggle("doc-hover", on);
  });
  reorderDocs(on ? unique : []);
}

// FLIP-animate a list: pinned/highlighted children slide to the top, the rest
// keep their original relative order. `keyFn(el)` returns the value to test
// for highlight membership; `origIdxFn(el)` returns the natural sort key.
function flipReorder(list, highlightedKeys, keyFn, origIdxFn) {
  const items = [...list.children];
  if (items.length < 2) return;

  const oldTops = new Map();
  items.forEach((el) => oldTops.set(el, el.getBoundingClientRect().top));

  const hi = new Set(highlightedKeys.map(String));
  const sorted = [...items].sort((a, b) => {
    const aHi = hi.has(keyFn(a)) ? 0 : 1;
    const bHi = hi.has(keyFn(b)) ? 0 : 1;
    if (aHi !== bHi) return aHi - bHi;
    return origIdxFn(a) - origIdxFn(b);
  });
  sorted.forEach((el) => list.appendChild(el));

  sorted.forEach((el) => {
    const dy = oldTops.get(el) - el.getBoundingClientRect().top;
    if (!dy) return;
    el.style.transition = "none";
    el.style.transform = `translateY(${dy}px)`;
    void el.offsetHeight;
    el.style.transition =
      "transform 0.35s cubic-bezier(0.2, 0, 0, 1), box-shadow 0.15s, background-color 0.15s";
    el.style.transform = "translateY(0)";
  });
}

function reorderClaims(highlightedIds) {
  flipReorder(
    els.claimsList,
    highlightedIds,
    (el) => el.dataset.claimId,
    (el) => Number(el.dataset.claimId)
  );
}

function reorderDocs(highlightedCiteNums) {
  flipReorder(
    els.docsList,
    highlightedCiteNums,
    (el) => el.dataset.citeNum,
    (el) => Number(el.dataset.origIdx)
  );
}

function setSentenceHover(claimId, on) {
  if (state.pinnedSource) return;  // pin takes precedence
  els.answerText.querySelectorAll(".sentence").forEach((s) => {
    const ids = parseClaimIds(s.dataset.claimIds);
    if (ids.includes(String(claimId))) s.classList.toggle("sentence-hover", on);
  });
  // Cascade: also highlight + reorder the docs cited by this claim.
  const claimEl = document.getElementById(`claim-${claimId}`);
  if (claimEl) setDocsHover(citeNumsForClaim(claimEl), on);
}

function renderMetrics(metrics) {
  els.metricsTable.innerHTML = Object.entries(metrics)
    .map(([k, v]) => `<tr>${metricLabelCell(k)}<td>${formatMetric(v)}</td></tr>`)
    .join("");
  bindMetricTooltips(els.metricsTable);
}

function renderBatchMetrics() {
  if (!state.summary || !state.summary.aggregate_metrics) {
    els.batchMetricsTable.innerHTML = `<tr><td colspan="2" style="color:var(--muted)">No summary.</td></tr>`;
    return;
  }
  els.batchMetricsTable.innerHTML = Object.entries(state.summary.aggregate_metrics)
    .map(([k, v]) => `<tr>${metricLabelCell(k)}<td>${formatMetric(v)}</td></tr>`)
    .join("");
  bindMetricTooltips(els.batchMetricsTable);
}

function metricLabelCell(key) {
  const meta = typeof METRIC_META !== "undefined" ? METRIC_META[key] : null;
  if (!meta) return `<td>${escapeHtml(key)}</td>`;
  return `<td>${escapeHtml(meta.name)} <button type="button" class="metric-help" data-metric="${escapeHtml(key)}" aria-describedby="tip-${escapeHtml(key)}">?<span id="tip-${escapeHtml(key)}" role="tooltip" class="metric-tip">${escapeHtml(meta.desc)}</span></button></td>`;
}

function bindMetricTooltips(tableEl) {
  tableEl.querySelectorAll(".metric-help").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const isOpen = btn.classList.contains("is-open");
      document.querySelectorAll(".metric-help.is-open").forEach((b) => b.classList.remove("is-open"));
      if (!isOpen) btn.classList.add("is-open");
    });
  });
  if (!bindMetricTooltips._docClickBound) {
    document.addEventListener("click", () => {
      document.querySelectorAll(".metric-help.is-open").forEach((b) => b.classList.remove("is-open"));
    });
    bindMetricTooltips._docClickBound = true;
  }
}

function renderClaims(claimResults, docIdToCiteNum) {
  els.claimsCount.textContent = `(${claimResults.length})`;
  els.claimsList.innerHTML = claimResults
    .map((c, idx) => {
      const label = (c.support && c.support.label) || "UNCLEAR";
      const conf = c.support && typeof c.support.confidence === "number"
        ? ` · ${c.support.confidence.toFixed(2)}`
        : "";
      const reason = (c.support && c.support.reason) || "";
      const citeNumsArr = (c.citation_doc_ids || [])
        .map((d) => docIdToCiteNum[d])
        .filter((n) => n != null);
      const cites = citeNumsArr.length
        ? citeNumsArr.map((n) => `[${n}]`).join(" ")
        : "(no citations)";
      const citeNumsData = citeNumsArr.join(",");
      return `
        <li class="claim-item" id="claim-${idx}" data-claim-id="${idx}" data-cite-nums="${citeNumsData}">
          <div class="claim-head">
            <div class="claim-text">${escapeHtml(c.claim?.text || "")}</div>
            <span class="pill ${label.toLowerCase()}">${label}${conf}</span>
          </div>
          <div class="claim-cites">cites: ${escapeHtml(cites)}</div>
          ${reason ? `<details class="claim-reason"><summary>reason</summary>${escapeHtml(reason)}</details>` : ""}
        </li>`;
    })
    .join("");

  // Hover claim → highlight its mapped sentence.
  // Click claim → pin (highlight sentence + reorder & highlight cited docs).
  els.claimsList.querySelectorAll(".claim-item").forEach((li) => {
    const id = li.dataset.claimId;
    li.addEventListener("mouseenter", () => setSentenceHover(id, true));
    li.addEventListener("mouseleave", () => setSentenceHover(id, false));
    li.addEventListener("click", (e) => {
      // Let the "reason" disclosure toggle without triggering a pin.
      if (e.target.closest(".claim-reason")) return;
      pinFromClaim(li);
    });
  });
}

function renderDocs(documents, docIdToCiteNum) {
  els.docsList.innerHTML = documents
    .map((d, i) => {
      const num = docIdToCiteNum[d.doc_id];
      const numHtml = num != null ? `<span class="doc-cite-num">[${num}]</span>` : "";
      const idAttr = num != null ? `id="doc-${num}"` : "";
      return `
        <div class="doc-block" ${idAttr} data-orig-idx="${i}" data-cite-num="${num != null ? num : ""}">
          <div class="doc-head">
            ${numHtml}<span>doc_id=${escapeHtml(d.doc_id || "?")}</span>
          </div>
          <div class="doc-text">${escapeHtml(d.text || "")}</div>
        </div>`;
    })
    .join("");
}

// ---------- Token highlighting ----------

function getKeywords(text) {
  return new Set(
    text.toLowerCase().replace(/[^\w\s]/g, " ").split(/\s+/)
      .filter((w) => w.length > 1 && !STOPWORDS.has(w))
  );
}

function highlightClaimTokensInDocs(claimIds, docCiteNums) {
  const rec = state.records[state.currentIdx];
  if (!rec) return;

  const keywords = new Set();
  claimIds.forEach((id) => {
    const text = rec.claims[id]?.claim?.text || "";
    getKeywords(text).forEach((w) => keywords.add(w));
  });
  if (!keywords.size) return;

  const citationToDoc = rec.citation_to_doc || {};
  docCiteNums.forEach((num) => {
    const docEl = document.getElementById(`doc-${num}`);
    if (!docEl) return;
    const textEl = docEl.querySelector(".doc-text");
    if (!textEl) return;
    const docId = citationToDoc[String(num)];
    const doc = rec.documents.find((d) => d.doc_id === docId);
    if (!doc) return;
    textEl.innerHTML = _tokenHighlightHtml(doc.text || "", keywords);
  });
}

function _tokenHighlightHtml(text, keywords) {
  return escapeHtml(text).replace(/\b\w+\b/g, (word) =>
    keywords.has(word.toLowerCase()) ? `<mark class="token-hit">${word}</mark>` : word
  );
}

function clearTokenHighlights() {
  els.docsList.querySelectorAll(".doc-text mark.token-hit").forEach((mark) => {
    mark.replaceWith(document.createTextNode(mark.textContent));
  });
  els.docsList.querySelectorAll(".doc-text").forEach((el) => el.normalize());
}

// ---------- Helpers ----------

function renderTextWithCitationLinks(text, textStart, markers, citationToDoc) {
  // `markers` are server-computed citation marker spans (from
  // acclaim.citations.parser.find_citation_markers — supports "[N]",
  // "[N][M]", and combined "[N, M]" styles) with offsets into the original
  // answer text; `text` is the literal slice [textStart, textStart+len)
  // of that same answer. Builds escaped HTML, wrapping each marker's exact
  // original substring in a link to its first known cited document.
  let html = "";
  let cursor = 0;
  markers.forEach((m) => {
    const localStart = m.start - textStart;
    const localEnd = m.end - textStart;
    if (localStart < cursor || localEnd > text.length) return;
    html += escapeHtml(text.slice(cursor, localStart));
    const markerText = text.slice(localStart, localEnd);
    const num = m.numbers.find((n) => citationToDoc[String(n)] != null);
    const docId = num != null ? citationToDoc[String(num)] : null;
    html += docId
      ? `<a class="cite" href="#doc-${num}" data-doc-id="${escapeHtml(docId)}">${escapeHtml(markerText)}</a>`
      : escapeHtml(markerText);
    cursor = localEnd;
  });
  html += escapeHtml(text.slice(cursor));
  return html;
}

function invertCitationMap(citationToDoc) {
  const out = {};
  for (const [num, docId] of Object.entries(citationToDoc || {})) {
    out[docId] = Number(num);
  }
  return out;
}

function truncate(s, n) {
  if (!s) return "";
  return s.length <= n ? s : s.slice(0, n - 1) + "…";
}

function formatMetric(v) {
  if (typeof v !== "number" || Number.isNaN(v)) return String(v);
  if (Number.isInteger(v)) return String(v);
  if (Math.abs(v) >= 100) return v.toFixed(1);
  return v.toFixed(3);
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

init();
