// First-visit welcome tutorial: a short walkthrough of the main controls,
// shown once automatically and replayable via the "?" help button.

(function () {
  const TUTORIAL_KEY = "attr-demo-tutorial-seen";

  const TUTORIAL_STEPS = [
    {
      selector: "#run-select",
      title: "Pick a run",
      text: "Pick which saved batch run to explore.",
    },
    {
      selector: "#example-select",
      extraSelectors: ["#prev-btn", "#next-btn"],
      title: "Browse examples",
      text: "Browse examples. Left/Right arrow keys also work.",
    },
    {
      selector: "#highlight-toggle",
      title: "Highlight",
      text: "Color answer sentences by support label.",
    },
    {
      selector: "#question-text",
      title: "Question",
      text: "This is the current question being answered.",
    },
    {
      selector: "#answer-text",
      title: "Answer",
      text: "The generated answer. Sentences are colored by support label, and [N] citation markers are clickable — click one to jump to that document below.",
    },
    {
      selector: "#docs-list",
      title: "Cited documents",
      text: "The documents cited by this answer.",
    },
    {
      selector: "#answer-text",
      title: "Sentence linkage",
      text: "Hover or click a sentence to see which claims it's built from — linked claims (and their cited documents) light up.",
    },
    {
      selector: "#claims-panel",
      title: "Claim linkage",
      text: "Hover or click a claim to see which sentence it supports and which documents it cites — they'll highlight and jump to the top.",
    },
    {
      selector: "#ask-btn",
      title: "Ask your own question",
      text: "Ask your own question — it retrieves documents, generates an answer, and evaluates it live.",
    },
    {
      selector: "#metrics-table",
      title: "Metrics",
      text: "Hover the ? next to each metric for a definition.",
    },
  ];

  const tState = { active: false, stepIndex: null };
  let overlayRoot = null;
  let spotlightEl = null;
  let calloutEl = null;
  let welcomeRoot = null;

  // ---------- Helpers ----------

  function esc(s) {
    return typeof escapeHtml === "function" ? escapeHtml(s) : String(s);
  }

  function resolveStepElements(step) {
    const selectors = [step.selector, ...(step.extraSelectors || [])];
    return selectors.map((s) => document.querySelector(s)).filter(Boolean);
  }

  function unionRect(elements) {
    const rects = elements.map((el) => el.getBoundingClientRect());
    const left = Math.min(...rects.map((r) => r.left));
    const top = Math.min(...rects.map((r) => r.top));
    const right = Math.max(...rects.map((r) => r.right));
    const bottom = Math.max(...rects.map((r) => r.bottom));
    return { left, top, right, bottom, width: right - left, height: bottom - top };
  }

  // ---------- Overlay construction ----------

  function buildOverlay() {
    overlayRoot = document.createElement("div");
    overlayRoot.id = "tutorial-overlay";

    spotlightEl = document.createElement("div");
    spotlightEl.className = "tutorial-spotlight";

    calloutEl = document.createElement("div");
    calloutEl.className = "tutorial-callout";
    calloutEl.setAttribute("role", "dialog");
    calloutEl.setAttribute("aria-modal", "true");

    overlayRoot.appendChild(spotlightEl);
    overlayRoot.appendChild(calloutEl);
    document.body.appendChild(overlayRoot);
  }

  function renderCalloutContent(step, i) {
    const total = TUTORIAL_STEPS.length;
    calloutEl.innerHTML = `
      <div class="tutorial-callout-step">${i + 1} of ${total}</div>
      <div class="tutorial-callout-title">${esc(step.title)}</div>
      <div class="tutorial-callout-text">${esc(step.text)}</div>
      <div class="tutorial-callout-actions">
        <button type="button" class="tutorial-skip">Skip</button>
        <div class="tutorial-nav">
          <button type="button" class="tutorial-back" ${i === 0 ? "disabled" : ""}>Back</button>
          <button type="button" class="tutorial-next">${i === total - 1 ? "Done" : "Next"}</button>
        </div>
      </div>`;
    calloutEl.querySelector(".tutorial-skip").addEventListener("click", () => endTutorial(true));
    calloutEl.querySelector(".tutorial-back").addEventListener("click", () => showStep(tState.stepIndex - 1));
    calloutEl.querySelector(".tutorial-next").addEventListener("click", () => showStep(tState.stepIndex + 1));
  }

  function positionForRect(rect) {
    const pad = 8;
    const sLeft = rect.left - pad;
    const sTop = rect.top - pad;
    const sWidth = rect.width + pad * 2;
    const sHeight = rect.height + pad * 2;

    spotlightEl.style.left = `${sLeft}px`;
    spotlightEl.style.top = `${sTop}px`;
    spotlightEl.style.width = `${sWidth}px`;
    spotlightEl.style.height = `${sHeight}px`;

    // Measure the callout off-screen first, since its size depends on its text.
    calloutEl.style.visibility = "hidden";
    calloutEl.style.left = "-9999px";
    calloutEl.style.top = "-9999px";
    const cw = calloutEl.offsetWidth;
    const ch = calloutEl.offsetHeight;

    const gap = 14;
    let top = sTop + sHeight + gap;
    if (top + ch > window.innerHeight - 8) {
      top = sTop - gap - ch;
    }
    top = Math.max(8, Math.min(top, window.innerHeight - ch - 8));

    let left = sLeft;
    left = Math.max(8, Math.min(left, window.innerWidth - cw - 8));

    calloutEl.style.left = `${left}px`;
    calloutEl.style.top = `${top}px`;
    calloutEl.style.visibility = "visible";
  }

  function showStep(i, _fromSkip) {
    if (i < 0) return;
    if (i >= TUTORIAL_STEPS.length) {
      endTutorial(true);
      return;
    }
    const step = TUTORIAL_STEPS[i];
    const elements = resolveStepElements(step);
    if (!elements.length) {
      // Target doesn't exist right now — skip forward past it defensively.
      showStep(i + 1);
      return;
    }
    tState.stepIndex = i;
    renderCalloutContent(step, i);
    positionForRect(unionRect(elements));
    const nextBtn = calloutEl.querySelector(".tutorial-next");
    if (nextBtn) nextBtn.focus();
  }

  function handleReposition() {
    if (!tState.active || tState.stepIndex == null) return;
    const elements = resolveStepElements(TUTORIAL_STEPS[tState.stepIndex]);
    if (!elements.length) return;
    positionForRect(unionRect(elements));
  }

  let repositionQueued = false;
  function onWindowChange() {
    if (repositionQueued) return;
    repositionQueued = true;
    requestAnimationFrame(() => {
      repositionQueued = false;
      handleReposition();
    });
  }
  window.addEventListener("resize", onWindowChange);
  window.addEventListener("scroll", onWindowChange, true);

  function startTutorial() {
    removeWelcomePrompt();
    tState.active = true;
    buildOverlay();
    showStep(0);
  }

  function endTutorial(markSeen) {
    tState.active = false;
    tState.stepIndex = null;
    if (overlayRoot) {
      overlayRoot.remove();
      overlayRoot = null;
      spotlightEl = null;
      calloutEl = null;
    }
    removeWelcomePrompt();
    if (markSeen) localStorage.setItem(TUTORIAL_KEY, "1");
  }

  // ---------- Welcome prompt ----------

  function showWelcomePrompt() {
    welcomeRoot = document.createElement("div");
    welcomeRoot.id = "tutorial-welcome";
    welcomeRoot.innerHTML = `
      <div class="tutorial-dim"></div>
      <div class="tutorial-callout tutorial-callout-centered" role="dialog" aria-modal="true">
        <div class="tutorial-callout-title">New here?</div>
        <div class="tutorial-callout-text">Take a 60-second tour of the explorer.</div>
        <div class="tutorial-callout-actions">
          <button type="button" class="tutorial-skip">Skip</button>
          <div class="tutorial-nav">
            <button type="button" class="tutorial-next">Start tour</button>
          </div>
        </div>
      </div>`;
    document.body.appendChild(welcomeRoot);
    welcomeRoot.querySelector(".tutorial-skip").addEventListener("click", () => endTutorial(true));
    welcomeRoot.querySelector(".tutorial-next").addEventListener("click", startTutorial);
    welcomeRoot.querySelector(".tutorial-next").focus();
  }

  function removeWelcomePrompt() {
    if (welcomeRoot) {
      welcomeRoot.remove();
      welcomeRoot = null;
    }
  }

  // ---------- Wiring ----------

  window.addEventListener(
    "acclaim:ready",
    () => {
      if (localStorage.getItem(TUTORIAL_KEY)) return;
      setTimeout(showWelcomePrompt, 400);
    },
    { once: true }
  );

  document.addEventListener(
    "keydown",
    (e) => {
      if (e.key === "Escape" && (tState.active || welcomeRoot)) {
        e.stopPropagation();
        endTutorial(true);
      }
    },
    true
  );

  const helpBtn = document.getElementById("help-btn");
  if (helpBtn) {
    helpBtn.addEventListener("click", () => {
      removeWelcomePrompt();
      startTutorial();
    });
  }

  const askBtn = document.getElementById("ask-btn");
  if (askBtn) {
    askBtn.addEventListener("click", () => {
      if (tState.active) endTutorial(true);
    });
  }
})();
