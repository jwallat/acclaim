// Interactive "Ask question" mode.
// Toggles via the topbar button; in ask mode the question bubble becomes an
// editable textarea and results stream progressively into the existing panels.

(function () {
  // ── DOM refs ────────────────────────────────────────────────────────────────
  const askBtn        = document.getElementById("ask-btn");
  const questionText  = document.getElementById("question-text");
  const questionInput = document.getElementById("question-input");
  const evalBtn       = document.getElementById("eval-btn");
  const answerRole    = document.getElementById("answer-role");

  if (!askBtn) return;

  // ── State ───────────────────────────────────────────────────────────────────
  let isAsking = false;
  let activeEs = null;  // currently open EventSource

  // ── Ask-mode lifecycle ──────────────────────────────────────────────────────

  function enterAskMode() {
    isAsking = true;
    document.body.classList.add("ask-mode");
    askBtn.textContent = "Cancel";
    _clearPanels();
    questionInput.value = "";
    questionInput.focus();
  }

  function exitAskMode() {
    isAsking = false;
    document.body.classList.remove("ask-mode");
    askBtn.textContent = "Ask question";
    if (activeEs) { activeEs.close(); activeEs = null; }
    _resetControls();
    _setStage("Answer");
    setStatus("");
    if (state.records.length) render(state.currentIdx);
    else questionText.textContent = "—";
  }

  function _clearPanels() {
    questionText.textContent    = "";
    els.answerText.innerHTML    = "—";
    els.docsList.innerHTML      = "";
    els.claimsList.innerHTML    = "";
    els.claimsCount.textContent = "";
    els.metricsTable.innerHTML  = "";
    _setStage("Answer");
    setStatus("");
  }

  // ── Helpers ─────────────────────────────────────────────────────────────────

  function _setStage(label) {
    answerRole.textContent = label;
    answerRole.classList.toggle("answer-role-busy", label !== "Answer");
    setStatus(label === "Answer" ? "" : label);
  }

  function _resetControls() {
    evalBtn.disabled = false;
    evalBtn.textContent = "Evaluate";
    questionInput.disabled = false;
  }

  // ── Pipeline ────────────────────────────────────────────────────────────────

  function runPipeline() {
    const question = questionInput.value.trim();
    if (!question) return;

    // Clear stale content from any previous query before new results arrive.
    _clearPanels();
    questionText.textContent = question;

    evalBtn.disabled = true;
    evalBtn.textContent = "Running…";
    questionInput.disabled = true;

    const url = `/query?q=${encodeURIComponent(question)}&retriever=bm25`;
    const es  = new EventSource(url);
    activeEs  = es;
    _setStage("Retrieving…");

    // Guard against the browser firing a spurious error event when the server
    // closes the connection normally after "done" — without this, a late error
    // event from query N would cancel query N+1.
    let completed = false;

    // ── Stage 1: docs received ───────────────────────────────────────────────
    es.addEventListener("retrieved", (e) => {
      const { documents } = JSON.parse(e.data);
      renderDocs(documents, {});  // no citation numbers yet
      _setStage("Generating…");
    });

    // ── Stage 2: answer received ─────────────────────────────────────────────
    es.addEventListener("generated", (e) => {
      const { answer } = JSON.parse(e.data);
      // Wrap as a single pseudo-sentence so renderAnswer gets the shape it expects.
      renderAnswer([{ text: answer, start: 0, end: answer.length }], [], [], {});
      _setStage("Evaluating…");
    });

    // ── Stage 3: full result ─────────────────────────────────────────────────
    es.addEventListener("done", (e) => {
      completed = true;
      es.close();
      activeEs = null;

      const { result } = JSON.parse(e.data);

      // Append to state so normal navigation keeps working.
      const idx = state.records.length;
      state.records.push(result);
      const opt = document.createElement("option");
      opt.value = idx;
      opt.textContent = `#${idx} — ${truncate(result.question, 60)}`;
      els.select.appendChild(opt);
      els.runBadge.textContent = `${state.records.length} examples`;

      // Full render: colored sentences, citation links, metrics, claims.
      render(idx);
      _setStage("Answer");
      _resetControls();
    });

    // ── Error ────────────────────────────────────────────────────────────────
    es.addEventListener("error", (e) => {
      // Ignore the connection-drop event that fires after a clean server close.
      if (completed) return;
      es.close();
      activeEs = null;
      let msg = "Connection error — is the server running?";
      try { msg = JSON.parse(e.data).message; } catch { /* use default */ }
      _setStage("Answer");
      setStatus(`Error: ${msg}`, true);
      _resetControls();
    });
  }

  // ── Event bindings ──────────────────────────────────────────────────────────

  askBtn.addEventListener("click", () => {
    if (isAsking) exitAskMode();
    else          enterAskMode();
  });

  evalBtn.addEventListener("click", runPipeline);

  questionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); runPipeline(); }
  });
})();
