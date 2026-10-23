(() => {
  // ---- Start screen elements ----
  const startScreen = document.getElementById('start-screen');
  const playerNameDisplay = document.getElementById('player-name-display');
  const setSelect = document.getElementById('set-select');
  const startBtn = document.getElementById('start-btn');
  const startError = document.getElementById('start-error');
  const retryResearchEndBtn = document.getElementById('retry-research-end');
  const researchSessionInput = document.getElementById('research-session-id');
  const researchSessionLinkBtn = document.getElementById('research-session-link');
  const researchSessionStatus = document.getElementById('research-session-status');
  const endStudyStartBtn = document.getElementById('end-study-start-btn');
  const endStudyPuzzleBtn = document.getElementById('end-study-puzzle-btn');
  const themeToggle = document.getElementById('theme-toggle');
  const themeToggleIcon = document.getElementById('theme-toggle-icon');
  const themeToggleLabel = document.getElementById('theme-toggle-label');

  const THEME_STORAGE_KEY = 'codeArrange:theme';

  function normalizeTheme(theme) {
    return theme === 'dark' ? 'dark' : 'light';
  }

  function readSavedTheme() {
    try {
      return normalizeTheme(localStorage.getItem(THEME_STORAGE_KEY));
    } catch (e) {
      return 'light';
    }
  }

  function applyTheme(theme) {
    const normalizedTheme = normalizeTheme(theme);
    const isDark = normalizedTheme === 'dark';
    document.documentElement.dataset.theme = normalizedTheme;
    themeToggleIcon.textContent = isDark ? '☀' : '☾';
    themeToggleLabel.textContent = isDark ? 'Mode Terang' : 'Mode Gelap';
    themeToggle.setAttribute('aria-pressed', String(isDark));
    themeToggle.setAttribute('aria-label', isDark ? 'Aktifkan mode terang' : 'Aktifkan mode gelap');
    themeToggle.title = isDark ? 'Aktifkan mode terang' : 'Aktifkan mode gelap';
  }

  let currentTheme = readSavedTheme();
  applyTheme(currentTheme);

  themeToggle.addEventListener('click', () => {
    currentTheme = currentTheme === 'dark' ? 'light' : 'dark';
    applyTheme(currentTheme);
    try {
      localStorage.setItem(THEME_STORAGE_KEY, currentTheme);
    } catch (e) {
      // The theme still applies for this page even when storage is unavailable.
    }
  });

  // ---- Puzzle screen elements ----
  const puzzleScreen = document.getElementById('puzzle-screen');
  const activePlayerLabel = document.getElementById('active-player');
  const activePuzzleTypeLabel = document.getElementById('active-puzzle-type');
  const activePuzzleModeLabel = document.getElementById('active-puzzle-mode');
  const timerLabel = document.getElementById('timer');
  const descriptionContent = document.getElementById('description-content');
  const modeHint = document.getElementById('mode-hint');
  const availableList = document.getElementById('available-list');
  const solutionList = document.getElementById('solution-list');
  const finishBtn = document.getElementById('finish-btn');
  const tryAgainBtn = document.getElementById('try-again-btn');
  const prevPuzzleBtn = document.getElementById('prev-puzzle-btn');
  const nextPuzzleBtn = document.getElementById('next-puzzle-btn');
  const feedback = document.getElementById('feedback');
  const explanationBox = document.getElementById('explanation-box');
  const explanationContent = document.getElementById('explanation-content');
  const ratingModal = document.getElementById('rating-modal');
  const ratingFeedback = document.getElementById('rating-feedback');
  const ratingButtons = document.getElementById('rating-buttons');

  let currentPuzzleId = null;
  let currentPuzzleData = null; // full {id,title,type,description,lines} from the server for the attempt in progress
  let puzzleList = []; // full ordered list from /api/puzzles, used to find "the next puzzle"
  let playerName = ''; // set once from /api/config; the player can't change it
  let researchSessionId = '';
  let researchStudyEnded = false;
  let researchSetOpen = false;
  let pendingSetEndOutcome = '';
  let ratingStartMarked = false;
  let ratingSaveInProgress = false;
  let activePuzzleSet = '';
  let availablePuzzleSets = [];
  let timeLimitMinutes = null; // set once from /api/config if master_config.json contains the active set
  let showTimerUI = true; // whether the timer display is shown in the UI (default true, can be toggled in master_config.json)
  const puzzleSessionCache = new Map(); // caches per-puzzle DOM arrangement and state so participants can freely skip and return
  let startTimeMs = null; // session start time (persists across Try Again and Next Puzzle)
  let timerIntervalId = null;
  let finished = false;
  let sessionExpired = false; // true when the global session time limit has run out
  let setCompleted = false; // true when all puzzles in the set have been solved
  let isDragging = false; // true while a drag gesture is active; prevents click-to-move from firing
  let finishedElapsedMs = null; // frozen elapsed time once Finish has been clicked
  let finishedPerLine = null; // correctness map from the last grading result, for restoring highlights
  let finishedExplanation = null; // explanation text that came back with the last grading result
  let reviewMode = false; // set once from /api/config; whether this server instance asks for ratings
  let pendingRatingAttemptId = null; // set while the rating dialog is open, for the attempt it's rating
  let pausedAtMs = null; // timestamp when exam timer was paused by the rating dialog

  function isTimerUIVisible(val) {
    if (val === undefined || val === null) return true;
    if (typeof val === 'boolean') return val;
    if (typeof val === 'number') return val !== 0;
    if (typeof val === 'string') {
      const s = val.trim().toLowerCase();
      if (s === 'off' || s === 'false' || s === '0' || s === 'hide' || s === 'hidden') return false;
      return true;
    }
    return true;
  }

  function applyTimerUIVisibility() {
    if (!showTimerUI) {
      timerLabel.classList.add('hidden');
    } else {
      timerLabel.classList.remove('hidden');
    }
  }

  function saveCurrentPuzzleToCache(isCorrectOverride) {
    if (!currentPuzzleId || !currentPuzzleData) return;
    const isCorrect = typeof isCorrectOverride === 'boolean'
      ? isCorrectOverride
      : (finishedPerLine ? Object.values(finishedPerLine).every(Boolean) : false);

    puzzleSessionCache.set(currentPuzzleId, {
      puzzleId: currentPuzzleId,
      puzzle: currentPuzzleData,
      availableOrder: [...availableList.children].map((li) => Number(li.dataset.id)),
      solutionOrder: [...solutionList.children].map((li) => Number(li.dataset.id)),
      finished,
      correct: isCorrect,
      finishedElapsedMs,
      finishedPerLine,
      finishedExplanation,
      pendingRatingAttemptId,
      feedbackText: feedback.textContent,
      feedbackClass: feedback.className,
    });
  }

  function areAllPuzzlesCompleted() {
    if (!puzzleList || puzzleList.length === 0) return false;
    return puzzleList.every((p) => {
      const cached = puzzleSessionCache.get(p.id);
      return cached && cached.finished && cached.correct;
    });
  }

  function getCompletedPuzzlesCount() {
    if (!puzzleList || puzzleList.length === 0) return 0;
    return puzzleList.filter((p) => {
      const cached = puzzleSessionCache.get(p.id);
      return cached && cached.finished && cached.correct;
    }).length;
  }

  // ---- Resuming after a page refresh (sessionStorage) --------------------
  //
  // sessionStorage is scoped to this browser tab and cleared when the tab
  // closes -- exactly the lifetime a "don't lose my progress on refresh,
  // but don't resurrect it days later in a new tab" feature wants, so
  // there's no need to reach for anything heavier (a server-side session,
  // localStorage with manual expiry, etc). Everything needed to fully
  // reconstruct the screen is saved: which puzzle, its exact line data as
  // originally fetched (so a resume doesn't get a *different* random
  // shuffle or different disambiguation tags than what the player was
  // looking at), the current arrangement in both lists, the original start
  // time (so the timer just keeps counting real elapsed time straight
  // through the refresh), and the finished/graded state if Finish was
  // already clicked.

  const STORAGE_KEY = 'codeArrange:puzzleState';

  function saveState() {
    if (!currentPuzzleId || !currentPuzzleData) return;
    saveCurrentPuzzleToCache();
    const cacheObj = {};
    for (const [k, v] of puzzleSessionCache.entries()) {
      cacheObj[k] = v;
    }
    const state = {
      puzzleSet: activePuzzleSet,
      puzzleId: currentPuzzleId,
      puzzleSessionCache: cacheObj,
      puzzle: currentPuzzleData,
      availableOrder: [...availableList.children].map((li) => Number(li.dataset.id)),
      solutionOrder: [...solutionList.children].map((li) => Number(li.dataset.id)),
      startTimeMs,
      finished,
      sessionExpired,
      setCompleted,
      finishedElapsedMs,
      finishedPerLine,
      finishedExplanation,
      pendingRatingAttemptId,
      pausedAtMs,
      feedbackText: feedback.textContent,
      feedbackClass: feedback.className,
    };
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (e) {
      // sessionStorage unavailable (private-mode quirks, quota, etc) --
      // persistence is a nice-to-have, so fail silently rather than
      // breaking the puzzle itself over it.
    }
  }

  function loadSavedState() {
    try {
      const raw = sessionStorage.getItem(STORAGE_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch (e) {
      return null;
    }
  }

  function clearSavedState() {
    try {
      sessionStorage.removeItem(STORAGE_KEY);
    } catch (e) {
      // nothing to do if storage isn't available
    }
  }

  // ---- Start screen: load the locked player name + puzzle list ----
  //
  // The player name is fixed for this server instance (passed as a CLI
  // argument when the server was started) and is only ever displayed here,
  // never editable -- there's no input to type a different name into.

  async function loadConfig() {
    const res = await fetch('/api/config');
    const config = await res.json();
    playerName = config.playerName;
    researchSessionId = config.researchSessionId || '';
    researchStudyEnded = Boolean(config.researchStudyEnded);
    researchSetOpen = Boolean(config.researchSetOpen);
    researchSessionInput.value = researchSessionId;
    updateResearchControls();
    activePuzzleSet = config.puzzleSet || '';
    playerNameDisplay.textContent = playerName;
    reviewMode = Boolean(config.reviewMode);
    if (config.setConfig && config.setConfig.timeLimitMinutes) {
      timeLimitMinutes = Number(config.setConfig.timeLimitMinutes);
    }
    if (config.setConfig && config.setConfig.showTimerUI !== undefined) {
      showTimerUI = isTimerUIVisible(config.setConfig.showTimerUI);
    } else {
      showTimerUI = true;
    }
    applyTimerUIVisibility();

    if (config.sessionStartedAtMs) {
      const elapsed = Date.now() - config.sessionStartedAtMs;
      const limitMs = timeLimitMinutes ? timeLimitMinutes * 60 * 1000 : Infinity;
      if (elapsed < limitMs) {
        startTimeMs = config.sessionStartedAtMs;
      } else {
        try {
          await fetch('/api/session/reset', { method: 'POST' });
        } catch (e) {}
        startTimeMs = null;
        sessionExpired = false;
      }
    }
  }

  function updateResearchControls() {
    const linked = Boolean(researchSessionId);
    researchSessionStatus.textContent = linked
      ? `Terhubung ke sesi ${researchSessionId}${researchStudyEnded ? ' (sudah diakhiri)' : ''}.`
      : 'Opsional untuk penggunaan biasa; wajib dihubungkan untuk sesi penelitian.';
    researchSessionStatus.className = linked ? 'feedback success' : 'feedback';
    endStudyStartBtn.disabled = !linked || researchStudyEnded;
    endStudyPuzzleBtn.disabled = !linked || researchStudyEnded;
    startBtn.disabled = researchStudyEnded || Boolean(pendingSetEndOutcome) || researchSetOpen;
    retryResearchEndBtn.classList.toggle('hidden', !pendingSetEndOutcome);
  }

  async function endResearchSet(outcome) {
    if (!researchSessionId || !researchSetOpen) return;
    try {
      await recordResearchEvent('set_ended', { outcome });
      pendingSetEndOutcome = '';
    } catch (error) {
      pendingSetEndOutcome = outcome;
      startError.textContent = `Penanda akhir set gagal disimpan: ${error.message}`;
    }
    updateResearchControls();
  }

  retryResearchEndBtn.addEventListener('click', () => endResearchSet(pendingSetEndOutcome));

  async function recordResearchEvent(eventType, details = {}) {
    if (!researchSessionId) return false;
    const res = await fetch('/api/research/events', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ eventType, eventTimeMs: Date.now(), ...details }),
    });
    let data = {};
    try { data = await res.json(); } catch (e) {}
    if (!res.ok) throw new Error(data.error || 'Event penelitian gagal disimpan.');
    if (eventType === 'study_ended') {
      researchStudyEnded = true;
      updateResearchControls();
    }
    if (eventType === 'set_started') researchSetOpen = true;
    if (eventType === 'set_ended' || eventType === 'study_ended') researchSetOpen = false;
    if (eventType === 'set_ended' || eventType === 'set_started') updateResearchControls();
    return true;
  }

  researchSessionLinkBtn.addEventListener('click', async () => {
    researchSessionStatus.textContent = 'Memvalidasi folder sesi sensor…';
    researchSessionStatus.className = 'feedback';
    researchSessionLinkBtn.disabled = true;
    try {
      const res = await fetch('/api/research/session/link', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId: researchSessionInput.value.trim() }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Sesi sensor tidak dapat dihubungkan.');
      researchSessionId = data.sessionId;
      researchStudyEnded = false;
      updateResearchControls();
    } catch (error) {
      researchSessionStatus.textContent = error.message;
      researchSessionStatus.className = 'feedback error';
    } finally {
      researchSessionLinkBtn.disabled = false;
    }
  });

  async function endResearchStudy() {
    if (!researchSessionId || researchStudyEnded) return;
    if (!window.confirm('Akhiri penandaan sesi penelitian ini? Perekaman sensor tetap berjalan sampai dihentikan terpisah.')) return;
    endStudyStartBtn.disabled = true;
    endStudyPuzzleBtn.disabled = true;
    try {
      await recordResearchEvent('study_ended');
    } catch (error) {
      researchSessionStatus.textContent = error.message;
      researchSessionStatus.className = 'feedback error';
      updateResearchControls();
    }
  }
  endStudyStartBtn.addEventListener('click', endResearchStudy);
  endStudyPuzzleBtn.addEventListener('click', endResearchStudy);

  async function loadPuzzleList() {
    const res = await fetch('/api/puzzles');
    if (!res.ok) throw new Error('Gagal memuat daftar soal.');
    puzzleList = await res.json();
    return puzzleList;
  }

  async function loadPuzzleSets() {
    const res = await fetch('/api/puzzle-sets');
    if (!res.ok) throw new Error('Gagal memuat daftar set soal.');
    availablePuzzleSets = await res.json();
    setSelect.innerHTML = '';
    for (const set of availablePuzzleSets) {
      const option = document.createElement('option');
      option.value = set.key;
      option.textContent = set.label;
      setSelect.appendChild(option);
    }
    if (activePuzzleSet && availablePuzzleSets.some((set) => set.key === activePuzzleSet)) {
      setSelect.value = activePuzzleSet;
    }
    startBtn.disabled = availablePuzzleSets.length === 0 || researchStudyEnded;
    if (availablePuzzleSets.length === 0) {
      startError.textContent = 'Tidak ada set soal yang tersedia.';
    }
  }

  // "Next puzzle" and "Prev puzzle" walk puzzleList in the same flat order
  // used to build the dropdown (server's filename sort), ignoring the type
  // groupings -- so they still have a well-defined neighbour even across a
  // type boundary.
  function hasNextPuzzle() {
    const index = puzzleList.findIndex((p) => p.id === currentPuzzleId);
    return index !== -1 && index < puzzleList.length - 1;
  }

  function getNextPuzzle() {
    const index = puzzleList.findIndex((p) => p.id === currentPuzzleId);
    return index === -1 ? null : puzzleList[index + 1] || null;
  }

  function hasPrevPuzzle() {
    const index = puzzleList.findIndex((p) => p.id === currentPuzzleId);
    return index > 0;
  }

  function getPrevPuzzle() {
    const index = puzzleList.findIndex((p) => p.id === currentPuzzleId);
    return index > 0 ? puzzleList[index - 1] : null;
  }

  // The label reflects what clicking the button will actually do. Both this
  // and the click handler call hasNextPuzzle()/getNextPuzzle() fresh rather
  // than caching a decision, so they can't drift out of sync with each
  // other -- nothing changes puzzleList or currentPuzzleId between an
  // attempt starting and the button being clicked, so recomputing is cheap
  // and keeps there being exactly one source of truth.
  function updateNextPuzzleButtonLabel() {
    if (sessionExpired || setCompleted || !hasNextPuzzle()) {
      nextPuzzleBtn.textContent = 'Kembali ke Menu Awal';
    } else {
      nextPuzzleBtn.textContent = 'Soal Berikutnya';
    }
  }

  function updateNextPuzzleButtonState() {
    if (sessionExpired || setCompleted) {
      nextPuzzleBtn.disabled = false;
      return;
    }
    if (hasNextPuzzle()) {
      // Peserta dapat langsung beralih ke soal berikutnya kapan saja (melewati/skip soal)
      nextPuzzleBtn.disabled = false;
    } else {
      // Di soal terakhir: HANYA aktif jika seluruh soal dalam set/folder telah selesai dengan benar
      nextPuzzleBtn.disabled = !areAllPuzzlesCompleted();
    }
  }

  // Tombol Soal Sebelumnya aktif jika ada soal sebelumnya dan sesi belum selesai/timeout
  function updatePrevPuzzleButtonState() {
    if (sessionExpired || setCompleted) {
      prevPuzzleBtn.disabled = true;
      return;
    }
    prevPuzzleBtn.disabled = !hasPrevPuzzle();
  }

  const TYPE_LABELS = {
    basics: 'Dasar',
    variables: 'Variabel',
    loops: 'Perulangan',
    arrays: 'Array',
    string: 'String',
    strings: 'String',
    uncategorized: 'Lainnya',
  };

  function humanizeType(type) {
    if (!type) return '';
    const lower = type.toLowerCase();
    if (TYPE_LABELS[lower]) return TYPE_LABELS[lower];
    return type
      .split(/[_-]+/)
      .filter(Boolean)
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  }

  // ---- Puzzle modes --------------------------------------------------------
  //
  // "mode" is what kind of task a puzzle is (separate from "type", its topic).
  // The default, "arrange", is the original behaviour and deliberately shows
  // no extra badge or hint, so existing puzzles look exactly as before. The
  // other modes get a badge in the header and a one-line instruction under
  // the description.

  const MODE_INFO = {
    arrange: { label: '', hint: '' },
    output: {
      label: 'Tebak Output',
      hint: 'Perhatikan apa yang dicetak oleh program pada deskripsi, lalu susun baris output sesuai urutan tampilannya.',
    },
    debug: {
      label: 'Debug & Perbaiki',
      hint:
        'Program pada deskripsi memiliki bug / kesalahan. Susun kembali versi yang benar dari potongan kode yang tersedia. ' +
        'Beberapa potongan kode terlihat benar namun masih memiliki bug, sehingga tidak semua potongan kode digunakan.',
    },
  };

  function applyMode(mode) {
    const info = MODE_INFO[mode || 'arrange'] || { label: humanizeType(mode), hint: '' };
    activePuzzleModeLabel.textContent = info.label;
    modeHint.textContent = info.hint;
    modeHint.classList.toggle('hidden', !info.hint);
  }

  function describeWrongLines(count, mode) {
    if (mode === 'debug') {
      return `${count} baris yang Anda gunakan masih mengandung bug.`;
    }
    return `${count} baris yang Anda gunakan tidak termasuk dalam solusi yang benar.`;
  }

  function showExplanation(text) {
    if (!text) {
      hideExplanation();
      return;
    }
    renderRichText(explanationContent, text);
    explanationBox.classList.remove('hidden');
  }

  function hideExplanation() {
    explanationBox.classList.add('hidden');
    explanationContent.innerHTML = '';
  }

  // ---- Rating dialog (review mode) ------------------------------------
  //
  // Shown after Finish (including timeout) when the server was started with
  // --review (reviewMode, set once from /api/config). Clicking a number
  // posts straight to the server; there is no separate "submit" step and no
  // skip button so the rating is captured reliably. The exam timer is paused
  // while this dialog is open.

  for (let i = 1; i <= 10; i++) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = String(i);
    btn.addEventListener('click', () => submitRating(i));
    ratingButtons.appendChild(btn);
  }

  async function showRatingDialog(attemptId) {
    pendingRatingAttemptId = attemptId;
    ratingStartMarked = false;
    ratingFeedback.textContent = feedback.textContent;
    ratingFeedback.className = feedback.className;
    ratingModal.classList.remove('hidden');

    // Freeze client-side timer while rating dialog is open (if not expired)
    if (!sessionExpired && !setCompleted) {
      stopTimer();
      pausedAtMs = Date.now();
    }
    saveState();
    if (researchSessionId) {
      try {
        await recordResearchEvent('rating_started', { attemptId });
        ratingStartMarked = true;
      } catch (error) {
        ratingFeedback.textContent += ` Penanda rating gagal disimpan: ${error.message}. Pilih rating untuk mencoba lagi.`;
        ratingFeedback.className = 'feedback error';
      }
    }
  }

  function hideRatingDialog() {
    pendingRatingAttemptId = null;
    ratingModal.classList.add('hidden');

    // Unfreeze client-side timer if exam is still active
    if (pausedAtMs && !sessionExpired && !setCompleted) {
      const pausedDuration = Math.max(0, Date.now() - pausedAtMs);
      pausedAtMs = null;
      if (startTimeMs) {
        startTimeMs += pausedDuration;
      }
      startTimer();
      updateTimerLabel();
    }
    saveState();
  }

  async function submitRating(rating) {
    const attemptId = pendingRatingAttemptId;
    if (!attemptId || ratingSaveInProgress) return;
    ratingSaveInProgress = true;
    try {
      if (researchSessionId && !ratingStartMarked) {
        try {
          await recordResearchEvent('rating_started', { attemptId });
          ratingStartMarked = true;
        } catch (error) {
          ratingFeedback.textContent = `Penanda rating gagal disimpan: ${error.message}. Coba lagi.`;
          ratingFeedback.className = 'feedback error';
          return;
        }
      }
      try {
        const res = await fetch(`/api/attempts/${encodeURIComponent(attemptId)}/rating`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ rating }),
        });
        if (res.ok) {
          const data = await res.json();
          if (data.sessionStartedAtMs) {
            startTimeMs = data.sessionStartedAtMs;
            pausedAtMs = null;
          }
        }
      } catch (err) {
        // Rating tetap opsional jika koneksi ke log attempt sempat terganggu.
      }
      if (researchSessionId) {
        try { await recordResearchEvent('rating_ended', { attemptId }); }
        catch (error) {
          ratingFeedback.textContent = `Penanda akhir rating gagal disimpan: ${error.message}. Coba lagi.`;
          ratingFeedback.className = 'feedback error';
          return;
        }
      }
      hideRatingDialog();
    } finally {
      ratingSaveInProgress = false;
    }
  }

  async function selectPuzzleSet(setKey) {
    const res = await fetch('/api/session/select-set', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ puzzleSet: setKey }),
    });
    let data = {};
    try {
      data = await res.json();
    } catch (e) {}
    if (!res.ok) throw new Error(data.error || 'Set soal tidak dapat dipilih.');

    activePuzzleSet = data.puzzleSet;
    const config = data.setConfig || {};
    timeLimitMinutes = config.timeLimitMinutes ? Number(config.timeLimitMinutes) : null;
    showTimerUI = config.showTimerUI === undefined ? true : isTimerUIVisible(config.showTimerUI);
    applyTimerUIVisibility();
    return data;
  }

  startBtn.addEventListener('click', async () => {
    if (!setSelect.value) {
      startError.textContent = 'Tidak ada set soal yang tersedia untuk dimulai.';
      return;
    }
    startError.textContent = '';

    try {
      await selectPuzzleSet(setSelect.value);
    } catch (e) {
      startError.textContent = e.message || 'Set soal tidak dapat dimuat.';
      return;
    }

    // Selecting a set always discards the previous set's browser state,
    // including when loading the new puzzle list subsequently fails.
    clearSavedState();
    stopTimer();
    startTimeMs = null;
    sessionExpired = false;
    setCompleted = false;
    currentPuzzleId = null;
    currentPuzzleData = null;
    puzzleSessionCache.clear();

    try {
      await loadPuzzleList();
    } catch (e) {
      startError.textContent = e.message || 'Gagal memuat soal dari set tersebut.';
      return;
    }
    if (puzzleList.length === 0) {
      startError.textContent = 'Tidak ada soal ditemukan pada set soal ini.';
      return;
    }

    // Starting from the home screen always creates a fresh set session.
    startScreen.classList.add('hidden');
    puzzleScreen.classList.remove('hidden');
    activePlayerLabel.textContent = `Peserta: ${playerName}`;

    await beginAttempt(puzzleList[0].id);
  });

  // Allow pressing Enter in the set picker to start.
  setSelect.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') startBtn.click();
  });

  // ---- Puzzle loading / attempt lifecycle ----

  // Starts an attempt at puzzleId.
  // Fetches a fresh copy of the puzzle from the server.
  // The global session timer keeps running continuously across attempts.
  async function beginAttempt(puzzleId, forceFresh = false) {
    // Simpan susunan tile soal yang sedang aktif sebelum berpindah
    if (currentPuzzleId && currentPuzzleId !== puzzleId && currentPuzzleData) {
      saveCurrentPuzzleToCache();
    }

    currentPuzzleId = puzzleId;
    ratingModal.classList.add('hidden');
    pendingRatingAttemptId = null;
    pausedAtMs = null;

    if (!forceFresh && puzzleSessionCache.has(puzzleId)) {
      const cached = puzzleSessionCache.get(puzzleId);
      currentPuzzleData = cached.puzzle;
      finished = Boolean(cached.finished);
      finishedElapsedMs = cached.finishedElapsedMs;
      finishedPerLine = cached.finishedPerLine;
      finishedExplanation = cached.finishedExplanation || null;
      pendingRatingAttemptId = cached.pendingRatingAttemptId || null;

      activePuzzleTypeLabel.textContent =
        cached.puzzle.type && cached.puzzle.type !== 'uncategorized' ? humanizeType(cached.puzzle.type) : '';
      applyMode(cached.puzzle.mode);
      renderDescription(cached.puzzle.description);
      renderTiles(cached.puzzle.lines, cached.availableOrder, cached.solutionOrder);

      if (researchSessionId) {
        try {
          await recordResearchEvent('puzzle_started', { puzzleId, eventTimeMs: Date.now() });
        } catch (error) {
          feedback.textContent = `Penanda soal gagal disimpan: ${error.message}. Hubungi peneliti sebelum melanjutkan.`;
          feedback.className = 'feedback error';
          finishBtn.disabled = true;
          nextPuzzleBtn.disabled = true;
          return;
        }
      }

      updatePrevPuzzleButtonState();
      updateNextPuzzleButtonState();
      updateNextPuzzleButtonLabel();

      if (finished) {
        finishBtn.disabled = true;
        tryAgainBtn.disabled = sessionExpired || setCompleted;
        revealIndentation();
        for (const li of solutionList.children) {
          const id = Number(li.dataset.id);
          if (finishedPerLine && Object.prototype.hasOwnProperty.call(finishedPerLine, id)) {
            li.classList.add(finishedPerLine[id] ? 'correct' : 'incorrect');
          }
        }
        feedback.textContent = cached.feedbackText || '';
        feedback.className = cached.feedbackClass || 'feedback';
        showExplanation(finishedExplanation);
      } else {
        finishBtn.disabled = sessionExpired;
        tryAgainBtn.disabled = sessionExpired;
        clearFeedback();
      }

      if (!startTimeMs) {
        startTimeMs = Date.now();
        sessionExpired = false;
      }
      applyTimerUIVisibility();
      if (!sessionExpired && !setCompleted) {
        startTimer();
      } else {
        updateTimerLabel();
      }
      saveState();
      return;
    }

    // Inisialisasi awal / fresh attempt dari server
    currentPuzzleData = null;
    finished = false;
    finishedElapsedMs = null;
    finishedPerLine = null;
    finishedExplanation = null;
    clearFeedback();

    const res = await fetch(`/api/puzzles/${encodeURIComponent(puzzleId)}`);
    if (!res.ok) {
      descriptionContent.innerHTML = '';
      descriptionContent.appendChild(textParagraph('Gagal memuat soal tersebut.'));
      return;
    }
    const puzzle = await res.json();
    currentPuzzleData = puzzle;
    activePuzzleTypeLabel.textContent = puzzle.type && puzzle.type !== 'uncategorized' ? humanizeType(puzzle.type) : '';
    applyMode(puzzle.mode);
    renderDescription(puzzle.description);
    renderTiles(puzzle.lines, puzzle.lines.map((line) => line.id), []);

    if (researchSessionId) {
      try {
        if (!researchSetOpen) await recordResearchEvent('set_started');
        await recordResearchEvent('puzzle_started', { puzzleId, eventTimeMs: Date.now() });
      } catch (error) {
        feedback.textContent = `Penanda soal gagal disimpan: ${error.message}. Hubungi peneliti sebelum melanjutkan.`;
        feedback.className = 'feedback error';
        finishBtn.disabled = true;
        nextPuzzleBtn.disabled = true;
        return;
      }
    }

    finishBtn.disabled = sessionExpired;
    tryAgainBtn.disabled = sessionExpired;
    updatePrevPuzzleButtonState();
    updateNextPuzzleButtonState();
    updateNextPuzzleButtonLabel();

    if (!startTimeMs) {
      startTimeMs = Date.now();
      sessionExpired = false;
    }

    applyTimerUIVisibility();
    if (!sessionExpired && !setCompleted) {
      startTimer();
    } else {
      updateTimerLabel();
    }

    saveCurrentPuzzleToCache();
    saveState();
  }

  // Reconstructs the screen from a previously saved state (page refresh)
  function resumeAttempt(saved) {
    if (saved.puzzleSessionCache) {
      puzzleSessionCache.clear();
      for (const [k, v] of Object.entries(saved.puzzleSessionCache)) {
        puzzleSessionCache.set(k, v);
      }
    }
    currentPuzzleId = saved.puzzleId;
    currentPuzzleData = saved.puzzle;
    finished = Boolean(saved.finished);
    finishedElapsedMs = saved.finishedElapsedMs;
    finishedPerLine = saved.finishedPerLine;
    finishedExplanation = saved.finishedExplanation || null;
    pendingRatingAttemptId = saved.pendingRatingAttemptId || null;
    pausedAtMs = saved.pausedAtMs || null;
    startTimeMs = saved.startTimeMs || startTimeMs;
    sessionExpired = Boolean(saved.sessionExpired);
    setCompleted = Boolean(saved.setCompleted);

    if (timeLimitMinutes && startTimeMs && !setCompleted) {
      if (Date.now() - startTimeMs >= timeLimitMinutes * 60 * 1000) {
        sessionExpired = true;
      }
    }

    startScreen.classList.add('hidden');
    puzzleScreen.classList.remove('hidden');
    activePlayerLabel.textContent = `Peserta: ${playerName}`;
    activePuzzleTypeLabel.textContent =
      saved.puzzle && saved.puzzle.type && saved.puzzle.type !== 'uncategorized' ? humanizeType(saved.puzzle.type) : '';
    applyMode(saved.puzzle ? saved.puzzle.mode : 'arrange');
    if (saved.puzzle) {
      renderDescription(saved.puzzle.description);
      const cached = puzzleSessionCache.get(saved.puzzleId);
      const avail = cached ? cached.availableOrder : saved.availableOrder;
      const sol = cached ? cached.solutionOrder : saved.solutionOrder;
      renderTiles(saved.puzzle.lines, avail, sol);
    }

    updateNextPuzzleButtonLabel();
    updateNextPuzzleButtonState();
    updatePrevPuzzleButtonState();
    applyTimerUIVisibility();

    if (finished) {
      finishBtn.disabled = true;
      tryAgainBtn.disabled = sessionExpired || setCompleted;
      if (sessionExpired || setCompleted || !hasNextPuzzle()) {
        nextPuzzleBtn.textContent = 'Kembali ke Menu Awal';
      }
      revealIndentation();
      for (const li of solutionList.children) {
        const id = Number(li.dataset.id);
        if (finishedPerLine && Object.prototype.hasOwnProperty.call(finishedPerLine, id)) {
          li.classList.add(finishedPerLine[id] ? 'correct' : 'incorrect');
        }
      }
      feedback.textContent = saved.feedbackText || '';
      feedback.className = saved.feedbackClass || 'feedback';
      showExplanation(finishedExplanation);

      if (setCompleted) {
        stopTimer();
        const displayLabel = formatElapsed(finishedElapsedMs || 0);
        timerLabel.textContent = timeLimitMinutes
          ? `${displayLabel} / ${timeLimitMinutes}:00`
          : displayLabel;
        timerLabel.classList.remove('timeout');
      } else if (!sessionExpired) {
        startTimer();
      } else {
        stopTimer();
        updateTimerLabel();
      }

      if (reviewMode && saved.pendingRatingAttemptId) {
        showRatingDialog(saved.pendingRatingAttemptId);
      }
    } else {
      if (sessionExpired) {
        handleSessionTimeout();
      } else {
        finishBtn.disabled = false;
        tryAgainBtn.disabled = false;
        clearFeedback();
        startTimer();
      }
    }
  }

  // ---- Description rendering ---------------------------------------------
  //
  // A puzzle's description is plain text, but may contain a fenced code
  // snippet delimited by ``` on its own, exactly like Markdown. Splitting on
  // ``` alternates between text segments (even indices) and code segments
  // (odd indices). Code segments are rendered line-by-line, each with its
  // own line number, in a monospaced block that preserves indentation
  // exactly -- rather than as a single wrapped paragraph, where the
  // browser's normal whitespace collapsing would run everything onto one
  // line and destroy the snippet's formatting.

  function renderDescription(description) {
    renderRichText(descriptionContent, description);
  }

  // Shared by the description and the post-Finish explanation, so both can
  // contain fenced code snippets that render the same line-numbered way.
  function renderRichText(container, text) {
    container.innerHTML = '';
    const segments = text.split('```');
    segments.forEach((segment, index) => {
      const isCode = index % 2 === 1;
      if (isCode) {
        const codeText = segment.replace(/^\n/, '').replace(/\n$/, '');
        if (codeText.trim() !== '') {
          container.appendChild(codeBlock(codeText));
        }
      } else if (segment.trim() !== '') {
        container.appendChild(textParagraph(segment.trim()));
      }
    });
  }

  function textParagraph(text) {
    const div = document.createElement('div');
    div.className = 'description-text-block';
    div.textContent = text;
    return div;
  }

  function codeBlock(codeText) {
    const container = document.createElement('div');
    container.className = 'description-code';
    const lines = codeText.split('\n');
    lines.forEach((lineText, i) => {
      const lineDiv = document.createElement('div');
      lineDiv.className = 'description-code-line';

      const number = document.createElement('span');
      number.className = 'description-code-line-number';
      number.textContent = String(i + 1);

      const text = document.createElement('span');
      text.className = 'description-code-line-text';
      text.textContent = lineText;

      lineDiv.appendChild(number);
      lineDiv.appendChild(text);
      container.appendChild(lineDiv);
    });
    return container;
  }

  // ---- Tile rendering ------------------------------------------------------
  //
  // Builds the Available/Your Program tiles from a puzzle's line data plus
  // two explicit id-orderings -- one per list. A fresh attempt passes the
  // server's shuffled order for "available" and an empty "solution"; a
  // resumed attempt passes back whatever arrangement was saved. Either way
  // the actual DOM-building logic (and therefore tag/indentation handling)
  // is identical, so the two cases can never visually drift apart.

  function renderTiles(lines, availableOrder, solutionOrder) {
    const lineById = new Map(lines.map((line) => [line.id, line]));
    availableList.innerHTML = '';
    solutionList.innerHTML = '';
    for (const id of availableOrder) {
      const line = lineById.get(id);
      if (line) availableList.appendChild(createLineElement(line));
    }
    for (const id of solutionOrder) {
      const line = lineById.get(id);
      if (line) solutionList.appendChild(createLineElement(line));
    }
  }

  function createLineElement(line) {
    const li = document.createElement('li');

    const codeSpan = document.createElement('span');
    codeSpan.className = 'code-text';
    codeSpan.textContent = stripLeadingWhitespace(line.code);
    li.appendChild(codeSpan);

    // The server tags lines that would otherwise look identical once
    // indentation is stripped (see computeDisambiguationTags in server.js).
    // The number reflects that line's real relative order among its
    // look-alikes, so it's something to reason about, not just a random label.
    if (line.tag !== null && line.tag !== undefined) {
      const tagSpan = document.createElement('span');
      tagSpan.className = 'line-tag';
      tagSpan.textContent = `#${line.tag}`;
      li.appendChild(tagSpan);
    }

    li.dataset.id = String(line.id);
    li.dataset.code = line.code; // full original text, indentation included; revealed after Finish
    li.draggable = true;
    li.addEventListener('dragstart', onDragStart);
    li.addEventListener('dragend', onDragEnd);

    // ---- Click-to-move -------------------------------------------------------
    // Clicking a tile moves it to the opposite list (available ↔ solution)
    // in one tap, without needing to drag. Only active when the puzzle is
    // in progress; ignored if a drag gesture is happening (isDragging guard
    // prevents this from firing at the end of a drag on touch devices).
    li.addEventListener('click', () => {
      if (finished || sessionExpired || isDragging) return;
      const isInAvailable = li.parentElement === availableList;
      const target = isInAvailable ? solutionList : availableList;
      target.appendChild(li);
      // Brief animation so the tile's arrival is noticed
      li.classList.remove('tile-moving');
      // Force reflow to restart animation even if already present
      void li.offsetWidth;
      li.classList.add('tile-moving');
      li.addEventListener('animationend', () => li.classList.remove('tile-moving'), { once: true });
      saveState();
    });

    return li;
  }

  // Leading indentation is a strong hint about a line's nesting/position,
  // so it's hidden while the puzzle is in progress and only shown once the
  // learner clicks Finish. Only leading whitespace is stripped -- spacing
  // in the middle of a line (e.g. around operators) is left untouched.
  function stripLeadingWhitespace(code) {
    return code.replace(/^[ \t]+/, '');
  }

  function revealIndentation() {
    for (const li of [...availableList.children, ...solutionList.children]) {
      li.textContent = li.dataset.code; // drops the disambiguation tag too; no longer needed once revealed
    }
  }

  // ---- Timer ----

  function startTimer() {
    stopTimer();
    updateTimerLabel();
    timerIntervalId = setInterval(updateTimerLabel, 200);
  }

  function stopTimer() {
    if (timerIntervalId !== null) {
      clearInterval(timerIntervalId);
      timerIntervalId = null;
    }
  }

  function updateTimerLabel() {
    applyTimerUIVisibility();
    if (!startTimeMs) {
      timerLabel.textContent = '0:00';
      return;
    }
    if (setCompleted) {
      const displayElapsed = finishedElapsedMs || (startTimeMs ? Date.now() - startTimeMs : 0);
      const label = formatElapsed(displayElapsed);
      timerLabel.textContent = timeLimitMinutes ? `${label} / ${timeLimitMinutes}:00` : label;
      timerLabel.classList.remove('timeout');
      return;
    }
    const elapsedMs = Math.max(0, Date.now() - startTimeMs);
    let label = formatElapsed(elapsedMs);

    if (timeLimitMinutes) {
      const limitMs = timeLimitMinutes * 60 * 1000;
      const displayElapsed = sessionExpired || elapsedMs >= limitMs ? limitMs : elapsedMs;
      label = `${formatElapsed(displayElapsed)} / ${timeLimitMinutes}:00`;

      if (elapsedMs >= limitMs) {
        if (!sessionExpired) {
          sessionExpired = true;
          handleSessionTimeout();
        }
      }
    }

    timerLabel.textContent = label;
    if (sessionExpired) {
      timerLabel.classList.add('timeout');
    } else {
      timerLabel.classList.remove('timeout');
    }
  }

  async function handleSessionTimeout() {
    if (setCompleted) return;
    sessionExpired = true;
    stopTimer();
    finishBtn.disabled = true;
    tryAgainBtn.disabled = true;
    nextPuzzleBtn.textContent = 'Kembali ke Menu Awal';
    nextPuzzleBtn.disabled = false;

    if (!finished && currentPuzzleId) {
      await submitAttempt('timeout');
    } else {
      await endResearchSet('timed_out');
      const msg = 'Batas waktu telah tercapai. Sesi ujian telah berakhir.';
      if (!feedback.textContent) {
        feedback.textContent = msg;
        feedback.className = 'feedback error';
      } else if (!feedback.textContent.includes('Sesi ujian telah berakhir')) {
        feedback.textContent += ' Sesi ujian telah berakhir.';
      }
      saveState();
    }
  }

  function formatElapsed(elapsedMs) {
    const totalSeconds = Math.floor(elapsedMs / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${String(seconds).padStart(2, '0')}`;
  }

  // ---- Drag and drop --------------------------------------------------------
  //
  // On every "dragover" we work out where the dragged <li> should land and
  // physically move it there immediately (a standard, well-tested vanilla-JS
  // pattern). By the time "drop" fires the element is already in place, so
  // drop just has to clean up styling.
  //
  // The dragged element is found live via `.dragging` on the DOM, not stored
  // in separate per-list JS state, so moving a line between the two lists
  // works exactly the same way as reordering within one of them.

  function onDragStart(e) {
    if (finished || sessionExpired) {
      e.preventDefault();
      return;
    }
    isDragging = true;
    e.target.classList.add('dragging');
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', e.target.dataset.id);
  }

  function onDragEnd(e) {
    isDragging = false;
    e.target.classList.remove('dragging');
    availableList.classList.remove('drag-over');
    solutionList.classList.remove('drag-over');
    // dragend always fires, whether the drag ended in a valid drop or was
    // cancelled -- and since dragover already physically relocates the
    // tile live as the drag happens, the DOM is already in its final,
    // settled arrangement by this point either way. So this is the one
    // reliable place to persist the arrangement after any drag gesture.
    saveState();
  }

  function getDragAfterElement(container, y) {
    const items = [...container.querySelectorAll('li:not(.dragging)')];
    return items.reduce(
      (closest, child) => {
        const box = child.getBoundingClientRect();
        const offset = y - box.top - box.height / 2;
        if (offset < 0 && offset > closest.offset) {
          return { offset, element: child };
        }
        return closest;
      },
      { offset: Number.NEGATIVE_INFINITY, element: null }
    ).element;
  }

  function setupDropZone(container) {
    container.addEventListener('dragover', (e) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
      container.classList.add('drag-over');

      const dragging = document.querySelector('.dragging');
      if (!dragging) return;

      const afterElement = getDragAfterElement(container, e.clientY);
      if (afterElement == null) {
        container.appendChild(dragging);
      } else {
        container.insertBefore(dragging, afterElement);
      }
    });

    container.addEventListener('dragleave', (e) => {
      if (e.target === container) {
        container.classList.remove('drag-over');
      }
    });

    container.addEventListener('drop', (e) => {
      e.preventDefault();
      container.classList.remove('drag-over');
      // The dragged element was already moved into place during dragover.
    });
  }

  setupDropZone(availableList);
  setupDropZone(solutionList);

  // ---- Finish / Try Again / Next Puzzle ----

  function clearFeedback() {
    feedback.textContent = '';
    feedback.className = 'feedback';
    hideExplanation();
    for (const li of [...availableList.children, ...solutionList.children]) {
      li.classList.remove('correct', 'incorrect');
    }
  }

  async function submitAttempt(submissionType = 'manual') {
    if (!currentPuzzleId || finished) return;

    finished = true;
    finishBtn.disabled = true;
    const elapsedMs = startTimeMs ? Math.max(0, Date.now() - startTimeMs) : 0;
    updateTimerLabel();

    const order = [...solutionList.children].map((li) => Number(li.dataset.id));

    let result;
    try {
      const res = await fetch(`/api/puzzles/${encodeURIComponent(currentPuzzleId)}/check`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ order, timeMs: elapsedMs, submissionType }),
      });
      if (!res.ok) throw new Error('Server returned an error');
      result = await res.json();
    } catch (err) {
      console.error('Check submission failed:', err);
      feedback.textContent = 'Terjadi kesalahan saat memeriksa jawaban Anda. Percobaan Anda belum dicatat.';
      feedback.className = 'feedback error';
      if (submissionType === 'timeout' || (timeLimitMinutes && elapsedMs >= timeLimitMinutes * 60 * 1000)) {
        sessionExpired = true;
        finishBtn.disabled = true;
        tryAgainBtn.disabled = true;
        nextPuzzleBtn.textContent = 'Kembali ke Menu Awal';
        nextPuzzleBtn.disabled = false;
        finished = true;
        stopTimer();
        saveState();
        return;
      }
      finishBtn.disabled = false;
      finished = false;
      startTimer();
      return;
    }

    for (const li of [...availableList.children, ...solutionList.children]) {
      li.classList.remove('correct', 'incorrect');
    }
    revealIndentation();
    for (const li of solutionList.children) {
      const id = Number(li.dataset.id);
      if (Object.prototype.hasOwnProperty.call(result.perLine, id)) {
        li.classList.add(result.perLine[id] ? 'correct' : 'incorrect');
      }
    }

    finishedElapsedMs = elapsedMs;
    finishedPerLine = result.perLine;
    finishedExplanation = result.explanation || null;
    showExplanation(finishedExplanation);

    const elapsedSeconds = (elapsedMs / 1000).toFixed(1);
    if (result.correct) {
      feedback.textContent = `Benar! Selesai dalam ${elapsedSeconds} detik. Hasil Anda telah dicatat.`;
      feedback.className = 'feedback success';
    } else {
      let message = `${result.correctPositions} dari ${result.total} baris berada di posisi yang benar. `;
      if (result.wrongLines > 0) {
        message += `${describeWrongLines(result.wrongLines, currentPuzzleData && currentPuzzleData.mode)} `;
      }
      if (submissionType === 'timeout') {
        message += `Batas waktu tercapai (${elapsedSeconds} detik). Hasil Anda telah dicatat.`;
      } else {
        message += `Waktu: ${elapsedSeconds} detik. Hasil Anda telah dicatat.`;
      }
      feedback.textContent = message;
      feedback.className = 'feedback error';
    }

    // Simpan hasil pengerjaan soal ini ke cache sebelum mengecek status seluruh soal
    saveCurrentPuzzleToCache(result.correct);
    saveState();

    if (submissionType === 'timeout' || sessionExpired || (timeLimitMinutes && elapsedMs >= timeLimitMinutes * 60 * 1000)) {
      await endResearchSet('timed_out');
      sessionExpired = true;
      stopTimer();
      finishBtn.disabled = true;
      tryAgainBtn.disabled = true;
      prevPuzzleBtn.disabled = true;
      nextPuzzleBtn.textContent = 'Kembali ke Menu Awal';
      nextPuzzleBtn.disabled = false;
      if (!feedback.textContent.includes('Sesi ujian telah berakhir')) {
        feedback.textContent += ' Sesi ujian telah berakhir.';
      }
    } else if (result.correct && areAllPuzzlesCompleted()) {
      await endResearchSet('completed');
      setCompleted = true;
      stopTimer();
      finishBtn.disabled = true;
      tryAgainBtn.disabled = true;
      prevPuzzleBtn.disabled = true;
      nextPuzzleBtn.textContent = 'Kembali ke Menu Awal';
      nextPuzzleBtn.disabled = false;
      const finalDisplay = formatElapsed(elapsedMs);
      timerLabel.textContent = timeLimitMinutes
        ? `${finalDisplay} / ${timeLimitMinutes}:00`
        : finalDisplay;
      timerLabel.classList.remove('timeout');
      feedback.textContent += ' Anda telah menyelesaikan semua soal dalam set ini!';
    } else {
      tryAgainBtn.disabled = false;
      updateNextPuzzleButtonState();
      updateNextPuzzleButtonLabel();
      prevPuzzleBtn.disabled = !hasPrevPuzzle();
      if (result.correct && !hasNextPuzzle() && !areAllPuzzlesCompleted()) {
        const completedCount = getCompletedPuzzlesCount();
        const totalCount = puzzleList ? puzzleList.length : 0;
        feedback.textContent += ` Anda telah menyelesaikan soal ini (${completedCount}/${totalCount} soal selesai). Masih ada soal yang belum Anda selesaikan, gunakan tombol Soal Sebelumnya jika ingin mengerjakannya.`;
      }
      // Continue session timer while participant reviews result
      startTimer();
    }

    saveCurrentPuzzleToCache(result.correct);
    saveState();

    if (reviewMode && result.attemptId && (result.correct || submissionType === 'timeout')) {
      showRatingDialog(result.attemptId);
    }
  }

  finishBtn.addEventListener('click', () => {
    if (!currentPuzzleId || finished || sessionExpired) return;
    submitAttempt('manual');
  });

  tryAgainBtn.addEventListener('click', () => {
    if (!currentPuzzleId || sessionExpired || setCompleted) return;
    if (timeLimitMinutes && startTimeMs && Date.now() - startTimeMs >= timeLimitMinutes * 60 * 1000) {
      handleSessionTimeout();
      return;
    }
    puzzleSessionCache.delete(currentPuzzleId);
    beginAttempt(currentPuzzleId, true);
  });

  prevPuzzleBtn.addEventListener('click', () => {
    if (prevPuzzleBtn.disabled) return;
    if (sessionExpired || setCompleted) return;

    if (timeLimitMinutes && startTimeMs && Date.now() - startTimeMs >= timeLimitMinutes * 60 * 1000) {
      handleSessionTimeout();
      return;
    }

    const prev = getPrevPuzzle();
    if (prev) {
      saveCurrentPuzzleToCache();
      saveState();
      beginAttempt(prev.id);
    }
  });

  nextPuzzleBtn.addEventListener('click', () => {
    if (nextPuzzleBtn.disabled) return;

    if (sessionExpired || setCompleted || (!hasNextPuzzle() && areAllPuzzlesCompleted())) {
      goToStartScreen();
      return;
    }

    if (timeLimitMinutes && startTimeMs && Date.now() - startTimeMs >= timeLimitMinutes * 60 * 1000) {
      handleSessionTimeout();
      return;
    }

    const next = getNextPuzzle();
    if (next) {
      saveCurrentPuzzleToCache();
      saveState();
      beginAttempt(next.id);
    } else {
      if (sessionExpired || setCompleted || areAllPuzzlesCompleted()) {
        goToStartScreen();
      }
    }
  });

  async function goToStartScreen() {
    stopTimer();
    finished = false;
    sessionExpired = false;
    setCompleted = false;
    isDragging = false;
    startTimeMs = null;
    currentPuzzleId = null;
    currentPuzzleData = null;
    pendingRatingAttemptId = null;
    pausedAtMs = null;
    puzzleSessionCache.clear();
    ratingModal.classList.add('hidden');
    clearFeedback();
    puzzleScreen.classList.add('hidden');
    startScreen.classList.remove('hidden');
    if (!pendingSetEndOutcome) startError.textContent = '';
    clearSavedState(); // leaving the puzzle screen deliberately means "start fresh" next time
    try {
      await fetch('/api/session/reset', { method: 'POST' });
    } catch (e) {
      // ignore network errors on reset
    }
  }

  // ---- Init ----

  async function init() {
    try {
      await loadConfig();
      await loadPuzzleSets();
    } catch (e) {
      startError.textContent = e.message || 'Gagal memuat konfigurasi aplikasi.';
      startBtn.disabled = true;
      return;
    }

    const saved = loadSavedState();
    if (researchSetOpen && (!saved || saved.setCompleted || saved.sessionExpired)) {
      pendingSetEndOutcome = saved?.sessionExpired ? 'timed_out' : saved?.setCompleted ? 'completed' : 'interrupted';
      startError.textContent = 'Penanda akhir set belum tersimpan. Coba lagi sebelum memulai set berikutnya.';
    }
    updateResearchControls();

    // If the server has no active session (e.g. server was restarted, or
    // the previous session's time limit already expired and was auto-reset
    // by loadConfig), any saved state in sessionStorage is stale -- the
    // server can't grade or continue anything from that session, so trying
    // to resume it would leave the UI stuck on a dead screen. Discard it
    // and let the player start fresh from the start screen.
    if (saved && !startTimeMs) {
      clearSavedState();
      return; // start screen is already visible by default
    }

    if (saved && saved.puzzleSet === activePuzzleSet && startTimeMs) {
      try {
        await loadPuzzleList();
      } catch (e) {
        clearSavedState();
        return;
      }
      const savedPuzzleStillExists = puzzleList.some((p) => p.id === saved.puzzleId);
      if (savedPuzzleStillExists) {
        resumeAttempt(saved);
      } else {
        // Stale state (e.g. the puzzle set changed since the last visit) --
        // don't try to resume something that no longer exists.
        clearSavedState();
      }
    } else if (saved) {
      // A state from another set, or a state whose server session no longer
      // exists, must never resurrect old progress on the home screen.
      clearSavedState();
    }
  }

  init();
})();
