(() => {
  // ---- Start screen elements ----
  const startScreen = document.getElementById('start-screen');
  const playerNameDisplay = document.getElementById('player-name-display');
  const puzzleSelect = document.getElementById('puzzle-select');
  const startBtn = document.getElementById('start-btn');
  const startError = document.getElementById('start-error');

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
  let timeLimitMinutes = null; // set once from /api/config if master_config.json contains the active set
  let startTimeMs = null; // session start time (persists across Try Again and Next Puzzle)
  let timerIntervalId = null;
  let finished = false;
  let sessionExpired = false; // true when the global session time limit has run out
  let setCompleted = false; // true when all puzzles in the set have been solved
  let finishedElapsedMs = null; // frozen elapsed time once Finish has been clicked
  let finishedPerLine = null; // correctness map from the last grading result, for restoring highlights
  let finishedExplanation = null; // explanation text that came back with the last grading result
  let reviewMode = false; // set once from /api/config; whether this server instance asks for ratings
  let pendingRatingAttemptId = null; // set while the rating dialog is open, for the attempt it's rating
  let pausedAtMs = null; // timestamp when exam timer was paused by the rating dialog

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
    const state = {
      puzzleId: currentPuzzleId,
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
    playerNameDisplay.textContent = playerName;
    reviewMode = Boolean(config.reviewMode);
    if (config.setConfig && config.setConfig.timeLimitMinutes) {
      timeLimitMinutes = Number(config.setConfig.timeLimitMinutes);
    }
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

  async function loadPuzzleList() {
    const res = await fetch('/api/puzzles');
    puzzleList = await res.json();

    puzzleSelect.innerHTML = '';

    // Group puzzles into <optgroup>s by their "type", preserving the order
    // types first appear in (server sorts puzzles by filename) rather than
    // alphabetizing, so puzzle authors control the order via filenames.
    const groups = new Map();
    for (const p of puzzleList) {
      const type = p.type || 'uncategorized';
      if (!groups.has(type)) groups.set(type, []);
      groups.get(type).push(p);
    }

    for (const [type, puzzlesOfType] of groups) {
      const optgroup = document.createElement('optgroup');
      optgroup.label = humanizeType(type);
      for (const p of puzzlesOfType) {
        const option = document.createElement('option');
        option.value = p.id;
        option.textContent = p.title;
        optgroup.appendChild(option);
      }
      puzzleSelect.appendChild(optgroup);
    }

    if (puzzleList.length === 0) {
      startError.textContent = 'Tidak ada soal ditemukan pada set soal ini.';
      startBtn.disabled = true;
    }
  }

  // "Next puzzle" walks puzzleList in the same flat order used to build the
  // dropdown (server's filename sort), ignoring the type groupings -- so it
  // still has a well-defined "next" even across a type boundary.
  function hasNextPuzzle() {
    const index = puzzleList.findIndex((p) => p.id === currentPuzzleId);
    return index !== -1 && index < puzzleList.length - 1;
  }

  function getNextPuzzle() {
    const index = puzzleList.findIndex((p) => p.id === currentPuzzleId);
    return index === -1 ? null : puzzleList[index + 1] || null;
  }

  // The label reflects what clicking the button will actually do. Both this
  // and the click handler call hasNextPuzzle()/getNextPuzzle() fresh rather
  // than caching a decision, so they can't drift out of sync with each
  // other -- nothing changes puzzleList or currentPuzzleId between an
  // attempt starting and the button being clicked, so recomputing is cheap
  // and keeps there being exactly one source of truth.
  function updateNextPuzzleButtonLabel() {
    nextPuzzleBtn.textContent = hasNextPuzzle() ? 'Soal Berikutnya' : 'Kembali ke Awal';
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

  function showRatingDialog(attemptId) {
    pendingRatingAttemptId = attemptId;
    ratingFeedback.textContent = feedback.textContent;
    ratingFeedback.className = feedback.className;
    ratingModal.classList.remove('hidden');

    // Freeze client-side timer while rating dialog is open (if not expired)
    if (!sessionExpired && !setCompleted) {
      stopTimer();
      pausedAtMs = Date.now();
    }
    saveState();
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
    if (!attemptId) return;
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
      // The rating is a nice-to-have on top of an attempt that's already
      // logged correctly either way -- don't trap the player in the
      // dialog over a network hiccup.
    }
    hideRatingDialog();
  }

  startBtn.addEventListener('click', async () => {
    if (!puzzleSelect.value) {
      startError.textContent = 'Tidak ada soal yang tersedia untuk dimulai.';
      return;
    }
    startError.textContent = '';

    // Starting fresh from the start screen resets any old session
    try {
      await fetch('/api/session/reset', { method: 'POST' });
    } catch (e) {}
    clearSavedState();
    startTimeMs = Date.now();
    sessionExpired = false;
    setCompleted = false;

    startScreen.classList.add('hidden');
    puzzleScreen.classList.remove('hidden');
    activePlayerLabel.textContent = `Peserta: ${playerName}`;

    await beginAttempt(puzzleSelect.value);
  });

  // Allow pressing Enter in the puzzle picker to start.
  puzzleSelect.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') startBtn.click();
  });

  // ---- Puzzle loading / attempt lifecycle ----

  // Starts an attempt at puzzleId.
  // Fetches a fresh copy of the puzzle from the server.
  // The global session timer keeps running continuously across attempts.
  async function beginAttempt(puzzleId) {
    currentPuzzleId = puzzleId;
    currentPuzzleData = null;
    finished = false;
    setCompleted = false;
    finishedElapsedMs = null;
    finishedPerLine = null;
    finishedExplanation = null;
    pendingRatingAttemptId = null;
    pausedAtMs = null;
    ratingModal.classList.add('hidden');
    finishBtn.disabled = sessionExpired;
    tryAgainBtn.disabled = sessionExpired;
    nextPuzzleBtn.disabled = true; // only enabled once Finish is clicked for this attempt
    updateNextPuzzleButtonLabel();
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

    // Ensure session timer is initialized if not yet started
    if (!startTimeMs) {
      startTimeMs = Date.now();
      sessionExpired = false;
    }

    if (!sessionExpired) {
      startTimer();
    } else {
      updateTimerLabel();
    }
    saveState();
  }

  // Reconstructs the screen from a previously saved state (page refresh),
  // without any request to /api/puzzles/:id -- using the exact line data
  // and tile arrangement that was saved means the player sees precisely
  // what they left, not a freshly (and differently) shuffled puzzle.
  function resumeAttempt(saved) {
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
      saved.puzzle.type && saved.puzzle.type !== 'uncategorized' ? humanizeType(saved.puzzle.type) : '';
    applyMode(saved.puzzle.mode); // state saved before modes existed has no mode; that's just "arrange"
    puzzleSelect.value = saved.puzzleId;

    renderDescription(saved.puzzle.description);
    renderTiles(saved.puzzle.lines, saved.availableOrder, saved.solutionOrder);

    updateNextPuzzleButtonLabel();

    if (finished) {
      finishBtn.disabled = true;
      tryAgainBtn.disabled = sessionExpired || setCompleted;
      nextPuzzleBtn.disabled = false;
      if (sessionExpired || setCompleted || !hasNextPuzzle()) {
        nextPuzzleBtn.textContent = 'Kembali ke Awal';
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
        nextPuzzleBtn.disabled = true;
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
    nextPuzzleBtn.textContent = 'Kembali ke Awal';
    nextPuzzleBtn.disabled = false;

    if (!finished && currentPuzzleId) {
      await submitAttempt('timeout');
    } else {
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
    e.target.classList.add('dragging');
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', e.target.dataset.id);
  }

  function onDragEnd(e) {
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
        nextPuzzleBtn.textContent = 'Kembali ke Awal';
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

    if (submissionType === 'timeout' || sessionExpired || (timeLimitMinutes && elapsedMs >= timeLimitMinutes * 60 * 1000)) {
      sessionExpired = true;
      stopTimer();
      finishBtn.disabled = true;
      tryAgainBtn.disabled = true;
      nextPuzzleBtn.textContent = 'Kembali ke Awal';
      nextPuzzleBtn.disabled = false;
      if (!feedback.textContent.includes('Sesi ujian telah berakhir')) {
        feedback.textContent += ' Sesi ujian telah berakhir.';
      }
    } else if (result.correct && !hasNextPuzzle()) {
      setCompleted = true;
      stopTimer();
      finishBtn.disabled = true;
      tryAgainBtn.disabled = true;
      nextPuzzleBtn.textContent = 'Kembali ke Awal';
      nextPuzzleBtn.disabled = false;
      const finalDisplay = formatElapsed(elapsedMs);
      timerLabel.textContent = timeLimitMinutes
        ? `${finalDisplay} / ${timeLimitMinutes}:00`
        : finalDisplay;
      timerLabel.classList.remove('timeout');
      feedback.textContent += ' Anda telah menyelesaikan semua soal dalam set ini!';
    } else {
      tryAgainBtn.disabled = false;
      nextPuzzleBtn.disabled = false;
      // Continue session timer while participant reviews result
      startTimer();
    }

    saveState();

    if (reviewMode && result.attemptId) {
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
    beginAttempt(currentPuzzleId);
  });

  nextPuzzleBtn.addEventListener('click', () => {
    if (nextPuzzleBtn.disabled) return;

    if (sessionExpired || setCompleted || !hasNextPuzzle()) {
      goToStartScreen();
      return;
    }

    if (timeLimitMinutes && startTimeMs && Date.now() - startTimeMs >= timeLimitMinutes * 60 * 1000) {
      handleSessionTimeout();
      return;
    }

    const next = getNextPuzzle();
    if (next) {
      puzzleSelect.value = next.id; // keep the picker in sync for if the player returns to it later
      beginAttempt(next.id);
    } else {
      goToStartScreen();
    }
  });

  async function goToStartScreen() {
    stopTimer();
    finished = false;
    sessionExpired = false;
    setCompleted = false;
    startTimeMs = null;
    currentPuzzleId = null;
    currentPuzzleData = null;
    pendingRatingAttemptId = null;
    pausedAtMs = null;
    ratingModal.classList.add('hidden');
    clearFeedback();
    puzzleScreen.classList.add('hidden');
    startScreen.classList.remove('hidden');
    startError.textContent = '';
    clearSavedState(); // leaving the puzzle screen deliberately means "start fresh" next time
    try {
      await fetch('/api/session/reset', { method: 'POST' });
    } catch (e) {
      // ignore network errors on reset
    }
  }

  // ---- Init ----

  async function init() {
    await loadConfig();
    await loadPuzzleList();

    const saved = loadSavedState();

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

    const savedPuzzleStillExists = saved && puzzleList.some((p) => p.id === saved.puzzleId);
    if (savedPuzzleStillExists) {
      resumeAttempt(saved);
    } else if (saved) {
      // Stale state (e.g. the puzzle set changed since the last visit) --
      // don't try to resume something that no longer exists.
      clearSavedState();
    }
  }

  init();
})();
