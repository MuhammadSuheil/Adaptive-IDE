'use strict';

const http = require('http');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { URL } = require('url');

const PORT = process.env.PORT || 3000;
const PUBLIC_DIR = path.join(__dirname, 'public');
const PUZZLES_ROOT = path.join(__dirname, 'puzzles');

// ---- Command-line arguments --------------------------------------------
//
// node server.js <player-name> <puzzle-set-subfolder> [--review]
//
// <player-name>          Locked in for the whole session -- the browser UI
//                         displays it but never lets the player change it.
// <puzzle-set-subfolder> A folder path relative to puzzles/ (use "." for
//                         puzzles/ itself). Only *.json files inside that
//                         folder are loaded; puzzles elsewhere under
//                         puzzles/ are not reachable by this server
//                         instance. This is meant for running one server
//                         per player/quiz-set, e.g.
//                         `node server.js "Alice" exam-set-a`.
// --review                Turns on review mode: after Finish, the player is
//                         asked to rate the puzzle's difficulty 1 (very
//                         easy) to 10 (very hard), and that rating is
//                         recorded in the attempt log. The exam timer is
//                         paused while the rating dialog is open. Can appear
//                         anywhere on the command line. Without it, every
//                         logged attempt's rating is "-".

const rawArgs = process.argv.slice(2);
const REVIEW_MODE = rawArgs.includes('--review');
const [cliPlayerName, cliPuzzleSubfolder] = rawArgs.filter((arg) => arg !== '--review');

if (!cliPlayerName || !cliPuzzleSubfolder) {
  console.error('Usage: node server.js <player-name> <puzzle-set-subfolder> [--review]');
  console.error('');
  console.error('  <player-name>            Name recorded in the attempt log; locked for the session.');
  console.error('  <puzzle-set-subfolder>   Folder inside puzzles/ whose *.json files will be served.');
  console.error('                           Use "." to serve puzzles/ itself.');
  console.error('  --review                 Ask the player to rate each puzzle\'s difficulty 1-10 after Finish, and');
  console.error('                           record that rating in the attempt log.');
  console.error('');
  console.error('Example: node server.js "Ada Lovelace" exam-set-a --review');
  process.exit(1);
}

const PLAYER_NAME = cliPlayerName;

const PUZZLES_DIR = path.resolve(PUZZLES_ROOT, cliPuzzleSubfolder);
const ACTIVE_PUZZLE_SET_KEY = path.relative(PUZZLES_ROOT, PUZZLES_DIR);
const isInsidePuzzlesRoot =
  PUZZLES_DIR === PUZZLES_ROOT || PUZZLES_DIR.startsWith(PUZZLES_ROOT + path.sep);
if (!isInsidePuzzlesRoot) {
  console.error(`Puzzle set subfolder must resolve to a path inside ${PUZZLES_ROOT}`);
  console.error(`Got: ${cliPuzzleSubfolder} -> ${PUZZLES_DIR}`);
  process.exit(1);
}
if (!fs.existsSync(PUZZLES_DIR) || !fs.statSync(PUZZLES_DIR).isDirectory()) {
  console.error(`Puzzle set folder not found: ${PUZZLES_DIR}`);
  process.exit(1);
}

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
};

// ---- Puzzle loading -------------------------------------------------------
//
// Puzzles are read fresh on every request (there are only ever a handful of
// small JSON files, so this costs nothing) which means you can add or edit
// files in puzzles/ without restarting the server.

function loadAllPuzzles() {
  const files = fs
    .readdirSync(PUZZLES_DIR)
    .filter((f) => f.endsWith('.json') && f !== 'master_config.json')
    .sort();
  return files.map((f) => normalizePuzzle(JSON.parse(fs.readFileSync(path.join(PUZZLES_DIR, f), 'utf8'))));
}

function loadActiveSetConfig() {
  const masterConfigPath = path.join(PUZZLES_ROOT, 'master_config.json');
  if (!fs.existsSync(masterConfigPath)) return {};
  try {
    const masterConfig = JSON.parse(fs.readFileSync(masterConfigPath, 'utf8'));
    return masterConfig[ACTIVE_PUZZLE_SET_KEY] || {};
  } catch (e) {
    console.error('Failed to parse master_config.json:', e);
    return {};
  }
}

function getActiveSetTimeLimitMs() {
  const setConfig = loadActiveSetConfig();
  const timeLimitMinutes = Number(setConfig.timeLimitMinutes);
  if (!Number.isFinite(timeLimitMinutes) || timeLimitMinutes <= 0) return null;
  return timeLimitMinutes * 60 * 1000;
}

// Puzzles are authored with a "solutions" field: an array of possible
// correct orderings (each itself an array of line ids), so a puzzle can
// accept more than one valid answer. For backwards compatibility, a file
// still written with the older singular "solution": [ids] field is
// upgraded in memory to "solutions": [[ids]].
function normalizePuzzle(puzzle) {
  if (!Array.isArray(puzzle.solutions)) {
    puzzle.solutions = Array.isArray(puzzle.solution) ? [puzzle.solution] : [];
  }
  // "type" is a free-form category label (e.g. "loops", "basics") used to
  // group puzzles in the picker. Older puzzle files without one still load
  // fine -- they just land in a generic "uncategorized" group.
  if (typeof puzzle.type !== 'string' || puzzle.type.trim() === '') {
    puzzle.type = 'uncategorized';
  }
  // "mode" says what kind of task a puzzle is, and is separate from "type"
  // (the topic category) so the two can vary independently:
  //   "arrange" - rebuild the source code from its lines (the default)
  //   "output"  - the description holds code; arrange what it prints
  //   "debug"   - the description holds buggy code; rebuild the corrected
  //               version from the tiles, some of which are plausible but
  //               still buggy distractors
  // The server grades every mode identically (against "solutions", with any
  // line not in a solution counting as a distractor) -- mode only changes
  // what the UI tells the player to do.
  if (typeof puzzle.mode !== 'string' || puzzle.mode.trim() === '') {
    puzzle.mode = 'arrange';
  }
  return puzzle;
}

function findPuzzleById(id) {
  return loadAllPuzzles().find((p) => p.id === id) || null;
}

function shuffle(array) {
  const result = array.slice();
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

// Only leading whitespace is stripped for display -- spacing in the middle
// of a line (e.g. around operators) is left untouched.
function stripLeadingWhitespace(code) {
  return code.replace(/^[ \t]+/, '');
}

// Some lines become visually identical once indentation is stripped for
// display -- most commonly bare closing braces like "}". If the player has
// no way to tell those tiles apart, the puzzle would require guessing which
// physical tile the grader expects in a given slot, even for someone who
// fully understands the code. To keep it solvable, lines that collide after
// stripping get a small "#n" tag reflecting their ACTUAL relative order in
// the correct solution (#1 really does belong before #2) -- so the tag is
// checkable, not just a random label. This has to happen server-side, since
// only the server holds the solution. It only reveals relative order
// *within* a group of identical-looking lines, never each line's absolute
// position or indentation depth.
//
// The first entry in puzzle.solutions is used as the reference ordering
// (fine even when a puzzle accepts multiple solutions, since alternates are
// meant for genuinely interchangeable lines -- e.g. swapping two variable
// declarations -- which almost never share identical stripped text with
// something else). A line that isn't part of that reference solution at all
// (a pure distractor) sorts after every line that is, tie-broken by id, so
// its tag carries no positional meaning.
function computeDisambiguationTags(puzzle) {
  const referenceSolution = puzzle.solutions[0] || [];
  const positionById = new Map();
  referenceSolution.forEach((id, index) => positionById.set(id, index));

  const groupsByStrippedText = new Map();
  for (const line of puzzle.lines) {
    const stripped = stripLeadingWhitespace(line.code);
    if (!groupsByStrippedText.has(stripped)) {
      groupsByStrippedText.set(stripped, []);
    }
    groupsByStrippedText.get(stripped).push(line);
  }

  const tagById = new Map();
  for (const group of groupsByStrippedText.values()) {
    if (group.length < 2) continue; // no collision, no tag needed
    const sorted = [...group].sort((a, b) => {
      const posA = positionById.has(a.id) ? positionById.get(a.id) : Infinity;
      const posB = positionById.has(b.id) ? positionById.get(b.id) : Infinity;
      return posA !== posB ? posA - posB : a.id - b.id;
    });
    sorted.forEach((line, index) => tagById.set(line.id, index + 1));
  }
  return tagById;
}

// Strip the solutions and shuffle line order before sending a puzzle to the
// browser, so opening dev tools doesn't just hand over the answer. Each
// line also gets its disambiguation tag (see computeDisambiguationTags),
// which is the one deliberate, minimal exception to that.
function toClientPuzzle(puzzle) {
  const tagById = computeDisambiguationTags(puzzle);
  return {
    id: puzzle.id,
    title: puzzle.title,
    type: puzzle.type,
    mode: puzzle.mode,
    description: puzzle.description,
    lines: shuffle(puzzle.lines).map(({ id, code }) => ({
      id,
      code,
      tag: tagById.has(id) ? tagById.get(id) : null,
    })),
  };
}

function gradeAgainst(expected, current) {
  const perLine = {};
  let correctPositions = 0;
  const checkLength = Math.min(current.length, expected.length);
  for (let i = 0; i < checkLength; i++) {
    const isCorrect = current[i] === expected[i];
    perLine[current[i]] = isCorrect;
    if (isCorrect) correctPositions++;
  }
  for (let i = checkLength; i < current.length; i++) {
    perLine[current[i]] = false;
  }

  const isFullyCorrect = current.length === expected.length && current.every((id, i) => id === expected[i]);

  return {
    correct: isFullyCorrect,
    correctPositions,
    total: expected.length,
    submittedCount: current.length,
    perLine,
  };
}

// A puzzle may accept more than one correct ordering (puzzle.solutions is
// an array of arrays). The submission only needs to match ONE of them. When
// there's no exact match, we still show the learner feedback based on
// whichever accepted solution they came closest to, so partial-credit
// messaging stays meaningful instead of just picking the first one.
function gradeSubmission(puzzle, order) {
  const current = Array.isArray(order) ? order.map(Number).filter((n) => Number.isFinite(n)) : [];
  const solutions = puzzle.solutions.length > 0 ? puzzle.solutions : [[]];

  let chosen = null;
  for (const expected of solutions) {
    const result = gradeAgainst(expected, current);
    if (result.correct) {
      chosen = result; // exact match against at least one accepted solution
      break;
    }
    if (!chosen || result.correctPositions > chosen.correctPositions) {
      chosen = result;
    }
  }

  // Lines the player used that appear in NO accepted solution -- i.e.
  // distractors. In a debug puzzle these are the plausible-but-buggy tiles,
  // so telling the player how many they used makes a wrong attempt
  // informative ("you kept a buggy line") rather than just "some lines are
  // in the wrong place".
  const acceptedIds = new Set(solutions.flat());
  chosen.wrongLines = current.filter((id) => !acceptedIds.has(id)).length;

  // An optional write-up of what the puzzle was really testing (e.g. what
  // the bug was). It only ever leaves the server here, in the grading
  // response, so it can't be read out of the puzzle payload beforehand.
  chosen.explanation = typeof puzzle.explanation === 'string' && puzzle.explanation.trim() !== '' ? puzzle.explanation : null;

  return chosen;
}

// ---- Attempt logging --------------------------------------------------
//
// Every time a learner clicks "Finish" (or the session time limit expires),
// the server (not the browser) grades the attempt and appends an identical
// row to BOTH logs/attempts.csv (master log) and logs/<puzzle-set>.csv.
// Doing the logging server-side, using the server's own grading result,
// means the log can't be spoofed by editing client-side JavaScript.
//
// Each row gets a server-generated attempt_id, and a rating column that is
// "-" until (in review mode) the player rates the puzzle's difficulty --
// at which point that same row is updated in place in both files.

const LOG_DIR = path.join(__dirname, 'logs');
const LOG_FILE = path.join(LOG_DIR, 'attempts.csv');
const LOG_LOCK_FILE = path.join(LOG_DIR, 'attempts.csv.lock');
const LOG_BACKUP_DIR = path.join(LOG_DIR, 'legacy-backup');
const sanitizedSetName = (cliPuzzleSubfolder === '.' ? 'root' : cliPuzzleSubfolder || 'default').replace(/[\\/:*?"<>|]/g, '_');
const SET_LOG_FILE = path.join(LOG_DIR, `${sanitizedSetName}.csv`);
// The files an attempt is written to / a rating is updated in. De-duplicated
// in case the active set's log name happens to collide with attempts.csv.
const ACTIVE_LOG_FILES = [...new Set([LOG_FILE, SET_LOG_FILE])];
fs.mkdirSync(LOG_DIR, { recursive: true }); // safe to call from multiple processes at once (mkdir -p semantics)

const LOG_COLUMNS = [
  'timestamp',
  'attempt_id',
  'name',
  'puzzle_set',
  'puzzle_id',
  'puzzle_title',
  'time_seconds',
  'submission_type',
  'correct',
  'correct_lines',
  'total_lines',
  'score_percent',
  'rating',
];
const LOG_HEADER = LOG_COLUMNS.join(',') + '\n';
// The header written by versions of this server from before attempt_id and
// rating existed. Files with exactly this header get upgraded in place.
const LEGACY_LOG_HEADER =
  'timestamp,name,puzzle_set,puzzle_id,puzzle_title,time_seconds,submission_type,correct,correct_lines,total_lines,score_percent';

// ---- Session timer state ----------------------------------------------
//
// sessionStartedAtMs / sessionDeadlineMs drive the global exam time limit.
// sessionPausedAtMs is set only while the review-mode rating dialog is
// open: the clock is frozen then, and on resume both the start and the
// deadline are shifted forward by exactly how long it was frozen, so time
// spent rating never counts against the exam.
let sessionStartedAtMs = null;
let sessionDeadlineMs = null;
let sessionPausedAtMs = null;

function resetSession() {
  sessionStartedAtMs = null;
  sessionDeadlineMs = null;
  sessionPausedAtMs = null;
}

// Freezes the session clock. Does nothing if there's no active session, it
// is already frozen, or the time limit has already passed (an expired
// session must stay expired -- pausing then resuming must never revive it).
function pauseSession(nowMs) {
  if (sessionStartedAtMs === null || sessionPausedAtMs !== null) return false;
  if (sessionDeadlineMs !== null && nowMs >= sessionDeadlineMs) return false;
  sessionPausedAtMs = nowMs;
  return true;
}

// Unfreezes the session clock, shifting start and deadline forward by the
// frozen duration. Safe to call when not paused (no-op).
function resumeSession(nowMs) {
  if (sessionPausedAtMs === null) return false;
  const pausedForMs = Math.max(0, nowMs - sessionPausedAtMs);
  sessionPausedAtMs = null;
  if (sessionStartedAtMs !== null) {
    sessionStartedAtMs += pausedForMs;
    if (sessionDeadlineMs !== null) sessionDeadlineMs += pausedForMs;
  }
  return true;
}

// ---- Cross-process locking for the shared log files -------------------
//
// The log files can be shared by several server.js processes at once -- e.g.
// one instance per student, all pointed at the same project directory. A
// single process is safe on its own: every fs call here is synchronous, and
// Node is single-threaded. But nothing stops two SEPARATE processes from
// touching the files at the same instant, and these operations are
// genuinely unsafe if that happens:
//   - creating a file with its header (a check-then-write race)
//   - upgrading a legacy file's header
//   - updateAttemptRating's read-the-whole-file-then-write-it-back cycle,
//     where one process's rewrite can silently clobber a row another
//     process appended (or re-rated) in between
// So every operation that touches a log file runs under one lock.
//
// The lock is a plain file created with the 'wx' flag, which fails with
// EEXIST if the file already exists -- an atomic create-if-absent usable as
// a cross-process mutex without any dependency. A lock older than
// LOCK_STALE_MS is assumed to belong to a crashed process and is stolen
// rather than waited on forever.

const LOCK_STALE_MS = 8000;
const LOCK_TIMEOUT_MS = 5000;
const LOCK_RETRY_MS = 20;

function acquireLogLock() {
  const start = Date.now();
  while (true) {
    try {
      const fd = fs.openSync(LOG_LOCK_FILE, 'wx');
      fs.writeSync(fd, String(process.pid));
      fs.closeSync(fd);
      return;
    } catch (err) {
      if (err.code !== 'EEXIST') throw err;
      try {
        if (Date.now() - fs.statSync(LOG_LOCK_FILE).mtimeMs > LOCK_STALE_MS) {
          fs.unlinkSync(LOG_LOCK_FILE); // previous holder likely crashed while holding it; steal it
          continue;
        }
      } catch (statErr) {
        continue; // lock file vanished between our open() and stat() -- just retry
      }
      if (Date.now() - start > LOCK_TIMEOUT_MS) {
        throw new Error(`Timed out waiting for the attempt log lock (held for over ${LOCK_TIMEOUT_MS}ms)`);
      }
      // A brief, genuinely synchronous sleep -- setTimeout can't be awaited
      // here without turning every caller async, and contention is expected
      // to be rare and measured in milliseconds.
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, LOCK_RETRY_MS);
    }
  }
}

function releaseLogLock() {
  try {
    fs.unlinkSync(LOG_LOCK_FILE);
  } catch (err) {
    // Already gone (e.g. stolen as stale by another process) -- fine.
  }
}

// Runs fn() with the log lock held, guaranteeing it can't interleave with
// any other process's log operation.
function withLogLock(fn) {
  acquireLogLock();
  try {
    return fn();
  } finally {
    releaseLogLock();
  }
}

function csvField(value) {
  const str = String(value);
  if (/[",\n]/.test(str)) {
    return '"' + str.replace(/"/g, '""') + '"';
  }
  return str;
}

function csvRow(values) {
  return values.map(csvField).join(',') + '\n';
}

// Parses one CSV line back into raw field values, respecting the quoting
// csvField() produces (quoted fields, doubled "" for an embedded quote).
// Needed to reliably find a specific row by attempt_id later, since a plain
// split(',') would break on any field -- like a puzzle title -- that
// contains a comma and is therefore quoted.
function parseCsvLine(line) {
  const fields = [];
  let current = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (inQuotes) {
      if (c === '"') {
        if (line[i + 1] === '"') {
          current += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        current += c;
      }
    } else if (c === '"') {
      inQuotes = true;
    } else if (c === ',') {
      fields.push(current);
      current = '';
    } else {
      current += c;
    }
  }
  fields.push(current);
  return fields;
}

// Upgrades a log file written before attempt_id/rating existed: every old
// row gets an attempt_id and rating "-". The id is derived from the row's
// own content, so the same legacy attempt gets the SAME id in attempts.csv
// and in its set log. A copy of the original is kept in logs/legacy-backup/
// first. Returns true if the file was upgraded. Must run under the lock.
function upgradeLegacyLogFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  const lines = content.split(/\r?\n/);
  if (lines.length > 0 && lines[lines.length - 1] === '') lines.pop();
  if (lines.length === 0 || lines[0].trim() !== LEGACY_LOG_HEADER) return false;

  fs.mkdirSync(LOG_BACKUP_DIR, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  fs.copyFileSync(filePath, path.join(LOG_BACKUP_DIR, `${path.basename(filePath, '.csv')}.${stamp}.pre-migration.csv`));

  const out = [LOG_COLUMNS.join(',')];
  for (let i = 1; i < lines.length; i++) {
    if (lines[i].trim() === '') continue;
    const f = parseCsvLine(lines[i]);
    if (f.length !== 11) {
      out.push(lines[i]); // unexpected shape: leave untouched rather than guess
      continue;
    }
    const h = crypto.createHash('sha1').update(lines[i]).digest('hex');
    const legacyId = `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20, 32)}`;
    out.push(csvRow([f[0], legacyId, ...f.slice(1), '-']).trimEnd());
  }
  fs.writeFileSync(filePath, out.join('\n') + '\n');
  return true;
}

// Makes sure filePath exists with the current header, upgrading it first if
// it still has the legacy one. Throws if it has some other unknown header,
// so rows are never appended to a file whose columns they don't match.
// Must run under the lock.
function ensureLogFile(filePath) {
  if (!fs.existsSync(filePath)) {
    fs.writeFileSync(filePath, LOG_HEADER);
    return;
  }
  const firstLine = fs.readFileSync(filePath, 'utf8').split(/\r?\n/)[0].trim();
  if (firstLine === '') {
    // An empty file (e.g. just created and never written to): start it properly.
    fs.writeFileSync(filePath, LOG_HEADER);
    return;
  }
  if (firstLine === LOG_COLUMNS.join(',')) return;
  if (firstLine === LEGACY_LOG_HEADER) {
    upgradeLegacyLogFile(filePath);
    return;
  }
  throw new Error(`Unrecognised log header in ${filePath}; refusing to append to it.`);
}

// At startup, upgrade every legacy-format CSV in logs/ (including set logs
// for sets this instance isn't serving), so the whole folder is consistent.
function upgradeAllLegacyLogs() {
  withLogLock(() => {
    for (const f of fs.readdirSync(LOG_DIR)) {
      if (!f.endsWith('.csv')) continue;
      const full = path.join(LOG_DIR, f);
      if (!fs.statSync(full).isFile()) continue;
      try {
        if (upgradeLegacyLogFile(full)) console.log(`  Upgraded legacy log to the new format: logs/${f}`);
      } catch (err) {
        console.error(`Could not upgrade logs/${f}:`, err);
      }
    }
  });
}
try {
  upgradeAllLegacyLogs();
} catch (err) {
  console.error('Legacy log upgrade skipped:', err);
}

// Logs an attempt immediately, with rating defaulted to "-". This always
// happens at Finish/timeout regardless of review mode, so every attempt is
// guaranteed exactly one row in each log -- a rating given later (see
// updateAttemptRating) just enriches those same rows rather than being a
// separate write. Returns the attempt's id so the caller can hand it back to
// the client for that enrichment step.
function logAttempt({ name, puzzleSet, puzzleId, puzzleTitle, timeSeconds, submissionType, correct, correctPositions, total }) {
  return withLogLock(() => {
    const scorePercent = total > 0 ? ((correctPositions / total) * 100).toFixed(1) : '0.0';
    const attemptId = crypto.randomUUID();
    const row = csvRow([
      new Date().toISOString(),
      attemptId,
      name,
      puzzleSet,
      puzzleId,
      puzzleTitle,
      timeSeconds.toFixed(2),
      submissionType || 'manual',
      correct ? 'true' : 'false',
      correctPositions,
      total,
      scorePercent,
      '-',
    ]);
    // Write the identical row to both the master attempts log and the set-specific log
    for (const file of ACTIVE_LOG_FILES) {
      ensureLogFile(file);
      fs.appendFileSync(file, row);
    }
    return attemptId;
  });
}

// Finds the row with this attempt_id in each active log file and rewrites
// its rating field in place; every other row is left byte-for-byte as it
// was. Returns false (and changes nothing) if the id isn't found in any
// file -- e.g. a log file was rotated or edited between Finish and the
// player submitting a rating.
function updateAttemptRating(attemptId, rating) {
  return withLogLock(() => {
    let updatedAny = false;
    for (const file of ACTIVE_LOG_FILES) {
      if (!fs.existsSync(file)) continue;
      const content = fs.readFileSync(file, 'utf8');
      const lines = content.split('\n');
      if (lines.length > 0 && lines[lines.length - 1] === '') lines.pop(); // trailing newline artifact
      if (lines.length < 2) continue;

      const header = parseCsvLine(lines[0].replace(/\r$/, ''));
      const idIndex = header.indexOf('attempt_id');
      const ratingIndex = header.indexOf('rating');
      if (idIndex === -1 || ratingIndex === -1) continue;

      for (let i = 1; i < lines.length; i++) {
        const fields = parseCsvLine(lines[i].replace(/\r$/, ''));
        if (fields[idIndex] === attemptId) {
          fields[ratingIndex] = String(rating);
          lines[i] = csvRow(fields).trimEnd();
          fs.writeFileSync(file, lines.join('\n') + '\n');
          updatedAny = true;
          break;
        }
      }
    }
    return updatedAny;
  });
}

// ---- HTTP helpers -----------------------------------------------------

function sendJson(res, statusCode, data) {
  const body = JSON.stringify(data);
  res.writeHead(statusCode, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(body),
  });
  res.end(body);
}

function sendFile(res, filePath) {
  fs.readFile(filePath, (err, content) => {
    if (err) {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('Not found');
      return;
    }
    const ext = path.extname(filePath);
    res.writeHead(200, {
      'Content-Type': MIME_TYPES[ext] || 'application/octet-stream',
      'Cache-Control': 'no-cache, no-store, must-revalidate',
      'Pragma': 'no-cache',
      'Expires': '0',
    });
    res.end(content);
  });
}

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', (chunk) => {
      data += chunk;
      if (data.length > 1e6) {
        req.destroy();
        reject(new Error('Payload too large'));
      }
    });
    req.on('end', () => {
      if (!data) return resolve({});
      try {
        resolve(JSON.parse(data));
      } catch (e) {
        reject(e);
      }
    });
    req.on('error', reject);
  });
}

// ---- Router ---------------------------------------------------------------

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  const pathname = decodeURIComponent(url.pathname);

  try {
    // GET /api/config  -> the locked player name, active puzzle set, review mode, and session state
    if (req.method === 'GET' && pathname === '/api/config') {
      const setConfig = loadActiveSetConfig();
      return sendJson(res, 200, {
        playerName: PLAYER_NAME,
        puzzleSet: cliPuzzleSubfolder,
        setConfig,
        reviewMode: REVIEW_MODE,
        sessionStartedAtMs,
        sessionDeadlineMs,
        sessionPausedAtMs,
      });
    }

    // POST /api/session/reset  -> reset the active session timer (e.g. going back to start)
    if (req.method === 'POST' && pathname === '/api/session/reset') {
      resetSession();
      return sendJson(res, 200, { ok: true });
    }

    // GET /api/puzzles  -> [{id, title, type, description}, ...]
    if (req.method === 'GET' && pathname === '/api/puzzles') {
      const list = loadAllPuzzles().map(({ id, title, type, mode, description }) => ({ id, title, type, mode, description }));
      return sendJson(res, 200, list);
    }

    // POST /api/puzzles/:id/check  -> grade a submitted line order
    let match = pathname.match(/^\/api\/puzzles\/([^/]+)\/check$/);
    if (req.method === 'POST' && match) {
      const puzzle = findPuzzleById(match[1]);
      if (!puzzle) return sendJson(res, 404, { error: 'Puzzle not found' });
      let body;
      try {
        body = await readJsonBody(req);
      } catch (e) {
        return sendJson(res, 400, { error: 'Invalid JSON body' });
      }
      // A check arriving while the clock is still frozen (e.g. the rating
      // request never got through) means the player has moved on: unfreeze
      // before measuring elapsed time, so it excludes the frozen period.
      resumeSession(Date.now());
      const nowMs = Date.now();
      const result = gradeSubmission(puzzle, body.order);

      const timeLimitMs = getActiveSetTimeLimitMs();
      let timeSeconds;
      let isTimeout = body.submissionType === 'timeout';

      if (sessionStartedAtMs !== null) {
        const elapsedMs = Math.max(0, nowMs - sessionStartedAtMs);
        if (sessionDeadlineMs !== null && nowMs >= sessionDeadlineMs) {
          isTimeout = true;
        }
        timeSeconds = isTimeout && timeLimitMs ? timeLimitMs / 1000 : elapsedMs / 1000;
      } else {
        const timeMs = Number(body.timeMs);
        timeSeconds = Number.isFinite(timeMs) && timeMs >= 0 ? timeMs / 1000 : 0;
      }
      const submissionType = isTimeout ? 'timeout' : 'manual';

      try {
        const attemptId = logAttempt({
          name: PLAYER_NAME,
          puzzleSet: cliPuzzleSubfolder,
          puzzleId: puzzle.id,
          puzzleTitle: puzzle.title,
          timeSeconds,
          submissionType,
          correct: result.correct,
          correctPositions: result.correctPositions,
          total: result.total,
        });
        // Only handed back when the server is running in review mode -- it's
        // how the client later identifies which row to attach a rating to. A
        // non-review server never exposes it, so a rating can never reach the
        // log for that kind of session even if someone tried calling the
        // rating endpoint directly (that endpoint also checks REVIEW_MODE).
        if (REVIEW_MODE) {
          result.attemptId = attemptId;
          // The rating dialog is about to open: freeze the exam clock while
          // it is. (No-op if the session already expired, so a timeout
          // submission is never revived by pausing.)
          if (!isTimeout) pauseSession(nowMs);
        }
      } catch (logErr) {
        // Logging failures shouldn't break grading for the learner.
        console.error('Failed to write attempt log:', logErr);
      }

      return sendJson(res, 200, result);
    }

    // POST /api/attempts/:attemptId/rating  -> attach a 1-10 rating to an
    // already-logged attempt (review mode only) and unfreeze the exam clock
    match = pathname.match(/^\/api\/attempts\/([^/]+)\/rating$/);
    if (req.method === 'POST' && match) {
      if (!REVIEW_MODE) {
        return sendJson(res, 403, { error: 'This server is not running in review mode.' });
      }
      let body;
      try {
        body = await readJsonBody(req);
      } catch (e) {
        return sendJson(res, 400, { error: 'Invalid JSON body' });
      }
      const rating = Number(body.rating);
      if (!Number.isInteger(rating) || rating < 1 || rating > 10) {
        return sendJson(res, 400, { error: 'rating must be an integer from 1 to 10' });
      }
      // The dialog is closing either way, so unfreeze the clock first and hand
      // the (possibly shifted) session start back so the client can resync.
      resumeSession(Date.now());
      const updated = updateAttemptRating(match[1], rating);
      if (!updated) {
        return sendJson(res, 404, { error: 'Attempt not found.', sessionStartedAtMs, sessionDeadlineMs });
      }
      return sendJson(res, 200, { ok: true, sessionStartedAtMs, sessionDeadlineMs });
    }

    // GET /api/puzzles/:id  -> puzzle detail (solution stripped, lines shuffled)
    match = pathname.match(/^\/api\/puzzles\/([^/]+)$/);
    if (req.method === 'GET' && match) {
      const puzzle = findPuzzleById(match[1]);
      if (!puzzle) return sendJson(res, 404, { error: 'Puzzle not found' });
      // Loading a puzzle means the player moved on from any rating dialog.
      resumeSession(Date.now());
      const timeLimitMs = getActiveSetTimeLimitMs();
      // Start session on first puzzle load if not already started
      if (sessionStartedAtMs === null) {
        sessionStartedAtMs = Date.now();
        sessionDeadlineMs = timeLimitMs === null ? null : sessionStartedAtMs + timeLimitMs;
      }
      return sendJson(res, 200, toClientPuzzle(puzzle));
    }

    // Static files (index.html, styles.css, app.js, ...)
    if (req.method === 'GET') {
      let filePath = pathname === '/' ? '/index.html' : pathname;
      filePath = path.normalize(filePath).replace(/^(\.\.[/\\])+/, '');
      const fullPath = path.join(PUBLIC_DIR, filePath);
      if (!fullPath.startsWith(PUBLIC_DIR)) {
        res.writeHead(403);
        return res.end('Forbidden');
      }
      return sendFile(res, fullPath);
    }

    res.writeHead(404, { 'Content-Type': 'text/plain' });
    res.end('Not found');
  } catch (err) {
    console.error(err);
    sendJson(res, 500, { error: 'Internal server error' });
  }
});

server.listen(PORT, () => {
  console.log(`Code Arrange running at http://localhost:${PORT}`);
  console.log(`  Player:      ${PLAYER_NAME}`);
  console.log(`  Puzzle set:  ${cliPuzzleSubfolder} (${PUZZLES_DIR})`);
  console.log(`  Review mode: ${REVIEW_MODE ? 'on' : 'off'}`);
});
