'use strict';

const http = require('http');
const fs = require('fs');
const path = require('path');
const { URL } = require('url');

const PORT = process.env.PORT || 3000;
const PUBLIC_DIR = path.join(__dirname, 'public');
const PUZZLES_ROOT = path.join(__dirname, 'puzzles');

// ---- Command-line arguments --------------------------------------------
//
// node server.js <player-name> <puzzle-set-subfolder>
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

const [, , cliPlayerName, cliPuzzleSubfolder] = process.argv;

if (!cliPlayerName || !cliPuzzleSubfolder) {
  console.error('Usage: node server.js <player-name> <puzzle-set-subfolder>');
  console.error('');
  console.error('  <player-name>            Name recorded in the attempt log; locked for the session.');
  console.error('  <puzzle-set-subfolder>   Folder inside puzzles/ whose *.json files will be served.');
  console.error('                           Use "." to serve puzzles/ itself.');
  console.error('');
  console.error('Example: node server.js "Ada Lovelace" exam-set-a');
  process.exit(1);
}

const PLAYER_NAME = cliPlayerName;

const PUZZLES_DIR = path.resolve(PUZZLES_ROOT, cliPuzzleSubfolder);
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
// Every time a learner clicks "Finish", the server (not the browser) grades
// the attempt and appends a row to logs/attempts.csv. Doing the logging
// server-side, using the server's own grading result, means the log can't
// be spoofed by editing client-side JavaScript.

const LOG_DIR = path.join(__dirname, 'logs');
const LOG_FILE = path.join(LOG_DIR, 'attempts.csv');
const LOG_HEADER = 'timestamp,name,puzzle_id,puzzle_title,time_seconds,correct,correct_lines,total_lines,score_percent\n';

function csvField(value) {
  const str = String(value);
  if (/[",\n]/.test(str)) {
    return '"' + str.replace(/"/g, '""') + '"';
  }
  return str;
}

function logAttempt({ name, puzzleId, puzzleTitle, timeSeconds, correct, correctPositions, total }) {
  if (!fs.existsSync(LOG_DIR)) {
    fs.mkdirSync(LOG_DIR, { recursive: true });
  }
  if (!fs.existsSync(LOG_FILE)) {
    fs.writeFileSync(LOG_FILE, LOG_HEADER);
  }
  const scorePercent = total > 0 ? ((correctPositions / total) * 100).toFixed(1) : '0.0';
  const row =
    [
      new Date().toISOString(),
      csvField(name),
      csvField(puzzleId),
      csvField(puzzleTitle),
      timeSeconds.toFixed(2),
      correct ? 'true' : 'false',
      correctPositions,
      total,
      scorePercent,
    ].join(',') + '\n';
  fs.appendFileSync(LOG_FILE, row);
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
    res.writeHead(200, { 'Content-Type': MIME_TYPES[ext] || 'application/octet-stream' });
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
    // GET /api/config  -> the locked player name and active puzzle set for this server instance
    if (req.method === 'GET' && pathname === '/api/config') {
      let setConfig = {};
      const masterConfigPath = path.join(PUZZLES_ROOT, 'master_config.json');
      if (fs.existsSync(masterConfigPath)) {
        try {
          const masterConfig = JSON.parse(fs.readFileSync(masterConfigPath, 'utf8'));
          if (masterConfig[cliPuzzleSubfolder]) {
            setConfig = masterConfig[cliPuzzleSubfolder];
          }
        } catch (e) {
          console.error('Failed to parse master_config.json:', e);
        }
      }
      return sendJson(res, 200, { playerName: PLAYER_NAME, puzzleSet: cliPuzzleSubfolder, setConfig });
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
      const result = gradeSubmission(puzzle, body.order);

      // The player name is locked in at server startup (see the CLI args
      // at the top of this file) and is never taken from the request body
      // -- otherwise a player could spoof a different name by editing
      // browser JavaScript. Every finished attempt gets logged, since
      // there's always a name for this server instance.
      const timeMs = Number(body.timeMs);
      const timeSeconds = Number.isFinite(timeMs) && timeMs >= 0 ? timeMs / 1000 : 0;
      try {
        logAttempt({
          name: PLAYER_NAME,
          puzzleId: puzzle.id,
          puzzleTitle: puzzle.title,
          timeSeconds,
          correct: result.correct,
          correctPositions: result.correctPositions,
          total: result.total,
        });
      } catch (logErr) {
        // Logging failures shouldn't break grading for the learner.
        console.error('Failed to write attempt log:', logErr);
      }

      return sendJson(res, 200, result);
    }

    // GET /api/puzzles/:id  -> puzzle detail (solution stripped, lines shuffled)
    match = pathname.match(/^\/api\/puzzles\/([^/]+)$/);
    if (req.method === 'GET' && match) {
      const puzzle = findPuzzleById(match[1]);
      if (!puzzle) return sendJson(res, 404, { error: 'Puzzle not found' });
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
});
