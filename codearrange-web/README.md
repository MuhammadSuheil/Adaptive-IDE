# Code Arrange (Web)

A Node.js web version of the Parsons-Problem-style learning tool: drag
scrambled lines of **Java** code into the correct order to build the
program described above the panels.

Zero external dependencies — the server uses only Node's built-in
`http`/`fs` modules, and the frontend is vanilla HTML/CSS/JS. No
`npm install` required.

## Requirements

- Node.js 18+

## Running it

The server takes two required arguments and one optional flag:

```bash
node server.js "<player name>" <puzzle-set-subfolder> --review
```

- **Player name** — recorded in the attempt log for the whole session. The
  browser UI only ever displays it; there's no field to change it, and the
  server ignores any name sent from the browser, so it can't be spoofed by
  editing page JavaScript either.
- **Puzzle set subfolder** — a folder path relative to `puzzles/` (use `.`
  to serve `puzzles/` itself). Only `*.json` files inside that folder are
  loaded; puzzles elsewhere under `puzzles/` aren't reachable by this
  server instance. `../`-style paths that would escape `puzzles/` are
  rejected.
- **`--review`** (optional) — turns on review mode: after each attempt (Finish
  or timeout), a modal asks the participant to rate the puzzle's difficulty
  from 1 (very easy) to 10 (very hard). The exam timer is paused while the
  dialog is open.

This is meant for running one server process per player and/or per
quiz-set — e.g. a teacher starting a separate instance (on its own port)
for each student, each locked to that student's name and an assigned set
of puzzles:

```bash
PORT=3001 node server.js "Ada Lovelace" exam-set-a --review
PORT=3002 node server.js "Grace Hopper" exam-set-a --review
PORT=3003 node server.js "Alan Turing"  exam-set-b
```

Running with no arguments (or only one) prints usage and exits without
starting the server.

Then open **http://localhost:3000** (or whichever `PORT` you set) in a
browser.

## How it works

- **Puzzles** live as JSON files in `puzzles/` — the exact same format
  used by the original desktop app, so the two are interchangeable.
  The `code` in each line is still Java; only the *app* changed to a
  web stack, not the exercises.
- **The server never sends the solution to the browser.** `GET
  /api/puzzles/:id` strips the `solution` field and shuffles the line
  order before responding, so opening dev tools won't hand over the
  answer. Grading happens server-side via `POST
  /api/puzzles/:id/check`, which compares the submitted line order
  against the solution kept only in the server's copy of the puzzle
  file.
- **Drag and drop** is implemented in plain JavaScript
  (`public/app.js`) using the browser's native HTML5 drag events, with
  a standard "live reorder on dragover" pattern: the dragged line is
  physically moved in the DOM as you drag over a list, found each time
  via a `.dragging` CSS class rather than tracked in separate JS
  state. That's a deliberate choice — it avoids the class of bug the
  desktop version had, where two independent handler objects each
  tracked "the current drag" separately and a cross-list drop landed
  on a handler that never saw the drag start.

## Using the app

1. The start screen shows your (locked-in) name and a dropdown of the
   puzzles available in the puzzle set this server was started with. Pick
   a puzzle and click **Start**. The puzzle's description and lines
   aren't fetched or shown until you do — no peeking at the shuffled
   lines beforehand, and the timer only starts once you've actually begun.
2. Drag lines into **Your Program**, in order. A timer in the top-right
   ticks while you work. Leading indentation is hidden on every line while
   you're solving (only the code content is shown) so indentation depth
   doesn't hint at nesting or ordering. If stripping indentation makes two
   or more lines look identical (classically, bare closing braces like
   `}`), each gets a small `#1` / `#2` tag reflecting their *actual*
   relative order in the correct solution — `#1` really does belong
   before `#2` — so you can still reason your way to a correct answer for
   those lines rather than having to guess which physical tile goes where.
   Clicking Finish reveals the original indentation on every line and the
   tags disappear.
3. Click **Finish**. The server grades your submitted order, stops the
   clock, and logs your name, completion time, and whether you got it
   right to `logs/attempts.csv`. If you used lines that belong in no
   correct answer, the feedback says how many (in a debug puzzle: "2 lines
   you used still contain bugs"), and a puzzle that has an explanation
   shows it in an **Explanation** box below.
4. **Try Again** reshuffles the same puzzle for a fresh attempt while the
   global session timer continues running without resetting.
   **Next Puzzle** is disabled until you click Finish for the current
   attempt; once enabled, it loads the next puzzle in the set (in the same
   order the dropdown lists them) and continues the same session timer. On
   the last puzzle in the set, the button relabels itself **Back to Start**
   and returns you to the start screen instead (your name stays the same
   either way — it's fixed for the whole server session).

## Resuming after a refresh

Refreshing (or accidentally closing and reopening) the tab mid-puzzle
doesn't send you back to the start screen — the app saves enough state
to `sessionStorage` to pick up exactly where you left off:

- Which puzzle you're on, using the *exact* line data and disambiguation
  tags you were already looking at (not a newly re-shuffled copy).
- The current arrangement of both lists, updated after every drag.
- The original start time, so the timer keeps counting real elapsed time
  straight through the refresh rather than resetting to zero.
- Whether you'd already clicked Finish — if so, the graded result
  (correct/incorrect highlighting, revealed indentation, the feedback
  message, frozen completion time) is restored too, rather than handing
  you a re-opened, editable puzzle.

This uses `sessionStorage`, not `localStorage`: it's scoped to this one
browser tab and is cleared automatically when the tab closes, so it
survives a refresh but doesn't linger indefinitely or leak between
tabs/devices. Clicking **Back to Start** clears the saved state
deliberately, since choosing to return to the start screen means
starting fresh next time. If the puzzle set changes between visits (a
saved puzzle id no longer exists), the stale state is discarded and you
land on the start screen as normal rather than the app trying to resume
something that isn't there anymore.

## Attempt log

Every time a puzzle attempt is finished (either manually by clicking **Finish** or automatically when the time limit expires), an identical row is appended to **two** CSV files:
1. `logs/attempts.csv` (global master log of all attempts across all sets)
2. `logs/<puzzle-set>.csv` (e.g. `logs/exam-set-a.csv`, specific to the active set)

Both files are created automatically on first use with the following schema:

```
timestamp,attempt_id,name,puzzle_set,puzzle_id,puzzle_title,time_seconds,submission_type,correct,correct_lines,total_lines,score_percent,rating
2026-09-15T20:16:17.749Z,8786c422-...,Ada Lovelace,exam-set-a,hello_world,"1. Hello, World!",12.35,manual,true,5,5,100.0,8
2026-09-15T20:22:03.410Z,3392e7c5-...,Grace Hopper,exam-set-a,hello_world,"1. Hello, World!",300.00,timeout,false,1,5,20.0,-
```

- `attempt_id`: A server-generated UUID unique to that row, used internally to associate a difficulty rating with the attempt.
- `puzzle_set`: Indicates which puzzle set folder was active during the session.
- `submission_type`: Records `'manual'` (player clicked Finish) or `'timeout'` (system auto-submitted due to the timer expiring).
- `correct`: Reflects whether the submission exactly matched one of the puzzle's accepted solutions.
- `correct_lines` / `total_lines` / `score_percent`: Capture partial-credit score even on a miss (matched against whichever accepted solution the learner came closest to).
- `rating`: The participant's 1–10 difficulty rating in review mode, or `-` outside review mode (or if unanswered).

Grading and logging both happen server-side, from the server's own copy of the solutions — the client only sends the submitted line order and submission type, ensuring logs cannot be spoofed. Fields containing commas or quotes are CSV-escaped automatically.

### Multi-instance log safety (Cross-process locking)

When multiple server instances are started from the same project folder (e.g. one process per student), all instances share `logs/attempts.csv`. File operations (appends, migrations, and rating updates) are protected by a file-based lock (`logs/attempts.csv.lock`) with atomic create-if-absent semantics and automatic stale-lock recovery, ensuring concurrent writes never overwrite each other.

## Review mode

When started with `--review`, a dialog prompts the participant to rate each puzzle's difficulty on a scale of 1 (very easy) to 10 (very hard) immediately following each attempt (including timeouts).
- The exam session timer is **paused** while the rating dialog is open, so rating deliberation does not penalize exam time.
- Clicking a rating submits `POST /api/attempts/:attemptId/rating`, updating the rating in place across both `attempts.csv` and the active `<puzzle-set>.csv`.
- The dialog state is persisted in `sessionStorage`, so refreshing the browser preserves the dialog.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/api/config` | `{ playerName, puzzleSet, setConfig, reviewMode, sessionStartedAtMs, sessionDeadlineMs, sessionPausedAtMs }` — locked server configuration and exam session state. |
| GET | `/api/puzzles` | List of `{ id, title, type, mode, description }` for every puzzle in the active puzzle set. |
| GET | `/api/puzzles/:id` | One puzzle's `{ id, title, type, mode, description, lines }`. Lines are shuffled, `code` has full original indentation. Solutions and explanations are never exposed here. |
| POST | `/api/puzzles/:id/check` | Body `{ "order": [<line ids>], "timeMs"?: number, "submissionType"?: string }`. Grades the submission, logs the attempt, and returns `{ correct, correctPositions, total, submittedCount, perLine, wrongLines, explanation, attemptId? }`. `attemptId` is provided only in review mode. |
| POST | `/api/attempts/:attemptId/rating` | Review mode only (`403` otherwise). Body `{ "rating": <integer 1-10> }`. Updates that attempt's row in both `attempts.csv` and the set log in place, and unfreezes the exam clock. |

## Puzzle sets (subfolders under `puzzles/`)

Which puzzles a running server can show is controlled entirely by the
`<puzzle-set-subfolder>` argument on the command line — see "Running it"
above. `puzzles/` in this repo ships with:

```
puzzles/
  master_config.json
  exam-set-a/
    1_hello_world.json
    2_sum_two_numbers.json
  exam-set-b/
    3_for_loop_sum.json
    4_fix_even_sum.json
```

`exam-set-a` and `exam-set-b` are just examples of grouping puzzles into
sets you can hand out separately (e.g. different quizzes, difficulty
tiers, or classes) — copy/rename/delete freely. A server started with
`exam-set-a` genuinely cannot serve `exam-set-b`'s puzzles: the puzzle
list only shows what's in that folder, and `GET`/`POST` requests for a
puzzle id from a different set 404, even if you know its id.

### Time Limits (master_config.json)

You can set a strict time limit for any puzzle set by editing `puzzles/master_config.json`. The keys must match the exact folder names of your puzzle sets.

```json
{
  "exam-set-a": {
    "timeLimitMinutes": 5
  },
  "exam-set-b": {
    "timeLimitMinutes": 10
  }
}
```

When a time limit is configured for the active set, the timer in the UI will display a session target (e.g., `0:15 / 5:00`). This time limit applies globally to the entire quiz session across all questions and retries. When the limit is reached, the active puzzle is automatically submitted, subsequent retries or moves are locked, and the exam session officially ends. If a set is not listed in `master_config.json`, it has no time limit (stopwatch mode).

## Puzzle modes

Every puzzle has an optional `mode` (default `arrange`), kept separate
from `type` (the topic) so the two can vary independently. **Mode only
changes what the UI asks the player to do** — the server grades every mode
identically, against `solutions`, and treats any line that appears in no
solution as a distractor.

| Mode | The player… | The tiles are… |
|---|---|---|
| `arrange` (default) | rebuilds a program from its lines | the program's own lines, optionally plus distractors |
| `output` | reads code in the description and arranges what it prints | output lines |
| `debug` | reads a *buggy* program in the description and rebuilds the corrected one | the corrected program's lines plus plausible-but-buggy distractor lines |

Puzzles in a non-default mode get a badge in the header and a one-line
instruction under the description; `arrange` puzzles look exactly as they
always have.

### Debug and repair

This is the most demanding mode, because it stacks the other two: the
player has to *trace* the buggy program to see where it departs from the
spec, and then *construct* the fix. The distractors are the buggy versions
of lines (`i < 10` where `i <= 10` belongs, `== 1` where `== 0` belongs),
so syntax alone can't rule them out — only understanding what the code
does can.

```json
{
  "id": "fix_even_sum",
  "type": "loops",
  "mode": "debug",
  "description": "…expected output: 30, but it prints 25…\n```\n<the buggy program>\n```",
  "explanation": "There were two bugs. The loop condition i < 10 stops one short…",
  "lines": [ /* the 11 correct lines, ids 1–11 */
    { "id": 12, "code": "        for (int i = 1; i < 10; i++) {" },
    { "id": 13, "code": "            if (i % 2 == 1) {" }
  ],
  "solutions": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
}
```

- The buggy program goes in `description` as a fenced code snippet (see
  below), along with what the program *should* do — ideally the expected
  and actual output, so the player has a symptom to work from.
- The buggy tiles are simply lines that no solution uses (ids 12 and 13
  here). Make each one a believable near-miss of the real line rather than
  an obvious wrong one.
- `explanation` is optional. It's shown in an **Explanation** box after
  Finish, and may contain fenced code snippets like the description can.
  It's held back by the server until grading, so it can't be read out of
  the puzzle beforehand.
- After Finish, the feedback says how many buggy tiles the player kept
  ("2 lines you used still contain bugs"), so a wrong attempt tells them
  something rather than only "some lines are out of place".
- See `puzzles/4_fix_even_sum.json` for a working example.

You don't have to build these by hand — `scripts/create-puzzle.js` can
generate them from a correct file and a buggy file (next section).

## Generating a puzzle from a Java file

Hand-writing the `lines`/`id`/`solutions` JSON for every puzzle gets
tedious fast, so `scripts/create-puzzle.js` builds a starter puzzle file
for you from a working `.java` file: one tile per line, in the file's own
order, with that order recorded as the accepted solution.

```bash
node scripts/create-puzzle.js path/to/MySolution.java
```

It prompts for anything you don't pass as a flag (puzzle id, title,
description); everything can also be supplied non-interactively:

```bash
node scripts/create-puzzle.js path/to/MySolution.java \
  --id=my_solution \
  --title="My Solution" \
  --type=loops \
  --description="Write a program that..." \
  --output=puzzles/exam-set-a \
  --force
```

Run `node scripts/create-puzzle.js --help` for the full option list
(including `--keep-blank-lines`, if you want blank lines to appear as
their own draggable tiles instead of being dropped).

In `arrange` mode this only gives you **one** accepted solution and **no
distractor lines** — both are easy to add by hand afterward (see "Writing
your own puzzles" below) but the script has no way to know which alternate
orderings are valid Java or which wrong lines you'd want to plant, so it
doesn't try to guess.

### Generating a debug puzzle

For `debug` mode you give it two files — the **correct** program, and a
**buggy** version of it — and it works out the distractors itself:

```bash
node scripts/create-puzzle.js Correct.java \
  --mode=debug --buggy=Buggy.java \
  --type=loops --title="Fix the Even Sum" \
  --description="This should print 30 but prints 25. Find the bugs and rebuild the program." \
  --explanation="There were two bugs. The loop condition..."
```

- The buggy program is appended to the description as a code snippet.
- Every line in the buggy file that has no counterpart in the correct file
  becomes a distractor tile, and the script lists the ones it found so you
  can sanity-check them. Lines are compared exactly, **indentation
  included**, and a line that legitimately appears twice in the correct
  program isn't mistaken for a distractor when it appears twice in the
  buggy one.
- A bug that only *reorders* lines, or *deletes* one, produces no
  distractor (the real line is already among the tiles) — the script
  warns you when it finds none, in case you passed the wrong files.
- The script can generate `arrange` and `debug` puzzles. `output`
  puzzles are hand-written, because their tiles are output lines rather
  than source code.

## Writing your own puzzles

Same schema as before — drop a new `*.json` file into the right puzzle
set's folder and it shows up automatically (no restart needed, the
server re-reads the folder on every request):

```json
{
  "id": "unique_id",
  "title": "Shown in the dropdown",
  "type": "basics",
  "mode": "arrange",
  "description": "Shown above the two panels.",
  "lines": [
    { "id": 1, "code": "public class Example {" },
    { "id": 2, "code": "    public static void main(String[] args) {" },
    { "id": 3, "code": "        System.out.println(\"Hi\");" },
    { "id": 4, "code": "    }" },
    { "id": 5, "code": "}" }
  ],
  "solutions": [
    [1, 2, 3, 4, 5]
  ]
}
```

`type` is a free-form category label (e.g. `basics`, `variables`,
`loops`) used to group puzzles in the dropdown under an `<optgroup>`.
Puzzles with the same `type` end up in the same group, in the order
their files sort in. It's optional — a puzzle file without one loads
fine and lands in a generic "Uncategorized" group — but the puzzle
picker gets a lot more useful once there's more than a handful of
puzzles and they're grouped by topic.

`mode` is what kind of task the puzzle is — `arrange` (the default),
`output`, or `debug` — and is independent of `type`. See "Puzzle modes"
above.

`description` can also contain a fenced code snippet, delimited by
` ``` ` on its own line, exactly like Markdown:

```json
"description": "This loop sums 1 through 5:\n```\nint total = 0;\nfor (int i = 1; i <= 5; i++) {\n    total += i;\n}\n```\nAdapt it to sum 1 through 10 instead."
```

The fenced part renders as its own monospaced, line-numbered block with
indentation preserved exactly — not as a wrapped paragraph, where a
browser's normal whitespace handling would collapse it onto one line and
destroy the formatting. Text outside the fence renders as normal
paragraph(s); a blank line in the JSON string (`\n\n`) starts a new
paragraph. `puzzles/3_for_loop_sum.json` is a working example of this.

`solutions` is a list of accepted orderings — the submission only needs to
match **one** of them to count as correct. This is meant for puzzles where
more than one ordering is genuinely valid Java, not as a place to list
near-misses. `puzzles/2_sum_two_numbers.json` is a real example: declaring
`a` before `b` or `b` before `a` are both fine, since neither declaration
depends on the other, so it lists both orderings:

```json
"solutions": [
  [1, 2, 3, 4, 5, 6, 7, 8],
  [1, 2, 4, 3, 5, 6, 7, 8]
]
```

When a submission doesn't exactly match any accepted solution, the
"X of Y lines correct" feedback is computed against whichever accepted
solution the learner came closest to, so partial credit stays meaningful
even for puzzles with several right answers.

(A puzzle file written with the older singular `"solution": [ids]` field
still works — it's treated as `"solutions": [[ids]]` — but new puzzles
should use `solutions`.)

Any line `id` not listed in **any** accepted solution is a distractor: it
appears in the pool but shouldn't end up in the final program (see
`puzzles/2_sum_two_numbers.json`'s red-herring `a - b` line, id `9`,
which isn't part of either accepted solution).

## Project layout

```
server.js              HTTP server: CLI args, static files + puzzle API + attempt logging (no dependencies)
package.json
scripts/
  create-puzzle.js      CLI helper: scaffold a puzzle JSON from a .java file
puzzles/                default puzzle set (served when you pass "." as the subfolder)
  exam-set-a/           example puzzle set — pass "exam-set-a" to serve only these
  exam-set-b/           another example puzzle set
logs/attempts.csv       master attempt log (created automatically on first submission)
logs/<set-name>.csv     set-specific attempt log (e.g. logs/exam-set-a.csv)
public/
  index.html
  styles.css
  app.js                start screen (locked name + puzzle picker), timer, drag-and-drop, finish/try-again flow
```

## Ideas for extending it

- Persist progress per learner (would need some form of accounts/sessions).
- A puzzle-authoring page instead of hand-editing JSON files.
- Per-line hints or a "reveal one correct line" button.
- Score indentation, not just line order.
- Deploy behind a real process manager (pm2, systemd) or containerize
  with a `Dockerfile` for production use.
