#!/usr/bin/env node
'use strict';

// Turns a working Java source file into a starter puzzle JSON: one line
// of code per tile, in the file's own order, with that order recorded as
// the puzzle's (single, initial) correct solution. Doesn't touch the
// running server or any existing puzzle files -- it just writes a new
// .json file for you to review, edit (add distractor lines, alternate
// accepted orderings, etc.) and drop into a puzzle set folder.
//
// Usage:
//   node scripts/create-puzzle.js <path-to-java-file> [options]
//
// Run with --help for the full option list.

const fs = require('fs');
const path = require('path');
const readline = require('readline');

const PROJECT_ROOT = path.join(__dirname, '..');
const DEFAULT_OUTPUT_DIR = path.join(PROJECT_ROOT, 'puzzles');

function printUsage() {
  console.log(`Usage: node scripts/create-puzzle.js <path-to-java-file> [options]

Turns a working Java source file into a starter puzzle JSON: one line of
code per tile, in the file's own order, with that order recorded as the
single correct solution. Prompts for anything not supplied as a flag.

Options:
  --id=<slug>            Puzzle id / output filename (without .json).
                          Default: derived from the public class name, or the source filename.
  --title=<text>         Puzzle title shown in the puzzle dropdown.
  --type=<category>      Category used to group puzzles in the dropdown (e.g. basics, loops).
                          Default: uncategorized.
  --description=<text>   Puzzle description shown above the panels. May contain a fenced
                          code snippet (delimited by \`\`\` on its own, like Markdown) which
                          renders as a monospaced, line-numbered block instead of plain text.
  --mode=<mode>          arrange (default) or debug. See "Debug and repair puzzles" below.
                          ("output" puzzles are hand-written: their tiles are output lines, not code.)
  --buggy=<file>         Debug mode: a buggy version of the same program. It is shown to the
                          player inside the description, and every line in it that isn't in the
                          correct file becomes a distractor tile.
  --explanation=<text>   Optional. Shown to the player after Finish (e.g. what the bug was).
  --output=<dir>         Directory to write <id>.json into. Default: puzzles/
  --keep-blank-lines     Keep blank lines as their own tiles (skipped by default).
  --force                Overwrite the output file if it already exists, without asking.
  --help                 Show this message.

Examples:
  node scripts/create-puzzle.js MySolution.java
  node scripts/create-puzzle.js MySolution.java --output=puzzles/exam-set-a --force
  node scripts/create-puzzle.js MySolution.java --id=my_solution --title="My Solution" --type=loops --description="..." --force

Debug and repair puzzles:
  Pass the CORRECT program as the main file, and a buggy version with --buggy.
  The script shows the buggy code in the description and adds each buggy line
  that differs from the correct file as a distractor tile (compared exactly,
  indentation included). It lists the distractors it found so you can check them.
    node scripts/create-puzzle.js Correct.java --mode=debug --buggy=Buggy.java \
      --description="This should print 30 but prints 25. Find the bugs." --force

Otherwise, after it writes the file, open it up: an arrange puzzle gets one
accepted solution and no distractor lines -- both are easy to add by hand (see
README.md's "Writing your own puzzles" section) but the script won't guess at
them for you.
`);
}

function parseArgs(argv) {
  const args = { flags: {}, positional: [] };
  for (const arg of argv) {
    if (arg === '--help' || arg === '-h') {
      args.flags.help = true;
    } else if (arg === '--keep-blank-lines') {
      args.flags.keepBlankLines = true;
    } else if (arg === '--force') {
      args.flags.force = true;
    } else if (arg.startsWith('--')) {
      const eq = arg.indexOf('=');
      if (eq === -1) {
        console.error(`Unrecognized option: ${arg} (expected --name=value)`);
        process.exit(1);
      }
      const key = arg.slice(2, eq);
      const value = arg.slice(eq + 1);
      args.flags[key] = value;
    } else {
      args.positional.push(arg);
    }
  }
  return args;
}

function slugify(text) {
  return text
    .replace(/([a-z0-9])([A-Z])/g, '$1_$2') // camelCase / PascalCase -> camel_Case
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .replace(/_+/g, '_');
}

function humanize(slug) {
  return slug
    .split('_')
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

function detectPublicTypeName(source) {
  const match = source.match(/\bpublic\s+(?:final\s+|abstract\s+)?(?:class|interface|enum|record)\s+(\w+)/);
  return match ? match[1] : null;
}

function extractLines(source, keepBlankLines) {
  const rawLines = source.split(/\r\n|\r|\n/).map((line) => line.replace(/[ \t]+$/, ''));
  // Drop a single trailing empty string caused by the file ending in a
  // newline -- that's a file-format artifact, not an intentional blank line.
  if (rawLines.length > 0 && rawLines[rawLines.length - 1] === '') {
    rawLines.pop();
  }
  const kept = keepBlankLines ? rawLines : rawLines.filter((line) => line.trim() !== '');
  return kept.map((code, index) => ({ id: index + 1, code }));
}

const SUPPORTED_MODES = ['arrange', 'debug'];

// The buggy program, tidied for display in the description: trailing
// whitespace stripped, leading/trailing blank lines dropped, but blank lines
// in the middle and all indentation kept exactly as written.
function snippetFromSource(source) {
  const lines = source.split(/\r\n|\r|\n/).map((line) => line.replace(/[ \t]+$/, ''));
  while (lines.length > 0 && lines[0].trim() === '') lines.shift();
  while (lines.length > 0 && lines[lines.length - 1].trim() === '') lines.pop();
  return lines.join('\n');
}

// Buggy lines that have no counterpart in the correct file. Each correct
// line can "absorb" at most one identical buggy line (a multiset difference,
// not a set difference), so a line that legitimately appears twice in the
// correct program isn't mistaken for a distractor when it appears twice in
// the buggy one. Whatever is left over is a line that only exists in the
// buggy version -- i.e. the bug itself -- and becomes a distractor tile.
function findDistractors(correctCodes, buggyCodes) {
  const remaining = new Map();
  for (const code of correctCodes) {
    remaining.set(code, (remaining.get(code) || 0) + 1);
  }
  const distractors = [];
  for (const code of buggyCodes) {
    const count = remaining.get(code) || 0;
    if (count > 0) {
      remaining.set(code, count - 1);
    } else {
      distractors.push(code);
    }
  }
  return distractors;
}

function createPrompter() {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  // Deliberately not using rl.question() in a loop: with piped (non-TTY)
  // input, lines that arrive before the next question() call registers its
  // listener get silently dropped, since question() only captures the
  // *next* 'line' event rather than queuing them. Consuming the interface
  // as an async iterator buffers correctly regardless of how fast input
  // arrives, and still works normally for a real interactive terminal.
  const lineIterator = rl[Symbol.asyncIterator]();
  const ask = async (question) => {
    process.stdout.write(question);
    const { value, done } = await lineIterator.next();
    return done ? '' : value;
  };
  return { ask, close: () => rl.close() };
}

async function resolveField({ flagValue, prompt, defaultValue, required, ask }) {
  if (flagValue !== undefined) {
    return flagValue;
  }
  const suffix = defaultValue ? ` [${defaultValue}]` : '';
  while (true) {
    const answer = (await ask(`${prompt}${suffix}: `)).trim();
    if (answer) return answer;
    if (defaultValue) return defaultValue;
    if (!required) return '';
    console.log('  (this field is required, please enter a value)');
  }
}

async function main() {
  const { flags, positional } = parseArgs(process.argv.slice(2));

  if (flags.help || positional.length === 0) {
    printUsage();
    process.exit(positional.length === 0 && !flags.help ? 1 : 0);
  }

  const javaFilePath = positional[0];
  if (!fs.existsSync(javaFilePath) || !fs.statSync(javaFilePath).isFile()) {
    console.error(`Java file not found: ${javaFilePath}`);
    process.exit(1);
  }

  const source = fs.readFileSync(javaFilePath, 'utf8');
  const detectedTypeName = detectPublicTypeName(source);
  const fallbackName = path.basename(javaFilePath, path.extname(javaFilePath));
  const defaultId = slugify(detectedTypeName || fallbackName);
  const defaultTitle = humanize(defaultId);

  const { ask, close } = createPrompter();

  console.log(`Reading ${javaFilePath}${detectedTypeName ? ` (detected public type: ${detectedTypeName})` : ''}\n`);

  const id = slugify(
    await resolveField({ flagValue: flags.id, prompt: 'Puzzle id (letters/numbers/underscores)', defaultValue: defaultId, required: true, ask })
  );
  const title = await resolveField({ flagValue: flags.title, prompt: 'Puzzle title', defaultValue: defaultTitle, required: true, ask });
  const category = await resolveField({
    flagValue: flags.type,
    prompt: 'Puzzle type / category (e.g. basics, variables, loops)',
    defaultValue: 'uncategorized',
    required: false,
    ask,
  });
  const mode = (
    await resolveField({
      flagValue: flags.mode,
      prompt: 'Puzzle mode (arrange or debug)',
      defaultValue: 'arrange',
      required: false,
      ask,
    })
  )
    .trim()
    .toLowerCase();
  if (!SUPPORTED_MODES.includes(mode)) {
    close();
    console.error(`Unsupported mode "${mode}". This script can generate: ${SUPPORTED_MODES.join(', ')}.`);
    console.error('("output" puzzles are hand-written, since their tiles are output lines rather than source code.)');
    process.exit(1);
  }

  const description = await resolveField({
    flagValue: flags.description,
    prompt:
      mode === 'debug'
        ? 'Puzzle description (what the program should do; the buggy code is appended for you)'
        : 'Puzzle description (shown to the player)',
    defaultValue: '',
    required: true,
    ask,
  });

  let buggySource = null;
  let explanation = '';
  if (mode === 'debug') {
    const buggyPath = await resolveField({
      flagValue: flags.buggy,
      prompt: 'Path to the BUGGY version of the Java file',
      defaultValue: '',
      required: true,
      ask,
    });
    if (!fs.existsSync(buggyPath) || !fs.statSync(buggyPath).isFile()) {
      close();
      console.error(`Buggy Java file not found: ${buggyPath}`);
      process.exit(1);
    }
    buggySource = fs.readFileSync(buggyPath, 'utf8');
    explanation = (
      await resolveField({
        flagValue: flags.explanation,
        prompt: 'Explanation shown after Finish (what the bug was; optional)',
        defaultValue: '',
        required: false,
        ask,
      })
    ).trim();
  } else if (flags.explanation !== undefined) {
    explanation = String(flags.explanation).trim();
  }

  close();

  const lines = extractLines(source, Boolean(flags.keepBlankLines));
  if (lines.length === 0) {
    console.error('No code lines found in that file (is it empty?). Nothing was written.');
    process.exit(1);
  }

  // Debug mode: show the buggy program inside the description, and turn every
  // buggy line with no counterpart in the correct file into a distractor tile
  // (ids continue on from the correct lines; they appear in no solution).
  let puzzleDescription = description;
  let allLines = lines;
  let distractorLines = [];
  if (mode === 'debug') {
    const buggyCodes = extractLines(buggySource, Boolean(flags.keepBlankLines)).map((line) => line.code);
    const distractorCodes = findDistractors(lines.map((line) => line.code), buggyCodes);
    distractorLines = distractorCodes.map((code, index) => ({ id: lines.length + index + 1, code }));
    allLines = lines.concat(distractorLines);
    const fence = '```';
    puzzleDescription = description + '\n' + fence + '\n' + snippetFromSource(buggySource) + '\n' + fence;
  }

  const puzzle = {
    id,
    title,
    type: category || 'uncategorized',
    mode,
    description: puzzleDescription,
  };
  if (explanation) {
    puzzle.explanation = explanation;
  }
  puzzle.lines = allLines;
  puzzle.solutions = [lines.map((line) => line.id)];

  const outputDir = flags.output ? path.resolve(process.cwd(), flags.output) : DEFAULT_OUTPUT_DIR;
  const outputPath = path.join(outputDir, `${id}.json`);

  if (fs.existsSync(outputPath) && !flags.force) {
    const { ask: askOverwrite, close: closeOverwrite } = createPrompter();
    const answer = (await askOverwrite(`${outputPath} already exists. Overwrite? (y/N): `)).trim().toLowerCase();
    closeOverwrite();
    if (answer !== 'y' && answer !== 'yes') {
      console.log('Cancelled -- nothing was written.');
      process.exit(0);
    }
  }

  fs.mkdirSync(outputDir, { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(puzzle, null, 2) + '\n');

  if (mode === 'debug') {
    console.log(`\nWrote ${lines.length} correct line(s) + ${distractorLines.length} distractor(s) to ${outputPath}`);
    if (distractorLines.length === 0) {
      console.log('\nWarning: no line in the buggy file differs from the correct file, so there are');
      console.log('no distractor tiles. (That is fine if the bug is purely one of ordering, or a');
      console.log('missing line; otherwise check that you passed the right two files.)');
    } else {
      console.log('\nDistractor tiles (buggy lines with no counterpart in the correct file):');
      for (const line of distractorLines) {
        console.log(`  id ${line.id}: ${line.code.trim()}`);
      }
    }
    console.log('\nThe buggy program was appended to the description as a code snippet.');
  } else {
    console.log(`\nWrote ${lines.length} line(s) to ${outputPath}`);
    console.log('This puzzle currently has exactly one accepted solution (the file\'s own order)');
    console.log('and no distractor lines. Open the file to add either by hand if you want them --');
    console.log('see README.md\'s "Writing your own puzzles" section for the schema.');
  }
}

main().catch((err) => {
  console.error('create-puzzle failed:', err);
  process.exit(1);
});
