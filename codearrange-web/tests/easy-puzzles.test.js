'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = path.join(__dirname, '..');
const setKeys = ['easy-1', 'easy-2', 'easy-3', 'easy-4'];

function readBank() {
  return setKeys.flatMap(set => {
    const directory = path.join(root, 'puzzles', set);
    assert.ok(fs.existsSync(directory), `${set} must be available`);
    const files = fs.readdirSync(directory).filter(file => file.endsWith('.json')).sort();
    assert.equal(files.length, 25, `${set} must contain 25 puzzles`);
    return files.map((file, index) => {
      assert.ok(file.startsWith(`${String(index + 1).padStart(2, '0')}_`), `${set}/${file}: numeric ordering`);
      return { set, file, puzzle: JSON.parse(fs.readFileSync(path.join(directory, file), 'utf8')) };
    });
  });
}

test('the easy bank provides 100 distinct puzzles in four balanced sets', () => {
  const bank = readBank();
  assert.equal(bank.length, 100);
  assert.equal(new Set(bank.map(({ puzzle }) => puzzle.id)).size, 100);
  const fingerprints = bank.map(({ puzzle }) => JSON.stringify({
    mode: puzzle.mode, code: puzzle.lines.map(line => line.code.trim()),
    descriptionCode: puzzle.description.match(/```[\s\S]*?```/)?.[0] || '',
  }));
  assert.equal(new Set(fingerprints).size, 100, 'identical exercises must not count twice');
  for (const set of setKeys) {
    const modes = { arrange: 0, output: 0, debug: 0 };
    for (const { puzzle } of bank.filter(item => item.set === set)) modes[puzzle.mode]++;
    assert.deepEqual(modes, { arrange: 17, output: 4, debug: 4 }, set);
  }
});

test('every accepted solution has valid IDs, distinct entries, and ordered identical tiles', () => {
  for (const { set, file, puzzle } of readBank()) {
    const label = `${set}/${file}`;
    assert.equal(puzzle.id, file.replace(/^\d+_/, '').replace(/\.json$/, ''), label);
    assert.ok(puzzle.title && puzzle.description && puzzle.type, label);
    assert.ok(!/^\d+[.)]\s/.test(puzzle.title), label);
    const ids = puzzle.lines.map(line => line.id);
    assert.equal(new Set(ids).size, ids.length, label);
    assert.ok(puzzle.solutions.length > 0, label);
    assert.equal(new Set(puzzle.solutions.map(order => JSON.stringify(order))).size, puzzle.solutions.length, label);
    const canonical = puzzle.solutions[0];
    for (const order of puzzle.solutions) {
      assert.equal(order.length, canonical.length, label);
      assert.equal(new Set(order).size, order.length, label);
      assert.ok(order.every(id => ids.includes(id) && canonical.includes(id)), label);
      for (const line of puzzle.lines) {
        const equivalent = canonical.filter(id => puzzle.lines.find(item => item.id === id).code.trimStart() === line.code.trimStart());
        assert.deepEqual(order.filter(id => equivalent.includes(id)), equivalent, label);
      }
    }
    if (puzzle.mode === 'arrange') assert.equal(canonical.length, ids.length, label);
    if (puzzle.mode === 'debug') {
      assert.ok(puzzle.explanation, label);
      assert.equal(ids.length - canonical.length, puzzle.id === 'fix_even_sum' ? 2 : 1, label);
    }
    if (puzzle.mode !== 'arrange') assert.match(puzzle.description, /```[\s\S]+```/, label);
  }
});

test('easy sets have no deadline and retain the easy research label', () => {
  const config = JSON.parse(fs.readFileSync(path.join(root, 'puzzles', 'master_config.json'), 'utf8'));
  for (const set of setKeys) {
    assert.equal(config[set].difficultyLevel, 'easy', set);
    assert.equal(config[set].showTimerUI, 'off', set);
    assert.ok(config[set].timeLimitMinutes === undefined || config[set].timeLimitMinutes === 0, set);
  }
  assert.equal(config.medium.timeLimitMinutes, 12);
  assert.equal(config.hard.timeLimitMinutes, 15);
});

test('each puzzle has independently specified Java execution expectations', () => {
  const location = path.join(__dirname, 'fixtures', 'easy-puzzle-cases.json');
  assert.ok(fs.existsSync(location), 'Java expectation fixtures must be supplied');
  const cases = JSON.parse(fs.readFileSync(location, 'utf8'));
  const bank = readBank();
  assert.deepEqual(Object.keys(cases).sort(), bank.map(({ puzzle }) => puzzle.id).sort());
  for (const { puzzle } of bank) {
    assert.equal(typeof cases[puzzle.id].stdout, 'string', puzzle.id);
    if (puzzle.mode === 'output') {
      const output = puzzle.solutions[0].map(id => puzzle.lines.find(line => line.id === id).code.trim()).join('\n') + '\n';
      assert.equal(output, cases[puzzle.id].stdout, puzzle.id);
    }
  }
});

test('the thirteen legacy exercises retain every code tile and accepted solution', () => {
  const location = path.join(__dirname, 'fixtures', 'easy-legacy-puzzles.json');
  assert.ok(fs.existsSync(location), 'legacy preservation fixture must be supplied');
  const legacy = JSON.parse(fs.readFileSync(location, 'utf8'));
  assert.equal(Object.keys(legacy).length, 13);
  const bank = readBank();
  for (const [id, original] of Object.entries(legacy)) {
    const current = bank.find(({ puzzle }) => puzzle.id === id);
    assert.equal(current.set, 'easy-1', id);
    assert.deepEqual(current.puzzle.lines, original.lines, id);
    assert.deepEqual(current.puzzle.solutions, original.solutions, id);
  }
});
