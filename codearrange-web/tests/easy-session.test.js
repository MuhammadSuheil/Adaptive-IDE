'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const net = require('node:net');
const { spawn } = require('node:child_process');
const { once } = require('node:events');

const project = path.join(__dirname, '..');
const sets = ['easy-1', 'easy-2', 'easy-3', 'easy-4'];

async function session(t, review) {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'codearrange-easy-'));
  await fs.copyFile(path.join(project, 'server.js'), path.join(directory, 'server.js'));
  await fs.cp(path.join(project, 'puzzles'), path.join(directory, 'puzzles'), { recursive: true });
  await fs.mkdir(path.join(directory, 'sessions', 'sensor_mock'), { recursive: true });
  const listener = net.createServer();
  listener.listen(0, '127.0.0.1');
  await once(listener, 'listening');
  const port = listener.address().port;
  await new Promise(resolve => listener.close(resolve));
  const child = spawn(process.execPath, [path.join(directory, 'server.js'), 'Easy Test', 'easy-1', ...(review ? ['--review'] : [])], {
    cwd: directory,
    env: { ...process.env, PORT: String(port), CODEARRANGE_SESSIONS_ROOT: path.join(directory, 'sessions') },
    stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true,
  });
  t.after(async () => {
    if (child.exitCode === null) { child.kill(); await once(child, 'exit'); }
    await fs.rm(directory, { recursive: true, force: true });
  });
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Server startup timeout')), 5000);
    child.stdout.on('data', data => {
      if (String(data).includes('Code Arrange running')) { clearTimeout(timer); resolve(); }
    });
    child.once('error', error => { clearTimeout(timer); reject(error); });
    child.once('exit', code => { clearTimeout(timer); reject(new Error('Unexpected exit: ' + code)); });
  });
  async function request(route, body) {
    const response = await fetch('http://127.0.0.1:' + port + route, body === undefined ? {} : {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    const data = await response.json();
    assert.equal(response.status, 200, route + ': ' + JSON.stringify(data));
    return data;
  }
  return { directory, request };
}

test('grading accepts a legacy alternative and rejects wrong dependencies, output, and distractors', async t => {
  const f = await session(t, false);
  const selected = await f.request('/api/session/select-set', { puzzleSet: 'easy-1' });
  const originals = await Promise.all((await fs.readdir(path.join(project, 'puzzles', 'easy-1'))).map(async file =>
    JSON.parse(await fs.readFile(path.join(project, 'puzzles', 'easy-1', file), 'utf8'))));
  const check = async (id, order) => {
    await f.request('/api/puzzles/' + id);
    return f.request('/api/puzzles/' + id + '/check', { order, setSessionId: selected.setSessionId });
  };
  const variables = originals.find(puzzle => puzzle.id === 'variables');
  assert.equal((await check(variables.id, variables.solutions[1])).correct, true);
  assert.equal((await check(variables.id, [...variables.solutions[0]].reverse())).correct, false);
  const output = originals.find(puzzle => puzzle.id === 'print_2d_array');
  assert.equal((await check(output.id, [...output.solutions[0]].reverse())).correct, false);
  const debug = originals.find(puzzle => puzzle.id === 'fix_purchase_total');
  const wrong = debug.lines.find(line => !debug.solutions[0].includes(line.id));
  const correct = debug.lines.find(line => line.code.includes('int total ='));
  const wrongOrder = debug.solutions[0].map(id => id === correct.id ? wrong.id : id);
  assert.equal((await check(debug.id, wrongOrder)).correct, false);
});

test('independent logic is accepted but matching output cannot override explicit instructions', async t => {
  const location = path.join(__dirname, 'fixtures', 'easy-order-rubric.json');
  assert.ok(await fs.access(location).then(() => true, () => false), 'order rubric must be supplied');
  const rubric = JSON.parse(await fs.readFile(location, 'utf8'));
  const f = await session(t, false);
  for (const example of rubric) {
    const selected = await f.request('/api/session/select-set', { puzzleSet: example.set });
    await f.request('/api/puzzles/' + example.puzzleId);
    const result = await f.request('/api/puzzles/' + example.puzzleId + '/check', {
      order: example.order, setSessionId: selected.setSessionId,
    });
    assert.equal(result.correct, example.correct, example.note);
    await f.request('/api/session/end-set', { setSessionId: selected.setSessionId });
  }
});

for (const review of [false, true]) {
  test('all 100 easy exercises work without restart ' + (review ? 'with ratings and a sensor ledger' : 'without sensors'), async t => {
    const f = await session(t, review);
    const available = await f.request('/api/puzzle-sets');
    assert.deepEqual(available.filter(set => sets.includes(set.key)).map(set => [set.key, set.puzzleCount]),
      sets.map(set => [set, 25]));
    if (review) await f.request('/api/research/session/link', { sessionId: 'sensor_mock' });
    let previousSessionId;
    for (const set of sets) {
      const selected = await f.request('/api/session/select-set', { puzzleSet: set });
      assert.notEqual(selected.setSessionId, previousSessionId);
      previousSessionId = selected.setSessionId;
      const setSessionId = selected.setSessionId;
      const event = (eventType, extra = {}) => f.request('/api/research/events', { eventType, setSessionId, ...extra });
      assert.equal(selected.setConfig.difficultyLevel, 'easy');
      const list = await f.request('/api/puzzles');
      const files = (await fs.readdir(path.join(project, 'puzzles', set))).filter(file => file.endsWith('.json')).sort();
      const originals = await Promise.all(files.map(async file => JSON.parse(await fs.readFile(path.join(project, 'puzzles', set, file), 'utf8'))));
      assert.deepEqual(list.map(puzzle => puzzle.id), originals.map(puzzle => puzzle.id));
      for (const [index, puzzle] of originals.entries()) {
        const detail = await f.request('/api/puzzles/' + puzzle.id + '?setSessionId=' + setSessionId);
        assert.equal(detail.id, puzzle.id);
        assert.equal(detail.solutions, undefined);
        if (index === 0 && review) await event('set_started');
        if (review) await event('puzzle_started', { puzzleId: puzzle.id });
        const beforeRefresh = await f.request('/api/config');
        const afterRefresh = await f.request('/api/config');
        assert.equal(afterRefresh.setSessionId, setSessionId);
        assert.equal(afterRefresh.sessionStartedAtMs, beforeRefresh.sessionStartedAtMs);
        assert.equal(afterRefresh.sessionDeadlineMs, null, 'easy must not acquire an exam deadline');
        const body = { order: puzzle.solutions[0], setSessionId, submissionType: 'manual', timeMs: 360000 };
        const result = await f.request('/api/puzzles/' + puzzle.id + '/check', body);
        assert.equal(result.correct, true, set + '/' + puzzle.id);
        if (review) {
          await event('rating_started', { puzzleId: puzzle.id, attemptId: result.attemptId });
          await f.request('/api/attempts/' + result.attemptId + '/rating', { rating: 2, setSessionId });
          await event('rating_ended', { puzzleId: puzzle.id, attemptId: result.attemptId });
        }
      }
      const ended = await f.request('/api/session/end-set', { setSessionId, outcome: 'completed' });
      assert.equal(ended.setSessionStatus, 'ended');
      assert.deepEqual(await f.request('/api/session/end-set', { setSessionId, outcome: 'completed' }), ended);
      const setLog = await fs.readFile(path.join(f.directory, 'logs', set + '.csv'), 'utf8');
      assert.equal(setLog.trim().split('\n').length, 26, set);
    }
    const master = await fs.readFile(path.join(f.directory, 'logs', 'attempts.csv'), 'utf8');
    assert.equal(master.trim().split('\n').length, 101);
    if (review) {
      await f.request('/api/research/events', { eventType: 'study_ended', setSessionId: previousSessionId });
      const ledger = await fs.readFile(path.join(f.directory, 'sessions', 'sensor_mock', 'task_events.csv'), 'utf8');
      const rows = ledger.trim().split('\n').slice(1).map(line => line.split(','));
      assert.equal(rows.filter(row => row[3] === 'set_ended').length, 4);
      assert.equal(rows.filter(row => row[3] === 'puzzle_started').length, 100);
      assert.equal(rows.filter(row => row[3] === 'attempt_submitted').length, 100);
      assert.equal(rows.filter(row => row[3] === 'rating_started').length, 100);
      assert.equal(rows.filter(row => row[3] === 'rating_ended').length, 100);
      for (const row of rows) assert.equal(row[8], 'easy', 'all four sets must carry the easy research label');
    }
  });
}
