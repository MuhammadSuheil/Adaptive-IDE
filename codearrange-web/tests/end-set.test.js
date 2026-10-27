'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const net = require('node:net');
const { spawn } = require('node:child_process');
const { once } = require('node:events');

async function fixture(t, review = false) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'codearrange-end-'));
  const sensor = path.join(root, 'sessions', 'sensor_test');
  await fs.mkdir(sensor, { recursive: true });
  await fs.copyFile(path.join(__dirname, '..', 'server.js'), path.join(root, 'server.js'));
  for (const set of ['demo', 'medium']) {
    await fs.mkdir(path.join(root, 'puzzles', set), { recursive: true });
    await fs.writeFile(path.join(root, 'puzzles', set, '1_q1.json'), JSON.stringify({
      id: 'q1', title: 'Example', description: 'Arrange the code.',
      lines: [{ id: 1, code: 'class A {}' }], solutions: [[1]],
    }));
  }
  const listener = net.createServer();
  listener.listen(0, '127.0.0.1');
  await once(listener, 'listening');
  const port = listener.address().port;
  await new Promise(resolve => listener.close(resolve));
  const child = spawn(process.execPath, [path.join(root, 'server.js'), 'Test', 'demo', ...(review ? ['--review'] : [])], {
    cwd: root, env: { ...process.env, PORT: String(port), CODEARRANGE_SESSIONS_ROOT: path.join(root, 'sessions') },
    stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true,
  });
  t.after(async () => {
    if (child.exitCode === null) { child.kill(); await once(child, 'exit'); }
    await fs.rm(root, { recursive: true, force: true });
  });
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Server startup timeout')), 5000);
    child.stdout.on('data', data => {
      if (String(data).includes('Code Arrange running')) { clearTimeout(timer); resolve(); }
    });
    child.once('error', error => { clearTimeout(timer); reject(error); });
    child.once('exit', code => { clearTimeout(timer); reject(new Error(`Server exited: ${code}`)); });
  });
  async function request(route, body) {
    const response = await fetch(`http://127.0.0.1:${port}${route}`, body === undefined ? {} : {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    const text = await response.text();
    let data;
    try { data = JSON.parse(text); } catch { data = { error: text }; }
    return { status: response.status, data };
  }
  const config = () => request('/api/config').then(r => r.data);
  const initial = await config();
  const event = (eventType, extra = {}) => request('/api/research/events', {
    eventType, setSessionId: initial.setSessionId, ...extra,
  });
  return { root, sensor, request, config, event, initial };
}

test('manual end without sensors stops only the active set, without grading', async t => {
  const f = await fixture(t);
  assert.equal(f.initial.setSessionStatus, 'idle');
  assert.ok(f.initial.setSessionId);
  await f.request(`/api/puzzles/q1?setSessionId=${f.initial.setSessionId}`);
  assert.equal((await f.config()).setSessionStatus, 'active');
  const ended = await f.request('/api/session/end-set', { setSessionId: f.initial.setSessionId, puzzleId: 'q1' });
  assert.equal(ended.status, 200);
  assert.equal(ended.data.setSessionStatus, 'ended');
  const config = await f.config();
  assert.equal(config.sessionStartedAtMs, null);
  assert.equal(config.sessionDeadlineMs, null);
  assert.equal(config.researchStudyEnded, false);
  assert.equal((await f.request('/api/puzzles/q1/check', { order: [1], setSessionId: f.initial.setSessionId })).status, 409);
  assert.equal((await f.request('/api/puzzles/q1')).status, 409);
  const attempts = await fs.readFile(path.join(f.root, 'logs', 'attempts.csv'), 'utf8')
    .catch(error => { if (error.code === 'ENOENT') return ''; throw error; });
  assert.ok(attempts.trim().split('\n').filter(Boolean).length <= 1);
  const next = await f.request('/api/session/select-set', { puzzleSet: 'medium' });
  assert.notEqual(next.data.setSessionId, f.initial.setSessionId);
  await f.request(`/api/puzzles/q1?setSessionId=${next.data.setSessionId}`);
  const retry = await f.request('/api/session/end-set', { setSessionId: f.initial.setSessionId });
  assert.deepEqual(retry.data, ended.data);
  assert.equal((await f.config()).setSessionId, next.data.setSessionId);
  assert.equal((await f.config()).setSessionStatus, 'active');
  assert.equal((await f.request(`/api/puzzles/q1?setSessionId=${f.initial.setSessionId}`)).status, 409);
});

test('invalid close requests do not alter an idle session', async t => {
  const f = await fixture(t);
  assert.equal((await f.request('/api/session/end-set', null)).status, 400);
  assert.equal((await f.request('/api/session/end-set', {})).status, 400);
  assert.equal((await f.request('/api/session/end-set', { setSessionId: f.initial.setSessionId })).status, 409);
  assert.equal((await f.config()).setSessionStatus, 'idle');
});

test('linked end writes one manual event and preserves existing attempts', async t => {
  const f = await fixture(t);
  await f.request('/api/research/session/link', { sessionId: 'sensor_test' });
  await f.request('/api/puzzles/q1');
  await f.event('set_started');
  await f.event('puzzle_started', { puzzleId: 'q1' });
  await f.request('/api/puzzles/q1/check', { order: [], setSessionId: f.initial.setSessionId });
  const attempts = await fs.readFile(path.join(f.root, 'logs', 'attempts.csv'), 'utf8');
  const body = { setSessionId: f.initial.setSessionId, puzzleId: 'q1' };
  const results = await Promise.all([f.request('/api/session/end-set', body), f.request('/api/session/end-set', body)]);
  assert.ok(results.every(r => r.status === 200));
  const events = await fs.readFile(path.join(f.sensor, 'task_events.csv'), 'utf8');
  assert.equal(events.split('\n').filter(row => row.includes(',set_ended,')).length, 1);
  assert.match(events, /stopped_manually/);
  assert.doesNotMatch(events, /,study_ended,/);
  assert.equal(await fs.readFile(path.join(f.root, 'logs', 'attempts.csv'), 'utf8'), attempts);
  assert.equal((await f.config()).researchSetOpen, false);
  assert.equal((await f.event('study_ended')).status, 200);
});

test('failed event write keeps the set and timer intact for retry', async t => {
  const f = await fixture(t);
  await f.request('/api/research/session/link', { sessionId: 'sensor_test' });
  await f.request('/api/puzzles/q1');
  await f.event('set_started');
  const before = await f.config();
  const eventPath = path.join(f.sensor, 'task_events.csv');
  await fs.rename(eventPath, eventPath + '.saved');
  await fs.mkdir(eventPath);
  assert.equal((await f.request('/api/session/end-set', { setSessionId: f.initial.setSessionId })).status, 500);
  const failed = await f.config();
  assert.equal(failed.setSessionStatus, 'active');
  assert.equal(failed.sessionStartedAtMs, before.sessionStartedAtMs);
  assert.equal(failed.researchSetOpen, true);
  await fs.rmdir(eventPath);
  await fs.rename(eventPath + '.saved', eventPath);
  assert.equal((await f.request('/api/session/end-set', { setSessionId: f.initial.setSessionId })).status, 200);
});

test('review pause blocks manual end until rating closes', async t => {
  const f = await fixture(t, true);
  await f.request('/api/puzzles/q1');
  const attempt = await f.request('/api/puzzles/q1/check', { order: [1], setSessionId: f.initial.setSessionId });
  assert.ok(attempt.data.attemptId);
  assert.equal((await f.request('/api/session/end-set', { setSessionId: f.initial.setSessionId })).status, 409);
  await f.request(`/api/attempts/${attempt.data.attemptId}/rating`, { rating: 5, setSessionId: f.initial.setSessionId });
  assert.equal((await f.request('/api/session/end-set', { setSessionId: f.initial.setSessionId })).status, 200);
});

for (const outcome of ['completed', 'timed_out']) {
  test(`${outcome} closes once while preserving the final rating flow`, async t => {
    const f = await fixture(t, true);
    await f.request('/api/research/session/link', { sessionId: 'sensor_test' });
    await f.request('/api/puzzles/q1');
    await f.event('set_started');
    const attempt = await f.request('/api/puzzles/q1/check', { order: [1],
      submissionType: outcome === 'timed_out' ? 'timeout' : 'manual', setSessionId: f.initial.setSessionId });
    const close = await f.request('/api/session/end-set', { setSessionId: f.initial.setSessionId, outcome });
    assert.equal(close.status, 200);
    const closed = await f.config();
    assert.equal(closed.setSessionStatus, 'ended');
    assert.equal(closed.pendingReviewAttemptId, attempt.data.attemptId);
    assert.equal((await f.event('rating_started', { attemptId: attempt.data.attemptId })).status, 200);
    await f.request(`/api/attempts/${attempt.data.attemptId}/rating`, { rating: 4, setSessionId: f.initial.setSessionId });
    assert.equal((await f.event('rating_ended')).status, 200);
    assert.equal((await f.event('set_ended', { outcome })).status, 200);
    const events = await fs.readFile(path.join(f.sensor, 'task_events.csv'), 'utf8');
    assert.equal(events.split('\n').filter(row => row.includes(',set_ended,')).length, 1);
    const next = await f.request('/api/session/select-set', { puzzleSet: 'medium' });
    assert.equal(next.status, 200);
    await f.request(`/api/puzzles/q1?setSessionId=${next.data.setSessionId}`);
    // A stale submission must not grade the new set, even if puzzle IDs match.
    assert.equal((await f.request('/api/puzzles/q1/check', { order: [1], setSessionId: f.initial.setSessionId })).status, 409);
  });
}
