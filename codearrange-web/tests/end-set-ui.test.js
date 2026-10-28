'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

// Minimal DOM boundary: run the actual app handlers without a browser dependency.
class Element {
  constructor() {
    this.children = []; this.dataset = {}; this.listeners = {}; this.className = '';
    this.disabled = false; this.value = ''; this.textContent = '';
    const update = fn => { const values = new Set(this.className.split(/\s+/).filter(Boolean)); fn(values); this.className = [...values].join(' '); };
    this.classList = {
      add: (...values) => update(set => values.forEach(value => set.add(value))),
      remove: (...values) => update(set => values.forEach(value => set.delete(value))),
      contains: value => this.className.split(/\s+/).includes(value),
      toggle: (value, force) => update(set => (force ?? !set.has(value)) ? set.add(value) : set.delete(value)),
    };
  }
  set innerHTML(_value) { this.children = []; }
  setAttribute(key, value) { this[key] = value; }
  appendChild(child) {
    if (child.parentElement) child.parentElement.children = child.parentElement.children.filter(item => item !== child);
    this.children.push(child); child.parentElement = this; return child;
  }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  async click() {
    if (this.disabled) return;
    await Promise.all((this.listeners.click || []).map(fn => fn({ target: this, preventDefault() {} })));
  }
  querySelectorAll() { return this.children; }
}

const flush = () => new Promise(resolve => setImmediate(resolve));
async function appFixture({ confirm = true, review = false, storage = new Map(), serverState, intercept } = {}) {
  const elements = new Map();
  const html = fs.readFileSync(path.join(__dirname, '..', 'public', 'index.html'), 'utf8');
  for (const match of html.matchAll(/<[a-z]+\b([^>]*\bid="([^"]+)"[^>]*)>/g)) {
    const element = new Element();
    element.className = match[1].match(/class="([^"]+)"/)?.[1] || '';
    element.disabled = /\bdisabled\b/.test(match[1]);
    elements.set(match[2], element);
  }
  const state = serverState || { setSessionId: 'set-1', setSessionStatus: 'idle',
    sessionStartedAtMs: null, researchSetOpen: false, researchStudyEnded: false };
  const calls = [], intervals = new Set();
  let generation = 1;
  const json = (data, status = 200) => ({ ok: status < 400, status, json: async () => structuredClone(data) });
  async function fetch(route, options = {}) {
    const body = options.body ? JSON.parse(options.body) : {};
    calls.push({ route, body });
    const custom = intercept && await intercept({ route, body, state, json });
    if (custom) return custom;
    if (route === '/api/config') return json({ playerName: 'Test', puzzleSet: 'demo', setConfig: {}, reviewMode: review, ...state });
    if (route === '/api/puzzle-sets') return json([{ key: 'demo', label: 'Demo' }]);
    if (route === '/api/session/select-set') {
      Object.assign(state, { setSessionId: `set-${++generation}`, setSessionStatus: 'idle', sessionStartedAtMs: null });
      return json({ puzzleSet: body.puzzleSet, setConfig: {}, ...state });
    }
    if (route === '/api/puzzles') return json([{ id: 'q1', title: 'One' }, { id: 'q2', title: 'Two' }]);
    if (route.startsWith('/api/puzzles/q1') && !route.includes('/check')) {
      Object.assign(state, { setSessionStatus: 'active', sessionStartedAtMs: Date.now() });
      return json({ id: 'q1', title: 'One', description: 'Arrange.', lines: [{ id: 1, code: 'class A {}' }], ...state });
    }
    if (route === '/api/session/end-set') {
      Object.assign(state, { setSessionStatus: 'ended', sessionStartedAtMs: null, researchSetOpen: false });
      return json({ ...state, outcome: body.outcome || 'stopped_manually' });
    }
    if (route.includes('/check')) return json({ correct: false, perLine: {}, correctPositions: 0, total: 1, wrongLines: 1 });
    if (route.includes('/rating')) return json({ ok: true });
    if (route === '/api/session/reset') return json({ ok: true });
    throw new Error(`Unexpected request: ${route}`);
  }
  const context = vm.createContext({
    document: { getElementById: id => elements.get(id), createElement: () => new Element(),
      documentElement: { dataset: {} }, querySelector: () => null },
    window: { confirm: () => confirm }, localStorage: { getItem: () => null, setItem() {} },
    sessionStorage: { getItem: key => storage.get(key) || null, setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key) },
    fetch, console, Date, setInterval: fn => { intervals.add(fn); return fn; }, clearInterval: fn => intervals.delete(fn),
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '..', 'public', 'app.js'), 'utf8'), context);
  await flush();
  async function start() { elements.get('set-select').value = 'demo'; await elements.get('start-btn').click(); await flush(); }
  return { elements, calls, state, storage, intervals, start };
}

test('standalone manual end clears draft, stops timer, and shows menu success', async () => {
  const f = await appFixture(); await f.start();
  assert.equal(f.elements.get('end-study-puzzle-btn').disabled, false);
  await f.elements.get('end-study-puzzle-btn').click();
  assert.equal(f.elements.get('puzzle-screen').classList.contains('hidden'), true);
  assert.equal(f.elements.get('start-screen').classList.contains('hidden'), false);
  assert.match(f.elements.get('start-error').textContent, /Set telah diakhiri/);
  assert.equal(f.storage.has('codeArrange:puzzleState'), false);
  assert.equal(f.intervals.size, 0);
  assert.equal(f.calls.filter(call => call.route.includes('/check')).length, 0);
});

test('cancel leaves the puzzle and timer running', async () => {
  const f = await appFixture({ confirm: false }); await f.start();
  assert.equal(f.elements.get('end-study-puzzle-btn').disabled, false);
  await f.elements.get('end-study-puzzle-btn').click();
  assert.equal(f.calls.filter(call => call.route === '/api/session/end-set').length, 0);
  assert.equal(f.elements.get('puzzle-screen').classList.contains('hidden'), false);
  assert.equal(f.intervals.size, 1);
});

test('failed close keeps draft and displays its error on the puzzle screen', async () => {
  const f = await appFixture({ intercept: ({ route, json }) => route === '/api/session/end-set'
    ? json({ error: 'Disk unavailable' }, 500) : undefined });
  await f.start();
  await f.elements.get('available-list').children[0].click();
  await f.elements.get('end-study-puzzle-btn').click();
  assert.equal(f.elements.get('puzzle-screen').classList.contains('hidden'), false);
  assert.match(f.elements.get('set-session-feedback').textContent, /Disk unavailable/);
  assert.equal(f.elements.get('set-session-feedback').classList.contains('hidden'), false);
  assert.deepEqual(JSON.parse(f.storage.get('codeArrange:puzzleState')).solutionOrder, [1]);
  assert.equal(f.elements.get('end-study-puzzle-btn').disabled, false);
});

test('lost close response is reconciled from server without a second end event', async () => {
  const f = await appFixture({ intercept: ({ route, state }) => {
    if (route === '/api/session/end-set') {
      Object.assign(state, { setSessionStatus: 'ended', sessionStartedAtMs: null, researchSetOpen: false });
      throw new Error('Response lost');
    }
  } });
  await f.start(); await f.elements.get('end-study-puzzle-btn').click();
  assert.equal(f.elements.get('start-screen').classList.contains('hidden'), false);
  assert.equal(f.calls.filter(call => call.route === '/api/session/end-set').length, 1);
});

test('refresh after server close never resumes an old saved puzzle', async () => {
  const first = await appFixture(); await first.start();
  const second = await appFixture({ storage: first.storage, serverState: {
    ...first.state, setSessionStatus: 'ended', sessionStartedAtMs: null,
  } });
  assert.equal(second.storage.has('codeArrange:puzzleState'), false);
  assert.equal(second.elements.get('start-screen').classList.contains('hidden'), false);
  assert.equal(second.calls.some(call => call.route === '/api/puzzles'), false);
});

test('end button is disabled during puzzle loading and submission', async () => {
  let resolveLoad, resolveGrade;
  const loading = new Promise(resolve => { resolveLoad = resolve; });
  const grading = new Promise(resolve => { resolveGrade = resolve; });
  const f = await appFixture({ intercept: async ({ route }) => {
    if (route.startsWith('/api/puzzles/q1') && !route.includes('/check')) await loading;
    if (route.includes('/check')) await grading;
  } });
  const start = f.start(); await flush();
  assert.equal(f.elements.get('end-study-puzzle-btn').disabled, true);
  resolveLoad(); await start;
  assert.equal(f.elements.get('end-study-puzzle-btn').disabled, false);
  await f.elements.get('finish-btn').click(); await flush();
  assert.equal(f.elements.get('end-study-puzzle-btn').disabled, true);
  resolveGrade(); await flush();
  assert.equal(f.elements.get('end-study-puzzle-btn').disabled, false);
});

test('rating modal blocks manual end until a rating is saved', async () => {
  const f = await appFixture({ review: true, intercept: ({ route, json }) => route.includes('/check')
    ? json({ correct: true, perLine: {}, correctPositions: 1, total: 1, wrongLines: 0, attemptId: 'a1' }) : undefined });
  await f.start(); await f.elements.get('finish-btn').click(); await flush();
  assert.equal(f.elements.get('rating-modal').classList.contains('hidden'), false);
  assert.equal(f.elements.get('end-study-puzzle-btn').disabled, true);
  await f.elements.get('rating-buttons').children[0].click(); await flush();
  assert.equal(f.elements.get('rating-modal').classList.contains('hidden'), true);
  assert.equal(f.elements.get('end-study-puzzle-btn').disabled, false);
});

test('unknown close outcome can be retried, then continued if server confirms active', async () => {
  let attempts = 0;
  const f = await appFixture({ intercept: ({ route, json }) => {
    if (route === '/api/session/end-set') {
      if (++attempts === 1) throw new Error('Offline');
      return json({ error: 'Disk unavailable' }, 500);
    }
    if (route === '/api/config' && attempts === 1) throw new Error('Offline');
  } });
  await f.start(); await f.elements.get('end-study-puzzle-btn').click();
  assert.equal(f.elements.get('finish-btn').disabled, true);
  assert.equal(f.elements.get('end-study-puzzle-btn').disabled, false);
  await f.elements.get('end-study-puzzle-btn').click();
  assert.equal(f.elements.get('finish-btn').disabled, false);
  assert.equal(f.elements.get('puzzle-screen').classList.contains('hidden'), false);
});

test('refresh after normal completion still presents the pending final rating', async () => {
  const f = await appFixture({ review: true, serverState: { setSessionId: 'set-1',
    setSessionStatus: 'ended', sessionStartedAtMs: null, pendingReviewAttemptId: 'a-final' } });
  assert.equal(f.elements.get('rating-modal').classList.contains('hidden'), false);
  assert.equal(f.elements.get('start-btn').disabled, true);
  await f.elements.get('rating-buttons').children[0].click(); await flush();
  assert.equal(f.elements.get('rating-modal').classList.contains('hidden'), true);
  assert.equal(f.elements.get('start-btn').disabled, false);
});

test('failed rating stays visible and can be retried before manual end', async () => {
  let ratings = 0;
  const f = await appFixture({ review: true, intercept: ({ route, json }) => {
    if (route.includes('/check')) return json({ correct: true, perLine: {}, correctPositions: 1,
      total: 1, wrongLines: 0, attemptId: 'a1' });
    if (route.includes('/rating') && ++ratings === 1) return json({ error: 'Rating not saved' }, 500);
  } });
  await f.start(); await f.elements.get('finish-btn').click(); await flush();
  await f.elements.get('rating-buttons').children[0].click();
  assert.equal(f.elements.get('rating-modal').classList.contains('hidden'), false);
  assert.match(f.elements.get('rating-feedback').textContent, /Rating not saved/);
  assert.equal(f.elements.get('end-study-puzzle-btn').disabled, true);
  await f.elements.get('rating-buttons').children[0].click();
  assert.equal(f.elements.get('rating-modal').classList.contains('hidden'), true);
  assert.equal(f.elements.get('end-study-puzzle-btn').disabled, false);
});

test('double click and tile edits are blocked while manual close is in flight', async () => {
  let release;
  const waiting = new Promise(resolve => { release = resolve; });
  const f = await appFixture({ intercept: async ({ route }) => {
    if (route === '/api/session/end-set') await waiting;
  } });
  await f.start();
  const closing = f.elements.get('end-study-puzzle-btn').click(); await flush();
  assert.equal(f.elements.get('end-study-puzzle-btn').disabled, true);
  await f.elements.get('end-study-puzzle-btn').click();
  await f.elements.get('available-list').children[0].click();
  assert.equal(f.elements.get('solution-list').children.length, 0);
  release(); await closing;
  assert.equal(f.calls.filter(call => call.route === '/api/session/end-set').length, 1);
});

test('refresh during an uncertain manual close preserves the draft for retry', async () => {
  let closing = false;
  const first = await appFixture({ intercept: ({ route }) => {
    if (route === '/api/session/end-set') { closing = true; throw new Error('Offline'); }
    if (closing && route === '/api/config') throw new Error('Offline');
  } });
  await first.start();
  await first.elements.get('available-list').children[0].click();
  await first.elements.get('end-study-puzzle-btn').click();
  const second = await appFixture({ storage: first.storage, serverState: { ...first.state } });
  assert.equal(second.elements.get('puzzle-screen').classList.contains('hidden'), false);
  assert.equal(second.elements.get('solution-list').children.length, 1);
  assert.equal(second.elements.get('finish-btn').disabled, true);
  assert.equal(second.elements.get('end-study-puzzle-btn').disabled, false);
  await second.elements.get('end-study-puzzle-btn').click();
  assert.equal(second.elements.get('start-screen').classList.contains('hidden'), false);
});
