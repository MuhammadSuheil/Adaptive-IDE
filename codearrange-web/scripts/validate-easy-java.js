#!/usr/bin/env node
'use strict';

// Maintainer check only: creates Java sources in an OS temporary directory.
// Expected stdout is hand-calculated in tests/fixtures, not inferred from Java.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const root = path.join(__dirname, '..');

function normalizeOutput(text) {
  return text.replace(/\r\n/g, '\n').split('\n').map(line => line.replace(/[ \t]+$/, '')).join('\n');
}

function javaString(value) {
  return JSON.stringify(value).replace(/\u2028/g, '\\u2028').replace(/\u2029/g, '\\u2029');
}

function codeBlock(description) {
  const fence = String.fromCharCode(96).repeat(3);
  const match = description.match(new RegExp(fence + '(?:java)?\\s*([\\s\\S]*?)' + fence));
  assert.ok(match, 'A Java description block is required');
  return match[1].trim();
}

function validationSource(bank, cases) {
  const classes = [];
  const checks = [];
  let index = 0;
  function addProgram(source, label, expected, shouldFail = false, probe = '') {
    const name = 'EasyCheck' + index++;
    assert.match(source, /\bpublic\s+class\s+([A-Za-z_]\w*)/, label);
    source = source.replace(/\bpublic\s+class\s+[A-Za-z_]\w*/, 'class ' + name);
    if (probe) {
      const closing = source.lastIndexOf('}', source.lastIndexOf('}') - 1);
      assert.ok(closing >= 0, label);
      source = source.slice(0, closing) + probe + '\n' + source.slice(closing);
    }
    classes.push(source);
    checks.push('        check(' + javaString(label) + ', () -> ' + name + '.main(new String[0]), ' + javaString(expected) + ', ' + shouldFail + ');');
  }
  for (const puzzle of bank) {
    const expected = cases[puzzle.id];
    assert.ok(expected, 'Missing expectation: ' + puzzle.id);
    if (puzzle.mode === 'output') {
      addProgram(codeBlock(puzzle.description), puzzle.id + '/output', expected.stdout);
      continue;
    }
    for (const [solutionIndex, order] of puzzle.solutions.entries()) {
      const source = order.map(id => puzzle.lines.find(line => line.id === id).code).join('\n');
      addProgram(source, puzzle.id + '/solution-' + (solutionIndex + 1), expected.stdout, false, expected.probe || '');
    }
    if (puzzle.mode === 'debug') {
      addProgram(codeBlock(puzzle.description), puzzle.id + '/buggy', expected.stdout, true);
    }
  }
  return {
    count: index,
    source: [
      'import java.io.*;',
      'import java.nio.charset.StandardCharsets;',
      'public class EasyValidation {',
      '    static int checked = 0;',
      '    static String normalize(String value) {',
      '        return value.replace("\\r\\n", "\\n").replaceAll("[ \\\\t]+(?=\\\\n|$)", "");',
      '    }',
      '    static void check(String label, Runnable program, String expected, boolean shouldFail) {',
      '        PrintStream previous = System.out;',
      '        ByteArrayOutputStream bytes = new ByteArrayOutputStream();',
      '        Throwable failure = null;',
      '        try {',
      '            System.setOut(new PrintStream(bytes, true, StandardCharsets.UTF_8));',
      '            program.run();',
      '        } catch (Throwable error) { failure = error; }',
      '        finally { System.setOut(previous); }',
      '        String actual = normalize(bytes.toString(StandardCharsets.UTF_8));',
      '        boolean matches = failure == null && actual.equals(expected);',
      '        if (shouldFail ? matches : !matches) {',
      '            throw new AssertionError(label + ": expected=" + expected.replace("\\n", "|")',
      '                + " actual=" + actual.replace("\\n", "|") + " error=" + failure);',
      '        }',
      '        checked++;',
      '    }',
      '    public static void main(String[] args) {',
      ...checks,
      '        System.out.println("Java checks passed: " + checked);',
      '    }',
      '}',
      ...classes,
    ].join('\n'),
  };
}

function run() {
  const bank = ['easy-1', 'easy-2', 'easy-3', 'easy-4'].flatMap(set => {
    const directory = path.join(root, 'puzzles', set);
    return fs.readdirSync(directory).filter(file => file.endsWith('.json')).sort()
      .map(file => JSON.parse(fs.readFileSync(path.join(directory, file), 'utf8')));
  });
  assert.equal(bank.length, 100);
  const cases = JSON.parse(fs.readFileSync(path.join(root, 'tests', 'fixtures', 'easy-puzzle-cases.json'), 'utf8'));
  const validation = validationSource(bank, cases);
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'codearrange-java-'));
  try {
    const source = path.join(directory, 'EasyValidation.java');
    fs.writeFileSync(source, validation.source, 'utf8');
    for (const [command, args] of [
      ['javac', ['-encoding', 'UTF-8', '-d', directory, source]],
      ['java', ['-cp', directory, 'EasyValidation']],
    ]) {
      const result = spawnSync(command, args, { encoding: 'utf8', timeout: 60000, maxBuffer: 4 * 1024 * 1024, windowsHide: true });
      if (result.error) throw result.error;
      assert.equal(result.status, 0, result.stderr || result.stdout || (command + ' failed'));
      if (result.stdout) process.stdout.write(result.stdout);
    }
    console.log('Validated ' + bank.length + ' puzzles and ' + validation.count + ' Java executions (accepted orders and buggy controls).');
  } finally {
    // Only remove the exact OS-created test directory; never a project folder.
    fs.rmSync(directory, { recursive: true, force: true });
  }
}

if (require.main === module) run();
module.exports = { normalizeOutput, codeBlock, validationSource };
