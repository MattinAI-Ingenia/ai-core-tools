const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const ts = require('typescript');

function load(relPath) {
  const source = fs.readFileSync(path.join(__dirname, relPath), 'utf8');
  const compiled = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS },
  }).outputText;
  const helpers = {};
  vm.runInNewContext(compiled, { exports: helpers });
  return helpers;
}

const ports = load('../src/components/visual-editor/agent-canvas/agentPorts.ts');
const picker = load('../src/components/visual-editor/agent-canvas/portPicker.ts');
const { AGENT_PORTS } = ports;
const { piecesForPort } = picker;

const modelPort = AGENT_PORTS.find((def) => def.id === 'model');
const skillsPort = AGENT_PORTS.find((def) => def.id === 'skills');
const toolsPort = AGENT_PORTS.find((def) => def.id === 'tools');
const videoPort = AGENT_PORTS.find((def) => def.id === 'video');

const ai = (id, name, extra = {}) => ({ kind: 'ai_service', id, name, ...extra });
const connectedModel = [{ port: 'model', kind: 'ai_service', id: 1 }];

test('lists only pieces the port accepts', () => {
  const pieces = [ai(1, 'GPT'), { kind: 'silo', id: 2, name: 'Docs' }, { kind: 'skill', id: 3, name: 'Greeting' }];
  const entries = piecesForPort(modelPort, pieces, []);
  assert.deepEqual(entries.map((entry) => entry.piece.id), [1]);
});

test('excludes pieces already connected to that port, keeps ones on other ports', () => {
  const pieces = [ai(1, 'GPT'), ai(2, 'Mistral')];
  const entries = piecesForPort(modelPort, pieces, connectedModel);
  assert.deepEqual(entries.map((entry) => entry.piece.id), [2]);
  // The same AI service connected to Transcription is still offered for Model.
  const elsewhere = piecesForPort(modelPort, pieces, [
    { port: 'transcription', kind: 'ai_service', id: 1 },
  ]);
  assert.deepEqual(elsewhere.map((entry) => entry.piece.id), [1, 2]);
});

test('lists every accepted kind on a multi-kind port', () => {
  const pieces = [
    { kind: 'mcp', id: 1, name: 'Files' },
    { kind: 'agent', id: 2, name: 'Helper' },
    { kind: 'silo', id: 3, name: 'Docs' },
  ];
  const entries = piecesForPort(toolsPort, pieces, []);
  assert.deepEqual(entries.map((entry) => entry.piece.id), [1, 2]);
});

test('excludes the exact (kind, id) connected to a multi-kind port', () => {
  const pieces = [{ kind: 'skill', id: 1, name: 'A' }, { kind: 'skill', id: 2, name: 'B' }];
  const entries = piecesForPort(skillsPort, pieces, [{ port: 'skills', kind: 'skill', id: 2 }]);
  assert.deepEqual(entries.map((entry) => entry.piece.id), [1]);
});

test('keeps only video-capable AI services on the Video port', () => {
  const pieces = [ai(1, 'GPT'), ai(2, 'Vision Pro', { supportsVideo: true }), ai(3, 'Unknown')];
  const entries = piecesForPort(videoPort, pieces, []);
  assert.deepEqual(entries.map((entry) => entry.piece.id), [2]);
});

test('marks retired services disabled and keeps the rest enabled', () => {
  const pieces = [ai(1, 'Old', { lifecycleStatus: 'retired' }), ai(2, 'New', { lifecycleStatus: 'deprecated' })];
  const entries = piecesForPort(modelPort, pieces, []);
  assert.deepEqual(
    entries.map((entry) => [entry.piece.id, entry.disabled]),
    [[1, true], [2, false]],
  );
});
