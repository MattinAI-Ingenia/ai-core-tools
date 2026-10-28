const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const ts = require('typescript');

const source = fs.readFileSync(path.join(__dirname, '../src/utils/buildCitedGraph.ts'), 'utf8');
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS },
}).outputText;
const helpers = {};
vm.runInNewContext(compiled, { exports: helpers });
const { buildCitedGraph, numberingFromContent } = helpers;

const graph = {
  data: {
    entities: [
      { id: 'e1', name: 'Alpha', source_id: 'c1<SEP>c2' },
      { id: 'e2', name: 'Beta', source_id: 'c3' },
      // Only reachable as the endpoint of a cited relationship, not from a cited chunk
      { id: 'e3', name: 'Gamma', source_id: 'c9' },
    ],
    relationships: [
      { id: 'r1', source: 'e1', target: 'e3', source_id: 'c2' },
      { id: 'r2', source: 'e2', target: 'e3', source_id: 'c3<SEP>c4' },
    ],
    chunks: [
      { id: 'c1', content: 'first' },
      { id: 'c2', content: 'second' },
      { id: 'c3', content: 'third' },
    ],
  },
};

test('keeps only cited chunks, in first-citation order, numbered 1..X', () => {
  const cited = buildCitedGraph(graph, ['c2', 'c1']);
  assert.ok(cited);
  assert.deepEqual(cited.graph.data.chunks.map((c) => c.id), ['c2', 'c1']);
  assert.deepEqual(cited.chunkNumbers, [1, 2]);
});

test('keeps entities/relationships from cited chunks plus cited-edge endpoints, drops the rest', () => {
  const cited = buildCitedGraph(graph, ['c2']);
  assert.ok(cited);
  assert.deepEqual(cited.graph.data.entities.map((e) => e.id), ['e1', 'e3']);
  assert.deepEqual(cited.graph.data.relationships.map((r) => r.id), ['r1']);
});

test('returns null with no citations or only unknown chunk ids', () => {
  assert.equal(buildCitedGraph(graph, []), null);
  assert.equal(buildCitedGraph(graph, ['nope']), null);
});

test('dedupes a chunk cited at several positions down to its first copy and number', () => {
  const repeated = {
    data: {
      entities: [],
      relationships: [],
      chunks: [
        { id: 'c1', content: 'a' },
        { id: 'c2', content: 'b' },
        { id: 'c1', content: 'a again' },
      ],
    },
  };
  const cited = buildCitedGraph(repeated, ['c1']);
  assert.ok(cited);
  assert.equal(cited.graph.data.chunks.length, 1);
  assert.deepEqual(cited.chunkNumbers, [1]);
});

const payloadChunks = [
  { id: 'c1' }, { id: 'c2' }, { id: 'c3' }, { id: 'c4' },
  { id: 'c1' }, // duplicate copy of the first chunk at payload position 5
];

test('renumbers cited sources 1..X in first-citation order, sharing numbers for repeats', () => {
  // The LLM cites payload positions 3, 1 and 5 (1-based) — the latter is a
  // duplicate copy of the chunk at position 1.
  const content = 'a [3](cite://3) b [1](cite://1) c [5](cite://5)';
  const { citedChunkIds, numbersByCite, numberByChunkId } = numberingFromContent(content, payloadChunks);
  // Round-trip through JSON: values built inside the vm realm fail deepEqual.
  const plain = JSON.parse(JSON.stringify({
    citedChunkIds,
    byCite: [...numbersByCite.entries()],
    byChunk: [...numberByChunkId.entries()],
  }));
  assert.deepEqual(plain.citedChunkIds, ['c3', 'c1']);
  assert.deepEqual(plain.byCite, [[3, 1], [1, 2], [5, 2]]);
  assert.deepEqual(plain.byChunk, [['c3', 1], ['c1', 2]]);
});

test('leaves unresolvable citations out of the numbering', () => {
  const content = 'x [99](cite://99) y [2](cite://2)';
  const { citedChunkIds, numbersByCite } = numberingFromContent(content, payloadChunks);
  const plain = JSON.parse(JSON.stringify({
    citedChunkIds,
    byCite: [...numbersByCite.entries()],
  }));
  assert.deepEqual(plain.citedChunkIds, ['c2']);
  assert.deepEqual(plain.byCite, [[2, 1]]);
});

test('empty content yields an empty numbering', () => {
  const numbering = numberingFromContent('', payloadChunks);
  assert.deepEqual(JSON.parse(JSON.stringify(numbering.citedChunkIds)), []);
  assert.equal(numbering.numbersByCite.size, 0);
});
