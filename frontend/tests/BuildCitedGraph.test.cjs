const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const ts = require('typescript');

// ES5 output would spread a Set as an empty array (no downlevelIteration), hence ES2020.
function load(file) {
  const out = {};
  vm.runInNewContext(
    ts.transpileModule(fs.readFileSync(path.join(__dirname, '../src', file), 'utf8'), {
      compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
    }).outputText,
    { exports: out },
  );
  return out;
}
const { buildCitedGraph, numberingFromContent } = load('utils/buildCitedGraph.ts');
const { splitGraphBySilo } = load('utils/splitGraphBySilo.ts');

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
  assert.deepEqual(cited.data.chunks.map((c) => c.id), ['c2', 'c1']);
});

test('keeps entities/relationships from cited chunks plus cited-edge endpoints, drops the rest', () => {
  const cited = buildCitedGraph(graph, ['c2']);
  assert.ok(cited);
  assert.deepEqual(cited.data.entities.map((e) => e.id), ['e1', 'e3']);
  assert.deepEqual(cited.data.relationships.map((r) => r.id), ['r1']);
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
  assert.equal(cited.data.chunks.length, 1);
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

// Two silos that both hold an entity called "Ikerlan". Entity ids are the name,
// so the two must stay separate nodes once the graph is split per silo.
const twoSilos = {
  data: {
    entities: [
      { id: 'Ikerlan', name: 'Ikerlan', source_id: 'res1-chunk-000', silo_id: 1, silo_name: 'Silo A' },
      { id: 'Unused', name: 'Unused', source_id: 'res1-chunk-009', silo_id: 1, silo_name: 'Silo A' },
      { id: 'Ikerlan', name: 'Ikerlan', source_id: 'res2-chunk-000', silo_id: 2, silo_name: 'Silo B' },
      { id: 'Other', name: 'Other', source_id: 'res2-chunk-009', silo_id: 2, silo_name: 'Silo B' },
    ],
    relationships: [
      { id: 'ra', source: 'Ikerlan', target: 'Unused', source_id: 'res1-chunk-000', silo_id: 1, silo_name: 'Silo A' },
      { id: 'rb', source: 'Ikerlan', target: 'Other', source_id: 'res2-chunk-000', silo_id: 2, silo_name: 'Silo B' },
    ],
    chunks: [
      { id: 'res1-chunk-000', content: 'a', silo_id: 1, silo_name: 'Silo A' },
      { id: 'res1-chunk-009', content: 'c', silo_id: 1, silo_name: 'Silo A' }, // not cited
      { id: 'res2-chunk-000', content: 'b', silo_id: 2, silo_name: 'Silo B' },
    ],
  },
};

// Round-trip values built inside the vm realm before comparing them.
const plain = (value) => JSON.parse(JSON.stringify(value));

test('multi-silo: the cited graph keeps the cited chunks of both silos and their entities', () => {
  const cited = buildCitedGraph(twoSilos, ['res2-chunk-000', 'res1-chunk-000']);
  assert.ok(cited);
  assert.deepEqual(plain(cited.data.chunks.map((c) => c.id)), ['res2-chunk-000', 'res1-chunk-000']);
  assert.deepEqual(
    plain(cited.data.entities.map((e) => `${e.silo_id}:${e.id}`)),
    ['1:Ikerlan', '1:Unused', '2:Ikerlan', '2:Other'],
  );
  assert.deepEqual(plain(cited.data.relationships.map((r) => r.id)), ['ra', 'rb']);
});

test('multi-silo: splitting the cited graph gives one group per silo with only its own parts', () => {
  const cited = buildCitedGraph(twoSilos, ['res2-chunk-000', 'res1-chunk-000']);
  const groups = plain(splitGraphBySilo(cited).map((g) => ({
    siloId: g.siloId,
    siloName: g.siloName,
    entities: g.graph.data.entities.map((e) => e.id),
    relationships: g.graph.data.relationships.map((r) => r.id),
    chunks: g.graph.data.chunks.map((c) => c.id),
  })));
  assert.deepEqual(groups, [
    { siloId: 1, siloName: 'Silo A', entities: ['Ikerlan', 'Unused'], relationships: ['ra'], chunks: ['res1-chunk-000'] },
    { siloId: 2, siloName: 'Silo B', entities: ['Ikerlan', 'Other'], relationships: ['rb'], chunks: ['res2-chunk-000'] },
  ]);
});

test('multi-silo: each silo group keeps the global citation numbers of its chunks', () => {
  // Payload positions: 1 = res1-chunk-000, 2 = res1-chunk-009, 3 = res2-chunk-000.
  // The answer cites position 3 first, then position 1.
  const { numberByChunkId } = numberingFromContent('a [3](cite://3) b [1](cite://1)', twoSilos.data.chunks);
  const cited = buildCitedGraph(twoSilos, ['res2-chunk-000', 'res1-chunk-000']);
  const numbers = plain(splitGraphBySilo(cited).map((g) =>
    g.graph.data.chunks.map((c) => numberByChunkId.get(c.id)),
  ));
  assert.deepEqual(numbers, [[2], [1]]);
});
