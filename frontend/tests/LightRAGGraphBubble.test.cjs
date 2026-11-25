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
    {
      exports: out,
      // Relative imports resolve against the loaded module's own directory.
      require: (rel) => load(path.join(path.dirname(file), rel).replace(/\.ts$/, '') + '.ts'),
    },
  );
  return out;
}

const { bubbleNodeFill, bubbleNodeIcon } = load('utils/bubbleGraphStyle.ts');
const { colorForType, PARTIAL_COLOR, NEUTRAL_COLOR, RING_CITED, RING_RETRIEVED } = load('utils/entityTypeColors.ts');

const cited = new Set(['c1']);

test('partial nodes keep their gray fill and never get a ring', () => {
  const entity = { id: 'p1', partial: true, entity_type: 'Machine' };
  assert.equal(bubbleNodeFill(entity, true), PARTIAL_COLOR);
  assert.equal(bubbleNodeIcon(entity, cited, true), undefined);
});

test('coloring on: fill is the entity type color', () => {
  const entity = { id: 'e1', entity_type: 'Machine' };
  assert.equal(bubbleNodeFill(entity, true), colorForType('Machine'));
});

test('coloring on: cited nodes get the green ring, retrieved the indigo one', () => {
  const citedEntity = { id: 'e1', entity_type: 'Machine', source_id: 'c1' };
  const retrievedEntity = { id: 'e2', entity_type: 'Person', source_id: 'c9' };
  const citedIcon = bubbleNodeIcon(citedEntity, cited, true);
  const retrievedIcon = bubbleNodeIcon(retrievedEntity, cited, true);
  assert.ok(decodeURIComponent(citedIcon.url).includes(`stroke="${RING_CITED}"`));
  assert.ok(decodeURIComponent(retrievedIcon.url).includes(`stroke="${RING_RETRIEVED}"`));
});

test('coloring off: the old semantics come back — green fill when cited, indigo otherwise, no rings', () => {
  const citedEntity = { id: 'e1', entity_type: 'Machine', source_id: 'c1' };
  const retrievedEntity = { id: 'e2', entity_type: 'Person' };
  assert.equal(bubbleNodeFill(citedEntity, false, cited), RING_CITED);
  assert.equal(bubbleNodeFill(retrievedEntity, false, cited), NEUTRAL_COLOR);
  assert.equal(bubbleNodeIcon(citedEntity, cited, false), undefined);
  assert.equal(bubbleNodeIcon(retrievedEntity, cited, false), undefined);
});

test('entities without a type still color and ring (no crash, gray fill)', () => {
  const entity = { id: 'e3', source_id: 'c1' };
  assert.equal(bubbleNodeFill(entity, true), PARTIAL_COLOR);
  assert.ok(decodeURIComponent(bubbleNodeIcon(entity, cited, true).url).includes(`stroke="${RING_CITED}"`));
});

test('partial wins over coloring-off semantics (stays gray)', () => {
  const entity = { id: 'p1', partial: true, source_id: 'c1' };
  assert.equal(bubbleNodeFill(entity, false), PARTIAL_COLOR);
});
