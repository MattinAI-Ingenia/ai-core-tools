const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const ts = require('typescript');

// The entity-types dialog must use the shared Modal: it opens on top of the
// ingestion-confirm modal, so it has to join the modal stack (z-index by
// depth, Escape, focus trap). A hand-rolled fixed overlay gets painted under
// the confirm modal and the flow silently dead-ends.
const source = fs.readFileSync(
  path.join(__dirname, '../src/components/forms/EntityTypeInferenceModal.tsx'),
  'utf8',
);
const compiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.CommonJS,
    jsx: ts.JsxEmit.ReactJSX,
    esModuleInterop: true,
  },
}).outputText;

function load() {
  const exports = {};
  let instance = null;
  let hookIndex = 0;

  const react = {
    useState(initial) {
      const index = hookIndex++;
      if (!(index in instance.states)) instance.states[index] = initial;
      return [instance.states[index], (v) => { instance.states[index] = typeof v === 'function' ? v(instance.states[index]) : v; }];
    },
    useRef(initial) {
      const index = hookIndex++;
      if (!(index in instance.refs)) instance.refs[index] = { current: initial };
      return instance.refs[index];
    },
    useCallback(fn) {
      hookIndex++;
      return fn;
    },
    useEffect() {
      hookIndex++;
    },
  };

  const apiService = {
    inferEntityTypes: () => new Promise(() => {}),
    getEntityTypeInferenceStatus: () => new Promise(() => {}),
  };

  vm.runInNewContext(compiled, {
    exports,
    require: (name) => {
      const modules = {
        react,
        'react/jsx-runtime': {
          jsx: (type, props) => ({ type, props }),
          jsxs: (type, props) => ({ type, props }),
        },
        '../ui/Modal': 'Modal',
        '../../services/api': { apiService },
        'lucide-react': { AlertTriangle: 'AlertTriangle', Loader2: 'Loader2' },
      };
      assert.ok(name in modules, `Unexpected dependency: ${name}`);
      return modules[name];
    },
  });

  return {
    render(props) {
      instance = { states: [], refs: [] };
      hookIndex = 0;
      return exports.EntityTypeInferenceModal(props);
    },
  };
}

test('renders through the shared modal stack, not a hand-rolled overlay', () => {
  const mod = load();
  const view = mod.render({
    appId: 1, siloId: 55, importJobId: 19,
    onConfirm() {}, onCancel() {},
  });
  assert.equal(view.type, 'Modal');
  assert.equal(view.props.isOpen, true);
  assert.equal(view.props.title, 'Entity types');
  assert.equal(typeof view.props.onClose, 'function');
  // The action bar must be the pinned footer, not scrolled out of view.
  assert.ok(view.props.footer, 'footer prop missing');
});
