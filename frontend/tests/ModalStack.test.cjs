const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const ts = require('typescript');

// The modal stack decides which modal receives clicks (top) and which is
// covered (aria-hidden, its overlay sits below the top one). Paint order is
// DOM order, so the stack position must drive the inline z-index — otherwise
// a modal opened earlier in the stack but later in the DOM is painted above
// an interactive one: the interaction target looks gone and clicks close the
// wrong modal.
const source = fs.readFileSync(
  path.join(__dirname, '../src/components/ui/Modal.tsx'),
  'utf8',
);
const compiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.CommonJS,
    jsx: ts.JsxEmit.ReactJSX,
    esModuleInterop: true,
  },
}).outputText;

let idCounter = 0;

function makeInstance() {
  // useId must be stable per component instance — the modal stack uses it as
  // the identity of the mounted modal.
  return { states: [], refs: [], effects: [], id: `test-id-${idCounter++}` };
}

function loadModal() {
  const exports = {};
  let active = null;
  let hookIndex = 0;

  const react = {
    useState(initial) {
      const inst = active;
      const index = hookIndex++;
      if (!(index in inst.states)) inst.states[index] = initial;
      return [
        inst.states[index],
        (next) => {
          inst.states[index] = typeof next === 'function' ? next(inst.states[index]) : next;
        },
      ];
    },
    useRef(initial) {
      const inst = active;
      const index = hookIndex++;
      if (!(index in inst.refs)) inst.refs[index] = { current: initial };
      return inst.refs[index];
    },
    useId() {
      hookIndex++;
      return active.id;
    },
    useEffect(callback, deps) {
      const inst = active;
      const index = hookIndex++;
      const previous = inst.effects[index];
      if (!previous || deps.some((dep, i) => !Object.is(dep, previous.deps[i]))) {
        previous?.cleanup?.();
        inst.effects[index] = { deps, callback, pending: true };
      }
    },
  };

  vm.runInNewContext(compiled, {
    exports,
    document: { addEventListener() {}, removeEventListener() {} },
    HTMLElement: class HTMLElement {},
    require: (name) => {
      const modules = {
        react,
        'react/jsx-runtime': {
          jsx: (type, props) => ({ type, props }),
          jsxs: (type, props) => ({ type, props }),
        },
        'lucide-react': { X: 'X' },
      };
      assert.ok(name in modules, `Unexpected dependency: ${name}`);
      return modules[name];
    },
  });

  return {
    render(instance, props) {
      active = instance;
      hookIndex = 0;
      const view = exports.default({ size: 'large', ...props });
      for (const effect of instance.effects) {
        if (effect?.pending) {
          effect.pending = false;
          effect.cleanup = effect.callback();
        }
      }
      return view;
    },
    // The stack state lands in useState during the effect flush, so a second
    // render (deps unchanged, effect skipped) is what actually reflects it.
    renderSettled(instance, props) {
      const view = this.render(instance, props);
      return this.render(instance, props) ?? view;
    },
  };
}

test('a modal opened later paints above an earlier one and covers it', () => {
  const modal = loadModal();
  const first = makeInstance();
  const second = makeInstance();

  const firstView = modal.renderSettled(first, { isOpen: true, title: 'First', onClose() {} });
  assert.equal(firstView.props.style.zIndex, 50);
  assert.notEqual(firstView.props.inert, true);

  const secondView = modal.renderSettled(second, { isOpen: true, title: 'Second', onClose() {} });
  assert.equal(secondView.props.style.zIndex, 51);
  assert.notEqual(secondView.props.inert, true);

  // Re-render the first: it is now covered by the second. Covered means
  // aria-hidden (screen readers skip it) — NOT inert, which would also hide
  // the nested top modal from hit-testing and let clicks pass through.
  const coveredView = modal.renderSettled(first, { isOpen: true, title: 'First', onClose() {} });
  assert.equal(coveredView.props.style.zIndex, 50);
  assert.equal(coveredView.props.inert, undefined);
  assert.equal(coveredView.props['aria-hidden'], true);
});

test('closing the top modal un-covers the one below', () => {
  const modal = loadModal();
  const first = makeInstance();
  const second = makeInstance();

  modal.renderSettled(first, { isOpen: true, title: 'First', onClose() {} });
  modal.renderSettled(second, { isOpen: true, title: 'Second', onClose() {} });
  modal.renderSettled(first, { isOpen: true, title: 'First', onClose() {} });

  assert.equal(modal.render(second, { isOpen: false, title: 'Second', onClose() {} }), null);

  const restored = modal.renderSettled(first, { isOpen: true, title: 'First', onClose() {} });
  assert.equal(restored.props.style.zIndex, 50);
  assert.notEqual(restored.props.inert, true);
  assert.equal(restored.props['aria-hidden'], undefined);
});