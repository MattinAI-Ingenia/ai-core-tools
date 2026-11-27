const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const ts = require('typescript');

const source = fs.readFileSync(
  path.join(__dirname, '../src/components/services/ServiceUsageModal.tsx'),
  'utf8',
);
const compiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.CommonJS,
    jsx: ts.JsxEmit.ReactJSX,
    esModuleInterop: true,
  },
}).outputText;

const usage = {
  service_id: 10,
  service_name: 'Source service',
  kind: 'ai',
  is_system: false,
  lifecycle_status: 'active',
  end_of_service_at: null,
  deprecation_notice: null,
  agents: [{ agent_id: 1, name: 'Agent', roles: ['llm'], app_id: 2 }],
  silos: [],
  repositories: [],
  middlewares: [],
  scheduled_tasks: [],
  apps_using_as_default: [],
};

const strings = {
  title: (name) => `Used by: ${name}`,
  loadError: 'Could not load usage',
  loading: 'Loading usage',
  retry: 'Retry',
  close: 'Close',
  nothing: 'Nothing uses this service',
  empty: 'None',
  sections: {
    agents: 'Agents',
    appsDefault: 'Apps using it as default',
    appsDefaultRerank: 'Apps using it as default reranker',
    silos: 'Silos',
    repositories: 'Repositories',
    middlewares: 'Middlewares',
    scheduledTasks: 'Scheduled tasks',
  },
  agentRoles: { llm: 'LLM', rerank: 'Reranker', rerank_app_default: 'Reranker (app default)' },
  siloRoles: {},
  moveTitle: 'Move agents',
  moveCount: (count) => `${count} agents use this service`,
  moveNone: 'No agents use this service',
  targetsLoading: 'Loading available services',
  targetsLoadError: 'Could not load available services',
  moveTargetLabel: 'Target service',
  moveTargetPlaceholder: 'Select a service',
  moveTargetEnds: (name) => name,
  moveNoTargets: 'No other service is available',
  moveButton: 'Move agents',
  moveConfirm: () => 'Confirm move',
  moveConfirmButton: 'Confirm',
  moveConfirming: 'Moving',
  moveCancel: 'Cancel',
  moveNote: 'Some references are not moved',
  platformDefaultBadge: 'Platform default',
};

function deferredFlush() {
  return new Promise((resolve) => setImmediate(resolve));
}

function mount(apiService, overrides = {}) {
  const states = [];
  const refs = [];
  const memoValues = [];
  const effects = [];
  let hookIndex = 0;

  const react = {
    useState(initial) {
      const index = hookIndex++;
      if (!(index in states)) states[index] = initial;
      return [
        states[index],
        (next) => {
          states[index] = typeof next === 'function' ? next(states[index]) : next;
        },
      ];
    },
    useEffect(callback, deps) {
      const index = hookIndex++;
      const previous = effects[index];
      if (!previous || deps.some((dep, i) => !Object.is(dep, previous.deps[i]))) {
        previous?.cleanup?.();
        effects[index] = { deps, callback, pending: true };
      }
    },
    useMemo(factory, deps) {
      const index = hookIndex++;
      const previous = memoValues[index];
      if (!previous || deps.some((dep, i) => !Object.is(dep, previous.deps[i]))) {
        memoValues[index] = { deps, value: factory() };
      }
      return memoValues[index].value;
    },
    useRef(initial) {
      const index = hookIndex++;
      if (!(index in refs)) refs[index] = { current: initial };
      return refs[index];
    },
    useId() {
      const index = hookIndex++;
      return `test-id-${index}`;
    },
  };

  const modules = {
    react,
    'react/jsx-runtime': {
      jsx: (type, props) => ({ type, props }),
      jsxs: (type, props) => ({ type, props }),
    },
    'react-router-dom': { Link: 'a' },
    '../ui/Modal': { default: 'Modal' },
    '../ui/ServiceStatusBadge': { default: 'ServiceStatusBadge' },
    '../ui/ButtonApp': { ButtonApp: 'ButtonApp' },
    '../ui/FormField': { FormSelect: 'FormSelect' },
    '../ui/Alert': { __esModule: true, default: 'Alert' },
    '../ui/CardApp': { __esModule: true, default: 'CardApp' },
    '../ui/Badge': { Badge: 'Badge' },
    '../../services/api': {
      apiService,
      ApiError: class ApiError extends Error {},
    },
    '../../utils/serviceLifecycle': {
      SERVICE_LIFECYCLE_STRINGS: { rerankWarning: (date) => `Rerank ends ${date}` },
      SERVICE_USAGE_STRINGS: strings,
      formatEndOfServiceDate: (date) => date,
    },
  };
  const exports = {};
  vm.runInNewContext(compiled, {
    exports,
    require: (name) => {
      assert.ok(name in modules, `Unexpected dependency: ${name}`);
      return modules[name];
    },
  });

  return {
    render() {
      hookIndex = 0;
      const view = exports.default({
        onClose() {},
        kind: 'ai',
        scope: 'app',
        appId: 2,
        serviceId: 10,
        serviceName: 'Source service',
        canMove: true,
        ...overrides,
      });
      for (const effect of effects) {
        if (effect?.pending) {
          effect.pending = false;
          effect.cleanup = effect.callback();
        }
      }
      return view;
    },
  };
}

function elements(node) {
  if (Array.isArray(node)) return node.flatMap(elements);
  if (!node || typeof node !== 'object') return [];
  return [node, ...elements(node.props?.children), ...elements(node.props?.message)];
}

function textContent(node) {
  if (typeof node === 'string' || typeof node === 'number') return String(node);
  if (Array.isArray(node)) return node.map(textContent).join(' ');
  if (!node || typeof node !== 'object') return '';
  return [textContent(node.props?.children), textContent(node.props?.message)].join(' ');
}

test('shows target-load errors and retries loading target services', async () => {
  let targetLoads = 0;
  const page = mount({
    getAIServiceUsedBy: async () => usage,
    getAIServices: async () => {
      if (++targetLoads === 1) throw new Error('Target service request failed');
      return [{ service_id: 20, name: 'Target service', lifecycle_status: 'active' }];
    },
  });

  page.render();
  await deferredFlush();
  const failedView = page.render();

  assert.ok(
    // Alert renders role="alert" for type="error" (see components/ui/Alert.tsx).
    elements(failedView).some((node) => node.type === 'Alert' && node.props.type === 'error'),
    'Expected an alert for the target-service loading error',
  );
  assert.match(textContent(failedView), /Could not load available services/);
  assert.doesNotMatch(textContent(failedView), /No other service is available/);

  const retry = elements(failedView).find(
    (node) => node.type === 'button' && node.props.children === 'Retry',
  );
  assert.ok(retry, 'a retry button should be available');
  retry.props.onClick();

  page.render();
  await deferredFlush();
  const recoveredView = page.render();

  assert.equal(targetLoads, 2);
  assert.ok(elements(recoveredView).some(
    (node) => node.type === 'FormSelect' && node.props.options.some((option) => option.value === 20),
  ));
});

test('rerank kind loads usage from the rerank endpoint, shows roles, default apps and platform badge, and offers no move', async () => {
  const calls = [];
  const rerankUsage = {
    ...usage,
    service_id: 11,
    kind: 'rerank',
    is_system: true,
    is_platform_default: true,
    agents: [{ agent_id: 1, name: 'Agent', roles: ['rerank', 'rerank_app_default'], app_id: 2 }],
    apps_using_as_default: [{ app_id: 2, app_name: 'Workspace' }],
  };
  const page = mount({
    getRerankServiceUsedBy: async (appId, serviceId) => {
      calls.push(['app', appId, serviceId]);
      return rerankUsage;
    },
    getSystemRerankServiceUsedBy: async (serviceId) => {
      calls.push(['system', serviceId]);
      return rerankUsage;
    },
    getAIServices: async () => assert.fail('rerank must not load move targets'),
    getSandboxServices: async () => assert.fail('rerank must not load move targets'),
  }, { kind: 'rerank', serviceId: 11 });

  page.render();
  await deferredFlush();
  const view = page.render();
  const text = textContent(view);
  const props = elements(view).map((n) => n.props ?? {});
  const has = (key, value) => props.some((p) => p[key] === value);

  assert.deepEqual(calls, [['app', 2, 11]]);
  assert.ok(has('detail', 'Reranker, Reranker (app default)'));
  assert.ok(has('title', 'Apps using it as default reranker'));
  assert.ok(has('name', 'Workspace'));
  assert.ok(elements(view).some((n) => n.type === 'Badge' && n.props.label === 'Platform default'));
  assert.ok(!elements(view).some((n) => n.type === 'FormSelect'), 'no move target select for rerank');
  assert.ok(!has('title', 'Silos') && !has('title', 'Repositories') && !has('title', 'Middlewares'));
  assert.doesNotMatch(text, /Silos/);
});

test('rerank kind with system scope uses the admin used-by endpoint', async () => {
  const calls = [];
  const page = mount({
    getSystemRerankServiceUsedBy: async (serviceId) => {
      calls.push(serviceId);
      return { ...usage, kind: 'rerank', agents: [] };
    },
  }, { kind: 'rerank', scope: 'system', appId: undefined, serviceId: 11 });
  page.render();
  await deferredFlush();
  page.render();
  assert.deepEqual(calls, [11]);
});
