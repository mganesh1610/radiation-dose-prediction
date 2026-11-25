const assert = require('node:assert/strict');
const fs = require('node:fs');
const test = require('node:test');
const vm = require('node:vm');

// Run the actual event handlers with independently resolvable HTTP responses.
function element() {
  return {
    children: [], value: '', dataset: {}, classList: { toggle() {} },
    setAttribute() {}, removeAttribute() {}, focus() {},
    append(...nodes) { this.children.push(...nodes); },
    replaceChildren(...nodes) { this.children = nodes; },
    get childElementCount() { return this.children.length; }
  };
}

const tick = () => new Promise(resolve => setImmediate(resolve));

async function workspace(examples = [{ id: 'first', study: 'study' }, { id: 'second', study: 'study' }],
  { search = '', methods = ['A', 'B', 'gradient_boosting_wb_wbc'] } = {}) {
  const nodes = new Map();
  function get(id) {
    if (!nodes.has(id)) nodes.set(id, element());
    return nodes.get(id);
  }
  const requests = [];
  const catalog = {
    role: 'reviewer', methods, model_release_overrides: { gradient_boosting_wb_wbc: 'wbc-release' }, uploads: [], examples
  };
  const response = data => ({ ok: true, json: async () => data });
  const fetch = async (path, options) => {
    if (path === '/api/prediction-catalog') return response(catalog);
    if (path.startsWith('/api/prediction-examples/')) {
      return response({ sample_id: path.split('/').pop(), values: [1] });
    }
    return new Promise(resolve => requests.push({
      path, body: JSON.parse(options.body), finish(data) { resolve(response(data)); }
    }));
  };
  vm.runInNewContext(['research-catalog.js', 'prediction-workflow.js'].map(name => fs.readFileSync(require.resolve('../app/static/' + name), 'utf8')).join('\n'), {
    fetch, document: { getElementById: get, createElement: element },
    window: { addEventListener() {}, matchMedia: () => ({ matches: false }) }, URLSearchParams, location: { hash: '#predictions', search }
  });
  await tick();
  async function select(id) {
    get('prediction-example').value = id;
    get('prediction-example').onchange();
    await tick();
  }
  return { get, select, requests };
}

function record(id, status = 'approved') {
  return { upload_id: 'uid-' + id, sample_id: id, status, qc: { usable_probes: 1, masked_probes: 0, blockers: [] } };
}

test('Predictions opens on combined boosting and respects explicit links and available methods', async () => {
  for (const [options, expected] of [
    [{}, 'gradient_boosting_wb_wbc'],
    [{ search: '?model=B' }, 'B'],
    [{ search: '?model=unknown' }, 'gradient_boosting_wb_wbc'],
    [{ methods: ['B', 'A'] }, 'B'],
    [{ methods: [] }, '']
  ]) {
    const { get, requests } = await workspace(undefined, options);
    assert.equal(get('prediction-model').value, expected);
    assert.equal(get('prediction-example').value, '');
    assert.equal(get('dose-run-status').textContent, 'Not run');
    assert.equal(get('predict-upload').disabled, true);
    assert.equal(requests.length, 0);
  }
});

test('The picker displays recorded doses for both groups and preserves accession values', async () => {
  const { get, requests } = await workspace([
    { id: 'training-zero', training_overlap: true, observed_dose_gy: 0 },
    { id: 'held-fraction', training_overlap: false, observed_dose_gy: 0.1 },
    { id: 'unknown', training_overlap: false, observed_dose_gy: null }
  ]);
  const groups = get('prediction-example').children;
  const options = groups.flatMap(group => group.children).map(option => ({
    value: option.value, label: option.children.map(child => child.textContent).join('')
  }));
  assert.deepEqual(groups.map(group => group.label), ['Used in training', 'Held out of training · GSE90909']);
  assert.deepEqual(options, [
    { value: 'training-zero', label: 'training-zero · 0\u00a0Gy' },
    { value: 'held-fraction', label: 'held-fraction · 0.1\u00a0Gy' },
    { value: 'unknown', label: 'unknown · Dose unavailable' }
  ]);
  assert.equal(requests.length, 0);
  assert.equal(get('prediction-result').dataset.ready, 'false');
});

test('Predict waits for the existing check and submits a single approved profile and model', async () => {
  const { get, select, requests } = await workspace();
  await select('first');
  assert.equal(requests.length, 1);
  assert.equal(requests[0].path, '/api/uploads?model=gradient_boosting_wb_wbc');
  assert.equal(get('predict-upload').disabled, false);
  const run = get('predict-upload').onclick();
  await get('predict-upload').onclick();
  assert.equal(requests.length, 1);
  requests[0].finish(record('first'));
  await tick();
  assert.equal(requests.length, 2);
  assert.equal(requests[1].path, '/api/predict');
  assert.deepEqual(requests[1].body, { upload_id: 'uid-first', model: 'gradient_boosting_wb_wbc' });

  // A review can revoke approval between the background check and inference.
  requests[1].finish({ status: 'rejected', record: record('first', 'rejected') });
  await run;
  assert.equal(get('predict-upload').disabled, true);
  assert.equal(get('prediction-result').dataset.ready, 'false');
  assert.match(get('prediction-status').textContent, /rejected/);
});

test('A late approval cannot run the previous profile or replace the current pending state', async () => {
  const { get, select, requests } = await workspace();
  await select('first');
  const queued = get('predict-upload').onclick();
  await select('second');
  assert.deepEqual(requests.map(r => r.body.sample_id), ['first', 'second']);
  requests[1].finish(record('second', 'pending_compatibility_review'));
  await tick();
  requests[0].finish(record('first'));
  await queued;
  assert.equal(requests.length, 2);
  assert.equal(get('predict-upload').disabled, true);
  assert.match(get('prediction-status').textContent, /Awaiting researcher review/);
  assert.match(get('input-preview').textContent, /second/);
});

test('Changing the model invalidates a pending estimate even when the user switches back', async () => {
  const { get, select, requests } = await workspace(undefined, { search: '?model=A' });
  await select('first');
  requests[0].finish(record('first'));
  await tick();
  const run = get('predict-upload').onclick();
  const model = get('prediction-model');
  model.value = 'B'; model.onchange();
  model.value = 'A'; model.onchange();
  requests[1].finish({ status: 'completed' });
  await run;
  assert.equal(get('prediction-result').dataset.ready, 'false');
  assert.equal(get('predict-upload').disabled, false);
  assert.equal(get('dose-placeholder-title').textContent, 'Ready to predict');
});

function completed(overrides = {}) {
  return {
    status: 'completed', sample_id: 'first', model: 'gradient_boosting_wb_wbc', estimate_gy: 0.096,
    dose_comparison: { study: 'study', observed_dose_gy: 0.1, estimate_minus_observed_gy: -0.004 },
    release: 'test', seed: 7, training_overlap: false, individual_prediction_interval: null,
    limitations: [], ...overrides
  };
}

test('The estimate stays hidden until the explicitly requested prediction completes', async () => {
  const { get, select, requests } = await workspace();
  const result = get('prediction-result'), predict = get('predict-upload');
  assert.equal(predict.disabled, true);
  assert.equal(result.dataset.ready, 'false');
  assert.equal(get('dose-placeholder-title').textContent, 'Waiting for a sample');
  await predict.onclick();
  assert.equal(requests.length, 0);
  await select('first');
  requests[0].finish(record('first'));
  await tick();
  assert.equal(get('dose-placeholder-title').textContent, 'Ready to predict');
  assert.equal(get('dose-run-status').textContent, 'Not run');
  assert.deepEqual(requests.map(r => r.path), ['/api/uploads?model=gradient_boosting_wb_wbc']);
  const run = predict.onclick();
  await predict.onclick();
  assert.equal(requests.length, 2);
  assert.equal(get('dose-run-status').textContent, 'Running');
  assert.equal(result.dataset.ready, 'false');
  requests[1].finish(completed());
  await run;
  assert.equal(result.dataset.ready, 'true');
  assert.equal(result.children[1].children[0].children[0].textContent, '0.096');
  const model = get('prediction-model');
  model.value = 'B';
  model.onchange();
  assert.equal(result.dataset.ready, 'false');
  assert.equal(requests[2].path, '/api/uploads/uid-first/check');
  requests[2].finish(record('first'));
  await tick();
  assert.equal(get('dose-placeholder-title').textContent, 'Ready to predict');
  assert.equal(requests.length, 3);
  await select('');
  assert.equal(get('dose-placeholder-title').textContent, 'Waiting for a sample');
  assert.equal(predict.disabled, true);
});

test('A partial or mismatched response leaves the dose hidden and permits a retry', async () => {
  const { get, select, requests } = await workspace();
  await select('first');
  requests[0].finish(record('first'));
  await tick();
  for (const overrides of [{ estimate_gy: null }, { sample_id: 'second' }, { model: 'B' }]) {
    const run = get('predict-upload').onclick();
    requests.at(-1).finish(completed(overrides));
    await run;
    assert.equal(get('prediction-result').dataset.ready, 'false');
    assert.equal(get('dose-run-status').textContent, 'Failed');
    assert.equal(get('predict-upload').disabled, false);
  }
});

test('Switching to combined boosting checks the new release and discards the older reply', async () => {
  const { get, requests, select } = await workspace(undefined, { search: '?model=A' });
  await select('first');
  const model = get('prediction-model');
  model.value = 'gradient_boosting_wb_wbc';
  model.onchange();
  await tick();
  assert.equal(requests[1].path, '/api/uploads?model=gradient_boosting_wb_wbc');
  requests[1].finish(record('first', 'approved'));
  await tick();
  requests[0].finish(record('first', 'rejected'));
  await tick();
  assert.equal(get('prediction-status').textContent, 'Compatibility approved');
  assert.equal(get('prediction-result').dataset.ready, 'false');
  get('predict-upload').onclick();
  await tick();
  assert.deepEqual(requests[2].body, {upload_id:'uid-first',model:'gradient_boosting_wb_wbc'});
});
