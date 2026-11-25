// With Playwright and Chromium installed, run: node --test tests/browser/experiments-workspace.js
// The public page loads its normal scripts. Only the result records are synthetic.
'use strict';
const { after, before, test } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { chromium } = require('playwright');

const app = path.resolve(__dirname, '../../app');
const origin = 'http://defender.test';
const dataPath = '/data/00000000000000000000';
const excluded = ['xgboost', 'random_forest', 'svr_rbf'];
let browser;
before(async () => { browser = await chromium.launch({channel: process.env.PLAYWRIGHT_CHANNEL || undefined}); });
after(async () => { await browser?.close(); });

function makeResults(version, extraLowScore) {
  const combined = 'gradient_boosting_wb_wbc';
  const ids = [...excluded, combined, 'at_cutoff', 'different_fit', ...Array.from({length: 14}, (_, i) => 'fit_' + i)];
  const regression = ids.map((model, i) => ({
    model, label: model, family: model === combined || i < 2 ? 'Ensemble' : 'Linear', mae_gy: 1 + i / 20
  }));
  const data = {schema_version: version, release_id: 'browser-test', purpose: 'test', core_profiles: 145,
    eligible_profiles: 567, final_test_included: false, regression, classification: []};
  if (version >= 2) {
    data.wbc_profiles = 422;
    data.held_out_profiles = 145;
    data.matched_gradient_boosting = {rows: [regression[3], {model: 'gradient_boosting', label: 'Whole blood only', mae_gy: 2}]};
  }
  if (version === 3) {
    const models = Object.fromEntries(regression.map(r => [r.model, {mae_gy: 1}]));
    models.at_cutoff.mae_gy = 0.770;
    models.different_fit.mae_gy = extraLowScore ? 0.7699 : 0.7701;
    data.retrospective_evaluation = {study: 'GSE90909', profiles: 52, donors: 12, run_date: '2026-01-01',
      models, fixed_whole_blood_control: {mae_gy: 1.5}};
  }
  return data;
}

async function serveFixtures(page, data) {
  await page.route('**/*', async route => {
    const url = new URL(route.request().url());
    if (url.origin !== origin) return route.abort();
    if (url.pathname === '/') return route.fulfill({path: path.join(app, 'public/index.html')});
    if (url.pathname === '/data/publication.json') {
      return route.fulfill({json: {schema_version: 1, data_path: dataPath, published_date: '2026-01-01', profiles: 145}});
    }
    if (url.pathname === dataPath + '/development-results.json') return route.fulfill({json: data});
    if (url.pathname.startsWith('/static/')) {
      const file = url.pathname.slice('/static/'.length);
      const dir = ['public-data.js', 'public-results.js', 'public.css'].includes(file) ? 'public' : 'static';
      return route.fulfill({path: path.join(app, dir, file)});
    }
    return route.fulfill({status: 404, json: {error: 'No fixture for this resource'}});
  });
}

for (const [version, width, extraLowScore] of [[3, 1440, false], [3, 390, true], [2, 1440, false], [1, 390, false]]) {
  test(`Development comparison · schema ${version} · ${width}px`, async () => {
    const data = makeResults(version, extraLowScore);
    const page = await browser.newPage({viewport: {width, height: 900}});
    try {
      await serveFixtures(page, data);
      await page.goto(origin + '/?comparison=development#experiments');
      const grid = page.locator('#development-regression');
      const names = grid.locator('tbody th');
      await names.first().waitFor();
      const shown = data.regression.filter(r => !excluded.includes(r.model) && !(extraLowScore && r.model === 'different_fit'));
      assert.deepEqual(await names.allTextContents(), shown.map(r => r.label));
      assert.deepEqual(await grid.locator('thead th').allTextContents(), ['Model', 'Development MAE (Gy)', 'GSE90909 MAE (Gy)']);
      assert.match(await page.locator('#model-count').innerText(), new RegExp(shown.length + ' of 20'));
      const evaluation = data.retrospective_evaluation;
      const externalScores = shown.map(r => evaluation ? evaluation.models[r.model].mae_gy.toFixed(3) : '—');
      assert.deepEqual(await grid.locator('.exp-external-mae').allTextContents(), externalScores);
      await grid.locator('.exp-dev-sort').click();
      assert.deepEqual(await names.allTextContents(), shown.map(r => r.label).reverse());
      assert.equal(await grid.locator('th[aria-sort]').getAttribute('aria-sort'), 'descending');
      await page.locator('#development-family').selectOption('Ensemble');
      assert.deepEqual(await names.allTextContents(), ['gradient_boosting_wb_wbc']);
      assert.equal(await page.locator('#matched-boosting').isVisible(), version >= 2);
      if (evaluation) {
        assert.deepEqual(await page.locator('#matched-boosting-table .exp-external-mae').allTextContents(), ['1.000', '1.500']);
      } else {
        assert.match(await page.locator('#comparison-run-date').innerText(), /unavailable/i);
      }
      // Export from the filtered view, including all hidden predictors and their original scores.
      await page.locator('#development-download').click();
      const saved = page.waitForEvent('download');
      await page.getByRole('button', {name: 'Save JSON'}).click();
      const stream = await (await saved).createReadStream();
      const chunks = [];
      for await (const chunk of stream) chunks.push(chunk);
      const result = JSON.parse(Buffer.concat(chunks).toString('utf8'));
      assert.deepEqual(result.regression, data.regression);
      assert.deepEqual(result.retrospective_evaluation, evaluation);
      assert.deepEqual(result.matched_gradient_boosting, data.matched_gradient_boosting);
      await page.getByRole('button', {name: 'Close export'}).click();
      await page.locator('#development-family').selectOption('Linear');
      assert.equal(await page.locator('#matched-boosting').isVisible(), false);
    } finally {
      await page.close();
    }
  });
}
