async (page) => {
  const root = new URL(page.url()).origin;
  const checks = [];
  const errors = [];
  const recordError = error => errors.push(error.message);
  page.on('pageerror', recordError);
  function check(name, actual, expected) {
    if (JSON.stringify(actual) !== JSON.stringify(expected)) {
      throw new Error(name + ' — expected ' + JSON.stringify(expected) + ', received ' + JSON.stringify(actual));
    }
    checks.push(name);
  }
  async function download(button) {
    const pending = page.waitForEvent('download');
    await button.click();
    const stream = await (await pending).createReadStream();
    let text = '';
    for await (const chunk of stream) text += chunk.toString('utf8');
    return JSON.parse(text);
  }
  const byId = id => page.locator('#' + id);
  const tab = name => page.getByRole('tab', { name, exact: true });
  const card = byId('reviewer-card-dialog');
  const development = tab('Development comparison');
  const final = tab('Final evaluation');
  const training = tab('Training');
  const gridValues = id => byId(id).locator('tbody td.exp-mae-cell').allTextContents();
  const checkTab = async (target, panel) => {
    check(panel + ' tab selected', await target.getAttribute('aria-selected'), 'true');
    check(panel + ' is the visible panel', await page.locator('#page-experiments [role=tabpanel]:visible').evaluateAll(nodes => nodes.map(n => n.id)), [panel]);
  };

  try {
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.goto(root + '/#experiments');
    await byId('reviewer-loaded').waitFor();
    await checkTab(final, 'reviewer-results');
    const result = await download(byId('reviewer-download'));
    const displayed = result.model_cards.filter(model => model.id !== 'gradient_boosting');
    const sorted = displayed.map(model => ({
      id: model.id, score: model.scores.find(s => s.seed === model.primary_seed),
    })).sort((a, b) => a.score.donor_mae - b.score.donor_mae);
    const ids = sorted.map(row => row.id);
    const order = () => byId('reviewer-neural').locator('button[data-model]').evaluateAll(buttons => buttons.map(b => b.dataset.model));
    check('Complete final evaluation download', [result.evaluated_date, result.profiles, result.donors, result.model_cards.length, result.model_cards.flatMap(m => m.scores).length], ['2026-11-22', 52, 12, 9, 21]);
    check('Eight displayed models ranked by their exported primary scores', await order(), ids);
    const combined = byId('reviewer-neural').locator('tr').filter({has: page.locator('button[data-model="gradient_boosting_wb_wbc"]')});
    check('Combined boosting shows its donor-balanced MAE in the ranking, default details and download',
      [await combined.locator('td.exp-mae-cell > span').first().textContent(), await byId('exp-mae').textContent(),
       sorted.find(row => row.id === 'gradient_boosting_wb_wbc').score.donor_mae.toFixed(3)], ['0.777', '0.777', '0.777']);
    check('Displayed errors agree with all exported models', await byId('reviewer-neural').locator('td.exp-mae-cell > span:first-child').allTextContents(), sorted.map(r => r.score.donor_mae.toFixed(3)));
    check('Only the whole-blood boosting row is omitted', [ids.length, ids[0], await byId('reviewer-neural').locator('[data-model="gradient_boosting"]').count()], [8, 'gradient_boosting_wb_wbc', 0]);
    await byId('exp-sort').click();
    check('Descending rank order', await order(), [...ids].reverse());
    await byId('exp-sort').click();
    for (const row of sorted) {
      await page.locator('button[data-model="' + row.id + '"]').click();
      check('Selected details agree with ' + row.id, [await byId('exp-mae').textContent(), await byId('exp-ci').textContent()],
        [row.score.donor_mae.toFixed(3), row.score.mae_ci.low.toFixed(3) + ' – ' + row.score.mae_ci.high.toFixed(3) + ' Gy']);
    }
    await byId('reviewer-open-card').click();
    await card.waitFor();
    const cardResult = await download(byId('reviewer-download-card'));
    check('Card download matches selected model', cardResult.model, result.model_cards.find(m => m.id === ids.at(-1)));
    await page.keyboard.press('Escape');
    await card.waitFor({ state: 'hidden' });
    await final.focus();
    await page.keyboard.press('ArrowRight');
    await byId('development-family').waitFor();
    await checkTab(development, 'development-comparisons');
    check('Initial model family', await byId('development-family').inputValue(), 'all');
    check('Every displayed predictor has a final-cohort score', await byId('development-regression').locator('td.exp-external-mae').count(), 17);
    await byId('development-family').selectOption('Ensemble');
    check('Ensemble GSE90909 errors', await byId('development-regression').locator('td.exp-external-mae').allTextContents(), ['0.777', '0.951']);
    check('Two visible tuned ensemble results', await gridValues('development-regression'), ['2.180', '2.442']);
    check('Fixed-settings comparison', await gridValues('matched-boosting-table'), ['2.016', '2.669']);
    await byId('development-regression').locator('.exp-dev-sort').click();
    check('Tuned sort works independently of the fixed pair', await gridValues('development-regression'), ['2.442', '2.180']);
    await byId('development-family').selectOption('Linear');
    check('Pair is hidden for another family', await byId('matched-boosting').isVisible(), false);
    await byId('development-family').selectOption('all');
    check('All 17 visible predictors available', (await gridValues('development-regression')).length, 17);
    await page.locator('.exp-classifiers > summary').click();
    check('Three exposure classifiers', await byId('development-classification').locator('tbody tr').count(), 3);
    await byId('development-download').click();
    const data = await download(page.getByRole('button', { name: 'Save JSON' }));
    check('Development export identifies its holdouts', [data.regression.length, data.matched_gradient_boosting.rows.length, data.eligible_profiles, data.wbc_profiles, data.held_out_profiles, data.development_scores_include_final_test], [20, 2, 567, 422, 145, false]);
    check('Retrospective export covers all 20 combined fits', [Object.keys(data.retrospective_evaluation.models).length,
      data.retrospective_evaluation.profiles, data.retrospective_evaluation.donors, data.retrospective_evaluation.retrospective], [20, 52, 12, true]);
    await page.getByRole('button', { name: 'Close export' }).click();
    await page.locator('dialog.development-export').waitFor({ state: 'detached' });
    await development.press('End');
    await checkTab(training, 'engineering-demo');
    if (await byId('train-form').count()) {
      await page.waitForFunction(() => document.getElementById('split-fit').textContent.includes('profiles to fit'));
      for (const [study, fit, heldOut] of [['GSE8917', 542, 25], ['GSE65292', 547, 20], ['GSE102971', 467, 100]]) {
        await byId('study').selectOption(study);
        check('Training split for ' + study, [await byId('split-fit').textContent(), await byId('split-test').textContent()], [fit + ' profiles to fit', heldOut + ' held out']);
      }
      check('Training pool', await page.locator('.exp-cohort dd:nth-child(2)').allTextContents(), ['567', '145', '422']);
      await byId('refresh').click();
      await page.waitForFunction(() => document.getElementById('allowance').textContent.includes(' of '));
    } else {
      await byId('published-runs').getByRole('button').first().waitFor();
      check('Published training result is available', await byId('results').isVisible(), true);
    }
    await page.goBack();
    await checkTab(development, 'development-comparisons');
    await page.goForward();
    await checkTab(training, 'engineering-demo');
    await page.setViewportSize({ width: 393, height: 845 });
    const guest = await byId('guest-training').count() > 0;
    await page.goto(root + '/?model=gradient_boosting#experiments');
    if (guest) {
      await byId('reviewer-loaded').waitFor();
      check('Guest router returns an archived model link to the default result', [
        new URL(page.url()).searchParams.get('model'), await byId('exp-mae').textContent()
      ], [null, '0.777']);
      const archived = result.model_cards.find(model => model.id === 'gradient_boosting');
      check('Guest export retains the whole-blood reference',
        archived.scores.find(score => score.seed === archived.primary_seed).donor_mae.toFixed(3), '0.795');
    } else {
      await card.waitFor();
      check('Existing whole-blood model link opens its own result', await card.locator('.exp-big-value span').textContent(), '0.795');
      await byId('reviewer-close-card').click();
      await card.waitFor({ state: 'hidden' });
    }
    check('Returning from the linked model restores the eight-row ranking', await order(), ids);
    await page.goto(root + '/?model=gradient_boosting_wb_wbc#experiments');
    await byId('reviewer-loaded').waitFor();
    const selected = page.locator('button[data-model="gradient_boosting_wb_wbc"]');
    await selected.click();
    await card.waitFor();
    const bounds = await card.boundingBox();
    check('Mobile detail fills the viewport from its top left', [bounds.x, bounds.y, bounds.width, bounds.height], [0, 0, 393, 845]);
    check('Mobile details show selected error', await card.locator('.exp-big-value span').textContent(), '0.777');
    await byId('reviewer-close-card').click();
    await card.waitFor({ state: 'hidden' });
    check('Mobile Back restores selection and keyboard focus', await selected.evaluate(b => b === document.activeElement && b.getAttribute('aria-pressed') === 'true'), true);
    await page.goForward();
    await card.waitFor();
    await page.reload();
    await card.waitFor();
    await page.goBack();
    await card.waitFor({ state: 'hidden' });
    await development.click();
    await byId('development-family').selectOption('Ensemble');
    check('Mobile comparison shows only the two MAE measures',
      await byId('development-regression').locator('thead th').allTextContents(),
      ['Model', 'Development MAE (Gy)', 'GSE90909 MAE (Gy)']);
    await development.focus();
    await page.keyboard.press('Home');
    await checkTab(final, 'reviewer-results');
    check('Page remains within the mobile viewport', await page.evaluate(() => document.documentElement.scrollWidth), 393);
    check('Last evaluation date', await page.locator('.main-footer time').innerText(), 'Last evaluated 22 Nov 2026');
    check('Browser errors', errors, []);
    return { status: 'passed', checks };
  } finally {
    page.off('pageerror', recordError);
  }
}
