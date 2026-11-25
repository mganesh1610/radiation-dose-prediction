async (page) => {
  const receipt = { startedAt: new Date().toISOString(), url: page.url(), checks: [], pageErrors: [] };
  const check = (name, pass, detail = {}) => {
    receipt.checks.push({ name, pass: Boolean(pass), detail });
    if (!pass) throw new Error(name + ': ' + JSON.stringify(detail));
  };
  const downloadText = async (trigger) => {
    const pending = page.waitForEvent('download');
    await trigger();
    const download = await pending;
    const stream = await download.createReadStream();
    let text = '';
    for await (const part of stream) text += part.toString('utf8');
    return text;
  };
  const openPalette = async () => {
    await page.getByRole('button', { name: 'Open command palette', exact: true }).click();
    await page.locator('#workspace-command-palette').ariaSnapshot();
    await page.getByRole('dialog', { name: 'Go to workspace', exact: true }).waitFor();
  };
  const navigate = async route => {
    await openPalette();
    await page.getByRole('combobox', { name: 'Search workspace pages', exact: true }).fill(route);
    await page.locator('#workspace-command-palette').ariaSnapshot();
    await page.locator(`ui-command-item[value="${route}"]`).click();
    await page.waitForFunction(name => location.hash === '#' + name && !document.getElementById('page-' + name).hidden, route);
  };
  const errors = error => receipt.pageErrors.push(String(error));
  page.on('pageerror', errors);
  try {
    await page.addInitScript(() => {
      window.uiCSPViolations = [];
      document.addEventListener('securitypolicyviolation', event => window.uiCSPViolations.push(event.effectiveDirective));
    });
    const entry = new URL(page.url());
    entry.hash = 'datasets';
    await page.goto(entry.href);
    await page.reload();
    await page.waitForFunction(() => document.querySelector('#dataset-study').hidden && document.querySelectorAll('#workspace-study-picker ui-select-item').length > 1);
    await page.locator('#dataset-study').evaluate(el => { el.value = 'all'; el.dispatchEvent(new Event('change', { bubbles: true })); });
    await page.locator('#sample-search').fill('');
    const catalog = await page.evaluate(async () => window.defenderPublic ? window.defenderPublic.read('/api/catalog') : (await fetch('/api/catalog')).json());
    const study = catalog.studies[1];
    const expected = catalog.samples.filter(sample => sample.study_id === study.study_id);
    receipt.catalog = { profiles: catalog.profiles, study: study.study_id, selectedProfiles: expected.length };
    check('Intended page renders meaningful content', await page.title() === 'Defender · Datasets' && await page.getByRole('heading', { name: 'Every sample has a source.', exact: true }).isVisible());
    check('Catalogue options arrive in the enhanced picker', await page.locator('#workspace-study-picker ui-select-item').count() === catalog.studies.length + 1);
    check('Native fallback retains the original form identity', await page.locator('#dataset-study').count() === 1 && await page.locator('#dataset-study').evaluate(el => el.hidden && el.value === 'all'));

    await page.getByRole('combobox', { name: 'Study', exact: true }).click();
    await page.locator('#dataset-study-field').ariaSnapshot();
    const search = page.getByRole('combobox', { name: 'Search studies', exact: true });
    await search.fill(study.study_id);
    await page.waitForFunction(() => [...document.querySelectorAll('ui-select-item')].filter(el => el.checkVisibility()).length === 1);
    check('Study search exposes only the matching option', await page.getByRole('option').count() === 1);
    await search.press('ArrowDown');
    await page.keyboard.press('Enter');
    await page.waitForFunction(id => document.querySelector('#dataset-study').value === id, study.study_id);
    check('Keyboard study selection drives the real table filter', await page.locator('#sample-rows tr').count() === expected.length && (await page.locator('#dataset-count').innerText()).startsWith(expected.length + ' of ' + catalog.profiles));
    check('All visible study values match the selected accession', (await page.locator('#sample-rows tr td:nth-child(2)').allTextContents()).every(value => value === study.study_id));

    const csv = await downloadText(() => page.getByRole('button', { name: 'Download visible rows', exact: true }).click());
    const fields = ['sample_id', 'study_id', 'group_id', 'dose_gy', 'masked_common_probes'];
    const expectedCSV = [fields.join(','), ...expected.map(sample => fields.map(key => sample[key]).join(','))].join('\n');
    check('CSV download preserves exact original field values and row order', csv === expectedCSV, { rows: expected.length });
    await page.locator('#sample-search').fill(expected[0].sample_id);
    check('Existing profile search combines with the study filter', await page.locator('#sample-rows tr').count() === 1 && await page.locator('#sample-rows tr td').first().innerText() === expected[0].sample_id);
    await page.locator('#sample-search').fill('no-such-profile');
    check('Existing zero-profile state remains available', await page.getByText('No profiles match these filters.', { exact: true }).isVisible());
    await page.locator('#sample-search').fill(expected[0].sample_id);

    await page.locator('#dataset-study').evaluate(el => { el.value = 'all'; el.dispatchEvent(new Event('change', { bubbles: true })); });
    check('Native change events synchronize the enhanced control', await page.locator('#workspace-study-picker').evaluate(el => el.value === 'all'));
    await page.locator('#dataset-study').evaluate((el, value) => { el.value = value; el.dispatchEvent(new Event('change', { bubbles: true })); }, study.study_id);

    const analysisRequests = [];
    const track = request => { if (request.url().includes('/api/analysis') || /\/analysis-[^/]+\.json/.test(request.url())) analysisRequests.push(request.url()); };
    page.on('request', track);
    await openPalette();
    const dialog = page.getByRole('dialog', { name: 'Go to workspace', exact: true });
    const query = page.getByRole('combobox', { name: 'Search workspace pages', exact: true });
    check('Palette opens with a named focused search field', await query.evaluate(el => el === document.activeElement));
    await query.fill('heatmap');
    await page.locator('#workspace-command-palette').ariaSnapshot();
    check('Topic search finds Analysis through its keywords', await page.locator('ui-command-item[value="analysis"]').isVisible() && await page.getByRole('option').count() === 1);
    await page.keyboard.press('ArrowDown');
    check('Highlighting a navigation choice starts no scientific request', analysisRequests.length === 0);
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => location.hash === '#analysis' && document.querySelector('#analysis-table tbody tr'));
    check('Enter navigates through the existing route', await page.title() === 'Defender · Analysis' && !await dialog.isVisible());
    await page.waitForFunction(() => document.activeElement === document.querySelector('#main-content'), undefined, { timeout: 5000 });
    check('Command navigation moves focus to main content', true);
    check('Dataset filtering does not change the scientific cohort', await page.locator('#analysis-study').inputValue() === 'all' && await page.locator('#analysis-table tbody tr').count() === catalog.profiles);
    const chartRows = await page.locator('#analysis-table tbody tr').allTextContents();
    await page.locator('#workspace-export').click();
    await page.locator('#workspace-export-options').ariaSnapshot();
    const snapshot = JSON.parse(await downloadText(() => page.locator('[data-workspace-export="snapshot"]').click()));
    check('Scientific export retains every measured profile and PCA score', snapshot.samples.length === catalog.profiles && snapshot.pca.scores.length === catalog.profiles && snapshot.options.study === 'all');
    for (const [name, expectedCells] of [
      ['Heatmap', snapshot.heatmap.gene_ids.length * snapshot.samples.length],
      ['Correlation', snapshot.correlation.gene_ids.length ** 2],
    ]) {
      await page.getByRole('tab', { name, exact: true }).click();
      const cells = page.locator('#analysis-chart rect[data-point]');
      check(name + ' renders every cell', await cells.count() === expectedCells);
      await page.waitForFunction(() => {
        const plot = document.getElementById('chart-scroll'), frame = plot.getBoundingClientRect();
        const chart = document.getElementById('analysis-chart'), bounds = chart.getBoundingClientRect();
        const header = document.querySelector('main > header').getBoundingClientRect();
        return plot.scrollWidth === plot.clientWidth && plot.scrollHeight === plot.clientHeight &&
          bounds.top >= Math.max(frame.top, header.bottom) && bounds.bottom <= Math.min(frame.bottom, innerHeight) &&
          [...chart.querySelectorAll('text, rect[data-point]')].every(mark => {
            const r = mark.getBoundingClientRect();
            return r.left >= frame.left && r.right <= frame.right && r.top >= frame.top && r.bottom <= frame.bottom;
          });
      });
      check(name + ' fits all cells and axis labels in the visible window', true);
      if (name === 'Correlation') {
        const cell = await cells.first().boundingBox();
        check('Correlation cells stay square', cell.width > 0 && Math.abs(cell.width - cell.height) < 0.1);
      }
      await cells.first().focus();
      await page.keyboard.press('End');
      check(name + ' last cell is keyboard inspectable', await cells.last().getAttribute('aria-label') === await page.locator('#point-detail').textContent());
    }
    await page.locator('#tab-qc').click();
    check('Existing chart tabs remain keyboard-addressable', await page.locator('#tab-qc').getAttribute('aria-selected') === 'true');
    await page.locator('#tab-pca').click();
    check('Returning to PCA preserves measured table values', JSON.stringify(await page.locator('#analysis-table tbody tr').allTextContents()) === JSON.stringify(chartRows));
    await navigate('datasets');
    check('Returning through the palette preserves both dataset filters', await page.locator('#dataset-study').inputValue() === study.study_id && await page.locator('#sample-search').inputValue() === expected[0].sample_id && await page.locator('#sample-rows tr').count() === 1);

    await page.locator('#sample-search').focus();
    await page.keyboard.press('ControlOrMeta+k');
    await dialog.waitFor();
    check('Platform command shortcut opens the palette from a text field', await query.evaluate(el => el === document.activeElement));
    check('Reopening the palette clears the previous query', await query.inputValue() === '');
    await page.keyboard.press('Escape');
    check('Escape closes an empty palette and restores trigger focus', !await dialog.isVisible() && await page.getByRole('button', { name: 'Open command palette', exact: true }).evaluate(el => el === document.activeElement));
    await openPalette();
    await query.fill('no-such-workspace-command-987654');
    check('Palette has a useful zero-result state', await page.getByText('No pages match your search.', { exact: true }).isVisible());
    await query.fill('');
    await page.keyboard.press('Escape');
    page.off('request', track);
    check('No page exceptions during the interaction path', receipt.pageErrors.length === 0, receipt.pageErrors);
    check('Existing CSP allows all enhanced interactions', await page.evaluate(() => window.uiCSPViolations.length === 0));
    receipt.scientificSnapshot = { analysisId: snapshot.analysis_id, options: snapshot.options, profiles: snapshot.samples.length, scores: snapshot.pca.scores.length };
    receipt.status = 'passed';
  } catch (error) {
    receipt.status = 'failed';
    receipt.error = String(error);
  } finally {
    page.off('pageerror', errors);
    receipt.completedAt = new Date().toISOString();
    await page.evaluate(value => { window.workspaceControlTestReceipt = value; }, receipt);
    console.log(JSON.stringify(receipt));
  }
}
