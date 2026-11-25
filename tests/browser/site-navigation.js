async (page) => {
  const root = new URL(page.url()).origin;
  const receipt = { startedAt: new Date().toISOString(), checks: [], pageErrors: [] };
  const check = (name, pass, detail = {}) => {
    receipt.checks.push({ name, pass: Boolean(pass), detail });
    if (!pass) throw Error(name + ': ' + JSON.stringify(detail));
  };
  const errors = error => receipt.pageErrors.push(String(error));
  page.on('pageerror', errors);
  const routes = ['overview', 'datasets', 'analysis', 'experiments', 'predictions', 'assistant'];
  const menu = () => page.getByRole('button', { name: 'Open workspace navigation', exact: true });
  const drawer = () => page.getByRole('dialog', { name: 'Research workspace', exact: true });
  const openMenu = async () => {
    await menu().click();
    await drawer().waitFor();
    await drawer().evaluate(async surface => {
      await new Promise(resolve => requestAnimationFrame(resolve));
      await Promise.all(surface.getAnimations().map(animation => animation.finished.catch(() => {})));
    });
  };
  try {
    await page.addInitScript(() => {
      window.siteCSPViolations = [];
      document.addEventListener('securitypolicyviolation', event => window.siteCSPViolations.push(event.effectiveDirective));
    });
    await page.goto(root + '/#datasets');
    await page.reload(); // Run the shell's startup scripts with the Overview hidden.
    await page.waitForFunction(() => document.body.dataset.mobileNavigation === 'ready');
    check('Direct entry to Datasets initializes without a script error', receipt.pageErrors.length === 0, receipt.pageErrors);
    await page.goto(root + '/#overview');
    await page.waitForFunction(() => !document.getElementById('dna-motion').hidden);
    check('Playback controls are available after returning to Overview', await page.locator('#dna-motion').isVisible());
    for (const width of [320, 768, 1440]) {
      await page.setViewportSize({ width, height: 1000 });
      await page.goto(root + '/?comparison=final#overview');
      await page.waitForFunction(() => document.body.dataset.mobileNavigation === 'ready');
      for (const route of routes) {
        await page.evaluate(route => { location.hash = route; }, route);
        await page.locator('#page-' + route).waitFor();
        if (route === 'datasets') await page.waitForFunction(() => document.querySelector('#sample-rows tr'));
        if (route === 'analysis') await page.waitForFunction(() => document.querySelector('#analysis-table tbody tr'));
        if (route === 'experiments') await page.locator('#reviewer-loaded').waitFor();
        if (route === 'assistant') await page.locator('#page-assistant form:visible').first().waitFor();
        const dimensions = await page.evaluate(() => ({ viewport: innerWidth, content: document.documentElement.scrollWidth }));
        check(route + ' fits the ' + width + 'px viewport', dimensions.content <= dimensions.viewport, dimensions);
      }
      check('Responsive routes retain a single active page at ' + width + 'px', await page.locator('.page:not([hidden])').count() === 1);
      check('No CSP violations at ' + width + 'px', await page.evaluate(() => window.siteCSPViolations.length === 0));
    }
    await page.setViewportSize({ width: 390, height: 844 });
    await page.evaluate(() => { location.hash = 'datasets'; });
    await page.locator('#sample-search').fill('GSM2751831');
    await openMenu();
    check('Mobile drawer exposes all six existing routes', await drawer().locator('a[data-route]').count() === 6);
    const surface = await drawer().evaluate(el => ({ background: getComputedStyle(el).backgroundColor, width: el.getBoundingClientRect().width, left: el.getBoundingClientRect().left }));
    check('Mobile drawer keeps an opaque bounded surface when portaled', surface.background === 'rgb(255, 255, 255)' && surface.width <= 320 && surface.left >= 0, surface);
    const links = await drawer().locator('a[data-route]').evaluateAll(links => links.map(link => ({ top: link.getBoundingClientRect().top, left: link.getBoundingClientRect().left, right: link.getBoundingClientRect().right })));
    check('Portaled mobile navigation retains one readable column', links.every((link, index) => link.left >= 0 && link.right <= 390 && (!index || link.top > links[index - 1].top)), { links });
    const access = page.locator('main > header .team-access');
    if (await access.count()) {
      check('Mobile drawer preserves the authenticated workspace link', await drawer().locator('.workspace-navigation-access').getAttribute('href') === await access.getAttribute('href'));
    }
    check('Mobile drawer moves focus inside its modal', await drawer().evaluate(el => el.contains(document.activeElement)));
    await page.keyboard.press('Shift+Tab');
    check('Mobile drawer keeps keyboard focus within its modal', await drawer().evaluate(el => el.contains(document.activeElement)));
    await page.keyboard.press('Escape');
    await drawer().waitFor({ state: 'hidden' });
    check('Escape returns focus to the mobile menu trigger', await menu().evaluate(el => el === document.activeElement));
    for (const route of routes) {
      await openMenu();
      await drawer().locator('a[data-route="' + route + '"]').click();
      await drawer().waitFor({ state: 'hidden' });
      await page.waitForFunction(route => location.hash === '#' + route && document.activeElement === document.getElementById('main-content'), route);
      check('Mobile menu opens ' + route + ' and focuses its content', await page.locator('#page-' + route).isVisible());
    }
    await page.evaluate(() => { location.hash = 'datasets'; });
    check('Mobile navigation retains existing dataset search', await page.locator('#sample-search').inputValue() === 'GSM2751831' && await page.locator('#sample-rows tr').count() === 1);
    check('Mobile navigation retains URL query parameters', new URL(page.url()).searchParams.get('comparison') === 'final');
    await openMenu();
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.waitForFunction(() => !document.querySelector('#workspace-navigation').open && document.activeElement === document.getElementById('main-content'));
    check('Resizing an open drawer restores desktop navigation and focus', await page.locator('body > aside nav a[data-route="analysis"]').isVisible() && !await menu().isVisible());
    check('Normal navigation produces no browser exceptions', receipt.pageErrors.length === 0, { errors: receipt.pageErrors });

    await page.setViewportSize({ width: 320, height: 844 });
    await page.route('**/static/vendor/ui-sdk/**/drawer.js', route => route.abort());
    await page.goto(root + '/?drawer=unavailable#datasets');
    await page.waitForFunction(() => document.querySelector('#dataset-study').hidden);
    check('An unavailable drawer preserves the searchable study picker', await page.getByRole('combobox', { name: 'Study', exact: true }).isVisible());
    check('An unavailable drawer preserves the command palette', await page.getByRole('button', { name: 'Open command palette', exact: true }).isVisible());
    check('An unavailable drawer leaves every original navigation link visible', (await page.locator('body > aside nav a').evaluateAll(links => links.every(link => link.checkVisibility()))));
    check('Drawer fallback has no horizontal page overflow', await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.unroute('**/static/vendor/ui-sdk/**/drawer.js');
    await page.route('**/static/vendor/ui-sdk/**', route => route.abort());
    await page.goto(root + '/?sdk=unavailable#datasets');
    await page.waitForFunction(() => document.querySelector('#sample-rows tr'));
    await page.locator('#dataset-study').selectOption('GSE8917');
    check('Full SDK fallback keeps the original working dataset filter', await page.locator('#dataset-study').isVisible() && await page.locator('#sample-rows tr').count() === 25);
    await page.locator('body > aside nav a[data-route="analysis"]').click();
    await page.waitForFunction(() => document.querySelector('#analysis-table tbody tr'));
    check('Full SDK fallback keeps the scientific chart data', await page.locator('#analysis-table tbody tr').count() === 145);
    await page.unroute('**/static/vendor/ui-sdk/**');
    receipt.status = 'passed';
  } catch (error) {
    receipt.status = 'failed'; receipt.error = String(error);
  } finally {
    await page.unroute('**/static/vendor/ui-sdk/**/drawer.js');
    await page.unroute('**/static/vendor/ui-sdk/**');
    page.off('pageerror', errors);
    receipt.completedAt = new Date().toISOString();
    await page.evaluate(value => { window.siteNavigationTestReceipt = value; }, receipt);
  }
}
