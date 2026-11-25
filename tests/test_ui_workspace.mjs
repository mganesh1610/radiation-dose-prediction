import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const read = name => fs.readFileSync(path.join(root, name), 'utf8');
const prefix = 'vendor/ui-sdk/0.1.0-beta.2/';
const assets = [...read('scripts/ui_sdk_assets.py').matchAll(/'([^']+)'/g)].map(match => match[1]);
const imports = /(?:from\s*|import\s*\(?\s*)["']([^"']+)["']/g;

test('selected browser modules resolve completely within the pinned local asset list', () => {
  const pending = ['web/select.js', 'web/command.js', 'web/dialog.js', 'web/drawer.js'];
  const visited = new Set();
  while (pending.length) {
    const name = pending.pop();
    if (visited.has(name)) continue;
    visited.add(name);
    assert(assets.includes(prefix + name), `Missing packaged module: ${name}`);
    const code = read('app/static/' + prefix + name);
    for (const [, specifier] of code.matchAll(imports)) {
      assert(specifier.startsWith('.'), `External SDK dependency: ${specifier}`);
      const dependency = path.posix.normalize(path.posix.join(path.posix.dirname(name), specifier));
      assert(!dependency.startsWith('../'), 'A dependency escaped the pinned version');
      pending.push(dependency);
    }
  }
  assert.equal(visited.size + 2, assets.length, 'Ship exactly the module closure, license and attribution');
  assert.equal(assets.length, new Set(assets).size);
  for (const asset of assets) assert(fs.statSync(path.join(root, 'app/static', asset)).isFile());
  assert(read('app/static/' + prefix + 'LICENSE').includes('Apache License'));
});

test('both deployment packages include the same selected modules and application adapter', () => {
  for (const name of ['scripts/public_site_assets.py', 'scripts/package_web.py']) {
    const source = read(name);
    assert(source.includes('from ui_sdk_assets import UI_SDK_FILES'));
    assert(source.includes('ui-workspace.mjs') && source.includes('ui-workspace.css') && source.includes('workspace-system.css'));
  }
  assert(!read('app/static/ui-workspace.mjs').includes('https://'));
  assert(!read('app/static/ui-workspace.css').includes('url('), 'No new remote or global-theme assets');
});

test('both page variants retain native filter/export contracts and declare the CSP style mode', () => {
  for (const name of ['app/static/index.html', 'app/public/index.html']) {
    const html = read(name);
    for (const id of ['dataset-study', 'sample-search', 'download-samples', 'analysis-chart', 'analysis-study', 'main-content']) {
      assert.equal((html.match(new RegExp(`id="${id}"`, 'g')) || []).length, 1, `${name}: ${id}`);
    }
    assert(html.includes('<label for="dataset-study">Study</label>'));
    assert(html.includes('<select id="dataset-study"><option value="all">All core studies</option>'));
    assert(html.includes('<meta property="csp-nonce">'));
    assert(html.includes('type="module" src="/static/ui-workspace.mjs"'));
    assert(html.includes('<body class="workspace-system">'));
    assert(html.includes('href="/static/workspace-system.css"'));
    assert(html.indexOf('property="csp-nonce"') < html.indexOf('ui-workspace.mjs'));
  }
});

test('private and public security policies still require same-origin scripts and styles', () => {
  for (const source of [read('app/server.py'), read('app/public/vercel.json')]) {
    assert(source.includes("script-src 'self'"));
    assert(source.includes("style-src 'self'"));
    assert(!source.includes('unsafe-inline') && !source.includes('unsafe-eval'));
  }
});

test('the Overview film and navigation ship in both site templates and packages', () => {
  const files = ['overview-dna.css', 'overview-dna.js', 'workspace-navigation.css', 'workspace-navigation.js',
    ...['overview-dna', 'overview-dna-wide'].flatMap(name => ['mp4', 'webp'].map(ext => 'assets/' + name + '.' + ext))];
  for (const template of ['app/public/index.html', 'app/static/index.html']) {
    const html = read(template);
    for (const file of files) assert.ok(html.includes('/static/' + file), template + ' is missing ' + file);
    for (const id of ['dna-film', 'dna-motion', 'sources', 'workspace-sidebar', 'sidebar-toggle', 'workspace-toolbar',
      'overview-toolbar-toggle', 'workspace-menu-controls', 'explore-datasets']) {
      assert.equal(html.split('id="' + id + '"').length, 2, template + ' must contain exactly one ' + id);
    }
  }
  for (const file of files) assert.ok(read('scripts/overview_ui_assets.py').includes("'" + file + "'"), file);
  for (const source of ['scripts/public_site_assets.py', 'scripts/package_web.py']) {
    assert.ok(read(source).includes('*OVERVIEW_UI_FILES'), source);
  }
});
