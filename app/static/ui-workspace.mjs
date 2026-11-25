const sdk = './vendor/ui-sdk/0.1.0-beta.2/web/';

function element(tag, attributes = {}, text) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

async function enhanceStudyFilter() {
  const native = document.getElementById('dataset-study');
  const field = document.getElementById('dataset-study-field');
  if (!native || !field) return;
  const label = field.querySelector('label');
  const picker = element('ui-select', { id: 'workspace-study-picker', class: 'workspace-study-picker' });
  picker.itemToLabel = value => [...native.options].find(option => option.value === value)?.label ?? String(value ?? '');
  const trigger = element('ui-select-trigger');
  const button = element('button', { type: 'button' });
  button.append(element('ui-select-value'), element('span', { class: 'workspace-picker-chevron', 'aria-hidden': 'true' }, '⌄'));
  trigger.append(button);
  const content = element('ui-select-content', { hidden: '' });
  const search = element('ui-select-input');
  search.append(element('input', { type: 'search', 'aria-label': 'Search studies', placeholder: 'Search a study…', autocomplete: 'off', spellcheck: 'false' }));
  const list = element('ui-select-list');
  content.append(search, element('ui-select-empty', {}, 'No studies match your search.'), list);
  picker.append(element('ui-select-label', {}, label.textContent), trigger, content);

  let synchronizing = false;
  function syncOptions() {
    synchronizing = true;
    try {
      const options = [...native.options];
      list.replaceChildren(...options.map(option => {
        const item = element('ui-select-item', { value: option.value }, option.label);
        if (option.disabled) item.setAttribute('disabled', '');
        return item;
      }));
      picker.disabled = native.disabled;
      picker.value = native.value;
    } finally {
      synchronizing = false;
    }
  }
  picker.addEventListener('valuechange', event => {
    if (synchronizing) return;
    const value = event.detail.value;
    if (native.disabled || ![...native.options].some(option => option.value === value && !option.disabled)) {
      event.preventDefault();
      return;
    }
    if (native.value === value) return;
    synchronizing = true;
    try {
      native.value = value;
      native.dispatchEvent(new Event('change', { bubbles: true }));
    } finally {
      synchronizing = false;
    }
  });
  native.addEventListener('change', () => {
    if (!synchronizing) picker.value = native.value;
  });
  field.append(picker);
  try {
    await new Promise(resolve => requestAnimationFrame(resolve));
    if (!picker.api) throw new Error('Study picker did not initialize');
    syncOptions();
    new MutationObserver(syncOptions).observe(native, { childList: true, subtree: true, attributes: true, characterData: true });
    native.hidden = true;
    label.hidden = true;
  } catch (error) {
    picker.remove();
    throw error;
  }
}

function mountCommandPalette() {
  const header = document.querySelector('main > header');
  if (!header) return;
  const hints = {
    overview: ['Research workspace and study sources', ['home', 'sources']],
    datasets: ['Study filters, profiles and CSV downloads', ['samples', 'cohorts', 'groups', 'export']],
    analysis: ['PCA, heatmaps, correlations and quality review', ['plots', 'heatmap', 'pca', 'qc', 'charts']],
    experiments: ['Model comparisons and recorded results', ['models', 'results', 'training']],
    predictions: ['Profile compatibility and dose estimates', ['upload', 'inference', 'dose']],
    assistant: ['Questions, records and cited answers', ['chat', 'copilot', 'evidence']]
  };
  const routes = [...document.querySelectorAll('aside nav a[data-route]')]
    .filter(link => hints[link.dataset.route] && document.getElementById('page-' + link.dataset.route))
    .map(link => ({ value: link.dataset.route, label: link.textContent.trim(), description: hints[link.dataset.route][0], keywords: hints[link.dataset.route][1] }));
  const palette = element('ui-dialog', { id: 'workspace-command-palette', class: 'workspace-command-palette' });
  const trigger = element('ui-dialog-trigger', { shortcut: 'mod+k' });
  const button = element('button', { type: 'button', class: 'workspace-command-trigger', 'aria-label': 'Open command palette' });
  button.append(element('span', {}, 'Search workspace'), element('ui-dialog-shortcut'));
  trigger.append(button);
  const content = element('ui-dialog-content', { class: 'workspace-command-content', hidden: '', 'initial-focus': '#workspace-command-input' });
  const heading = element('div', { class: 'workspace-command-heading' });
  const close = element('ui-dialog-close');
  close.append(element('button', { type: 'button', 'aria-label': 'Close command palette' }, 'Close'));
  heading.append(element('ui-dialog-title', {}, 'Go to workspace'), close);
  const command = element('ui-command', { id: 'workspace-commands' });
  const input = element('ui-command-input');
  input.append(element('input', { id: 'workspace-command-input', 'aria-label': 'Search workspace pages', placeholder: 'Search pages or topics…', autocomplete: 'off', spellcheck: 'false' }));
  const list = element('ui-command-list', { 'aria-label': 'Workspace pages' });
  list.append(...routes.map(route => {
    const item = element('ui-command-item', { value: route.value, label: route.label });
    item.keywords = route.keywords;
    item.append(element('span', { class: 'workspace-command-name' }, route.label), element('small', {}, route.description));
    return item;
  }));
  command.append(input, element('ui-command-empty', {}, 'No pages match your search.'), list);
  content.append(heading, element('ui-dialog-description', {}, 'Find a page by name or topic.'), command);
  palette.append(trigger, content);
  palette.addEventListener('openchange', event => {
    if (event.detail.value) {
      command.query = '';
    }
  });
  command.addEventListener('action', event => {
    const route = routes.find(candidate => candidate.value === event.detail.value);
    if (!route) return;
    palette.open = false;
    location.hash = route.value;
    requestAnimationFrame(() => {
      if (!palette.open) document.getElementById('main-content').focus({ preventScroll: true });
    });
  });
  header.classList.add('workspace-header');
  header.append(palette);
}

function mountMobileNavigation() {
  const header = document.querySelector('main > header');
  const source = document.querySelector('body > aside nav');
  if (!header || !source) return;
  const drawer = element('ui-drawer', { id: 'workspace-navigation', class: 'workspace-navigation', side: 'left' });
  const trigger = element('ui-drawer-trigger');
  const button = element('button', { type: 'button', class: 'workspace-navigation-trigger', 'aria-label': 'Open workspace navigation' });
  const icon = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  icon.setAttribute('viewBox', '0 0 24 24');
  icon.setAttribute('aria-hidden', 'true');
  const lines = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  lines.setAttribute('d', 'M4 6h16M4 12h16M4 18h16');
  icon.append(lines);
  const menuHost = document.getElementById('workspace-menu-controls');
  const wordmark = menuHost?.querySelector('.menu-wordmark');
  button.append(icon, wordmark?.cloneNode(true) ?? element('span', {}, 'Menu'));
  trigger.append(button);
  const backdrop = element('ui-drawer-backdrop', { class: 'workspace-navigation-backdrop', hidden: '' });
  const content = element('ui-drawer-content', { class: 'workspace-navigation-content', hidden: '' });
  const heading = element('div', { class: 'workspace-navigation-heading' });
  const close = element('ui-drawer-close');
  close.append(element('button', { type: 'button', class: 'workspace-navigation-close', 'aria-label': 'Close workspace navigation' }, 'Close'));
  heading.append(element('ui-drawer-title', {}, 'Research workspace'), close);
  const nav = element('nav', { 'aria-label': 'Workspace sections' });
  const links = [...source.querySelectorAll('a[data-route]')];
  for (const original of links) {
    const link = original.cloneNode(true);
    link.addEventListener('click', () => {
      drawer.open = false;
      requestAnimationFrame(() => document.getElementById('main-content').focus({ preventScroll: true }));
    });
    nav.append(link);
  }
  content.append(heading, element('ui-drawer-description', {}, 'Explore the research and continue your work.'), nav);
  const access = header.querySelector('.team-access');
  if (access) {
    const link = access.cloneNode(true);
    link.className = 'workspace-navigation-access';
    content.append(link);
  }
  drawer.append(trigger, backdrop, content);
  (menuHost ?? header).prepend(drawer);
  document.body.dataset.mobileNavigation = 'ready';
  const desktop = matchMedia('(min-width: 901px)');
  desktop.addEventListener('change', event => {
    if (event.matches && drawer.open) {
      drawer.open = false;
      requestAnimationFrame(() => document.getElementById('main-content').focus({ preventScroll: true }));
    }
  });
}

async function start() {
  try {
    await Promise.all(['select', 'command', 'dialog'].map(name => import(sdk + name + '.js')));
    if (document.readyState === 'loading') await new Promise(resolve => document.addEventListener('DOMContentLoaded', resolve, { once: true }));
    const results = await Promise.allSettled([enhanceStudyFilter(), Promise.resolve().then(mountCommandPalette), import(sdk + 'drawer.js').then(mountMobileNavigation)]);
    for (const result of results) if (result.status === 'rejected') console.warn('An optional workspace control could not load.', result.reason);
  } catch {
    console.warn('Optional workspace controls are unavailable; the original navigation and study filter remain available.');
  }
}

start();
