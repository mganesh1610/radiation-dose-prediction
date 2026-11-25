'use strict';

(() => {
  const body = document.body;
  const sidebar = document.getElementById('workspace-sidebar');
  const menu = document.getElementById('sidebar-toggle');
  const overview = document.getElementById('page-overview');
  const toolbar = document.getElementById('workspace-toolbar');
  const reveal = document.getElementById('overview-toolbar-toggle');
  const desktop = matchMedia('(min-width: 901px)');
  const hover = matchMedia('(hover: hover)');
  const storageKey = 'defender.sidebar';
  let preference;
  try { preference = localStorage.getItem(storageKey); } catch { /* The menu also works without storage. */ }
  let home = false;
  let lastScroll = 0;
  let hideTimer;

  function syncSidebar() {
    const expanded = preference === 'expanded' || (preference !== 'collapsed' && !home);
    body.dataset.sidebar = expanded ? 'expanded' : 'collapsed';
    menu.setAttribute('aria-expanded', String(expanded));
    menu.title = expanded ? 'Collapse navigation' : 'Expand navigation';
    menu.setAttribute('aria-label', menu.title);
    if (desktop.matches && !expanded && sidebar.contains(document.activeElement)) menu.focus({preventScroll: true});
    sidebar.inert = desktop.matches && !expanded;
  }

  function toggleSidebar() {
    preference = body.dataset.sidebar === 'expanded' ? 'collapsed' : 'expanded';
    try { localStorage.setItem(storageKey, preference); } catch { /* Keep the choice for this visit. */ }
    syncSidebar();
  }

  function showToolbar(open) {
    clearTimeout(hideTimer);
    const state = open ? 'open' : 'closed';
    if (!home || body.dataset.homeToolbar === state) return;
    body.dataset.homeToolbar = state;
    reveal.setAttribute('aria-expanded', String(open));
    reveal.title = open ? 'Hide workspace bar' : 'Show workspace bar';
    reveal.setAttribute('aria-label', reveal.title);
    if (!open && toolbar.contains(document.activeElement)) reveal.focus({preventScroll: true});
  }

  function toolbarInUse() {
    return toolbar.contains(document.activeElement) || document.getElementById('workspace-command-palette')?.open;
  }

  function hideWhenIdle() {
    clearTimeout(hideTimer);
    if (!home || body.dataset.homeToolbar !== 'open') return;
    hideTimer = setTimeout(() => {
      if (toolbarInUse() || (hover.matches && toolbar.matches(':hover'))) return;
      if (document.activeElement === reveal && reveal.matches(':focus-visible')) return;
      showToolbar(false);
    }, 2200);
  }

  function syncRoute() {
    home = !overview.hidden;
    clearTimeout(hideTimer);
    lastScroll = Math.max(0, window.scrollY);
    syncSidebar();
    reveal.hidden = !home;
    if (home) showToolbar(false);
    else delete body.dataset.homeToolbar;
  }

  menu.addEventListener('click', toggleSidebar);
  menu.hidden = false;
  desktop.addEventListener('change', syncSidebar);
  reveal.addEventListener('click', () => {
    showToolbar(body.dataset.homeToolbar !== 'open');
    hideWhenIdle();
  });
  toolbar.addEventListener('pointerleave', hideWhenIdle);
  toolbar.addEventListener('focusout', hideWhenIdle);
  reveal.addEventListener('blur', hideWhenIdle);
  toolbar.addEventListener('openchange', event => {
    if (event.target.id !== 'workspace-command-palette') return;
    if (event.detail.value) showToolbar(true);
    else hideWhenIdle();
  });

  document.addEventListener('pointermove', event => {
    if (!home || event.pointerType !== 'mouse' || event.clientY > 8 || body.dataset.homeToolbar === 'open') return;
    if (sidebar.contains(event.target) || event.target.closest('#workspace-menu-controls, #overview-toolbar-toggle')) return;
    showToolbar(true);
    hideWhenIdle();
  }, {passive: true});

  window.addEventListener('scroll', () => {
    if (!home) return;
    const top = Math.max(0, window.scrollY);
    if (Math.abs(top - lastScroll) < 12) return;
    const movingUp = top < lastScroll;
    lastScroll = top;
    if (toolbarInUse()) return;
    showToolbar(movingUp);
    if (movingUp) hideWhenIdle();
  }, {passive: true});

  window.addEventListener('keydown', event => {
    if (home && (event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
      showToolbar(true);
      hideWhenIdle();
    }
    if (event.key !== 'Escape' || document.querySelector(':modal') || document.getElementById('workspace-navigation')?.open ||
        document.getElementById('workspace-command-palette')?.open) return;
    if (home && body.dataset.homeToolbar === 'open') {
      showToolbar(false);
    } else if (desktop.matches && body.dataset.sidebar === 'expanded' && (sidebar.contains(event.target) || event.target === menu)) {
      toggleSidebar();
      menu.focus({preventScroll: true});
    }
  }, true);

  window.addEventListener('hashchange', syncRoute);
  syncRoute();
})();
