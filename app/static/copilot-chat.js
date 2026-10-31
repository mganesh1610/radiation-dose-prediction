'use strict';
(() => {
  const el = (tag, text, cls) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (cls) node.className = cls;
    return node;
  };
  const icon = name => {
    const img = el('img', undefined, 'copilot-icon');
    img.src = '/static/icons/phosphor/' + name + '.svg'; img.alt = ''; img.width = img.height = 20;
    return img;
  };
  const button = (label, name, cls = 'copilot-icon-button') => {
    const node = el('button', undefined, cls); node.type = 'button';
    node.title = label; node.setAttribute('aria-label', label);
    if (name) node.append(icon(name));
    return node;
  };
  const safeLink = (label, href) => {
    try {
      const url = new URL(href, location.origin);
      if (url.protocol !== 'https:' && !(href.startsWith('#') && url.origin === location.origin)) return el('span', label);
      const a = el('a', label); a.href = url.href;
      if (url.origin !== location.origin) { a.target = '_blank'; a.rel = 'noopener noreferrer'; }
      return a;
    } catch { return el('span', label); }
  };
  // Model text is always parsed into text nodes; HTML is never evaluated.
  function inline(parent, text) {
    const pattern = /(\*\*([^*\n]+)\*\*|`([^`\n]+)`|\[([^\]\n]+)\]\(([^\s)]+)\))/g;
    let end = 0;
    for (const match of text.matchAll(pattern)) {
      parent.append(document.createTextNode(text.slice(end, match.index)));
      parent.append(match[2] ? el('strong', match[2]) : match[3] ? el('code', match[3]) : safeLink(match[4], match[5]));
      end = match.index + match[0].length;
    }
    parent.append(document.createTextNode(text.slice(end)));
  }
  function markdown(parent, text) {
    const lines = String(text).replace(/\r/g, '').split('\n'); let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) { i++; continue; }
      if (/^```/.test(line)) {
        const code = []; i++;
        while (i < lines.length && !/^```/.test(lines[i])) code.push(lines[i++]);
        const pre = el('pre'); pre.append(el('code', code.join('\n'))); parent.append(pre); i++; continue;
      }
      if (/^#{1,6}\s/.test(line)) {
        const heading = el('h3'); inline(heading, line.replace(/^#{1,6}\s+/, '')); parent.append(heading); i++; continue;
      }
      if (/^\s*(?:[-*]|\d+[.])\s+/.test(line)) {
        const ordered = /^\s*\d+[.]/.test(line), list = el(ordered ? 'ol' : 'ul');
        while (i < lines.length && /^\s*(?:[-*]|\d+[.])\s+/.test(lines[i])) {
          const item = el('li'); inline(item, lines[i++].replace(/^\s*(?:[-*]|\d+[.])\s+/, '')); list.append(item);
        }
        parent.append(list); continue;
      }
      if (line.includes('|') && i + 1 < lines.length && /^\s*\|?\s*:?-{3}/.test(lines[i + 1])) {
        const wrap = el('div', undefined, 'copilot-table'), table = el('table');
        const cells = value => value.trim().replace(/^\||\|$/g, '').split('|').map(x => x.trim());
        const head = el('tr'); for (const value of cells(line)) { const cell = el('th'); inline(cell, value); head.append(cell); }
        const thead = el('thead'); thead.append(head); table.append(thead); i += 2;
        const tbody = el('tbody');
        while (i < lines.length && lines[i].includes('|') && lines[i].trim()) {
          const row = el('tr'); for (const value of cells(lines[i++])) { const cell = el('td'); inline(cell, value); row.append(cell); } tbody.append(row);
        }
        table.append(tbody); wrap.append(table); parent.append(wrap); continue;
      }
      const paragraph = el('p'), content = [line]; i++;
      while (i < lines.length && lines[i].trim() && !/^\s*(?:```|#{1,6}\s|[-*]\s|\d+[.]\s)/.test(lines[i])) content.push(lines[i++]);
      inline(paragraph, content.join('\n')); parent.append(paragraph);
    }
  }
  const api = async (path, method = 'GET', body) => {
    const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 195000);
    try {
      const response = await fetch('/api/copilot/' + path, {
        method, credentials: 'same-origin', signal: controller.signal,
        headers: method === 'GET' ? {} : {'Content-Type': 'application/json', 'X-Defender-Request': '1'},
        ...(body === undefined ? {} : {body: JSON.stringify(body)})
      });
      const data = await response.json(); if (!response.ok) throw Error(data.error || 'Chat request failed'); return data;
    } finally { clearTimeout(timer); }
  };
  const route = () => document.body.classList.toggle('assistant-workspace', location.hash === '#assistant');
  window.addEventListener('hashchange', route); route();
  document.querySelector('.skip-link')?.addEventListener('click', event => {
    if (location.hash === '#assistant') {event.preventDefault(); document.getElementById('copilot-question')?.focus();}
  });

  async function workspace(host, options = {}) {
    const records = Boolean(options.records), titles = new Map(), histories = new Map();
    let chat = null, chats = [], pending = null, busy = false, latestReceipt = null, messageItems = [], createKey = null, stopRequested = false;
    let runtimeReady = records || options.executionReady !== false;
    const shell = el('div', undefined, 'copilot-shell');
    const sidebar = el('div', undefined, 'copilot-sidebar'); sidebar.id = 'copilot-sidebar';
    const brand = el('a', undefined, 'copilot-brand'); brand.href = '#overview';
    brand.append(icon('flask'), el('span', 'Research workspace'));
    const hideSidebar = button('Close chat history', 'sidebar-simple');
    const sidebarBrand = el('div', undefined, 'copilot-sidebar-brand'); sidebarBrand.append(brand, hideSidebar);
    const fresh = button('New chat', 'note-pencil', 'copilot-new-chat'); fresh.append(el('span', 'New chat'));
    const find = button('Search chats', 'magnifying-glass', 'copilot-search-toggle'); find.append(el('span', 'Search chats'));
    const search = el('input', undefined, 'copilot-chat-search'); search.type = 'search'; search.placeholder = 'Find a chat by title or date';
    search.setAttribute('aria-label', 'Search saved chats'); search.hidden = true;
    const returnLink = el('a', undefined, 'copilot-return'); returnLink.href = '#overview'; returnLink.append(icon('squares-four'), el('span', 'Back to research'));
    const historyLabel = el('p', 'Your chats', 'copilot-list-label');
    const chatList = el('div', undefined, 'copilot-chat-list'); chatList.setAttribute('aria-label', 'Saved chats');
    const sidebarFoot = el('div', undefined, 'copilot-sidebar-foot'); sidebarFoot.append(icon('shield-check'));
    const footText = el('div'); footText.append(el('strong', records ? 'Public research viewer' : 'Private workspace'), el('span', records ? 'This visit only' : 'History saved for 30 days')); sidebarFoot.append(footText);
    sidebar.append(sidebarBrand, fresh, find, search, returnLink, historyLabel, chatList, sidebarFoot);
    const scrim = button('Close chat history', undefined, 'copilot-sidebar-scrim'); scrim.hidden = true;
    const main = el('div', undefined, 'copilot-main');
    const header = el('div', undefined, 'copilot-header');
    const headerLeft = el('div', undefined, 'copilot-header-left');
    const toggle = button('Open chat history', 'sidebar-simple'); toggle.setAttribute('aria-controls', sidebar.id); toggle.setAttribute('aria-expanded', 'true');
    const title = el('div', undefined, 'copilot-title'); title.append(el('strong', 'Research copilot'), el('span', records ? 'Approved records' : 'Core research assistant'));
    headerLeft.append(toggle, title);
    const actions = el('div', undefined, 'copilot-header-actions');
    const scope = button('About this copilot', 'database', 'copilot-scope'); scope.append(el('span', records ? 'Published evidence' : '145 profiles'));
    const sources = button('Open sources', 'book-open', 'copilot-sources-button'); sources.append(el('span', 'Sources')); sources.disabled = true;
    const menu = el('details', undefined, 'copilot-menu'), summary = el('summary'); summary.setAttribute('role', 'button'); summary.setAttribute('aria-label', 'Conversation options'); summary.title = 'Conversation options'; summary.append(icon('dots-three'));
    const menuBody = el('div', undefined, 'copilot-menu-body');
    const refresh = button('Refresh conversation', 'arrows-clockwise', 'copilot-menu-item'); refresh.append(el('span', 'Refresh conversation'));
    const about = button('About this copilot', 'info', 'copilot-menu-item'); about.append(el('span', 'About this copilot'));
    const remove = button('Delete chat', 'trash', 'copilot-menu-item copilot-danger'); remove.append(el('span', 'Delete conversation'));
    menuBody.append(refresh, about, remove); menu.append(summary, menuBody); actions.append(scope, sources, menu); header.append(headerLeft, actions);
    if (!records && options.memoryEnabled) {
      const memory = button('Memory preferences', 'book-open', 'copilot-menu-item'); memory.append(el('span', 'Memory preferences'));
      menuBody.insertBefore(memory, remove);
      memory.onclick = async () => {
        menu.open = false;
        const pane = dialog('Memory preferences'), content = el('div', undefined, 'copilot-evidence-content');
        content.append(el('p', 'Save presentation preferences for your account across chats. Each save expires after 30 days. Conversations and scientific data are not automatically added to memory.'));
        const feedback = el('p', 'Loading…'); feedback.setAttribute('role', 'status'); content.append(feedback); pane.append(content); pane.showModal();
        try {
          const data = await api('memories'); feedback.textContent = '';
          for (const item of data.preferences) {
            const row = el('div', undefined, 'copilot-preference');
            const label = el('label', item.key === 'response_style' ? 'Response length' : 'Answer format');
            const select = el('select'); select.id = 'pref-' + item.key; label.htmlFor = select.id;
            for (const value of item.choices) {const option = el('option', value); option.value = value; select.append(option);}
            if (item.value) select.value = item.value;
            const note = el('p', item.value ? 'Saved: ' + item.value : 'Not saved (' + item.status + ')');
            const save = el('button', 'Save'), forget = el('button', 'Forget'); save.type = forget.type = 'button';
            forget.disabled = item.revision === 0;
            let requestId = null, requestedValue = null;
            const act = async deleting => {
              save.disabled = forget.disabled = true; feedback.textContent = deleting ? 'Forgetting…' : 'Saving…';
              try {
                if (!deleting && requestedValue !== select.value) {requestId = crypto.randomUUID(); requestedValue = select.value;}
                const updated = await api('memories/' + item.key, deleting ? 'DELETE' : 'PATCH', deleting ?
                  {expected_revision: item.revision} : {request_id: requestId, expected_revision: item.revision, value: requestedValue});
                Object.assign(item, updated.preferences.find(p => p.key === item.key));
                requestId = requestedValue = null; note.textContent = item.value ? 'Saved: ' + item.value : 'Forgotten'; feedback.textContent = 'Updated.';
              } catch (e) {feedback.textContent = e.message + '. Retry the same action, or close and reopen to refresh.';}
              finally {save.disabled = false; forget.disabled = item.revision === 0;}
            };
            save.onclick = () => act(false); forget.onclick = () => act(true);
            row.append(label, select, note, save, forget); content.append(row);
          }
        } catch (e) {feedback.textContent = e.message;}
      };
    }
    const scroll = el('div', undefined, 'copilot-scroll');
    const messages = el('div', undefined, 'copilot-messages'); messages.setAttribute('role', 'log'); messages.setAttribute('aria-label', 'Conversation'); messages.setAttribute('aria-live', 'polite');
    scroll.append(messages);
    const bottom = el('div', undefined, 'copilot-bottom');
    const state = el('p', '', 'copilot-status'); state.setAttribute('role', 'status'); state.setAttribute('aria-live', 'polite'); state.hidden = true;
    const retry = button('Check or retry the same request', 'arrows-clockwise', 'copilot-retry'); retry.append(el('span', 'Check / retry')); retry.hidden = true;
    const form = el('form', undefined, 'copilot-composer');
    const label = el('label', 'Research question', 'sr-only'), input = el('textarea'); input.id = 'copilot-question'; label.htmlFor = input.id;
    input.maxLength = records ? 500 : 2000; input.rows = 1; input.required = true; input.placeholder = 'Ask about the research';
    input.setAttribute('autocomplete', 'off');
    const formTools = el('div', undefined, 'copilot-composer-tools');
    const modeLabel = el('span', undefined, 'copilot-mode'); modeLabel.append(icon(records ? 'book-open' : 'database'), el('span', records ? 'Approved records' : 'Core records'));
    const send = button('Send message', 'arrow-up', 'copilot-send'); send.type = 'submit';
    const stop = button('Stop response', 'stop', 'copilot-send'); stop.hidden = true;
    const formActions = el('div', undefined, 'copilot-form-actions'); formActions.append(retry, send, stop); formTools.append(modeLabel, formActions); form.append(label, input, formTools);
    let attach;
    if (!records && options.predictionEnabled) {
      attach=button('Add expression profile','flask','copilot-attach');
      attach.append(el('span','Add profile')); formTools.insertBefore(attach,modeLabel);
      attach.onclick=predictionDialog;
    }
    if (!records && options.trainingEnabled) {
      const training=button('Training jobs','flask','copilot-attach');training.append(el('span','Training'));
      formTools.insertBefore(training,modeLabel);training.onclick=trainingDialog;
    }
    const hint = el('p', records ? 'Answers from published records. Research use only.' : 'Check the sources. Recorded doses are separate from model estimates. Research use only.', 'copilot-composer-hint');
    bottom.append(state, form, hint); main.append(header, scroll, bottom); shell.append(sidebar, scrim, main); host.replaceChildren(shell);
    if (options.preview) { const badge = el('span', 'Local UI preview', 'copilot-preview-label'); title.append(badge); }

    function status(text = '', error = false) { state.textContent = text; state.hidden = !text; state.classList.toggle('is-error', error); }
    function resize() {
      input.style.height = 'auto';
      input.style.height = Math.min(Math.max(36, input.scrollHeight), 160) + 'px';
      input.style.overflowY = input.scrollHeight > 160 ? 'auto' : 'hidden';
    }
    function controls() {
      fresh.disabled = busy || Boolean(pending) || !runtimeReady; remove.disabled = busy || !chat; refresh.disabled = !chat;
      send.hidden = busy; stop.hidden = !busy; send.disabled = busy || Boolean(pending) || !runtimeReady || !input.value.trim();
      input.readOnly = Boolean(pending); sources.disabled = !latestReceipt;
      if (attach) attach.disabled=busy || Boolean(pending) || !runtimeReady;
      for (const node of chatList.querySelectorAll('button')) node.disabled = busy || Boolean(pending);
      messages.setAttribute('aria-busy', String(busy || !runtimeReady));
    }
    function mobileClose() {
      const wasOpen = shell.classList.contains('history-open');
      shell.classList.remove('history-open'); scrim.hidden = true; main.inert = false;
      toggle.setAttribute('aria-expanded', 'false'); if (wasOpen) toggle.focus();
    }
    function historyToggle() {
      if (matchMedia('(max-width: 760px)').matches) {
        const open = shell.classList.toggle('history-open'); scrim.hidden = !open; main.inert = open;
        toggle.setAttribute('aria-expanded', String(open));
        if (open) hideSidebar.focus(); else toggle.focus();
      } else {
        const collapsed = shell.classList.toggle('history-collapsed'); toggle.setAttribute('aria-expanded', String(!collapsed));
      }
    }
    toggle.onclick = historyToggle; hideSidebar.onclick = historyToggle; scrim.onclick = mobileClose;
    matchMedia('(max-width: 760px)').addEventListener('change', () => {
      mobileClose(); toggle.setAttribute('aria-expanded', String(!matchMedia('(max-width: 760px)').matches && !shell.classList.contains('history-collapsed')));
    });
    find.onclick = () => { search.hidden = !search.hidden; if (!search.hidden) search.focus(); else {search.value = ''; renderList();} };
    search.oninput = renderList;
    function urlChat(id) {
      if (records) return;
      const url = new URL(location.href); if (id) url.searchParams.set('chat', id); else url.searchParams.delete('chat');
      history.replaceState(null, '', url.pathname + url.search + url.hash);
    }
    function renderList() {
      chatList.replaceChildren();
      const filter = search.value.toLowerCase();
      for (const item of [...chats].sort((a, b) => b.created_at - a.created_at)) {
        const name = item.title || titles.get(item.chat_id) || 'New conversation';
        const date = new Date(item.created_at * 1000).toLocaleDateString(undefined, {month: 'short', day: 'numeric'});
        if (filter && !(name + ' ' + date).toLowerCase().includes(filter)) continue;
        const row = button(name, undefined, 'copilot-chat-row'); row.dataset.chatId = item.chat_id;
        row.append(el('span', name), el('small', date)); row.title = name + ' · ' + date;
        if (item.chat_id === chat?.chat_id) row.setAttribute('aria-current', 'true');
        row.onclick = () => openChat(item).catch(e => status(e.message, true)); chatList.append(row);
      }
      if (!chatList.children.length) chatList.append(el('p', filter ? 'No matching conversations.' : 'Your conversations will appear here.', 'copilot-list-empty'));
      controls();
    }
    function welcome() {
      const box = el('div', undefined, 'copilot-welcome');
      box.append(el('p', 'RADIATION RESEARCH', 'copilot-welcome-kicker'), el('h1', 'Explore the evidence.'), el('p', records ? 'Ask about the studies, models, and published results.' : 'Ask about samples, recorded doses, and gene expression.', 'copilot-welcome-description'));
      const suggestions = records ? [
        ['Compare the models', 'Did Model A outperform B, C and D?', 'flask'],
        ['Understand the data', 'Did clustering improve prediction?', 'database'],
        ['Look up a sample', 'What is the dose for GSM2751831?', 'magnifying-glass']
      ] : [
        ['Explore recorded doses', 'How many core profiles have a recorded dose of 2 Gy?', 'database'],
        ['Look up a sample', 'Show the recorded details for sample GSM2751831.', 'magnifying-glass'],
        ['Inspect gene expression', 'For sample GSM2751831, retrieve NCBI Gene ID 7157 expression values. State the scale and normalization.', 'flask']
      ];
      const rows = el('div', undefined, 'copilot-starters');
      for (const [name, question, image] of suggestions) {
        const node = button(name, image, 'copilot-starter'); node.append(el('span', name), icon('arrow-right'));
        node.onclick = () => {input.value = question; resize(); controls(); input.focus();}; rows.append(node);
      }
      box.append(rows); return box;
    }
    async function copyText(text, node) {
      try { await navigator.clipboard.writeText(text); node.replaceChildren(icon('check')); node.title = 'Copied'; }
      catch { status('Copy is unavailable in this browser. You can select the answer text.', true); }
    }
    function renderMessages(items) {
      messages.replaceChildren();
      shell.classList.toggle('has-messages', Boolean(items.length));
      if (!items.length) { messages.append(welcome()); return; }
      for (const message of items) {
        const article = el('article', undefined, 'copilot-message ' + message.role), body = el('div', undefined, 'copilot-message-body');
        if (message.role === 'user') body.append(el('p', message.text));
        else {
          const author = el('div', undefined, 'copilot-author'); author.append(icon('flask'), el('span', 'Research copilot')); article.append(author);
          markdown(body, message.text);
        }
        article.append(body);
        if (message.role !== 'user') {
          const tools = el('div', undefined, 'copilot-message-tools'), copy = button('Copy answer', 'copy');
          copy.onclick = () => copyText(message.text, copy); tools.append(copy);
          if (message.invocation_id?.startsWith('run-')) {
            const evidence = button('View sources for this answer', 'book-open', 'copilot-message-source'); evidence.append(el('span', 'Sources'));
            evidence.onclick = async () => {
              try { const receipt = records ? histories.get(chat.chat_id)?.receipt : await api('runs/' + message.invocation_id + '/receipt'); showSources(receipt); }
              catch (e) { status(e.message, true); }
            }; tools.append(evidence);
          }
          article.append(tools);
          if (latestReceipt?.run_id === message.invocation_id && !records) {renderAnalysis(article, latestReceipt); renderPrediction(article,latestReceipt);renderTraining(article,latestReceipt);}
        }
        messages.append(article);
      }
      scroll.scrollTop = scroll.scrollHeight;
    }
    function thinking() {
      renderMessages([...messageItems, {role: 'user', text: pending.text}]);
      const row = el('div', undefined, 'copilot-thinking'); row.append(icon('arrows-clockwise'), el('span', records ? 'Finding the approved record…' : 'Checking the approved records…')); messages.append(row); scroll.scrollTop = scroll.scrollHeight;
    }
    async function renderAnalysis(parent, receipt) {
      const sources = (receipt.receipts || []).filter(r => r.tool === 'analysis' && r.artifact);
      for (const source of sources) {
        const runId = receipt.run_id, ref = source.artifact;
        if (!/^run-[a-f0-9]{32}$/.test(runId || '') || !/^plot-[a-f0-9]{32}$/.test(ref.artifact_id || '')) continue;
        const card = el('figure', undefined, 'copilot-analysis');
        card.append(el('figcaption', 'Expression by recorded dose'));
        const note = el('p', 'Loading verified plot…'); card.append(note); parent.append(card);
        try {
          const result = await api('runs/' + runId + '/artifacts/' + ref.artifact_id);
          if (typeof result.png_base64 !== 'string' || result.png_base64.length > 300000 || !/^[A-Za-z0-9+/]+={0,2}$/.test(result.png_base64)) throw new Error('Invalid plot data');
          const bytes = Uint8Array.from(atob(result.png_base64), c => c.charCodeAt(0));
          const sha = async value => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', value)), b => b.toString(16).padStart(2, '0')).join('');
          if (await sha(bytes) !== ref.png_sha256 || typeof result.csv !== 'string' || await sha(new TextEncoder().encode(result.csv)) !== ref.csv_sha256) throw new Error('Artifact integrity check failed');
          const image = el('img'); image.src = 'data:image/png;base64,' + result.png_base64;
          const params = result.provenance.parameters;
          image.alt = params.study_id + ', NCBI Gene ' + params.gene_id + ': separate-probe expression by recorded laboratory dose, source linear intensity.';
          image.style.cssText = 'width:100%;height:auto;border-radius:12px'; card.append(image);
          const counts = result.summary;
          note.textContent = counts.profiles + ' profiles · ' + counts.study_local_donor_groups + ' study-local donor groups · ' + counts.missing_expression_cells + ' missing expression cells';
          card.append(el('p', result.limitations, 'copilot-evidence-note'));
          const downloads = el('div', undefined, 'copilot-message-tools');
          for (const [label, extension, blob] of [['Download PNG','png',new Blob([bytes], {type:'image/png'})], ['Download data CSV','csv',new Blob([result.csv], {type:'text/csv;charset=utf-8'})], ['Download provenance','json',new Blob([JSON.stringify({summary:result.summary,provenance:result.provenance,limitations:result.limitations},null,2)],{type:'application/json'})]]) {
            const control = el('button', label, 'copilot-message-source'); control.type = 'button';
            control.onclick = () => { const link = document.createElement('a'), url = URL.createObjectURL(blob); link.href = url; link.download = ref.artifact_id + '.' + extension; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); };
            downloads.append(control);
          }
          card.append(downloads);
        } catch (error) { note.textContent = 'Plot unavailable: ' + error.message + '. Refresh to retry.'; }
      }
    }
    async function list() { if (!records) chats = (await api('chats')).chats; renderList(); }
    async function loadHistory() {
      if (!chat) return;
      const id = chat.chat_id; let data, items = [], offset = 0;
      if (records) { const saved = histories.get(id); data = {chat, messages: saved?.messages || []}; items = data.messages; latestReceipt = saved?.receipt || null; }
      else {
        do { data = await api('chats/' + id + '/messages?offset=' + offset); items.push(...data.messages); offset = data.next_offset; } while (offset !== null);
        if (chat?.chat_id !== id) return;
        const latest = [...items].reverse().find(m => m.invocation_id?.startsWith('run-'));
        latestReceipt = latest ? await api('runs/' + latest.invocation_id + '/receipt') : null;
      }
      if (chat?.chat_id !== id) return;
      chat = data.chat; messageItems = items;
      const first = items.find(m => m.role === 'user'); if (first) titles.set(id, first.text.replace(/\s+/g, ' ').slice(0, 72));
      chats = chats.map(item => item.chat_id === id ? {...item, ...chat} : item);
      busy = Boolean(chat.active_run); renderMessages(items); renderList(); controls();
      status(busy ? 'A response is in progress. Refresh to check it, or stop the turn.' : '');
    }
    async function openChat(item) {
      if (busy || pending) return;
      chat = item; pending = null; retry.hidden = true; status('Loading conversation…'); mobileClose(); urlChat(item.chat_id);
      await loadHistory();
    }
    let navigationEpoch = 0;
    function draft() {
      if (busy || pending) return;
      navigationEpoch++;
      chat = null; createKey = null; messageItems = []; latestReceipt = null; input.value = ''; retry.hidden = true;
      status(); renderMessages([]); renderList(); controls(); resize(); urlChat(null); mobileClose(); input.focus();
    }
    fresh.onclick = draft;
    async function sendMessage() {
      if (!pending || busy) return;
      busy = true; stopRequested = false; retry.hidden = true; controls(); status(); thinking();
      try {
        if (!chat) {
          if (records) { chat = {chat_id: 'visit-' + crypto.randomUUID(), created_at: Date.now() / 1000, active_run: null}; chats.push(chat); }
          else { createKey ||= crypto.randomUUID(); chat = await api('chats', 'POST', {request_id: createKey}); chats.push(chat); }
          urlChat(chat.chat_id);
        }
        if (stopRequested) {
          pending = null; input.value = ''; resize(); await loadHistory();
          status('Stopped before the question was sent.'); return;
        }
        let result;
        if (records) {
          const answer = options.answer(pending.text), id = 'record-' + crypto.randomUUID();
          histories.set(chat.chat_id, {messages: [...messageItems, {role: 'user', text: pending.text, invocation_id: id}, {role: 'assistant', text: answer.answer, invocation_id: id}], receipt: {status: 'approved_records', sources: answer.sources}});
          result = {status: 'completed'};
        } else result = await api('chats/' + chat.chat_id + '/messages', 'POST', pending);
        await loadHistory();
        if (['completed', 'failed', 'interrupted', 'cancelled'].includes(result.status)) {
          pending = null; input.value = ''; resize();
          if (result.status !== 'completed') status('Turn ' + result.status + '. This request will not be repeated automatically.', true);
        } else { retry.hidden = false; status('Your request is still running. Refresh to check it.'); }
      } catch (error) {
        status(error.message + '. Use Check / retry to recover this same request.', true); retry.hidden = false;
        messages.querySelector('.copilot-thinking')?.remove();
      } finally { busy = Boolean(chat?.active_run); renderList(); controls(); }
    }
    form.onsubmit = async event => {
      event.preventDefault(); if (busy || pending || !runtimeReady || !input.value.trim()) return;
      pending = {request_id: crypto.randomUUID(), text: input.value.trim()}; await sendMessage();
    };
    input.oninput = () => {resize(); controls();};
    input.onkeydown = event => {
      if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {event.preventDefault(); if (!send.disabled && !busy) form.requestSubmit();}
    };
    retry.onclick = () => sendMessage();
    refresh.onclick = async () => {menu.open = false; try {await loadHistory();} catch (e) {status(e.message, true);} };
    stop.onclick = async () => {
      stopRequested = true;
      try {
        if (!chat) {status('Stopping before the question is sent…'); return;}
        if (chat && !records) {
          const active = chat.active_run || (await api('chats/' + chat.chat_id + '/messages?offset=0')).chat.active_run;
          if (active) await api('runs/' + active + '/cancel', 'POST', {});
          else {status('No active turn was found. Refresh the conversation to check its status.'); return;}
        }
        status('Stop requested. The turn will stop at its next guarded step.');
      } catch (e) {status(e.message, true);}
    };

    function dialog(titleText, cls = '') {
      const node = el('dialog', undefined, 'copilot-dialog ' + cls), head = el('div', undefined, 'copilot-dialog-header');
      const heading = el('h2', titleText), close = button('Close ' + titleText.toLowerCase(), 'x');
      heading.id = 'copilot-dialog-title-' + crypto.randomUUID(); node.setAttribute('aria-labelledby', heading.id);
      close.onclick = () => node.close(); head.append(heading, close); node.append(head);
      node.addEventListener('click', event => {if (event.target === node) node.close();});
      node.addEventListener('close', () => node.remove()); shell.append(node); return node;
    }
    async function trainingApi(path,body) {
      const r=await fetch('/api/training/'+path,{method:body===undefined?'GET':'POST',credentials:'same-origin',
        headers:body===undefined?{}:{'Content-Type':'application/json','X-Defender-Request':'1'},body:body===undefined?undefined:JSON.stringify(body)});
      if(!r.headers.get('content-type')?.includes('application/json'))throw Error('Sign in again to open training jobs.');
      const data=await r.json();if(!r.ok)throw Error(data.error||'Training request unavailable');return data;
    }
    async function trainingDialog() {
      if(busy||pending)return;
      const pane=dialog('Mac training jobs'),content=el('div',undefined,'copilot-evidence-content');
      const method=el('select'),holdout=el('select'),saved=el('select');
      method.setAttribute('aria-label','Engineering preset');holdout.setAttribute('aria-label','Held-out core study');saved.setAttribute('aria-label','Your training jobs');
      const prepare=el('button','Review training plan'),approve=el('button','Approve and queue this exact job'),
        refresh=el('button','Check job status'),cancel=el('button','Cancel selected job'),ask=el('button','Ask about this job');
      const planView=el('div'),jobView=el('div'),feedback=el('p');feedback.setAttribute('role','status');
      let catalog,plan=null,requestBody=null,current=null,working=false,uncertain=false;
      const update=()=>{prepare.disabled=working||uncertain||!catalog?.training_enabled||!method.value||!holdout.value;
        approve.disabled=working||!plan;method.disabled=working||uncertain;holdout.disabled=working||uncertain;
        refresh.disabled=working||!saved.value;cancel.disabled=working||!current||!['QUEUED','RUNNING','RETRYABLE'].includes(current.status);
        ask.disabled=working||!current;saved.disabled=working;};
      async function act(fn){if(working)return;working=true;update();feedback.textContent='Working…';try{await fn();}catch(e){feedback.textContent=e.message;}finally{working=false;update();}}
      function show(result){current=result;jobView.replaceChildren();renderTrainingResult(jobView,result);}
      function clear(){plan=null;requestBody=null;planView.replaceChildren();update();}
      method.onchange=clear;holdout.onchange=clear;saved.onchange=()=>{current=null;jobView.replaceChildren();update();};
      content.append(el('p','Bounded engineering tests on the existing Mac worker. These jobs do not retrain the frozen research models or use the final test study.'),
        el('h3','Prepare a job'),method,holdout,prepare,planView,approve,el('h3','Your jobs'),saved,refresh,jobView,cancel,ask,feedback);
      pane.append(content);pane.showModal();update();
      prepare.onclick=()=>act(async()=>{
        requestBody={job_id:'web_'+catalog.month.replace('-','')+'_'+crypto.randomUUID().replaceAll('-',''),method:method.value,holdout_study:holdout.value};
        plan=await trainingApi('prepare',requestBody);const p=plan.plan;
        planView.replaceChildren(el('h3','Review before approval'),el('p',p.job.method+' · hold out '+p.job.holdout_study),
          el('p','Preset: '+JSON.stringify(p.job.parameters)+' · seed '+p.seed),el('p',p.evaluation),el('p',p.compute),el('p',p.cost),
          el('p','Maximum 720 seconds per attempt; 3 attempts. No automatic model deployment.'),
          el('p','Dataset SHA-256: '+p.job.dataset.sha256),el('p','Approval SHA-256: '+plan.approval_sha256));
        const details=el('details'),summary=el('summary','Full plan and worker code hashes'),pre=el('pre',JSON.stringify(p,null,2));details.append(summary,pre);planView.append(details);
        feedback.textContent='Review the exact plan, then approve to queue one job.';
      });
      approve.onclick=()=>act(async()=>{uncertain=true;
        const result=await trainingApi('approve',{...requestBody,approval_sha256:plan.approval_sha256});
        if(![...saved.options].some(o=>o.value===result.job_id)){const o=el('option',result.job_id);o.value=result.job_id;saved.append(o);}
        saved.value=result.job_id;show(result);uncertain=false;clear();feedback.textContent='Approval recorded. The durable queue determines execution status.';
      });
      refresh.onclick=()=>act(async()=>{show(await trainingApi('jobs/'+saved.value));feedback.textContent='Current durable job state loaded.';});
      cancel.onclick=()=>act(async()=>{show(await trainingApi('jobs/'+current.job_id+'/cancel',{}));feedback.textContent='Cancellation state checked. Local computation may finish its current checkpoint.';});
      ask.onclick=()=>{input.value='Show the current status and engineering results for job '+current.job_id+'.';pane.close();resize();controls();input.focus();};
      await act(async()=>{catalog=await trainingApi('catalog');
        for(const [select,label,values] of [[method,'Choose a preset',Object.keys(catalog.methods)],[holdout,'Choose a held-out study',Object.keys(catalog.studies)],[saved,'Choose one of your jobs',catalog.jobs]]){
          const empty=el('option',label);empty.value='';select.append(empty);for(const value of values){const o=el('option',value);o.value=value;select.append(o);}}
        feedback.textContent=catalog.training_enabled?'Select a preset and holdout. Nothing runs until approval.':'New training is paused. Existing jobs remain readable.';
      });
    }
    function renderTrainingResult(parent,r){
      const card=el('section',undefined,'copilot-prediction');card.append(el('h3','Mac engineering job'),el('code',r.job_id),
        el('p',r.status+' · '+(r.stage||'waiting')+' · attempt '+r.attempt),el('p',r.method+' · held-out '+r.holdout_study));
      if(r.observed_at)card.append(el('p','Status observed: '+new Date(r.observed_at*1000).toLocaleString()));
      if(r.execution)card.append(el('p',r.execution));
      if(r.metrics_gy)card.append(el('p','Donor-balanced MAE '+r.metrics_gy.mae.toFixed(3)+' Gy · RMSE '+r.metrics_gy.rmse.toFixed(3)+' Gy · bias '+r.metrics_gy.bias.toFixed(3)+' Gy'));
      if(r.worker_last_checked_at)card.append(el('p','Last worker check-in: '+new Date(r.worker_last_checked_at*1000).toLocaleString()));
      card.append(el('p',r.notice));parent.append(card);
    }
    function renderTraining(parent,receipt){for(const item of (receipt.receipts||[]).filter(r=>r.tool==='training_status'))renderTrainingResult(parent,item.verified_result);}
    async function predictionDialog() {
      if (busy || pending) return;
      const pane=dialog('Add expression profile'), content=el('div',undefined,'copilot-evidence-content');
      content.append(el('p','Upload one de-identified Agilent adapter JSON, or select a real development example. Values stay out of the language-model prompt. New inputs require researcher compatibility review.'));
      const fileLabel=el('label','Expression profile JSON'), file=el('input');file.type='file';file.accept='.json,application/json';file.id='copilot-profile-file';fileLabel.htmlFor=file.id;
      const examples=el('select');examples.setAttribute('aria-label','Real development example');
      const saved=el('select');saved.setAttribute('aria-label','Your saved uploads');
      const chooseExample=el('button','Load real example'), chooseSaved=el('button','Load saved upload'), upload=el('button','Upload and check');
      const model=el('select');model.setAttribute('aria-label','Frozen prediction model');
      const qcQuestion=el('button','Ask about QC'), predictQuestion=el('button','Use for prediction');
      const feedback=el('p','Loading…');feedback.setAttribute('role','status');const details=el('div',undefined,'copilot-upload-details');
      const review=el('a','Open Predictions for researcher review');review.href='#predictions';review.onclick=()=>pane.close();
      let documentInput=null, selected=null, working=false;
      const webApi=async (path,body)=>{
        const response=await fetch('/api/'+path,{credentials:'same-origin',method:body===undefined?'GET':'POST',
          headers:body===undefined?{}:{'Content-Type':'application/json','X-Defender-Request':'1'},...(body===undefined?{}:{body:JSON.stringify(body)})});
        const result=await response.json();if(!response.ok)throw Error(result.error||'Upload request failed');return result;
      };
      function update(){
        for(const control of [file,examples,saved,model])control.disabled=working;
        chooseExample.disabled=working||!examples.value;chooseSaved.disabled=working||!saved.value;
        upload.disabled=working||!documentInput;qcQuestion.disabled=working||!selected;
        predictQuestion.disabled=working||selected?.status!=='approved'||!model.value;
      }
      function show(record){
        selected=record;details.replaceChildren(el('h3',record.sample_id),el('p','Status: '+record.status.replaceAll('_',' ')),
          el('p',record.qc.usable_probes+' usable probes · '+record.qc.masked_probes+' masked probes'));
        for(const reason of record.qc.blockers||[])details.append(el('p',reason,'copilot-upload-warning'));
        if(record.curated)details.append(el('p','Real development example with training overlap. It tests the workflow, not predictive accuracy.'));
        else if(record.status!=='approved')details.append(el('p','No estimate is available until the exact input and frozen release pass researcher review.'));
        update();
      }
      async function act(fn){if(working)return;working=true;update();feedback.textContent='Working…';try{await fn();}catch(e){feedback.textContent=e.message+' Retry the same upload if its outcome is uncertain.';}finally{working=false;update();}}
      for(const b of [chooseExample,chooseSaved,upload,qcQuestion,predictQuestion])b.type='button';
      content.append(fileLabel,file,upload,el('p','Or use an audited development profile:'),examples,chooseExample,el('p','Your saved uploads:'),saved,chooseSaved,details,el('h3','Choose a frozen model'),model,qcQuestion,predictQuestion,feedback,review);
      pane.append(content);pane.showModal();update();
      file.onchange=()=>act(async()=>{documentInput=null;selected=null;details.replaceChildren();const f=file.files[0];if(!f)return;
        if(f.size>800000)throw Error('Use a JSON adapter no larger than 800 kB.');
        documentInput=JSON.parse(await f.text());feedback.textContent='File ready. Upload and check compatibility.';});
      chooseExample.onclick=()=>act(async()=>{selected=null;details.replaceChildren();documentInput=await webApi('prediction-examples/'+examples.value);feedback.textContent='Real example loaded. Upload and check compatibility.';});
      chooseSaved.onclick=()=>act(async()=>{documentInput=null;show(await webApi('uploads/'+saved.value));feedback.textContent='Saved upload checked.';});
      upload.onclick=()=>act(async()=>{show(await webApi('uploads',documentInput));feedback.textContent='Upload recorded. Review QC before choosing the next action.';});
      model.onchange=update;
      qcQuestion.onclick=()=>{input.value='Show compatibility and QC for upload '+selected.upload_id+'.';pane.close();resize();controls();input.focus();};
      predictQuestion.onclick=()=>{input.value='Predict the research dose using model '+model.value+' for upload '+selected.upload_id+'.';pane.close();resize();controls();input.focus();};
      await act(async()=>{const catalog=await webApi('prediction-chat-catalog');
        for(const row of catalog.examples){const o=el('option',row.id+' · '+row.study);o.value=row.id;examples.append(o);}
        for(const row of catalog.uploads){const o=el('option','Upload '+row.upload_id.slice(0,12));o.value=row.upload_id;saved.append(o);}
        const empty=el('option','Choose a model');empty.value='';model.append(empty);
        for(const method of catalog.methods){const o=el('option',method);o.value=method;model.append(o);}
        feedback.textContent='Choose a file, real example or saved upload. No model is selected automatically.';
      });
    }
    function renderPrediction(parent,receipt){
      for(const item of (receipt.receipts||[]).filter(r=>r.tool==='prediction')){
        const verified=item.verified_result, r=verified?.answer;if(!r)continue;
        const card=el('section',undefined,'copilot-prediction');card.append(el('h3','Frozen-model research result'));
        if(r.status==='completed'&&Number.isFinite(r.estimate_gy)){
          card.append(el('p',r.estimate_gy.toFixed(3)+' Gy','copilot-dose'),el('p','Model '+r.model+' · '+verified.sample_id));
          card.append(el('p','Model estimate, not a recorded exposure. No individual prediction interval is available.'));
        }else card.append(el('p',r.status.replaceAll('_',' ')),el('p',r.explanation||'No dose estimate.'));
        card.append(el('p','Training overlap: '+String(r.training_overlap)),el('p',r.qc.usable_probes+' usable probes · '+r.qc.masked_probes+' masked probes'));
        if(r.training_overlap===true||r.training_overlap==='yes')card.append(el('p','This profile overlaps model training. Use the result to test functionality, not accuracy.','copilot-upload-warning'));
        for(const warning of [...(r.qc.blockers||[]),...(r.limitations||[])])card.append(el('p',warning,'copilot-evidence-note'));
        card.append(el('small','Frozen release '+r.release_sha256.slice(0,16)+' · '+item.evidence_id));
        const download=el('button','Download result & provenance','copilot-message-source');download.type='button';
        download.onclick=()=>{const a=el('a'),url=URL.createObjectURL(new Blob([JSON.stringify(item,null,2)],{type:'application/json'}));a.href=url;a.download='research-prediction-'+r.upload_id.slice(0,12)+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
        card.append(download);parent.append(card);
      }
    }
    function showSources(receipt) {
      if (!receipt) return;
      const pane = dialog('Sources & evidence', 'copilot-evidence-dialog'), content = el('div', undefined, 'copilot-evidence-content');
      content.append(el('p', 'The records behind this answer.', 'copilot-dialog-description'));
      if (!records) {renderAnalysis(content, receipt);renderPrediction(content,receipt);renderTraining(content,receipt);}
      if (receipt.sources) for (const source of receipt.sources) content.append(safeLink(source.label, source.url));
      for (const evidence of receipt.receipts || []) {
        if (evidence.tool === 'training_status' || evidence.tool === 'prediction' || evidence.tool === 'analysis') continue;
        const result = evidence.verified_result || {}, card = el('div', undefined, 'copilot-evidence-card');
        card.append(icon(evidence.tool === 'expression' ? 'flask' : 'database'), el('h3', evidence.tool === 'protocol' ? 'Study methods & protocols' : evidence.tool === 'expression' ? 'Gene expression record' : 'Recorded sample data'));
        if (evidence.evidence_id) card.append(el('code', evidence.evidence_id, 'copilot-evidence-id'));
        if (result.count !== undefined) card.append(el('p', result.count + ' ' + (result.unit || 'records'), 'copilot-evidence-count'));
        const details = el('dl');
        for (const [labelText, value] of [['Release', result.release_id || chat?.release_id], ['Study', result.samples?.map(s => s.study_id).filter((s, i, all) => all.indexOf(s) === i).join(', ')], ['Scale', result.scale?.replaceAll('_', ' ')], ['Normalization', result.normalization_status?.replaceAll('_', ' ')]]) {
          if (!value) continue; details.append(el('dt', labelText), el('dd', value));
        }
        card.append(details);
        for (const passage of result.passages || []) {
          const section = el('section', undefined, 'copilot-protocol-passage');
          section.append(safeLink(passage.study_id + ' · ' + passage.section.replaceAll('_', ' '), passage.source_url),
            el('code', passage.chunk_id), el('blockquote', passage.text),
            el('small', 'GEO source version ' + passage.document_version.slice(0, 12)));
          card.append(section);
        }
        if (result.limitations) card.append(el('p', result.limitations, 'copilot-evidence-note'));
        if (result.features && result.samples && result.values) {
          const wrap = el('div', undefined, 'copilot-table'), table = el('table'), caption = el('caption', 'Exact retrieved values'); table.append(caption);
          const head = el('tr'); for (const field of ['Sample / probe', 'Value']) head.append(el('th', field)); const thead = el('thead'); thead.append(head); table.append(thead);
          const tbody = el('tbody'); let count = 0;
          for (let p = 0; p < result.features.length && count < 200; p++) for (let s = 0; s < result.samples.length && count < 200; s++) {
            const row = el('tr'), name = el('td'); name.append(el('span', result.samples[s].sample_id), el('small', result.features[p].probe_id));
            row.append(name, el('td', result.values[p]?.[s] == null ? 'Not recorded' : String(result.values[p][s]))); tbody.append(row); count++;
          }
          table.append(tbody); wrap.append(table); card.append(wrap);
        }
        if (result.comparison_note) card.append(el('p', result.comparison_note, 'copilot-evidence-note'));
        content.append(card);
      }
      if (receipt.model_calls !== undefined) content.append(el('p', receipt.model_calls + ' model calls · ' + receipt.tool_calls + ' data lookup' + (receipt.tool_calls === 1 ? '' : 's'), 'copilot-dialog-description'));
      const advanced = el('details', undefined, 'copilot-raw-receipt'); advanced.append(el('summary', 'Technical receipt'), el('pre', JSON.stringify(receipt, null, 2))); content.append(advanced);
      pane.append(content); pane.showModal();
    }
    sources.onclick = () => showSources(latestReceipt);
    scope.onclick = () => {
      const pane = dialog('About this copilot'), content = el('div', undefined, 'copilot-evidence-content');
      content.append(el('h3', records ? 'Published research evidence' : 'Verified core release'), el('p', records ? 'Answers use the existing approved study and result records. Conversations are kept only for this visit.' : '145 profiles from GSE102971, GSE8917, and GSE65292. Ask for recorded doses, sample counts, and gene expression using NCBI Gene IDs.'));
      content.append(el('h3', 'History & privacy'), el('p', records ? 'No generative model is called in this viewer.' : 'Your conversations are saved with Google-managed Sessions for 30 days and are scoped to your signed-in account. ' + (options.memoryEnabled ? 'Memory preferences are explicit, editable and expire after 30 days. Deleting a chat does not delete preferences; use Forget in Memory preferences.' : 'Long-term personal memory is not connected.')));
      content.append(el('h3', 'Current limits'), el('p', records ? 'Unknown facts remain “not recorded”.' : (options.protocolEnabled ? 'Protocol search covers approved public GEO metadata for the three core studies; it does not search full papers or project decisions. ' : 'Protocol search is not connected. ') + (options.predictionEnabled?'Owned uploaded profiles can use frozen prediction after compatibility approval. Use Add profile; the agent cannot approve inputs. ':'Prediction execution is not connected. ')+(options.trainingEnabled?'Use Training to review and approve an engineering preset; the agent can retrieve owned status/results. ':'Training commands are not connected. ')+'The workspace has a limited pilot allowance.'));
      content.append(el('p', 'Research use only. This assistant is not a clinical decision tool.', 'copilot-dialog-description')); pane.append(content); pane.showModal();
    };
    about.onclick = () => {menu.open = false; scope.click();};
    remove.onclick = () => {
      menu.open = false; if (!chat || busy) return;
      const pane = dialog('Delete this conversation?', 'copilot-confirm-dialog'), body = el('div', undefined, 'copilot-evidence-content');
      body.append(el('p', records ? 'This removes the conversation from this visit.' : 'This deletes the saved conversation and its Evidence-agent history. Scientific records and audit receipts are retained.'));
      const actions = el('div', undefined, 'copilot-confirm-actions'), keep = el('button', 'Keep conversation', 'copilot-secondary'), del = el('button', 'Delete conversation', 'copilot-delete-confirm');
      keep.type = del.type = 'button'; keep.onclick = () => pane.close();
      del.onclick = async () => {
        const id = chat.chat_id; del.disabled = true;
        try {if (!records) await api('chats/' + id, 'DELETE', {}); histories.delete(id); chats = chats.filter(c => c.chat_id !== id); pane.close(); pending = null; busy = false; draft(); status('Conversation deleted.');}
        catch (e) {del.disabled = false; status(e.message, true);}
      };
      actions.append(keep, del); body.append(actions); pane.append(body); pane.showModal(); keep.focus();
    };
    shell.addEventListener('keydown', event => {if (event.key === 'Escape') {mobileClose(); menu.open = false;}});
    document.addEventListener('click', event => {if (!menu.contains(event.target)) menu.open = false;});
    renderMessages([]); controls(); resize();
    if (!runtimeReady) status('Starting research services… Your saved chats will appear shortly.');
    try {
      const initialEpoch = navigationEpoch;
      await list();
      runtimeReady = true; controls(); status();
      const selected = new URL(location.href).searchParams.get('chat');
      const initial = chats.find(c => c.chat_id === selected) || [...chats].sort((a, b) => b.created_at - a.created_at)[0];
      if (initial && navigationEpoch === initialEpoch) await openChat(initial);
    } catch (e) {status(e.message, true);}
    return true;
  }
  window.defenderCopilot = {
    async mount(host) {if (window.defenderPublic) return false; const status = await api('status'); if (!status.enabled) return false; return workspace(host, {executionReady:status.execution_ready !== false,preview: status.preview, memoryEnabled: status.memory_enabled, protocolEnabled: status.protocol_rag_enabled,predictionEnabled:status.prediction_enabled,trainingEnabled:status.training_enabled});},
    async mountRecords(host, data) {return workspace(host, {records: true, answer: question => window.defenderAnswers.answer(question, data)});}
  };
})();
