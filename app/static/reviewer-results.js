'use strict';

// The comparator reads saved aggregate results. Selecting a model is a display action.
(() => {
 const get = id => document.getElementById(id);
 const host = get('reviewer-results');
 const phone = matchMedia('(max-width: 760px)');
 const combined = 'gradient_boosting_wb_wbc';
 const wholeBloodBoosting = 'gradient_boosting';
 const badges = {elastic_net:'EN', pls:'PLS', gradient_boosting:'GB', [combined]:'GB', median_null:'M'};
 const format = value => Number(value).toFixed(3);
 const signed = value => (value >= 0 ? '+' : '−') + format(Math.abs(value));
 const primary = card => card.scores.find(score => score.seed === card.primary_seed);
 let snapshot, loading = false, selected, ascending = true, opener;

 // Model cards and downloads use the complete evaluation snapshot.
 function comparisonCards() {
  return snapshot.model_cards.filter(card => card.id !== wholeBloodBoosting);
 }

 function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
 }

 host.innerHTML = `
  <div class="exp-panel-heading"><div><h2>Final evaluation</h2>
   <p>GSE90909 · 52 <span class="exp-wide">whole-blood </span>profiles · 12 donors · 0–4 Gy</p></div>
   <button type="button" class="exp-button" id="reviewer-download" disabled><img src="/static/assets/experiments/download.svg" alt="">Download results</button></div>
  <p id="reviewer-status" role="status">Loading saved results…</p><button type="button" id="reviewer-retry" class="exp-button" hidden>Retry results</button>
  <div id="reviewer-loaded" class="exp-final-grid" hidden>
   <section class="exp-card exp-model-table" aria-labelledby="exp-performance-title">
    <div class="exp-card-heading"><div><h3 id="exp-performance-title">Model performance</h3>
     <p class="exp-sub">Donor-balanced <abbr title="Mean absolute error">MAE</abbr> · lower is better</p></div></div>
    <table class="exp-table exp-models" aria-labelledby="exp-performance-title">
     <colgroup><col class="exp-name-col"><col class="exp-pool-col"><col class="exp-ci-col"><col class="exp-mae-col"></colgroup>
     <thead><tr><th scope="col">Model</th><th scope="col" class="exp-pool-col">Training data</th>
      <th scope="col" class="exp-ci-col"><abbr title="95% donor-bootstrap interval for mean absolute error">95% interval</abbr></th>
      <th scope="col" id="exp-sort-heading" aria-sort="ascending"><button type="button" id="exp-sort" aria-label="Sort mean absolute error, highest first">MAE (Gy)<img src="/static/assets/experiments/sort.svg" alt=""></button></th>
     </tr></thead><tbody id="reviewer-neural"></tbody></table>
   </section>
   <section class="exp-card exp-selected" aria-labelledby="reviewer-name">
    <div class="exp-highlight"><p class="exp-eyebrow">Selected model</p><h3 id="reviewer-name"></h3><p id="exp-variant" class="exp-variant"></p>
     <p class="exp-big-value"><span id="exp-mae"></span> <small>Gy</small></p><p class="exp-metric-label">Donor-balanced MAE</p></div>
    <div class="exp-selected-body"><p class="exp-sub">95% donor interval</p><p class="exp-ci-value" id="exp-ci"></p><div id="exp-ci-scale"></div>
     <dl class="exp-facts" id="exp-model-facts"></dl><p class="exp-note" id="exp-note"></p>
     <button type="button" class="exp-button" id="reviewer-open-card"><img src="/static/assets/experiments/books.svg" alt="">Read model card</button></div>
   </section>
  </div><p id="reviewer-selection-status" class="sr-only" role="status"></p>
  <dialog id="reviewer-card-dialog" class="exp-model-dialog" aria-labelledby="reviewer-card-title">
   <header><button type="button" class="exp-back" id="reviewer-close-card"><img src="/static/assets/experiments/back.svg" alt="">Final evaluation</button></header>
   <div class="exp-detail-body"><h2 id="reviewer-card-title">Model details</h2><div id="reviewer-card-content"></div>
    <button type="button" class="exp-button exp-download-card" id="reviewer-download-card"><img src="/static/assets/experiments/download.svg" alt="">Download model card</button>
    <details class="exp-methods"><summary>Methods and seed results</summary><div id="exp-card-methods"></div></details>
    <footer class="exp-date"><img src="/static/assets/experiments/clock.svg" alt=""><span id="reviewer-card-date"></span></footer>
   </div></dialog>`;

 function labels(card) {
  const name = window.defenderModelNames[card.id] || card.name;
  if (card.id === wholeBloodBoosting || card.id === combined) {
   return ['Gradient boosting', card.id === combined ? 'WBC + whole blood data' : 'Whole blood only'];
  }
  if (card.id === 'C') return ['Multitask neural network', 'Exposure and dose'];
  if (card.id === 'D') return ['Weighted neural regression', 'WBC and whole blood'];
  return [name.replace(/^[A-D] · /, ''), ''];
 }

 function interval(score, withAxis = false) {
  const ns = 'http://www.w3.org/2000/svg';
  const {low, high} = score.mae_ci;
  const limit = Math.max(4, Math.ceil(Math.max(...snapshot.model_cards.map(card => primary(card).mae_ci.high))));
  const display = el('div', undefined, 'exp-interval');
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', '0 0 160 16');
  svg.setAttribute('preserveAspectRatio', 'none');
  svg.setAttribute('aria-hidden', 'true');
  const x = value => 2 + 156 * value / limit;
  function path(d, type) {
   const mark = document.createElementNS(ns, 'path');
   mark.setAttribute('d', d);
   mark.setAttribute('class', type);
   mark.setAttribute('vector-effect', 'non-scaling-stroke');
   svg.append(mark);
  }
  path('M2 8H158', 'exp-axis-line');
  path('M'+x(low)+' 8H'+x(high)+' M'+x(low)+' 3V13 M'+x(high)+' 3V13', 'exp-ci-line');
  path('M'+x(score.donor_mae)+' 5.5V10.5', 'exp-ci-point');
  const text = format(low) + ' to ' + format(high) + ' Gy';
  display.title = text;
  display.append(el('span', text, 'sr-only'), svg);
  if (withAxis) {
   const axis = el('div', undefined, 'exp-axis-labels');
   axis.append(el('span', '0'), el('span', limit + ' Gy'));
   display.append(axis);
  }
  return display;
 }

 function fact(list, label, value) {
  const row = el('div');
  row.append(el('dt', label), el('dd', value));
  list.append(row);
 }

 function note(card) {
  return card.id === combined ? 'Retrospective comparison on the same 52 previously evaluated profiles.' :
   'Evaluated on 52 whole-blood profiles from 12 donors in GSE90909.';
 }

 function showSelection(card) {
  selected = card;
  const score = primary(card);
  const [name, variant] = labels(card);
  for (const button of host.querySelectorAll('button[data-model]')) {
   button.setAttribute('aria-pressed', String(button.dataset.model === card.id));
   button.closest('tr').classList.toggle('exp-chosen', button.dataset.model === card.id);
  }
  get('reviewer-name').textContent = name;
  get('exp-variant').textContent = variant;
  get('exp-variant').hidden = !variant;
  get('exp-mae').textContent = format(score.donor_mae);
  get('exp-ci').textContent = format(score.mae_ci.low) + ' – ' + format(score.mae_ci.high) + ' Gy';
  get('exp-ci-scale').replaceChildren(interval(score, true));
  const facts = get('exp-model-facts');
  facts.replaceChildren();
  fact(facts, 'RMSE', format(score.donor_rmse) + ' Gy');
  fact(facts, 'Signed bias', signed(score.donor_bias) + ' Gy');
  fact(facts, 'Training profiles', card.core_profiles + card.auxiliary_profiles);
  get('exp-note').textContent = note(card);
  get('reviewer-selection-status').textContent = 'Selected ' + card.name;
 }

 function renderTable() {
  const body = get('reviewer-neural');
  const rows = comparisonCards().sort((a,b) => (primary(a).donor_mae - primary(b).donor_mae) * (ascending ? 1 : -1));
  body.replaceChildren();
  for (const card of rows) {
   const score = primary(card), [name, variant] = labels(card);
   const tr = el('tr'), title = el('th'), button = el('button', undefined, 'exp-model-pick');
   title.scope = 'row';
   button.type = 'button';
   button.dataset.model = card.id;
   button.setAttribute('aria-label', card.name + ', MAE ' + format(score.donor_mae) + ' Gy');
   const badge = el('span', badges[card.id] || card.id, 'exp-badge');
   badge.setAttribute('aria-hidden', 'true');
   const text = el('span', undefined, 'exp-model-copy');
   text.append(el('span', name));
   if (variant) text.append(el('small', variant));
   button.append(badge, text);
   title.append(button);
   const ci = el('td', undefined, 'exp-ci-col');
   ci.append(interval(score));
   const metric = el('td', undefined, 'exp-mae-cell');
   metric.append(el('span', format(score.donor_mae)), el('span', '', 'exp-row-chevron'));
   tr.append(title, el('td', card.auxiliary_profiles ? 'WBC + whole blood' : 'Whole blood', 'exp-pool-col'), ci, metric);
   tr.addEventListener('click', () => {
    showSelection(card);
    const url = new URL(location.href);
    url.searchParams.set('model', card.id);
    url.searchParams.delete('chapter');
    history.replaceState(history.state, '', url);
    if (phone.matches) openCard(button);
   });
   body.append(tr);
  }
  get('exp-sort-heading').setAttribute('aria-sort', ascending ? 'ascending' : 'descending');
  get('exp-sort').setAttribute('aria-label', 'Sort mean absolute error, ' + (ascending ? 'highest' : 'lowest') + ' first');
 }

 function textSection(parent, title, copy) {
  parent.append(el('h4', title), el('p', copy));
 }
 function detailTable(parent, headings, rows) {
  const wrap = el('div', undefined, 'table-scroll'), table = el('table'), head = table.createTHead().insertRow();
  for (const label of headings) {const th = el('th', label); th.scope = 'col'; head.append(th);}
  const body = table.createTBody();
  for (const row of rows) {const tr = body.insertRow(); for (const value of row) tr.insertCell().textContent = value;}
  wrap.append(table); parent.append(wrap);
 }
 function renderDetail() {
  const card = selected, score = primary(card), [name, variant] = labels(card);
  const body = get('reviewer-card-content');
  const hero = el('section', undefined, 'exp-highlight');
  hero.append(el('p', variant ? name : 'Selected model', 'exp-eyebrow'), el('h3', variant || name));
  const readout = el('p', undefined, 'exp-big-value');
  readout.append(el('span', format(score.donor_mae)), el('small', 'Gy'));
  hero.append(readout, el('p', 'Donor-balanced mean absolute error', 'exp-metric-label'));
  const measures = el('section', undefined, 'exp-card exp-detail-card');
  const facts = el('dl', undefined, 'exp-facts');
  fact(facts, '95% donor interval', format(score.mae_ci.low) + ' – ' + format(score.mae_ci.high) + ' Gy');
  fact(facts, 'RMSE', format(score.donor_rmse) + ' Gy'); fact(facts, 'Signed bias', signed(score.donor_bias) + ' Gy');
  measures.append(el('h3', 'Evaluation measures'), facts);
  const training = el('section', undefined, 'exp-card exp-detail-card');
  const pool = card.core_profiles + ' whole-blood' + (card.auxiliary_profiles ? ' + ' + card.auxiliary_profiles + ' WBC' : '') + ' profiles';
  training.append(el('h3', 'Training and test cohort'), el('p', pool), el('p', note(card), 'exp-sub'));
  body.replaceChildren(hero, measures, training);
  const methods = get('exp-card-methods');
  methods.replaceChildren();
  textSection(methods, 'Purpose', card.description);
  textSection(methods, 'Features', card.selected_genes ? card.selected_genes+' genes · '+card.representation : 'Constant training median');
  if (card.selected_genes) {textSection(methods, 'Preprocessing', snapshot.preprocessing); textSection(methods, 'Dose scale', snapshot.target_scaling);}
  textSection(methods, 'Frozen settings', JSON.stringify(card.settings));
  methods.append(el('h4', 'Final test and seed sensitivity'));
  detailTable(methods, ['Seed / role','MAE [95% interval]','RMSE','Bias'], card.scores.map(s => [s.seed+' / '+s.seed_role,
   format(s.donor_mae)+' ['+format(s.mae_ci.low)+', '+format(s.mae_ci.high)+']', format(s.donor_rmse),signed(s.donor_bias)]));
  methods.append(el('p', snapshot.interval_definition));
  textSection(methods, 'Interpretation', card.interpretation);
  detailTable(methods, ['Dose slice', 'Profiles / donors', 'MAE (Gy)', 'Bias (Gy)'], card.primary_strata.filter(r => ['control','above_2gy'].includes(r.stratum)).map(r =>
   [r.stratum === 'control' ? 'Controls' : '>2 Gy (all 4 Gy)', r.profiles+' / '+r.donors, format(r.donor_mae), signed(r.donor_bias)]));
  textSection(methods, 'Selection', card.selection || snapshot.selection);
  methods.append(el('h4', 'Limitations'));
  const list = el('ul'); for (const item of snapshot.limitations) list.append(el('li', item)); methods.append(list);
  get('reviewer-card-date').textContent = document.querySelector('.main-footer time').textContent;
 }

 const dialog = get('reviewer-card-dialog');
 function displayCard() {
  renderDetail();
  if (dialog.open) return;
  dialog.querySelector('details').open = false;
  dialog.showModal();
  dialog.scrollTop = 0;
  get('reviewer-close-card').focus({preventScroll:true});
 }
 function openCard(button) {
  opener = button;
  const url = new URL(location.href);
  url.searchParams.set('model', selected.id);
  url.searchParams.set('model-view','details');
  history.pushState({defenderModelCard:true}, '', url);
  displayCard();
 }
 function dismiss() {if (dialog.open) dialog.close();}
 function closeCard() {
  if (history.state?.defenderModelCard) {history.back(); return;}
  const url = new URL(location.href);
  url.searchParams.delete('model-view');
  if (url.searchParams.get('model') === wholeBloodBoosting) url.searchParams.delete('model');
  history.replaceState(history.state, '', url);
  syncFromUrl();
 }
 dialog.addEventListener('cancel', event => {event.preventDefault(); closeCard();});
 dialog.addEventListener('close', () => {
  if (location.hash === '#experiments') (opener || host.querySelector('button[aria-pressed="true"]'))?.focus({preventScroll:true});
  opener = undefined;
 });

 function syncFromUrl() {
  const params = new URLSearchParams(location.search);
  const requested = snapshot.model_cards.find(c => c.id === params.get('model'));
  const first = comparisonCards().reduce((a,b) => primary(a).donor_mae <= primary(b).donor_mae ? a : b);
  showSelection(requested || first);
  const detail = params.get('model-view') === 'details' || (phone.matches && requested?.id === wholeBloodBoosting);
  if (detail && requested && location.hash === '#experiments' && !host.hidden) displayCard();
  else dismiss();
 }

 function validateSnapshot(value) {
  const names = Object.keys(window.defenderModelNames);
  const legacy = value.schema_version === 1 && value.release_id === 'locked-evaluation-2026-10-03-v1';
  const current = value.schema_version === 2 && value.evaluated_date === document.querySelector('.main-footer time').dateTime &&
   value.release_id === 'gse90909-evaluation-' + value.evaluated_date + '-v2';
  const expected = legacy ? names.filter(id => id !== combined) : names;
  const ids = value.model_cards?.map(c => c.id) || [];
  if ((!current && !legacy) || value.status !== 'completed_consumed' || value.live_prediction_enabled !== false ||
      ids.length !== expected.length || !expected.every(id => ids.includes(id))) throw Error('Unsupported result snapshot.');
 }

 function download(value, name) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value,null,2)], {type:'application/json'}));
  const a = el('a'); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
 }
 async function ensure() {
  if (snapshot) {syncFromUrl(); return;}
  if (loading) return;
  loading = true; get('reviewer-status').hidden = false; get('reviewer-retry').hidden = true;
  get('reviewer-status').textContent = 'Loading saved results…';
  try {
   let result;
   if (window.defenderPublic) result = await window.defenderPublic.read('/published/final-results');
   else {
    const response = await fetch('/static/research-results.json', {redirect:'error'});
    if (!response.ok) throw Error('Results unavailable.');
    result = await response.json();
   }
   validateSnapshot(result);
   for (const card of result.model_cards) card.name = window.defenderModelNames[card.id];
   snapshot = result; renderTable(); syncFromUrl();
   get('reviewer-loaded').hidden = false; get('reviewer-download').disabled = false; get('reviewer-status').hidden = true;
  } catch {
   snapshot = undefined; get('reviewer-status').textContent = 'Saved results are unavailable.'; get('reviewer-retry').hidden = false;
  } finally {loading = false;}
 }
 get('exp-sort').onclick = () => {ascending = !ascending; renderTable(); showSelection(selected);};
 get('reviewer-retry').onclick = ensure;
 get('reviewer-open-card').onclick = event => openCard(event.currentTarget);
 get('reviewer-close-card').onclick = closeCard;
 get('reviewer-download').onclick = () => download(snapshot, 'defender-final-results.json');
 get('reviewer-download-card').onclick = () => {
  const {release_id,evaluated_date,status,live_prediction_enabled,study,profiles,donors,metric_definition,interval_definition,preprocessing,training,limitations} = snapshot;
  download({release_id,evaluated_date,status,live_prediction_enabled,study,profiles,donors,metric_definition,interval_definition,preprocessing,training,
   selection:selected.selection || snapshot.selection,limitations,model:selected}, 'defender-model-'+selected.id+'.json');
 };
 window.defenderReviewer = Object.freeze({ensure,dismiss});
})();
