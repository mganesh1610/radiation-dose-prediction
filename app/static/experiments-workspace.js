'use strict';

(() => {
 const tabs = [...document.querySelectorAll('#experiment-tabs [role="tab"]')];
 const panels = tabs.map(tab => document.getElementById(tab.getAttribute('aria-controls')));

 function enter() {
  const params = new URLSearchParams(location.search);
  const selected = params.get('workspace') === 'training' ? 'training' :
   params.get('comparison') === 'development' ? 'development' : 'final';
  tabs.forEach((tab, index) => {
   const active = tab.dataset.experiment === selected;
   tab.setAttribute('aria-selected', String(active));
   tab.tabIndex = active ? 0 : -1;
   panels[index].hidden = !active;
  });
  if (selected === 'final') window.defenderReviewer.ensure();
  else {
   window.defenderReviewer.dismiss();
   if (selected === 'training') window.defenderTraining.ensure();
   else window.defenderDevelopment.ensureComparison();
  }
 }

 function activate(tab) {
  if (tab.getAttribute('aria-selected') === 'true') return;
  const url = new URL(location.href);
  url.searchParams.delete('workspace');
  url.searchParams.delete('comparison');
  url.searchParams.delete('model-view');
  if (tab.dataset.experiment === 'training') url.searchParams.set('workspace', 'training');
  if (tab.dataset.experiment === 'development') url.searchParams.set('comparison', 'development');
  history.pushState(null, '', url);
  enter();
 }

 tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => activate(tab));
  tab.addEventListener('keydown', event => {
   const target = {ArrowRight:(index + 1) % tabs.length, ArrowLeft:(index + tabs.length - 1) % tabs.length,
    Home:0, End:tabs.length - 1}[event.key];
   if (target === undefined) return;
   event.preventDefault();
   tabs[target].focus();
   activate(tabs[target]);
  });
 });

 window.addEventListener('popstate', () => {
  if (location.hash === '#experiments') enter();
  else window.defenderReviewer.dismiss();
 });
 window.defenderExperiments = Object.freeze({enter});
})();
