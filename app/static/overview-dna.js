'use strict';

(() => {
  const overview = document.getElementById('page-overview');
  const hero = overview.querySelector('.overview-hero');
  const header = document.querySelector('main > header');
  const sidebar = document.querySelector('body > aside');
  const sources = document.getElementById('sources');
  let headerHeight;
  let observePlayback;

  // Only in-flow chrome takes space from the cover. The retractable toolbar overlays it.
  function measureHeader() {
    if (overview.hidden) return;
    const brandHeight = getComputedStyle(sidebar).position === 'sticky' ? sidebar.getBoundingClientRect().height : 0;
    const toolbarHeight = getComputedStyle(header).position === 'fixed' ? 0 : header.getBoundingClientRect().height;
    const height = toolbarHeight + brandHeight;
    if (height === headerHeight) return;
    headerHeight = height;
    overview.style.setProperty('--overview-header-height', height + 'px');
    observePlayback?.();
  }
  const sizeObserver = new ResizeObserver(measureHeader);
  sizeObserver.observe(header);
  sizeObserver.observe(sidebar);
  window.addEventListener('hashchange', measureHeader);
  measureHeader();

  const arrival = new IntersectionObserver(([entry]) => {
    if (!entry.isIntersecting) return;
    sources.classList.add('dna-arrival');
    arrival.disconnect();
  });
  arrival.observe(sources.querySelector('.source-heading'));

  const film = document.getElementById('dna-film');
  const control = document.getElementById('dna-motion');
  if (!film.canPlayType('video/mp4')) return;

  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  const portrait = matchMedia(hero.querySelector('picture source').media);
  let manualChoice = null;
  let inView = false;

  function showAction() {
    const playing = !film.paused;
    const label = playing ? 'Pause DNA animation' : 'Play DNA animation';
    control.setAttribute('aria-label', label);
    control.title = label;
    control.classList.toggle('is-playing', playing);
  }

  function syncPlayback() {
    const wantsMotion = manualChoice ?? !motion.matches;
    if (!wantsMotion || !inView || document.hidden || film.error) {
      film.pause();
      return;
    }
    if (!film.hasAttribute('src')) film.src = portrait.matches ? film.dataset.portrait : film.dataset.wide;
    if (film.paused) film.play().catch(error => {
      if (error.name === 'NotAllowedError') manualChoice = false;
      showAction();
    });
  }

  film.addEventListener('play', showAction);
  film.addEventListener('pause', showAction);
  film.addEventListener('playing', () => film.classList.add('has-frame'));
  film.addEventListener('error', () => {
    film.classList.remove('has-frame');
    control.hidden = true;
  });
  control.addEventListener('click', () => {
    manualChoice = film.paused;
    syncPlayback();
  });
  motion.addEventListener('change', () => {
    if (motion.matches && manualChoice === true) manualChoice = null;
    syncPlayback();
  });
  portrait.addEventListener('change', () => {
    // The picture element has switched crops. Use its still until the matching film plays.
    film.classList.remove('has-frame');
    if (film.hasAttribute('src')) {
      film.removeAttribute('src');
      film.load();
    }
    control.hidden = false;
    syncPlayback();
  });
  document.addEventListener('visibilitychange', syncPlayback);
  let visibilityObserver;
  observePlayback = () => {
    visibilityObserver?.disconnect();
    visibilityObserver = new IntersectionObserver(([entry]) => {
      inView = entry.isIntersecting && entry.intersectionRatio >= .01;
      syncPlayback();
    }, {rootMargin: `-${Math.ceil(headerHeight ?? 0)}px 0px 0px`, threshold: .01});
    visibilityObserver.observe(hero);
  };
  observePlayback();
  control.hidden = false;
})();
