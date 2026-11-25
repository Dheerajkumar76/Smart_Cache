// Shared helpers for every SmartCache page: header, footer, formatting, ribbon.
const SC = (() => {
  const PAGES = [
    ['home', '/', 'Overview'],
    ['explorer', '/explorer', 'Replay explorer'],
    ['impact', '/impact', 'Impact'],
    ['live', '/metrics-ui', 'Live cache'],
    ['catalog', '/catalog', 'Catalog'],
    ['docs', '/docs', 'API'],
  ];

  const esc = value => String(value).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
  const num = (value, digits = 0) => Number(value).toLocaleString('en-IN', {minimumFractionDigits: digits, maximumFractionDigits: digits});
  const inr = value => (value < 0 ? '−₹' : '₹') + Math.abs(Math.round(value)).toLocaleString('en-IN');

  function header(active) {
    const el = document.getElementById('site-header');
    if (!el) return;
    el.className = 'site-header';
    el.innerHTML = `<div class="inner">
      <a class="brand" href="/"><span class="brand-mark" aria-hidden="true"><span></span><span></span><span></span></span>SmartCache</a>
      <nav class="site-nav" aria-label="Main">${PAGES.map(([key, href, label]) =>
        `<a href="${href}"${key === active ? ' aria-current="page"' : ''}>${label}</a>`).join('')}</nav>
    </div>`;
  }

  async function footer() {
    const el = document.getElementById('site-footer');
    if (!el) return;
    el.className = 'site-footer';
    el.innerHTML = '<div class="inner"><span>SmartCache: cost-aware caching for AI chatbot answers</span><span id="runtime-line">Checking the live cache…</span></div>';
    try {
      const info = await (await fetch('/runtime-info')).json();
      document.getElementById('runtime-line').textContent =
        `Live cache: ${info.origin_backend} origin, ${info.cache_backend} store, policy ${info.policy}, ${num(info.capacity_bytes)} bytes`;
    } catch {
      document.getElementById('runtime-line').textContent = 'Live cache status unavailable';
    }
  }

  // Draw a request ribbon: one cell per request, green for a hit, amber for a miss.
  // `results` is an array of 'HIT' / 'MISS'; cells after `upto` are left empty.
  function ribbon(el, results, {cols = 60, upto = results.length, current = -1} = {}) {
    el.style.setProperty('--cols', cols);
    el.innerHTML = results.map((r, i) => {
      const cls = i >= upto ? '' : r === 'HIT' ? 'h' : 'm';
      return `<i class="${cls}${i === current ? ' now' : ''}"></i>`;
    }).join('');
  }

  function init(active) {
    header(active);
    footer();
  }

  return {esc, num, inr, ribbon, init};
})();
