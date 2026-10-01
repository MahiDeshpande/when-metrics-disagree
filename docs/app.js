(() => {
  'use strict';

  // ------------------------------------------------------------------
  // Perceptual Quality Index (PQI): piecewise-linear map of each metric
  // onto 0–100, where each literature tier spans exactly 25 points.
  // ------------------------------------------------------------------
  const ANCHORS = {
    psnr: [[10, 0], [20, 25], [30, 50], [40, 75], [50, 100]],
    ssim: [[0, 0], [0.6, 25], [0.85, 50], [0.95, 75], [1, 100]],
    lpips_alex: [[0, 100], [0.1, 75], [0.3, 50], [0.6, 25], [1, 0]],
    lpips_vgg: [[0, 100], [0.1, 75], [0.3, 50], [0.6, 25], [1, 0]],
    tpips: [[0, 100], [0.05, 75], [0.15, 50], [0.3, 25], [0.6, 0]],
  };
  const TIERS = [
    { name: 'Indistinguishable', short: 'Indist.' },
    { name: 'Slight', short: 'Slight' },
    { name: 'Noticeable', short: 'Notice.' },
    { name: 'Heavy', short: 'Heavy' },
  ];
  const METRIC_KEYS = ['psnr', 'ssim', 'lpips_alex', 'lpips_vgg', 'tpips'];
  const METRIC_LABEL = { psnr: 'PSNR', ssim: 'SSIM', lpips_alex: 'LPIPS-Alex', lpips_vgg: 'LPIPS-VGG', tpips: 'TPIPS' };
  const METRIC_COLOR = { psnr: '#0ea5e9', ssim: '#f59e0b', lpips_alex: '#6366f1', lpips_vgg: '#a855f7', tpips: '#10b981' };
  const LEVEL_FMT = {
    gaussian_noise: (v) => `σ ${v}`,
    gaussian_blur: (v) => `r ${v} px`,
    jpeg: (v) => `Q ${v}`,
    translation: (v) => `${v} px`,
    brightness: (v) => `×${v}`,
    contrast: (v) => `×${v}`,
    saturation: (v) => `×${v}`,
    rotation: (v) => `${v}°`,
    crop_resize: (v) => `${v}%`,
    hue: (v) => `${v}°`,
  };
  const FAMILY_CLASS = {
    Noise: 'fam-noise', Blur: 'fam-blur', Compression: 'fam-compression', Geometric: 'fam-geo', Photometric: 'fam-photometric',
  };

  function pqi(metric, value) {
    const a = ANCHORS[metric];
    const asc = a[0][0] < a[a.length - 1][0];
    const pts = asc ? a : [...a].reverse();
    if (value <= pts[0][0]) return pts[0][1];
    if (value >= pts[pts.length - 1][0]) return pts[pts.length - 1][1];
    for (let i = 0; i < pts.length - 1; i++) {
      const [x0, y0] = pts[i];
      const [x1, y1] = pts[i + 1];
      if (value >= x0 && value <= x1) return y0 + ((value - x0) / (x1 - x0)) * (y1 - y0);
    }
    return 0;
  }
  const tierOf = (p) => (p >= 75 ? 0 : p >= 50 ? 1 : p >= 25 ? 2 : 3);
  const median = (arr) => {
    const s = [...arr].sort((x, y) => x - y);
    const m = Math.floor(s.length / 2);
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  };
  const mean = (arr) => arr.reduce((s, v) => s + v, 0) / arr.length;
  const std = (arr) => {
    const m = mean(arr);
    return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / (arr.length - 1));
  };
  const fmt = (metric, v) => (metric === 'psnr' ? v.toFixed(2) : v.toFixed(metric === 'ssim' ? 3 : 4));
  const pill = (t, label) => `<span class="tier-pill tier-${t}">${label || TIERS[t].name}</span>`;
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const isDark = () => document.documentElement.classList.contains('dark');

  let DATA = null;
  let DIST = {};
  let IMG = {};
  const state = { img: 'img_ref', dist: 'gaussian_noise', level: 0, view: 'side', chartScope: 'image' };
  const tableState = { search: '', family: 'All', scope: 'mean', mode: 'raw', sortKey: null, sortDir: 1 };
  let sweepChart = null;
  let mappingChart = null;
  let mappingMetric = 'psnr';

  const tilePath = (img, dist, i) => `assets/img/${img}/${dist}_${i}.webp`;
  const refPath = (img) => `assets/img/${img}/ref.webp`;
  const levelLabel = (distId, v) => LEVEL_FMT[distId](v);

  function scoresFor(scope, distId, i) {
    if (scope !== 'mean') {
      const r = DATA.results[distId][scope][i];
      const out = {};
      METRIC_KEYS.forEach((k) => (out[k] = { v: r[k], sd: null }));
      return out;
    }
    const imgs = Object.keys(DATA.results[distId]);
    const out = {};
    METRIC_KEYS.forEach((k) => {
      const vals = imgs.map((id) => DATA.results[distId][id][i][k]);
      out[k] = { v: mean(vals), sd: std(vals) };
    });
    return out;
  }
  function summarize(scores) {
    const p = {};
    METRIC_KEYS.forEach((k) => (p[k] = pqi(k, scores[k].v)));
    const vals = Object.values(p);
    return { pqi: p, consensus: median(vals), spread: Math.max(...vals) - Math.min(...vals) };
  }

  // ------------------------------------------------------------------
  // Theme, nav, progress
  // ------------------------------------------------------------------
  function initChrome() {
    $('#themeToggle').addEventListener('click', () => {
      const dark = document.documentElement.classList.toggle('dark');
      localStorage.setItem('theme', dark ? 'dark' : 'light');
      renderSweepChart();
      renderMappingChart();
    });
    const bar = $('#progress');
    const onScroll = () => {
      const h = document.documentElement;
      const pct = (h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight)) * 100;
      bar.style.width = `${pct}%`;
    };
    document.addEventListener('scroll', onScroll, { passive: true });
    onScroll();

    const links = $$('#navlinks a');
    const sections = links.map((a) => document.querySelector(a.getAttribute('href'))).filter(Boolean);
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (!e.isIntersecting) return;
          links.forEach((a) => a.classList.toggle('active', a.getAttribute('href') === `#${e.target.id}`));
        });
      },
      { rootMargin: '-45% 0px -50% 0px' }
    );
    sections.forEach((s) => io.observe(s));

    $$('.code-tab').forEach((btn) =>
      btn.addEventListener('click', () => {
        $$('.code-tab').forEach((b) => b.classList.toggle('active', b === btn));
        $$('.code-pane').forEach((p) => p.classList.toggle('hidden', p.dataset.pane !== btn.dataset.tab));
      })
    );
    if (window.hljs) window.hljs.highlightAll();

    const revealIO = new IntersectionObserver(
      (entries) => entries.forEach((e) => e.isIntersecting && (e.target.classList.add('in'), revealIO.unobserve(e.target))),
      { rootMargin: '0px 0px -8% 0px' }
    );
    $$('.card, .pipe-step, .takeaways li').forEach((el) => {
      el.classList.add('reveal');
      revealIO.observe(el);
    });
  }

  // ------------------------------------------------------------------
  // Explorer
  // ------------------------------------------------------------------
  function initExplorer() {
    $('#imagePicker').innerHTML = DATA.images
      .map(
        (im) => `<button class="thumb" data-img="${im.id}" title="${esc(im.label)} · ${esc(im.source)}">
          <img src="${refPath(im.id)}" alt="${esc(im.label)}" loading="lazy" /><span>${esc(im.label)}</span></button>`
      )
      .join('');
    $('#imagePicker').addEventListener('click', (e) => {
      const b = e.target.closest('.thumb');
      if (!b) return;
      state.img = b.dataset.img;
      renderExplorer();
    });

    $('#distortionTabs').innerHTML = DATA.distortions
      .map((d) => `<button class="tab-btn" data-dist="${d.id}">${esc(d.name)}</button>`)
      .join('');
    $('#distortionTabs').addEventListener('click', (e) => {
      const b = e.target.closest('.tab-btn');
      if (!b) return;
      state.dist = b.dataset.dist;
      state.level = Math.min(state.level, DIST[state.dist].levels.length - 1);
      renderExplorer();
    });

    $('#levelButtons').addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      state.level = +b.dataset.level;
      renderExplorer();
    });

    $('#viewModes').addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      state.view = b.dataset.mode;
      renderExplorer();
    });

    $('#chartScope').addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      state.chartScope = b.dataset.scope;
      $$('#chartScope button').forEach((x) => x.classList.toggle('active', x === b));
      renderSweepChart();
      renderSweepTable();
    });

    const range = $('#swipeRange');
    const setSwipe = (pct) => {
      $('#swipeTop').style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
      $('#swipeHandle').style.left = `${pct}%`;
    };
    range.addEventListener('input', () => setSwipe(+range.value));
    const box = $('#swipeBox');
    const drag = (clientX) => {
      const r = box.getBoundingClientRect();
      const pct = Math.max(0, Math.min(100, ((clientX - r.left) / r.width) * 100));
      range.value = pct;
      setSwipe(pct);
    };
    let dragging = false;
    box.addEventListener('pointerdown', (e) => { dragging = true; box.setPointerCapture(e.pointerId); drag(e.clientX); });
    box.addEventListener('pointermove', (e) => dragging && drag(e.clientX));
    box.addEventListener('pointerup', () => (dragging = false));

    document.addEventListener('keydown', (e) => {
      if (!['ArrowLeft', 'ArrowRight'].includes(e.key)) return;
      const rect = $('#explorer').getBoundingClientRect();
      if (rect.top > window.innerHeight * 0.5 || rect.bottom < window.innerHeight * 0.3) return;
      if (document.activeElement && ['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
      const n = DIST[state.dist].levels.length;
      state.level = (state.level + (e.key === 'ArrowRight' ? 1 : n - 1)) % n;
      renderExplorer();
    });
  }

  function swapImg(el, src) {
    if (el.getAttribute('src') === src) return;
    el.src = src;
    el.classList.remove('img-fade');
    void el.offsetWidth;
    el.classList.add('img-fade');
  }

  function renderExplorer() {
    const d = DIST[state.dist];
    const im = IMG[state.img];
    $$('#imagePicker .thumb').forEach((b) => b.classList.toggle('active', b.dataset.img === state.img));
    $$('#distortionTabs .tab-btn').forEach((b) => b.classList.toggle('active', b.dataset.dist === state.dist));
    $('#levelButtons').innerHTML = d.levels
      .map((v, i) => `<button data-level="${i}" class="${i === state.level ? 'active' : ''}">${levelLabel(d.id, v)}</button>`)
      .join('');
    $$('#viewModes button').forEach((b) => b.classList.toggle('active', b.dataset.mode === state.view));
    $('#viewSide').classList.toggle('hidden', state.view !== 'side');
    $('#viewSwipe').classList.toggle('hidden', state.view !== 'swipe');
    $('#viewDiff').classList.toggle('hidden', state.view !== 'diff');

    const ref = refPath(im.id);
    const dist = tilePath(im.id, d.id, state.level);
    swapImg($('#imgRef'), ref);
    swapImg($('#imgDist'), dist);
    $('#swipeRef').src = ref;
    $('#swipeDist').src = dist;
    $('#distCaption').textContent = `${d.name} · ${levelLabel(d.id, d.levels[state.level])}`;
    if (state.view === 'diff') renderDiff(ref, dist);

    const scores = scoresFor(im.id, d.id, state.level);
    const sum = summarize(scores);
    const ct = tierOf(sum.consensus);
    const badge = $('#consensusBadge');
    badge.className = `tier-pill tier-${ct}`;
    badge.textContent = `Consensus: ${TIERS[ct].name} · ${Math.round(sum.consensus)}%`;
    $('#scoreCards').innerHTML = METRIC_KEYS.map((k) => {
      const p = sum.pqi[k];
      const t = tierOf(p);
      return `<div class="score-row">
        <div class="score-name"><i style="background:${METRIC_COLOR[k]}"></i>${METRIC_LABEL[k]}</div>
        <div class="score-bar" title="PQI ${p.toFixed(0)}%"><div style="width:${Math.max(2, p)}%;background:${METRIC_COLOR[k]}"></div></div>
        <div class="score-val">${fmt(k, scores[k].v)}<small class="tier-text-${t}">${TIERS[t].name}</small></div>
      </div>`;
    }).join('');

    renderSweepChart();
    renderSweepTable();
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = src;
    });
  }
  let diffToken = 0;
  async function renderDiff(refSrc, distSrc) {
    const token = ++diffToken;
    const [a, b] = await Promise.all([loadImage(refSrc), loadImage(distSrc)]);
    if (token !== diffToken) return;
    const c = $('#diffCanvas');
    const w = (c.width = a.naturalWidth);
    const h = (c.height = a.naturalHeight);
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(a, 0, 0);
    const da = ctx.getImageData(0, 0, w, h);
    ctx.drawImage(b, 0, 0, w, h);
    const db = ctx.getImageData(0, 0, w, h);
    const out = ctx.createImageData(w, h);
    for (let i = 0; i < da.data.length; i += 4) {
      out.data[i] = Math.min(255, Math.abs(da.data[i] - db.data[i]) * 4);
      out.data[i + 1] = Math.min(255, Math.abs(da.data[i + 1] - db.data[i + 1]) * 4);
      out.data[i + 2] = Math.min(255, Math.abs(da.data[i + 2] - db.data[i + 2]) * 4);
      out.data[i + 3] = 255;
    }
    ctx.putImageData(out, 0, 0);
  }

  function chartTheme() {
    const dark = isDark();
    return {
      grid: dark ? 'rgba(63,63,70,0.6)' : 'rgba(228,228,231,0.9)',
      text: dark ? '#a1a1aa' : '#71717a',
      tooltipBg: dark ? '#27272a' : '#18181b',
    };
  }
  const tierBandPlugin = {
    id: 'tierBands',
    beforeDraw(chart) {
      const y = chart.scales.y;
      if (!y || chart.config.options.plugins.tierBands === false) return;
      const { ctx, chartArea } = chart;
      const dark = isDark();
      const bands = [
        [75, 100, dark ? 'rgba(16,185,129,0.07)' : 'rgba(16,185,129,0.07)'],
        [50, 75, dark ? 'rgba(14,165,233,0.06)' : 'rgba(14,165,233,0.06)'],
        [25, 50, dark ? 'rgba(245,158,11,0.06)' : 'rgba(245,158,11,0.07)'],
        [0, 25, dark ? 'rgba(244,63,94,0.07)' : 'rgba(244,63,94,0.06)'],
      ];
      ctx.save();
      bands.forEach(([lo, hi, color]) => {
        const top = y.getPixelForValue(hi);
        const bot = y.getPixelForValue(lo);
        ctx.fillStyle = color;
        ctx.fillRect(chartArea.left, top, chartArea.right - chartArea.left, bot - top);
      });
      ctx.restore();
    },
  };

  function pqiAxis(th) {
    return {
      min: 0, max: 100,
      grid: { color: th.grid },
      ticks: { color: th.text, stepSize: 25, callback: (v) => `${v}%` },
      title: { display: true, text: 'Perceptual Quality Index', color: th.text, font: { size: 11 } },
    };
  }

  function renderSweepChart() {
    if (!window.Chart) return;
    const d = DIST[state.dist];
    const scope = state.chartScope === 'mean' ? 'mean' : state.img;
    const rows = d.levels.map((_, i) => scoresFor(scope, d.id, i));
    const th = chartTheme();
    const datasets = METRIC_KEYS.map((k) => ({
      label: METRIC_LABEL[k],
      data: rows.map((r) => pqi(k, r[k].v)),
      raw: rows.map((r) => r[k]),
      metric: k,
      borderColor: METRIC_COLOR[k],
      backgroundColor: METRIC_COLOR[k],
      borderWidth: 2.5,
      pointRadius: rows.map((_, i) => (i === state.level ? 6 : 3.5)),
      pointHoverRadius: 7,
      tension: 0.3,
    }));
    const labels = d.levels.map((v) => levelLabel(d.id, v));
    if (sweepChart) sweepChart.destroy();
    sweepChart = new Chart($('#sweepChart'), {
      type: 'line',
      data: { labels, datasets },
      plugins: [tierBandPlugin],
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        animation: { duration: 350 },
        onClick: (_, els) => {
          if (els.length) {
            state.level = els[0].index;
            renderExplorer();
          }
        },
        scales: {
          x: { grid: { display: false }, ticks: { color: th.text }, title: { display: true, text: `${d.name} severity →`, color: th.text, font: { size: 11 } } },
          y: pqiAxis(th),
        },
        plugins: {
          legend: { labels: { color: th.text, usePointStyle: true, pointStyle: 'circle', boxWidth: 8, padding: 14 } },
          tooltip: {
            backgroundColor: th.tooltipBg,
            padding: 10,
            callbacks: {
              label: (ctx) => {
                const r = ctx.dataset.raw[ctx.dataIndex];
                const sd = r.sd != null ? ` ± ${fmt(ctx.dataset.metric, r.sd)}` : '';
                return ` ${ctx.dataset.label}: ${fmt(ctx.dataset.metric, r.v)}${sd}  →  ${ctx.parsed.y.toFixed(0)}% (${TIERS[tierOf(ctx.parsed.y)].name})`;
              },
            },
          },
        },
      },
    });
  }

  function renderSweepTable() {
    const d = DIST[state.dist];
    const scope = state.chartScope === 'mean' ? 'mean' : state.img;
    $('#tableTitle').textContent = `${d.name} · ${scope === 'mean' ? 'mean ± std, 12 images' : IMG[state.img].label}`;
    const head = `<thead><tr><th>${esc(d.param)}</th>${METRIC_KEYS.map((k) => `<th class="num">${METRIC_LABEL[k]}</th>`).join('')}</tr></thead>`;
    const body = d.levels
      .map((v, i) => {
        const s = scoresFor(scope, d.id, i);
        const cells = METRIC_KEYS.map((k) => {
          const t = tierOf(pqi(k, s[k].v));
          const sd = s[k].sd != null ? `<small>±${fmt(k, s[k].sd)}</small>` : '';
          return `<td class="num"><span class="cell tier-${t}">${fmt(k, s[k].v)}${sd}</span></td>`;
        }).join('');
        return `<tr data-level="${i}" class="cursor-pointer ${i === state.level ? 'row-active' : ''}"><td class="mono whitespace-nowrap">${levelLabel(d.id, v)}</td>${cells}</tr>`;
      })
      .join('');
    const tbl = $('#sweepTable');
    tbl.innerHTML = head + `<tbody>${body}</tbody>`;
    tbl.onclick = (e) => {
      const tr = e.target.closest('tr[data-level]');
      if (!tr) return;
      state.level = +tr.dataset.level;
      renderExplorer();
    };
  }

  // ------------------------------------------------------------------
  // Translation table
  // ------------------------------------------------------------------
  function buildRows(scope) {
    const rows = [];
    DATA.distortions.forEach((d) => {
      d.levels.forEach((v, i) => {
        const scores = scoresFor(scope, d.id, i);
        const sum = summarize(scores);
        rows.push({
          distId: d.id, idx: i, name: d.name, family: d.family, level: v, levelLabel: levelLabel(d.id, v),
          scores, ...sum,
          text: `${d.name} ${d.family} ${d.param} ${levelLabel(d.id, v)} ${v}`.toLowerCase(),
        });
      });
    });
    return rows;
  }

  function initTranslation() {
    const families = ['All', ...new Set(DATA.distortions.map((d) => d.family))];
    $('#familyChips').innerHTML = families
      .map((f) => `<button class="fchip ${f === 'All' ? 'active' : ''}" data-family="${f}">${f}</button>`)
      .join('');
    $('#familyChips').addEventListener('click', (e) => {
      const b = e.target.closest('.fchip');
      if (!b) return;
      tableState.family = b.dataset.family;
      $$('#familyChips .fchip').forEach((x) => x.classList.toggle('active', x === b));
      renderMaster();
    });
    $('#tableScope').innerHTML =
      `<option value="mean">Mean of all 12 images</option>` +
      DATA.images.map((im) => `<option value="${im.id}">${esc(im.label)}</option>`).join('');
    $('#tableScope').addEventListener('change', (e) => {
      tableState.scope = e.target.value;
      renderMaster();
    });
    $('#tableSearch').addEventListener('input', (e) => {
      tableState.search = e.target.value.trim().toLowerCase();
      renderMaster();
    });
    $('#cellMode').addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      tableState.mode = b.dataset.mode;
      $$('#cellMode button').forEach((x) => x.classList.toggle('active', x === b));
      renderMaster();
    });
    $('#downloadCsv').addEventListener('click', downloadCsv);

    $('#mappingMetric').innerHTML = METRIC_KEYS.map(
      (k) => `<button data-metric="${k}" class="${k === mappingMetric ? 'active' : ''}">${METRIC_LABEL[k]}</button>`
    ).join('');
    $('#mappingMetric').addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      mappingMetric = b.dataset.metric;
      $$('#mappingMetric button').forEach((x) => x.classList.toggle('active', x === b));
      renderMappingChart();
    });
  }

  const COLUMNS = [
    { key: 'name', label: 'Degradation' },
    { key: 'levelLabel', label: 'Level' },
    ...METRIC_KEYS.map((k) => ({ key: k, label: `${METRIC_LABEL[k]} ${k === 'psnr' || k === 'ssim' ? '↑' : '↓'}`, num: true })),
    { key: 'consensus', label: 'Consensus', num: true },
    { key: 'spread', label: 'Spread', num: true },
  ];

  function sortValue(r, key) {
    if (METRIC_KEYS.includes(key)) return r.pqi[key];
    if (key === 'levelLabel') return r.level;
    return r[key];
  }

  function filteredRows() {
    let rows = buildRows(tableState.scope);
    if (tableState.family !== 'All') rows = rows.filter((r) => r.family === tableState.family);
    if (tableState.search) {
      const toks = tableState.search.split(/\s+/);
      rows = rows.filter((r) => toks.every((t) => r.text.includes(t)));
    }
    if (tableState.sortKey) {
      const k = tableState.sortKey;
      rows.sort((a, b) => {
        const va = sortValue(a, k);
        const vb = sortValue(b, k);
        return (typeof va === 'string' ? va.localeCompare(vb) : va - vb) * tableState.sortDir;
      });
    }
    return rows;
  }

  function renderMaster() {
    const rows = filteredRows();
    const head = `<thead><tr>${COLUMNS.map((c) => {
      const sorted = tableState.sortKey === c.key;
      const arrow = sorted ? (tableState.sortDir === 1 ? '▲' : '▼') : '↕';
      return `<th class="sortable ${c.num ? 'num' : ''} ${sorted ? 'sorted' : ''}" data-key="${c.key}">${c.label}<span class="arrow">${arrow}</span></th>`;
    }).join('')}</tr></thead>`;
    const body = rows.length
      ? rows
          .map((r) => {
            const metricCells = METRIC_KEYS.map((k) => {
              const p = r.pqi[k];
              const t = tierOf(p);
              const s = r.scores[k];
              const main = tableState.mode === 'pqi' ? `${p.toFixed(0)}%` : fmt(k, s.v);
              const sub = tableState.mode === 'pqi' ? fmt(k, s.v) : s.sd != null ? `±${fmt(k, s.sd)}` : `${p.toFixed(0)}%`;
              return `<td class="num"><span class="cell tier-${t}" title="${TIERS[t].name} · PQI ${p.toFixed(1)}%">${main}<small>${sub}</small></span></td>`;
            }).join('');
            const ct = tierOf(r.consensus);
            const spreadCls = r.spread >= 50 ? 'text-rose-600 dark:text-rose-400 font-semibold' : r.spread >= 30 ? 'text-amber-600 dark:text-amber-400 font-semibold' : '';
            return `<tr>
              <td><div class="font-medium text-zinc-900 dark:text-zinc-100 whitespace-nowrap">${esc(r.name)}</div><div class="text-xs text-zinc-500">${esc(r.family)}</div></td>
              <td class="mono whitespace-nowrap">${esc(r.levelLabel)}</td>
              ${metricCells}
              <td class="num">${pill(ct, `${Math.round(r.consensus)}% · ${TIERS[ct].name}`)}</td>
              <td class="num ${spreadCls}">${r.spread.toFixed(0)}</td>
            </tr>`;
          })
          .join('')
      : `<tr><td colspan="${COLUMNS.length}" class="text-center py-10 text-zinc-500">No degradations match “${esc(tableState.search)}”.</td></tr>`;
    const tbl = $('#masterTable');
    tbl.innerHTML = head + `<tbody>${body}</tbody>`;
    tbl.querySelectorAll('th.sortable').forEach((th) =>
      th.addEventListener('click', () => {
        const k = th.dataset.key;
        if (tableState.sortKey === k) {
          if (tableState.sortDir === 1) tableState.sortDir = -1;
          else { tableState.sortKey = null; tableState.sortDir = 1; }
        } else {
          tableState.sortKey = k;
          tableState.sortDir = k === 'spread' ? -1 : 1;
        }
        renderMaster();
      })
    );
    const total = DATA.distortions.reduce((s, d) => s + d.levels.length, 0);
    const scopeLabel = tableState.scope === 'mean' ? 'mean of 12 images' : IMG[tableState.scope].label;
    $('#masterCount').textContent = `Showing ${rows.length} of ${total} rows · ${scopeLabel}`;
  }

  function downloadCsv() {
    const rows = filteredRows();
    const header = ['distortion', 'family', 'level', ...METRIC_KEYS.flatMap((k) => [k, `${k}_std`, `${k}_pqi`]), 'consensus_pqi', 'consensus_tier', 'spread'];
    const lines = [header.join(',')];
    rows.forEach((r) => {
      lines.push(
        [
          `"${r.name}"`, r.family, r.level,
          ...METRIC_KEYS.flatMap((k) => [r.scores[k].v.toFixed(4), r.scores[k].sd != null ? r.scores[k].sd.toFixed(4) : '', r.pqi[k].toFixed(1)]),
          r.consensus.toFixed(1), TIERS[tierOf(r.consensus)].name, r.spread.toFixed(1),
        ].join(',')
      );
    });
    const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `metric_translation_table_${tableState.scope}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  function renderMappingChart() {
    if (!window.Chart) return;
    const k = mappingMetric;
    const th = chartTheme();
    const a = ANCHORS[k];
    const xs = [a[0][0], a[a.length - 1][0]];
    const xmin = Math.min(...xs);
    const xmax = Math.max(...xs);
    const curve = [];
    for (let i = 0; i <= 120; i++) {
      const x = xmin + ((xmax - xmin) * i) / 120;
      curve.push({ x, y: pqi(k, x) });
    }
    const points = [];
    DATA.distortions.forEach((d) =>
      Object.values(DATA.results[d.id]).forEach((rows) => rows.forEach((r) => points.push({ x: Math.max(xmin, Math.min(xmax, r[k])), y: pqi(k, r[k]), d: d.name }))));
    if (mappingChart) mappingChart.destroy();
    mappingChart = new Chart($('#mappingChart'), {
      type: 'scatter',
      data: {
        datasets: [
          { type: 'line', label: 'Mapping', data: curve, borderColor: METRIC_COLOR[k], borderWidth: 2.5, pointRadius: 0, order: 1 },
          { label: 'Our measurements', data: points, backgroundColor: `${METRIC_COLOR[k]}55`, borderColor: 'transparent', pointRadius: 3, order: 2 },
        ],
      },
      plugins: [tierBandPlugin],
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        scales: {
          x: {
            type: 'linear', min: xmin, max: xmax,
            grid: { color: th.grid }, ticks: { color: th.text },
            title: { display: true, text: `${METRIC_LABEL[k]} raw score${k === 'psnr' ? ' (dB)' : ''}`, color: th.text, font: { size: 11 } },
          },
          y: pqiAxis(th),
        },
        plugins: {
          legend: { labels: { color: th.text, usePointStyle: true, boxWidth: 8 } },
          tooltip: {
            backgroundColor: th.tooltipBg,
            callbacks: {
              label: (ctx) => (ctx.raw.d ? ` ${ctx.raw.d}: ${fmt(k, ctx.raw.x)} → ${ctx.raw.y.toFixed(0)}%` : ` ${fmt(k, ctx.raw.x)} → ${ctx.raw.y.toFixed(0)}%`),
            },
          },
        },
      },
    });
  }

  // ------------------------------------------------------------------
  // Failure cases
  // ------------------------------------------------------------------
  const FAILURES = [
    {
      dist: 'translation', idx: 0, img: 'img_ref',
      title: 'PSNR panics over a 1-pixel shift',
      text: (s) => `Move the image one pixel to the right. Nothing visible changes, yet PSNR drops to ${fmt('psnr', s.psnr.v)} dB and SSIM to ${fmt('ssim', s.ssim.v)} — both in the <em>Noticeable</em> tier. LPIPS (${fmt('lpips_alex', s.lpips_alex.v)}) and TPIPS (${fmt('tpips', s.tpips.v)}) correctly call it indistinguishable. Pixel-aligned math cannot tell “moved” from “damaged”.`,
    },
    {
      dist: 'brightness', idx: 0, img: 'ILSVRC2012_val_00000003',
      title: 'Brightness shifts trick pixel-wise math',
      text: (s) => `Dimming to 70% leaves every object perfectly recognizable, but PSNR collapses to ${fmt('psnr', s.psnr.v)} dB — <em>Heavy</em> degradation, worse than σ = 20 noise. SSIM, which normalizes local luminance, stays at ${fmt('ssim', s.ssim.v)}, and both deep metrics stay near zero.`,
    },
    {
      dist: 'saturation', idx: 0, img: 'ILSVRC2012_val_00000008',
      title: 'SSIM barely notices color disappearing',
      text: (s) => `Fully desaturating an image is an obvious change to any viewer. SSIM still reports ${fmt('ssim', s.ssim.v)} (<em>Slight</em>), because its structure term is dominated by luminance. TPIPS jumps to ${fmt('tpips', s.tpips.v)} — the largest TPIPS value of any mild distortion — flagging a real semantic change.`,
    },
    {
      dist: 'gaussian_noise', idx: 3, img: 'ILSVRC2012_val_00001403',
      title: 'TPIPS shrugs off heavy noise',
      text: (s) => `At σ = 40, the image is visibly grainy: SSIM falls to ${fmt('ssim', s.ssim.v)} and LPIPS-Alex climbs to ${fmt('lpips_alex', s.lpips_alex.v)}, both <em>Heavy</em>. TPIPS reports only ${fmt('tpips', s.tpips.v)} (<em>Slight</em>) — the content is still a bus on a street, so the semantic judge is not worried. Invariance is a feature until it isn't.`,
    },
  ];

  function renderFailures() {
    $('#failureCards').innerHTML = FAILURES.map((f) => {
      const d = DIST[f.dist];
      const s = scoresFor('mean', f.dist, f.idx);
      const metrics = METRIC_KEYS.map((k) => {
        const t = tierOf(pqi(k, s[k].v));
        return `<div class="fail-metric tier-${t}"><b>${METRIC_LABEL[k].replace('LPIPS-', 'LP-')}</b><span>${fmt(k, s[k].v)}</span><em>±${fmt(k, s[k].sd)}</em></div>`;
      }).join('');
      return `<article class="card fail-card">
        <div>
          <div class="card-kicker">${esc(d.name)} · ${levelLabel(d.id, d.levels[f.idx])}</div>
          <h3 class="mt-1">${esc(f.title)}</h3>
        </div>
        <div class="fail-imgs">
          <figure><img src="${refPath(f.img)}" alt="Original ${esc(IMG[f.img].label)}" loading="lazy" /><figcaption>Original</figcaption></figure>
          <figure><img src="${tilePath(f.img, f.dist, f.idx)}" alt="${esc(d.name)} applied" loading="lazy" /><figcaption>${esc(d.name)} · ${levelLabel(d.id, d.levels[f.idx])}</figcaption></figure>
        </div>
        <div class="fail-metrics">${metrics}</div>
        <p>${f.text(s)}</p>
      </article>`;
    }).join('');
  }

  function initIso() {
    const all = [];
    DATA.distortions.forEach((d) =>
      Object.entries(DATA.results[d.id]).forEach(([img, rows]) => rows.forEach((r, i) => all.push({ d, img, i, r }))));
    const range = $('#isoRange');
    const render = () => {
      const target = +range.value;
      $('#isoValue').textContent = target.toFixed(1);
      const hits = all.filter((h) => Math.abs(h.r.psnr - target) <= 0.75);
      const fams = new Set(hits.map((h) => h.d.name));
      // pick a diverse set: one per distortion type, closest to target first
      const byDist = {};
      hits.sort((a, b) => Math.abs(a.r.psnr - target) - Math.abs(b.r.psnr - target)).forEach((h) => {
        if (!byDist[h.d.id]) byDist[h.d.id] = h;
      });
      const shown = Object.values(byDist).sort((a, b) => pqi('tpips', b.r.tpips) - pqi('tpips', a.r.tpips)).slice(0, 6);
      if (!hits.length) {
        $('#isoSummary').innerHTML = 'No measurements within ±0.75 dB of this value.';
        $('#isoGrid').innerHTML = '';
        return;
      }
      const lp = hits.map((h) => h.r.lpips_alex);
      const tp = hits.map((h) => h.r.tpips);
      const tierCounts = [0, 0, 0, 0];
      hits.forEach((h) => tierCounts[tierOf(pqi('lpips_alex', h.r.lpips_alex))]++);
      $('#isoSummary').innerHTML = `<strong class="text-zinc-900 dark:text-white">${hits.length}</strong> measurements from
        <strong class="text-zinc-900 dark:text-white">${fams.size}</strong> distortion types land at ${target.toFixed(1)} ± 0.75 dB.
        Among them, LPIPS-Alex ranges <span class="mono">${Math.min(...lp).toFixed(3)}–${Math.max(...lp).toFixed(3)}</span> and TPIPS
        <span class="mono">${Math.min(...tp).toFixed(3)}–${Math.max(...tp).toFixed(3)}</span>. LPIPS-Alex tiers:
        ${tierCounts.map((c, t) => (c ? `${pill(t)} ×${c}` : '')).filter(Boolean).join(' ')}`;
      $('#isoGrid').innerHTML = shown
        .map((h) => {
          const tl = tierOf(pqi('lpips_alex', h.r.lpips_alex));
          const tt = tierOf(pqi('tpips', h.r.tpips));
          return `<div class="iso-item">
            <img src="${tilePath(h.img, h.d.id, h.i)}" alt="${esc(h.d.name)}" loading="lazy" />
            <div>
              <div class="t">${esc(h.d.name)} · ${levelLabel(h.d.id, h.r.level)}</div>
              <div class="text-zinc-500 mb-1">${esc(IMG[h.img].label)} · <span class="mono">${h.r.psnr.toFixed(2)} dB</span></div>
              <div class="flex flex-wrap gap-1">${pill(tl, `LPIPS ${h.r.lpips_alex.toFixed(3)}`)}${pill(tt, `TPIPS ${h.r.tpips.toFixed(3)}`)}</div>
            </div>
          </div>`;
        })
        .join('');
    };
    range.addEventListener('input', render);
    render();
  }

  function renderDisagreements() {
    const rows = buildRows('mean').sort((a, b) => b.spread - a.spread).slice(0, 6);
    $('#disagreeList').innerHTML = rows
      .map((r) => {
        const entries = METRIC_KEYS.map((k) => [k, r.pqi[k]]).sort((a, b) => a[1] - b[1]);
        const lo = entries[0];
        const hi = entries[entries.length - 1];
        const dots = METRIC_KEYS.map(
          (k) => `<span class="dis-dot" title="${METRIC_LABEL[k]}: ${fmt(k, r.scores[k].v)} → ${r.pqi[k].toFixed(0)}%" style="left:${r.pqi[k]}%;background:${METRIC_COLOR[k]}"></span>`
        ).join('');
        return `<div class="dis-row">
          <div>
            <div class="font-semibold text-zinc-900 dark:text-white">${esc(r.name)} · <span class="mono">${esc(r.levelLabel)}</span></div>
            <div class="text-xs text-zinc-500 mt-0.5"><span style="color:${METRIC_COLOR[hi[0]]}">${METRIC_LABEL[hi[0]]}</span> says ${TIERS[tierOf(hi[1])].name.toLowerCase()},
              <span style="color:${METRIC_COLOR[lo[0]]}">${METRIC_LABEL[lo[0]]}</span> says ${TIERS[tierOf(lo[1])].name.toLowerCase()}</div>
          </div>
          <div class="dis-track">${dots}</div>
          <div class="text-right"><div class="mono text-lg font-semibold text-rose-600 dark:text-rose-400">${r.spread.toFixed(0)}</div><div class="text-[10px] uppercase tracking-wider text-zinc-500">pt spread</div></div>
        </div>`;
      })
      .join('');
    const legend = METRIC_KEYS.map((k) => `<span class="inline-flex items-center gap-1.5"><i class="inline-block h-2.5 w-2.5 rounded-full" style="background:${METRIC_COLOR[k]}"></i>${METRIC_LABEL[k]}</span>`).join('');
    $('#disagreeList').insertAdjacentHTML('beforeend', `<div class="flex flex-wrap gap-4 text-xs text-zinc-500 mt-1">${legend}<span class="ml-auto">Track: Heavy → Noticeable → Slight → Indistinguishable</span></div>`);
  }

  function renderStats() {
    const levels = DATA.distortions.reduce((s, d) => s + d.levels.length, 0);
    const set = (k, v) => { const el = document.querySelector(`[data-stat="${k}"]`); if (el) el.textContent = v.toLocaleString(); };
    set('images', DATA.images.length);
    set('distortions', DATA.distortions.length);
    set('levels', levels);
    set('measurements', DATA.images.length * levels * METRIC_KEYS.length);
  }

  // ------------------------------------------------------------------
  async function main() {
    initChrome();
    try {
      const res = await fetch('data.json');
      DATA = await res.json();
    } catch (err) {
      const msg = '<div class="card p-6 text-sm">Could not load <code>data.json</code>. Serve this folder over HTTP (e.g. <code>python -m http.server</code> inside <code>docs/</code>) or open the GitHub Pages site.</div>';
      ['#explorer .max-w-6xl', '#translation'].forEach((s) => document.querySelector(s).insertAdjacentHTML('beforeend', msg));
      return;
    }
    DATA.distortions.forEach((d) => (DIST[d.id] = d));
    DATA.images.forEach((im) => (IMG[im.id] = im));
    renderStats();
    initExplorer();
    renderExplorer();
    initTranslation();
    renderMaster();
    renderMappingChart();
    renderFailures();
    initIso();
    renderDisagreements();
  }

  main();
})();
