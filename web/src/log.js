// In-page log console: captures our own step logs, console output from LiteRT/MediaPipe,
// uncaught errors, failed resource loads and an environment report. Viewable in a drawer,
// copyable and saveable as .txt so a run on a phone can be diagnosed afterwards.

const MAX = 6000;
const LEVELS = { debug: 0, info: 1, warn: 2, error: 3 };
const lines = [];
const t0 = performance.now();
let counts = { warn: 0, error: 0 };
let filter = 1; // show info and up by default; "Verbose" shows debug
let drawer, list, badge;
const orig = { log: console.log, info: console.info, warn: console.warn, error: console.error, debug: console.debug };

function fmt(v) {
  if (v instanceof Error || (v && typeof v === 'object' && 'stack' in v && 'message' in v)) {
    let s = `${v.name || 'Error'}: ${v.message}`;
    if (v.code) s += ` [code ${v.code}]`;
    if (v.stack) s += `\n${String(v.stack).split('\n').filter((l) => !l.startsWith(`${v.name}:`)).slice(0, 12).join('\n')}`;
    if (v.cause) s += `\ncaused by: ${fmt(v.cause)}`;
    return s;
  }
  if (typeof v === 'string') return v;
  if (v instanceof Event) return `${v.type} event${v.target?.src ? ` on ${v.target.src}` : ''}`;
  try {
    const s = JSON.stringify(v, (k, x) => (typeof x === 'bigint' ? `${x}n` : x instanceof Float32Array ? `Float32Array(${x.length})` : x));
    return s && s.length > 3000 ? `${s.slice(0, 3000)}…` : s;
  } catch { return String(v); }
}

export function log(level, msg, ...data) {
  const t = (performance.now() - t0) / 1000;
  const text = [msg, ...data.map(fmt)].filter((x) => x !== '' && x !== undefined).join(' ');
  const line = { t, level, text };
  lines.push(line);
  if (lines.length > MAX) lines.splice(0, lines.length - MAX);
  if (level === 'warn' || level === 'error') { counts[level]++; updateBadge(); }
  if (level === 'error') openDrawer();
  if (list && LEVELS[level] >= filter) appendLine(line);
}
export const L = {
  debug: (m, ...d) => log('debug', m, ...d),
  info: (m, ...d) => log('info', m, ...d),
  warn: (m, ...d) => log('warn', m, ...d),
  error: (m, ...d) => log('error', m, ...d),
  /** Time an async step: logs start, duration and failure. */
  async step(name, fn) {
    const s = performance.now();
    log('info', `▶ ${name}`);
    try {
      const r = await fn();
      log('info', `✔ ${name} (${(performance.now() - s).toFixed(0)} ms)`);
      return r;
    } catch (e) {
      log('error', `✖ ${name} failed after ${(performance.now() - s).toFixed(0)} ms:`, e);
      throw e;
    }
  },
};

export function text() {
  return lines.map((l) => `${l.t.toFixed(3).padStart(9)}s ${l.level.toUpperCase().padEnd(5)} ${l.text}`).join('\n');
}

// ---------- capture ----------
for (const k of Object.keys(orig)) {
  console[k] = (...args) => {
    orig[k].apply(console, args);
    const lvl = k === 'log' ? 'debug' : k === 'info' ? 'debug' : k;
    // Runtime chatter (MediaPipe glog lines, LiteRT's optional NPU/OpenVINO probes, GL perf notes)
    // is verbose-only so it does not count as a warning.
    const first = String(args[0] ?? '');
    const noise = /^(W|I)\d{4}|npu_registry|compiler_plugin|OpenVino|GL Driver Message|Automatic fallback to software WebGL|^INFO:/.test(first);
    log(noise && lvl !== 'debug' ? 'debug' : lvl, `[console] ${args.map(fmt).join(' ')}`);
  };
}
addEventListener('error', (e) => {
  if (e.target && e.target !== window && (e.target.src || e.target.href)) log('error', `Failed to load resource: ${e.target.src || e.target.href}`);
  else log('error', `Uncaught ${e.message} at ${e.filename}:${e.lineno}:${e.colno}`, e.error || '');
}, true);
addEventListener('unhandledrejection', (e) => log('error', 'Unhandled promise rejection:', e.reason));
document.addEventListener('visibilitychange', () => {
  log(document.hidden ? 'warn' : 'info', document.hidden ? 'Tab hidden: phones pause background tabs, keep this page in front while it renders.' : 'Tab visible again.');
});
addEventListener('online', () => log('info', 'Network: online'));
addEventListener('offline', () => log('warn', 'Network: offline'));

// ---------- environment report ----------
export async function envReport(build) {
  const n = navigator;
  L.info(`Wav2Lip web ${build}`);
  L.info('Page:', location.href);
  L.info('User agent:', n.userAgent);
  if (n.userAgentData) L.info('Platform:', { platform: n.userAgentData.platform, mobile: n.userAgentData.mobile, brands: n.userAgentData.brands?.map((b) => `${b.brand} ${b.version}`).join(', ') });
  L.info('Device:', { cpuThreads: n.hardwareConcurrency, memoryGB: n.deviceMemory ?? 'n/a', screen: `${screen.width}×${screen.height} @${devicePixelRatio}x`, viewport: `${innerWidth}×${innerHeight}` });
  const c = n.connection;
  if (c) L.info('Connection:', { type: c.type, effectiveType: c.effectiveType, downlinkMbps: c.downlink, rttMs: c.rtt, saveData: c.saveData });
  L.info('Features:', {
    webgpu: !!n.gpu, webcodecsVideoEncoder: typeof VideoEncoder !== 'undefined', webcodecsAudioEncoder: typeof AudioEncoder !== 'undefined',
    offscreenCanvas: typeof OffscreenCanvas !== 'undefined', cacheStorage: typeof caches !== 'undefined', crossOriginIsolated: self.crossOriginIsolated,
    jspi: typeof WebAssembly.Suspending === 'function', wakeLock: 'wakeLock' in n, secureContext: isSecureContext,
  });
  if (n.gpu) {
    try {
      const a = await n.gpu.requestAdapter();
      if (!a) L.warn('WebGPU: navigator.gpu exists but no adapter was returned (GPU blocklisted or disabled).');
      else {
        const info = a.info || (a.requestAdapterInfo ? await a.requestAdapterInfo() : {});
        L.info('WebGPU adapter:', { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description, fallback: a.isFallbackAdapter ?? info.isFallbackAdapter });
        L.debug('WebGPU features:', [...a.features].join(', '));
        L.debug('WebGPU limits:', { maxBufferSize: a.limits.maxBufferSize, maxStorageBufferBindingSize: a.limits.maxStorageBufferBindingSize, maxComputeWorkgroupStorageSize: a.limits.maxComputeWorkgroupStorageSize });
      }
    } catch (e) { L.warn('WebGPU adapter query failed:', e); }
  }
  if (n.storage?.estimate) {
    try { const e = await n.storage.estimate(); L.info('Storage:', { usedMB: +(e.usage / 1e6).toFixed(1), quotaMB: +(e.quota / 1e6).toFixed(0) }); } catch {}
  }
}

export function memory() {
  const m = performance.memory;
  return m ? `JS heap ${(m.usedJSHeapSize / 1e6).toFixed(0)}/${(m.jsHeapSizeLimit / 1e6).toFixed(0)} MB` : '';
}

// ---------- drawer UI ----------
function appendLine(l) {
  const div = document.createElement('div');
  div.className = `ll ${l.level}`;
  div.textContent = `${l.t.toFixed(2).padStart(7)}  ${l.text}`;
  const stick = list.scrollTop + list.clientHeight >= list.scrollHeight - 30;
  list.appendChild(div);
  while (list.childElementCount > 1500) list.firstChild.remove();
  if (stick) list.scrollTop = list.scrollHeight;
}
function render() {
  list.textContent = '';
  for (const l of lines.slice(-1500)) if (LEVELS[l.level] >= filter) appendLine(l);
  list.scrollTop = list.scrollHeight;
}
function updateBadge() {
  if (!badge) return;
  badge.textContent = counts.error ? `${counts.error} error${counts.error > 1 ? 's' : ''}` : counts.warn ? `${counts.warn} warn` : '';
  badge.className = `badge ${counts.error ? 'error' : counts.warn ? 'warn' : ''}`;
}
export function openDrawer() { if (drawer) { drawer.hidden = false; render(); } }

export function mountLogUI(fileStem) {
  const btn = document.createElement('button');
  btn.id = 'log-btn'; btn.className = 'log-btn'; btn.type = 'button';
  btn.innerHTML = 'Log <span class="badge"></span>';
  badge = btn.querySelector('.badge');
  drawer = document.createElement('section');
  drawer.className = 'log-drawer'; drawer.hidden = true; drawer.setAttribute('aria-label', 'Log');
  drawer.innerHTML = `
    <div class="log-bar">
      <strong>System log</strong>
      <select id="log-level" aria-label="Log level"><option value="0">Verbose</option><option value="1" selected>Info</option><option value="2">Warnings</option><option value="3">Errors</option></select>
      <button type="button" data-a="copy">Copy</button><button type="button" data-a="save">Save .txt</button>
      <button type="button" data-a="clear">Clear</button><button type="button" data-a="close" aria-label="Close log">Close</button>
      <span class="log-note" id="log-note"></span>
    </div>
    <div class="log-list" id="log-list" role="log"></div>`;
  document.body.append(btn, drawer);
  list = drawer.querySelector('#log-list');
  const note = (m) => { drawer.querySelector('#log-note').textContent = m; setTimeout(() => (drawer.querySelector('#log-note').textContent = ''), 2500); };
  btn.addEventListener('click', () => (drawer.hidden ? openDrawer() : (drawer.hidden = true)));
  drawer.querySelector('#log-level').addEventListener('change', (e) => { filter = +e.target.value; render(); });
  drawer.querySelector('.log-bar').addEventListener('click', async (e) => {
    const a = e.target.dataset?.a;
    if (a === 'close') drawer.hidden = true;
    if (a === 'clear') { lines.length = 0; counts = { warn: 0, error: 0 }; updateBadge(); render(); }
    if (a === 'copy') {
      try { await navigator.clipboard.writeText(text()); note('Copied full log'); }
      catch { const r = document.createRange(); r.selectNodeContents(list); getSelection().removeAllRanges(); getSelection().addRange(r); note('Copy blocked: text selected, copy it manually'); }
    }
    if (a === 'save') {
      const url = URL.createObjectURL(new Blob([text()], { type: 'text/plain' }));
      const link = Object.assign(document.createElement('a'), { href: url, download: `${fileStem}-${new Date().toISOString().replace(/[:.]/g, '-')}.txt` });
      document.body.append(link); link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
      note('Saved');
    }
  });
  updateBadge();
}
