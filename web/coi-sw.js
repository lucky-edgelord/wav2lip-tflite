// Cross-origin isolation for static hosts that cannot set headers (GitHub Pages).
// Adds COOP/COEP to this site's own responses so the page gets SharedArrayBuffer,
// which LiteRT.js needs for multithreaded CPU inference. COEP "credentialless" keeps
// cross-origin no-cors loads (Google Fonts) working without CORP headers.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (new URL(req.url).origin !== self.location.origin) return; // leave other sites alone
  if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;
  e.respondWith((async () => {
    const res = await fetch(req);
    if (res.status === 0 || res.type === 'opaqueredirect') return res;
    const h = new Headers(res.headers);
    h.set('Cross-Origin-Opener-Policy', 'same-origin');
    h.set('Cross-Origin-Embedder-Policy', 'credentialless');
    h.set('Cross-Origin-Resource-Policy', 'same-origin');
    // The body handed on is already decompressed, so the original encoding/length no longer apply.
    h.delete('Content-Encoding');
    h.delete('Content-Length');
    return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
  })());
});
