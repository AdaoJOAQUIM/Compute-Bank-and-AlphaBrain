'use strict';

const CACHE = 'alphabrain-v10';
const ASSETS = ['/', '/index.html', '/alphabrain-core.js', '/app.js', '/manifest.json'];

// ── Install: pre-cache all assets ────────────────────────────────────────────
self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(ASSETS))
  );
  self.skipWaiting();
});

// ── Activate: evict old caches ────────────────────────────────────────────────
self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// ── Fetch: cache-first for assets, network-first for API ──────────────────────
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);

  // Only intercept same-origin GETs
  if (e.request.method !== 'GET' || url.origin !== self.location.origin) return;

  e.respondWith(
    caches.match(e.request).then(cached => {
      if (cached) return cached;
      return fetch(e.request).then(resp => {
        if (resp.ok) {
          const clone = resp.clone();
          caches.open(CACHE).then(c => c.put(e.request, clone));
        }
        return resp;
      }).catch(() => cached || new Response('offline', { status: 503 }));
    })
  );
});

// ── Background pheromone tick (Axiom K) ───────────────────────────────────────
// Runs at 1 Hz even when the main page is backgrounded; clients receive messages.
let _bgTickInterval = null;

self.addEventListener('message', e => {
  if (e.data === 'START_BG_TICK') {
    if (_bgTickInterval) return;
    _bgTickInterval = setInterval(() => {
      self.clients.matchAll().then(clients =>
        clients.forEach(c => c.postMessage({ type: 'BG_TICK' }))
      );
    }, 1000);
  }
  if (e.data === 'STOP_BG_TICK') {
    clearInterval(_bgTickInterval);
    _bgTickInterval = null;
  }
});
