// Prosty Service Worker dla PWA "M.A.R.K.E.T."
// Cache'uje statyczną powłokę strony; dane z Supabase zawsze idą przez sieć na żywo.

const CACHE_NAME = 'market-shell-v1';
const ASSETS_DO_CACHE = [
  './',
  './index.html',
  './manifest.json',
  './icon.svg'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS_DO_CACHE))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const url = event.request.url;

  // Dane z Supabase (API + realtime) muszą zawsze iść na żywo do sieci - nigdy z cache.
  if (url.includes('supabase.co')) {
    return;
  }

  event.respondWith(
    caches.match(event.request).then((cached) => cached || fetch(event.request))
  );
});
