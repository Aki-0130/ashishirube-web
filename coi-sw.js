// WASM を複数スレッドで動かすための Service Worker。
// GitHub Pages では HTTP ヘッダを設定できないので、このページ（同じオリジン）の応答に
// COOP / COEP ヘッダをブラウザ内で付け足して crossOriginIsolated の状態にする。
// 無効にしたいときは URL の末尾に ?coi=0 を付けて開く（登録が解除される）。

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (new URL(req.url).origin !== self.location.origin) return;   // CDN など外部はそのまま
  if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;
  e.respondWith((async () => {
    const res = await fetch(req);
    if (res.status === 0) return res;
    const h = new Headers(res.headers);
    h.set('Cross-Origin-Embedder-Policy', 'credentialless');
    h.set('Cross-Origin-Opener-Policy', 'same-origin');
    return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
  })());
});
