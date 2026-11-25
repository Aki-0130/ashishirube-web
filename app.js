// あししるべ スマホ単体版：カメラ → 床推定(ONNX Runtime Web) → 指示 → Web Bluetooth → ESP32
import { Guide, DEFAULTS } from './guide.js?v=5';

const APP_VERSION = 5;  // 更新のたびに index.html の app.js?v= と一緒に上げる

const qs = new URLSearchParams(location.search);
const ORT_VERSION = qs.get('ort') || '1.30.0';
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const NUS = '6e400001-b5a3-f393-e0a9-e50e24dcca9e';     // Nordic UART Service
const NUS_RX = '6e400002-b5a3-f393-e0a9-e50e24dcca9e';  // スマホ → ESP32（write）
const NUS_TX = '6e400003-b5a3-f393-e0a9-e50e24dcca9e';  // ESP32 → スマホ（notify, 往復時間計測用）
const STALL_MS = 1000;     // これ以上処理が止まったら床ロスト扱いの指示を出す
const WARMUP = 5;
const SKIP_FOR_STATS = 10; // 集計から除く最初のフレーム数

const $ = (id) => document.getElementById(id);
const video = $('video'), canvas = $('canvas'), octx = canvas.getContext('2d');
const inCanvas = document.createElement('canvas');
const ictx = inCanvas.getContext('2d', { willReadFrequently: true });

function log(...a) {
  const s = a.join(' ');
  console.log(s);
  $('log').textContent = `${new Date().toLocaleTimeString()} ${s}\n` + $('log').textContent.slice(0, 5000);
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------------- 設定（ブラウザに保存） ----------------
const store = {
  get(k, d) { try { const v = localStorage.getItem('ashi_' + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('ashi_' + k, JSON.stringify(v)); } catch { /* 保存できなくても動作は続ける */ } },
};
const S = {
  model: store.get('model', ''), ep: store.get('ep', 'webgpu'), cam: store.get('cam', ''),
  rot: store.get('rot', 'auto'), thresh: store.get('thresh', DEFAULTS.thresh),
  lostCmd: store.get('lostCmd', DEFAULTS.lostCmd), holdTime: store.get('holdTime', DEFAULTS.holdTime),
  smooth: store.get('smooth', DEFAULTS.smooth), minFloor: store.get('minFloor', DEFAULTS.minFloor),
  lostFrames: store.get('lostFrames', DEFAULTS.lostFrames), draw: store.get('draw', '1'),
  opt: store.get('opt', 'all'), layout: store.get('layout', 'NHWC'),
  threads: store.get('threads', '4'),  // WebGPU の最適化レベルとデータの並び
};
const gpuOpts = (opt, layout) => ({ executionProviders: [{ name: 'webgpu', preferredLayout: layout }], graphOptimizationLevel: opt });
function guideParams() {
  return { thresh: S.thresh, lostCmd: S.lostCmd, holdTime: +S.holdTime, smooth: +S.smooth,
           minFloor: +S.minFloor, lostFrames: Math.max(1, Math.round(+S.lostFrames)) };
}

// ---------------- 状態 ----------------
let models = [], modelInfo = null, sess = null, epUsed = '', inName = 'image', outName = 'floor';
let stream = null, camLabel = '';
let guide = new Guide(guideParams());
let busy = false, recording = false, useRvfc = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;
let lastCbAt = 0, lastDoneAt = 0, lastMediaTime = -1, loopStarted = false;
let rows = [], frameIdx = 0, tStart = 0, collisions = 0, startMeta = null;
let lat = [], age = [], inf = [], rtts = [];
let wakeLock = null, lastStallTx = 0;
let ble = { dev: null, rx: null, tx: null, busy: false, pending: null, seq: 0, sent: new Map(), want: false, connected: false };
let lastResult = null, fpsT = [], skipped = 0, preparing = false;
let visData = null, inputBuf = null;

const COLS = ['frame', 'elapsed_s', 'latency_ms', 'infer_ms', 'frame_age_ms', 'floor_ratio', 'offset', 'slope',
  'angle_deg', 'command', 'thresh', 'lost', 'collisions', 'ble', 'sent_seq', 'ble_rtt_ms'];
const C_SEQ = COLS.indexOf('sent_seq'), C_RTT = COLS.indexOf('ble_rtt_ms');

// ---------------- ONNX Runtime ----------------
function loadScript(src) {
  return new Promise((res, rej) => {
    const s = document.createElement('script');
    s.src = src; s.onload = res; s.onerror = () => rej(new Error('読み込み失敗 ' + src));
    document.head.appendChild(s);
  });
}
let ortReady = false;
async function loadOrt() {
  if (ortReady) return;
  for (const f of window.ort ? [] : ['ort.webgpu.min.js', 'ort.all.min.js', 'ort.min.js']) {
    try { await loadScript(ORT_BASE + f); } catch (e) { log(e.message); continue; }
    if (window.ort) { log(`ONNX Runtime Web ${ORT_VERSION} (${f})`); break; }
  }
  if (!window.ort) throw new Error('ONNX Runtime Web を読み込めません（通信を確認してください）');
  window.ort.env.wasm.wasmPaths = ORT_BASE;
  wasmThreads = window.crossOriginIsolated ? Math.max(1, +S.threads) : 1;
  window.ort.env.wasm.numThreads = wasmThreads;
  log(window.crossOriginIsolated
    ? `WASM: ${wasmThreads}スレッド（並列化できる状態）`
    : 'WASM: 1スレッド（このページは並列化できない状態。再読み込みすると直ることがあります）');
  ortReady = true;
}
let wasmThreads = 1;

async function loadModelList() {
  try {
    const r = await fetch('models/models.json', { cache: 'no-cache' });
    models = await r.json();
  } catch {
    models = [];
    log('models/models.json が見つかりません。PCで tools/export_onnx.py を実行して web/models/ に置いてください');
  }
  const sel = $('selModel');
  sel.innerHTML = '';
  for (const m of models) sel.add(new Option(`${m.label} (${m.mb}MB)`, m.file));
  if (!models.find((m) => m.file === S.model) && models.length) S.model = models[0].file;
  sel.value = S.model;
}

async function prepareSession() {
  modelInfo = models.find((m) => m.file === S.model);
  if (!modelInfo) throw new Error('モデルがありません');
  await loadOrt();
  preparing = true;
  try {
    while (busy) await sleep(5);           // 処理中のフレームが終わるのを待つ
    try { await sess?.release(); } catch { /* 無視 */ }
    await createSession();
  } finally { preparing = false; }
}
async function createSession() {
  setPill('pEp', '推論 準備中…', 'warn');
  const url = 'models/' + modelInfo.file;
  sess = null;
  if (S.ep === 'webgpu') {
    if (!navigator.gpu) log('この端末/ブラウザでは WebGPU が使えません → WASM で動かします');
    try {
      if (navigator.gpu) { sess = await window.ort.InferenceSession.create(url, gpuOpts(S.opt, S.layout)); epUsed = 'webgpu'; }
    } catch (e) { log('WebGPU での準備に失敗 → WASM に切替:', e.message); }
  }
  if (!sess) { sess = await window.ort.InferenceSession.create(url, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' }); epUsed = 'wasm'; }
  inName = sess.inputNames[0]; outName = sess.outputNames.includes('floor') ? 'floor' : sess.outputNames[0];
  const { height: H, width: W } = modelInfo;
  inCanvas.width = canvas.width = W; inCanvas.height = canvas.height = H;
  visData = octx.createImageData(W, H);
  inputBuf = new Float32Array(3 * W * H);
  const t = [];
  for (let i = 0; i < WARMUP; i++) {
    const t0 = performance.now();
    await sess.run({ [inName]: new window.ort.Tensor('float32', inputBuf, [1, 3, H, W]) });
    t.push(performance.now() - t0);
  }
  log(`モデル準備完了 ${modelInfo.label} / ${epUsed} / ウォームアップ最終 ${t[t.length - 1].toFixed(0)}ms`);
  setPill('pEp', `推論 ${epUsed}${epUsed === 'webgpu' ? `(${S.opt}/${S.layout})` : `×${wasmThreads}`} ${modelInfo.label}`, epUsed === 'webgpu' ? 'ok' : 'warn');
  guide = new Guide(guideParams());
}

// ---------------- カメラ ----------------
async function openCamera() {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  const v = { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } };
  if (S.cam) v.deviceId = { exact: S.cam }; else v.facingMode = { ideal: 'environment' };
  stream = await navigator.mediaDevices.getUserMedia({ video: v, audio: false });
  video.srcObject = stream;
  await video.play();
  const tr = stream.getVideoTracks()[0], st = tr.getSettings();
  camLabel = `${tr.label} ${st.width}x${st.height}@${st.frameRate ?? '?'}`;
  log('カメラ:', camLabel);
  const sel = $('selCam');
  if (sel.options.length <= 1) {
    for (const d of await navigator.mediaDevices.enumerateDevices()) {
      if (d.kind === 'videoinput') sel.add(new Option(d.label || d.deviceId.slice(0, 8), d.deviceId));
    }
    sel.value = S.cam;
  }
  if (!loopStarted) {
    loopStarted = true; lastDoneAt = performance.now(); scheduleNext();
    // requestVideoFrameCallback が来ない端末向けの保険
    setTimeout(() => {
      if (useRvfc && lastCbAt === 0) { log('requestVideoFrameCallback が動かないので rAF 方式に切替'); useRvfc = false; scheduleNext(); }
    }, 3000);
  }
}

function scheduleNext() {
  if (useRvfc) video.requestVideoFrameCallback(onFrame);
  else requestAnimationFrame(pollFrame);
}
function pollFrame() {
  if (video.currentTime !== lastMediaTime) { lastMediaTime = video.currentTime; onFrame(performance.now(), null, true); }
  else requestAnimationFrame(pollFrame);
}

function onFrame(now, md) {
  lastCbAt = performance.now();
  scheduleNext();
  if (busy || preparing || !sess || !video.videoWidth) { skipped++; return; }
  busy = true;
  processFrame(md).catch((e) => log('処理エラー:', e.message)).finally(() => { busy = false; });
}

function drawInput() {
  const W = inCanvas.width, H = inCanvas.height, vw = video.videoWidth, vh = video.videoHeight;
  let rot = S.rot === 'auto' ? (vw > vh ? '90' : '0') : S.rot;
  ictx.setTransform(1, 0, 0, 1, 0, 0);
  if (rot === '90') { ictx.translate(W, 0); ictx.rotate(Math.PI / 2); ictx.drawImage(video, 0, 0, H, W); }
  else if (rot === '270') { ictx.translate(0, H); ictx.rotate(-Math.PI / 2); ictx.drawImage(video, 0, 0, H, W); }
  else if (rot === '180') { ictx.translate(W, H); ictx.rotate(Math.PI); ictx.drawImage(video, 0, 0, W, H); }
  else ictx.drawImage(video, 0, 0, W, H);
  return rot;
}

async function processFrame(md) {
  const t0 = performance.now();
  const W = inCanvas.width, H = inCanvas.height, n = W * H;
  drawInput();
  const img = ictx.getImageData(0, 0, W, H).data;
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    inputBuf[i] = img[p] / 255; inputBuf[n + i] = img[p + 1] / 255; inputBuf[2 * n + i] = img[p + 2] / 255;
  }
  const t1 = performance.now();
  const out = await sess.run({ [inName]: new window.ort.Tensor('float32', inputBuf, [1, 3, H, W]) });
  const prob = out[outName].data;
  const t2 = performance.now();
  const r = guide.step(prob, W, H, t2 / 1000);
  const tDone = performance.now();
  lastDoneAt = tDone;
  lastResult = r;
  fpsT.push(tDone); while (fpsT.length && tDone - fpsT[0] > 2000) fpsT.shift();

  let ageMs = null;
  if (md && md.captureTime > 0) { const a = tDone - md.captureTime; if (a >= 0 && a < 5000) ageMs = a; }

  if (recording) {
    const row = [frameIdx, ((tDone - tStart) / 1000).toFixed(3), (tDone - t0).toFixed(1), (t2 - t1).toFixed(1),
      ageMs === null ? '' : ageMs.toFixed(1), r.floorRatio.toFixed(4),
      r.offset === null ? '' : r.offset.toFixed(4), r.slope === null ? '' : r.slope.toFixed(4),
      r.angleDeg === null ? '' : r.angleDeg.toFixed(2), r.cmd, S.thresh.toFixed(2), r.lost ? 1 : 0, collisions,
      ble.connected ? 1 : 0, '', ''];
    rows.push(row);
    if (r.send) sendCmd(r.cmd, rows.length - 1);
    if (frameIdx >= SKIP_FOR_STATS) {
      lat.push(tDone - t0); inf.push(t2 - t1); if (ageMs !== null) age.push(ageMs);
    }
    frameIdx++;
    if (frameIdx % 150 === 0) backup();
  }
  if (S.draw !== '0') drawOverlay(img, r, prob);
  showStats(r, tDone - t0, t2 - t1, ageMs, probStats(prob));
}

// ---------------- 表示 ----------------
function drawOverlay(img, r, prob) {
  const W = canvas.width, H = canvas.height, m = r.mask, vis = visData.data, probMode = S.draw === 'p';
  for (let i = 0, p = 0; i < W * H; i++, p += 4) {
    let R = img[p], G = img[p + 1], B = img[p + 2];
    if (probMode) {                       // 確率マップ：赤いほど床らしい、青いほど床らしくない
      const q = Math.min(1, Math.max(0, prob[i]));
      R = (R * 0.35 + 165 * q) | 0; G = (G * 0.35) | 0; B = (B * 0.35 + 165 * (1 - q)) | 0;
    } else if (m[i]) {
      const x = i % W;
      const edge = x === 0 || x === W - 1 || i < W || i >= W * (H - 1) || !m[i - 1] || !m[i + 1] || !m[i - W] || !m[i + W];
      if (edge) { R = 255; G = 255; B = 0; } else { R = (R + 60) >> 1; G = (G + 200) >> 1; B = (B + 60) >> 1; }
    }
    vis[p] = R; vis[p + 1] = G; vis[p + 2] = B; vis[p + 3] = 255;
  }
  octx.putImageData(visData, 0, 0);
  const ty = H * 0.6, th = (W / 2) * S.thresh;
  octx.lineWidth = 2; octx.strokeStyle = '#0ff';
  octx.beginPath(); octx.moveTo(W / 2 - th, ty - 15); octx.lineTo(W / 2 - th, ty + 15);
  octx.moveTo(W / 2 + th, ty - 15); octx.lineTo(W / 2 + th, ty + 15); octx.stroke();
  octx.fillStyle = '#f0f';
  for (const [x, y] of r.pts) { octx.beginPath(); octx.arc(x, y, 3, 0, 2 * Math.PI); octx.fill(); }
  if (r.line) {
    const [a, b] = r.line, y0 = H - 1, y1 = Math.floor(H * 0.35);
    octx.strokeStyle = '#f00'; octx.lineWidth = 4;
    octx.beginPath(); octx.moveTo(a * y0 + b, y0); octx.lineTo(a * y1 + b, y1); octx.stroke();
    octx.fillStyle = '#ff0'; octx.beginPath(); octx.arc(a * H * 0.6 + b, ty, 6, 0, 2 * Math.PI); octx.fill();
  }
  octx.strokeStyle = '#fff'; octx.lineWidth = 2;
  octx.beginPath(); octx.moveTo(W / 2, H - 1); octx.lineTo(W / 2, H - 60); octx.stroke();
  if (r.cmd === 'S') { octx.strokeStyle = '#f00'; octx.lineWidth = 8; octx.strokeRect(0, 0, W, H); }
}

const CMD_VIEW = { L: ['←', '#4aa3ff'], R: ['→', '#ffb347'], C: ['↑', '#3cb371'], S: ['■', '#e04848'] };
function probStats(p) {
  let mx = -Infinity, mn = Infinity, sum = 0, over = 0;
  for (let i = 0; i < p.length; i++) { const v = p[i]; if (v > mx) mx = v; if (v < mn) mn = v; sum += v; if (v > 0.5) over++; }
  return { min: mn, max: mx, mean: sum / p.length, over: over / p.length, nan: Number.isNaN(sum) };
}
function showStats(r, latency, infer, ageMs, ps) {
  const [sym, col] = CMD_VIEW[r.cmd] || ['?', '#fff'];
  $('cmdBig').textContent = sym; $('cmdBig').style.color = col;
  const fps = fpsT.length > 1 ? (fpsT.length - 1) / ((fpsT[fpsT.length - 1] - fpsT[0]) / 1000) : 0;
  const rtt = rtts.length ? median(rtts.slice(-20)).toFixed(0) + 'ms' : '―';
  $('stats').textContent =
    `fps ${fps.toFixed(1)}   遅延 ${latency.toFixed(0)}ms (推論 ${infer.toFixed(0)})   映像→指示 ${ageMs === null ? '―' : ageMs.toFixed(0) + 'ms'}\n` +
    `offset ${r.offset === null ? 'None' : (r.offset >= 0 ? '+' : '') + r.offset.toFixed(2)}  angle ${r.angleDeg === null ? '―' : r.angleDeg.toFixed(1)}°  床 ${(r.floorRatio * 100).toFixed(0)}%\n` +
    `床確率 max ${ps.nan ? 'NaN' : ps.max.toFixed(2)} 平均 ${ps.mean.toFixed(2)} >0.5 ${(ps.over * 100).toFixed(0)}%\n` +
    `指示 ${r.cmd}${recording ? '' : '（プレビュー・未送信）'}   BLE往復 ${rtt}   衝突 ${collisions}`;
}
function setPill(id, text, cls) { const el = $(id); el.textContent = text; el.className = 'pill ' + (cls || ''); }

// ---------------- Bluetooth ----------------
async function bleConnect() {
  if (!navigator.bluetooth) { alert('Web Bluetooth が使えません。Android の Chrome で開いてください。'); return; }
  try {
    ble.dev = await navigator.bluetooth.requestDevice({ filters: [{ services: [NUS] }, { namePrefix: 'ASHI' }], optionalServices: [NUS] });
    ble.dev.addEventListener('gattserverdisconnected', onBleDisconnected);
    ble.want = true;
    await bleOpen();
  } catch (e) { log('BLE接続失敗:', e.message); }
}
async function bleOpen() {
  setPill('pBle', 'BLE 接続中…', 'warn');
  const server = await ble.dev.gatt.connect();
  const svc = await server.getPrimaryService(NUS);
  ble.rx = await svc.getCharacteristic(NUS_RX);
  ble.tx = await svc.getCharacteristic(NUS_TX);
  ble.tx.addEventListener('characteristicvaluechanged', onEcho);
  await ble.tx.startNotifications();
  ble.connected = true; ble.busy = false;
  setPill('pBle', `BLE ${ble.dev.name || ''} 接続`, 'ok');
  log('BLE接続:', ble.dev.name);
}
async function onBleDisconnected() {
  ble.connected = false; ble.rx = null;
  setPill('pBle', 'BLE 切断', 'ng');
  log('BLEが切断されました（ESP32側は1秒後に S に切り替わります）');
  for (let i = 0; ble.want && !ble.connected && i < 30; i++) {
    await sleep(1000);
    try { await bleOpen(); } catch (e) { log('再接続待ち…', e.message); }
  }
}
function onEcho(e) {
  const s = e.target.value.getUint8(0), rec = ble.sent.get(s);
  if (!rec) return;
  ble.sent.delete(s);
  const rtt = performance.now() - rec.t;
  rtts.push(rtt);
  if (rec.row >= 0 && rows[rec.row]) rows[rec.row][C_RTT] = rtt.toFixed(1);
}
// 最新の指示だけを送る（書き込み中に来た古い指示は捨てる）
function sendCmd(c, row = -1) {
  ble.seq = (ble.seq + 1) & 0xff;
  ble.pending = { c, s: ble.seq, row };
  pump();
}
async function pump() {
  if (ble.busy || !ble.pending || !ble.rx) return;
  const { c, s, row } = ble.pending;
  ble.pending = null; ble.busy = true;
  try {
    ble.sent.set(s, { t: performance.now(), row });
    await ble.rx.writeValueWithoutResponse(new Uint8Array([c.charCodeAt(0), s]));
    if (row >= 0 && rows[row]) rows[row][C_SEQ] = s;
  } catch (e) { log('送信エラー:', e.message); }
  finally { ble.busy = false; if (ble.pending) pump(); }
}
async function sendNow(c) {
  for (let i = 0; i < 50 && ble.busy; i++) await sleep(10);
  sendCmd(c);
}

// ---------------- 実験の開始・停止 ----------------
async function start() {
  if (!sess) { alert('モデルの準備ができていません'); return; }
  if (!ble.connected && !confirm('BLE未接続です。振動なし（記録のみ）で開始しますか？')) return;
  guide = new Guide(guideParams());
  rows = []; frameIdx = 0; collisions = 0; lat = []; age = []; inf = []; rtts = [];
  tStart = performance.now(); lastDoneAt = tStart;
  startMeta = {
    started: new Date().toISOString(), model: modelInfo.label, ep: epUsed, ort: ORT_VERSION, camera: camLabel,
    wasmThreads, webgpuOpt: `${S.opt}/${S.layout}`, app: APP_VERSION,
    rotate: S.rot, params: guideParams(), ua: navigator.userAgent, tag: $('inTag').value,
  };
  recording = true;
  sendCmd('C');
  try { wakeLock = await navigator.wakeLock?.request('screen'); } catch (e) { log('画面ONの維持に失敗:', e.message); }
  setPill('pRun', '記録中', 'ok');
  $('btnStart').disabled = true; $('btnStop').disabled = false; $('summary').style.display = 'none';
  log('開始');
}
async function stop() {
  recording = false;
  guide.reset();
  await sendNow('Q');   // 振動停止＋ESP32のウォッチドッグ解除
  try { await wakeLock?.release(); } catch { /* 無視 */ }
  wakeLock = null;
  setPill('pRun', '停止中');
  $('btnStart').disabled = false; $('btnStop').disabled = true;
  const sm = summaryText();
  $('summary').textContent = sm; $('summary').style.display = 'block';
  log('停止');
  save();
}

function median(a) { if (!a.length) return NaN; const s = [...a].sort((x, y) => x - y); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }
function pct(a, q) { if (!a.length) return NaN; const s = [...a].sort((x, y) => x - y); const i = (s.length - 1) * q, lo = Math.floor(i); return s[lo] + (s[Math.min(lo + 1, s.length - 1)] - s[lo]) * (i - lo); }
function mean(a) { return a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN; }
const f1 = (x) => (Number.isFinite(x) ? x.toFixed(1) : '―');
function summaryText() {
  const el = rows.length ? +rows[rows.length - 1][1] : 0;
  const ageMed = median(age), rttMed = median(rtts);
  const lines = [
    '==== 実験サマリ ====',
    `フレーム数 ${rows.length} / 経過 ${el.toFixed(1)} s / 平均FPS ${(rows.length / Math.max(el, 1e-6)).toFixed(1)}`,
    `処理遅延 latency_ms: 平均 ${f1(mean(lat))} / 中央値 ${f1(median(lat))} / p95 ${f1(pct(lat, 0.95))}`,
    `  うち推論 infer_ms: 中央値 ${f1(median(inf))}`,
    `撮影→指示 frame_age_ms: 平均 ${f1(mean(age))} / 中央値 ${f1(ageMed)} / p95 ${f1(pct(age, 0.95))}` + (age.length ? '' : '（この端末では取得不可）'),
    `BLE往復 rtt_ms: 中央値 ${f1(rttMed)} / p95 ${f1(pct(rtts, 0.95))} / 件数 ${rtts.length}`,
    `推定 撮影→ESP32受信: ${Number.isFinite(ageMed) && Number.isFinite(rttMed) ? (ageMed + rttMed / 2).toFixed(1) + ' ms（frame_age中央値 + RTT中央値/2）' : '―'}`,
    `衝突回数 ${collisions}`,
    `条件: ${startMeta ? `${startMeta.model} / ${startMeta.ep} / ORT ${startMeta.ort} / rot ${startMeta.rotate} / ${JSON.stringify(startMeta.params)}` : ''}`,
    `カメラ: ${startMeta?.camera ?? ''}`,
  ];
  return lines.join('\n');
}
function stamp() {
  const d = new Date(), p = (x) => String(x).padStart(2, '0');
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}
function csvText() { return COLS.join(',') + '\n' + rows.map((r) => r.join(',')).join('\n') + '\n'; }
function download(name, text, type) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(text instanceof Blob ? text : new Blob([text], { type }));
  a.download = name; document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}
function save() {
  if (!rows.length) { log('保存する記録がありません'); return; }
  const tag = ($('inTag').value || '').replace(/[^\w-]/g, '');
  const base = `ashi_${tag ? tag + '_' : ''}${stamp()}`;
  download(base + '.csv', csvText(), 'text/csv');
  download(base + '_summary.txt', summaryText() + '\n\n' + JSON.stringify(startMeta, null, 2) + '\n', 'text/plain');
  log('保存:', base);
}
function backup() { store.set('backup', { name: `ashi_backup_${stamp()}.csv`, csv: csvText() }); }

// ---------------- 診断：同じ入力を WebGPU と WASM(CPU) で計算して比べる ----------------
let wasmSess = null, wasmFile = '';
async function diagnose() {
  if (!sess || !video.videoWidth) { alert('カメラとモデルの準備ができてから押してください'); return; }
  if (recording) { alert('停止してから押してください'); return; }
  preparing = true;
  const out = [];
  try {
    while (busy) await sleep(5);
    $('summary').style.display = 'block'; $('summary').textContent = '診断中…（WASMでの計算に数秒かかります）';
    const W = inCanvas.width, H = inCanvas.height, n = W * H;
    const rot = drawInput();
    const img = ictx.getImageData(0, 0, W, H).data;
    const x = new Float32Array(3 * n);
    for (let i = 0, p = 0; i < n; i++, p += 4) { x[i] = img[p] / 255; x[n + i] = img[p + 1] / 255; x[2 * n + i] = img[p + 2] / 255; }
    const ch = [0, 1, 2].map((c) => { let s = 0; for (let i = 0; i < n; i++) s += x[c * n + i]; return s / n; });
    const fmt = (st) => `max ${st.max.toFixed(3)} / 平均 ${st.mean.toFixed(3)} / min ${st.min.toFixed(3)} / >0.5 ${(st.over * 100).toFixed(1)}%${st.nan ? ' / NaNあり' : ''}`;
    out.push('==== 診断 ====', `モデル ${modelInfo.label} / 実行先 ${epUsed} / ORT ${ORT_VERSION} / 回転 ${rot} / app v${APP_VERSION}`,
      `WASM ${wasmThreads}スレッド / crossOriginIsolated ${window.crossOriginIsolated}`,
      `カメラ ${video.videoWidth}x${video.videoHeight} → 入力 ${W}x${H}`,
      `入力の平均 R ${ch[0].toFixed(3)} G ${ch[1].toFixed(3)} B ${ch[2].toFixed(3)}（0〜1。ほぼ0なら映像が真っ黒）`);
    let t0 = performance.now();
    const resA = await sess.run({ [inName]: new window.ort.Tensor('float32', new Float32Array(x), [1, 3, H, W]) });
    const a = Float32Array.from(resA[outName].data);
    const sa = probStats(a);
    out.push(`${epUsed}: ${fmt(sa)}  (${(performance.now() - t0).toFixed(0)}ms)`);
    let sb = null, diff = 0, agree = 1;
    if (epUsed !== 'wasm') {
      if (!wasmSess || wasmFile !== modelInfo.file) {
        wasmSess = await window.ort.InferenceSession.create('models/' + modelInfo.file, { executionProviders: ['wasm'] });
        wasmFile = modelInfo.file;
      }
      t0 = performance.now();
      const resB = await wasmSess.run({ [inName]: new window.ort.Tensor('float32', new Float32Array(x), [1, 3, H, W]) });
      const b = resB[outName].data;
      sb = probStats(b);
      let same = 0;
      for (let i = 0; i < n; i++) { const d = Math.abs(a[i] - b[i]); if (!(d <= diff)) diff = d; if ((a[i] > 0.5) === (b[i] > 0.5)) same++; }
      agree = same / n;
      out.push(`wasm  : ${fmt(sb)}  (${(performance.now() - t0).toFixed(0)}ms)`,
        `差: 最大 ${Number.isFinite(diff) ? diff.toFixed(4) : 'NaN'} / 床判定(>0.5)の一致率 ${(agree * 100).toFixed(2)}%`);
      if (sess.outputNames.length > 1) {   // 診断用モデル：途中の値ごとに GPU と CPU を比べる
        out.push('---- 途中の値ごとの比較（相対誤差 = 最大差 / CPU側の最大絶対値。0.01未満なら一致）----');
        for (const name of sess.outputNames) {
          const ga = resA[name].data, cb = resB[name].data;
          let md = 0, mref = 0;
          for (let i = 0; i < cb.length; i++) { const d = Math.abs(ga[i] - cb[i]); if (!(d <= md)) md = d; const r = Math.abs(cb[i]); if (r > mref) mref = r; }
          const rel = md / Math.max(mref, 1e-6);
          out.push(`  ${name.padEnd(9)} [${resB[name].dims.join('x')}] 相対誤差 ${Number.isFinite(rel) ? rel.toFixed(4) : 'NaN'} ${rel < 0.01 ? '○' : '×'}`);
        }
      }
    }
    out.push('---- 判定 ----');
    if (Math.max(...ch) < 0.02) out.push('入力がほぼ真っ黒です。カメラ映像が取れていません。');
    else if (sb && (!(diff < 0.05) || agree < 0.99)) {
      out.push('WebGPU の計算結果が CPU(WASM) と違います（GPU側の計算ミス）。',
        sb.over > 0.01 ? '→ WASM では床が出ています。設定の「推論の実行先」を WASM にすると正しく動きます（遅くなります）。' : '→ WASM でも床が少ないので、モデル側の問題も重なっています。');
    } else if (sa.over < 0.01) out.push('GPUとCPUの結果は一致しています。モデル自体がこの映像で床を見つけていません。保存した画像をPCの tools/check_image.py で調べてください。');
    else out.push('床確率は出ています。マスク処理か表示の問題なので、この結果を送ってください。');
    $('summary').textContent = out.join('\n');
    log('診断完了');
    inCanvas.toBlob((blob) => { if (blob) download(`ashi_diag_${stamp()}.png`, blob); }, 'image/png');
    download(`ashi_diag_${stamp()}.txt`, out.join('\n') + '\n', 'text/plain');
  } catch (e) {
    out.push('診断エラー: ' + e.message);
    $('summary').textContent = out.join('\n');
    log('診断エラー:', e.message);
  } finally { preparing = false; }
}
$('btnDiag').onclick = diagnose;

// ---------------- WebGPU の設定の組み合わせを総当たりして、WASM と一致するものを探す ----------------
function captureInput() {
  const W = inCanvas.width, H = inCanvas.height, n = W * H;
  drawInput();
  const img = ictx.getImageData(0, 0, W, H).data;
  const x = new Float32Array(3 * n);
  for (let i = 0, p = 0; i < n; i++, p += 4) { x[i] = img[p] / 255; x[n + i] = img[p + 1] / 255; x[2 * n + i] = img[p + 2] / 255; }
  return { x, W, H, n };
}
async function gpuMatrix() {
  if (!sess || !video.videoWidth) { alert('カメラとモデルの準備ができてから押してください'); return; }
  if (recording) { alert('停止してから押してください'); return; }
  if (!navigator.gpu) { alert('この端末では WebGPU が使えません'); return; }
  preparing = true;
  const out = ['==== WebGPU 組み合わせテスト ====', `モデル ${modelInfo.label} / ORT ${ORT_VERSION} / app v${APP_VERSION}`];
  const show = (extra = '') => { $('summary').style.display = 'block'; $('summary').textContent = out.join('\n') + extra; };
  let best = null;
  try {
    while (busy) await sleep(5);
    const { x, W, H, n } = captureInput();
    const url = 'models/' + modelInfo.file;
    const feed = () => ({ [inName]: new window.ort.Tensor('float32', new Float32Array(x), [1, 3, H, W]) });
    show('\nWASM で正解を計算中…');
    if (!wasmSess || wasmFile !== modelInfo.file) {
      wasmSess = await window.ort.InferenceSession.create(url, { executionProviders: ['wasm'] });
      wasmFile = modelInfo.file;
    }
    const ref = Float32Array.from((await wasmSess.run(feed()))[outName].data);
    const sr = probStats(ref);
    out.push(`正解(wasm): 床確率>0.5 ${(sr.over * 100).toFixed(1)}%`, '最適化 / 並び : 一致率 / 最大差 / 推論時間');
    // 今動いているセッションは一度解放して、GPUのメモリを空けておく
    try { await sess.release(); } catch { /* 無視 */ }
    sess = null;
    for (const opt of ['all', 'extended', 'basic', 'disabled']) {
      for (const layout of ['NHWC', 'NCHW']) {
        show(`\n${opt} / ${layout} を試しています…`);
        let s = null;
        try {
          s = await window.ort.InferenceSession.create(url, gpuOpts(opt, layout));
          await s.run(feed());                                 // ウォームアップ
          const ts = []; let a = null;
          for (let k = 0; k < 3; k++) { const t0 = performance.now(); a = (await s.run(feed()))[outName].data; ts.push(performance.now() - t0); }
          let diff = 0, same = 0;
          for (let i = 0; i < n; i++) { const d = Math.abs(a[i] - ref[i]); if (!(d <= diff)) diff = d; if ((a[i] > 0.5) === (ref[i] > 0.5)) same++; }
          const agree = same / n, t = median(ts), ok = agree >= 0.99 && diff < 0.05;
          out.push(`${opt.padEnd(8)} / ${layout} : ${(agree * 100).toFixed(2)}% / ${Number.isFinite(diff) ? diff.toFixed(3) : 'NaN'} / ${t.toFixed(0)}ms ${ok ? '○' : '×'}`);
          if (ok && (!best || t < best.t)) best = { opt, layout, t };
        } catch (e) {
          out.push(`${opt.padEnd(8)} / ${layout} : エラー ${e.message.slice(0, 80)}`);
        } finally { try { await s?.release(); } catch { /* 無視 */ } }
      }
    }
    out.push('---- 結果 ----', best
      ? `○ の中で最速: ${best.opt} / ${best.layout}（${best.t.toFixed(0)}ms）`
      : 'どの組み合わせでも WASM と一致しませんでした。URL の末尾を ?ort=1.24.2 などに変えて、このテストをもう一度試してください。');
    show();
    download(`ashi_gputest_${stamp()}.txt`, out.join('\n') + '\n', 'text/plain');
  } catch (e) {
    out.push('テストエラー: ' + e.message); show();
  } finally {
    preparing = false;
  }
  if (best && confirm(`${best.opt} / ${best.layout} を使うように設定しますか？`)) { S.opt = best.opt; S.layout = best.layout; store.set('opt', S.opt); store.set('layout', S.layout); $('selOpt').value = S.opt; $('selLayout').value = S.layout; }
  await reprepare();
}
$('btnGpuTest').onclick = gpuMatrix;

// ---------------- 処理の停止を検知（カメラ途絶・推論の固まり） ----------------
setInterval(() => {
  if (!recording) return;
  const now = performance.now();
  if (now - lastDoneAt > STALL_MS && now - lastStallTx > DEFAULTS.resend * 1000) {
    if (guide.force(S.lostCmd, now / 1000)) log(`処理が${STALL_MS}ms以上止まっています → ${S.lostCmd}`);
    sendCmd(S.lostCmd);
    lastStallTx = now;
  }
}, 100);
document.addEventListener('visibilitychange', async () => {
  if (document.visibilityState === 'visible' && recording) {
    try { wakeLock = await navigator.wakeLock?.request('screen'); } catch { /* 無視 */ }
  } else if (recording) log('画面が裏に回りました（処理が止まり、ESP32側で S になります）');
});

// ---------------- ボタン・設定 ----------------
$('btnBle').onclick = bleConnect;
$('btnStart').onclick = start;
$('btnStop').onclick = stop;
$('btnSave').onclick = save;
$('btnHit').onclick = () => { collisions++; log(`衝突 +1 → ${collisions}`); };
$('btnUnhit').onclick = () => { collisions = Math.max(0, collisions - 1); log(`衝突 −1 → ${collisions}`); };
function setThresh(v) {
  S.thresh = Math.round(Math.min(0.6, Math.max(0.05, v)) * 100) / 100;
  guide.p.thresh = S.thresh; store.set('thresh', S.thresh); $('threshVal').textContent = S.thresh.toFixed(2);
}
$('btnThMinus').onclick = () => setThresh(S.thresh - 0.05);
$('btnThPlus').onclick = () => setThresh(S.thresh + 0.05);
const ROTS = ['auto', '90', '180', '270', '0'];
$('btnRot').onclick = () => { S.rot = ROTS[(ROTS.indexOf(S.rot) + 1) % ROTS.length]; $('selRot').value = S.rot; store.set('rot', S.rot); guide.probAvg = null; log('回転:', S.rot); };
$('btnTest').onclick = async () => {
  if (!ble.connected) { alert('先にBluetooth接続してください'); return; }
  if (recording) return;
  for (const c of ['L', 'R', 'S', 'C']) {
    log('振動テスト:', c);
    for (let i = 0; i < 6; i++) { sendCmd(c); await sleep(250); }
  }
  await sendNow('Q');
};
$('btnRestore').onclick = () => { const b = store.get('backup', null); if (b) download(b.name, b.csv, 'text/csv'); else alert('バックアップはありません'); };

function bindSetting(id, key, after) {
  const el = $(id);
  el.value = S[key];
  el.onchange = async () => {
    S[key] = el.type === 'number' ? +el.value : el.value; store.set(key, S[key]);
    if (!recording) Object.assign(guide.p, guideParams());
    if (after) await after();
  };
}
const reprepare = async () => { if (recording) { alert('記録中は変更できません'); return; } try { await prepareSession(); } catch (e) { log('準備エラー:', e.message); } };
bindSetting('selModel', 'model', reprepare);
bindSetting('selEp', 'ep', reprepare);
bindSetting('selOpt', 'opt', reprepare);
bindSetting('selThreads', 'threads', () => location.reload());  // スレッド数は最初の読み込み時にしか変えられない
bindSetting('selLayout', 'layout', reprepare);
bindSetting('selCam', 'cam', async () => { try { await openCamera(); } catch (e) { log('カメラエラー:', e.message); } });
bindSetting('selRot', 'rot', () => { guide.probAvg = null; });
bindSetting('selLost', 'lostCmd');
bindSetting('inHold', 'holdTime');
bindSetting('inSmooth', 'smooth');
bindSetting('inMinFloor', 'minFloor');
bindSetting('inLostFrames', 'lostFrames');
bindSetting('selDraw', 'draw');
setThresh(S.thresh);

// ---------------- 起動 ----------------
(async () => {
  log(`あししるべ app v${APP_VERSION} / ORT ${ORT_VERSION}`);
  if (!window.isSecureContext) log('注意: https で開いていないため、カメラとBluetoothが使えません');
  await loadModelList();
  try { await openCamera(); } catch (e) { log('カメラを開けません:', e.message); }
  if (models.length) { try { await prepareSession(); } catch (e) { log('準備エラー:', e.message); setPill('pEp', '推論 エラー', 'ng'); } }
})();
