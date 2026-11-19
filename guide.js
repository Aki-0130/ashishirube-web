// 床確率マップ → 床マスク → 中心線 → 指示（L/R/C/S）
// 前回の realtime_guide.py（OpenCV版）と同じ処理を JavaScript で書き直したもの。
// ブラウザとNodeの両方から import できるよう、DOMには触らない。

export const DEFAULTS = {
  thresh: 0.22,      // 直進とみなすoffsetの幅
  returnRatio: 0.55, // 戻りしきい値 = thresh * returnRatio（ヒステリシス）
  holdTime: 0.15,    // L/R 切替の最小間隔 [s]
  minFloor: 0.02,    // 床面積比率がこれ未満なら床ロスト
  lostFrames: 3,     // 床ロストが何フレーム続いたら lostCmd を出すか
  lostCmd: 'S',      // 'S' 両側振動(止まれ) / 'C'
  resend: 0.3,       // 同じ指示の再送間隔 [s]（ESP32側ウォッチドッグ用）
  smooth: 0.25,      // 前フレームの確率との平均の重み
};

// Python の round()（偶数丸め）と OpenCV の cvRound に合わせる
export function roundHalfEven(x) {
  const f = Math.floor(x), d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

// cv2.getStructuringElement(MORPH_ELLIPSE, (ks, ks)) の各行の半幅
export function ellipseHalfWidths(ks) {
  const r = ks >> 1, c = r, invR2 = r ? 1 / (r * r) : 0;
  const hws = [];
  for (let i = 0; i < ks; i++) {
    const dy = i - r;
    const dx = roundHalfEven(c * Math.sqrt((r * r - dy * dy) * invR2));
    const j1 = Math.max(c - dx, 0), j2 = Math.min(c + dx + 1, ks);
    hws.push({ dy, left: c - j1, right: j2 - 1 - c });
  }
  return hws;
}

function rowPrefix(m, w, h) {
  const P = new Int32Array((w + 1) * h);
  for (let y = 0; y < h; y++) {
    let s = 0, base = y * (w + 1), row = y * w;
    for (let x = 0; x < w; x++) { s += m[row + x]; P[base + x + 1] = s; }
  }
  return P;
}

// 行ごとの横方向の判定を先に作り（半幅の種類ぶんだけ）、縦方向に AND / OR で重ねる。
// 結果は cv2.erode / cv2.dilate（楕円カーネル、既定の境界処理）と一致する
function morph(m, w, h, hws, isErode) {
  const P = rowPrefix(m, w, h), cache = new Map();
  const out = new Uint8Array(w * h);
  if (isErode) out.fill(1);
  const w4 = w % 4 === 0 ? w / 4 : 0, out32 = w4 ? new Uint32Array(out.buffer) : null, cache32 = new Map();
  for (const k of hws) {
    const lo = isErode ? k.left : k.right, hi = isErode ? k.right : k.left, key = lo * 1000 + hi;
    let Hk = cache.get(key);
    if (!Hk) {
      Hk = new Uint8Array(w * h);
      for (let y = 0; y < h; y++) {
        const b = y * (w + 1), o = y * w;
        for (let x = 0; x < w; x++) {
          const x0 = x - lo < 0 ? 0 : x - lo, x1 = x + hi > w - 1 ? w - 1 : x + hi;
          const s = P[b + x1 + 1] - P[b + x0];
          Hk[o + x] = isErode ? (s === x1 - x0 + 1 ? 1 : 0) : (s > 0 ? 1 : 0);
        }
      }
      cache.set(key, Hk);
      if (w4) cache32.set(key, new Uint32Array(Hk.buffer));
    }
    for (let y = 0; y < h; y++) {
      const yy = y + k.dy;
      if (yy < 0 || yy >= h) continue;            // 画像外の行：収縮では1、膨張では0 扱い
      if (w4) {                                    // 4画素ずつまとめて処理（幅が4の倍数のとき）
        const o = y * w4, sOff = yy * w4, H32 = cache32.get(key);
        if (isErode) for (let x = 0; x < w4; x++) out32[o + x] &= H32[sOff + x];
        else for (let x = 0; x < w4; x++) out32[o + x] |= H32[sOff + x];
      } else {
        const o = y * w, sOff = yy * w;
        if (isErode) for (let x = 0; x < w; x++) out[o + x] &= Hk[sOff + x];
        else for (let x = 0; x < w; x++) out[o + x] |= Hk[sOff + x];
      }
    }
  }
  return out;
}
export const erode = (m, w, h, hws) => morph(m, w, h, hws, true);
export const dilate = (m, w, h, hws) => morph(m, w, h, hws, false);

// 8連結ラベリング。ラベルはラスタ順に最初に見つかった画素の順。
// 周囲に1画素の枠を足して、境界チェックを省いている（速度のため）
function label8(m, w, h) {
  const W2 = w + 2, lab = new Int32Array(W2 * (h + 2)), stack = new Int32Array(w * h);
  const pad = new Uint8Array(W2 * (h + 2));
  for (let y = 0; y < h; y++) pad.set(m.subarray(y * w, y * w + w), (y + 1) * W2 + 1);
  const nb = [-W2 - 1, -W2, -W2 + 1, -1, 1, W2 - 1, W2, W2 + 1];
  const areas = [0];
  let n = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0, i = (y + 1) * W2 + 1; x < w; x++, i++) {
      if (!pad[i] || lab[i]) continue;
      n++; let area = 0, sp = 0;
      stack[sp++] = i; lab[i] = n;
      while (sp) {
        const p = stack[--sp]; area++;
        for (let k = 0; k < 8; k++) {
          const q = p + nb[k];
          if (pad[q] && !lab[q]) { lab[q] = n; stack[sp++] = q; }
        }
      }
      areas.push(area);
    }
  }
  const out = new Int32Array(w * h);
  for (let y = 0; y < h; y++) out.set(lab.subarray((y + 1) * W2 + 1, (y + 1) * W2 + 1 + w), y * w);
  return { n: n + 1, lab: out, areas };
}

// 穴埋め：外周から4連結でたどれない背景を埋める（外輪郭の塗りつぶしと同じ）
function fillHoles(m, w, h) {
  const W2 = w + 2, N2 = W2 * (h + 2);
  const seen = new Uint8Array(N2), stack = new Int32Array(N2);
  const fg = new Uint8Array(N2);
  for (let y = 0; y < h; y++) fg.set(m.subarray(y * w, y * w + w), (y + 1) * W2 + 1);
  // 外枠（画像外）は背景としてつながっている扱い。枠の外へは出ない
  let sp = 0;
  seen[0] = 1; stack[sp++] = 0;
  while (sp) {
    const p = stack[--sp], px = p % W2;
    if (px > 0 && !fg[p - 1] && !seen[p - 1]) { seen[p - 1] = 1; stack[sp++] = p - 1; }
    if (px < W2 - 1 && !fg[p + 1] && !seen[p + 1]) { seen[p + 1] = 1; stack[sp++] = p + 1; }
    if (p >= W2 && !fg[p - W2] && !seen[p - W2]) { seen[p - W2] = 1; stack[sp++] = p - W2; }
    if (p < N2 - W2 && !fg[p + W2] && !seen[p + W2]) { seen[p + W2] = 1; stack[sp++] = p + W2; }
  }
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0, i = (y + 1) * W2 + 1; x < w; x++, i++) out[y * w + x] = seen[i] ? 0 : 1;
  }
  return out;
}

const hwCache = new Map();
export function cleanMask(mask, w, h) {
  const ks = Math.max(5, roundHalfEven(15 * Math.max(h, w) / 640) | 1);
  if (!hwCache.has(ks)) hwCache.set(ks, ellipseHalfWidths(ks));
  const hws = hwCache.get(ks);
  let m = dilate(erode(mask, w, h, hws), w, h, hws);   // open
  m = erode(dilate(m, w, h, hws), w, h, hws);          // close
  const { n, lab, areas } = label8(m, w, h);
  if (n <= 1) return m;
  const counts = new Int32Array(n);
  let any = false;
  for (let y = Math.max(0, h - 5); y < h; y++) {
    for (let x = Math.floor(w / 4); x < Math.floor(3 * w / 4); x++) {
      const l = lab[y * w + x];
      if (l > 0) { counts[l]++; any = true; }
    }
  }
  let keep = 1;
  if (any) { for (let l = 1; l < n; l++) if (counts[l] > counts[keep]) keep = l; }
  else { for (let l = 1; l < n; l++) if (areas[l] > areas[keep]) keep = l; }
  const sel = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) sel[i] = lab[i] === keep ? 1 : 0;
  return fillHoles(sel, w, h);
}

export function centerLine(mask, w, h) {
  const pts = [];
  const step = Math.max(4, roundHalfEven(10 * h / 640));
  const yEnd = Math.floor(h * 0.3);
  for (let y = h - 1; y > yEnd; y -= step) {
    let first = -1, last = -1, cnt = 0;
    const row = y * w;
    for (let x = 0; x < w; x++) if (mask[row + x]) { if (first < 0) first = x; last = x; cnt++; }
    if (cnt < w * 0.05) continue;
    pts.push([(first + last) / 2, y]);
  }
  if (pts.length < 5) return { pts, line: null };
  // np.polyfit(y, x, 1)
  let sy = 0, sx = 0;
  for (const [x, y] of pts) { sx += x; sy += y; }
  const my = sy / pts.length, mx = sx / pts.length;
  let num = 0, den = 0;
  for (const [x, y] of pts) { num += (y - my) * (x - mx); den += (y - my) * (y - my); }
  const a = num / den, b = mx - a * my;
  return { pts, line: [a, b] };
}

export function decideCmd(cur, offset, thresh, ret) {
  if (offset === null) return cur;
  if (cur === 'C' || cur === 'S') {
    if (offset < -thresh) return 'L';
    if (offset > thresh) return 'R';
    return 'C';
  }
  if (cur === 'L') {
    if (offset > thresh) return 'R';
    if (offset > -ret) return 'C';
    return 'L';
  }
  if (cur === 'R') {
    if (offset < -thresh) return 'L';
    if (offset < ret) return 'C';
    return 'R';
  }
  return cur;
}

// 1フレーム分の判定と送信タイミングを管理する
export class Guide {
  constructor(params = {}) {
    this.p = { ...DEFAULTS, ...params };
    this.reset();
  }
  reset() {
    this.probAvg = null;
    this.cmd = 'C';
    this.lastSwitch = -1e9;
    this.lastTx = -1e9;
    this.lostCount = 0;
  }
  // prob: Float32Array (w*h) 床確率, now: 秒
  step(prob, w, h, now) {
    const p = this.p;
    if (!this.probAvg || this.probAvg.length !== prob.length) this.probAvg = Float32Array.from(prob);
    else for (let i = 0; i < prob.length; i++) this.probAvg[i] = p.smooth * this.probAvg[i] + (1 - p.smooth) * prob[i];

    const raw = new Uint8Array(w * h);
    for (let i = 0; i < raw.length; i++) raw[i] = this.probAvg[i] > 0.5 ? 1 : 0;
    const mask = cleanMask(raw, w, h);
    let area = 0;
    for (let i = 0; i < mask.length; i++) area += mask[i];
    const floorRatio = area / mask.length;

    const { pts, line } = centerLine(mask, w, h);
    let offset = null, slope = null, angleDeg = null;
    if (line) {
      const [a, b] = line;
      offset = Math.min(1, Math.max(-1, ((a * h * 0.6 + b) - w / 2) / (w / 2)));
      slope = a;
      angleDeg = Math.atan(-a) * 180 / Math.PI;
    }

    const lostNow = offset === null || floorRatio < p.minFloor;
    this.lostCount = lostNow ? this.lostCount + 1 : 0;
    const lost = this.lostCount >= p.lostFrames;
    let desired;
    if (lost) desired = p.lostCmd;
    else if (lostNow) desired = this.cmd;          // 一瞬のロストは直前の指示を維持
    else desired = decideCmd(this.cmd, offset, p.thresh, p.thresh * p.returnRatio);

    let send = false;
    if (desired !== this.cmd) {
      if (desired === 'C' || desired === 'S' || now - this.lastSwitch >= p.holdTime) {
        this.cmd = desired; this.lastSwitch = now; send = true;
      }
    }
    if (!send && now - this.lastTx >= p.resend) send = true;
    if (send) this.lastTx = now;

    return { mask, pts, line, floorRatio, offset, slope, angleDeg, lost, cmd: this.cmd, send };
  }
  // カメラ途絶など外部要因で指示を強制する
  force(cmd, now) {
    const changed = cmd !== this.cmd;
    this.cmd = cmd; this.lastSwitch = now; this.lastTx = now;
    return changed;
  }
}
