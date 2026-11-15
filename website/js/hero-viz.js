// Hero visualisation: simulated orders flowing through Sentinel's model.
// Order -> 41 behavioural signals -> 5 gradient-boosted tree models (the bag) -> calibrated P(fraud)
// -> expected-cost policy -> allow / verify / decline, with the novelty gate (an Isolation Forest on the same
// signals) upgrading the most unusual 0.5% of would-be allows to verify (sentinel/pipeline.py).
// Orders and decisions come from engine.js. About 97% of real orders are allowed, so the animation shows
// every flagged order and 1 in ALLOW_SAMPLE allowed ones; the counters still count every order screened.
import { makeRng, makeOrder, decide, DEFAULT_COSTS } from './engine.js';

const COLORS = Object.freeze({
  line: 'rgba(255,255,255,0.09)', lineHot: 'rgba(232,201,171,0.75)', node: '#2a2e33',
  text: '#9aa0a6', ink: '#eceef0', allow: '#7fb69a', verify: '#c9a58a', decline: '#e0806f', order: '#eceef0',
});
const ACTION_COLOR = { allow: COLORS.allow, verify: COLORS.verify, decline: COLORS.decline };
const ACTION_LABEL = { allow: 'Allow', verify: 'Verify', decline: 'Decline' };
const N_FEATURES = 41;
const N_TREES = 5;
const TRAVEL_MS = 3400;
const SPAWN_MS = 850;
const GLOW_MS = 700;
const SEED = 412;
const WARM_SCREENED = 6000;
const ALLOW_SAMPLE = 30;
const NOVELTY_RATE = 0.005;
const MAX_DRAWS = 5000;

const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);
const lerp = (a, b, t) => a + (b - a) * t;
const along = (p, q, t) => ({ x: lerp(p.x, q.x, t), y: lerp(p.y, q.y, t) });

/** Positions of every node for a canvas of W x H (CSS pixels). */
function layout(W, H) {
  const narrow = W < 560;
  const pad = narrow ? 16 : 28;
  const top = narrow ? 34 : 44;
  const midY = top + (H - top - pad) / 2;
  const cols = narrow ? 5 : 6;
  const rows = Math.ceil(N_FEATURES / cols);
  const s = Math.min(narrow ? 11 : 15, (H - top - pad) / (rows + 1));
  const featX = W * (narrow ? 0.2 : 0.21);
  const features = Array.from({ length: N_FEATURES }, (_, i) => ({
    x: featX + ((i % cols) - (cols - 1) / 2) * s,
    y: midY + (Math.floor(i / cols) - (rows - 1) / 2) * s,
  }));
  const treeX = W * (narrow ? 0.4 : 0.42);
  const tw = Math.min(narrow ? 56 : 150, W * 0.15);
  const gateStrip = narrow ? 26 : 34;
  const band = (H - top - pad - gateStrip) / N_TREES;
  const trees = Array.from({ length: N_TREES }, (_, k) => {
    const y = top + band * (k + 0.5);
    const th = band * 0.78;
    return {
      root: { x: treeX, y },
      kids: [{ x: treeX + tw / 2, y: y - th / 4 }, { x: treeX + tw / 2, y: y + th / 4 }],
      leaves: [-3, -1, 1, 3].map((m) => ({ x: treeX + tw, y: y + (m * th) / 8 })),
    };
  });
  const score = { x: W * 0.68, y: midY, r: narrow ? 20 : 28 };
  const binX = W - pad - (narrow ? 34 : 70);
  const gap = Math.min(70, (H - top - pad) / 3.2);
  const bins = { allow: { x: binX, y: midY - gap }, verify: { x: binX, y: midY }, decline: { x: binX, y: midY + gap } };
  const gate = { x: W * 0.68, y: H - pad - (narrow ? 8 : 12) };
  return { W, H, narrow, top, midY, features, featCenter: { x: featX, y: midY }, trees, score, bins, gate, entryX: pad };
}

function pickLeaf(rng, risk) {
  const hi = Math.min(1, risk * 3 + rng() * 0.35);
  if (hi > 0.75) return 3;
  if (hi > 0.45) return 2;
  return rng() < 0.6 ? 0 : 1;
}

// Screens one order the way the pipeline does: the cost policy decides, then the novelty gate may
// upgrade an allow to verify (never to decline).
function spawn(rng, id, L) {
  const order = makeOrder(rng, id, id);
  const base = decide(order.risk, order.amount, DEFAULT_COSTS).action;
  const novel = base === 'allow' && rng() < NOVELTY_RATE;
  const action = novel ? 'verify' : base;
  return {
    order, action, novel, born: 0, flashed: false, scored: false, done: false,
    leaves: Array.from({ length: N_TREES }, () => pickLeaf(rng, order.risk)),
    hot: Array.from({ length: 6 + Math.floor(order.risk * 14) }, () => Math.floor(rng() * N_FEATURES)),
    entryY: L.midY + (rng() - 0.5) * 60,
  };
}

// The next order worth drawing. Every order screened on the way is counted; skipped orders are allows.
function nextShown(rng, s, L) {
  for (let i = 0; i < MAX_DRAWS; i++) {
    const o = spawn(rng, s.nextId++, L);
    s.screened += 1;
    if (o.action !== 'allow' || rng() < 1 / ALLOW_SAMPLE) return o;
    s.counts = { ...s.counts, allow: s.counts.allow + 1 };
  }
  return spawn(rng, s.nextId++, L);
}

const glow = (s, at) => Math.max(0, 1 - (s.now - (at ?? -1e9)) / GLOW_MS);

function drawStatic(ctx, L, s) {
  ctx.lineWidth = 1;
  ctx.strokeStyle = COLORS.line;
  ctx.beginPath();
  for (const t of L.trees) {
    ctx.moveTo(L.featCenter.x + 30, L.featCenter.y); ctx.lineTo(t.root.x, t.root.y);
    t.kids.forEach((k, i) => {
      ctx.moveTo(t.root.x, t.root.y); ctx.lineTo(k.x, k.y);
      [t.leaves[i * 2], t.leaves[i * 2 + 1]].forEach((l) => { ctx.moveTo(k.x, k.y); ctx.lineTo(l.x, l.y); });
    });
    t.leaves.forEach((l) => { ctx.moveTo(l.x, l.y); ctx.lineTo(L.score.x - L.score.r, L.score.y); });
  }
  Object.values(L.bins).forEach((b) => { ctx.moveTo(L.score.x + L.score.r, L.score.y); ctx.lineTo(b.x - 8, b.y); });
  ctx.stroke();
  ctx.setLineDash([3, 4]);
  ctx.beginPath();
  ctx.moveTo(L.featCenter.x, L.featCenter.y + 60); ctx.lineTo(L.featCenter.x, L.gate.y); ctx.lineTo(L.gate.x - 10, L.gate.y);
  ctx.moveTo(L.gate.x + 10, L.gate.y); ctx.lineTo(L.bins.verify.x - 8, L.bins.verify.y);
  ctx.stroke();
  ctx.setLineDash([]);
  L.features.forEach((f, i) => {
    const g = glow(s, s.featGlow[i]);
    ctx.fillStyle = g > 0 ? `rgba(232,201,171,${0.25 + g * 0.75})` : COLORS.node;
    ctx.beginPath(); ctx.arc(f.x, f.y, L.narrow ? 2.6 : 3.4, 0, Math.PI * 2); ctx.fill();
  });
  L.trees.forEach((t, k) => {
    [t.root, ...t.kids, ...t.leaves].forEach((n, j) => {
      const g = glow(s, s.treeGlow[k][j]);
      ctx.fillStyle = g > 0 ? `rgba(232,201,171,${0.3 + g * 0.7})` : COLORS.node;
      ctx.beginPath(); ctx.arc(n.x, n.y, j === 0 ? 4 : 3, 0, Math.PI * 2); ctx.fill();
    });
  });
}

function drawGate(ctx, L, s) {
  const g = glow(s, s.gateAt);
  ctx.fillStyle = '#17191c';
  ctx.strokeStyle = g > 0 ? `rgba(201,165,138,${0.4 + g * 0.6})` : 'rgba(255,255,255,0.18)';
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.roundRect(L.gate.x - 10, L.gate.y - 8, 20, 16, 4); ctx.fill(); ctx.stroke();
  ctx.fillStyle = g > 0 ? COLORS.verify : COLORS.text;
  ctx.font = `${L.narrow ? 10 : 12}px "Familjen Grotesk", system-ui, sans-serif`;
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  ctx.fillText(L.narrow ? 'Novelty gate' : 'Novelty gate (Isolation Forest)', L.gate.x - 18, L.gate.y);
}

function drawScore(ctx, L, s) {
  const g = glow(s, s.scoreAt);
  ctx.fillStyle = '#17191c';
  ctx.strokeStyle = g > 0 ? `rgba(232,201,171,${0.4 + g * 0.6})` : 'rgba(255,255,255,0.18)';
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(L.score.x, L.score.y, L.score.r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  const p = s.lastRisk;
  ctx.fillStyle = COLORS.ink;
  ctx.font = `500 ${L.narrow ? 11 : 13}px "IBM Plex Mono", ui-monospace, monospace`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(p < 0.001 ? '<0.1%' : `${(p * 100).toFixed(p < 0.01 ? 1 : 0)}%`, L.score.x, L.score.y);
}

function drawBins(ctx, L, s) {
  ctx.textBaseline = 'middle';
  Object.entries(L.bins).forEach(([a, b]) => {
    const g = glow(s, s.binAt[a]);
    ctx.fillStyle = ACTION_COLOR[a];
    ctx.globalAlpha = 0.16 + g * 0.5;
    ctx.beginPath(); ctx.roundRect(b.x - 8, b.y - 13, L.narrow ? 44 : 84, 26, 6); ctx.fill();
    ctx.globalAlpha = 1;
    ctx.font = `500 ${L.narrow ? 10 : 12}px "Familjen Grotesk", system-ui, sans-serif`;
    ctx.textAlign = 'left';
    const n = s.counts[a].toLocaleString('en-US');
    ctx.fillText(L.narrow ? n : `${ACTION_LABEL[a]} ${n}`, b.x, b.y);
  });
}

function drawLabels(ctx, L) {
  ctx.fillStyle = COLORS.text;
  ctx.font = `${L.narrow ? 10 : 12}px "Familjen Grotesk", system-ui, sans-serif`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
  const y = L.top - 16;
  const t0 = L.trees[0];
  ctx.fillText(L.narrow ? 'Signals' : '41 behavioural signals', L.featCenter.x, y);
  ctx.fillText(L.narrow ? '5 tree models' : '5 gradient-boosted tree models', (t0.root.x + t0.leaves[0].x) / 2, y);
  ctx.fillText(L.narrow ? 'Risk' : 'Calibrated risk', L.score.x, y);
  ctx.fillText(L.narrow ? 'Action' : 'Cheapest action', L.bins.verify.x + (L.narrow ? 14 : 34), y);
}

// One order at progress t in [0, 1]; marks glows and counts once as it passes each stage.
function drawOrder(ctx, L, s, o, t) {
  const dot = (p, r, c) => { ctx.fillStyle = c; ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2); ctx.fill(); };
  if (t < 0.22) { dot(along({ x: L.entryX, y: o.entryY }, L.featCenter, ease(t / 0.22)), 4, COLORS.order); return; }
  if (!o.flashed) { o.flashed = true; o.hot.forEach((i) => { s.featGlow[i] = s.now; }); }
  if (t < 0.56) {
    const u = (t - 0.22) / 0.34;
    const seg = Math.min(2, Math.floor(u * 3));
    const f = Math.min(1, u * 3 - seg);
    L.trees.forEach((tree, k) => {
      const leaf = o.leaves[k], kid = Math.floor(leaf / 2);
      const path = [{ x: L.featCenter.x + 30, y: L.featCenter.y }, tree.root, tree.kids[kid], tree.leaves[leaf]];
      dot(along(path[seg], path[seg + 1], ease(f)), 2.6, COLORS.lineHot);
      s.treeGlow[k][[0, 1 + kid, 3 + leaf][seg]] = s.now;
    });
    return;
  }
  if (t < 0.74) {
    const u = ease((t - 0.56) / 0.18);
    L.trees.forEach((tree, k) => dot(along(tree.leaves[o.leaves[k]], { x: L.score.x - L.score.r, y: L.score.y }, u), 2.6, COLORS.lineHot));
    return;
  }
  if (!o.scored) { o.scored = true; s.lastRisk = o.order.risk; s.scoreAt = s.now; }
  if (t < 1) {
    const bin = L.bins[o.action];
    const u = (t - 0.74) / 0.26;
    if (o.novel) {
      // the gate catches an unusual order the policy would have allowed and sends it to verify
      if (u < 0.5) { dot(along(L.featCenter, L.gate, ease(u * 2)), 4, COLORS.verify); return; }
      s.gateAt = s.now;
      dot(along(L.gate, { x: bin.x - 8, y: bin.y }, ease(u * 2 - 1)), 4.5, COLORS.verify);
      return;
    }
    dot(along({ x: L.score.x + L.score.r, y: L.score.y }, { x: bin.x - 8, y: bin.y }, ease(u)), 4.5, ACTION_COLOR[o.action]);
    return;
  }
  if (!o.done) { o.done = true; s.counts = { ...s.counts, [o.action]: s.counts[o.action] + 1 }; s.binAt[o.action] = s.now; }
}

/** Mounts the animation on a container holding a <canvas> and an optional [data-viz-pause] button. */
export function mountHeroViz(root) {
  const canvas = root?.querySelector('canvas');
  if (!canvas) return;
  const btn = root.querySelector('[data-viz-pause]');
  const ctx = canvas.getContext('2d');
  const rng = makeRng(SEED);
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const note = root.querySelector('[data-viz-screened]');
  const s = { now: 0, nextId: 1, screened: 0, featGlow: [], treeGlow: Array.from({ length: N_TREES }, () => []), scoreAt: -1e9, gateAt: -1e9, binAt: {}, lastRisk: 0.34, counts: { allow: 0, verify: 0, decline: 0 } };
  let L = null, lastSpawn = -1e9, raf = 0, paused = reduce, visible = true, orders = [];

  const paint = (list, t) => {
    ctx.clearRect(0, 0, L.W, L.H);
    drawLabels(ctx, L); drawStatic(ctx, L, s);
    list.forEach((o) => drawOrder(ctx, L, s, o, t ?? (s.now - o.born) / TRAVEL_MS));
    drawGate(ctx, L, s); drawScore(ctx, L, s); drawBins(ctx, L, s);
    if (note) note.textContent = s.screened.toLocaleString('en-US');
  };
  const frame = (now) => {
    s.now = now;
    if (now - lastSpawn > SPAWN_MS) { orders = [...orders, { ...nextShown(rng, s, L), born: now }]; lastSpawn = now; }
    paint(orders);
    orders = orders.filter((o) => now - o.born < TRAVEL_MS + 50);
    if (!paused && visible) raf = requestAnimationFrame(frame);
  };
  // The still picture for reduced motion or a pause: one order part-way through the trees.
  const still = () => { const o = nextShown(rng, s, L); s.now = 1e6; paint([o], 0.45); paint([o], 0.45); };
  const start = () => { cancelAnimationFrame(raf); if (!paused && visible) raf = requestAnimationFrame(frame); };
  const resize = () => {
    const r = canvas.getBoundingClientRect(), dpr = Math.min(2, devicePixelRatio || 1);
    canvas.width = Math.round(r.width * dpr); canvas.height = Math.round(r.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    L = layout(r.width, r.height);
  };

  resize();
  for (let i = 0; i < WARM_SCREENED; i++) { const o = spawn(rng, s.nextId++, L); s.screened += 1; s.counts[o.action] += 1; }
  if (paused) still(); else start();

  new ResizeObserver(() => { resize(); if (paused) still(); }).observe(canvas);
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; start(); }).observe(canvas);
  document.addEventListener('visibilitychange', () => { visible = !document.hidden; start(); });
  btn?.addEventListener('click', () => {
    paused = !paused;
    btn.setAttribute('aria-pressed', String(paused));
    btn.setAttribute('aria-label', paused ? 'Play model animation' : 'Pause model animation');
    if (paused) { cancelAnimationFrame(raf); } else { orders = []; start(); }
  });
  root.dataset.ready = 'true';
}
