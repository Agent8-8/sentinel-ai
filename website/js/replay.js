// Live replay console: streams simulated orders through the policy, keeps running totals,
// and explains any selected order (risk, reason codes, the three expected costs).
import { makeRng, makeOrder, decide, realise, allowAllCost, reasonCodes, DEFAULT_COSTS, FRAUD_LABELS } from './engine.js';
import { overviewHtml } from './overview.js';

const HISTORY_EVERY = 10;
const HISTORY_MAX = 160;

const SEED = 74;
const TICK_MS = 600;
const ORDERS_PER_TICK = 6;
const STEP_ORDERS = 100;
const FEED_ROWS = 24;
const ROW_H = 48;
const MIN_ROWS = 4;
const WARM_UP_ORDERS = 1200;
const START_MINUTE = 74 * 24 * 60;

const usd = (v, d = 0) => '$' + v.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (v, d = 0) => (v * 100).toFixed(d) + '%';
const riskText = (p) => (p < 0.001 ? '<0.1' : p < 0.01 ? (p * 100).toFixed(1) : Math.round(p * 100)) + '%';
const clock = (minute) => {
  const day = Math.floor(minute / 1440), m = minute % 1440;
  return `Day ${day} · ${String(Math.floor(m / 60)).padStart(2, '0')}:${String(m % 60).padStart(2, '0')}`;
};
const ACTION_WORD = { allow: 'Allowed', verify: 'Verify', decline: 'Declined' };

function freshState() {
  return {
    rng: makeRng(SEED), outcomeRng: makeRng(SEED * 31), nextId: 58000, minute: START_MINUTE,
    rows: [], totals: { n: 0, cost: 0, allowAll: 0, fraudDollars: 0, fraudStopped: 0, genuine: 0, falseDeclines: 0, verified: 0, fraud: 0 },
    history: [{ n: 0, cost: 0, allowAll: 0 }], actions: { allow: 0, verify: 0, decline: 0 }, byType: {},
  };
}

// Keep the chart's points bounded: when full, drop every other point (the newest always stays).
function pushHistory(history, point) {
  const next = [...history, point];
  return next.length <= HISTORY_MAX ? next : next.filter((_, i) => i % 2 === 0 || i === next.length - 1);
}

function addType(byType, order, result) {
  if (!order.isFraud) return byType;
  const prev = byType[order.type] ?? { n: 0, caught: 0 };
  return { ...byType, [order.type]: { n: prev.n + 1, caught: prev.caught + (result.fraudLoss ? 0 : 1) } };
}

function addTotals(t, order, result, decision, c) {
  const genuine = !order.isFraud;
  return {
    n: t.n + 1,
    cost: t.cost + result.cost,
    allowAll: t.allowAll + allowAllCost(order, c),
    fraudDollars: t.fraudDollars + (order.isFraud ? order.amount : 0),
    fraudStopped: t.fraudStopped + (order.isFraud && !result.fraudLoss ? order.amount : 0),
    genuine: t.genuine + (genuine ? 1 : 0),
    falseDeclines: t.falseDeclines + (result.falseDecline ? 1 : 0),
    verified: t.verified + (decision.action === 'verify' ? 1 : 0),
    fraud: t.fraud + (order.isFraud ? 1 : 0),
  };
}

function advance(state, count, c) {
  let { totals, rows, minute, nextId, history, actions, byType } = state;
  const added = [];
  for (let i = 0; i < count; i++) {
    minute += state.rng() < 0.5 ? 1 : 2;
    const order = makeOrder(state.rng, nextId++, minute);
    const decision = decide(order.risk, order.amount, c);
    const result = realise(order, decision.action, c, state.outcomeRng);
    totals = addTotals(totals, order, result, decision, c);
    actions = { ...actions, [decision.action]: actions[decision.action] + 1 };
    byType = addType(byType, order, result);
    if (totals.n % HISTORY_EVERY === 0) history = pushHistory(history, { n: totals.n, cost: totals.cost, allowAll: totals.allowAll });
    added.push({ order, decision, result });
  }
  rows = [...added.reverse(), ...rows].slice(0, FEED_ROWS);
  return { ...state, totals, rows, minute, nextId, history, actions, byType };
}

function rowHtml({ order, decision }, selectedId, newestSeen) {
  const sel = order.id === selectedId ? ' is-on' : '';
  return `<li${order.id > newestSeen ? " class=\"is-new\"" : ""}><button type="button" class="rp-row${sel}" data-id="${order.id}" aria-pressed="${order.id === selectedId}">
    <span class="rp-row__id">#${order.id}<small>${order.category}</small></span>
    <span>${usd(order.amount, 2)}</span>
    <span class="rp-risk">${riskText(order.risk)}</span>
    <span class="rp-chip rp-chip--${decision.action}">${ACTION_WORD[decision.action]}</span></button></li>`;
}

function caseHtml({ order, decision, result }) {
  const reasons = reasonCodes(order);
  const max = Math.max(...Object.values(decision.costs), 0.01);
  const costRow = (a) => `<div class="mini-cost__row${a === decision.action ? ' is-pick' : ''}"><span>${a[0].toUpperCase() + a.slice(1)}</span><i style="--w:${Math.max(1, (decision.costs[a] / max) * 100).toFixed(1)}%"></i><b>${usd(decision.costs[a], 2)}</b></div>`;
  const truth = order.isFraud ? `Fraud, ${FRAUD_LABELS[order.type]}` : 'Genuine order';
  const verification = result.verification ? `<div><dt>Verification</dt><dd>${result.verification}</dd></div>` : '';
  const reasonList = reasons.length
    ? reasons.map((r) => `<li><span>${r.label}</span><b>+${(r.drop * 100).toFixed(1)} pts</b></li>`).join('')
    : '<li><span>No signal stands out from typical genuine orders</span></li>';
  return `<div class="rp-case">
    <button type="button" class="rp-back" data-rp="overview">← Back to run overview</button>
    <p class="t-label muted">Order #${order.id} · ${clock(order.minute)}</p>
    <p class="rp-case__head"><b>${usd(order.amount, 2)}</b> ${order.category} · risk ${riskText(order.risk)}</p>
    <p class="rp-chip rp-chip--${decision.action} rp-chip--lg">${ACTION_WORD[decision.action]}</p>
    <h3 class="rp-h">Expected cost of each action</h3>
    <div class="mini-cost mini-cost--flat">${['allow', 'verify', 'decline'].map(costRow).join('')}</div>
    <h3 class="rp-h">Top reasons</h3>
    <ul class="rp-reasons">${reasonList}</ul>
    <dl class="rp-truth"><div><dt>Simulator truth</dt><dd>${truth}</dd></div>${verification}</dl>
  </div>`;
}

export function mountReplay(root) {
  if (!root) return;
  const c = DEFAULT_COSTS;
  const feed = root.querySelector('[data-feed]');
  const pane = root.querySelector('[data-case]');
  const playBtn = root.querySelector('[data-rp="play"]');
  const kpi = (k) => root.querySelector(`[data-k="${k}"]`);
  let state = advance(freshState(), WARM_UP_ORDERS, c);
  let selectedId = null;
  let timer = null;
  let newestSeen = Infinity;

  const renderKpis = () => {
    const t = state.totals;
    kpi('n').textContent = t.n.toLocaleString('en-US');
    kpi('stop').textContent = pct(t.fraudDollars ? t.fraudStopped / t.fraudDollars : 0);
    kpi('save').textContent = usd(Math.max(0, t.allowAll - t.cost));
    kpi('fd').textContent = pct(t.genuine ? t.falseDeclines / t.genuine : 0, 2);
  };
  const renderPane = () => {
    const row = state.rows.find((r) => r.order.id === selectedId);
    pane.innerHTML = row ? caseHtml(row) : overviewHtml(state);
  };
  const render = () => {
    const fit = Math.max(MIN_ROWS, Math.floor(feed.clientHeight / ROW_H) || MIN_ROWS);
    feed.innerHTML = state.rows.slice(0, fit).map((r) => rowHtml(r, selectedId, newestSeen)).join('');
    newestSeen = state.rows[0]?.order.id ?? newestSeen;
    renderKpis();
    if (!selectedId) renderPane();
  };
  const stop = () => { clearInterval(timer); timer = null; playBtn.textContent = 'Run replay'; playBtn.setAttribute('aria-pressed', 'false'); };
  const play = () => {
    timer = setInterval(() => { state = advance(state, ORDERS_PER_TICK, c); render(); }, TICK_MS);
    playBtn.textContent = 'Pause replay'; playBtn.setAttribute('aria-pressed', 'true');
  };

  root.addEventListener('click', (e) => {
    const ctrl = e.target.closest('[data-rp]');
    if (ctrl?.dataset.rp === 'play') return timer ? stop() : play();
    if (ctrl?.dataset.rp === 'overview') { selectedId = null; render(); return renderPane(); }
    if (ctrl?.dataset.rp === 'step') { state = advance(state, STEP_ORDERS, c); return render(); }
    if (ctrl?.dataset.rp === 'reset') { stop(); state = advance(freshState(), WARM_UP_ORDERS, c); selectedId = null; newestSeen = Infinity; return render(); }
    const row = e.target.closest('.rp-row');
    if (row) {
      const id = Number(row.dataset.id);
      selectedId = selectedId === id ? null : id;
      render();
      renderPane();
    }
  });

  render();
  new ResizeObserver(() => render()).observe(feed);
  root.dataset.ready = "true";
}
