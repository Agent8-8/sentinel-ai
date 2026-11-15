// Run overview for the replay console: drawn when no order is selected.
// A cumulative-cost chart (Sentinel against allowing everything), the decision mix,
// fraud caught by pattern, and a one-line briefing. All values come from the replay state.
import { FRAUD_LABELS } from './engine.js';

const CHART_W = 560;
const CHART_H = 180;
const PATTERN_ORDER = ['card_testing', 'stolen_card', 'ato', 'sleeper_ring', 'friendly'];

const usd = (v) => '$' + Math.round(v).toLocaleString('en-US');
const pct = (v, d = 0) => (v * 100).toFixed(d) + '%';
const plural = (n, one, many) => `${n.toLocaleString('en-US')} ${n === 1 ? one : many}`;

/**
 * Two cumulative lines over the orders screened, as SVG paths.
 * @param {{n:number, cost:number, allowAll:number}[]} history
 * @returns {string}
 */
function costChart(history) {
  if (history.length < 2) return '';
  const last = history[history.length - 1];
  const maxY = Math.max(last.allowAll, last.cost, 1) * 1.08;
  const x = (n) => ((n - history[0].n) / Math.max(1, last.n - history[0].n)) * CHART_W;
  const y = (v) => CHART_H - (v / maxY) * CHART_H;
  const line = (key) => history.map((p, i) => `${i ? 'L' : 'M'}${x(p.n).toFixed(1)},${y(p[key]).toFixed(1)}`).join('');
  const area = (key) => `${line(key)}L${CHART_W},${CHART_H}L0,${CHART_H}Z`;
  const grid = [0.25, 0.5, 0.75].map((f) => `<line x1="0" x2="${CHART_W}" y1="${(CHART_H * f).toFixed(1)}" y2="${(CHART_H * f).toFixed(1)}" />`).join('');
  return `<svg class="ov-chart" viewBox="0 0 ${CHART_W} ${CHART_H}" preserveAspectRatio="none" role="img"
      aria-label="Cumulative simulated cost: allowing every order reaches ${usd(last.allowAll)}, Sentinel ${usd(last.cost)}">
    <g class="ov-grid">${grid}</g>
    <path class="ov-area ov-area--all" d="${area('allowAll')}" />
    <path class="ov-area ov-area--sen" d="${area('cost')}" />
    <path class="ov-line ov-line--all" d="${line('allowAll')}" vector-effect="non-scaling-stroke" />
    <path class="ov-line ov-line--sen" d="${line('cost')}" vector-effect="non-scaling-stroke" />
  </svg>`;
}

function mixBlock(actions, n) {
  const share = (a) => (n ? actions[a] / n : 0);
  const seg = (a) => `<i class="ov-mix__${a}" style="flex-grow:${share(a).toFixed(4)}"></i>`;
  const row = (a, label) => `<li><i class="ov-dot ov-dot--${a}"></i>${label}<b>${actions[a].toLocaleString('en-US')}</b></li>`;
  return `<div class="ov-card">
    <p class="ov-h">Decisions</p>
    <div class="ov-mix" role="img" aria-label="Allowed ${pct(share('allow'), 1)}, verified ${pct(share('verify'), 1)}, declined ${pct(share('decline'), 1)}">${seg('allow')}${seg('verify')}${seg('decline')}</div>
    <ul class="ov-legend">${row('allow', 'Allowed')}${row('verify', 'Verified')}${row('decline', 'Declined')}</ul>
  </div>`;
}

function patternBlock(byType) {
  const rows = PATTERN_ORDER.filter((t) => byType[t]?.n).map((t) => {
    const { n, caught } = byType[t];
    return `<li><span>${FRAUD_LABELS[t]}</span><i style="--v:${((caught / n) * 100).toFixed(0)}%"></i><b>${caught}/${n}</b></li>`;
  }).join('');
  return `<div class="ov-card">
    <p class="ov-h">Fraud stopped by pattern</p>
    <ul class="ov-patterns">${rows || '<li><span>No fraud yet</span></li>'}</ul>
  </div>`;
}

/**
 * The overview pane's markup.
 * @param {{totals: object, history: object[], actions: object, byType: object}} state
 * @returns {string}
 */
export function overviewHtml(state) {
  const t = state.totals;
  const saved = t.allowAll - t.cost;
  const stopShare = t.fraudDollars ? t.fraudStopped / t.fraudDollars : 0;
  return `<div class="ov">
    <div class="ov-head">
      <p class="t-label muted">Run overview</p>
      <p class="ov-saved"><span class="ov-saved__chip">${saved >= 0 ? usd(saved) + ' saved' : usd(-saved) + ' over'}</span></p>
    </div>
    <div class="ov-figs">
      <div><p class="ov-fig ov-fig--sen">${usd(t.cost)}</p><p class="ov-cap"><i class="ov-key ov-key--sen"></i>Cost with Sentinel</p></div>
      <div><p class="ov-fig">${usd(t.allowAll)}</p><p class="ov-cap"><i class="ov-key ov-key--all"></i>Cost if every order is allowed</p></div>
    </div>
    ${costChart(state.history)}
    <div class="ov-cards">${mixBlock(state.actions, t.n)}${patternBlock(state.byType)}</div>
    <p class="ov-brief">${plural(t.fraud, 'fraud order', 'fraud orders')} worth ${usd(t.fraudDollars)} came in; <b>${pct(stopShare)}</b> of that value was stopped, with ${plural(t.falseDeclines, 'good order', 'good orders')} declined. Pick any row to see why it was decided.</p>
  </div>`;
}
