// What-if panel: re-optimises the expected-cost policy over a fixed set of simulated orders
// for the merchant's own cost assumptions. Outcomes are expected values, so they do not jitter.
import { makeRng, makeOrder, decide, DEFAULT_COSTS } from './engine.js';

const SAMPLE_ORDER = Object.freeze({ amount: 412, risk: 0.34 });

const SAMPLE_SIZE = 20000;
const SEED = 91;
const ATO_CATCH_PENALTY = 0.25;

const FORMAT = {
  margin: (v) => Math.round(v * 100) + '%',
  chargeback_fee: (v) => '$' + Math.round(v),
  churn_cost: (v) => '$' + Math.round(v),
  friction: (v) => '$' + v.toFixed(2),
  catch_rate: (v) => Math.round(v * 100) + '%',
  abandon: (v) => Math.round(v * 100) + '%',
};

function buildSample() {
  const rng = makeRng(SEED);
  return Array.from({ length: SAMPLE_SIZE }, (_, i) => makeOrder(rng, i, i));
}

// Expected realised outcome of the policy over the sample, given the simulator's truth.
export function evaluate(orders, c) {
  const acc = { cost: 0, allowAll: 0, fraud$: 0, stopped$: 0, genuine: 0, fd: 0, fr: 0, allow: 0, verify: 0, decline: 0 };
  for (const o of orders) {
    const { action } = decide(o.risk, o.amount, c);
    acc[action] += 1;
    const lossIfFraud = o.amount + c.chargeback_fee;
    if (o.isFraud) {
      const catchRate = o.type === 'ato' ? c.catch_rate - ATO_CATCH_PENALTY : c.catch_rate;
      const slipShare = action === 'allow' ? 1 : action === 'verify' ? 1 - catchRate : 0;
      acc.fraud$ += o.amount;
      acc.stopped$ += o.amount * (1 - slipShare);
      acc.allowAll += lossIfFraud;
      acc.cost += slipShare * lossIfFraud + (action === 'verify' ? c.friction : 0);
    } else {
      acc.genuine += 1;
      if (action === 'decline') { acc.fd += 1; acc.cost += c.margin * o.amount + c.churn_cost; }
      if (action === 'verify') { acc.fr += 1; acc.cost += c.friction + c.abandon * (c.margin * o.amount + c.churn_cost / 2); }
    }
  }
  return acc;
}

function readCosts(form) {
  return Object.fromEntries(Object.keys(DEFAULT_COSTS).map((k) => [k, Number(form.elements[k].value)]));
}

function renderSample(box, c) {
  if (!box) return;
  const { action, costs } = decide(SAMPLE_ORDER.risk, SAMPLE_ORDER.amount, c);
  const max = Math.max(...Object.values(costs), 0.01);
  box.innerHTML = ['allow', 'verify', 'decline'].map((a) => `<div class="mini-cost__row${a === action ? ' is-pick' : ''}"><span>${a[0].toUpperCase() + a.slice(1)}</span><i style="--w:${Math.max(1, (costs[a] / max) * 100).toFixed(1)}%"></i><b>$${costs[a].toFixed(2)}</b></div>`).join('');
}

export function mountWhatIf(section) {
  const form = section?.querySelector('[data-whatif]');
  if (!form) return;
  const out = (k) => section.querySelector(`[data-w="${k}"]`);
  const orders = buildSample();
  const pct = (v, d = 0) => (v * 100).toFixed(d) + '%';

  const update = () => {
    const c = readCosts(form);
    Object.entries(c).forEach(([k, v]) => { section.querySelector(`[data-o="${k}"]`).textContent = FORMAT[k](v); });
    const r = evaluate(orders, c);
    const n = orders.length;
    out('save').textContent = '$' + Math.round(r.allowAll - r.cost).toLocaleString('en-US');
    ['allow', 'verify', 'decline'].forEach((a, i) => {
      const share = r[a] / n;
      out(['a', 'v', 'd'][i]).style.setProperty('--s', share);
      out(['ap', 'vp', 'dp'][i]).textContent = pct(share, share < 0.1 ? 1 : 0);
    });
    out('splitlabel').setAttribute('aria-label', `Allowed ${pct(r.allow / n, 1)}, verified ${pct(r.verify / n, 1)}, declined ${pct(r.decline / n, 1)}`);
    out('stop').textContent = pct(r.fraud$ ? r.stopped$ / r.fraud$ : 0);
    out('fd').textContent = pct(r.genuine ? r.fd / r.genuine : 0, 2);
    out('fr').textContent = pct(r.genuine ? r.fr / r.genuine : 0, 2);
    renderSample(section.querySelector('[data-sample]'), c);
  };

  form.addEventListener('input', update);
  update();
  form.dataset.ready = 'true';
}
