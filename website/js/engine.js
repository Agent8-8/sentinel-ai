// Browser port of the Sentinel decision logic: a small order simulator, a risk score built from
// behavioural signals, and the expected-cost policy from sentinel/policy.py in the source repo.
// Everything here is simulated; no real payment data is used.

export const DEFAULT_COSTS = Object.freeze({
  margin: 0.30, chargeback_fee: 25, churn_cost: 20, friction: 0.40, catch_rate: 0.85, abandon: 0.07,
});

export const ACTIONS = Object.freeze(['allow', 'verify', 'decline']);

const FRAUD_RATE = 0.021;
const FRAUD_MIX = [['card_testing', 0.35], ['stolen_card', 0.30], ['ato', 0.20], ['sleeper_ring', 0.10], ['friendly', 0.05]];
const CATEGORIES = ['Electronics', 'Apparel', 'Home', 'Beauty', 'Books', 'Gift cards', 'Toys', 'Sport'];
const ATO_CATCH_PENALTY = 0.25;
const SIGNAL_SCALE = 3.1;
const BASE_LOG_ODDS = -8.2;
const NOISE_RANGE = 1.6;

// Each signal: a label shown in reason codes and its weight in the log-odds score.
const SIGNALS = Object.freeze({
  deviceCards: { label: 'Device used with other cards today', weight: 1.35 },
  cardVelocity: { label: 'Card used several times in the last hour', weight: 1.1 },
  newAccount: { label: 'Account created in the last day', weight: 0.9 },
  shipMismatch: { label: 'Shipping address differs from the card', weight: 0.8 },
  foreignIp: { label: 'IP country differs from billing country', weight: 0.95 },
  linkedChargeback: { label: 'Address linked to an earlier chargeback', weight: 2.2 },
  tinyAmount: { label: 'Very small amount, typical of card testing', weight: 1.2 },
  nightHour: { label: 'Placed between 1 and 5 am local time', weight: 0.45 },
  aboveHistory: { label: 'Far above this customer’s usual spend', weight: 0.85 },
});

// Seeded PRNG (mulberry32) so a replay is the same every time it is reset.
export function makeRng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const pick = (rng, list) => list[Math.floor(rng() * list.length)];
const chance = (rng, p) => rng() < p;
const logNormal = (rng, median, spread) => {
  const z = Math.sqrt(-2 * Math.log(rng() || 1e-9)) * Math.cos(2 * Math.PI * rng());
  return median * Math.exp(spread * z);
};

function fraudType(rng) {
  let u = rng();
  for (const [name, w] of FRAUD_MIX) { if ((u -= w) < 0) return name; }
  return FRAUD_MIX[0][0];
}

// Genuine hard cases (travellers, gifts, new phones) light some signals too.
function genuineSignals(rng, amount) {
  return {
    deviceCards: chance(rng, 0.02) ? 0.5 : 0,
    cardVelocity: chance(rng, 0.03) ? 0.4 : 0,
    newAccount: chance(rng, 0.05) ? 1 : 0,
    shipMismatch: chance(rng, 0.07) ? 1 : 0,
    foreignIp: chance(rng, 0.05) ? 1 : 0,
    linkedChargeback: chance(rng, 0.003) ? 1 : 0,
    tinyAmount: amount < 10 ? 0.2 : 0,
    nightHour: chance(rng, 0.06) ? 1 : 0,
    aboveHistory: chance(rng, 0.06) ? 0.6 : 0,
  };
}

const FRAUD_SIGNALS = {
  card_testing: (rng) => ({ cardVelocity: 1, deviceCards: chance(rng, 0.7) ? 1 : 0.3, foreignIp: chance(rng, 0.5) ? 1 : 0, tinyAmount: 1 }),
  stolen_card: (rng) => ({ newAccount: 1, shipMismatch: chance(rng, 0.8) ? 1 : 0, foreignIp: chance(rng, 0.45) ? 1 : 0, deviceCards: chance(rng, 0.4) ? 1 : 0 }),
  ato: (rng) => ({ aboveHistory: 1, shipMismatch: chance(rng, 0.75) ? 1 : 0, foreignIp: chance(rng, 0.45) ? 1 : 0 }),
  sleeper_ring: (rng) => ({ deviceCards: 1, linkedChargeback: chance(rng, 0.6) ? 1 : 0 }),
  friendly: () => ({}),
};

function signalsFor(rng, type, amount) {
  const base = genuineSignals(rng, amount);
  if (!type) return base;
  const night = chance(rng, 0.3) ? 1 : base.nightHour;
  return { ...base, nightHour: type === 'friendly' ? base.nightHour : night, ...FRAUD_SIGNALS[type](rng) };
}

function amountFor(rng, type) {
  if (type === 'card_testing') return 1 + rng() * 8;
  if (type === 'stolen_card') return 150 + rng() * 750;
  if (type === 'ato') return 80 + rng() * 520;
  if (type === 'sleeper_ring') return 60 + rng() * 140;
  return Math.min(1500, Math.max(4, logNormal(rng, 58, 0.75)));
}

// Calibrated-looking probability from the signals, plus per-order noise the model cannot explain.
export function scoreSignals(signals, noise = 0) {
  let z = BASE_LOG_ODDS + noise;
  for (const [key, value] of Object.entries(signals)) z += SIGNALS[key].weight * value * SIGNAL_SCALE;
  return 1 / (1 + Math.exp(-z));
}

// Reason codes by occlusion, as in sentinel/explain.py: reset one signal to 0 and measure the drop.
export function reasonCodes(order, top = 3) {
  const { signals, noise } = order;
  const p = scoreSignals(signals, noise);
  return Object.keys(signals)
    .filter((k) => signals[k] > 0)
    .map((k) => ({ key: k, label: SIGNALS[k].label, drop: p - scoreSignals({ ...signals, [k]: 0 }, noise) }))
    .filter((r) => r.drop > 0.0005)
    .sort((a, b) => b.drop - a.drop)
    .slice(0, top);
}

export function makeOrder(rng, id, minute) {
  const isFraud = chance(rng, FRAUD_RATE);
  const type = isFraud ? fraudType(rng) : null;
  const amount = Math.round(amountFor(rng, type) * 100) / 100;
  const signals = signalsFor(rng, type, amount);
  const noise = (rng() - 0.5) * NOISE_RANGE;
  return Object.freeze({
    id, minute, amount, type, isFraud, signals, noise,
    category: type === 'card_testing' ? 'Gift cards' : pick(rng, CATEGORIES),
    risk: scoreSignals(signals, noise),
  });
}

// Expected cost of each action for calibrated risk p and amount A (sentinel/policy.py).
export function expectedCosts(p, A, c) {
  const lostSale = c.margin * A + c.churn_cost;
  return {
    allow: p * (A + c.chargeback_fee),
    verify: c.friction + p * (1 - c.catch_rate) * (A + c.chargeback_fee) + (1 - p) * c.abandon * (c.margin * A + c.churn_cost / 2),
    decline: (1 - p) * lostSale,
  };
}

export function decide(p, A, c) {
  const costs = expectedCosts(p, A, c);
  const action = ACTIONS.reduce((best, a) => (costs[a] < costs[best] ? a : best), 'allow');
  return { action, costs };
}

// What actually happens after the decision, using the simulator's truth.
export function realise(order, action, c, rng) {
  const A = order.amount;
  const catchRate = order.type === 'ato' ? c.catch_rate - ATO_CATCH_PENALTY : c.catch_rate;
  const u = rng();
  const verify = action === 'verify';
  const fraudPasses = verify && order.isFraud && u > catchRate;
  const abandons = verify && !order.isFraud && u < c.abandon;
  const fraudLoss = (action === 'allow' && order.isFraud) || fraudPasses;
  const falseDecline = action === 'decline' && !order.isFraud;
  const cost = (fraudLoss ? A + c.chargeback_fee : 0)
    + (falseDecline ? c.margin * A + c.churn_cost : 0)
    + (verify ? c.friction : 0)
    + (abandons ? c.margin * A + c.churn_cost / 2 : 0);
  let verification = null;
  if (verify) verification = order.isFraud ? (fraudPasses ? 'passed' : 'failed') : (abandons ? 'abandoned' : 'passed');
  return { cost, fraudLoss, falseDecline, verification };
}

export function allowAllCost(order, c) {
  return order.isFraud ? order.amount + c.chargeback_fee : 0;
}

export const FRAUD_LABELS = Object.freeze({
  card_testing: 'card testing', stolen_card: 'stolen card', ato: 'account takeover',
  sleeper_ring: 'sleeper ring', friendly: 'friendly fraud',
});
