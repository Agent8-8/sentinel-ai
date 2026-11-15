import { test } from 'node:test';
import assert from 'node:assert/strict';
import { expectedCosts, decide, makeRng, makeOrder, reasonCodes, DEFAULT_COSTS } from '../../js/engine.js';
import { evaluate } from '../../js/whatif.js';

test('expected costs match the policy formula from sentinel/policy.py', () => {
  // Arrange
  const c = DEFAULT_COSTS;
  // Act
  const costs = expectedCosts(0.34, 412, c);
  // Assert
  assert.equal(costs.allow.toFixed(2), (0.34 * (412 + 25)).toFixed(2));
  assert.equal(costs.decline.toFixed(2), (0.66 * (0.3 * 412 + 20)).toFixed(2));
  assert.ok(costs.verify < costs.allow && costs.verify < costs.decline);
});

test('decide allows a low-risk order and declines a near-certain fraud', () => {
  assert.equal(decide(0.001, 60, DEFAULT_COSTS).action, 'allow');
  assert.equal(decide(0.99, 600, DEFAULT_COSTS).action, 'decline');
});

test('action severity never decreases as risk rises for a fixed amount', () => {
  const rank = { allow: 0, verify: 1, decline: 2 };
  let last = 0;
  for (let p = 0; p <= 1; p += 0.01) {
    const r = rank[decide(p, 250, DEFAULT_COSTS).action];
    assert.ok(r >= last, `severity dropped at p=${p.toFixed(2)}`);
    last = r;
  }
});

test('seeded orders are reproducible', () => {
  const a = makeOrder(makeRng(7), 1, 0);
  const b = makeOrder(makeRng(7), 1, 0);
  assert.deepEqual(a, b);
});

test('reason codes are ordered by their effect on risk', () => {
  const rng = makeRng(3);
  const fraud = Array.from({ length: 4000 }, (_, i) => makeOrder(rng, i, i)).find((o) => o.type === 'stolen_card');
  const reasons = reasonCodes(fraud);
  assert.ok(reasons.length > 0);
  for (let i = 1; i < reasons.length; i++) assert.ok(reasons[i - 1].drop >= reasons[i].drop);
});

test('the policy beats allowing every order on the simulated sample', () => {
  const rng = makeRng(91);
  const orders = Array.from({ length: 5000 }, (_, i) => makeOrder(rng, i, i));
  const r = evaluate(orders, DEFAULT_COSTS);
  assert.ok(r.cost < r.allowAll * 0.5);
  assert.ok(r.fd / r.genuine < 0.005, 'false declines stay under 0.5%');
});
