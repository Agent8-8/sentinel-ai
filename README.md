# Sentinel AI: fraud screening for small online stores

**Track:** AI + Finance. This project is a prototype and is not financial advice. All payments, customers and verification outcomes are simulated.

The full write-up and credits are in [SUBMISSION.md](SUBMISSION.md).

## Who it's for

The owner of a small online store. They lose money when a fraudulent order goes through, and they also lose sales when they block a real customer. They don't have a fraud team, so Sentinel makes the call for each order.

## Input and output

- **Input:** one order (amount, account, card, device, IP, shipping address) plus what the store has seen from those before.
- **Output:** a fraud risk score, an action (allow, verify, or decline), and the signals behind it.

A gradient-boosted model (5 copies averaged, then calibrated) scores the risk from 41 features built only from earlier orders. A cost rule then picks the cheapest action, so a $9 order and a $900 order with the same risk can get different actions. An anomaly detector can send unusual orders to verification. Gemini is optional and only rewrites the explanation in plain English. It never changes a decision.

## Results vs. baselines (later test period the model never trained on)

| Check | Baseline | Sentinel |
|---|---|---|
| Simulated cost, one store, 21,596 test orders | $77,219 (approve everything) | **$14,776** |
| Fraud caught when 0.5% of real orders are flagged, 3 simulated stores | 41% (typical store rules) | **91%** |
| Average precision on real public card data (ULB/Worldline, 284,807 transactions) | 0.754 (logistic regression) | **0.791** |

With a cutoff picked on validation data, the test period had 93.8% precision and 88.7% recall (377 frauds caught, 25 real orders flagged, 48 frauds missed).

## Where it fails

- **Missed fraud:** 48 frauds got through in that test. One was "friendly fraud" (the real cardholder buys, then disputes it), which looks exactly like a normal order by design.
- **Guest shoppers get flagged more:** 4 of 217 real guest orders were flagged vs. 21 of 20,954 non-guest orders. IP country mismatch shows the same issue (10 of 405). See [reports/scenario_audit.json](reports/scenario_audit.json).
- **Most results are on simulated data.** The public dataset has no account or device IDs, so it only tests the scoring model, not the behavior features or the cost policy. The modest gain there has overlapping confidence intervals.
- **Other caveats:** the 91% row sets its cutoff using test labels (a ranking comparison, not a real cutoff). Calibration uses simulator labels a real store wouldn't have. Costs (30% margin, $25 chargeback fee, etc.) are assumptions. The test periods have been looked at more than once, so later numbers are after the fact.

## Run it

```bash
./run.sh                 # installs deps, runs tests, builds the model, starts the dashboard
# open http://localhost:8000   (optional: export GEMINI_API_KEY=... first)
```

Rebuild every result (about 20 min):

```bash
python -m sentinel.experiments && python -m sentinel.experiments_novelty
python -m sentinel.evaluate && python -m sentinel.replay_corrected
python -m sentinel.check_metrics && python -m sentinel.audit_groups
python -m sentinel.benchmark_public   # needs Kaggle creditcard.csv in data/public/
```

## Files

```
sentinel/simulator.py       simulated store: customers, cards, devices, fraud types, late chargebacks
sentinel/features.py        features built only from earlier orders (no peeking at the future)
sentinel/data.py            train / validation / test split by time
sentinel/models.py          baselines (rules, logistic regression) and the boosted model
sentinel/pipeline.py        final model + anomaly detector
sentinel/policy.py          allow / verify / decline by expected cost
sentinel/explain.py         reason codes + optional Gemini wording
sentinel/novelty.py         rarity scores used in explanations
sentinel/experiments*.py    model choices, made on validation data only
sentinel/evaluate.py        final test evaluation -> reports/results.json
sentinel/replay_corrected.py  order-by-order replay -> reports/corrected_replay_seed7.json
sentinel/check_metrics.py   precision, recall and a missed-fraud example
sentinel/audit_groups.py    false flags by customer situation -> reports/scenario_audit.json
sentinel/benchmark_public.py  real-data check -> reports/public_benchmark.json
sentinel/api.py, dashboard/ live demo dashboard
tests/                      checks for data leakage and policy logic
```
