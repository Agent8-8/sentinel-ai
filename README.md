# Sentinel AI: fraud screening for small online stores

**Track:** AI + Finance. This project is a prototype and is not financial advice. All payments, customers and verification outcomes are simulated.

The full write-up and credits are in [SUBMISSION.md](SUBMISSION.md).

## Who it's for

The owner of a small online store. They lose money when a fraudulent order goes through, and they also lose sales when they block a real customer. They don't have a fraud team, so Sentinel makes the call for each order.

## Input and output

- **Input:** one order (amount, account, card, device, IP, shipping address) plus what the store has seen from those before.
- **Output:** a fraud risk score, an action (allow, verify, or decline), and the signals behind it.

A gradient-boosted model (5 copies averaged, then calibrated) scores the risk from 41 features built only from earlier orders. A cost rule then picks the cheapest action, so a $9 order and a $900 order with the same risk can get different actions. An anomaly detector can send unusual orders to verification. The explanation shown for each order comes from the model itself (which signals pushed the risk up), not from a chatbot.

## What we added to the suggested card fraud project

- **Our own user:** a small online store owner with no fraud team, instead of a bank.
- **Our own data:** a store simulator (`sentinel/simulator.py`) with customers, devices, five kinds of fraud, and chargebacks that show up days late, the way they do in real life.
- **Our own question:** not just "is this fraud?" but "what should the store do?" A cost rule picks allow, verify, or decline for each order.
- **A real-data check:** the scoring model is also tested on the public ULB/Worldline card dataset.

## How the AI is built (where to find the code)

All the AI code is in the [`sentinel/`](sentinel/) folder. It uses [scikit-learn](https://scikit-learn.org/) for the models. No chatbot or outside AI service is used to make decisions. Here is the order things happen in, with a link to the exact code for each step:

| Step | What it does | Code |
|---|---|---|
| 1. Make the data | Simulates a store: customers, cards, devices, normal orders and five fraud types, with late chargebacks | [`simulator.py` → `simulate()`](sentinel/simulator.py#L82) |
| 2. Build features | Turns each order into 41 numbers (velocity, new device, cards per device, etc.), using **only orders that came before it** so the model can't peek at the future | [`features.py` → feature list](sentinel/features.py#L26), [`FeatureEngine`](sentinel/features.py#L81) |
| 3. Split by time | The final model trains on days 0–63 (using only chargebacks that had arrived by day 63), validation is days 63–74, test is days 74–91. The model never sees the test days while training | [`data.py` → `splits()`](sentinel/data.py#L37), [`evaluate.py`](sentinel/evaluate.py#L71) |
| 4. Handle late labels | Recent orders with no chargeback yet might still be fraud, so they count less in training | [`models.py` → `maturity_weights()`](sentinel/models.py#L106) |
| 5. Train the model | **The main AI:** 5 gradient-boosted tree models (`HistGradientBoostingClassifier`), each with a different random seed, averaged together | [`models.py` → `hgb()`](sentinel/models.py#L91), [`pipeline.py` → `SentinelModel.fit()`](sentinel/pipeline.py#L28) |
| 6. Calibrate | Adjusts the score so "0.30" really means about a 30% chance of fraud (isotonic regression, fit on validation data) | [`models.py` → `Calibrator`](sentinel/models.py#L130), [`pipeline.py` → `calibrate()`](sentinel/pipeline.py#L53) |
| 7. Anomaly check | An `IsolationForest` trained only on normal orders flags very unusual ones for verification (never for decline) | [`pipeline.py`](sentinel/pipeline.py#L28) |
| 8. Pick an action | For each order, works out the expected cost of allow, verify, and decline, and picks the cheapest | [`policy.py` → `expected_costs()` and `decide()`](sentinel/policy.py#L43) |
| 9. Explain | Shows which signals raised the risk the most by setting each one back to normal and seeing how much the score drops | [`explain.py` → `reason_codes()`](sentinel/explain.py#L38) |
| 10. Test | Trains on 3 separate simulated stores and scores the untouched test days | [`evaluate.py` → `run_world()`](sentinel/evaluate.py#L63) |

The baselines it's compared against are in [`models.py`](sentinel/models.py): hand-made store rules ([`rules_score()`](sentinel/models.py#L69)) and logistic regression ([`logistic()`](sentinel/models.py#L85)). The model settings were picked on validation data only, in [`experiments.py`](sentinel/experiments.py).

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
# open http://localhost:8000
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
sentinel/explain.py         reason codes (which signals raised the risk)
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
