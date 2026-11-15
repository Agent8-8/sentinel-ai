# Sentinel AI: autonomous fraud prevention for small online stores

Sentinel scores every card-not-present order in real time and **decides on its own** whether to
**allow** it, **hold it for automated customer verification** (3DS/OTP-style step-up), or **decline** it.
The goal is to stop fraud without turning away good customers and without a merchant reviewing orders by hand.

> Hackathon prototype. All payments are simulated. Detection is trained and tested on a behavioural
> simulator (below) and the core ML is independently checked on a real public dataset.

```bash
./run.sh                      # installs deps, runs tests, (re)builds reports if missing, starts the dashboard
# open http://localhost:8000  # optional: export GEMINI_API_KEY=... first for AI-written explanations
```

---

## Headline results (held-out test window, mean of 3 independent simulated stores)

Test = simulated days 74–91. Nothing in this window was used for training, tuning, calibration or threshold
selection. It contains a holiday sales spike (legit distribution shift) and a fraud pattern that never occurs
in training ("sleeper ring").

**Detection** (training used only chargebacks that had *arrived* by the label cutoff):

| Model | AUPRC | 95% CI† | Fraud caught @ 0.5% of genuine orders flagged | Fraud $ caught |
|---|---|---|---|---|
| Rules engine (typical gateway rules) | 0.38 | 0.41–0.51* | 41% | 25% |
| Logistic regression, transaction fields | 0.60 | 0.56–0.65 | 54% | 31% |
| Gradient boosting, transaction fields | 0.59 | 0.52–0.62 | 59% | 39% |
| **Sentinel: behavioural GBM bag** | **0.89** | **0.84–0.90** | **91%** | **91%** |
| **Sentinel + novelty gate** (+0.5% genuine verified) | – | – | **92%** | **92%** |

Calibration on test: Brier 0.0052, ECE 0.004 (predicted mean 1.7% vs actual 2.1%).
Unseen "sleeper ring" pattern: 93% caught (88–96% across worlds); transaction-only models: 0%.

**Business outcome of the autonomous decision policy** (realised costs incl. chargeback fees, lost margin on
false declines, customer churn, verification friction and abandonment):

| Policy | Total cost | Fraud $ stopped | False declines | Genuine asked to verify | Merchant reviews |
|---|---|---|---|---|---|
| Approve everything | $82,037 | 0% | 0% | 0% | 0 |
| Rules → manual review queue | $67,959 | 17% | 0% | 0.20% | **162** |
| Rules → auto-decline | $73,399 | 20% | 0.20% | 0% | 0 |
| Sentinel score, tuned binary threshold | $21,986 | 76% | 0.15% | 0% | 0 |
| Sentinel score, tuned 3-way thresholds | $23,852 | 72% | 0.07% | 0.08% | 0 |
| **Sentinel expected-cost policy** | **$15,007** | **82%** | **0.03%** | 0.20% | 0 |
| **Sentinel full** (+ novelty gate + trust memory) | **$15,097** | **83%** | **0.03%** | 0.84% | 0 |

→ **−82% fraud-related cost vs. doing nothing, 1 in 3,000 genuine customers declined, zero manual reviews.**

**Real public data** (ULB/Worldline credit-card dataset, 284,807 real transactions, strict temporal split):
Sentinel GBM bag test AUPRC **0.79** [0.71–0.86] vs logistic regression 0.75 [0.67–0.85]. A modest gain with
overlapping CIs. That dataset has no identities, so it only validates the modelling stack, not behaviour.

† percentile bootstrap, averaged over worlds. *Rule scores are heavily tied, which biases bootstrap AP upward,
so its CI does not contain the point estimate. It doesn't change any comparison.

---

## How it works

```
order ─► streaming feature engine ─► calibrated GBM bag ─► P(fraud) ─┐
          (41 causal features:         (5 seeds, isotonic)           ├─► expected-cost policy ─► allow / verify / decline
           behaviour, linkage,      ─► IsolationForest novelty gate ─┘        ▲                       │
           reputation, bot pulse)                                    trust memory ◄── verification ◄──┘
                       ▲                                                                              │
                       └──────────── failed verification / chargebacks mark device, card, address bad ┘
```

### 1. Data: a behavioural merchant simulator (`sentinel/simulator.py`)
Public card datasets (ULB, Sparkov, IEEE-CIS) either lack entity identities or are hard to get, and none let me
test *unseen* attacks or label delay. So I simulate a 91-day US online store (~85k orders, ~16k customers)
with explicit accounts, cards, devices, IPs and shipping addresses, including the hard parts:

* **Hard genuine cases:** travellers, VPN users, new phones, gifts shipped elsewhere, family-shared devices,
  office-shared IPs, brand-new guest shoppers, occasional splurges, holiday rush in the test window.
* **Adaptive fraud:** card-testing bots (incl. "low-and-slow" campaigns with US proxies and normal basket sizes),
  account takeover (55% via residential proxies, 25% ship to the victim's own address), stolen cards on fresh
  accounts with purchased aged emails, friendly fraud (indistinguishable by design), and a **test-only "sleeper
  ring"** (aged accounts, local IPs, normal amounts; visible only through shared devices/drop addresses).
* **Realistic labels:** chargebacks arrive 5–60 days late; ~45% of card-testing and ~7% of other fraud is never reported.

### 2. Causal streaming features (`sentinel/features.py`)
One `FeatureEngine` class, used identically offline and live (no train/serve skew). For each order it computes
features from state built only from *earlier* orders, then updates state. Groups: transaction, velocity,
deviation-from-own-history, cross-entity linkage (device→cards/accounts, address→cards), reputation
(linked to a chargeback *that has already arrived*), and merchant-wide bot pulse.

### 3. Learning from delayed, incomplete labels (`sentinel/models.py`)
At training time only 47% of training-window fraud had been charged back. Recent "no chargeback yet" orders are
weak evidence of being genuine, so each one is weighted by P(its chargeback would have arrived by now | fraud),
using a delay distribution fitted from older chargebacks. That makes it safe to also train on the last week
before the cutoff.

### 4. Novelty gate for attacks with no labels yet (`sentinel/pipeline.py`)
An IsolationForest fitted only on presumed-genuine traffic. If an order is more anomalous than 99.5% of genuine
validation traffic, an *allow* becomes *verify*. It never declines, so a false alarm costs a few seconds of a
customer's time, not a sale.

### 5. Autonomous expected-cost decisions (`sentinel/policy.py`)
For each order with calibrated risk *p* and amount *A*, Sentinel computes the expected cost of allowing (fraud
loss + fee), declining (lost margin + churn) and verifying (friction + abandonment + fraud that passes
verification) and picks the cheapest. A $9 order and a $900 order with the same risk get different treatment.
**Trust memory:** an (account, device) pair that passed verification isn't challenged again for mid-risk
orders. **Instant feedback:** a failed verification marks the device/card/address bad immediately, instead of
waiting weeks for a chargeback. The only escalation to a human is "passed verification yet risk ≥ 90%" (0 in test).

### 6. Explanations (`sentinel/explain.py`)
Reason codes are computed by **occlusion on the real model**: reset each signal group to typical-genuine values
and measure the drop in P(fraud). For novelty holds, Sentinel reports which features are rarer than X% of
genuine orders. **Gemini (optional)** only rewrites these computed facts into plain English for the merchant
(per case, and as a daily briefing). It never makes or changes a decision, and everything works without a key
(template narratives).

---

## Evidence for each decision (what was tried, what was kept)

All selection used the validation window (days 63–74), averaged over 3 worlds. Full logs: `reports/*.json`.

| Experiment | Result | Decision |
|---|---|---|
| First simulator version | Transaction-only logistic already AUPRC 0.93, so it couldn't separate approaches | **Rebuilt simulator harder** (adaptive fraud + legit hard cases) |
| Behavioural vs transaction-only features | 0.90 vs 0.81 AUPRC; ATO recall 0.99 vs 0.65 | Kept |
| Drop-one-group ablation | Each group alone costs ≤0.011 AUPRC (redundant signals), linkage largest | Kept all (cheap, robust) |
| Maturity-weighted labels (+ gap week) | 0.898 → 0.902–0.906; $-recall 0.87 → 0.90; oracle-label ceiling 0.915 | Kept |
| Hyperparameter grid (6 configs) | Small trees (7 leaves) best/equal; lr 0.1 clearly worse | 7 leaves, lr 0.05 |
| 5-seed bagging | 0.906 → 0.910 | Kept |
| IsolationForest *rank blend* into the score | Hurt slightly (0.903–0.907) | **Rejected** |
| **Leave-one-pattern-out**: supervised only | Unseen card testing 5%, ATO 1%, stolen card 54% recall | Supervised alone is blind to new attacks |
| IsolationForest as a **verification gate** (+0.5% genuine) | Unseen card testing 5→80%, ATO 1→61%, stolen 54→73% | **Kept** |
| My own tail-probability novelty detector | ≤23% on unseen patterns | **Rejected** (reused only as an explainer; its feature orientation was corrected afterwards and the detector was not re-evaluated) |
| Expected-cost vs tuned fixed thresholds | $15.0k vs $22.0k (binary) / $23.9k (3-way) | Kept: amount-awareness matters |
| Single-row latency | 138 ms/tx (thread fan-out) → ~14 ms/tx single-threaded | `OMP_NUM_THREADS=1` in API |

## Leakage controls (tested in `tests/test_sentinel.py`)
* **No look-ahead:** rewriting all future transactions leaves every past feature bit-identical (tested).
* **Reputation** features only see chargebacks whose arrival time precedes the order (tested).
* **Temporal splits:** train < day 56 | gap | valid 63–74 | test ≥ 74; training labels censored at day 63 (tested).
* Novel pattern exists only in test (tested). Test window scored once, after the config was frozen
  (`evaluate.py` reads choices from `reports/experiments.json`).
* Policy sanity: action severity monotone in risk; expected cost equals realised cost on average (tested).

## Honest limitations
* **Synthetic data.** The fraud patterns are my model of reality. Absolute numbers will not transfer; the
  *relative* evidence (behaviour > transaction-only, gate catches unseen attacks, cost policy > thresholds) is
  the claim. The real-data check covers only the modelling stack and shows a modest, CI-overlapping gain.
* **The "unseen" sleeper ring shares fingerprints** (shared devices, drop addresses) with training-era crews,
  so 93% recall there is generalisation to a new *combination*, not to a fundamentally new signal. The
  leave-one-pattern-out results (61–80% with the gate) are the more sober estimate for genuine zero-days.
* **The novelty gate didn't pay for itself on this test set** (+$90 cost, +0.6% genuine verified) because the
  supervised model already covered every test pattern. It's insurance against attacks with no labels yet.
* **Friendly fraud (~5% of fraud) is undetectable** here by construction and caps recall.
* **Cost parameters are assumptions** (margin 30%, $25 chargeback fee, $20 churn, 85% verification catch rate,
  7% abandonment). The *What-if* tab re-optimises the policy for any merchant's numbers.
* Verification is simulated. Real 3DS/OTP pass rates, especially for ATO with SIM-swap, vary widely.
* Single-merchant scope: no consortium data, device-fingerprinting vendor, or BIN intelligence, which real
  systems use heavily.

## Project layout
```
sentinel/simulator.py     behavioural merchant world (accounts, cards, devices, IPs, addresses, label delays)
sentinel/features.py      causal streaming feature engine (shared by training and live API)
sentinel/data.py          dataset build + temporal splits + label censoring
sentinel/models.py        GBM / logistic / rules baselines, maturity weighting, metrics, bootstrap
sentinel/pipeline.py      production model: calibrated GBM bag + novelty gate
sentinel/policy.py        expected-cost allow/verify/decline, trust memory, outcome simulation
sentinel/explain.py       occlusion reason codes + optional Gemini narration
sentinel/novelty.py       tail-probability scorer (rejected as detector, used as explainer)
sentinel/experiments*.py  validation-only model development (multi-world)
sentinel/evaluate.py      final one-shot held-out evaluation
sentinel/benchmark_public.py  real-data check on ULB credit-card dataset
sentinel/api.py           FastAPI live replay server
dashboard/                live ops, model evidence, what-if (Chart.js)
tests/                    leakage, causality and policy tests
```

Reproduce from scratch: `python -m sentinel.experiments && python -m sentinel.experiments_novelty &&
python -m sentinel.evaluate && python -m sentinel.benchmark_public` (~20 min on a laptop; the ULB step needs
`data/public/creditcard.pq` from OpenML dataset 1597).
