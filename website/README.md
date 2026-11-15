# Sentinel AI website

Live: https://sentinel-ai-site-gamma.vercel.app

A static showcase site for this project. Plain HTML, CSS and JavaScript, no build step. Every figure on it comes
from this repository's `reports/` (`results.json`, `corrected_replay_seed7.json`, `public_benchmark.json`). All
payments are simulated.

## What is on it
- **Model animation (hero):** simulated orders flow through 41 behavioural signals, the bag of 5 gradient-boosted
  models, calibration and the expected-cost policy, with the Isolation Forest novelty gate. It shows every flagged
  order and 1 in 30 allowed ones; its counters count every order screened.
- **Cost breakdown:** the seed-7 replay's $77,219 (allow every order) and $14,776 (Sentinel) split into fraud
  losses, chargeback fees, verification codes and lost sales, rebuilt with the cost formula in `sentinel/policy.py`.
- **Live simulator:** a browser replay with running totals, a run overview (cumulative cost chart, decision mix,
  fraud stopped by pattern) and a case view for any order (risk, reason codes, the three action costs).
- **What-if:** six cost sliders that re-optimise the policy over 20,000 simulated orders.
- **Model evidence** and **known limits**, from the reports and the main README.

The browser simulator (`js/engine.js`) is a small stand-in calibrated to land near the repository's results; its
policy maths is a direct port of `sentinel/policy.py`. It does not run the trained Python model.

## Run and test
```bash
cd website
npm install
npx playwright install chromium
npm run serve      # http://127.0.0.1:4173
npm test           # unit tests for the decision engine
npm run test:e2e   # Playwright, desktop and phone
```

## Deploy
From this folder: `npx vercel deploy --prod` (files in `.vercelignore` are left out).

## Layout
- `index.html`, `site.css`, `hero.css`: the page
- `js/engine.js`: order simulator, scoring, reason codes, expected-cost policy
- `js/hero-viz.js`, `js/replay.js`, `js/overview.js`, `js/whatif.js`, `js/main.js`: the interactive parts
- `assets/`: the parts of the design kit the page uses (system CSS, motion, diagram, moments, effect)
- `tests/`: unit tests, Playwright tests and a small static server
