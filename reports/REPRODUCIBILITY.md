# Reproduction and evaluation notes

## Public-data benchmark

On November 14, 2026, the benchmark was rerun on the original ULB/Worldline credit-card CSV downloaded from Kaggle. Its SHA-256 was `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`. It had 284,807 rows and 31 columns, including `Time`; the OpenML 1597 version retrieved during this check did not include `Time` and could not reproduce the temporal split. The CSV is not included in this repository.

The split produced 170,882 training transactions (360 fraud), 42,723 validation transactions (38 fraud), and 71,202 test transactions (94 fraud). The rerun reproduced the saved test average precision to the displayed precision: 0.791 for the five-model boosted ensemble and 0.754 for logistic regression. There are only 94 positive examples in that test period. The intervals in the original report are bootstrap intervals on individual transactions, and do not measure changes across merchants or future time periods.

The rerun used Python 3.14.5, NumPy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1 and SciPy 1.18.1. Run `python -m sentinel.benchmark_public` after placing the original `creditcard.csv` in `data/public/`. The script overwrites `public_benchmark.json`.

## Operating points and data limits

In the original detection report, the threshold used for the 0.5% genuine-flagged comparisons is calculated from genuine test scores. That is an after-the-fact ranking comparison, not a rule available before test data arrives. `check_submission_metrics.py` and `sentinel/audit_groups.py` instead choose a threshold on validation and apply it to the later test period. The latter is an exploratory, score-only behavioral-scenario audit; the original test set has already been inspected.

Simulation calibration uses complete validation ground truth, including unreported fraud, which a real store would not know. The original three simulated test periods are consumed. New simulation seeds can check repeatability within this simulator, not external validity. Never infer real cost savings or demographic fairness from these synthetic results.

The corrected sequential replay processed all 21,596 seed-7 simulated test orders; see `corrected_replay_seed7.json` and run `python -m sentinel.replay_corrected`. It retains assumed costs, simulated verification, and complete validation labels. It does not recompute the previous three-store average and uses a different replay random seed.
