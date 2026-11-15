"""Models, baselines and metrics."""
from __future__ import annotations

import numpy as np
from scipy.stats import gamma
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import CATEGORICAL, FEATURES, GROUPS
from .simulator import DAY

FIDX = {f: i for i, f in enumerate(FEATURES)}


def cols(groups):
    return [FIDX[f] for g in groups for f in GROUPS[g]]


# ---------------------------------------------------------------- metrics
def recall_at_fpr(y, s, fpr=0.005, weight=None):
    """Fraud caught (count or $-weighted) when only `fpr` of legit traffic is flagged."""
    thr = np.quantile(s[y == 0], 1 - fpr)
    hit = s > thr
    w = np.ones_like(s) if weight is None else weight
    return float((w * hit * y).sum() / max((w * y).sum(), 1e-9))


def metrics(y, s, amount):
    return {
        "auprc": float(average_precision_score(y, s)),
        "auroc": float(roc_auc_score(y, s)),
        "recall@0.5%fpr": recall_at_fpr(y, s, 0.005),
        "recall@1%fpr": recall_at_fpr(y, s, 0.01),
        "dollar_recall@0.5%fpr": recall_at_fpr(y, s, 0.005, amount),
    }


def bootstrap_ci(y, s, amount, fn, n=300, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() < 5:
            continue
        vals.append(fn(y[i], s[i], amount[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------------------------------------------------------- baselines
def rules_score(X):
    """What a small merchant typically configures in a gateway rule engine."""
    g = lambda f: X[:, FIDX[f]]
    amt = np.expm1(g("amount_log"))
    score = (
        (amt > 500) * 1.0
        + g("ip_bill_country_mismatch") * 1.0
        + g("ship_bill_mismatch") * 0.5
        + (g("card_tx_24h") >= 3) * 1.0
        + (g("ip_cards_1h") >= 3) * 1.0
        + (g("email_age_days_log") < np.log1p(7)) * 0.5
        + (X[:, FIDX["category"]] == 5) * 0.5  # gift cards
    )
    return score + amt / 1e5  # tie-break


def logistic(X, y, w=None):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"))
    m.fit(X, y, logisticregression__sample_weight=w)
    return m


def hgb(X, y, w=None, cat_mask=None, seed=0, **kw):
    params = dict(learning_rate=0.05, max_iter=500, max_leaf_nodes=31, min_samples_leaf=40,
                  l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                  n_iter_no_change=40, random_state=seed)
    params.update(kw)
    m = HistGradientBoostingClassifier(categorical_features=cat_mask, **params)
    m.fit(X, y, sample_weight=w)
    return m


def cat_mask_for(col_idx):
    return [FEATURES[i] in CATEGORICAL for i in col_idx] if col_idx else None


# ------------------------------------------------- label-maturity weighting
def maturity_weights(ts, y_obs, label_cutoff, a=2.0, scale=9.0, base_delay=5.0):
    """A recent transaction with no chargeback yet is only weakly evidence of
    being legit: its fraud label may simply not have arrived. Weight each
    unlabeled sample by P(chargeback would have arrived by cutoff | fraud),
    using the chargeback-delay distribution estimated from history."""
    age_days = (label_cutoff - ts) / DAY
    p_mature = gamma.cdf(np.clip(age_days - base_delay, 0, None), a, scale=scale)
    return np.where(y_obs == 1, 1.0, np.clip(p_mature, 0.05, 1.0))


def fit_delay_distribution(delays_days):
    """Estimate chargeback-delay distribution from observed (matured) chargebacks."""
    d = np.asarray(delays_days)
    base = max(0.0, d.min() - 0.1)
    a, _, scale = gamma.fit(d - base, floc=0)
    return a, scale, base


# ------------------------------------------------------------ unsupervised
def isolation_scores(X_fit, X_apply, seed=0):
    iso = IsolationForest(n_estimators=300, max_samples=4096, random_state=seed).fit(X_fit)
    return -iso.score_samples(X_apply)


class Calibrator:
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=0.999)

    def fit(self, s, y):
        self.iso.fit(s, y)
        return self

    def __call__(self, s):
        return self.iso.predict(s)
