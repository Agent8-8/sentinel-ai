"""Novelty detection for attacks the supervised model has never seen.

`TailNovelty`: for every feature we know the empirical distribution over
presumed-legit traffic. A transaction's novelty = how jointly extreme it is:
we take each feature's one-sided tail probability (in the direction that is
rarer for legit traffic), and combine the k smallest via a Fisher-style sum of
-log p. Simple, label-free, fast to score one transaction, interpretable
("this device has more distinct cards than 99.98% of legit traffic").
"""
from __future__ import annotations

import numpy as np

from .features import FEATURES, GROUPS

BEHAVIOUR = [f for g in ["velocity", "deviation", "linkage", "reputation", "merchant_pulse"] for f in GROUPS[g]]
TXN_RISKY_HIGH = ["amount_log", "ip_bill_distance_km_log", "ip_bill_country_mismatch", "ship_bill_mismatch",
                  "card_bill_country_mismatch", "foreign_ip"]
TXN_RISKY_LOW = ["account_age_days_log", "email_age_days_log"]
LOW_IS_RISKY = TXN_RISKY_LOW + ["dev_age_days_log", "acct_history_n_log", "secs_since_card_last_log"]


class TailNovelty:
    def __init__(self, features=None, k=3, max_ref=60000, seed=0):
        self.features = features or (BEHAVIOUR + TXN_RISKY_HIGH + TXN_RISKY_LOW)
        self.idx = [FEATURES.index(f) for f in self.features]
        self.low = np.array([f in LOW_IS_RISKY for f in self.features])
        self.k, self.max_ref, self.seed = k, max_ref, seed

    def fit(self, X_legit):
        rng = np.random.default_rng(self.seed)
        R = X_legit[:, self.idx]
        if len(R) > self.max_ref:
            R = R[rng.choice(len(R), self.max_ref, replace=False)]
        R = np.where(self.low, -R, R)  # orient so "high = rare/risky"
        self.sorted = np.sort(R, axis=0)
        self.n = len(R)
        return self

    def tail_p(self, X):
        Z = np.atleast_2d(X)[:, self.idx]
        Z = np.where(self.low, -Z, Z)
        # P(legit value >= z), with +1 smoothing
        ge = np.stack([self.n - np.searchsorted(self.sorted[:, j], Z[:, j], side="left")
                       for j in range(Z.shape[1])], axis=1)
        return (ge + 1) / (self.n + 1)

    def score(self, X):
        p = np.sort(self.tail_p(X), axis=1)[:, :self.k]
        return -np.log(p).sum(axis=1)

    def top_signals(self, x, top=3):
        p = self.tail_p(x)[0]
        o = np.argsort(p)[:top]
        return [(self.features[i], float(p[i])) for i in o]
