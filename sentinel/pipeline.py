"""Production model.

Two paths, scored one transaction at a time:
  1. Supervised risk: bagged GBM -> isotonic calibration -> P(fraud). Drives the
     expected-cost allow/hold/decline decision.
  2. Novelty gate: IsolationForest fitted on presumed-legit traffic only. If an
     order is more anomalous than (1 - budget) of genuine validation traffic, an
     ALLOW is upgraded to HOLD (step-up verification) - never to DECLINE. This
     is the safety net for attack patterns that have no labels yet.
"""
from __future__ import annotations

import pickle

import numpy as np
from sklearn.ensemble import IsolationForest

from . import models as M
from .features import FEATURES


class SentinelModel:
    def __init__(self, params: dict, n_bag=5, iso_weight=0.0, novelty_budget=0.005):
        self.params, self.n_bag, self.iso_weight = params, n_bag, iso_weight
        self.novelty_budget = novelty_budget
        self.cat_mask = M.cat_mask_for(list(range(len(FEATURES))))

    def fit(self, X, y, w=None):
        self.bag = [M.hgb(X, y, w=w, cat_mask=self.cat_mask, seed=k, **self.params) for k in range(self.n_bag)]
        if self.iso_weight > 0 or self.novelty_budget > 0:
            self.iso = IsolationForest(n_estimators=300, max_samples=4096, random_state=0).fit(X[y == 0])
        return self

    def novelty(self, X):
        return -self.iso.score_samples(np.atleast_2d(X))

    def is_novel(self, X):
        if self.novelty_budget <= 0:
            return np.zeros(len(np.atleast_2d(X)), bool)
        return self.novelty(X) > self.novelty_thr

    def _gbm(self, X):
        return np.mean([m.predict_proba(X)[:, 1] for m in self.bag], axis=0)

    def _raw(self, X):
        g = self._gbm(X)
        if self.iso_weight <= 0:
            return g
        qg = np.searchsorted(self.ref_g, g) / len(self.ref_g)
        qi = np.searchsorted(self.ref_i, -self.iso.score_samples(X)) / len(self.ref_i)
        return (1 - self.iso_weight) * qg + self.iso_weight * qi

    def calibrate(self, Xv, yv):
        if self.iso_weight > 0:
            self.ref_g = np.sort(self._gbm(Xv))
            self.ref_i = np.sort(-self.iso.score_samples(Xv))
        self.cal = M.Calibrator().fit(self._raw(Xv), yv)
        if self.novelty_budget > 0:  # budget = share of genuine traffic the gate may send to verification
            self.novelty_thr = float(np.quantile(self.novelty(Xv[yv == 0]), 1 - self.novelty_budget))
        return self

    def predict(self, X):
        """Calibrated P(fraud)."""
        return self.cal(self._raw(np.atleast_2d(X)))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
