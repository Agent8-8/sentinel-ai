"""Experiment: catching fraud patterns absent from training (validation only).

For each pattern P in {card_testing, ato, stolen_card}:
  * train the supervised GBM with every P example removed
  * fit label-free novelty detectors on presumed-legit training traffic
  * on validation (legit + P only), measure recall when each path may flag a
    fixed 0.5% of genuine traffic, and for the union "GBM OR novelty gate"
    (total genuine friction budget ~1%).

    python -m sentinel.experiments_novelty
"""
from __future__ import annotations

import json

import numpy as np

from . import models as M
from .data import ROOT
from .experiments import SEEDS, World
from .features import FEATURES, GROUPS
from .novelty import BEHAVIOUR, TailNovelty

PATTERNS = ["card_testing", "ato", "stolen_card"]


def at_budget(s, y, budget):
    thr = np.quantile(s[y == 0], 1 - budget)
    return s > thr


def main():
    params = json.loads((ROOT / "reports" / "experiments.json").read_text())["best_params"]
    worlds = [World(s) for s in SEEDS]
    rows = {}
    bidx = [FEATURES.index(f) for f in BEHAVIOUR]
    for P in PATTERNS:
        acc = {}
        for w in worlds:
            fit = w.ext & ~((w.ftype == P) & (w.y_obs == 1))
            Xf, yf, wf = w.X[fit], w.y_obs[fit], w.w_mat[fit]
            gbm = M.hgb(Xf, yf, w=wf, cat_mask=M.cat_mask_for(list(range(len(FEATURES)))), **params)
            keep = w.va & ((w.y_true == 0) | (w.ftype == P))
            Xv, yv = w.X[keep], w.y_true[keep]
            legit_ref = Xf[yf == 0]
            scores = {
                "GBM (pattern unseen)": gbm.predict_proba(Xv)[:, 1],
                "IsolationForest all feats": M.isolation_scores(legit_ref, Xv),
                "IsolationForest behaviour": M.isolation_scores(legit_ref[:, bidx], Xv[:, bidx]),
                "TailNovelty k=1": TailNovelty(k=1).fit(legit_ref).score(Xv),
                "TailNovelty k=3": TailNovelty(k=3).fit(legit_ref).score(Xv),
                "TailNovelty k=5": TailNovelty(k=5).fit(legit_ref).score(Xv),
            }
            g = at_budget(scores["GBM (pattern unseen)"], yv, 0.005)
            for name, s in scores.items():
                acc.setdefault(name + " | recall @0.5%", []).append(at_budget(s, yv, 0.005)[yv == 1].mean())
                acc.setdefault(name + " | AUPRC", []).append(M.metrics(yv, s, w.amt[keep])["auprc"])
                if name != "GBM (pattern unseen)":
                    u = g | at_budget(s, yv, 0.005)
                    acc.setdefault(f"GBM OR {name} gate | recall", []).append(u[yv == 1].mean())
                    acc.setdefault(f"GBM OR {name} gate | genuine flagged", []).append(u[yv == 0].mean())
        rows[P] = {k: float(np.mean(v)) for k, v in acc.items()}
        print(f"\n== unseen pattern: {P} ==")
        for k, v in rows[P].items():
            print(f"  {k:<60} {v:.3f}")
    (ROOT / "reports" / "novelty_experiments.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
