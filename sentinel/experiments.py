"""Model-development experiments. VALIDATION WINDOW ONLY - test windows are
never scored here. Every number is averaged over independent simulated
worlds (seeds) to reduce noise. Results -> reports/experiments.json

    python -m sentinel.experiments
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata

from . import models as M
from .data import LABEL_CUTOFF, ROOT, load, splits
from .features import GROUPS
from .simulator import DAY

ALL = list(GROUPS)
SEEDS = [7, 11, 23]
FRAUD_TYPES = ["card_testing", "ato", "stolen_card", "friendly"]


class World:
    def __init__(self, seed):
        self.df, self.X = load(seed)
        tr, va, te, y_obs, y_true = splits(self.df)
        self.ts, self.amt = self.df.ts.to_numpy(), self.df.amount.to_numpy()
        self.va, self.y_true, self.y_obs = va, y_true, y_obs
        self.ftype = self.df.fraud_type.to_numpy()
        cb = self.df.chargeback_ts.to_numpy()
        early = tr & (self.ts < 20 * DAY) & ~np.isnan(cb) & (cb <= LABEL_CUTOFF)
        self.delay = M.fit_delay_distribution((cb[early] - self.ts[early]) / DAY)
        self.w_mat = M.maturity_weights(self.ts, y_obs, LABEL_CUTOFF, *self.delay)
        self.tr = tr
        self.ext = self.ts < LABEL_CUTOFF  # train + gap week


def score_config(w: World, cfg: dict):
    groups = cfg.get("groups", ALL)
    c = M.cols(groups)
    fit_mask = w.ext if cfg.get("gap_week") else w.tr
    if cfg.get("drop_type"):  # leave-one-pattern-out: remove that pattern's positives from training
        fit_mask = fit_mask & ~((w.ftype == cfg["drop_type"]) & (w.y_obs == 1))
    X, y = w.X[fit_mask][:, c], w.y_obs[fit_mask]
    if cfg.get("oracle"):
        y = w.y_true[fit_mask]
    sw = w.w_mat[fit_mask] if cfg.get("maturity") else None
    Xv = w.X[w.va][:, c]
    kind = cfg.get("model", "gbm")
    if kind == "rules":
        return M.rules_score(w.X[w.va])
    if kind == "logistic":
        return M.logistic(X, y, sw).predict_proba(Xv)[:, 1]
    params = cfg.get("params", {})
    bags = cfg.get("bag", 1)
    s = np.mean([M.hgb(X, y, w=sw, cat_mask=M.cat_mask_for(c), seed=k, **params).predict_proba(Xv)[:, 1]
                 for k in range(bags)], axis=0)
    if cfg.get("iso_blend"):
        iso = M.isolation_scores(X[y == 0], Xv)
        s = (1 - cfg["iso_blend"]) * rankdata(s) / len(s) + cfg["iso_blend"] * rankdata(iso) / len(s)
    return s


def evaluate(worlds, name, cfg, results, target_type=None):
    agg = defaultdict(list)
    for w in worlds:
        s = score_config(w, cfg)
        yv, av, ft = w.y_true[w.va], w.amt[w.va], w.ftype[w.va]
        if target_type:  # evaluate only on legit + the held-out pattern
            keep = (yv == 0) | (ft == target_type)
            yv, av, s, ft = yv[keep], av[keep], s[keep], ft[keep]
        m = M.metrics(yv, s, av)
        thr = np.quantile(s[yv == 0], 0.995)
        for t in FRAUD_TYPES:
            sel = ft == t
            if sel.any():
                m[f"recall_{t}"] = float((s[sel] > thr).mean())
        for k, v in m.items():
            agg[k].append(v)
    out = {k: float(np.mean(v)) for k, v in agg.items()}
    out["auprc_sd"] = float(np.std(agg["auprc"]))
    results[name] = out
    types = "  ".join(f"{t[:5]} {out.get('recall_' + t, float('nan')):.2f}" for t in FRAUD_TYPES)
    print(f"{name:<46} AUPRC {out['auprc']:.3f}±{out['auprc_sd']:.3f}  R@.5% {out['recall@0.5%fpr']:.3f}  "
          f"$R@.5% {out['dollar_recall@0.5%fpr']:.3f} | {types}", flush=True)
    return out


def main(quick=False):
    worlds = [World(s) for s in (SEEDS[:1] if quick else SEEDS)]
    for w, s in zip(worlds, SEEDS):
        yv = w.y_true[w.va]
        print(f"world {s}: valid n={w.va.sum()} fraud={yv.sum()}  delay fit={np.round(w.delay, 2)}")
    R = {}
    best = dict(learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0)

    print("\n== Baselines ==")
    evaluate(worlds, "rules engine", {"model": "rules"}, R)
    evaluate(worlds, "logistic, transaction-only", {"model": "logistic", "groups": ["transaction"]}, R)
    evaluate(worlds, "GBM, transaction-only", {"groups": ["transaction"]}, R)
    evaluate(worlds, "logistic, all features", {"model": "logistic"}, R)
    evaluate(worlds, "GBM, all features", {}, R)

    print("\n== Feature-group ablation (drop one) ==")
    for g in ALL[1:]:
        evaluate(worlds, f"GBM without [{g}]", {"groups": [x for x in ALL if x != g]}, R)

    print("\n== Label handling ==")
    evaluate(worlds, "GBM + maturity weighting", {"maturity": True}, R)
    evaluate(worlds, "GBM + maturity + gap week", {"maturity": True, "gap_week": True}, R)
    evaluate(worlds, "GBM oracle labels (upper bound)", {"oracle": True}, R)

    print("\n== Hyperparameters (maturity + gap week) ==")
    grid = [(0.05, 31, 40, 1.0), (0.05, 15, 20, 1.0), (0.05, 7, 20, 1.0), (0.03, 15, 20, 3.0),
            (0.1, 15, 40, 0.0), (0.03, 63, 80, 3.0)]
    best_auprc = -1
    for lr, l, msl, l2 in grid:
        p = dict(learning_rate=lr, max_leaf_nodes=l, min_samples_leaf=msl, l2_regularization=l2)
        o = evaluate(worlds, f"GBM lr={lr} leaves={l} min_leaf={msl} l2={l2}",
                     {"maturity": True, "gap_week": True, "params": p}, R)
        if o["auprc"] > best_auprc:
            best, best_auprc = p, o["auprc"]
    print("best params:", best)

    print("\n== Ensembling ==")
    base = {"maturity": True, "gap_week": True, "params": best}
    evaluate(worlds, "GBM 5-seed bag", {**base, "bag": 5}, R)
    for b in (0.1, 0.25):
        evaluate(worlds, f"GBM bag + {b} IsolationForest rank blend", {**base, "bag": 5, "iso_blend": b}, R)

    print("\n== Generalisation to UNSEEN fraud patterns (leave-one-pattern-out) ==")
    print("   trained without any example of the pattern; scored on legit + that pattern only")
    for t in ["card_testing", "ato", "stolen_card"]:
        for label, groups in [("transaction-only", ["transaction"]), ("all behavioural", ALL)]:
            evaluate(worlds, f"unseen {t:<12} | GBM {label}",
                     {**base, "groups": groups, "drop_type": t}, R, target_type=t)

    out = {"seeds": SEEDS[:len(worlds)], "best_params": best, "valid": R}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "experiments.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    t = time.time()
    main(quick="--quick" in sys.argv)
    print(f"\n{time.time() - t:.0f}s")
