"""Independent check of the core ML on REAL public data: the ULB/Worldline
credit-card fraud dataset (284,807 European card transactions over 2 days,
492 frauds; features V1-V28 are anonymised PCA components).

It has no card/device/account identifiers, so behavioural features cannot be
built - this benchmark only tests the modelling + calibration stack.

Strict temporal split (no shuffling): first 60% of time = train,
next 15% = validation (model selection), last 25% = test (scored once).

    python -m sentinel.benchmark_public
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import models as M
from .data import ROOT

PATH = ROOT / "data" / "public" / "creditcard.pq"


def main():
    df = pd.read_parquet(PATH).sort_values("Time", kind="stable").reset_index(drop=True)
    y = (df["Class"].astype(str).str.strip("'") == "1").astype(int).to_numpy()
    amt = df["Amount"].to_numpy(dtype=float)
    feats = [f"V{i}" for i in range(1, 29)]
    X = np.c_[df[feats].to_numpy(dtype=float), np.log1p(amt), (df.Time.to_numpy() / 3600) % 24]
    t = df.Time.to_numpy()
    q60, q75 = np.quantile(t, [0.60, 0.75])
    tr, va, te = t < q60, (t >= q60) & (t < q75), t >= q75
    print(f"train {tr.sum()} ({y[tr].sum()} fraud)  valid {va.sum()} ({y[va].sum()})  test {te.sum()} ({y[te].sum()})")

    res = {}

    def report(name, s_va, s_te):
        res[name] = {"valid": M.metrics(y[va], s_va, amt[va]), "test": M.metrics(y[te], s_te, amt[te])}
        res[name]["test"]["auprc_ci95"] = M.bootstrap_ci(
            y[te], s_te, amt[te], lambda a, b, c: M.metrics(a, b, c)["auprc"], n=200)
        v, s = res[name]["valid"], res[name]["test"]
        print(f"{name:<36} valid AUPRC {v['auprc']:.3f} | TEST AUPRC {s['auprc']:.3f} "
              f"[{s['auprc_ci95'][0]:.3f},{s['auprc_ci95'][1]:.3f}]  R@0.5%FPR {s['recall@0.5%fpr']:.3f}  "
              f"$R@0.5% {s['dollar_recall@0.5%fpr']:.3f}")

    # Simple baselines
    amt_rule = lambda m: amt[m] + 0  # "big transactions are risky"
    report("baseline: amount only", amt_rule(va), amt_rule(te))
    lr = M.logistic(X[tr], y[tr])
    report("baseline: logistic regression", lr.predict_proba(X[va])[:, 1], lr.predict_proba(X[te])[:, 1])

    # Sentinel stack: tuned GBM bag (+ optional IsolationForest blend), chosen on validation
    best, best_v = None, -1
    for p in [dict(learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0),
              dict(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0),
              dict(learning_rate=0.03, max_leaf_nodes=7, min_samples_leaf=50, l2_regularization=3.0)]:
        m = M.hgb(X[tr], y[tr], **p)
        v = M.metrics(y[va], m.predict_proba(X[va])[:, 1], amt[va])["auprc"]
        print(f"  valid {p} -> {v:.3f}")
        if v > best_v:
            best, best_v = p, v
    bag = [M.hgb(X[tr], y[tr], seed=k, **best) for k in range(5)]
    g_va = np.mean([m.predict_proba(X[va])[:, 1] for m in bag], axis=0)
    g_te = np.mean([m.predict_proba(X[te])[:, 1] for m in bag], axis=0)
    report("Sentinel GBM 5-bag", g_va, g_te)
    iso_va = M.isolation_scores(X[tr][y[tr] == 0], X[va])
    iso_te = M.isolation_scores(X[tr][y[tr] == 0], X[te])
    blend = lambda g, i: 0.85 * rankdata(g) / len(g) + 0.15 * rankdata(i) / len(i)
    report("Sentinel GBM bag + IsoForest blend", blend(g_va, iso_va), blend(g_te, iso_te))

    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "public_benchmark.json").write_text(json.dumps({"best_params": best, "results": res}, indent=1))


if __name__ == "__main__":
    main()
