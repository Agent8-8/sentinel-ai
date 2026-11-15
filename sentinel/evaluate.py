"""FINAL held-out evaluation. Run once after model selection is frozen.

For each independent world (seed):
  1. train the selected config on days < 63 with censored labels + maturity weights
  2. calibrate on validation (days 63-74); tune baseline thresholds on validation
  3. score the untouched TEST window (days 74-91: holiday spike + unseen fraud ring)
Outputs reports/results.json and saves the seed-7 model for the live demo.

    python -m sentinel.evaluate
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import models as M
from . import policy as P
from .data import LABEL_CUTOFF, ROOT, load, splits
from .features import FEATURES
from .pipeline import SentinelModel
from .simulator import DAY

SEEDS = [7, 11, 23]
TYPES = ["card_testing", "ato", "stolen_card", "sleeper_ring", "friendly"]


def config():
    exp = json.loads((ROOT / "reports" / "experiments.json").read_text())
    v = exp["valid"]
    bag = v["GBM 5-seed bag"]["auprc"]
    iso = max((v[k]["auprc"], float(k.split("+ ")[1].split(" ")[0])) for k in v if "IsolationForest rank blend" in k)
    iso_w = iso[1] if iso[0] > bag + 0.002 else 0.0  # blend only if it clearly helps on validation
    return exp["best_params"], iso_w


def ece(y, p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def tune_threshold_policy(scores, y, amt, c, rng, three_way=False):
    """Amount-blind thresholds picked on VALIDATION to minimise realised cost."""
    qs = np.unique(np.quantile(scores, np.linspace(0.80, 0.9995, 60)))
    best = (np.inf, None)
    for t2 in qs:
        for t1 in (qs[qs <= t2] if three_way else [np.inf]):
            a = np.where(scores > t2, P.DECLINE, np.where(scores > t1, P.HOLD, P.ALLOW))
            cost = np.mean([P.realized_cost(a, y, amt, c, np.random.default_rng(k))["cost"].sum() for k in range(3)])
            if cost < best[0]:
                best = (cost, (t1, t2))
    return best[1]


def apply_thresholds(scores, t):
    t1, t2 = t
    return np.where(scores > t2, P.DECLINE, np.where(scores > t1, P.HOLD, P.ALLOW))


def run_world(seed, params, iso_w, costs):
    df, X = load(seed)
    tr, va, te, y_obs, y_true = splits(df)
    ts, amt, ftype = df.ts.to_numpy(), df.amount.to_numpy(), df.fraud_type.to_numpy()
    cb = df.chargeback_ts.to_numpy()
    early = tr & (ts < 20 * DAY) & ~np.isnan(cb) & (cb <= LABEL_CUTOFF)
    delay = M.fit_delay_distribution((cb[early] - ts[early]) / DAY)
    w = M.maturity_weights(ts, y_obs, LABEL_CUTOFF, *delay)
    fit = ts < LABEL_CUTOFF

    model = SentinelModel(params, n_bag=5, iso_weight=iso_w).fit(X[fit], y_obs[fit], w[fit])
    model.calibrate(X[va], y_true[va])
    p_va, p_te = model.predict(X[va]), model.predict(X[te])
    nov_te = model.is_novel(X[te])

    # baselines trained the same way (censored labels, same window)
    tcols = M.cols(["transaction"])
    lr_txn = M.logistic(X[fit][:, tcols], y_obs[fit])
    gbm_txn = M.hgb(X[fit][:, tcols], y_obs[fit], cat_mask=M.cat_mask_for(tcols))
    scores = {
        "rules engine": (M.rules_score(X[va]), M.rules_score(X[te])),
        "logistic (transaction-only)": (lr_txn.predict_proba(X[va][:, tcols])[:, 1], lr_txn.predict_proba(X[te][:, tcols])[:, 1]),
        "GBM (transaction-only)": (gbm_txn.predict_proba(X[va][:, tcols])[:, 1], gbm_txn.predict_proba(X[te][:, tcols])[:, 1]),
        "Sentinel (behavioural GBM bag)": (p_va, p_te),
    }
    yt, at, ft = y_true[te], amt[te], ftype[te]
    det = {}
    for name, (sv, st) in scores.items():
        m = M.metrics(yt, st, at)
        m["auprc_ci95"] = M.bootstrap_ci(yt, st, at, lambda a, b, c: M.metrics(a, b, c)["auprc"], n=200, seed=seed)
        thr = np.quantile(st[yt == 0], 0.995)
        m["recall_by_type@0.5%fpr"] = {t: float((st[ft == t] > thr).mean()) for t in TYPES}
        det[name] = m
    # whole-system flagging: GBM top-0.5% OR novelty gate (another <=0.5% of genuine)
    thr = np.quantile(p_te[yt == 0], 0.995)
    flag = (p_te > thr) | nov_te
    det["Sentinel + novelty gate (flag = review or verify)"] = {
        "genuine_flagged": float(flag[yt == 0].mean()), "recall": float(flag[yt == 1].mean()),
        "dollar_recall": float(at[flag & (yt == 1)].sum() / at[yt == 1].sum()),
        "recall_by_type": {t: float(flag[ft == t].mean()) for t in TYPES}}
    calib = {"brier": float(np.mean((p_te - yt) ** 2)), "ece": ece(yt, p_te),
             "mean_p": float(p_te.mean()), "fraud_rate": float(yt.mean())}

    # ------------------------------------------------------ decision policies
    yv, av = y_true[va], amt[va]
    rng = lambda: np.random.default_rng(1000 + seed)
    pol = {}

    def record(name, action, out, esc=0):
        s = P.summarize(action, out, yt, at)
        s["merchant_reviews"] = int(esc)
        pol[name] = s

    a = np.zeros(te.sum(), int)
    record("approve everything", a, P.realized_cost(a, yt, at, costs, rng()))
    rs_v, rs_t = scores["rules engine"]
    t = tune_threshold_policy(rs_v, yv, av, costs, rng())
    a = apply_thresholds(rs_t, t)
    flagged = a == P.DECLINE
    # typical small-merchant setup: rule hits go to a MANUAL review queue
    record("rules -> manual review", np.zeros_like(a), P.realized_cost(np.where(flagged, P.HOLD, P.ALLOW), yt, at, costs, rng()),
           esc=int(flagged.sum()))
    pol["rules -> manual review"]["note"] = "every flagged order needs a human decision"
    record("rules -> auto-decline", a, P.realized_cost(a, yt, at, costs, rng()))
    t = tune_threshold_policy(p_va, yv, av, costs, rng())
    a = apply_thresholds(p_te, t)
    record("Sentinel score, binary threshold", a, P.realized_cost(a, yt, at, costs, rng()))
    t = tune_threshold_policy(p_va, yv, av, costs, rng(), three_way=True)
    a = apply_thresholds(p_te, t)
    record("Sentinel score, 3-way fixed thresholds", a, P.realized_cost(a, yt, at, costs, rng()))
    a = P.decide(p_te, at, costs)
    record("Sentinel expected-cost policy", a, P.realized_cost(a, yt, at, costs, rng(), ft))
    a = P.apply_novelty_gate(P.decide(p_te, at, costs), nov_te)
    record("Sentinel expected-cost + novelty gate", a, P.realized_cost(a, yt, at, costs, rng(), ft))
    a, out, esc = P.run_sequential(df[te], p_te, costs, rng(), trust_memory=True, novel=nov_te)
    record("Sentinel full (cost + novelty gate + trust memory)", a, out, esc)

    if seed == SEEDS[0]:
        (ROOT / "artifacts").mkdir(exist_ok=True)
        model.save(ROOT / "artifacts" / "sentinel_model.pkl")
        legit_ref = np.median(X[fit][y_obs[fit] == 0], axis=0)
        np.save(ROOT / "artifacts" / "legit_reference.npy", legit_ref)
    return {"detection": det, "calibration": calib, "policy": pol,
            "n_test": int(te.sum()), "n_test_fraud": int(yt.sum()), "test_fraud_dollars": float(at[yt == 1].sum())}


def mean_over(worlds, path):
    def get(d, p):
        for k in p:
            d = d[k]
        return d
    vals = [get(w, path) for w in worlds]
    if isinstance(vals[0], dict):
        return {k: mean_over(worlds, path + [k]) for k in vals[0]}
    if isinstance(vals[0], (int, float)):
        return float(np.mean(vals))
    if isinstance(vals[0], list):
        return [float(np.mean([v[i] for v in vals])) for i in range(len(vals[0]))]
    return vals[0]


def main():
    params, iso_w = config()
    costs = P.Costs()
    print("config:", params, "iso_weight:", iso_w, "\ncosts:", costs.dict())
    worlds = {}
    for s in SEEDS:
        t = time.time()
        worlds[s] = run_world(s, params, iso_w, costs)
        print(f"world {s} done in {time.time() - t:.0f}s")
    W = list(worlds.values())
    summary = {k: mean_over(W, [k]) for k in ["detection", "calibration", "policy"]}
    out = {"config": {"params": params, "iso_weight": iso_w, "costs": costs.dict(), "seeds": SEEDS,
                      "features": FEATURES},
           "mean": summary, "per_world": {str(k): v for k, v in worlds.items()}}
    (ROOT / "reports" / "results.json").write_text(json.dumps(out, indent=1))

    print("\n=== TEST detection (mean of 3 worlds) ===")
    for n, m in summary["detection"].items():
        if "novelty gate" in n:
            print(f"{n}: recall {m['recall']:.3f} $recall {m['dollar_recall']:.3f} genuine flagged {m['genuine_flagged']:.4f} | "
                  + " ".join(f"{k[:6]} {v:.2f}" for k, v in m["recall_by_type"].items()))
            continue
        bt = m["recall_by_type@0.5%fpr"]
        print(f"{n:<32} AUPRC {m['auprc']:.3f}  R@0.5%FPR {m['recall@0.5%fpr']:.3f}  $R {m['dollar_recall@0.5%fpr']:.3f} | "
              + " ".join(f"{k[:6]} {v:.2f}" for k, v in bt.items()))
    print("calibration:", summary["calibration"])
    print("\n=== TEST decision policies (mean of 3 worlds) ===")
    for n, s in summary["policy"].items():
        print(f"{n:<40} cost ${s['total_cost']:>9,.0f}  fraud$ stopped {s['fraud_dollar_stop_rate']:.1%}  "
              f"false-decline {s['false_decline_rate']:.2%}  legit friction {s['legit_friction_rate']:.2%}  "
              f"legit lost {s['legit_lost_rate']:.2%}  reviews {s['merchant_reviews']:.0f}")


if __name__ == "__main__":
    main()
