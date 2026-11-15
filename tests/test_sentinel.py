"""Correctness tests for the leakage / causality / policy claims."""
import math

import numpy as np
import pandas as pd
import pytest

from sentinel import policy as P
from sentinel.data import LABEL_CUTOFF, TRAIN_END, VALID_END, VALID_START, splits
from sentinel.features import FeatureEngine, build_feature_matrix
from sentinel.simulator import DAY, simulate


@pytest.fixture(scope="module")
def world():
    df = simulate(seed=3, n_customers=2500)
    X, _ = build_feature_matrix(df)
    return df, X


def test_no_lookahead(world):
    """Features of a transaction must not change when the future is deleted or rewritten."""
    df, X = world
    k = len(df) // 2
    past = df.iloc[:k].copy()
    Xp, _ = build_feature_matrix(past)
    assert np.allclose(Xp, X[:k])
    # rewrite the future wildly: features of the past still identical
    fut = df.iloc[k:].copy()
    fut["amount"] *= 50
    fut["device_id"] = "dev_attack"
    fut["chargeback_ts"] = fut.ts + 1.0
    X2, _ = build_feature_matrix(pd.concat([past, fut]))
    assert np.allclose(X2[:k], X[:k])


def test_reputation_waits_for_chargeback():
    eng = FeatureEngine()
    tx = dict(ts=0.0, account_id="a", is_guest=False, account_created_ts=-100 * DAY, email_age_days=500,
              card_id="c", card_country="US", billing_country="US", billing_lat=40, billing_lon=-90,
              ship_address_id="s", ship_country="US", device_id="d", ip_id="i", ip_country="US",
              ip_lat=40, ip_lon=-90, category="home", n_items=1, amount=50.0)
    eng.process(tx)
    eng.register_chargeback(10 * DAY, tx)
    f_before = eng.compute({**tx, "ts": 9 * DAY})
    f_after = eng.compute({**tx, "ts": 11 * DAY})
    assert f_before["dev_confirmed_fraud"] == 0 and f_after["dev_confirmed_fraud"] == 1


def test_splits_and_label_censoring(world):
    df, _ = world
    tr, va, te, y_obs, y_true = splits(df)
    assert not (tr & va).any() and not (va & te).any() and not (tr & te).any()
    ts, cb = df.ts.to_numpy(), df.chargeback_ts.to_numpy()
    assert ts[tr].max() < TRAIN_END <= VALID_START and ts[va].max() < VALID_END <= ts[te].min()
    # training labels only from chargebacks that had arrived by the cutoff
    assert np.all(cb[y_obs == 1] <= LABEL_CUTOFF)
    assert y_obs[~tr].sum() == 0


def test_novel_pattern_only_in_test(world):
    df, _ = world
    _, _, te, _, _ = splits(df)
    ring = (df.fraud_type == "sleeper_ring").to_numpy()
    assert ring.any() and not (ring & ~te).any()


def test_policy_sane():
    c = P.Costs()
    amt = np.full(101, 120.0)
    a = P.decide(np.linspace(0, 1, 101), amt, c)
    assert a[0] == P.ALLOW and a[-1] == P.DECLINE
    assert np.all(np.diff(a) >= 0), "action severity must be monotone in risk"
    # same risk, bigger order -> never a softer action
    p = np.full(50, 0.08)
    b = P.decide(p, np.linspace(5, 3000, 50), c)
    assert np.all(np.diff(b) >= 0)


def test_expected_costs_match_realized_on_average():
    c = P.Costs()
    rng = np.random.default_rng(0)
    p, amt = 0.2, 200.0
    n = 200_000
    y = rng.random(n) < p
    for act in (P.ALLOW, P.HOLD, P.DECLINE):
        real = P.realized_cost(np.full(n, act), y, np.full(n, amt), c, rng)["cost"].mean()
        exp = P.expected_costs(p, amt, c)[act]
        assert math.isclose(real, exp, rel_tol=0.03), (act, real, exp)
