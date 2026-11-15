"""Autonomous decision policy: ALLOW / HOLD (automated step-up verification) / DECLINE.

For every transaction we know p = calibrated P(fraud) and A = order amount.
Each action has an expected cost to the merchant:

  allow   : p * (A + chargeback_fee)                             fraud slips through
  decline : (1-p) * (margin*A + churn_cost)                      a real customer is turned away
  hold    : friction + p*(1-catch)*(A+fee)                       step-up (3DS / OTP / ID check)
            + (1-p)*abandon*(margin*A + churn_cost/2)            some genuine buyers abandon

The engine picks the cheapest action. No merchant is in the loop; the only
escalation is "verification passed but risk still extreme" (rare), which is
counted as merchant workload.

Trust memory: once a device has passed step-up for an account, a later
mid-risk transaction from that same (account, device) pair is allowed
without asking the customer again - unless risk is very high.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Costs:
    margin: float = 0.30          # gross margin lost when a genuine sale is declined/abandoned
    chargeback_fee: float = 25.0  # dispute fee + ops cost per chargeback
    churn_cost: float = 20.0      # lifetime-value damage of falsely declining a good customer
    friction: float = 0.40        # cost of sending a verification (SMS/3DS) + annoyance
    catch_rate: float = 0.85      # share of fraudsters that fail step-up
    abandon: float = 0.07         # share of genuine customers that abandon at step-up

    def dict(self):
        return asdict(self)


ALLOW, HOLD, DECLINE = 0, 1, 2
NAMES = {ALLOW: "allow", HOLD: "hold", DECLINE: "decline"}


def expected_costs(p, amount, c: Costs):
    p, A = np.asarray(p, float), np.asarray(amount, float)
    lost_sale = c.margin * A + c.churn_cost
    allow = p * (A + c.chargeback_fee)
    decline = (1 - p) * lost_sale
    hold = c.friction + p * (1 - c.catch_rate) * (A + c.chargeback_fee) + (1 - p) * c.abandon * (c.margin * A + c.churn_cost / 2)
    return np.stack([allow, hold, decline], axis=-1)


def decide(p, amount, c: Costs):
    return expected_costs(p, amount, c).argmin(axis=-1)


# ------------------------------------------------------- outcome simulation
def realized_cost(action, is_fraud, amount, c: Costs, rng, ftype=None):
    """Simulate what actually happens. Returns dict of per-tx outcome arrays."""
    n = len(action)
    action, y, A = np.asarray(action), np.asarray(is_fraud).astype(bool), np.asarray(amount, float)
    u = rng.random(n)
    # ATO fraudsters sometimes control the victim's phone -> pass step-up more often
    catch = np.full(n, c.catch_rate)
    if ftype is not None:
        catch = np.where(np.asarray(ftype) == "ato", c.catch_rate - 0.25, catch)
    held = action == HOLD
    fraud_passes = held & y & (u > catch)
    legit_abandons = held & ~y & (u < c.abandon)
    fraud_loss = ((action == ALLOW) & y) | fraud_passes
    false_decline = (action == DECLINE) & ~y
    cost = (fraud_loss * (A + c.chargeback_fee)
            + false_decline * (c.margin * A + c.churn_cost)
            + held * c.friction
            + legit_abandons * (c.margin * A + c.churn_cost / 2))
    return dict(cost=cost, fraud_loss=fraud_loss, false_decline=false_decline,
                legit_abandon=legit_abandons, held=held, fraud_passes=fraud_passes,
                verified_ok=held & ~legit_abandons & (~y | fraud_passes))


def summarize(action, out, is_fraud, amount):
    y, A = np.asarray(is_fraud).astype(bool), np.asarray(amount, float)
    legit = ~y
    return {
        "total_cost": float(out["cost"].sum()),
        "fraud_dollars": float(A[y].sum()),
        "fraud_dollars_stopped": float(A[y & ~out["fraud_loss"]].sum()),
        "fraud_dollar_stop_rate": float(A[y & ~out["fraud_loss"]].sum() / max(A[y].sum(), 1e-9)),
        "fraud_count_stop_rate": float((y & ~out["fraud_loss"]).sum() / max(y.sum(), 1)),
        "false_decline_rate": float(out["false_decline"][legit].mean()),
        "legit_friction_rate": float(out["held"][legit].mean()),
        "legit_lost_rate": float((out["false_decline"] | out["legit_abandon"])[legit].mean()),
        "legit_revenue_lost": float(A[legit & (out["false_decline"] | out["legit_abandon"])].sum()),
        "allow_rate": float((action == ALLOW).mean()),
        "hold_rate": float((action == HOLD).mean()),
        "decline_rate": float((action == DECLINE).mean()),
    }


def apply_novelty_gate(action, novel):
    """Upgrade ALLOW -> HOLD for orders the novelty detector finds extreme."""
    return np.where((np.asarray(action) == ALLOW) & np.asarray(novel), HOLD, action)


def run_sequential(df_rows, p, c: Costs, rng, trust_memory=True, escalate_p=0.9, novel=None):
    """Stream through transactions in time order, applying trust memory and
    escalation. Returns action array, outcomes, and number of merchant escalations."""
    n = len(p)
    amount = df_rows.amount.to_numpy()
    base = decide(p, amount, c)
    if novel is not None:
        base = apply_novelty_gate(base, novel)
    acct, dev = df_rows.account_id.to_numpy(), df_rows.device_id.to_numpy()
    y, ftype = df_rows.is_fraud.to_numpy(), df_rows.fraud_type.to_numpy()
    trusted: set = set()
    action = base.copy()
    escalations = 0
    outcomes = {k: np.zeros(n, bool) for k in ["fraud_loss", "false_decline", "legit_abandon", "held",
                                                  "fraud_passes", "verified_ok"]}
    cost = np.zeros(n)
    for i in range(n):
        key = (acct[i], dev[i])
        a = base[i]
        if trust_memory and a == HOLD and key in trusted and p[i] < 0.5:
            a = ALLOW
        action[i] = a
        o = realized_cost(np.array([a]), y[i:i + 1], amount[i:i + 1], c, rng, ftype[i:i + 1])
        if o["verified_ok"][0]:
            trusted.add(key)
            if p[i] >= escalate_p:
                escalations += 1  # passed step-up yet extreme risk -> merchant glance
        for k in outcomes:
            outcomes[k][i] = o[k][0]
        cost[i] = o["cost"][0]
    outcomes["cost"] = cost
    return action, outcomes, escalations
