"""Streaming, strictly-causal behavioural feature engine.

One `FeatureEngine` instance processes transactions in time order. For each
transaction it first computes features from state built from *earlier*
transactions only, then updates the state. The exact same code path is used
for offline training and for live scoring in the API, so there is no
train/serve skew and no look-ahead leakage by construction.

Label-derived features ("entity linked to a confirmed fraud") only use
chargebacks whose arrival time is earlier than the current transaction.
"""
from __future__ import annotations

import bisect
import heapq
import math
from collections import defaultdict

import numpy as np

from .simulator import CATEGORIES, DAY, GEO, HOUR

MIN = 60.0

# Feature groups (used for ablations and reason codes)
GROUPS = {
    "transaction": ["amount_log", "category", "n_items", "local_hour", "is_guest", "account_age_days_log",
                    "email_age_days_log", "ship_bill_mismatch", "ip_bill_country_mismatch",
                    "card_bill_country_mismatch", "ip_bill_distance_km_log", "foreign_ip"],
    "velocity": ["card_tx_1h", "card_tx_24h", "card_amt_24h_log", "acct_tx_1h", "acct_tx_24h",
                 "secs_since_card_last_log", "ip_cards_1h"],
    "deviation": ["amt_vs_acct_median", "acct_history_n_log", "card_first_seen", "new_device_for_acct",
                  "new_addr_for_acct", "new_ip_country_for_acct", "hour_vs_acct_typical",
                  "acct_distinct_devices_7d", "new_category_for_acct"],
    "linkage": ["dev_distinct_cards_24h", "dev_distinct_cards_all", "dev_distinct_accts_all", "dev_tx_1h",
                "addr_distinct_cards_30d", "addr_distinct_accts_30d", "dev_age_days_log"],
    "reputation": ["dev_confirmed_fraud", "addr_confirmed_fraud", "card_confirmed_fraud",
                   "acct_confirmed_fraud"],
    "merchant_pulse": ["merchant_small_tx_10m", "merchant_new_cards_10m"],
}
FEATURES = [f for g in GROUPS.values() for f in g]
CATEGORICAL = ["category"]
CAT_CODE = {c: i for i, c in enumerate(CATEGORIES)}


def haversine_km(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * math.asin(min(1.0, math.sqrt(a)))


class _Times:
    """Append-only sorted timestamps (+ cumulative values) for window counts/sums."""
    __slots__ = ("t", "cum")

    def __init__(self):
        self.t: list[float] = []
        self.cum: list[float] = [0.0]

    def count(self, now, window):
        return len(self.t) - bisect.bisect_left(self.t, now - window)

    def total(self, now, window):
        i = bisect.bisect_left(self.t, now - window)
        return self.cum[-1] - self.cum[i]

    def add(self, now, v=0.0):
        self.t.append(now)
        self.cum.append(self.cum[-1] + v)

    def last(self):
        return self.t[-1] if self.t else None


def _distinct_recent(d: dict, now, window):
    lo = now - window
    return sum(1 for ts in d.values() if ts >= lo)


class FeatureEngine:
    def __init__(self):
        self.card_t = defaultdict(_Times)
        self.acct_t = defaultdict(_Times)
        self.ip_cards = defaultdict(dict)        # ip -> {card: last_ts}
        self.dev_cards = defaultdict(dict)       # device -> {card: last_ts}
        self.dev_accts = defaultdict(set)
        self.dev_t = defaultdict(_Times)
        self.dev_first = {}
        self.addr_cards = defaultdict(dict)
        self.addr_accts = defaultdict(dict)
        self.acct_amounts = defaultdict(list)
        self.acct_hours = defaultdict(list)
        self.acct_devices = defaultdict(dict)    # acct -> {device: last_ts}
        self.acct_addrs = defaultdict(set)
        self.acct_ipc = defaultdict(set)
        self.acct_cats = defaultdict(set)
        self.cards_seen = set()
        self.merchant_small = _Times()
        self.merchant_newcard = _Times()
        # reputation: pending chargebacks (arrival_ts, entities) -> confirmed sets
        self._pending: list = []
        self.bad = {"dev": set(), "addr": set(), "card": set(), "acct": set()}

    # -- labels -------------------------------------------------------------
    def register_chargeback(self, arrival_ts, tx):
        """Schedule a chargeback that becomes visible at `arrival_ts`."""
        heapq.heappush(self._pending, (arrival_ts, tx["device_id"], tx["ship_address_id"],
                                       tx["card_id"], tx["account_id"]))

    def mark_bad(self, tx):
        """Immediate confirmation (e.g. failed step-up verification in live mode)."""
        self.bad["dev"].add(tx["device_id"]); self.bad["addr"].add(tx["ship_address_id"])
        self.bad["card"].add(tx["card_id"]); self.bad["acct"].add(tx["account_id"])

    def _release(self, now):
        while self._pending and self._pending[0][0] < now:
            _, d, a, c, ac = heapq.heappop(self._pending)
            self.bad["dev"].add(d); self.bad["addr"].add(a); self.bad["card"].add(c); self.bad["acct"].add(ac)

    # -- core ---------------------------------------------------------------
    def compute(self, tx) -> dict:
        now = tx["ts"]
        self._release(now)
        card, acct, dev, ip, addr = tx["card_id"], tx["account_id"], tx["device_id"], tx["ip_id"], tx["ship_address_id"]
        amt = tx["amount"]
        local_hour = ((now / HOUR) + tx["ip_lon"] / 15.0) % 24
        dist = haversine_km(tx["billing_lat"], tx["billing_lon"], tx["ip_lat"], tx["ip_lon"])
        hist = self.acct_amounts[acct]
        hours = self.acct_hours[acct]
        ct, at = self.card_t[card], self.acct_t[acct]
        last_card = ct.last()
        f = {
            "amount_log": math.log1p(amt),
            "category": CAT_CODE[tx["category"]],
            "n_items": tx["n_items"],
            "local_hour": local_hour,
            "is_guest": float(tx["is_guest"]),
            "account_age_days_log": math.log1p(max(0.0, (now - tx["account_created_ts"]) / DAY)),
            "email_age_days_log": math.log1p(max(0.0, tx["email_age_days"])),
            "ship_bill_mismatch": float(tx["ship_country"] != tx["billing_country"]),
            "ip_bill_country_mismatch": float(tx["ip_country"] != tx["billing_country"]),
            "card_bill_country_mismatch": float(tx["card_country"] != tx["billing_country"]),
            "ip_bill_distance_km_log": math.log1p(dist),
            "foreign_ip": float(tx["ip_country"] != "US"),
            # velocity
            "card_tx_1h": ct.count(now, HOUR),
            "card_tx_24h": ct.count(now, DAY),
            "card_amt_24h_log": math.log1p(ct.total(now, DAY)),
            "acct_tx_1h": at.count(now, HOUR),
            "acct_tx_24h": at.count(now, DAY),
            "secs_since_card_last_log": math.log1p(now - last_card) if last_card is not None else -1.0,
            "ip_cards_1h": _distinct_recent(self.ip_cards[ip], now, HOUR),
            # deviation from the account's own history
            "amt_vs_acct_median": (amt / (np.median(hist[-20:]) + 1.0)) if hist else -1.0,
            "acct_history_n_log": math.log1p(len(hist)),
            "card_first_seen": float(card not in self.cards_seen),
            "new_device_for_acct": float(bool(hist) and dev not in self.acct_devices[acct]),
            "new_addr_for_acct": float(bool(hist) and addr not in self.acct_addrs[acct]),
            "new_ip_country_for_acct": float(bool(hist) and tx["ip_country"] not in self.acct_ipc[acct]),
            "hour_vs_acct_typical": (abs(((local_hour - np.median(hours[-20:])) + 12) % 24 - 12)) if hours else -1.0,
            "acct_distinct_devices_7d": _distinct_recent(self.acct_devices[acct], now, 7 * DAY),
            "new_category_for_acct": float(bool(hist) and tx["category"] not in self.acct_cats[acct]),
            # linkage across entities
            "dev_distinct_cards_24h": _distinct_recent(self.dev_cards[dev], now, DAY),
            "dev_distinct_cards_all": len(self.dev_cards[dev]),
            "dev_distinct_accts_all": len(self.dev_accts[dev]),
            "dev_tx_1h": self.dev_t[dev].count(now, HOUR),
            "addr_distinct_cards_30d": _distinct_recent(self.addr_cards[addr], now, 30 * DAY),
            "addr_distinct_accts_30d": _distinct_recent(self.addr_accts[addr], now, 30 * DAY),
            "dev_age_days_log": math.log1p((now - self.dev_first[dev]) / DAY) if dev in self.dev_first else -1.0,
            # reputation (only chargebacks already received)
            "dev_confirmed_fraud": float(dev in self.bad["dev"]),
            "addr_confirmed_fraud": float(addr in self.bad["addr"]),
            "card_confirmed_fraud": float(card in self.bad["card"]),
            "acct_confirmed_fraud": float(acct in self.bad["acct"]),
            # merchant-wide pulse (bot attacks)
            "merchant_small_tx_10m": self.merchant_small.count(now, 10 * MIN),
            "merchant_new_cards_10m": self.merchant_newcard.count(now, 10 * MIN),
        }
        return f

    def update(self, tx):
        now = tx["ts"]
        card, acct, dev, ip, addr = tx["card_id"], tx["account_id"], tx["device_id"], tx["ip_id"], tx["ship_address_id"]
        if card not in self.cards_seen:
            self.merchant_newcard.add(now)
        if tx["amount"] < 10:
            self.merchant_small.add(now)
        self.card_t[card].add(now, tx["amount"])
        self.acct_t[acct].add(now)
        self.ip_cards[ip][card] = now
        self.dev_cards[dev][card] = now
        self.dev_accts[dev].add(acct)
        self.dev_t[dev].add(now)
        self.dev_first.setdefault(dev, now)
        self.addr_cards[addr][card] = now
        self.addr_accts[addr][acct] = now
        self.acct_amounts[acct].append(tx["amount"])
        self.acct_hours[acct].append(((now / HOUR) + tx["ip_lon"] / 15.0) % 24)
        self.acct_devices[acct][dev] = now
        self.acct_addrs[acct].add(addr)
        self.acct_ipc[acct].add(tx["ip_country"])
        self.acct_cats[acct].add(tx["category"])
        self.cards_seen.add(card)

    def process(self, tx) -> dict:
        f = self.compute(tx)
        self.update(tx)
        return f


def build_feature_matrix(df, engine: FeatureEngine | None = None):
    """Run the stream over a time-sorted dataframe. Chargebacks are registered
    as they occur so reputation features only see already-arrived labels."""
    engine = engine or FeatureEngine()
    out = np.empty((len(df), len(FEATURES)), dtype=np.float64)
    for i, tx in enumerate(df.to_dict("records")):
        f = engine.process(tx)
        out[i] = [f[k] for k in FEATURES]
        if not math.isnan(tx.get("chargeback_ts", float("nan"))):
            engine.register_chargeback(tx["chargeback_ts"], tx)
    return out, engine
