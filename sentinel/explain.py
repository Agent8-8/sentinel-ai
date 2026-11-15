"""Explanations: faithful reason codes (occlusion on the real model) + optional
Gemini narration. Gemini never decides anything; it only rewrites facts the
model already produced, so the system is fully functional without a key.

Set GEMINI_API_KEY (and optionally GEMINI_MODEL) to enable narration.
"""
from __future__ import annotations

import json
import math
import os
import urllib.request

import numpy as np

from .features import FEATURES, GROUPS

FIDX = {f: i for i, f in enumerate(FEATURES)}

# Finer-grained "signals" for reason codes: name -> (features, human description)
SIGNALS = {
    "amount": (["amount_log", "n_items"], "order value"),
    "account_newness": (["account_age_days_log", "email_age_days_log", "is_guest", "acct_history_n_log"],
                        "new / unestablished account or email"),
    "geo_mismatch": (["ip_bill_country_mismatch", "ip_bill_distance_km_log", "foreign_ip",
                      "card_bill_country_mismatch", "ship_bill_mismatch"], "location doesn't match billing details"),
    "velocity": (GROUPS["velocity"], "burst of activity on this card / account / IP"),
    "behaviour_change": (["amt_vs_acct_median", "new_device_for_acct", "new_addr_for_acct",
                          "new_ip_country_for_acct", "hour_vs_acct_typical", "acct_distinct_devices_7d",
                          "new_category_for_acct", "card_first_seen"], "differs from this customer's usual behaviour"),
    "linked_entities": (GROUPS["linkage"], "device or shipping address shared across many cards/accounts"),
    "known_bad": (GROUPS["reputation"], "linked to a previous confirmed fraud"),
    "attack_wave": (GROUPS["merchant_pulse"], "store-wide spike of small orders / new cards (bot attack)"),
    "category_time": (["category", "local_hour"], "product category and time of day"),
}


class Explainer:
    def __init__(self, scorer, reference_row: np.ndarray):
        """scorer: fn(X)->calibrated p. reference_row: median feature vector of legit traffic."""
        self.scorer = scorer
        self.ref = reference_row

    def reason_codes(self, x: np.ndarray, top=4):
        base = float(self.scorer(x[None, :])[0])
        rows = []
        for name, (feats, _) in SIGNALS.items():
            z = x.copy()
            idx = [FIDX[f] for f in feats]
            z[idx] = self.ref[idx]
            rows.append(z)
        ps = self.scorer(np.vstack(rows))
        out = []
        for (name, (feats, desc)), p in zip(SIGNALS.items(), ps):
            out.append({"signal": name, "description": desc, "impact": round(base - float(p), 4)})
        out.sort(key=lambda r: -r["impact"])
        return base, [r for r in out[:top] if r["impact"] > 0.005]


def facts(tx: dict, f: dict) -> dict:
    """Readable facts about a transaction for narration."""
    return {
        "amount": tx["amount"], "category": tx["category"], "guest_checkout": bool(tx["is_guest"]),
        "account_age_days": round(math.expm1(f["account_age_days_log"]), 1),
        "email_age_days": round(math.expm1(f["email_age_days_log"]), 1),
        "ip_country": tx["ip_country"], "billing_country": tx["billing_country"],
        "ip_to_billing_km": round(math.expm1(f["ip_bill_distance_km_log"])),
        "new_device_for_account": bool(f["new_device_for_acct"]),
        "new_shipping_address": bool(f["new_addr_for_acct"]),
        "amount_vs_customer_median": round(f["amt_vs_acct_median"], 2) if f["amt_vs_acct_median"] >= 0 else None,
        "cards_seen_on_device_24h": int(f["dev_distinct_cards_24h"]),
        "accounts_on_device": int(f["dev_distinct_accts_all"]),
        "cards_shipping_to_address_30d": int(f["addr_distinct_cards_30d"]),
        "card_tx_last_hour": int(f["card_tx_1h"]),
        "store_small_orders_10min": int(f["merchant_small_tx_10m"]),
        "linked_to_confirmed_fraud": bool(f["dev_confirmed_fraud"] or f["addr_confirmed_fraud"]
                                          or f["card_confirmed_fraud"] or f["acct_confirmed_fraud"]),
    }


def template_narrative(decision: str, p: float, reasons: list, fx: dict) -> str:
    lead = {"allow": "Approved automatically", "hold": "Held for customer verification",
            "decline": "Declined automatically"}[decision]
    if not reasons:
        return f"{lead}: risk {p:.1%}. Nothing unusual compared with this store's normal traffic."
    why = "; ".join(r["description"] for r in reasons[:3])
    return f"{lead} (risk {p:.1%}, ${fx['amount']:.2f} {fx['category']}). Main drivers: {why}."


def gemini_available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


def gemini(prompt: str, max_tokens=300) -> str | None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.load(r)
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:  # never let narration break the decision path
        return f"(Gemini unavailable: {type(e).__name__})"


def narrate_case(decision, p, reasons, fx):
    prompt = (
        "You are the analyst voice of Sentinel AI, a fraud-prevention system for a small online store. "
        "Explain this automated decision to the store owner in 2-3 short sentences, plain English, no "
        "jargon, no speculation beyond the facts. Do not invent facts. Say what the customer experiences.\n"
        f"Decision: {decision}\nCalibrated fraud probability: {p:.3f}\n"
        f"Top model reason codes (impact = drop in probability if signal were normal): {json.dumps(reasons)}\n"
        f"Transaction facts: {json.dumps(fx)}")
    return gemini(prompt) or template_narrative(decision, p, reasons, fx)


def narrate_report(stats: dict):
    prompt = (
        "Write a concise daily fraud briefing (max 120 words, 3 bullet points + 1 recommendation) for a "
        "small online store owner, based ONLY on these numbers from Sentinel AI. Be specific with numbers.\n"
        + json.dumps(stats))
    return gemini(prompt, 400)
