"""Sentinel AI live demo server.

Replays the held-out TEST period (unseen by the model) through the real
streaming pipeline: feature engine -> calibrated model -> expected-cost policy
-> simulated step-up verification -> feedback into entity reputation.

    uvicorn sentinel.api:app --port 8000
"""
from __future__ import annotations

import os

# Single-row scoring: OpenMP thread fan-out costs ~20x more than it saves. Must be
# set before numpy/sklearn load, and applies to FastAPI's worker threads too.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import math
import threading
from collections import Counter, deque

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import explain as E
from . import policy as P
from .data import ROOT, VALID_END, load
from .features import FEATURES, FeatureEngine
from .novelty import TailNovelty
from .pipeline import SentinelModel
from .simulator import DAY, HOUR

SEED = 7
app = FastAPI(title="Sentinel AI")
LOCK = threading.RLock()


class Live:
    def __init__(self):
        self.df, self.X = load(SEED)
        self.model = SentinelModel.load(ROOT / "artifacts" / "sentinel_model.pkl")
        self.ref = np.load(ROOT / "artifacts" / "legit_reference.npy")
        self.explainer = E.Explainer(self.model.predict, self.ref)
        hist = (self.df.ts.to_numpy() < VALID_END - 11 * DAY) & np.isnan(self.df.chargeback_ts.to_numpy())
        self.tails = TailNovelty().fit(self.X[hist])  # "how rare is this vs genuine traffic" for novelty holds
        self.costs = P.Costs()
        test = self.df.ts.to_numpy() >= VALID_END
        self.test_idx = np.flatnonzero(test)
        self.p_test_offline = self.model.predict(self.X[test])  # for what-if analysis
        self.nov_test_offline = self.model.is_novel(self.X[test])
        self.reset()

    def reset(self):
        self.engine = FeatureEngine()
        hist = self.df[self.df.ts < VALID_END]
        for tx in hist.to_dict("records"):  # warm up behavioural state on history
            self.engine.process(tx)
            if not math.isnan(tx["chargeback_ts"]):
                self.engine.register_chargeback(tx["chargeback_ts"], tx)
        self.cursor = 0
        self.rng = np.random.default_rng(42)
        self.trusted = set()
        self.log: list[dict] = []
        self.by_id: dict[str, dict] = {}
        self.feats: dict[str, dict] = {}
        self.recent = deque(maxlen=60)
        self.hourly: dict[int, Counter] = {}
        self.tot = Counter()

    def step(self, n):
        rows = self.df.iloc[self.test_idx[self.cursor:self.cursor + n]].to_dict("records")
        for tx in rows:
            f = self.engine.compute(tx)
            x = np.array([f[k] for k in FEATURES])
            p = float(self.model.predict(x)[0])
            ec = P.expected_costs(p, tx["amount"], self.costs)
            a = int(np.argmin(ec))
            novel = bool(self.model.is_novel(x)[0])
            gated = novel and a == P.ALLOW
            if gated:
                a = P.HOLD  # unseen-pattern safety net: verify, never decline
            key = (tx["account_id"], tx["device_id"])
            trusted = False
            if a == P.HOLD and key in self.trusted and p < 0.5:
                a, trusted = P.ALLOW, True
            out = P.realized_cost(np.array([a]), np.array([tx["is_fraud"]]), np.array([tx["amount"]]),
                                  self.costs, self.rng, np.array([tx["fraud_type"]]))
            verification = None
            if a == P.HOLD:
                if out["verified_ok"][0]:
                    verification = "passed"
                    self.trusted.add(key)
                elif out["legit_abandon"][0]:
                    verification = "abandoned"
                else:
                    verification = "failed"
                    self.engine.mark_bad(tx)  # instant feedback: learn from failed step-up
            self.engine.update(tx)
            # Only a fraud loss that occurs can produce this simulated chargeback.
            # Offline benchmark features remain based on the original approve-all stream.
            if out["fraud_loss"][0] and not math.isnan(tx["chargeback_ts"]):
                self.engine.register_chargeback(tx["chargeback_ts"], tx)
            rec = {
                "tx_id": tx["tx_id"], "ts": tx["ts"], "amount": tx["amount"], "category": tx["category"],
                "account": tx["account_id"], "ip_country": tx["ip_country"], "is_guest": bool(tx["is_guest"]),
                "p": round(p, 4), "decision": P.NAMES[a], "trusted_skip": trusted, "novelty_gate": gated, "verification": verification,
                "expected_costs": {"allow": round(float(ec[0]), 2), "hold": round(float(ec[1]), 2),
                                   "decline": round(float(ec[2]), 2)},
                "truth_fraud": bool(tx["is_fraud"]), "truth_type": tx["fraud_type"],
                "fraud_loss": bool(out["fraud_loss"][0]), "false_decline": bool(out["false_decline"][0]),
                "cost": round(float(out["cost"][0]), 2),
            }
            self.log.append(rec)
            self.by_id[rec["tx_id"]] = (tx, rec)
            self.feats[rec["tx_id"]] = f
            self.recent.appendleft(rec)
            h = int(tx["ts"] // HOUR)
            c = self.hourly.setdefault(h, Counter())
            c[rec["decision"]] += 1
            c["fraud"] += rec["truth_fraud"]
            c["fraud_stopped"] += rec["truth_fraud"] and not rec["fraud_loss"]
            t = self.tot
            t["n"] += 1; t[rec["decision"]] += 1; t["trusted_skip"] += trusted; t["novelty_gate"] += gated
            t["fraud"] += tx["is_fraud"]; t["fraud_usd"] += tx["amount"] * tx["is_fraud"]
            t["fraud_stopped"] += tx["is_fraud"] and not rec["fraud_loss"]
            t["fraud_usd_stopped"] += tx["amount"] * (tx["is_fraud"] and not rec["fraud_loss"])
            t["legit"] += not tx["is_fraud"]; t["false_decline"] += rec["false_decline"]
            t["legit_held"] += (a == P.HOLD) and not tx["is_fraud"]
            t["legit_abandon"] += bool(out["legit_abandon"][0])
            t["cost"] += rec["cost"]
            t["cost_approve_all"] += (tx["amount"] + self.costs.chargeback_fee) * tx["is_fraud"]
            t["escalations"] += verification == "passed" and p >= 0.9
            t["volume_usd"] += tx["amount"]
        self.cursor += len(rows)
        return len(rows)

    def state(self):
        t = self.tot
        hours = sorted(self.hourly)[-72:]
        sim_now = self.log[-1]["ts"] if self.log else VALID_END
        return {
            "cursor": self.cursor, "total": len(self.test_idx),
            "sim_day": round(sim_now / DAY, 2),
            "kpis": {
                "transactions": t["n"], "volume_usd": round(t["volume_usd"], 2),
                "allowed": t["allow"], "held": t["hold"], "declined": t["decline"],
                "auto_resolved_pct": round(100 * (1 - t["escalations"] / max(t["n"], 1)), 3),
                "merchant_escalations": t["escalations"],
                "fraud_caught": t["fraud_stopped"], "fraud_total": t["fraud"],
                "fraud_usd_stopped": round(t["fraud_usd_stopped"], 2), "fraud_usd_total": round(t["fraud_usd"], 2),
                "false_decline_rate_pct": round(100 * t["false_decline"] / max(t["legit"], 1), 3),
                "legit_friction_pct": round(100 * t["legit_held"] / max(t["legit"], 1), 3),
                "trusted_skips": t["trusted_skip"], "novelty_holds": t["novelty_gate"],
                "net_savings_usd": round(t["cost_approve_all"] - t["cost"], 2),
                "cost_usd": round(t["cost"], 2), "cost_approve_all_usd": round(t["cost_approve_all"], 2),
            },
            "hourly": [{"hour": h, **{k: self.hourly[h][k] for k in ["allow", "hold", "decline", "fraud", "fraud_stopped"]}}
                       for h in hours],
            "recent": list(self.recent),
            "risk_hist": np.histogram([r["p"] for r in self.log[-3000:]], bins=20, range=(0, 1))[0].tolist() if self.log else [],
        }


LIVE: Live | None = None


def live() -> Live:
    global LIVE
    if LIVE is None:
        with LOCK:
            if LIVE is None:
                LIVE = Live()
    return LIVE


@app.on_event("startup")
def _warm():
    live()  # build behavioural state + load model before the first request


class Tick(BaseModel):
    n: int = 25


class CostsIn(BaseModel):
    margin: float = 0.30
    chargeback_fee: float = 25.0
    churn_cost: float = 20.0
    friction: float = 0.40
    catch_rate: float = 0.85
    abandon: float = 0.07


@app.get("/api/state")
def get_state():
    with LOCK:
        return live().state()


@app.post("/api/tick")
def tick(t: Tick):
    with LOCK:
        L = live()
        L.step(max(1, min(t.n, 2000)))
        return L.state()


@app.post("/api/reset")
def reset():
    with LOCK:
        live().reset()
        return live().state()


@app.get("/api/tx/{tx_id}")
def tx_detail(tx_id: str):
    L = live()
    if tx_id not in L.by_id:
        raise HTTPException(404, "unknown or not yet processed transaction")
    tx, rec = L.by_id[tx_id]
    f = L.feats[tx_id]
    x = np.array([f[k] for k in FEATURES])
    _, reasons = L.explainer.reason_codes(x)
    if rec.get("novelty_gate"):
        rare = "; ".join(f"{f.replace('_log', '').replace('_', ' ')} (rarer than "
                         f"{'99.99' if p < 1e-4 else f'{100 * (1 - p):.2f}'}% of genuine orders)"
                         for f, p in L.tails.top_signals(x, 3))
        reasons = [{"signal": "novelty", "impact": 0.0, "description": "novelty safety net: unusual combination "
                    f"sent to verification. Most unusual: {rare}"}] + reasons
    fx = E.facts(tx, f)
    narrative = E.template_narrative(rec["decision"], rec["p"], reasons, fx)
    return {"record": rec, "reasons": reasons, "facts": fx, "narrative": narrative}


@app.post("/api/whatif")
def whatif(c: CostsIn):
    """Re-run the expected-cost policy on the whole test period with new business costs."""
    L = live()
    costs = P.Costs(**c.model_dump())
    d = L.df.iloc[L.test_idx]
    y, a = d.is_fraud.to_numpy(), d.amount.to_numpy()
    act = P.apply_novelty_gate(P.decide(L.p_test_offline, a, costs), L.nov_test_offline)
    out = P.realized_cost(act, y, a, costs, np.random.default_rng(0), d.fraud_type.to_numpy())
    base = P.realized_cost(np.zeros_like(act), y, a, costs, np.random.default_rng(0))
    s = P.summarize(act, out, y, a)
    s["savings_vs_approve_all"] = float(base["cost"].sum() - out["cost"].sum())
    return s


@app.get("/api/reports")
def reports():
    out = {}
    for name in ["results", "experiments", "public_benchmark"]:
        p = ROOT / "reports" / f"{name}.json"
        if p.exists():
            out[name] = json.loads(p.read_text())
    return out


@app.post("/api/briefing")
def briefing():
    s = live().state()["kpis"]
    text = (f"• Processed {s['transactions']:,} orders (${s['volume_usd']:,.0f}); {s['auto_resolved_pct']}% resolved "
                f"with no merchant action.\n• Stopped {s['fraud_caught']}/{s['fraud_total']} fraudulent orders "
                f"(${s['fraud_usd_stopped']:,.0f} of ${s['fraud_usd_total']:,.0f}).\n"
                f"• False declines: {s['false_decline_rate_pct']}% of genuine customers; "
                f"{s['legit_friction_pct']}% were asked to verify.\n"
                f"Recommendation: net savings vs. approving everything are ${s['net_savings_usd']:,.0f}; keep current settings.")
    return {"text": text}


DASH = ROOT / "dashboard"
app.mount("/static", StaticFiles(directory=DASH), name="static")


@app.get("/")
def index():
    return FileResponse(DASH / "index.html")
