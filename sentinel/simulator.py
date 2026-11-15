"""Synthetic merchant world for Sentinel AI.

Simulates 91 days of card-not-present transactions for a small US-based online
store. The point is *entity structure* (accounts, cards, devices, IPs, shipping
addresses) so that behavioural signals exist the way they do in production,
plus the messy parts that make fraud detection hard:

* legitimate customers who travel, buy new phones, ship gifts, splurge, or
  check out as brand-new guests (hard negatives -> false-decline risk)
* a holiday sales spike inside the test window (legit distribution shift)
* chargeback labels that arrive 5-60 days late, and some fraud never reported
* "friendly fraud" (legit-looking purchases later disputed) - undetectable by design
* a brand-new fraud pattern ("sleeper ring") that only starts in the test window

Ground truth `is_fraud` is used for evaluation only. Models train on
`chargeback_ts`-censored labels (what a merchant would actually know).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY = 86400.0
HOUR = 3600.0
SIM_DAYS = 91

# country -> (lat, lon, utc_offset_hours)
GEO = {
    "US": (39.0, -98.0, -6), "CA": (50.0, -100.0, -6), "GB": (53.0, -1.5, 0),
    "DE": (51.0, 10.0, 1), "FR": (46.5, 2.5, 1), "AU": (-27.0, 140.0, 10),
    "MX": (23.0, -102.0, -6), "NG": (9.0, 8.0, 1), "RO": (45.9, 25.0, 2),
    "VN": (16.0, 107.0, 7), "BR": (-10.0, -52.0, -3), "RU": (56.0, 38.0, 3),
    "ID": (-2.0, 118.0, 7), "PH": (12.0, 122.0, 8),
    "NL": (52.3, 4.9, 1), "SG": (1.35, 103.8, 8),
}
CUSTOMER_COUNTRIES = ["US", "CA", "GB", "DE", "FR", "AU", "MX"]
CUSTOMER_COUNTRY_P = [0.80, 0.06, 0.04, 0.03, 0.02, 0.02, 0.03]
FRAUD_COUNTRIES = ["NG", "RO", "VN", "BR", "RU", "ID", "PH"]

CATEGORIES = ["apparel", "home", "beauty", "accessories", "electronics", "gift_cards"]
# lognormal (mu, sigma) of item price per category
PRICE = {
    "apparel": (3.7, 0.5), "home": (3.9, 0.6), "beauty": (3.2, 0.45),
    "accessories": (3.3, 0.6), "electronics": (5.4, 0.6), "gift_cards": (4.0, 0.5),
}
LEGIT_CAT_P = [0.32, 0.20, 0.18, 0.15, 0.10, 0.05]


class IdGen:
    def __init__(self):
        self.n = {}

    def __call__(self, prefix: str) -> str:
        self.n[prefix] = self.n.get(prefix, 0) + 1
        return f"{prefix}{self.n[prefix]:06d}"


def _geo_point(rng, country, spread=4.0):
    lat, lon, _ = GEO[country]
    return lat + rng.normal(0, spread), lon + rng.normal(0, spread * 1.5)


def _price(rng, cat, scale=1.0):
    mu, s = PRICE[cat]
    return float(np.round(np.exp(rng.normal(mu, s)) * scale, 2))


def _local_hour_ts(rng, day, country, peak_hour, spread=3.5):
    """Timestamp on `day` whose *local* hour is around peak_hour."""
    h = (rng.normal(peak_hour, spread)) % 24
    utc_h = (h - GEO[country][2]) % 24
    return day * DAY + utc_h * HOUR + rng.uniform(0, HOUR)


def _demand_multiplier(day):
    """Weekly seasonality + a holiday spike in the test window (days 80-85)."""
    weekly = 1.0 + 0.15 * np.sin(2 * np.pi * day / 7)
    holiday = 2.2 if 80 <= day <= 85 else 1.0
    return weekly * holiday


def simulate(seed: int = 7, n_customers: int = 16000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = IdGen()
    rows: list[dict] = []
    drop_addresses = [ids("addr") for _ in range(25)]  # reshippers used by fraud crews
    drop_country = {a: "US" for a in drop_addresses}

    def add(**kw):
        rows.append(kw)

    # ------------------------------------------------------------------ legit
    customers = []
    for _ in range(n_customers):
        country = rng.choice(CUSTOMER_COUNTRIES, p=CUSTOMER_COUNTRY_P)
        joins_during = rng.random() < 0.18
        created = rng.uniform(0, SIM_DAYS - 3) * DAY if joins_during else -rng.uniform(30, 1500) * DAY
        guest = joins_during and rng.random() < 0.35
        home = _geo_point(rng, country)
        c = dict(
            account=ids("guest") if guest else ids("acct"), guest=guest, country=country,
            created=created, home=home,
            email_age=float(rng.lognormal(6.8, 0.9)) if rng.random() > 0.07 else float(rng.uniform(0, 30)),
            cards=[ids("card") for _ in range(1 + (rng.random() < 0.3))],
            devices=[ids("dev") for _ in range(1 + (rng.random() < 0.45))],
            ips=[ids("ip") for _ in range(1 + rng.integers(0, 3))],
            addr=ids("addr"),
            rate=float(rng.gamma(1.3, 0.045)),  # purchases per day
            peak=float(rng.normal(19, 3)),
            cat_p=rng.dirichlet(np.array(LEGIT_CAT_P) * 12),
            spend=float(rng.lognormal(0, 0.35)),
            travel=None,
            vpn=rng.random() < 0.04,
        )
        if rng.random() < 0.05:  # one trip abroad / cross-border
            start = rng.uniform(0, SIM_DAYS - 10)
            dest = rng.choice([k for k in CUSTOMER_COUNTRIES if k != country] + ["BR", "VN", "PH"])
            c["travel"] = (start * DAY, (start + rng.uniform(3, 10)) * DAY, dest)
        customers.append(c)

    # households share a device; offices / campuses share IPs (legit linkage)
    for i in range(0, int(n_customers * 0.08), 2):
        a, b = customers[i * 7 % n_customers], customers[(i * 7 + 3) % n_customers]
        b["devices"].append(a["devices"][0])
    office_ips = [ids("ip") for _ in range(60)]
    for c in customers:
        if rng.random() < 0.15:
            c["ips"].append(office_ips[rng.integers(60)])

    for c in customers:
        first_day = max(0.0, c["created"] / DAY)
        if c["guest"]:  # guests: 1-2 purchases shortly after "creation"
            n = 1 + (rng.random() < 0.25)
            days = [first_day + rng.uniform(0, 0.2)] + [first_day + rng.uniform(1, 20)] * (n - 1)
        else:
            days = []
            d = int(first_day)
            for day in range(d, SIM_DAYS):
                k = rng.poisson(c["rate"] * _demand_multiplier(day))
                days += [day] * k
            if c["created"] > 0 and (not days or days[0] > first_day + 1):
                days = [first_day + rng.uniform(0, 0.1)] + days  # new signups buy soon
        for day in days:
            ts = _local_hour_ts(rng, int(day), c["country"], c["peak"])
            ts = max(ts, c["created"] + rng.uniform(60, 1800)) if c["created"] > 0 else ts
            if ts >= SIM_DAYS * DAY:
                continue
            country_ip, ip_geo = c["country"], (c["home"][0] + rng.normal(0, 0.8), c["home"][1] + rng.normal(0, 0.8))
            ip = c["ips"][rng.integers(len(c["ips"]))]
            if rng.random() < 0.12:
                ip = ids("ip")  # mobile / coffee-shop IP
            if c["travel"] and c["travel"][0] <= ts <= c["travel"][1]:
                country_ip = c["travel"][2]
                ip_geo = _geo_point(rng, country_ip, 2)
                ip = ids("ip")
            if c["vpn"] and rng.random() < 0.5:
                country_ip = rng.choice(["NL", "SG", "GB", "DE", "RO"])
                ip_geo = _geo_point(rng, country_ip, 1)
                ip = ids("ip")
            if rng.random() < 0.025:
                c["devices"].append(ids("dev"))  # new phone / laptop
            dev = c["devices"][-1] if rng.random() < 0.7 else c["devices"][rng.integers(len(c["devices"]))]
            addr, ship_country = c["addr"], c["country"]
            if rng.random() < 0.06:  # gift shipped elsewhere
                addr = ids("addr")
                ship_country = c["country"] if rng.random() < 0.85 else rng.choice(CUSTOMER_COUNTRIES)
            cat = rng.choice(CATEGORIES, p=c["cat_p"])
            items = 1 + rng.poisson(0.5)
            amount = sum(_price(rng, cat, c["spend"]) for _ in range(items))
            if rng.random() < 0.015:
                amount *= rng.uniform(3, 8)  # occasional splurge
            friendly = rng.random() < 0.0012
            add(ts=ts, account_id=c["account"], is_guest=c["guest"], account_created_ts=c["created"],
                email_age_days=c["email_age"] + max(0, ts / DAY), card_id=c["cards"][rng.integers(len(c["cards"]))],
                card_country=c["country"], billing_country=c["country"],
                billing_lat=c["home"][0], billing_lon=c["home"][1],
                ship_address_id=addr, ship_country=ship_country, device_id=dev, ip_id=ip,
                ip_country=country_ip, ip_lat=ip_geo[0], ip_lon=ip_geo[1],
                category=cat, n_items=items, amount=round(amount, 2),
                is_fraud=friendly, fraud_type="friendly" if friendly else "none")

    established = [c for c in customers if not c["guest"] and c["created"] < -60 * DAY]

    # --------------------------------------------------------- card testing
    day = rng.uniform(0, 4)
    while day < SIM_DAYS - 1:
        start = day * DAY + rng.uniform(0, DAY)
        fc = rng.choice(FRAUD_COUNTRIES + ["US"] * 3)
        bot_devs = [ids("dev") for _ in range(rng.integers(1, 4))]
        stealth = rng.random() < 0.4  # low-and-slow: normal baskets, US proxies, many devices
        if stealth:
            fc = "US"
            bot_devs = [ids("dev") for _ in range(rng.integers(5, 12))]
        n_cards = int(rng.integers(15, 70))
        good_cards = []
        t = start
        for _ in range(n_cards):
            t += rng.exponential(25) if not stealth else rng.exponential(15 * 60)
            card, cc = ids("card"), rng.choice(["US", "US", "US", "CA", "GB"])
            bill = _geo_point(rng, cc)
            ipg = _geo_point(rng, fc, 3)
            acct = ids("guest")
            add(ts=t, account_id=acct, is_guest=True, account_created_ts=t - rng.uniform(5, 60),
                email_age_days=float(rng.uniform(0, 3)), card_id=card, card_country=cc, billing_country=cc,
                billing_lat=bill[0], billing_lon=bill[1], ship_address_id=rng.choice(drop_addresses),
                ship_country="US", device_id=rng.choice(bot_devs), ip_id=ids("ip"), ip_country=fc,
                ip_lat=(bill[0] + rng.normal(0, 1.5)) if stealth else ipg[0],
                ip_lon=(bill[1] + rng.normal(0, 2)) if stealth else ipg[1],
                category=rng.choice(["gift_cards", "accessories", "beauty"]), n_items=1,
                amount=round(float(rng.uniform(8, 35) if stealth else rng.uniform(0.5, 6)), 2),
                email_age_days_override=float(rng.lognormal(5.5, 1)) if stealth else None,
                is_fraud=True, fraud_type="card_testing")
            if rng.random() < 0.08:
                good_cards.append((card, cc, bill))
        for card, cc, bill in good_cards[:5]:  # cash-out on validated cards
            t += rng.uniform(600, 4 * HOUR)
            ipg = _geo_point(rng, fc, 3)
            cat = rng.choice(["electronics", "gift_cards"])
            add(ts=t, account_id=ids("guest"), is_guest=True, account_created_ts=t - rng.uniform(30, 300),
                email_age_days=float(rng.uniform(0, 10)), card_id=card, card_country=cc, billing_country=cc,
                billing_lat=bill[0], billing_lon=bill[1], ship_address_id=rng.choice(drop_addresses),
                ship_country="US", device_id=rng.choice(bot_devs + [ids("dev")]), ip_id=ids("ip"),
                ip_country=fc, ip_lat=ipg[0], ip_lon=ipg[1], category=cat, n_items=1,
                amount=round(_price(rng, cat, 1.6), 2), is_fraud=True, fraud_type="card_testing")
        day += rng.uniform(2.5, 6)

    # ---------------------------------------------------- account takeover
    for _ in range(int(SIM_DAYS * 1.1)):
        c = established[rng.integers(len(established))]
        t = rng.uniform(1, SIM_DAYS - 0.5) * DAY
        proxy = rng.random() < 0.55  # US residential proxy hides geography
        fc = c["country"] if proxy else rng.choice(FRAUD_COUNTRIES)
        dev = ids("dev")
        r = rng.random()
        addr = c["addr"] if r < 0.25 else (rng.choice(drop_addresses) if r < 0.6 else ids("addr"))
        for k in range(int(rng.integers(1, 5))):
            t += rng.uniform(60, 3 * HOUR) if k else 0
            ipg = (c["home"][0] + rng.normal(0, 3), c["home"][1] + rng.normal(0, 4)) if proxy else _geo_point(rng, fc, 3)
            cat = rng.choice(["electronics", "gift_cards", "electronics", "accessories"])
            add(ts=t, account_id=c["account"], is_guest=False, account_created_ts=c["created"],
                email_age_days=c["email_age"] + t / DAY, card_id=c["cards"][0], card_country=c["country"],
                billing_country=c["country"], billing_lat=c["home"][0], billing_lon=c["home"][1],
                ship_address_id=addr, ship_country="US", device_id=dev, ip_id=ids("ip"), ip_country=fc,
                ip_lat=ipg[0], ip_lon=ipg[1], category=cat, n_items=1 + rng.poisson(0.7),
                amount=round(_price(rng, cat, 1.3) * (1 + rng.poisson(0.7)), 2), is_fraud=True, fraud_type="ato")

    # ------------------------------------------ stolen card on new account
    crew_devices = [ids("dev") for _ in range(12)]
    for _ in range(int(SIM_DAYS * 1.6)):
        t = rng.uniform(0, SIM_DAYS - 0.1) * DAY
        cc = rng.choice(["US", "US", "US", "CA", "GB"])
        bill = _geo_point(rng, cc)
        proxy = rng.random() < 0.6
        fc = cc if proxy else rng.choice(FRAUD_COUNTRIES)
        ipg = (bill[0] + rng.normal(0, 1.2), bill[1] + rng.normal(0, 1.5)) if proxy else _geo_point(rng, fc, 5)
        acct = ids("guest") if rng.random() < 0.5 else ids("acct")
        dev = crew_devices[rng.integers(len(crew_devices))] if rng.random() < 0.3 else ids("dev")
        addr = rng.choice(drop_addresses) if rng.random() < 0.55 else ids("addr")
        card = ids("card")
        created = t - rng.uniform(120, 2 * DAY)
        for k in range(int(rng.integers(1, 4))):
            t += rng.uniform(300, 6 * HOUR) if k else 0
            cat = rng.choice(["electronics", "gift_cards", "electronics", "home", "apparel"])
            add(ts=t, account_id=acct, is_guest=acct.startswith("guest"), account_created_ts=created,
                email_age_days=float(rng.uniform(0, 40)) if rng.random() < 0.55 else float(rng.lognormal(6.3, 1)),
                card_id=card, card_country=cc, billing_country=cc, billing_lat=bill[0], billing_lon=bill[1],
                ship_address_id=addr, ship_country="US", device_id=dev, ip_id=ids("ip"), ip_country=fc,
                ip_lat=ipg[0], ip_lon=ipg[1], category=cat, n_items=1 + rng.poisson(0.5),
                amount=round(_price(rng, cat, 1.1) * (1 + rng.poisson(0.5)), 2), is_fraud=True,
                fraud_type="stolen_card")

    # ---------------- NOVEL pattern, test window only: "sleeper ring"
    # Aged accounts + aged emails, US residential IPs near billing, normal hours,
    # modest amounts, one purchase per card. Only linkage reveals it: a few
    # shared device fingerprints and two fresh drop addresses.
    ring_devices = [ids("dev") for _ in range(4)]
    ring_addrs = [ids("addr") for _ in range(3)]
    for _ in range(140):
        t = rng.uniform(75, SIM_DAYS - 0.2) * DAY
        cc = "US"
        bill = _geo_point(rng, cc)
        acct = ids("acct")
        cat = rng.choice(["electronics", "home", "apparel", "accessories"])
        add(ts=_local_hour_ts(rng, int(t / DAY), "US", 18), account_id=acct, is_guest=False,
            account_created_ts=t - rng.uniform(20, 70) * DAY, email_age_days=float(rng.lognormal(6.3, 0.6)),
            card_id=ids("card"), card_country=cc, billing_country=cc, billing_lat=bill[0], billing_lon=bill[1],
            ship_address_id=ring_addrs[rng.integers(3)], ship_country="US",
            device_id=ring_devices[rng.integers(4)], ip_id=ids("ip"), ip_country="US",
            ip_lat=bill[0] + rng.normal(0, 0.6), ip_lon=bill[1] + rng.normal(0, 0.6), category=cat,
            n_items=1, amount=round(float(rng.uniform(120, 420)), 2), is_fraud=True, fraud_type="sleeper_ring")

    df = pd.DataFrame(rows)
    if "email_age_days_override" in df:
        o = df.pop("email_age_days_override")
        df["email_age_days"] = np.where(o.notna(), o, df.email_age_days)
    df = df[(df.ts >= 0) & (df.ts < SIM_DAYS * DAY)].sort_values("ts", kind="stable").reset_index(drop=True)
    df.insert(0, "tx_id", [f"tx{i:07d}" for i in range(len(df))])

    # Chargeback (label) arrival. Some fraud is never reported.
    n = len(df)
    delay = (5 + rng.gamma(2.0, 9.0, n)) * DAY
    reported = df.is_fraud.to_numpy() & (rng.random(n) < np.where(df.fraud_type == "card_testing", 0.55, 0.93))
    df["chargeback_ts"] = np.where(reported, df.ts + delay, np.nan)
    return df


if __name__ == "__main__":
    d = simulate()
    print(d.shape, d.is_fraud.mean())
    print(d.fraud_type.value_counts())
