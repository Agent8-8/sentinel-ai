"""Exploratory seed-7 customer-situation audit using a validation-fixed cutoff.

Synthetic groups are behavioral scenarios, not protected demographic groups.
Fraud outcomes and fully observed validation labels are simulator ground truth.
"""
import json
import numpy as np
from .data import ROOT, load, splits
from .models import validation_threshold
from .pipeline import SentinelModel
from .features import FEATURES

def main():
    df, X = load(7)
    _, va, te, _, y = splits(df)
    model = SentinelModel.load(ROOT/'artifacts'/'sentinel_model.pkl')
    pv = model.predict(X[va]); pt = model.predict(X[te]); yt = y[te]
    threshold = validation_threshold(pv, y[va])
    x = X[te]
    groups = {'guest': x[:, FEATURES.index('is_guest')] == 1,
              'non_guest': x[:, FEATURES.index('is_guest')] == 0,
              'new_device': x[:, FEATURES.index('new_device_for_acct')] == 1,
              'known_device': x[:, FEATURES.index('new_device_for_acct')] == 0,
              'ip_country_mismatch': x[:, FEATURES.index('ip_bill_country_mismatch')] == 1,
              'ip_country_match': x[:, FEATURES.index('ip_bill_country_mismatch')] == 0}
    rows = {}
    for name, mask in groups.items():
        good = mask & (yt == 0); bad = mask & (yt == 1); flag = pt > threshold
        rows[name] = {'genuine_n': int(good.sum()), 'genuine_flagged': int((good & flag).sum()),
                      'genuine_flag_rate': float(flag[good].mean()) if good.any() else None,
                      'fraud_n': int(bad.sum()), 'fraud_flagged': int((bad & flag).sum()),
                      'fraud_recall': float(flag[bad].mean()) if bad.any() else None}
    report = {'seed': 7, 'threshold': threshold, 'threshold_method': '99.5th percentile of genuine validation scores; strict greater than',
              'scope': 'simulated behavioral scenarios, not demographics; analysis on previously inspected test period', 'groups': rows}
    (ROOT/'reports'/'scenario_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
if __name__ == '__main__': main()
