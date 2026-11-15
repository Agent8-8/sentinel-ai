"""Dataset build + leakage-safe temporal splits.

Timeline (days):
  [0, 56)   TRAIN  - labels censored: fraud only if chargeback arrived by day 63
  [56, 63)  GAP    - unused (labels still maturing; prevents boundary leakage)
  [63, 74)  VALID  - model selection, calibration, policy tuning (matured labels)
  [74, 91)  TEST   - touched once for the final report (includes holiday spike
                      and the never-before-seen "sleeper ring" fraud pattern)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURES, build_feature_matrix
from .simulator import DAY, simulate

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data"
TRAIN_END, LABEL_CUTOFF, VALID_START, VALID_END = 56 * DAY, 63 * DAY, 63 * DAY, 74 * DAY


def load(seed: int = 7, rebuild: bool = False):
    CACHE.mkdir(exist_ok=True)
    tx_path, x_path = CACHE / f"tx_{seed}.pkl", CACHE / f"X_{seed}.npy"
    if tx_path.exists() and x_path.exists() and not rebuild:
        return pd.read_pickle(tx_path), np.load(x_path)
    df = simulate(seed)
    X, _ = build_feature_matrix(df)
    df.to_pickle(tx_path)
    np.save(x_path, X)
    return df, X


def splits(df):
    ts = df.ts.to_numpy()
    train = ts < TRAIN_END
    valid = (ts >= VALID_START) & (ts < VALID_END)
    test = ts >= VALID_END
    # what the merchant actually knows at training time
    y_train_observed = (df.chargeback_ts.to_numpy() <= LABEL_CUTOFF) & train
    y_true = df.is_fraud.to_numpy().astype(int)
    return train, valid, test, y_train_observed.astype(int), y_true


__all__ = ["load", "splits", "FEATURES"]
