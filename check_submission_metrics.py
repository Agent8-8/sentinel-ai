"""Report a validation-selected operating point for the saved seed-7 demo model."""
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sentinel.data import ROOT, load, splits
from sentinel.models import validation_threshold
from sentinel.pipeline import SentinelModel

df, X = load(7)
_, valid, test, _, y = splits(df)
model = SentinelModel.load(ROOT / 'artifacts' / 'sentinel_model.pkl')
p_valid = model.predict(X[valid])
threshold = validation_threshold(p_valid, y[valid])
p = model.predict(X[test])
pred = p > threshold
yt = y[test]
tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
print(f'Validation 99.5th percentile threshold: {threshold:.8f}')
print(f'Test TN={tn} FP={fp} FN={fn} TP={tp}')
print(f'Precision={precision_score(yt, pred):.6f}, recall={recall_score(yt, pred):.6f}')
errors = df.loc[test].iloc[np.flatnonzero((yt == 1) & ~pred)]
if len(errors):
    row = errors.iloc[0]
    print(f'First missed synthetic fraud: id={row.tx_id}, type={row.fraud_type}, amount=${row.amount:.2f}')
