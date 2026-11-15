"""Compare frozen weight choices on three NEW synthetic test worlds.

Selection: guest-unlabelled 3x reduced guest false flags during exploratory validation
with a small average-precision reduction. The original production weight is the
comparison. No new-world test result is used to select either configuration.
Both use the existing five-model ensemble and simulator-truth validation
calibration; these are not independent real-world data.
"""
import json
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score
from .data import ROOT, load, splits, LABEL_CUTOFF
from .models import validation_threshold
from .experiments import World
from .features import FEATURES
from .pipeline import SentinelModel

PARAMS = dict(learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=20, l2_regularization=1.0)
SEEDS = [61, 67, 71]

def main():
    results = {}
    for seed in SEEDS:
        world = World(seed)
        _, va, te, _, y = splits(world.df)
        results[str(seed)] = {}
        for name, exponent in [('original', 1.0), ('guest_unlabelled_3x', 3.0)]:
            fit = world.ext
            weights = world.w_mat[fit].copy()
            guest_unlabelled = (world.X[fit, FEATURES.index('is_guest')] == 1) & (world.y_obs[fit] == 0)
            weights *= np.where(guest_unlabelled, exponent, 1.0)
            model = SentinelModel(PARAMS, n_bag=5, novelty_budget=0).fit(world.X[fit], world.y_obs[fit], weights)
            model.calibrate(world.X[va], y[va])
            pval, ptest = model.predict(world.X[va]), model.predict(world.X[te])
            threshold = validation_threshold(pval, y[va])
            yt = y[te]; flag = ptest > threshold
            genuine = yt == 0; guest = world.X[te, FEATURES.index('is_guest')] == 1
            out = {'ap': float(average_precision_score(yt, ptest)),
                   'brier': float(brier_score_loss(yt, ptest)),
                   'threshold': threshold, 'tp': int((flag & ~genuine).sum()),
                   'fp': int((flag & genuine).sum()), 'fn': int((~flag & ~genuine).sum()),
                   'precision': float(precision_score(yt, flag, zero_division=0)),
                   'recall': float(recall_score(yt, flag, zero_division=0)),
                   'genuine_fpr': float(flag[genuine].mean()),
                   'guest_genuine_n': int((genuine & guest).sum()),
                   'guest_fp': int((flag & genuine & guest).sum()),
                   'nonguest_genuine_n': int((genuine & ~guest).sum()),
                   'nonguest_fp': int((flag & genuine & ~guest).sum())}
            results[str(seed)][name] = out
            print(seed, name, out, flush=True)
        (ROOT/'reports'/'independent_guest_check.json').write_text(json.dumps({'seeds': SEEDS, 'per_world': results}, indent=2)+'\n')
    means = {name: {k: float(np.mean([r[name][k] for r in results.values()])) for k in ['ap','brier','precision','recall','genuine_fpr']} for name in ['original','guest_unlabelled_3x']}
    (ROOT/'reports'/'independent_guest_check.json').write_text(json.dumps({'seeds': SEEDS, 'per_world': results, 'means': means}, indent=2)+'\n')
    print('MEANS',json.dumps(means),flush=True)
if __name__ == '__main__': main()
