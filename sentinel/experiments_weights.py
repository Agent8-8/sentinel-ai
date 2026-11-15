"""Validation-only comparison of delayed-label sample weights and tree settings.

Uses the original development worlds and their validation windows; never scores
those worlds' test periods. The selected configuration is exploratory until
checked on independent simulation worlds and independent real data.
"""
from __future__ import annotations
import json
from collections import defaultdict
import numpy as np
from sklearn.metrics import average_precision_score
from . import models as M
from .data import ROOT
from .experiments import World

SEEDS = [7, 11, 23]
PARAMS = dict(learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=20, l2_regularization=1.0)
CONFIGS = {
    'original': (1., 1., {}),
    'softer_maturity': (.5, 1., {}),
    'stronger_maturity': (2., 1., {}),
    'positive_2x': (1., 2., {}),
    'positive_4x': (1., 4., {}),
    'softer_positive_2x': (.5, 2., {}),
    'leaves_15': (1., 1., {'max_leaf_nodes': 15}),
    'leaf_50': (1., 1., {'min_samples_leaf': 50}),
    'more_regularization': (1., 1., {'l2_regularization': 10.}),
    'slower_learning': (1., 1., {'learning_rate': .025, 'max_iter': 800}),
}

def score(w, spec):
    exponent, pos, extra = spec
    fit = w.ext
    weight = np.where(w.y_obs[fit] == 1, pos, w.w_mat[fit] ** exponent)
    model = M.hgb(w.X[fit], w.y_obs[fit], w=weight,
                  cat_mask=M.cat_mask_for(list(range(w.X.shape[1]))),
                  seed=0, **(PARAMS | extra))
    p = model.predict_proba(w.X[w.va])[:, 1]
    y, amt = w.y_true[w.va], w.amt[w.va]
    return {'ap': float(average_precision_score(y, p)),
            'recall_at_0_5pct_fpr': M.recall_at_fpr(y, p),
            'dollar_recall_at_0_5pct_fpr': M.recall_at_fpr(y, p, weight=amt),
            'iterations': int(model.n_iter_)}

def main():
    results = defaultdict(dict)
    for seed in SEEDS:
        w = World(seed)
        for name, spec in CONFIGS.items():
            results[name][str(seed)] = score(w, spec)
            print(seed, name, results[name][str(seed)], flush=True)
    out = {'scope': 'development validation only; no final test periods scored',
           'seeds': SEEDS, 'configs': {k: {'maturity_exponent': v[0], 'positive_weight': v[1], 'params': PARAMS | v[2]} for k,v in CONFIGS.items()},
           'per_world': results,
           'means': {name: {key: float(np.mean([v[key] for v in worlds.values()])) for key in next(iter(worlds.values()))} for name, worlds in results.items()}}
    (ROOT/'reports'/'weight_experiments.json').write_text(json.dumps(out, indent=2)+'\n')
    print('MEANS', json.dumps(out['means'], indent=2), flush=True)

if __name__ == '__main__':
    main()
