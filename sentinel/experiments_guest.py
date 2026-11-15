"""Exploratory validation-only checks for reducing simulated guest false flags.
These scenarios cannot establish demographic fairness or real-world safety.
"""
import json
import numpy as np
from sklearn.metrics import average_precision_score
from . import models as M
from .data import ROOT
from .experiments import World
from .features import FEATURES

PARAMS = dict(learning_rate=.05,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=1.)
SEEDS = [7,11,23]
CONFIGS = ['original','no_guest_feature','guest_unlabelled_3x','guest_unlabelled_6x']

def main():
    results = {}
    guest_i = FEATURES.index('is_guest')
    for seed in SEEDS:
        w = World(seed); results[str(seed)]={}; fit=w.ext
        for name in CONFIGS:
            cols = [i for i in range(len(FEATURES)) if not (name == 'no_guest_feature' and i == guest_i)]
            weight = w.w_mat[fit].copy()
            if name.startswith('guest_unlabelled_'):
                factor = 3 if name.endswith('3x') else 6
                weight *= np.where((w.X[fit, guest_i] == 1) & (w.y_obs[fit] == 0), factor, 1)
            model = M.hgb(w.X[fit][:,cols],w.y_obs[fit],w=weight,cat_mask=M.cat_mask_for(cols),seed=0,**PARAMS)
            p=model.predict_proba(w.X[w.va][:,cols])[:,1]; y=w.y_true[w.va]; g=w.X[w.va,guest_i] == 1
            threshold=M.validation_threshold(p,y)
            flag=p > threshold
            out={'ap': float(average_precision_score(y,p)), 'fraud_recall': float(flag[y==1].mean()),
                 'guest_genuine_n':int(((y==0)&g).sum()),'guest_fp':int((flag&(y==0)&g).sum()),
                 'nonguest_genuine_n':int(((y==0)&~g).sum()),'nonguest_fp':int((flag&(y==0)&~g).sum())}
            results[str(seed)][name]=out; print(seed,name,out,flush=True)
    summary={}
    for name in CONFIGS:
        rows=[r[name] for r in results.values()]
        summary[name]={'mean_ap':float(np.mean([r['ap'] for r in rows])),
                       'mean_recall':float(np.mean([r['fraud_recall'] for r in rows])),
                       'guest_fp':sum(r['guest_fp'] for r in rows), 'guest_genuine_n':sum(r['guest_genuine_n'] for r in rows),
                       'nonguest_fp':sum(r['nonguest_fp'] for r in rows), 'nonguest_genuine_n':sum(r['nonguest_genuine_n'] for r in rows)}
    report={'scope':'exploratory validation only; same validation labels used to select threshold, no test results; labels include never-reported simulated fraud', 'per_world':results,'summary':summary}
    (ROOT/'reports'/'guest_experiments.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
if __name__ == '__main__': main()
