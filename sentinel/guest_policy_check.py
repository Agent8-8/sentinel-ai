"""Exploratory static-policy comparison after guest-weight experiment.

The seed 61/67/71 test results were previously inspected. Scores use offline
features with the original chargeback stream; this is not a corrected sequential
replay. Both policies use identical simulator assumptions and random seeds.
"""
import json
import numpy as np
from . import policy as P
from .data import ROOT, splits
from .experiments import World
from .features import FEATURES
from .pipeline import SentinelModel

PARAMS=dict(learning_rate=.05,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=1.)
SEEDS=[61,67,71]

def main():
    results={}
    for seed in SEEDS:
        w=World(seed); _,va,te,_,y=splits(w.df); fit=w.ext
        amount=w.amt[te]; ftype=w.ftype[te]; results[str(seed)]={}
        for name,factor in [('original',1.),('guest_unlabelled_3x',3.)]:
            weight=w.w_mat[fit].copy()
            guest=(w.X[fit,FEATURES.index('is_guest')]==1)&(w.y_obs[fit]==0)
            weight *= np.where(guest,factor,1.)
            model=SentinelModel(PARAMS,n_bag=5,novelty_budget=0).fit(w.X[fit],w.y_obs[fit],weight)
            model.calibrate(w.X[va],y[va]); p=model.predict(w.X[te]); costs=P.Costs()
            action=P.decide(p,amount,costs)
            outcome=P.realized_cost(action,y[te],amount,costs,np.random.default_rng(1000+seed),ftype)
            summary=P.summarize(action,outcome,y[te],amount)
            results[str(seed)][name]=summary
            print(seed,name,'cost',summary['total_cost'],'fraud_stopped',summary['fraud_dollar_stop_rate'],
                  'false_decline',summary['false_decline_rate'],'legit_held',summary['legit_friction_rate'],flush=True)
        (ROOT/'reports'/'guest_policy_check.json').write_text(json.dumps({'scope':__doc__, 'per_world':results},indent=2)+'\n')
    fields=['total_cost','fraud_dollar_stop_rate','false_decline_rate','legit_friction_rate']
    means={n:{f:float(np.mean([r[n][f] for r in results.values()])) for f in fields} for n in ['original','guest_unlabelled_3x']}
    (ROOT/'reports'/'guest_policy_check.json').write_text(json.dumps({'scope':__doc__,'per_world':results,'means':means},indent=2)+'\n')
    print('MEANS',json.dumps(means),flush=True)
if __name__=='__main__': main()
