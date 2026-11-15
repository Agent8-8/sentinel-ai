"""Replay the seed-7 test stream using action-aware chargeback feedback.

This is an independent recomputation of the single-world dashboard policy,
not an external validation. Both the data and verification outcomes remain
simulated. Model training and calibration still use oracle validation labels.
"""
import json
from .api import Live
from .data import ROOT

def main():
    live=Live()
    while live.cursor < len(live.test_idx):
        live.step(min(1000,len(live.test_idx)-live.cursor))
        print('processed',live.cursor,'/',len(live.test_idx),flush=True)
    state=live.state()
    report={'scope':'seed-7 simulated test replay; schedules chargeback only if simulated loss occurred',
            'model_calibration':'uses complete simulated validation labels; not realistic delayed-label calibration',
            'policy':state['kpis'], 'n_test':state['cursor']}
    (ROOT/'reports'/'corrected_replay_seed7.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__': main()
