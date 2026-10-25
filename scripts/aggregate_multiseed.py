"""
Aggregate multi-seed results into mean-of-run statistics.

Reads results.json from:
  experiments/results_StrategyB/         (seed 42, canonical)
  experiments/results_StrategyB_s123/    (seed 123)
  experiments/results_StrategyB_s2026/   (seed 2026)

Emits per-dataset per-phase table:
  MSR = mean(mean-per-run), stddev = std(mean-per-run)

Answers reviewer R1-3 / R2-5.
"""
import json, sys, io
import numpy as np
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

REPO = Path(__file__).resolve().parents[1]
LINKS = ['Link_1_Core','Link_2_Aggregation_L','Link_3_Aggregation_R','Link_4_Edge_L','Link_5_Edge_R']

SEED_DIRS = {
    42:   'results_StrategyB',
    123:  'results_StrategyB_s123',
    2026: 'results_StrategyB_s2026',
}

PHASES = ['mosar_independent', 'mosar_fedavg', 'mosar_fedprox', 'fuzzy_retrain']
PHASE_LABEL = {
    'mosar_independent': 'B1 (Independent)',
    'mosar_fedavg':      'B2 (FedAvg)',
    'mosar_fedprox':     'B3 (FedProx)',
    'fuzzy_retrain':     'B4 (Fuzzy)',
}

def link_mean(res, metric='msr'):
    return float(np.mean([res[l][metric] for l in LINKS if l in res]))

def load(seed, ds, phase):
    p = REPO / 'experiments' / SEED_DIRS[seed] / ds / phase / 'results.json'
    if not p.exists(): return None
    return json.load(open(p))

def main():
    for ds in ['ciciot2023', 'diad2024']:
        print(f'\n=== {ds.upper()} — mean over {len([s for s in SEED_DIRS if (REPO/"experiments"/SEED_DIRS[s]/ds).exists()])} available seed(s) ===')
        print(f'{"Phase":22s} {"MSR mean-of-run":>18s} {"MSR std-of-run":>17s} {"n_seeds":>8s}')
        for phase in PHASES:
            run_msrs = []
            for seed in SEED_DIRS:
                r = load(seed, ds, phase)
                if r is None: continue
                run_msrs.append(link_mean(r, 'msr'))
            if not run_msrs:
                print(f'{PHASE_LABEL[phase]:22s} {"--":>18s} {"--":>17s} {"0":>8s}')
                continue
            mu, sd = np.mean(run_msrs), np.std(run_msrs) if len(run_msrs)>1 else 0.0
            print(f'{PHASE_LABEL[phase]:22s} {mu:18.4f} {sd:17.4f} {len(run_msrs):8d}')

    print('\nDone. If n_seeds < 3, additional runs are still needed (Batch A of RUN_ORDER).')

if __name__ == '__main__':
    main()
