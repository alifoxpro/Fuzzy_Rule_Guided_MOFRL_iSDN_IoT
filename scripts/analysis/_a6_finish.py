# -*- coding: utf-8 -*-
"""Resume/finish the A6 Dirichlet sweep: parse completed (alpha,seed) results
from the log, run only the missing combos, then aggregate + plot."""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import re, sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'scripts/analysis')
import profile_a6_dirichlet as A6D

LOG = Path('experiments/results_A5run/a6_dirichlet.log')
ds = 'ciciot2023'
ds_dir = Path('experiments/results_A5run') / ds
out_dir = ds_dir / 'a6_dirichlet'
out_dir.mkdir(parents=True, exist_ok=True)

pat = re.compile(r'alpha=([\d.]+) seed=(\d+): FPR HetStd=([\d.]+) HetGap=([\d.]+)'
                 r' \| MSR HetStd=([\d.]+) HetGap=([\d.]+)')
done = {}
for m in pat.finditer(LOG.read_text()):
    a, s = float(m.group(1)), int(m.group(2))
    done[(a, s)] = {'alpha': a, 'seed': s,
                    'FPR_HetStd': float(m.group(3)), 'FPR_HetGap': float(m.group(4)),
                    'MSR_HetStd': float(m.group(5)), 'MSR_HetGap': float(m.group(6))}
print(f'parsed {len(done)} completed combos', flush=True)

missing = [(a, s) for a in A6D.ALPHAS for s in A6D.SEEDS if (a, s) not in done]
print(f'missing combos: {missing}', flush=True)

if missing:
    X, y, sm, emb, fo = A6D.load_pooled(ds, ds_dir)
    for (a, s) in missing:
        done[(a, s)] = A6D.run_alpha_seed(a, s, X, y, sm, emb, fo, out_dir)

agg_rows = []
for a in A6D.ALPHAS:
    seed_rows = [done[(a, s)] for s in A6D.SEEDS]
    agg_rows.append(A6D.aggregate(seed_rows))

json.dump(agg_rows, open(out_dir / 'a6_summary.json', 'w'), indent=2)

a = [r['alpha'] for r in agg_rows]
fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(a, [r['FPR_HetStd_mean'] for r in agg_rows],
            yerr=[r['FPR_HetStd_std'] for r in agg_rows],
            fmt='o-', color='#D1495B', capsize=4, label='FPR HetStd')
ax.errorbar(a, [r['MSR_HetStd_mean'] for r in agg_rows],
            yerr=[r['MSR_HetStd_std'] for r in agg_rows],
            fmt='s-', color='#2C6FBB', capsize=4, label='MSR HetStd')
ax.set_xscale('log')
ax.set_xlabel(r'Dirichlet concentration $\alpha$  (smaller = more non-IID)')
ax.set_ylabel('Client heterogeneity (std across clients)')
ax.set_title(f'A6 client heterogeneity vs non-IID level '
             f'({ds}, mean$\\pm$std over {len(A6D.SEEDS)} seeds)', fontsize=10)
ax.grid(True, which='both', alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / 'a6_dirichlet.png', dpi=150)
fig.savefig(out_dir / 'a6_dirichlet.pdf')
plt.close(fig)
print('A6_FINISH_DONE', flush=True)
