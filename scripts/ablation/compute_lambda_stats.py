# -*- coding: utf-8 -*-
"""
============================================================================
Compute Welch t-tests for the lambda ablation.

Reads:  paper4-Fuzzy-Augmented-Reward-Shaping/ablation/lambda_ablation.json
Writes: paper4-Fuzzy-Augmented-Reward-Shaping/ablation/lambda_stats.json
        paper4-Fuzzy-Augmented-Reward-Shaping/ablation/lambda_stats_summary.txt

For each dataset, compares the phase_b4 MSR at every lambda value against
the lambda=0.00 cell (which disables fuzzy shaping entirely and is
therefore our "no-shaping" control condition). Reports:

  t, df, p-value (two-sided Welch)
  Cohen's d
  mean_delta (B - A)
  std_ratio (sigma_0 / sigma_lambda)
  ci_95 (95% CI for the mean difference, assuming n=50)

Then also produces:

  lambda_placeholders.json
    a flat dictionary of {"LAM_CIC_00_MSR": 0.xxxx, ...} values that the
    inject_lambda_results.py script can use to fill in the paper template.
============================================================================
"""
import json
import math
import sys
from pathlib import Path

try:
    from scipy.stats import t as tdist
except ImportError:
    print('ERROR: scipy is required. Install with:  pip install scipy')
    sys.exit(1)

PROJECT = Path(__file__).resolve().parent.parent.parent
ABL_DIR = PROJECT / 'paper4-Fuzzy-Augmented-Reward-Shaping' / 'ablation'

N_EVAL = 50  # evaluation episodes per run (matches the rest of the paper)


def welch(m1, s1, m2, s2, n=N_EVAL):
    """Two-sided Welch t-test and Cohen's d.

    Returns (t, df, p, cohen_d, mean_diff, pooled_sd).
    """
    if s1 is None or s2 is None or m1 is None or m2 is None:
        return None
    se2 = s1 * s1 / n + s2 * s2 / n
    if se2 <= 0:
        return None
    t = (m2 - m1) / math.sqrt(se2)
    num = (s1 * s1 / n + s2 * s2 / n) ** 2
    den_a = (s1 * s1 / n) ** 2 / (n - 1)
    den_b = (s2 * s2 / n) ** 2 / (n - 1)
    df = num / (den_a + den_b) if (den_a + den_b) > 0 else float('inf')
    p = 2 * (1 - tdist.cdf(abs(t), df))
    pooled = math.sqrt((s1 * s1 + s2 * s2) / 2)
    d = (m2 - m1) / pooled if pooled > 0 else 0.0
    return {
        't': t, 'df': df, 'p': p,
        'cohen_d': d,
        'mean_diff': m2 - m1,
        'pooled_sd': pooled,
    }


def load_b4_rows():
    path = ABL_DIR / 'lambda_ablation.json'
    if not path.exists():
        print(f'ERROR: {path} does not exist.')
        print('       Run aggregate_lambda_results.py first.')
        sys.exit(1)
    with open(path, 'r') as f:
        data = json.load(f)
    return [r for r in data['rows'] if r.get('phase') == 'phase_b4']


def main():
    b4_rows = load_b4_rows()
    if not b4_rows:
        print('ERROR: no phase_b4 rows found. The sweep has not been run yet.')
        sys.exit(1)

    # Index by (dataset, lambda)
    by_key = {(r['dataset'], float(r['lambda'])): r for r in b4_rows}

    # Per dataset: find the lambda=0.00 control and compare every other
    # lambda against it.
    results = []  # list of dict rows
    placeholders = {}  # flat dict for template substitution

    datasets = sorted({r['dataset'] for r in b4_rows})

    summary_lines = []
    summary_lines.append('=' * 72)
    summary_lines.append('Lambda ablation statistical tests (phase B4, MSR)')
    summary_lines.append('=' * 72)

    for ds in datasets:
        # Fill placeholders first (MSR, JFI, FPR at each lambda)
        ds_short = 'CIC' if 'ciciot' in ds else 'DIAD'
        lams_here = sorted([lam for (d, lam) in by_key if d == ds])
        for lam in lams_here:
            r = by_key[(ds, lam)]
            tag = f'{ds_short}_{int(round(lam * 100)):02d}'
            placeholders[f'LAM_{tag}_MSR']     = r.get('MSR')
            placeholders[f'LAM_{tag}_MSR_STD'] = r.get('MSR_std')
            placeholders[f'LAM_{tag}_JFI']     = r.get('JFI')
            placeholders[f'LAM_{tag}_FPR']     = r.get('FPR')
            placeholders[f'LAM_{tag}_FNR']     = r.get('FNR')
            placeholders[f'LAM_{tag}_REWARD']  = r.get('Reward')

        # Welch test against lambda=0.00 control
        ctrl = by_key.get((ds, 0.0))
        if ctrl is None:
            summary_lines.append('')
            summary_lines.append(f'{ds}: no lambda=0.00 control, skipping tests')
            continue

        summary_lines.append('')
        summary_lines.append(f'--- {ds} (control: lambda=0.00) ---')
        summary_lines.append(f'{"lambda":>7} {"MSR":>8} {"std":>8}'
                             f' {"delta":>9} {"t":>8} {"p":>11}'
                             f' {"Cohens d":>10} {"sigma_0/sigma":>14}')

        for lam in lams_here:
            r = by_key[(ds, lam)]
            if lam == 0.0:
                summary_lines.append(
                    f'{lam:>7.2f} {r["MSR"]:>8.4f} {r["MSR_std"]:>8.4f}'
                    f' {0.0:>9.4f} {"--":>8} {"(control)":>11}'
                    f' {"--":>10} {"1.00":>14}'
                )
                continue
            st = welch(ctrl['MSR'], ctrl['MSR_std'], r['MSR'], r['MSR_std'])
            if st is None:
                summary_lines.append(f'{lam:>7.2f} (stats unavailable)')
                continue
            sd_ratio = (ctrl['MSR_std'] / r['MSR_std']) if r['MSR_std'] > 0 else float('inf')
            row = {
                'dataset': ds,
                'lambda': lam,
                'MSR': r['MSR'],
                'MSR_std': r['MSR_std'],
                'control_lambda': 0.0,
                'control_MSR': ctrl['MSR'],
                'control_MSR_std': ctrl['MSR_std'],
                't': st['t'],
                'df': st['df'],
                'p': st['p'],
                'cohen_d': st['cohen_d'],
                'mean_diff': st['mean_diff'],
                'sigma_ratio_0_over_lambda': sd_ratio,
            }
            results.append(row)

            # Store stats as placeholders too
            ds_short = 'CIC' if 'ciciot' in ds else 'DIAD'
            tag = f'{ds_short}_{int(round(lam * 100)):02d}_vs_00'
            placeholders[f'STAT_{tag}_T']      = st['t']
            placeholders[f'STAT_{tag}_P']      = st['p']
            placeholders[f'STAT_{tag}_D']      = st['cohen_d']
            placeholders[f'STAT_{tag}_DIFF']   = st['mean_diff']
            placeholders[f'STAT_{tag}_SDRATIO'] = sd_ratio

            summary_lines.append(
                f'{lam:>7.2f} {r["MSR"]:>8.4f} {r["MSR_std"]:>8.4f}'
                f' {st["mean_diff"]:>+9.4f} {st["t"]:>+8.3f}'
                f' {st["p"]:>11.3e}'
                f' {st["cohen_d"]:>+10.3f} {sd_ratio:>14.2f}'
            )

    summary_lines.append('')
    summary_lines.append('=' * 72)
    summary_lines.append(f'Computed {len(results)} tests across {len(datasets)} datasets.')
    summary_lines.append('=' * 72)

    # === Write results ===
    stats_json = ABL_DIR / 'lambda_stats.json'
    with open(stats_json, 'w') as f:
        json.dump({
            'n_per_group': N_EVAL,
            'control_lambda': 0.0,
            'tests': results,
        }, f, indent=2)

    placeholders_json = ABL_DIR / 'lambda_placeholders.json'
    with open(placeholders_json, 'w') as f:
        json.dump(placeholders, f, indent=2)

    summary_path = ABL_DIR / 'lambda_stats_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))

    print('\n'.join(summary_lines))
    print()
    print(f'Wrote: {stats_json}')
    print(f'Wrote: {placeholders_json}')
    print(f'Wrote: {summary_path}')


if __name__ == '__main__':
    main()
