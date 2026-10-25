#!/usr/bin/env python
"""Tier 4: complete the 2D lambda x mu sensitivity sweep on DIAD2024.

Runs the same six (lambda, mu) configurations reported on CICIoT2023
(Table lambda_mu_sweep in the manuscript) on DIAD2024, minus the
canonical (0.25, 0.10) point which already lives in
experiments/results_StrategyB/diad2024/.

Five new invocations:
  (0.10, 0.10)  -- diagonal
  (0.50, 0.10)  -- diagonal
  (0.25, 0.05)  -- off-diagonal
  (0.25, 0.20)  -- off-diagonal
  (0.10, 0.20)  -- off-diagonal

Progress: experiments/tier4_progress.json (resumable).
"""

import argparse, json, subprocess, sys, time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
TRAIN_B = PROJECT / 'scripts' / 'training' / 'train_rl_multiagent_StrategyB.py'
PROGRESS = PROJECT / 'experiments' / 'tier4_progress.json'
MAX_STEPS = 500

QUEUE = [
    dict(id='t4_lam010_mu010_diad', lam=0.10, mu=0.10, suffix='_lam0.10_mu0.10_diad'),
    dict(id='t4_lam050_mu010_diad', lam=0.50, mu=0.10, suffix='_lam0.50_mu0.10_diad'),
    dict(id='t4_lam025_mu005_diad', lam=0.25, mu=0.05, suffix='_lam0.25_mu0.05_diad'),
    dict(id='t4_lam025_mu020_diad', lam=0.25, mu=0.20, suffix='_lam0.25_mu0.20_diad'),
    dict(id='t4_lam010_mu020_diad', lam=0.10, mu=0.20, suffix='_lam0.10_mu0.20_diad'),
]


def load_progress():
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text())
    return {'completed': [], 'failed': [], 'started_at': None}


def save_progress(p):
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(p, indent=2))


def build_cmd(run):
    return [
        sys.executable, str(TRAIN_B),
        '--dataset', 'diad2024',
        '--seed', '42',
        '--max-steps', str(MAX_STEPS),
        '--suffix', run['suffix'],
        '--use-rls',
        '--lambda-fuzzy', str(run['lam']),
        '--mu-link', str(run['mu']),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--only', type=str, default=None)
    args = ap.parse_args()

    prog = load_progress()
    if prog['started_at'] is None:
        prog['started_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        save_progress(prog)

    queue = QUEUE if not args.only else [r for r in QUEUE if r['id'] == args.only]

    print('=' * 72)
    print(f'Tier 4 (lambda x mu on DIAD2024)  |  {len(queue)} runs  |  max_steps={MAX_STEPS}')
    print('=' * 72)

    for i, run in enumerate(queue, 1):
        rid = run['id']
        if rid in prog['completed']:
            print(f'[{i}/{len(queue)}] SKIP {rid}'); continue
        cmd = build_cmd(run)
        print(f'\n[{i}/{len(queue)}] {rid}  lam={run["lam"]}, mu={run["mu"]}')
        print(f'         cmd: {" ".join(cmd)}')
        if args.dry_run: continue
        started = time.time()
        try:
            r = subprocess.run(cmd, cwd=str(PROJECT))
            rc = r.returncode
        except KeyboardInterrupt:
            save_progress(prog); return 130
        elapsed = time.time() - started
        if rc == 0:
            prog['completed'].append(rid)
            prog[rid] = {'status': 'ok', 'seconds': elapsed,
                         'finished_at': time.strftime('%Y-%m-%d %H:%M:%S')}
            print(f'         DONE in {elapsed/60:.1f} min')
        else:
            prog['failed'].append(rid)
            prog[rid] = {'status': 'fail', 'return_code': rc, 'seconds': elapsed}
            print(f'         FAIL rc={rc} after {elapsed/60:.1f} min')
        save_progress(prog)

    print(f'\ncompleted: {len(prog["completed"])}   failed: {len(prog["failed"])}')
    return 0 if not prog['failed'] else 1


if __name__ == '__main__':
    sys.exit(main())
