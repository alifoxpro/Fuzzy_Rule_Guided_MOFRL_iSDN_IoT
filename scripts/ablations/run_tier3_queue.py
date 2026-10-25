#!/usr/bin/env python
"""Tier 3 experiment queue for the Major-revision closure.

Runs seven training invocations back-to-back with resume support:
  * 2 runs to complete a strict 5-seed protocol per benchmark
  * 2 runs for ablation cell (a): conventional scalar reward
  * 3 runs for the 2D lambda x mu preliminary sweep on CICIoT2023

Progress log: experiments/tier3_progress.json
Usage:
    python scripts/ablations/run_tier3_queue.py
    python scripts/ablations/run_tier3_queue.py --only <run_id>
    python scripts/ablations/run_tier3_queue.py --dry-run
"""

import argparse, json, subprocess, sys, time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
TRAIN_B = PROJECT / 'scripts' / 'training' / 'train_rl_multiagent_StrategyB.py'
PROGRESS = PROJECT / 'experiments' / 'tier3_progress.json'

QUEUE = [
    # -- Recover the lost seed_7 CICIoT run + add the missing seed_2026 DIAD --
    dict(id='t3_seed7_ciciot_recover', dataset='ciciot2023', seed=7,   suffix='_seed_7',
         lam=None, mu=None, scalar=False, tier='T3 seed recovery', use_rls=True),
    dict(id='t3_seed2026_diad',        dataset='diad2024',   seed=2026, suffix='_seed_2026',
         lam=None, mu=None, scalar=False, tier='T3 seed 2026 DIAD', use_rls=True),
    # -- Ablation cell (a): conventional scalar reward, no MOSAR --
    dict(id='t3_cellA_ciciot', dataset='ciciot2023', seed=42, suffix='_cell_a_scalar',
         lam=0.0, mu=0.0, scalar=True, tier='T3 cell a scalar', use_rls=True),
    dict(id='t3_cellA_diad',   dataset='diad2024',   seed=42, suffix='_cell_a_scalar',
         lam=0.0, mu=0.0, scalar=True, tier='T3 cell a scalar', use_rls=True),
    # -- 2D lambda x mu preliminary sweep on CICIoT (3 off-diagonal points) --
    dict(id='t3_lam025_mu005_ciciot', dataset='ciciot2023', seed=42, suffix='_lam0.25_mu0.05',
         lam=0.25, mu=0.05, scalar=False, tier='T3 lambda x mu sweep', use_rls=True),
    dict(id='t3_lam025_mu020_ciciot', dataset='ciciot2023', seed=42, suffix='_lam0.25_mu0.20',
         lam=0.25, mu=0.20, scalar=False, tier='T3 lambda x mu sweep', use_rls=True),
    dict(id='t3_lam010_mu020_ciciot', dataset='ciciot2023', seed=42, suffix='_lam0.10_mu0.20',
         lam=0.10, mu=0.20, scalar=False, tier='T3 lambda x mu sweep', use_rls=True),
]

MAX_STEPS = 500


def load_progress():
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text())
    return {'completed': [], 'failed': [], 'started_at': None}


def save_progress(p):
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(p, indent=2))


def build_cmd(run):
    cmd = [sys.executable, str(TRAIN_B),
           '--dataset', run['dataset'],
           '--seed', str(run['seed']),
           '--max-steps', str(MAX_STEPS),
           '--suffix', run['suffix']]
    if run.get('use_rls'):
        cmd += ['--use-rls']
    if run.get('lam') is not None:
        cmd += ['--lambda-fuzzy', str(run['lam'])]
    if run.get('mu') is not None:
        cmd += ['--mu-link', str(run['mu'])]
    if run.get('scalar'):
        cmd += ['--scalar-reward']
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--only', type=str, default=None)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    prog = load_progress()
    if prog['started_at'] is None:
        prog['started_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        save_progress(prog)

    queue = QUEUE if not args.only else [r for r in QUEUE if r['id'] == args.only]
    if args.only and not queue:
        print(f'[queue] no run matches --only {args.only!r}')
        print('        available ids:', [r['id'] for r in QUEUE])
        return 2

    print('=' * 72)
    print(f'Tier 3 queue  |  {len(queue)} runs  |  max_steps={MAX_STEPS}')
    print(f'Progress file: {PROGRESS}')
    print('=' * 72)

    for i, run in enumerate(queue, 1):
        rid = run['id']
        if not args.force and rid in prog['completed']:
            print(f'[{i}/{len(queue)}] SKIP (already done) {rid}')
            continue

        cmd = build_cmd(run)
        pretty = ' '.join(cmd)
        print(f'\n[{i}/{len(queue)}] {run["tier"]:30s}  id={rid}')
        print(f'         cmd: {pretty}')
        if args.dry_run:
            continue

        started = time.time()
        try:
            r = subprocess.run(cmd, cwd=str(PROJECT))
            rc = r.returncode
        except KeyboardInterrupt:
            print('[queue] interrupted; progress saved.')
            save_progress(prog)
            return 130
        elapsed = time.time() - started

        if rc == 0:
            prog['completed'].append(rid)
            prog[rid] = {'status': 'ok', 'seconds': elapsed,
                         'finished_at': time.strftime('%Y-%m-%d %H:%M:%S')}
            print(f'         DONE in {elapsed/60:.1f} min')
        else:
            prog['failed'].append(rid)
            prog[rid] = {'status': 'fail', 'return_code': rc,
                         'seconds': elapsed,
                         'finished_at': time.strftime('%Y-%m-%d %H:%M:%S')}
            print(f'         FAIL rc={rc} after {elapsed/60:.1f} min')
        save_progress(prog)

    print('\n' + '=' * 72)
    print(f'completed: {len(prog["completed"])}   failed: {len(prog["failed"])}')
    if prog['failed']:
        print(f'  failed ids: {prog["failed"]}')
    return 0 if not prog['failed'] else 1


if __name__ == '__main__':
    sys.exit(main())
