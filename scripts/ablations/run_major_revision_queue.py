#!/usr/bin/env python
"""
Major-revision experiment queue: Tier 1 (multi-seed) + Tier 2 (ablation cells).

Runs eight training invocations back-to-back with resume support. Each
invocation runs the full B1/B2/B3/B4 pipeline once for one dataset and one
configuration, and writes its artefacts to
    experiments/results_StrategyB<suffix>/
The queue keeps its own progress log at
    experiments/major_revision_progress.json
so re-launching the script after a crash or reboot skips completed runs.

Usage:
    python scripts/ablations/run_major_revision_queue.py
    python scripts/ablations/run_major_revision_queue.py --dry-run
    python scripts/ablations/run_major_revision_queue.py --only <run_id>
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
TRAIN_B = PROJECT / 'scripts' / 'training' / 'train_rl_multiagent_StrategyB.py'
PROGRESS = PROJECT / 'experiments' / 'major_revision_progress.json'

# Queue in priority order — Tier 1 seeds first, Tier 2 ablations next.
QUEUE = [
    # Tier 1: two new seeds × two datasets
    dict(id='t1_seed7_ciciot',   dataset='ciciot2023', seed=7,   suffix='_seed_7',
         lam=None, mu=None, tier='T1 multi-seed', use_rls=True),
    dict(id='t1_seed7_diad',     dataset='diad2024',   seed=7,   suffix='_seed_7',
         lam=None, mu=None, tier='T1 multi-seed', use_rls=True),
    dict(id='t1_seed314_ciciot', dataset='ciciot2023', seed=314, suffix='_seed_314',
         lam=None, mu=None, tier='T1 multi-seed', use_rls=True),
    dict(id='t1_seed314_diad',   dataset='diad2024',   seed=314, suffix='_seed_314',
         lam=None, mu=None, tier='T1 multi-seed', use_rls=True),
    # Tier 2: cells (b) MOSAR-only and (d) MOSAR + AFRAS-FL
    dict(id='t2_cellB_ciciot', dataset='ciciot2023', seed=42, suffix='_cell_b_mosar_only',
         lam=0.0, mu=0.0, tier='T2 cell b MOSAR-only', use_rls=True),
    dict(id='t2_cellB_diad',   dataset='diad2024',   seed=42, suffix='_cell_b_mosar_only',
         lam=0.0, mu=0.0, tier='T2 cell b MOSAR-only', use_rls=True),
    dict(id='t2_cellD_ciciot', dataset='ciciot2023', seed=42, suffix='_cell_d_mosar_afras',
         lam=0.25, mu=0.0, tier='T2 cell d MOSAR+AFRAS-FL', use_rls=True),
    dict(id='t2_cellD_diad',   dataset='diad2024',   seed=42, suffix='_cell_d_mosar_afras',
         lam=0.25, mu=0.0, tier='T2 cell d MOSAR+AFRAS-FL', use_rls=True),
]

MAX_STEPS = 500  # matches recent runs; keeps each invocation ~90-120 min


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
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--only', type=str, default=None,
                    help='Run only the given run id and exit.')
    ap.add_argument('--force', action='store_true',
                    help='Re-run even if marked completed.')
    args = ap.parse_args()

    prog = load_progress()
    if prog['started_at'] is None:
        prog['started_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        save_progress(prog)

    queue = QUEUE
    if args.only:
        queue = [r for r in QUEUE if r['id'] == args.only]
        if not queue:
            print(f'[queue] no run matches --only {args.only!r}')
            print('        available ids:', [r['id'] for r in QUEUE])
            return 2

    print('=' * 72)
    print(f'Major-revision queue  |  {len(queue)} runs  |  max_steps={MAX_STEPS}')
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
            print('[queue] interrupted by user; progress saved.')
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
