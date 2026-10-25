# -*- coding: utf-8 -*-
"""
============================================================================
A8 --- Operational Cost profiling
============================================================================
Measures the deployment costs of the federated detector:

  (1) inference latency per window: frozen Mamba encoder forward, PCA
      projection, and DQN policy forward -- single-window (batch=1, the
      online SDN path) and batched (throughput), wall-clock on the GPU.
  (2) communication per round per client: exact parameter payload of the
      federated component (DQN only; the frozen Mamba never leaves the
      client), plus the measured on-disk checkpoint size.
  (3) wall-clock seconds per FL round, recovered from the modification
      times of the per-round global checkpoints saved during the actual
      A5 training runs (no retraining needed).

Run:  python scripts/analysis/profile_a8_cost.py
============================================================================
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import json
import time
from pathlib import Path

import numpy as np
import torch

TRAIN_DIR = Path(__file__).resolve().parent.parent / 'training'
sys.path.insert(0, str(TRAIN_DIR))
import train_rl_multiagent as M

BASE = Path('experiments/results_A5run').resolve()
OUT = BASE / 'a8_cost'
OUT.mkdir(parents=True, exist_ok=True)

DEV = M.DEVICE
W, F = M.WINDOW_SIZE, 14
N_WARM, N_ITER = 200, 2000
BATCH = 512


def time_fn(fn, n_warm=N_WARM, n_iter=N_ITER):
    """Wall-clock a CUDA-synchronized callable; returns (mean_ms, std_ms)."""
    for _ in range(n_warm):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ts = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.array(ts)
    return float(ts.mean()), float(ts.std())


def main():
    print(f'A8 cost profiling | device={DEV}', flush=True)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}', flush=True)

    # ---------------- (1) inference latency ----------------
    mamba_ckpt = Path('experiments/results/ciciot2023/checkpoints/'
                      'Link_1_Core_Mamba_Only.pt')
    extractor = M.MambaEmbeddingExtractor(mamba_ckpt, n_features=F)
    mamba = extractor.model

    agent = M.DQNAgent()
    gp = torch.load(BASE / 'ciciot2023' / 'fedavg' / 'global_dqn.pt',
                    weights_only=False, map_location=DEV)
    agent.set_params(gp)
    dqn = agent.q_net.eval()

    # PCA projection (128 -> 16) as deployed: a single matmul + centering
    pca_W = torch.randn(128, 16, device=DEV)
    pca_mu = torch.randn(128, device=DEV)

    x1 = torch.randn(1, W, F, device=DEV)
    xB = torch.randn(BATCH, W, F, device=DEV)
    s1 = torch.randn(1, M.STATE_DIM, device=DEV)
    sB = torch.randn(BATCH, M.STATE_DIM, device=DEV)
    e1 = torch.randn(1, 128, device=DEV)

    @torch.no_grad()
    def mamba_fwd(x):
        h = mamba.input_proj(x)
        h = mamba.mamba(h)
        emb = h[:, -1, :]
        return torch.softmax(mamba.fc(emb), dim=-1), emb

    @torch.no_grad()
    def pca_fwd(e):
        return (e - pca_mu) @ pca_W

    @torch.no_grad()
    def dqn_fwd(s):
        return dqn(s)

    lat = {}
    lat['mamba_b1_ms'] = time_fn(lambda: mamba_fwd(x1))
    lat['mamba_b512_ms'] = time_fn(lambda: mamba_fwd(xB), n_iter=300)
    lat['pca_b1_ms'] = time_fn(lambda: pca_fwd(e1))
    lat['dqn_b1_ms'] = time_fn(lambda: dqn_fwd(s1))
    lat['dqn_b512_ms'] = time_fn(lambda: dqn_fwd(sB), n_iter=300)
    e2e = lat['mamba_b1_ms'][0] + lat['pca_b1_ms'][0] + lat['dqn_b1_ms'][0]
    lat['end_to_end_b1_ms'] = e2e
    lat['throughput_b512_wps'] = BATCH / (
        (lat['mamba_b512_ms'][0] + lat['dqn_b512_ms'][0]) / 1e3)
    for k, v in lat.items():
        print(f'  {k}: {v}', flush=True)

    # ---------------- (2) communication ----------------
    n_params = sum(p.numel() for p in dqn.parameters())
    payload_up = n_params * 4                       # fp32 bytes, client -> server
    per_round_per_client = 2 * payload_up           # upload + broadcast
    ckpt_file = BASE / 'ciciot2023' / 'fedavg' / 'round_ckpts' / 'global_round_01.pt'
    comm = {
        'dqn_params': int(n_params),
        'payload_one_way_bytes': int(payload_up),
        'per_round_per_client_bytes': int(per_round_per_client),
        'per_round_total_bytes_K5': int(per_round_per_client * 5),
        'full_training_total_bytes_T30_K5': int(per_round_per_client * 5 * 30),
        'checkpoint_file_bytes': ckpt_file.stat().st_size,
        'mamba_params_local_only': int(sum(p.numel() for p in mamba.parameters())),
    }
    for k, v in comm.items():
        print(f'  {k}: {v}', flush=True)

    # ---------------- (3) wall-clock per round from ckpt mtimes ----------------
    rounds = {}
    for ds in ('ciciot2023', 'diad2024'):
        for ph in ('fedavg', 'fedprox'):
            d = BASE / ds / ph / 'round_ckpts'
            files = sorted(d.glob('global_round_*.pt'))
            mt = np.array([f.stat().st_mtime for f in files])
            dt = np.diff(mt)
            med = np.median(dt)
            dt = dt[dt < 3 * med]        # drop restart/idle gaps
            rounds[f'{ds}/{ph}'] = {
                'n_deltas_used': int(len(dt)),
                'sec_per_round_mean': float(dt.mean()),
                'sec_per_round_std': float(dt.std()),
            }
            print(f'  {ds}/{ph}: {dt.mean():.1f} +/- {dt.std():.1f} s/round '
                  f'(n={len(dt)})', flush=True)

    summary = {
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'latency_ms': lat,
        'communication': comm,
        'seconds_per_round': rounds,
        'notes': ('Latency on random inputs (timing is input-independent); '
                  'batch=1 is the online SDN path. Round wall-clock recovered '
                  'from per-round checkpoint mtimes of the real A5 runs; '
                  'deltas >3x median dropped as restart gaps.'),
    }
    with open(OUT / 'a8_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nWrote {OUT / "a8_summary.json"}', flush=True)
    print('A8_DONE', flush=True)


if __name__ == '__main__':
    main()
