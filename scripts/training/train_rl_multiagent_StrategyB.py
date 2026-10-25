# -*- coding: utf-8 -*-
"""
============================================================================
FL-RLMAMS Phase 4 — Strategy B: MOSAR + AFRAS-FL + Fuzzy-Augmented RL
============================================================================
Extends Strategy A with 3 innovations:
  Innovation 1: MOSAR Reward (5 multi-objective components)
  Innovation 3: AFRAS-FL Pipeline (7 phases → fuzzy IF-THEN rules)
  Innovation 4: Fuzzy-Augmented Re-training (R_total = R_MOSAR + R_fuzzy + R_link)

Training Phases:
  Phase B0: AFRAS-FL Pipeline (7 sub-phases)
  Phase B1: Independent DQN + MOSAR Reward
  Phase B2: FL-DQN with FedAvg + MOSAR
  Phase B3: FL-DQN with FedProx + MOSAR
  Phase B4: Fuzzy-Augmented Re-training (R_MOSAR + R_fuzzy + R_link)
  Phase B5: Master Agent Load Balancing

Usage:
  python train_rl_multiagent_StrategyB.py --dataset ciciot2023
  python train_rl_multiagent_StrategyB.py --dataset diad2024
  python train_rl_multiagent_StrategyB.py --dataset all
============================================================================
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

import sys, io, json, time, warnings, math, argparse, gc, copy, random
from pathlib import Path
from collections import Counter, deque

import numpy as np
import pandas as pd

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except AttributeError:
    pass

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# seaborn is imported for styling only; it is never called in this file,
# so we degrade gracefully if it is not installed (some minimal envs
# used for the lambda ablation do not ship it).
try:
    import seaborn as sns  # noqa: F401
except ImportError:
    sns = None

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.spatial import KDTree
from scipy.interpolate import BSpline

# R-Tree (rtree package — fallback to brute-force if not installed)
try:
    from rtree import index as rtree_index
    HAS_RTREE = True
except ImportError:
    HAS_RTREE = False

# pywt for wavelet denoising
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

# Import from train_isolated.py (same project)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_isolated import (
    MambaOnlyModel, MambaBlock, SelectiveSSM,
    WindowedFlowDataset, create_windows,
    seed_everything, count_params,
    SEED, WINDOW_SIZE, STRIDE, BATCH_SIZE, NUM_CLASSES, DEVICE,
    SEV_MAP, SEV_NAMES,
)

# Record-Level Split loader (leakage-free preprocessing; reviewer R2-6).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'data'))
from load_link_rls import load_link_rls  # noqa: E402

# Runtime toggles for the record-level split (RLS) data path. Defaults
# preserve the legacy flow-level behaviour; --use-rls flips them on.
USE_RLS = False
RLS_ROOT = None       # set by --rls-root, defaults to data/unified_14features_5links_rls
RLS_SPLIT = 'train'   # which split feeds RL training when USE_RLS is on


# ============================================================================
# Configuration
# ============================================================================
PROJECT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT / 'data'
# RESULTS_BASE can be overridden with the RESULTS_SUFFIX env var so that
# ablation runs write to e.g. results_StrategyB_lam0.10 without overwriting
# the canonical results_StrategyB folder.
_SUFFIX = os.environ.get('RESULTS_SUFFIX', '')
RESULTS_BASE = PROJECT / 'experiments' / f'results_StrategyB{_SUFFIX}'
STRATEGY_A_RESULTS = PROJECT / 'experiments' / 'results_StrategyA'

MAX_ROWS = 3_000_000

# DQN Hyperparameters (same as Strategy A for fair comparison)
STATE_DIM = 28
N_ACTIONS = 3
DQN_HIDDEN = [128, 64, 32]
DQN_LR = 1e-3
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BUFFER_SIZE = 10000
DQN_BATCH = 64
TAU = 0.005
MAX_STEPS_PER_EPISODE = 5000

# Training
INDEPENDENT_EPISODES = 100
FL_ROUNDS = 30
LOCAL_EPISODES_PER_ROUND = 5
FEDPROX_MU = 0.01
BLEND_ALPHA = 0.7

# MOSAR Reward Weights (Innovation 1)
MOSAR_W_DETECTION = 0.30
MOSAR_W_ACCURACY = 0.20
MOSAR_W_LATENCY = 0.15
MOSAR_W_CONFIDENCE = 0.15
MOSAR_W_IMPACT = 0.20

# Fuzzy-Augmented Weights (Innovation 4) — can be overridden via CLI
# (see main() for the argparse overrides and the RESULTS_BASE suffix)
LAMBDA_FUZZY = float(os.environ.get('LAMBDA_FUZZY', 0.25))
MU_LINK = float(os.environ.get('MU_LINK', 0.10))

# Ablation cell (a): conventional scalar reward. When SCALAR_REWARD is True,
# the MOSAR five-component decomposition is replaced by a single scalar
# reward = R_detection (+1 correct / -0.5 wrong). LAMBDA_FUZZY and MU_LINK
# are forced to zero so nothing else contributes.
SCALAR_REWARD = False

# --- PBRS (Potential-Based Reward Shaping) baseline ------------------------
# When USE_PBRS is True, Phase B4 replaces the fuzzy-agreement bonus
#   R_fuzzy = fuzzy_engine(s, a)
# with a standard PBRS shaping term
#   F(s, s') = gamma_RL * Phi(s') - Phi(s)
# where Phi(s) = mamba_softmax[s][attack_class] is the state-only attack
# probability estimated by the Mamba backbone. LAMBDA_PBRS scales the term
# and defaults to LAMBDA_FUZZY so B4 uses R_MOSAR + LAMBDA_PBRS * F + MU_LINK
# * R_link. See paper Section L3 / R13 (PBRS baseline).
USE_PBRS   = int(os.environ.get('USE_PBRS', 0)) == 1
LAMBDA_PBRS = float(os.environ.get('LAMBDA_PBRS', LAMBDA_FUZZY))
PBRS_GAMMA  = float(os.environ.get('PBRS_GAMMA', 0.99))
PBRS_ATTACK_CLASS = int(os.environ.get('PBRS_ATTACK_CLASS', 0))  # 0=High-sev

# AFRAS-FL Configuration
AFRAS_N_FUZZY_SETS = 3          # Low, Medium, High membership functions
AFRAS_MIN_RULE_CONFIDENCE = 0.5
AFRAS_MIN_RULE_SUPPORT = 50
AFRAS_GRID_COARSE = 5           # Coarse grid divisions
AFRAS_GRID_FINE = 10            # Fine grid divisions
AFRAS_DENSITY_THRESHOLD = 0.3   # Below = sparse → coarse grid

ACTION_NAMES = ['ALLOW', 'RATE_LIMIT', 'BLOCK']
LINK_NAMES = [
    'Link_1_Core', 'Link_2_Aggregation_L', 'Link_3_Aggregation_R',
    'Link_4_Edge_L', 'Link_5_Edge_R'
]
LINK_SHORT = ['L1-Core', 'L2-AggrL', 'L3-AggrR', 'L4-EdgeL', 'L5-EdgeR']

DATASETS = {
    'ciciot2023': {
        'data_dir': DATA_DIR / 'unified_14features_5links' / 'ciciot2023',
        'json_file': DATA_DIR / 'unified_selected_14.json',
    },
    'diad2024': {
        'data_dir': DATA_DIR / 'unified_14features_5links' / 'diad2024',
        'json_file': DATA_DIR / 'unified_selected_14.json',
    },
}

DS_LABELS = {'ciciot2023': 'CICIoT2023', 'diad2024': 'DIAD2024'}

# Phase colors for charts
PHASE_COLORS = {
    'Independent': '#E74C3C', 'FedAvg': '#3498DB',
    'FedProx': '#27AE60', 'FuzzyRetrain': '#9B59B6',
}
PHASE_HATCHES = {
    'Independent': '///', 'FedAvg': '...',
    'FedProx': 'xxx', 'FuzzyRetrain': '\\\\\\',
}
LINK_COLORS = ['#E74C3C', '#3498DB', '#27AE60', '#F39C12', '#9B59B6']
ACTION_COLORS = ['#27AE60', '#F39C12', '#E74C3C']
FUZZY_SET_COLORS = ['#3498DB', '#F39C12', '#E74C3C']

# IEEE Academic Style
plt.rcParams.update({
    'font.size': 10, 'font.family': 'serif',
    'axes.labelsize': 11, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
})
IEEE_W1 = 3.5
IEEE_W2 = 7.16

CKPT_EVERY_EP = 10
CKPT_EVERY_ROUND = 5


# ============================================================================
# Logging
# ============================================================================
_log_file = None

def log(msg=''):
    sys.stdout.write(str(msg) + '\n')
    try:
        sys.stdout.flush()
    except Exception:
        pass
    if _log_file:
        os.makedirs(_log_file.parent, exist_ok=True)
        with open(_log_file, 'a', encoding='utf-8') as f:
            f.write(str(msg) + '\n')

def fmt(n):
    return f'{n:,}'


# ============================================================================
# Progress Tracking (Resume)
# ============================================================================
def load_progress(progress_file):
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {'completed': [], 'results': {}}

def save_progress(state, progress_file):
    os.makedirs(progress_file.parent, exist_ok=True)
    tmp = str(progress_file) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    if progress_file.exists():
        os.remove(progress_file)
    os.rename(tmp, progress_file)


# ============================================================================
# Data Loading (same as Strategy A)
# ============================================================================
def load_link_csv(csv_path, feature_order, dataset_key=None, link_name=None):
    """Load features + severity for one link.

    When USE_RLS is on, dispatches to the record-level split loader
    (leakage-free preprocessing per reviewer R2-6). Otherwise falls
    back to the legacy flow-level CSV loader that fits its scaler on
    the concatenated data downstream.
    """
    if USE_RLS:
        if dataset_key is None or link_name is None:
            raise ValueError(
                'load_link_csv called with USE_RLS=True but dataset_key or '
                'link_name was not provided. Update the caller to pass both.')
        log(f'    [RLS/{RLS_SPLIT}] Loading {dataset_key}/{link_name}...')
        X, y = load_link_rls(dataset=dataset_key,
                             link_name=link_name,
                             feature_order=feature_order,
                             split=RLS_SPLIT,
                             rls_root=RLS_ROOT)
        # Subsample AFTER the split, still stratified, so the reduction
        # matches the legacy MAX_ROWS behaviour without re-introducing
        # leakage across the train/val/test boundary.
        if len(y) > MAX_ROWS:
            log(f'      RLS rows={len(y):,} > MAX_ROWS={MAX_ROWS:,}; stratified subsample.')
            idx_all = np.arange(len(y))
            idx_keep, _ = train_test_split(
                idx_all, train_size=MAX_ROWS,
                random_state=SEED, stratify=y)
            X = X[idx_keep]
            y = y[idx_keep]
        log(f'      Rows: {fmt(len(y))}  '
            f'Class dist: H={np.sum(y==0):,} M={np.sum(y==1):,} L={np.sum(y==2):,}')
        return X, y

    log(f'    Loading {csv_path.name}...')
    needed_cols = list(feature_order) + ['severity']
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    usecols = [c for c in needed_cols if c in header]
    chunks = []
    total_rows = 0
    for chunk in pd.read_csv(csv_path, usecols=usecols, low_memory=True,
                             chunksize=500_000):
        chunks.append(chunk)
        total_rows += len(chunk)
        if total_rows >= MAX_ROWS * 2:
            break
    df = pd.concat(chunks, ignore_index=True)
    del chunks; gc.collect()
    total_rows = len(df)
    log(f'      Total rows: {fmt(total_rows)}')
    if total_rows > MAX_ROWS:
        log(f'      Subsampling to {fmt(MAX_ROWS)} rows...')
        df, _ = train_test_split(df, train_size=MAX_ROWS, random_state=SEED,
                                  stratify=df['severity'])
        df = df.reset_index(drop=True)
    available = [f for f in feature_order if f in df.columns]
    X = df[available].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y_str = df['severity'].values
    y = np.array([SEV_MAP.get(s, 2) for s in y_str], dtype=np.int64)
    del df; gc.collect()
    log(f'      Class dist: H={np.sum(y==0):,} M={np.sum(y==1):,} L={np.sum(y==2):,}')
    return X, y


# ============================================================================
# Mamba Embedding Extractor (frozen — reused from Strategy A)
# ============================================================================
class MambaEmbeddingExtractor:
    def __init__(self, checkpoint_path, n_features=14, device=DEVICE):
        self.device = device
        self.model = MambaOnlyModel(n_features=n_features, n_classes=NUM_CLASSES).to(device)
        state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.pca = None
        log(f'      Loaded Mamba checkpoint: {Path(checkpoint_path).name}')

    @torch.no_grad()
    def extract_all(self, X_windows):
        all_softmax, all_embeds = [], []
        for i in range(0, len(X_windows), 512):
            batch = torch.FloatTensor(X_windows[i:i+512]).to(self.device)
            x = self.model.input_proj(batch)
            x = self.model.mamba(x)
            embed = x[:, -1, :]
            logits = self.model.fc(embed)
            probs = F.softmax(logits, dim=-1)
            all_softmax.append(probs.cpu().numpy())
            all_embeds.append(embed.cpu().numpy())
        return np.concatenate(all_softmax), np.concatenate(all_embeds)

    def fit_pca(self, embeddings, n_components=16):
        self.pca = PCA(n_components=n_components, random_state=SEED)
        self.pca.fit(embeddings)
        log(f'      PCA fitted: 128D -> {n_components}D (explained: {self.pca.explained_variance_ratio_.sum():.3f})')

    def transform_pca(self, embeddings):
        return self.pca.transform(embeddings).astype(np.float32)


# ============================================================================
# ██████████████████████████████████████████████████████████████████████████████
# INNOVATION 3: AFRAS-FL PIPELINE (7 Phases)
# ██████████████████████████████████████████████████████████████████████████████
# ============================================================================

class AFRASFLPipeline:
    """
    Adaptive Fuzzy Rule-based Aggregation System for Federated Learning.
    7-phase pipeline: Denoise → Balance → Boundary → Grid → Rules → Organize → Deploy
    """

    def __init__(self, X_2d, y, feature_names, link_name, cache_dir):
        """
        X_2d: (N, 2) — top 2 XAI-selected features for this link
        y: (N,) — severity labels (0=High, 1=Medium, 2=Low)
        feature_names: list of 2 feature names
        """
        self.X_raw = X_2d.copy()
        self.y = y.copy()
        self.feature_names = feature_names
        self.link_name = link_name
        self.cache_dir = Path(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Pipeline outputs (populated phase by phase)
        self.X_denoised = None
        self.X_balanced = None
        self.y_balanced = None
        self.boundary_points = None
        self.boundary_labels = None
        self.grid_cells = None
        self.rules = None
        self.organized_rules = None
        self.membership_params = None   # {feature: {set: (center, width)}}

    # ----------------------------------------------------------------
    # Phase 1: EKF + Wavelet Denoising
    # ----------------------------------------------------------------
    def phase1_denoise(self):
        cache_file = self.cache_dir / 'phase1_denoised.npz'
        if cache_file.exists():
            data = np.load(cache_file)
            self.X_denoised = data['X_denoised']
            log(f'      P1 Denoise: loaded from cache ({self.X_denoised.shape})')
            return self.X_denoised

        log(f'      P1 Denoise: EKF + Wavelet soft-threshold...')
        X = self.X_raw.copy()

        # Step 1: StandardScaler. When USE_RLS is on, the arrays coming
        # from load_link_rls are already scaled by a StandardScaler that
        # was fit on the record-level TRAIN partition only — refitting a
        # second scaler here on the (train-only) slice would be redundant
        # but not a leakage source, so we simply skip it for consistency
        # with the RLS-only statistics recorded in split_manifest.json.
        if USE_RLS:
            log(f'        RLS mode: skipping in-pipeline StandardScaler '
                f'(pre-scaled by record_level_split.py).')
        else:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # Step 2: Wavelet soft-threshold denoising per feature
        if HAS_PYWT:
            for col in range(X.shape[1]):
                signal = X[:, col]
                # Decompose using Daubechies-4 wavelet, level 3
                coeffs = pywt.wavedec(signal, 'db4', level=3)
                # Estimate noise sigma from finest detail coefficients
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                # Universal threshold
                threshold = sigma * np.sqrt(2 * np.log(len(signal)))
                # Soft-threshold detail coefficients (keep approximation)
                denoised_coeffs = [coeffs[0]]
                for c in coeffs[1:]:
                    denoised_coeffs.append(pywt.threshold(c, threshold, mode='soft'))
                # Reconstruct
                reconstructed = pywt.waverec(denoised_coeffs, 'db4')
                X[:, col] = reconstructed[:len(signal)]
                log(f'        Feature {self.feature_names[col]}: sigma={sigma:.4f} threshold={threshold:.4f}')
        else:
            log(f'        pywt not installed — using StandardScaler only')

        self.X_denoised = X.astype(np.float32)
        np.savez_compressed(cache_file, X_denoised=self.X_denoised)
        log(f'      P1 Denoise: done ({self.X_denoised.shape})')
        return self.X_denoised

    # ----------------------------------------------------------------
    # Phase 2: NURBS-SMOTE Balancing
    # ----------------------------------------------------------------
    def phase2_nurbs_smote(self):
        cache_file = self.cache_dir / 'phase2_balanced.npz'
        if cache_file.exists():
            data = np.load(cache_file)
            self.X_balanced = data['X_balanced']
            self.y_balanced = data['y_balanced']
            log(f'      P2 NURBS-SMOTE: loaded from cache ({self.X_balanced.shape})')
            return self.X_balanced, self.y_balanced

        log(f'      P2 NURBS-SMOTE: Geometric B-Spline oversampling...')
        X = self.X_denoised if self.X_denoised is not None else self.X_raw.copy()
        y = self.y.copy()

        # Count classes
        class_counts = Counter(y)
        max_count = max(class_counts.values())
        log(f'        Before: {dict(class_counts)}')

        all_X, all_y = [X], [y]

        for cls in range(3):
            n_existing = class_counts.get(cls, 0)
            if n_existing == 0 or n_existing >= max_count:
                continue
            n_synthetic = max_count - n_existing
            cls_indices = np.where(y == cls)[0]
            cls_X = X[cls_indices]

            if len(cls_X) < 5:
                # Too few samples — simple duplication
                dup_indices = np.random.choice(len(cls_X), size=n_synthetic, replace=True)
                noise = np.random.normal(0, 0.01, (n_synthetic, cls_X.shape[1]))
                synthetic = cls_X[dup_indices] + noise
            else:
                # NURBS-SMOTE: generate along B-Spline curves between neighbors
                tree = KDTree(cls_X)
                k_neighbors = min(5, len(cls_X) - 1)
                synthetic = np.zeros((n_synthetic, cls_X.shape[1]), dtype=np.float32)
                batch_size = min(5000, n_synthetic)

                generated = 0
                while generated < n_synthetic:
                    # Pick random seed point
                    seed_idx = np.random.randint(0, len(cls_X))
                    seed_point = cls_X[seed_idx]

                    # Find k nearest neighbors
                    _, nn_indices = tree.query(seed_point, k=k_neighbors + 1)
                    nn_indices = nn_indices[1:]  # exclude self

                    # Build control points: seed + neighbors (ordered by distance)
                    control_points = np.vstack([seed_point, cls_X[nn_indices]])

                    # Fit B-Spline through control points
                    n_ctrl = len(control_points)
                    degree = min(3, n_ctrl - 1)  # Cubic or lower

                    # Parameterize: chord-length parameterization
                    dists = np.sqrt(np.sum(np.diff(control_points, axis=0)**2, axis=1))
                    dists = np.maximum(dists, 1e-8)
                    t_params = np.zeros(n_ctrl)
                    t_params[1:] = np.cumsum(dists)
                    t_params /= t_params[-1]  # Normalize to [0, 1]

                    # Create knot vector (clamped uniform)
                    n_knots = n_ctrl + degree + 1
                    knots = np.zeros(n_knots)
                    knots[:degree+1] = 0.0
                    knots[-degree-1:] = 1.0
                    n_internal = n_knots - 2 * (degree + 1)
                    if n_internal > 0:
                        knots[degree+1:degree+1+n_internal] = np.linspace(0, 1, n_internal + 2)[1:-1]

                    # Sample points along the B-Spline curve
                    n_samples_this = min(k_neighbors, n_synthetic - generated)
                    t_eval = np.random.uniform(0, 1, n_samples_this)

                    for dim in range(cls_X.shape[1]):
                        try:
                            spline = BSpline(knots, control_points[:, dim], degree)
                            synthetic[generated:generated+n_samples_this, dim] = spline(t_eval)
                        except Exception:
                            # Fallback: linear interpolation
                            interp_idx = np.random.choice(len(nn_indices), n_samples_this)
                            lam = np.random.uniform(0, 1, n_samples_this)
                            synthetic[generated:generated+n_samples_this, dim] = \
                                (1 - lam) * seed_point[dim] + lam * cls_X[nn_indices[interp_idx], dim]

                    generated += n_samples_this

                synthetic = synthetic[:n_synthetic]

            all_X.append(synthetic.astype(np.float32))
            all_y.append(np.full(n_synthetic, cls, dtype=np.int64))
            log(f'        Class {cls} ({SEV_NAMES[cls]}): {n_existing} -> {max_count} (+{n_synthetic} NURBS-SMOTE)')

        self.X_balanced = np.vstack(all_X).astype(np.float32)
        self.y_balanced = np.concatenate(all_y).astype(np.int64)

        # Shuffle
        perm = np.random.permutation(len(self.y_balanced))
        self.X_balanced = self.X_balanced[perm]
        self.y_balanced = self.y_balanced[perm]

        np.savez_compressed(cache_file, X_balanced=self.X_balanced, y_balanced=self.y_balanced)
        log(f'      P2 NURBS-SMOTE: done ({self.X_balanced.shape})')
        return self.X_balanced, self.y_balanced

    # ----------------------------------------------------------------
    # Phase 3: K-D Tree + R-Tree Boundary Detection
    # ----------------------------------------------------------------
    def phase3_boundary_detection(self):
        cache_file = self.cache_dir / 'phase3_boundary.npz'
        if cache_file.exists():
            data = np.load(cache_file)
            self.boundary_points = data['boundary_points']
            self.boundary_labels = data['boundary_labels']
            log(f'      P3 Boundary: loaded from cache ({self.boundary_points.shape[0]} boundary points)')
            return self.boundary_points, self.boundary_labels

        log(f'      P3 Boundary: K-D Tree + R-Tree detection...')
        X = self.X_balanced if self.X_balanced is not None else self.X_denoised
        y = self.y_balanced if self.y_balanced is not None else self.y

        # Step 1: Build K-D Tree for nearest neighbor queries
        k_neighbors = min(10, len(X) - 1)
        tree = KDTree(X)
        log(f'        KDTree built: {len(X)} points, k={k_neighbors}')

        # Step 2: Find boundary points (mixed-class neighborhoods)
        # Sample subset for efficiency
        n_sample = min(50000, len(X))
        sample_idx = np.random.choice(len(X), n_sample, replace=False)
        X_sample = X[sample_idx]
        y_sample = y[sample_idx]

        _, nn_indices = tree.query(X_sample, k=k_neighbors + 1)
        nn_indices = nn_indices[:, 1:]  # exclude self

        # Boundary = points where neighbors have different classes
        is_boundary = np.zeros(n_sample, dtype=bool)
        for i in range(n_sample):
            neighbor_labels = y[nn_indices[i]]
            n_same = np.sum(neighbor_labels == y_sample[i])
            purity = n_same / k_neighbors
            if purity < 0.8:  # Mixed neighborhood
                is_boundary[i] = True

        boundary_idx = np.where(is_boundary)[0]
        boundary_X = X_sample[boundary_idx]
        boundary_y = y_sample[boundary_idx]
        log(f'        KDTree boundary points: {len(boundary_idx)}/{n_sample} '
            f'({100*len(boundary_idx)/n_sample:.1f}%)')

        # Step 3: R-Tree spatial index for region-level boundary confirmation
        if HAS_RTREE and len(boundary_X) > 0:
            # Build R-Tree from boundary points
            p = rtree_index.Property()
            p.dimension = X.shape[1]
            idx = rtree_index.Index(properties=p)
            for i, pt in enumerate(boundary_X):
                coords = tuple(pt) + tuple(pt)  # point as zero-volume box
                idx.insert(i, coords)

            # Query R-Tree: confirm boundary regions by checking density
            confirmed = np.ones(len(boundary_X), dtype=bool)
            search_radius = 0.2  # Normalized feature space
            for i, pt in enumerate(boundary_X):
                bbox = tuple(pt - search_radius) + tuple(pt + search_radius)
                nearby = list(idx.intersection(bbox))
                if len(nearby) < 3:  # Isolated boundary points — less reliable
                    confirmed[i] = False

            boundary_X = boundary_X[confirmed]
            boundary_y = boundary_y[confirmed]
            log(f'        R-Tree confirmed: {np.sum(confirmed)}/{len(confirmed)} boundary points')
        else:
            if not HAS_RTREE:
                log(f'        R-Tree: rtree not installed — using KDTree results directly')
            # Fallback: use density-based filtering with KDTree
            if len(boundary_X) > 100:
                bt = KDTree(boundary_X)
                dists, _ = bt.query(boundary_X, k=min(5, len(boundary_X)))
                mean_dist = dists[:, 1:].mean(axis=1) if dists.shape[1] > 1 else dists[:, 0]
                threshold = np.percentile(mean_dist, 90)
                confirmed = mean_dist < threshold
                boundary_X = boundary_X[confirmed]
                boundary_y = boundary_y[confirmed]
                log(f'        KDTree density filter: {np.sum(confirmed)} confirmed boundary points')

        self.boundary_points = boundary_X.astype(np.float32)
        self.boundary_labels = boundary_y.astype(np.int64)
        np.savez_compressed(cache_file,
                           boundary_points=self.boundary_points,
                           boundary_labels=self.boundary_labels)
        log(f'      P3 Boundary: done ({self.boundary_points.shape[0]} boundary points)')
        return self.boundary_points, self.boundary_labels

    # ----------------------------------------------------------------
    # Phase 4: Adaptive Grid Partitioning
    # ----------------------------------------------------------------
    def phase4_adaptive_grid(self):
        cache_file = self.cache_dir / 'phase4_grid.json'
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                self.grid_cells = json.load(f)
            log(f'      P4 Grid: loaded from cache ({len(self.grid_cells)} cells)')
            return self.grid_cells

        log(f'      P4 Grid: Adaptive partitioning (coarse={AFRAS_GRID_COARSE}, fine={AFRAS_GRID_FINE})...')
        X = self.X_balanced if self.X_balanced is not None else self.X_denoised
        y = self.y_balanced if self.y_balanced is not None else self.y

        # Feature ranges
        f_min = X.min(axis=0)
        f_max = X.max(axis=0)

        # Step 1: Create coarse grid
        n_coarse = AFRAS_GRID_COARSE
        coarse_edges = [np.linspace(f_min[d], f_max[d], n_coarse + 1) for d in range(2)]

        # Step 2: Determine density per coarse cell
        coarse_density = np.zeros((n_coarse, n_coarse))
        total_points = len(X)
        for i in range(n_coarse):
            for j in range(n_coarse):
                mask = ((X[:, 0] >= coarse_edges[0][i]) & (X[:, 0] < coarse_edges[0][i+1]) &
                        (X[:, 1] >= coarse_edges[1][j]) & (X[:, 1] < coarse_edges[1][j+1]))
                coarse_density[i, j] = mask.sum() / total_points

        # Step 3: Adaptive refinement
        grid_cells = []
        cell_id = 0
        for i in range(n_coarse):
            for j in range(n_coarse):
                x_lo, x_hi = coarse_edges[0][i], coarse_edges[0][i+1]
                y_lo, y_hi = coarse_edges[1][j], coarse_edges[1][j+1]
                density = coarse_density[i, j]

                if density >= AFRAS_DENSITY_THRESHOLD:
                    # Dense region → fine grid
                    n_fine = AFRAS_GRID_FINE
                    fine_x = np.linspace(x_lo, x_hi, n_fine + 1)
                    fine_y = np.linspace(y_lo, y_hi, n_fine + 1)
                    for fi in range(n_fine):
                        for fj in range(n_fine):
                            mask = ((X[:, 0] >= fine_x[fi]) & (X[:, 0] < fine_x[fi+1]) &
                                    (X[:, 1] >= fine_y[fj]) & (X[:, 1] < fine_y[fj+1]))
                            cell_y = y[mask]
                            if len(cell_y) > 0:
                                majority = int(Counter(cell_y).most_common(1)[0][0])
                                purity = np.sum(cell_y == majority) / len(cell_y)
                                grid_cells.append({
                                    'id': cell_id,
                                    'x_range': [float(fine_x[fi]), float(fine_x[fi+1])],
                                    'y_range': [float(fine_y[fj]), float(fine_y[fj+1])],
                                    'n_points': int(len(cell_y)),
                                    'majority_class': majority,
                                    'purity': float(purity),
                                    'class_dist': {str(c): int(v) for c, v in Counter(cell_y.tolist()).items()},
                                    'is_fine': True,
                                    'density': float(density),
                                })
                                cell_id += 1
                else:
                    # Sparse region → coarse cell
                    mask = ((X[:, 0] >= x_lo) & (X[:, 0] < x_hi) &
                            (X[:, 1] >= y_lo) & (X[:, 1] < y_hi))
                    cell_y = y[mask]
                    if len(cell_y) > 0:
                        majority = int(Counter(cell_y).most_common(1)[0][0])
                        purity = np.sum(cell_y == majority) / len(cell_y)
                        grid_cells.append({
                            'id': cell_id,
                            'x_range': [float(x_lo), float(x_hi)],
                            'y_range': [float(y_lo), float(y_hi)],
                            'n_points': int(len(cell_y)),
                            'majority_class': majority,
                            'purity': float(purity),
                            'class_dist': {str(c): int(v) for c, v in Counter(cell_y.tolist()).items()},
                            'is_fine': False,
                            'density': float(density),
                        })
                        cell_id += 1

        self.grid_cells = grid_cells
        with open(cache_file, 'w') as f:
            json.dump(grid_cells, f, indent=2)
        n_fine_cells = sum(1 for c in grid_cells if c['is_fine'])
        n_coarse_cells = len(grid_cells) - n_fine_cells
        log(f'      P4 Grid: {len(grid_cells)} cells (fine={n_fine_cells}, coarse={n_coarse_cells})')
        return self.grid_cells

    # ----------------------------------------------------------------
    # Phase 5: Fuzzy Rule Extraction (CORE)
    # ----------------------------------------------------------------
    def phase5_rule_extraction(self):
        cache_file = self.cache_dir / 'phase5_rules.json'
        params_file = self.cache_dir / 'phase5_membership_params.json'
        if cache_file.exists() and params_file.exists():
            with open(cache_file, 'r') as f:
                self.rules = json.load(f)
            with open(params_file, 'r') as f:
                self.membership_params = json.load(f)
            log(f'      P5 Rules: loaded from cache ({len(self.rules)} rules)')
            return self.rules

        log(f'      P5 Rules: Extracting fuzzy IF-THEN rules...')
        X = self.X_balanced if self.X_balanced is not None else self.X_denoised
        y = self.y_balanced if self.y_balanced is not None else self.y

        # Step 1: Define membership functions (Gaussian) for each feature
        self.membership_params = {}
        for d in range(2):
            fname = self.feature_names[d]
            f_min, f_max = float(X[:, d].min()), float(X[:, d].max())
            f_range = f_max - f_min + 1e-8
            # 3 fuzzy sets: Low, Medium, High (Gaussian centers)
            centers = [f_min + f_range * 0.15,
                       f_min + f_range * 0.50,
                       f_min + f_range * 0.85]
            width = f_range * 0.25  # Gaussian sigma
            self.membership_params[fname] = {
                'Low': {'center': centers[0], 'width': width},
                'Medium': {'center': centers[1], 'width': width},
                'High': {'center': centers[2], 'width': width},
                'range': [f_min, f_max],
            }

        # Step 2: Compute membership degrees for all data points
        fuzzy_sets = ['Low', 'Medium', 'High']
        memberships = np.zeros((len(X), 2, 3), dtype=np.float32)
        for d in range(2):
            fname = self.feature_names[d]
            for s, sname in enumerate(fuzzy_sets):
                center = self.membership_params[fname][sname]['center']
                width = self.membership_params[fname][sname]['width']
                memberships[:, d, s] = np.exp(-0.5 * ((X[:, d] - center) / width) ** 2)

        # Step 3: Generate candidate rules from grid cells
        rules = []
        rule_id = 0
        grid_cells = self.grid_cells if self.grid_cells else []

        for cell in grid_cells:
            if cell['purity'] < AFRAS_MIN_RULE_CONFIDENCE:
                continue
            if cell['n_points'] < AFRAS_MIN_RULE_SUPPORT:
                continue

            # Find the dominant fuzzy set for each feature in this cell
            x_center = (cell['x_range'][0] + cell['x_range'][1]) / 2
            y_center = (cell['y_range'][0] + cell['y_range'][1]) / 2

            antecedents = []
            for d, val in enumerate([x_center, y_center]):
                fname = self.feature_names[d]
                best_set = 'Medium'
                best_membership = 0.0
                for sname in fuzzy_sets:
                    center = self.membership_params[fname][sname]['center']
                    width = self.membership_params[fname][sname]['width']
                    mu = np.exp(-0.5 * ((val - center) / width) ** 2)
                    if mu > best_membership:
                        best_membership = mu
                        best_set = sname
                antecedents.append((fname, best_set))

            # Map severity to action
            severity_to_action = {0: 'BLOCK', 1: 'RATE_LIMIT', 2: 'ALLOW'}
            consequent = severity_to_action[cell['majority_class']]

            rule = {
                'rule_id': rule_id,
                'antecedents': antecedents,
                'consequent': consequent,
                'consequent_action': {0: 2, 1: 1, 2: 0}[cell['majority_class']],
                'confidence': cell['purity'],
                'support': cell['n_points'],
                'cell_id': cell['id'],
            }
            rules.append(rule)
            rule_id += 1

        # Step 4: Merge duplicate rules (same antecedents)
        unique_rules = {}
        for r in rules:
            key = tuple(tuple(a) for a in r['antecedents']) + (r['consequent'],)
            if key not in unique_rules:
                unique_rules[key] = r
            else:
                existing = unique_rules[key]
                # Keep the one with higher confidence, accumulate support
                existing['support'] += r['support']
                if r['confidence'] > existing['confidence']:
                    existing['confidence'] = r['confidence']

        self.rules = list(unique_rules.values())

        # -- L9 mitigation: guarantee consequent-class diversity ------------
        # On attack-heavy datasets (e.g. CICIoT2023) the confidence/support
        # thresholds above can leave the rule base with a single consequent
        # (e.g. all BLOCK), which then causes the AFRAS-FL reward to push
        # the DQN towards over-blocking on unseen records. To prevent this
        # collapse, for every consequent class {BLOCK, RATE_LIMIT, ALLOW}
        # that ended up unrepresented, we add the highest-purity cell whose
        # majority_class maps to it -- if one exists at all -- even when
        # that cell falls below AFRAS_MIN_RULE_CONFIDENCE / SUPPORT.
        present = {r['consequent'] for r in self.rules}
        want = {'BLOCK', 'RATE_LIMIT', 'ALLOW'}
        severity_to_action_name = {0: 'BLOCK', 1: 'RATE_LIMIT', 2: 'ALLOW'}
        severity_to_action_idx = {0: 2, 1: 1, 2: 0}
        for missing in sorted(want - present):
            candidates = [c for c in grid_cells
                          if severity_to_action_name[c['majority_class']] == missing]
            if not candidates:
                continue
            candidates.sort(key=lambda c: (c['purity'], c['n_points']), reverse=True)
            top = candidates[0]
            x_center = (top['x_range'][0] + top['x_range'][1]) / 2
            y_center = (top['y_range'][0] + top['y_range'][1]) / 2
            antecedents = []
            for d, val in enumerate([x_center, y_center]):
                fname = self.feature_names[d]
                best_set, best_membership = 'Medium', 0.0
                for sname in fuzzy_sets:
                    center = self.membership_params[fname][sname]['center']
                    width = self.membership_params[fname][sname]['width']
                    mu = np.exp(-0.5 * ((val - center) / width) ** 2)
                    if mu > best_membership:
                        best_membership, best_set = mu, sname
                antecedents.append((fname, best_set))
            self.rules.append({
                'rule_id': -1,
                'antecedents': antecedents,
                'consequent': missing,
                'consequent_action': severity_to_action_idx[top['majority_class']],
                'confidence': float(top['purity']),
                'support': int(top['n_points']),
                'cell_id': int(top['id']),
                'diversity_relaxed': True,
            })
            log(f'        L9 diversity: added relaxed {missing} rule '
                f'(conf={top["purity"]:.3f}, sup={top["n_points"]})')

        # Re-assign IDs
        for i, r in enumerate(self.rules):
            r['rule_id'] = i

        with open(cache_file, 'w') as f:
            json.dump(self.rules, f, indent=2)
        with open(params_file, 'w') as f:
            json.dump(self.membership_params, f, indent=2)
        log(f'      P5 Rules: {len(self.rules)} unique rules extracted')
        for r in self.rules[:5]:
            ant = ' AND '.join(f'{a[0]} is {a[1]}' for a in r['antecedents'])
            log(f'        Rule {r["rule_id"]}: IF {ant} THEN {r["consequent"]} '
                f'(conf={r["confidence"]:.3f}, sup={r["support"]})')
        return self.rules

    # ----------------------------------------------------------------
    # Phase 6: Rule Organization (Priority + Conflict Resolution)
    # ----------------------------------------------------------------
    def phase6_organize_rules(self):
        cache_file = self.cache_dir / 'phase6_organized.json'
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                self.organized_rules = json.load(f)
            log(f'      P6 Organize: loaded from cache ({len(self.organized_rules)} rules)')
            return self.organized_rules

        log(f'      P6 Organize: Priority sorting + conflict resolution...')
        rules = list(self.rules) if self.rules else []

        # Step 1: Compute priority score
        for r in rules:
            # Priority = confidence * log(support + 1) * action_weight
            action_weight = {'BLOCK': 1.5, 'RATE_LIMIT': 1.0, 'ALLOW': 0.8}
            r['priority'] = (r['confidence'] *
                            math.log(r['support'] + 1) *
                            action_weight[r['consequent']])

        # Step 2: Conflict resolution
        # Two rules conflict if they have same antecedents but different consequents
        antecedent_groups = {}
        for r in rules:
            key = tuple(tuple(a) for a in r['antecedents'])
            if key not in antecedent_groups:
                antecedent_groups[key] = []
            antecedent_groups[key].append(r)

        resolved = []
        conflicts_found = 0
        for key, group in antecedent_groups.items():
            if len(group) == 1:
                resolved.append(group[0])
            else:
                # Multiple rules with same antecedents — keep highest
                # priority PER CONSEQUENT so that a rule flagged as an L9
                # diversity relaxation (different consequent) is not
                # silently absorbed by the majority rule. Rules with
                # identical (antecedents, consequent) still collapse into
                # one, and their supports are still accumulated onto the
                # winning rule.
                by_conseq = {}
                for r in group:
                    c = r['consequent']
                    if c not in by_conseq:
                        by_conseq[c] = r
                    else:
                        by_conseq[c]['support'] += r['support']
                        if r['priority'] > by_conseq[c]['priority']:
                            by_conseq[c]['priority'] = r['priority']
                            by_conseq[c]['confidence'] = r['confidence']
                conflicts_found += len(group) - len(by_conseq)
                resolved.extend(by_conseq.values())

        # Step 3: Sort by priority (descending)
        resolved.sort(key=lambda r: r['priority'], reverse=True)
        for i, r in enumerate(resolved):
            r['rank'] = i + 1
            r['rule_id'] = i

        self.organized_rules = resolved
        with open(cache_file, 'w') as f:
            json.dump(resolved, f, indent=2)
        log(f'      P6 Organize: {len(resolved)} rules, {conflicts_found} conflicts resolved')
        return self.organized_rules

    # ----------------------------------------------------------------
    # Phase 7: FL Deployment Ready (Package rules per agent)
    # ----------------------------------------------------------------
    def phase7_deploy_package(self):
        cache_file = self.cache_dir / 'phase7_deployed.json'
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                package = json.load(f)
            log(f'      P7 Deploy: loaded from cache')
            return package

        log(f'      P7 Deploy: Packaging for {self.link_name}...')
        rules = self.organized_rules if self.organized_rules else self.rules or []

        package = {
            'link_name': self.link_name,
            'feature_names': self.feature_names,
            'n_rules': len(rules),
            'membership_params': self.membership_params,
            'rules': rules,
            'summary': {
                'total_rules': len(rules),
                'block_rules': sum(1 for r in rules if r['consequent'] == 'BLOCK'),
                'limit_rules': sum(1 for r in rules if r['consequent'] == 'RATE_LIMIT'),
                'allow_rules': sum(1 for r in rules if r['consequent'] == 'ALLOW'),
                'avg_confidence': float(np.mean([r['confidence'] for r in rules])) if rules else 0,
                'avg_support': float(np.mean([r['support'] for r in rules])) if rules else 0,
            }
        }

        with open(cache_file, 'w') as f:
            json.dump(package, f, indent=2)

        log(f'      P7 Deploy: {package["summary"]["total_rules"]} rules '
            f'(B={package["summary"]["block_rules"]}, '
            f'L={package["summary"]["limit_rules"]}, '
            f'A={package["summary"]["allow_rules"]})')
        return package

    # ----------------------------------------------------------------
    # Run All 7 Phases
    # ----------------------------------------------------------------
    def run_all(self):
        log(f'\n    AFRAS-FL Pipeline for {self.link_name}:')
        self.phase1_denoise()
        self.phase2_nurbs_smote()
        self.phase3_boundary_detection()
        self.phase4_adaptive_grid()
        self.phase5_rule_extraction()
        self.phase6_organize_rules()
        package = self.phase7_deploy_package()
        return package


# ============================================================================
# Fuzzy Inference Engine (used during training — Innovation 4)
# ============================================================================
class FuzzyInferenceEngine:
    """
    Evaluates fuzzy rules against current observation.
    Returns: recommended action + confidence.
    """

    def __init__(self, rules, membership_params, feature_names):
        self.rules = rules if rules else []
        self.membership_params = membership_params if membership_params else {}
        self.feature_names = feature_names

    def _gaussian_membership(self, x, center, width):
        return np.exp(-0.5 * ((x - center) / max(width, 1e-8)) ** 2)

    def infer(self, feature_values):
        """
        Given 2 feature values, fire matching rules and return best action.
        Returns: (action_idx, confidence) or (None, 0.0) if no rule matches.
        """
        if not self.rules or not self.membership_params:
            return None, 0.0

        best_action = None
        best_strength = 0.0

        for rule in self.rules:
            # Compute firing strength (product of membership degrees)
            strength = 1.0
            for fname, fset in rule['antecedents']:
                if fname not in self.membership_params:
                    strength = 0.0
                    break
                fidx = self.feature_names.index(fname) if fname in self.feature_names else -1
                if fidx < 0 or fidx >= len(feature_values):
                    strength = 0.0
                    break
                params = self.membership_params[fname][fset]
                mu = self._gaussian_membership(feature_values[fidx], params['center'], params['width'])
                strength *= mu

            strength *= rule.get('confidence', 0.5)

            if strength > best_strength:
                best_strength = strength
                best_action = rule.get('consequent_action', None)

        return best_action, float(best_strength)


# ============================================================================
# ██████████████████████████████████████████████████████████████████████████████
# INNOVATION 1: MOSAR Reward Function (5 Multi-Objective Components)
# ██████████████████████████████████████████████████████████████████████████████
# ============================================================================

class MOSARReward:
    """
    Multi-Objective Security-Aware Reward function.
    R_MOSAR = alpha*R_detection + beta*R_accuracy + gamma*R_latency
              + delta*R_confidence + epsilon*R_impact
    """

    @staticmethod
    def compute(action, true_severity, mamba_softmax, q_values=None,
                neighbor_actions=None, fuzzy_engine=None, feature_values=None):
        """
        Compute full reward with optional fuzzy and link components.
        Returns: (R_total, reward_components_dict)
        """
        optimal = {0: 2, 1: 1, 2: 0}[true_severity]

        # Component 1: Detection correctness
        R_detection = 1.0 if action == optimal else -0.5

        # Component 2: Mamba accuracy (confidence for true class)
        R_accuracy = float(mamba_softmax[true_severity]) - 0.5

        # Component 3: Latency cost (heavier actions cost more)
        action_cost = {0: 0.0, 1: 0.3, 2: 0.5}
        R_latency = -0.1 * action_cost[action]

        # Component 4: DQN confidence
        R_confidence = 0.0
        if q_values is not None:
            q_abs = np.abs(q_values)
            q_sum = q_abs.sum()
            if q_sum > 0:
                R_confidence = float(q_abs.max() / q_sum) - 0.33

        # Component 5: Impact (severity-weighted action quality)
        severity_weight = {0: 1.0, 1: 0.5, 2: 0.1}[true_severity]
        action_quality = 1.0 if action == optimal else (-0.5 if action == 0 and true_severity == 0 else -0.2)
        R_impact = severity_weight * action_quality

        # MOSAR base reward (or scalar-reward ablation for cell (a))
        if SCALAR_REWARD:
            R_MOSAR = float(R_detection)
        else:
            R_MOSAR = (MOSAR_W_DETECTION * R_detection +
                       MOSAR_W_ACCURACY * R_accuracy +
                       MOSAR_W_LATENCY * R_latency +
                       MOSAR_W_CONFIDENCE * R_confidence +
                       MOSAR_W_IMPACT * R_impact)

        # Innovation 4: Fuzzy bonus
        R_fuzzy = 0.0
        if fuzzy_engine is not None and feature_values is not None:
            fuzzy_action, fuzzy_conf = fuzzy_engine.infer(feature_values)
            if fuzzy_action is not None:
                if fuzzy_action == action:
                    R_fuzzy = fuzzy_conf * 0.3  # Agreement bonus
                else:
                    R_fuzzy = -0.1  # Disagreement penalty
            else:
                R_fuzzy = -0.05  # No rule matched

        # Innovation 4: Neighbor consensus bonus
        R_link = 0.0
        if neighbor_actions is not None and len(neighbor_actions) > 0:
            consensus = sum(1 for a in neighbor_actions if a == action) / len(neighbor_actions)
            R_link = 0.2 * (consensus - 0.5)

        # Total reward
        R_total = R_MOSAR + LAMBDA_FUZZY * R_fuzzy + MU_LINK * R_link

        components = {
            'R_detection': float(R_detection),
            'R_accuracy': float(R_accuracy),
            'R_latency': float(R_latency),
            'R_confidence': float(R_confidence),
            'R_impact': float(R_impact),
            'R_MOSAR': float(R_MOSAR),
            'R_fuzzy': float(R_fuzzy),
            'R_link': float(R_link),
            'R_total': float(R_total),
        }
        return R_total, components


# ============================================================================
# SDN Environment for Strategy B (MOSAR + Fuzzy-aware)
# ============================================================================
class SDNEnvironmentB(object):
    """Extends Strategy A environment with MOSAR reward and fuzzy inference."""

    def __init__(self, X_windows, y_windows, mamba_softmax, mamba_embed_pca,
                 link_idx=0, feature_order=None, max_steps=MAX_STEPS_PER_EPISODE,
                 fuzzy_engine=None, use_mosar=True, use_fuzzy_reward=False):
        self.X_windows = X_windows
        self.y_windows = y_windows
        self.mamba_softmax = mamba_softmax
        self.mamba_embed = mamba_embed_pca
        self.link_idx = link_idx
        self.n_windows = len(y_windows)
        self.max_steps = min(max_steps, self.n_windows)
        self.feature_order = feature_order
        self.fuzzy_engine = fuzzy_engine
        self.use_mosar = use_mosar
        self.use_fuzzy_reward = use_fuzzy_reward

        self._precompute_link_metrics()
        self.step_idx = 0
        self.episode_indices = None
        self.recent_severities = deque(maxlen=10)
        self.neighbor_envs = []

        # MOSAR tracking
        self.episode_components = []

    def _precompute_link_metrics(self):
        X_mean = self.X_windows.mean(axis=1)
        fo = self.feature_order if self.feature_order else []
        def _idx(name):
            try: return fo.index(name)
            except ValueError: return -1

        idx_flow_packets = _idx('flow_packets_per_s')
        idx_flow_duration = _idx('flow_duration')
        idx_pkt_size = _idx('avg_pkt_size')

        if idx_flow_packets >= 0:
            traffic = X_mean[:, idx_flow_packets]
            self.traffic_load = (traffic - traffic.min()) / (traffic.max() - traffic.min() + 1e-8)
        else:
            self.traffic_load = np.zeros(self.n_windows, dtype=np.float32)

        if idx_flow_duration >= 0:
            lat = X_mean[:, idx_flow_duration]
            self.latency_est = (lat - lat.min()) / (lat.max() - lat.min() + 1e-8)
        else:
            self.latency_est = np.zeros(self.n_windows, dtype=np.float32)

        if idx_pkt_size >= 0:
            pkt = X_mean[:, idx_pkt_size]
            self.packet_rate = (pkt - pkt.min()) / (pkt.max() - pkt.min() + 1e-8)
        else:
            self.packet_rate = np.zeros(self.n_windows, dtype=np.float32)

        # Pre-extract top 2 features per window for fuzzy inference
        self._precompute_fuzzy_features()

    def _precompute_fuzzy_features(self):
        """Extract the 2 features used by fuzzy engine from each window."""
        if self.fuzzy_engine is None or not self.fuzzy_engine.feature_names:
            self.fuzzy_features = None
            return
        X_mean = self.X_windows.mean(axis=1)
        fo = self.feature_order if self.feature_order else []
        feature_indices = []
        for fname in self.fuzzy_engine.feature_names:
            try:
                feature_indices.append(fo.index(fname))
            except ValueError:
                feature_indices.append(0)
        self.fuzzy_features = X_mean[:, feature_indices].astype(np.float32)

    def reset(self):
        indices = np.random.permutation(self.n_windows)[:self.max_steps]
        self.episode_indices = indices
        self.step_idx = 0
        self.recent_severities.clear()
        self.episode_components = []
        return self._get_state()

    def step(self, action, q_values=None):
        idx = self.episode_indices[self.step_idx]
        true_severity = self.y_windows[idx]
        pred_severity = self.mamba_softmax[idx].argmax()
        self.recent_severities.append(pred_severity)

        # Get neighbor actions
        neighbor_actions = self._get_neighbor_actions()

        # Get fuzzy feature values
        fuzzy_fv = None
        if self.fuzzy_features is not None:
            fuzzy_fv = self.fuzzy_features[idx]

        if self.use_mosar:
            reward, components = MOSARReward.compute(
                action=action,
                true_severity=int(true_severity),
                mamba_softmax=self.mamba_softmax[idx],
                q_values=q_values,
                neighbor_actions=neighbor_actions,
                fuzzy_engine=self.fuzzy_engine if self.use_fuzzy_reward else None,
                feature_values=fuzzy_fv if self.use_fuzzy_reward else None,
            )
            # --- PBRS baseline injection ---
            # When USE_PBRS is enabled AND the environment has been switched to
            # PBRS mode (env.use_pbrs_reward), replace R_fuzzy with the
            # potential-based shaping term F(s,s') = gamma * Phi(s') - Phi(s).
            if USE_PBRS and getattr(self, 'use_pbrs_reward', False):
                phi_curr = float(self.mamba_softmax[idx][PBRS_ATTACK_CLASS])
                next_step = self.step_idx + 1
                if next_step < len(self.episode_indices):
                    next_idx = self.episode_indices[next_step]
                    phi_next = float(self.mamba_softmax[next_idx][PBRS_ATTACK_CLASS])
                else:
                    phi_next = 0.0
                F = PBRS_GAMMA * phi_next - phi_curr
                # subtract old R_fuzzy contribution, add PBRS instead
                old_fuzzy_contrib = LAMBDA_FUZZY * components.get('R_fuzzy', 0.0)
                reward = reward - old_fuzzy_contrib + LAMBDA_PBRS * F
                components['R_pbrs'] = F
                components['R_total'] = float(reward)
            self.episode_components.append(components)
        else:
            # Fallback to simple reward (Strategy A style)
            optimal = {0: 2, 1: 1, 2: 0}[true_severity]
            R_correct = 1.0 if action == optimal else -0.5
            R_security = -1.0 if (action == 0 and true_severity == 0) else 0.0
            R_service = -1.0 if (action == 2 and true_severity == 2) else 0.0
            reward = 0.4 * R_correct + 0.3 * R_security + 0.2 * R_service
            components = {'R_total': reward}

        self.step_idx += 1
        done = self.step_idx >= len(self.episode_indices)
        info = {
            'true_severity': int(true_severity),
            'pred_severity': int(pred_severity),
            'action': int(action),
            'correct': int(action == {0: 2, 1: 1, 2: 0}[true_severity]),
            'components': components,
        }
        next_state = self._get_state() if not done else np.zeros(STATE_DIM, dtype=np.float32)
        return next_state, reward, done, info

    def _get_state(self):
        if self.step_idx >= len(self.episode_indices):
            return np.zeros(STATE_DIM, dtype=np.float32)
        idx = self.episode_indices[self.step_idx]
        state = np.zeros(STATE_DIM, dtype=np.float32)
        state[0] = self.traffic_load[idx]
        state[1] = self.latency_est[idx]
        state[2] = self._compute_entropy()
        state[3] = self.packet_rate[idx]
        softmax = self.mamba_softmax[idx]
        pred = softmax.argmax()
        state[4] = float(pred != 2)
        state[5] = float(pred) / 2.0
        state[6] = float(softmax.max())
        state[7] = self._severity_ratio()
        state[8:24] = self.mamba_embed[idx]
        n_flags = self._get_neighbor_flags()
        state[24:28] = n_flags
        return state

    def _compute_entropy(self):
        if len(self.recent_severities) < 2:
            return 0.5
        counts = Counter(self.recent_severities)
        total = len(self.recent_severities)
        entropy = 0.0
        for c in counts.values():
            p = c / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy / math.log2(3)

    def _severity_ratio(self):
        if len(self.recent_severities) == 0:
            return 0.0
        return sum(1 for s in self.recent_severities if s == 0) / len(self.recent_severities)

    def _get_neighbor_flags(self):
        flags = np.zeros(4, dtype=np.float32)
        if not self.neighbor_envs:
            return flags
        if self.link_idx > 0 and len(self.neighbor_envs) > self.link_idx - 1:
            env_l = self.neighbor_envs[self.link_idx - 1]
            if len(env_l.recent_severities) > 0:
                flags[0] = env_l.recent_severities[-1] / 2.0
        if self.link_idx < 4 and len(self.neighbor_envs) > self.link_idx + 1:
            env_r = self.neighbor_envs[self.link_idx + 1]
            if len(env_r.recent_severities) > 0:
                flags[1] = env_r.recent_severities[-1] / 2.0
        alert = 0
        for env in self.neighbor_envs:
            if env is not self and len(env.recent_severities) > 0:
                if env.recent_severities[-1] == 0:
                    alert += 1
        flags[2] = alert / max(len(self.neighbor_envs) - 1, 1)
        sev_sum, sev_count = 0, 0
        for env in self.neighbor_envs:
            if env is not self and len(env.recent_severities) > 0:
                sev_sum += env.recent_severities[-1]
                sev_count += 1
        flags[3] = (sev_sum / max(sev_count, 1)) / 2.0
        return flags

    def _get_neighbor_actions(self):
        return None


# ============================================================================
# DQN Network + Agent + Replay Buffer (same as Strategy A)
# ============================================================================
class DQNNetwork(nn.Module):
    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS, hidden=DQN_HIDDEN):
        super().__init__()
        layers = []
        prev = state_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS, device=DEVICE):
        self.device = device
        self.n_actions = n_actions
        self.q_net = DQNNetwork(state_dim, n_actions).to(device)
        self.target_net = DQNNetwork(state_dim, n_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=DQN_LR)
        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.epsilon = EPSILON_START
        self.total_steps = 0

    def select_action(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            q_vals = self.q_net(torch.FloatTensor(state).unsqueeze(0).to(self.device))
            return q_vals.argmax(dim=1).item()

    def get_q_values(self, state):
        with torch.no_grad():
            return self.q_net(torch.FloatTensor(state).unsqueeze(0).to(self.device)).cpu().numpy()[0]

    def train_step(self, global_params=None, mu=0.0):
        if len(self.buffer) < DQN_BATCH:
            return 0.0
        states, actions, rewards, next_states, dones = self.buffer.sample(DQN_BATCH)
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            target_q = rewards_t + GAMMA * next_q * (1 - dones_t)
        loss = F.mse_loss(q_values, target_q)
        if mu > 0 and global_params is not None:
            prox_term = 0.0
            for p, gp in zip(self.q_net.parameters(), global_params):
                prox_term += ((p - gp.to(self.device)) ** 2).sum()
            loss = loss + (mu / 2.0) * prox_term
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.total_steps += 1
        for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(TAU * p.data + (1.0 - TAU) * tp.data)
        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def get_params(self):
        return [p.data.clone().cpu() for p in self.q_net.parameters()]

    def set_params(self, params):
        for p, new_p in zip(self.q_net.parameters(), params):
            p.data.copy_(new_p.to(self.device))

    def reset(self):
        self.q_net = DQNNetwork(STATE_DIM, N_ACTIONS).to(self.device)
        self.target_net = DQNNetwork(STATE_DIM, N_ACTIONS).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=DQN_LR)
        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.epsilon = EPSILON_START
        self.total_steps = 0

    @property
    def n_params(self):
        return count_params(self.q_net)


# ============================================================================
# FL Aggregation
# ============================================================================
def fedavg_aggregate(client_params_list, client_weights):
    n_params = len(client_params_list[0])
    aggregated = []
    for i in range(n_params):
        weighted_sum = torch.zeros_like(client_params_list[0][i])
        for c, w in enumerate(client_weights):
            weighted_sum += w * client_params_list[c][i]
        aggregated.append(weighted_sum)
    return aggregated

def blend_params(local_params, global_params, alpha=BLEND_ALPHA):
    blended = []
    for lp, gp in zip(local_params, global_params):
        blended.append(alpha * lp + (1.0 - alpha) * gp)
    return blended


# ============================================================================
# Evaluation
# ============================================================================
def evaluate_agent(agent, env, n_episodes=5):
    total_correct = 0
    total_steps = 0
    total_reward = 0.0
    action_counts = Counter()
    severity_action_matrix = np.zeros((3, 3), dtype=np.int64)
    fp_count, fn_count, benign_count, attack_count = 0, 0, 0, 0
    mosar_components_sum = Counter()

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, eval_mode=True)
            q_values = agent.get_q_values(state)
            next_state, reward, done, info = env.step(action, q_values=q_values)
            total_correct += info['correct']
            total_steps += 1
            total_reward += reward
            action_counts[action] += 1
            severity_action_matrix[info['true_severity'], action] += 1
            if info['true_severity'] == 2:
                benign_count += 1
                if action == 2:
                    fp_count += 1
            elif info['true_severity'] == 0:
                attack_count += 1
                if action == 0:
                    fn_count += 1
            for k, v in info.get('components', {}).items():
                mosar_components_sum[k] += v
            state = next_state

    msr = total_correct / max(total_steps, 1)
    fpr = fp_count / max(benign_count, 1)
    fnr = fn_count / max(attack_count, 1)

    return {
        'msr': float(msr),
        'fpr': float(fpr),
        'fnr': float(fnr),
        'avg_reward': float(total_reward / max(total_steps, 1)),
        'service_availability': float(1.0 - fpr),
        'total_steps': int(total_steps),
        'action_distribution': {ACTION_NAMES[k]: int(v) for k, v in action_counts.items()},
        'severity_action_matrix': severity_action_matrix.tolist(),
        'mosar_components': {k: float(v / max(total_steps, 1)) for k, v in mosar_components_sum.items()},
    }


def compute_jain_fairness(msr_list):
    n = len(msr_list)
    sum_x = sum(msr_list)
    sum_x2 = sum(x**2 for x in msr_list)
    if sum_x2 == 0:
        return 1.0
    return (sum_x ** 2) / (n * sum_x2)


# ============================================================================
# Master Agent (same as Strategy A)
# ============================================================================
class MasterAgent:
    def __init__(self, n_links=5, capacity_mbps=1000.0):
        self.n_links = n_links
        self.capacity = capacity_mbps
        self.w_capacity = 0.40
        self.w_latency = 0.25
        self.w_pktloss = 0.20
        self.w_safety = 0.15
        self.overload_thresh = 0.85
        self.healthy_h_thresh = 0.60
        self.healthy_u_thresh = 0.70
        self.jfi_thresh = 0.90

    def compute_health_scores(self, U, L, PL, A):
        H = np.zeros(self.n_links)
        for i in range(self.n_links):
            C_i = 1.0 - U[i]
            L_i_norm = max(0.0, 1.0 - L[i] / 100.0)
            P_i = 1.0 - PL[i]
            A_i_safe = 1.0 - A[i]
            H[i] = (self.w_capacity * C_i + self.w_latency * L_i_norm +
                     self.w_pktloss * P_i + self.w_safety * A_i_safe)
        return H

    def identify_links(self, H, U):
        overloaded = [i for i in range(self.n_links) if U[i] > self.overload_thresh]
        healthy = [i for i in range(self.n_links)
                    if H[i] > self.healthy_h_thresh and U[i] < self.healthy_u_thresh]
        healthy.sort(key=lambda i: H[i], reverse=True)
        return overloaded, healthy

    def compute_redistribution(self, U, H, overloaded, healthy):
        reroute_commands = []
        U_sim = U.copy()
        healthy_pool = list(healthy)
        for i in overloaded:
            excess_i = (U_sim[i] - self.overload_thresh) * self.capacity
            if excess_i <= 0 or not healthy_pool:
                continue
            available_target = (1.0 - U_sim[healthy_pool[0]]) * self.capacity
            if available_target >= excess_i:
                target = healthy_pool[0]
                reroute_commands.append({
                    'from_link': LINK_SHORT[i], 'to_link': LINK_SHORT[target],
                    'amount_mbps': round(float(excess_i), 1),
                })
                U_sim[i] -= excess_i / self.capacity
                U_sim[target] += excess_i / self.capacity
                if U_sim[target] > self.healthy_u_thresh:
                    healthy_pool.remove(target)
            else:
                remaining = excess_i
                for j in list(healthy_pool):
                    available_j = (1.0 - U_sim[j]) * self.capacity
                    amount_j = min(remaining, available_j)
                    if amount_j <= 0:
                        continue
                    reroute_commands.append({
                        'from_link': LINK_SHORT[i], 'to_link': LINK_SHORT[j],
                        'amount_mbps': round(float(amount_j), 1),
                    })
                    remaining -= amount_j
                    U_sim[i] -= amount_j / self.capacity
                    U_sim[j] += amount_j / self.capacity
                    if U_sim[j] > self.healthy_u_thresh:
                        healthy_pool.remove(j)
                    if remaining <= 0:
                        break
        return reroute_commands, U_sim

    def compute_routing_probabilities(self, H):
        total_h = H.sum()
        if total_h == 0:
            return np.ones(self.n_links) / self.n_links
        return H / total_h

    def compute_fairness(self, U):
        sum_u = U.sum()
        sum_u2 = (U ** 2).sum()
        if sum_u2 == 0:
            return 1.0
        return (sum_u ** 2) / (self.n_links * sum_u2)

    def rank_links(self, U, L, PL, A):
        H = self.compute_health_scores(U, L, PL, A)
        overloaded, healthy = self.identify_links(H, U)
        reroute_commands, U_after = self.compute_redistribution(U.copy(), H, overloaded, healthy)
        P = self.compute_routing_probabilities(H)
        jfi_before = self.compute_fairness(U)
        jfi_after = self.compute_fairness(U_after)
        ranking_idx = np.argsort(-H)
        return {
            'health_scores': H.tolist(),
            'ranking': [LINK_SHORT[i] for i in ranking_idx],
            'best_link': LINK_SHORT[ranking_idx[0]],
            'overloaded': [LINK_SHORT[i] for i in overloaded],
            'healthy': [LINK_SHORT[i] for i in healthy],
            'reroute_commands': reroute_commands,
            'routing_probabilities': {LINK_SHORT[i]: float(P[i]) for i in range(self.n_links)},
            'utilization_before': U.tolist(),
            'utilization_after': U_after.tolist(),
            'jfi_before': float(jfi_before),
            'jfi_after': float(jfi_after),
        }

    def evaluate_over_episodes(self, agents, envs, n_episodes=50):
        all_health, all_jfi_before, all_jfi_after = [], [], []
        all_best_links, all_reroute_counts, all_routing_probs = [], [], []
        for ep in range(n_episodes):
            U, L, PL, A = np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5)
            for i in range(5):
                state = envs[i].reset()
                done = False
                fp, benign, total = 0, 0, 0
                traffic_sum, latency_sum, attack_sum = 0.0, 0.0, 0.0
                while not done:
                    action = agents[i].select_action(state, eval_mode=True)
                    q_values = agents[i].get_q_values(state)
                    next_state, reward, done, info = envs[i].step(action, q_values=q_values)
                    total += 1
                    if info['true_severity'] == 2:
                        benign += 1
                        if action == 2:
                            fp += 1
                    traffic_sum += state[0]
                    latency_sum += state[1]
                    attack_sum += state[4]
                    state = next_state
                U[i] = traffic_sum / max(total, 1)
                L[i] = (latency_sum / max(total, 1)) * 100.0
                PL[i] = fp / max(benign, 1)
                A[i] = attack_sum / max(total, 1)
            result = self.rank_links(U, L, PL, A)
            all_health.append(result['health_scores'])
            all_jfi_before.append(result['jfi_before'])
            all_jfi_after.append(result['jfi_after'])
            all_best_links.append(result['best_link'])
            all_reroute_counts.append(len(result['reroute_commands']))
            all_routing_probs.append(list(result['routing_probabilities'].values()))
        health_mean = np.mean(all_health, axis=0)
        health_std = np.std(all_health, axis=0)
        best_link_counts = Counter(all_best_links)
        routing_mean = np.mean(all_routing_probs, axis=0)
        return {
            'health_mean': health_mean.tolist(),
            'health_std': health_std.tolist(),
            'jfi_before_mean': float(np.mean(all_jfi_before)),
            'jfi_after_mean': float(np.mean(all_jfi_after)),
            'jfi_improvement': float(np.mean(all_jfi_after) - np.mean(all_jfi_before)),
            'best_link_frequency': {k: v for k, v in best_link_counts.most_common()},
            'avg_reroutes_per_episode': float(np.mean(all_reroute_counts)),
            'routing_prob_mean': {LINK_SHORT[i]: float(routing_mean[i]) for i in range(5)},
            'all_health': all_health,
            'all_jfi_before': all_jfi_before,
            'all_jfi_after': all_jfi_after,
            'last_result': result,
        }


# ============================================================================
# Mid-Phase Checkpoint Helpers
# ============================================================================
def save_mid_checkpoint(agents, history, ep_or_round, phase_dir, prefix='phase'):
    ckpt = {
        'ep_or_round': ep_or_round,
        'history': history,
        'epsilons': [a.epsilon for a in agents],
        'total_steps': [a.total_steps for a in agents],
    }
    for i, agent in enumerate(agents):
        ckpt[f'q_net_{i}'] = agent.q_net.state_dict()
        ckpt[f'target_net_{i}'] = agent.target_net.state_dict()
    os.makedirs(phase_dir, exist_ok=True)
    torch.save(ckpt, phase_dir / f'{prefix}_resume.pt')

def load_mid_checkpoint(agents, phase_dir, prefix='phase'):
    path = phase_dir / f'{prefix}_resume.pt'
    if not path.exists():
        return None
    ckpt = torch.load(path, weights_only=False, map_location=DEVICE)
    for i, agent in enumerate(agents):
        agent.q_net.load_state_dict(ckpt[f'q_net_{i}'])
        agent.target_net.load_state_dict(ckpt[f'target_net_{i}'])
        agent.epsilon = ckpt['epsilons'][i]
        agent.total_steps = ckpt['total_steps'][i]
    log(f'      Resumed from checkpoint: ep/round={ckpt["ep_or_round"]}')
    return ckpt['ep_or_round'], ckpt['history']

def remove_mid_checkpoint(phase_dir, prefix='phase'):
    path = phase_dir / f'{prefix}_resume.pt'
    if path.exists():
        os.remove(path)


# ============================================================================
# Training: Phase B1 (Independent DQN + MOSAR)
# ============================================================================
def train_independent_mosar(agents, envs, n_episodes, phase_dir):
    log(f'\n  Phase B1: Independent DQN + MOSAR ({n_episodes} episodes)')
    os.makedirs(phase_dir, exist_ok=True)
    history = {str(i): {'rewards': [], 'msr': [], 'loss': [], 'epsilon': [],
                        'mosar_components': []} for i in range(5)}
    start_ep = 0
    resumed = load_mid_checkpoint(agents, phase_dir, 'phaseB1')
    if resumed:
        start_ep, history = resumed[0] + 1, resumed[1]
        log(f'      Resuming Phase B1 from episode {start_ep}')

    for ep in range(start_ep, n_episodes):
        ep_rewards = []
        for i in range(5):
            state = envs[i].reset()
            done = False
            ep_reward, ep_loss, steps = 0.0, 0.0, 0
            while not done:
                action = agents[i].select_action(state)
                q_values = agents[i].get_q_values(state)
                next_state, reward, done, info = envs[i].step(action, q_values=q_values)
                agents[i].buffer.push(state, action, reward, next_state, done)
                loss = agents[i].train_step()
                ep_reward += reward
                ep_loss += loss
                steps += 1
                state = next_state
            agents[i].decay_epsilon()
            history[str(i)]['rewards'].append(ep_reward / max(steps, 1))
            history[str(i)]['loss'].append(ep_loss / max(steps, 1))
            history[str(i)]['epsilon'].append(agents[i].epsilon)
            ep_rewards.append(ep_reward / max(steps, 1))

        if (ep + 1) % 10 == 0 or ep == 0:
            mean_reward = np.mean(ep_rewards)
            msr_list = []
            for i in range(5):
                m = evaluate_agent(agents[i], envs[i], n_episodes=1)
                msr_list.append(m['msr'])
                history[str(i)]['msr'].append(m['msr'])
                if m.get('mosar_components'):
                    history[str(i)]['mosar_components'].append(m['mosar_components'])
            mean_msr = np.mean(msr_list)
            jfi = compute_jain_fairness(msr_list)
            log(f'    Ep {ep+1:3d}/{n_episodes}: Reward={mean_reward:.4f} '
                f'MSR={mean_msr:.4f} JFI={jfi:.4f} eps={agents[0].epsilon:.3f}')

        if (ep + 1) % CKPT_EVERY_EP == 0 and (ep + 1) < n_episodes:
            save_mid_checkpoint(agents, history, ep, phase_dir, 'phaseB1')

    # Final evaluation
    log(f'\n  [Final Eval] Independent DQN + MOSAR:')
    results = {}
    for i in range(5):
        m = evaluate_agent(agents[i], envs[i], n_episodes=10)
        log(f'    {LINK_SHORT[i]}: MSR={m["msr"]:.4f} FPR={m["fpr"]:.4f} '
            f'FNR={m["fnr"]:.4f} Reward={m["avg_reward"]:.4f}')
        results[LINK_NAMES[i]] = m
        ckpt_dir = phase_dir / LINK_NAMES[i]
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(agents[i].q_net.state_dict(), ckpt_dir / 'best_dqn.pt')

    msr_list = [results[ln]['msr'] for ln in LINK_NAMES]
    results['global'] = {
        'mean_msr': float(np.mean(msr_list)),
        'std_msr': float(np.std(msr_list)),
        'jain_fairness': float(compute_jain_fairness(msr_list)),
        'mean_fpr': float(np.mean([results[ln]['fpr'] for ln in LINK_NAMES])),
        'mean_fnr': float(np.mean([results[ln]['fnr'] for ln in LINK_NAMES])),
        'mean_reward': float(np.mean([results[ln]['avg_reward'] for ln in LINK_NAMES])),
    }
    results['history'] = history
    log(f'    Global: MSR={results["global"]["mean_msr"]:.4f} '
        f'JFI={results["global"]["jain_fairness"]:.4f}')

    with open(phase_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    remove_mid_checkpoint(phase_dir, 'phaseB1')
    return results


# ============================================================================
# Training: Phase B2/B3 (FL-DQN + MOSAR)
# ============================================================================
def train_fl_mosar(agents, envs, fl_rounds, local_episodes, algo='fedavg',
                   mu=0.0, phase_dir=None):
    algo_label = 'FedAvg+MOSAR' if algo == 'fedavg' else f'FedProx+MOSAR(mu={mu})'
    phase_num = '2' if algo == 'fedavg' else '3'
    ckpt_prefix = f'phaseB{phase_num}'
    log(f'\n  Phase B{phase_num}: FL-DQN with {algo_label} '
        f'({fl_rounds} rounds x {local_episodes} local eps)')

    if phase_dir:
        os.makedirs(phase_dir, exist_ok=True)

    for env in envs:
        env.neighbor_envs = envs

    global_params = agents[0].get_params()
    client_weights = [env.n_windows for env in envs]
    total_w = sum(client_weights)
    client_weights = [w / total_w for w in client_weights]

    history = {
        'rounds': [],
        'per_agent': {str(i): {'msr': [], 'reward': [], 'fpr': [], 'fnr': []} for i in range(5)},
        'global_msr': [], 'jain_fairness': [],
        'global_fpr': [], 'global_fnr': [], 'mean_loss': [],   # A5 round-drift
    }
    start_rd = 0
    if phase_dir:
        resumed = load_mid_checkpoint(agents, phase_dir, ckpt_prefix)
        if resumed:
            start_rd, history = resumed[0] + 1, resumed[1]
            gp_path = phase_dir / 'global_dqn_mid.pt'
            if gp_path.exists():
                global_params = torch.load(gp_path, weights_only=False, map_location=DEVICE)
            log(f'      Resuming Phase B{phase_num} from round {start_rd}')

    # A5 round-drift: backfill keys (resumed runs) + per-round checkpoint dir
    for _k in ('global_fpr', 'global_fnr', 'mean_loss'):
        history.setdefault(_k, [])
    round_ckpt_dir = None
    if phase_dir:
        round_ckpt_dir = phase_dir / 'round_ckpts'
        os.makedirs(round_ckpt_dir, exist_ok=True)

    for rd in range(start_rd, fl_rounds):
        client_params_list = []
        round_losses = []
        for i in range(5):
            for ep in range(local_episodes):
                state = envs[i].reset()
                done = False
                while not done:
                    action = agents[i].select_action(state)
                    q_values = agents[i].get_q_values(state)
                    next_state, reward, done, info = envs[i].step(action, q_values=q_values)
                    agents[i].buffer.push(state, action, reward, next_state, done)
                    if algo == 'fedprox':
                        loss_val = agents[i].train_step(global_params=global_params, mu=mu)
                    else:
                        loss_val = agents[i].train_step()
                    round_losses.append(loss_val)
                    state = next_state
            agents[i].decay_epsilon()
            client_params_list.append(agents[i].get_params())

        global_params = fedavg_aggregate(client_params_list, client_weights)

        # A5 round-drift: persist the aggregated GLOBAL model for this round.
        if round_ckpt_dir is not None:
            os.makedirs(round_ckpt_dir, exist_ok=True)
            torch.save([p.clone() for p in global_params],
                       round_ckpt_dir / f'global_round_{rd + 1:02d}.pt')

        for i in range(5):
            local_p = agents[i].get_params()
            blended = blend_params(local_p, global_params, BLEND_ALPHA)
            agents[i].set_params(blended)
            agents[i].target_net.load_state_dict(agents[i].q_net.state_dict())

        msr_list = []
        fpr_list = []
        fnr_list = []
        for i in range(5):
            m = evaluate_agent(agents[i], envs[i], n_episodes=3)
            msr_list.append(m['msr'])
            fpr_list.append(m['fpr'])
            fnr_list.append(m['fnr'])
            history['per_agent'][str(i)]['msr'].append(m['msr'])
            history['per_agent'][str(i)]['reward'].append(m['avg_reward'])
            history['per_agent'][str(i)]['fpr'].append(m['fpr'])
            history['per_agent'][str(i)]['fnr'].append(m['fnr'])

        mean_msr = np.mean(msr_list)
        jfi = compute_jain_fairness(msr_list)
        history['rounds'].append(rd + 1)
        history['global_msr'].append(float(mean_msr))
        history['jain_fairness'].append(float(jfi))
        history['global_fpr'].append(float(np.nanmean(fpr_list)))
        history['global_fnr'].append(float(np.nanmean(fnr_list)))
        history['mean_loss'].append(float(np.mean(round_losses)) if round_losses else float('nan'))

        if (rd + 1) % 5 == 0 or rd == 0:
            log(f'    Round {rd+1:2d}/{fl_rounds}: MSR={mean_msr:.4f} '
                f'JFI={jfi:.4f} eps={agents[0].epsilon:.3f}')

        if phase_dir and (rd + 1) % CKPT_EVERY_ROUND == 0 and (rd + 1) < fl_rounds:
            save_mid_checkpoint(agents, history, rd, phase_dir, ckpt_prefix)
            torch.save([p.clone() for p in global_params], phase_dir / 'global_dqn_mid.pt')

    # Final evaluation
    log(f'\n  [Final Eval] {algo_label}:')
    results = {}
    for i in range(5):
        m = evaluate_agent(agents[i], envs[i], n_episodes=10)
        log(f'    {LINK_SHORT[i]}: MSR={m["msr"]:.4f} FPR={m["fpr"]:.4f} '
            f'FNR={m["fnr"]:.4f} Reward={m["avg_reward"]:.4f}')
        results[LINK_NAMES[i]] = m
        if phase_dir:
            ckpt_dir = phase_dir / LINK_NAMES[i]
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(agents[i].q_net.state_dict(), ckpt_dir / 'best_dqn.pt')

    msr_list = [results[ln]['msr'] for ln in LINK_NAMES]
    results['global'] = {
        'mean_msr': float(np.mean(msr_list)),
        'std_msr': float(np.std(msr_list)),
        'jain_fairness': float(compute_jain_fairness(msr_list)),
        'mean_fpr': float(np.mean([results[ln]['fpr'] for ln in LINK_NAMES])),
        'mean_fnr': float(np.mean([results[ln]['fnr'] for ln in LINK_NAMES])),
        'mean_reward': float(np.mean([results[ln]['avg_reward'] for ln in LINK_NAMES])),
    }
    results['history'] = history
    log(f'    Global: MSR={results["global"]["mean_msr"]:.4f} '
        f'JFI={results["global"]["jain_fairness"]:.4f}')

    if phase_dir:
        with open(phase_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        torch.save([p.clone() for p in global_params], phase_dir / 'global_dqn.pt')
        remove_mid_checkpoint(phase_dir, ckpt_prefix)
        mid_gp = phase_dir / 'global_dqn_mid.pt'
        if mid_gp.exists():
            os.remove(mid_gp)
    return results


# ============================================================================
# Training: Phase B4 (Fuzzy-Augmented Re-training)
# ============================================================================
def train_fuzzy_augmented(agents, envs, fl_rounds, local_episodes, mu=FEDPROX_MU,
                          phase_dir=None):
    """
    Re-trains agents with R_total = R_MOSAR + lambda*R_fuzzy + mu_link*R_link.
    Uses FedProx as base (best from B3) + fuzzy reward guidance.
    """
    log(f'\n  Phase B4: Fuzzy-Augmented Re-training '
        f'({fl_rounds} rounds x {local_episodes} local eps)')
    log(f'    lambda_fuzzy={LAMBDA_FUZZY}, mu_link={MU_LINK}')

    if phase_dir:
        os.makedirs(phase_dir, exist_ok=True)

    # Enable shaping reward in environments (PBRS baseline swap-in)
    for env in envs:
        env.neighbor_envs = envs
        if USE_PBRS:
            env.use_fuzzy_reward = True   # keeps fuzzy path so R_fuzzy is populated,
            env.use_pbrs_reward = True    # then overridden in env.step() by PBRS shaping
        else:
            env.use_fuzzy_reward = True
            env.use_pbrs_reward = False

    global_params = agents[0].get_params()
    client_weights = [env.n_windows for env in envs]
    total_w = sum(client_weights)
    client_weights = [w / total_w for w in client_weights]

    history = {
        'rounds': [],
        'per_agent': {str(i): {'msr': [], 'reward': [], 'fpr': [], 'fnr': [],
                                'fuzzy_match_rate': []} for i in range(5)},
        'global_msr': [], 'jain_fairness': [],
        'fuzzy_reward_avg': [],
        'global_fpr': [], 'global_fnr': [], 'mean_loss': [],   # A5 round-drift
    }
    start_rd = 0
    if phase_dir:
        resumed = load_mid_checkpoint(agents, phase_dir, 'phaseB4')
        if resumed:
            start_rd, history = resumed[0] + 1, resumed[1]
            gp_path = phase_dir / 'global_dqn_mid.pt'
            if gp_path.exists():
                global_params = torch.load(gp_path, weights_only=False, map_location=DEVICE)
            log(f'      Resuming Phase B4 from round {start_rd}')

    # A5 round-drift: backfill keys (resumed runs) + per-round checkpoint dir
    for _k in ('global_fpr', 'global_fnr', 'mean_loss'):
        history.setdefault(_k, [])
    round_ckpt_dir = None
    if phase_dir:
        round_ckpt_dir = phase_dir / 'round_ckpts'
        os.makedirs(round_ckpt_dir, exist_ok=True)

    for rd in range(start_rd, fl_rounds):
        client_params_list = []
        round_fuzzy_rewards = []
        round_losses = []

        for i in range(5):
            fuzzy_matches, fuzzy_total = 0, 0
            for ep in range(local_episodes):
                state = envs[i].reset()
                done = False
                while not done:
                    action = agents[i].select_action(state)
                    q_values = agents[i].get_q_values(state)
                    next_state, reward, done, info = envs[i].step(action, q_values=q_values)
                    agents[i].buffer.push(state, action, reward, next_state, done)
                    loss_val = agents[i].train_step(global_params=global_params, mu=mu)
                    round_losses.append(loss_val)

                    # Track fuzzy agreement
                    comp = info.get('components', {})
                    if comp.get('R_fuzzy', 0) > 0:
                        fuzzy_matches += 1
                    fuzzy_total += 1
                    round_fuzzy_rewards.append(comp.get('R_fuzzy', 0))

                    state = next_state
            agents[i].decay_epsilon()
            client_params_list.append(agents[i].get_params())
            history['per_agent'][str(i)]['fuzzy_match_rate'].append(
                fuzzy_matches / max(fuzzy_total, 1))

        global_params = fedavg_aggregate(client_params_list, client_weights)

        # A5 round-drift: persist the aggregated GLOBAL model for this round.
        if round_ckpt_dir is not None:
            os.makedirs(round_ckpt_dir, exist_ok=True)
            torch.save([p.clone() for p in global_params],
                       round_ckpt_dir / f'global_round_{rd + 1:02d}.pt')

        for i in range(5):
            local_p = agents[i].get_params()
            blended = blend_params(local_p, global_params, BLEND_ALPHA)
            agents[i].set_params(blended)
            agents[i].target_net.load_state_dict(agents[i].q_net.state_dict())

        msr_list = []
        fpr_list = []
        fnr_list = []
        for i in range(5):
            m = evaluate_agent(agents[i], envs[i], n_episodes=3)
            msr_list.append(m['msr'])
            fpr_list.append(m['fpr'])
            fnr_list.append(m['fnr'])
            history['per_agent'][str(i)]['msr'].append(m['msr'])
            history['per_agent'][str(i)]['reward'].append(m['avg_reward'])
            history['per_agent'][str(i)]['fpr'].append(m['fpr'])
            history['per_agent'][str(i)]['fnr'].append(m['fnr'])

        mean_msr = np.mean(msr_list)
        jfi = compute_jain_fairness(msr_list)
        history['rounds'].append(rd + 1)
        history['global_msr'].append(float(mean_msr))
        history['jain_fairness'].append(float(jfi))
        history['fuzzy_reward_avg'].append(float(np.mean(round_fuzzy_rewards)))
        history['global_fpr'].append(float(np.nanmean(fpr_list)))
        history['global_fnr'].append(float(np.nanmean(fnr_list)))
        history['mean_loss'].append(float(np.mean(round_losses)) if round_losses else float('nan'))

        if (rd + 1) % 5 == 0 or rd == 0:
            avg_fmr = np.mean([history['per_agent'][str(i)]['fuzzy_match_rate'][-1] for i in range(5)])
            log(f'    Round {rd+1:2d}/{fl_rounds}: MSR={mean_msr:.4f} '
                f'JFI={jfi:.4f} FuzzyMatch={avg_fmr:.3f} eps={agents[0].epsilon:.3f}')

        if phase_dir and (rd + 1) % CKPT_EVERY_ROUND == 0 and (rd + 1) < fl_rounds:
            save_mid_checkpoint(agents, history, rd, phase_dir, 'phaseB4')
            torch.save([p.clone() for p in global_params], phase_dir / 'global_dqn_mid.pt')

    # Final evaluation
    log(f'\n  [Final Eval] Fuzzy-Augmented:')
    results = {}
    for i in range(5):
        m = evaluate_agent(agents[i], envs[i], n_episodes=10)
        log(f'    {LINK_SHORT[i]}: MSR={m["msr"]:.4f} FPR={m["fpr"]:.4f} '
            f'FNR={m["fnr"]:.4f} Reward={m["avg_reward"]:.4f}')
        results[LINK_NAMES[i]] = m
        if phase_dir:
            ckpt_dir = phase_dir / LINK_NAMES[i]
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(agents[i].q_net.state_dict(), ckpt_dir / 'best_dqn.pt')

    msr_list = [results[ln]['msr'] for ln in LINK_NAMES]
    results['global'] = {
        'mean_msr': float(np.mean(msr_list)),
        'std_msr': float(np.std(msr_list)),
        'jain_fairness': float(compute_jain_fairness(msr_list)),
        'mean_fpr': float(np.mean([results[ln]['fpr'] for ln in LINK_NAMES])),
        'mean_fnr': float(np.mean([results[ln]['fnr'] for ln in LINK_NAMES])),
        'mean_reward': float(np.mean([results[ln]['avg_reward'] for ln in LINK_NAMES])),
    }
    results['history'] = history
    log(f'    Global: MSR={results["global"]["mean_msr"]:.4f} '
        f'JFI={results["global"]["jain_fairness"]:.4f}')

    if phase_dir:
        with open(phase_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        torch.save([p.clone() for p in global_params], phase_dir / 'global_dqn.pt')
        remove_mid_checkpoint(phase_dir, 'phaseB4')
        mid_gp = phase_dir / 'global_dqn_mid.pt'
        if mid_gp.exists():
            os.remove(mid_gp)
    return results


# ============================================================================
# Main Training Pipeline
# ============================================================================
def train_dataset(dataset_key):
    global _log_file
    ds_label = DS_LABELS[dataset_key]
    ds_dir = RESULTS_BASE / dataset_key
    _log_file = ds_dir / 'training.log'
    progress_file = ds_dir / 'progress_state.json'
    charts_dir = ds_dir / 'charts'
    afras_base = ds_dir / 'afras_fl'

    os.makedirs(ds_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)
    seed_everything(SEED)

    log('=' * 70)
    log(f'  FL-RLMAMS Phase 4 Strategy B: MOSAR + AFRAS-FL + Fuzzy RL')
    log(f'  Dataset: {ds_label}')
    log(f'  Device: {DEVICE}')
    if torch.cuda.is_available():
        log(f'  GPU: {torch.cuda.get_device_name(0)}')
    log(f'  State dim: {STATE_DIM}, Actions: {N_ACTIONS}')
    log(f'  Phases: B0(AFRAS-FL) + B1({INDEPENDENT_EPISODES} eps) + '
        f'B2(FedAvg {FL_ROUNDS}r) + B3(FedProx {FL_ROUNDS}r) + '
        f'B4(Fuzzy {FL_ROUNDS}r) + B5(Master)')
    log('=' * 70)

    # Load feature order
    ds_cfg = DATASETS[dataset_key]
    with open(ds_cfg['json_file'], 'r') as f:
        feature_data = json.load(f)
        if isinstance(feature_data, dict):
            feature_order = feature_data.get('selected_features', list(feature_data.keys()))
        else:
            feature_order = feature_data
    log(f'  Features: {len(feature_order)}')

    progress = load_progress(progress_file)

    # ================================================================
    # Step 1: Load Data & Mamba Pre-computation
    # ================================================================
    log(f'\n{"="*60}')
    log(f'  Step 1: Loading Data & Mamba Pre-computation')
    log(f'{"="*60}')

    # RLS mode uses a different set of windows (fewer rows) than the
    # legacy flow-level pipeline, so its Mamba outputs are NOT compatible
    # with the shared Strategy A cache. Isolate its cache and, crucially,
    # do not fall back to the Strategy A cache.
    cache_tag = f'_rls_{RLS_SPLIT}' if USE_RLS else ''
    mamba_cache_dir = ds_dir / f'mamba_cache{cache_tag}'
    if USE_RLS:
        mamba_cache_a = None
    else:
        mamba_cache_a = STRATEGY_A_RESULTS / dataset_key / 'mamba_cache'
    os.makedirs(mamba_cache_dir, exist_ok=True)

    all_X, all_y, all_softmax, all_embed_pca, all_envs = [], [], [], [], []
    all_X_windows = []

    for i, link_name in enumerate(LINK_NAMES):
        log(f'  [{i+1}/5] {link_name}')
        csv_path = ds_cfg['data_dir'] / f'{link_name}.csv'
        X, y = load_link_csv(csv_path, feature_order,
                             dataset_key=dataset_key, link_name=link_name)
        log(f'    Creating windows (W={WINDOW_SIZE}, S={STRIDE})...')
        X_w, y_w = create_windows(X, y, WINDOW_SIZE, STRIDE)
        log(f'      Windows: {fmt(len(y_w))}')
        all_X_windows.append(X_w)

        # Load/compute Mamba outputs
        softmax_cache = mamba_cache_dir / f'{link_name}_softmax.npy'
        embed_cache = mamba_cache_dir / f'{link_name}_embed.npy'

        # Try Strategy A cache first (skipped in RLS mode: its cached
        # arrays were built on the full flow-level stream and are not
        # size-compatible with the record-level TRAIN split).
        softmax_cache_a = (mamba_cache_a / f'{link_name}_softmax.npy'
                           if mamba_cache_a is not None else None)
        embed_cache_a = (mamba_cache_a / f'{link_name}_embed.npy'
                         if mamba_cache_a is not None else None)

        def _cache_size_ok(path):
            try:
                arr = np.load(path, mmap_mode='r')
                return arr.shape[0] == len(X_w)
            except Exception:
                return False

        if (softmax_cache.exists() and embed_cache.exists()
                and _cache_size_ok(softmax_cache)):
            log(f'      Loading cached Mamba outputs...')
            softmax = np.load(softmax_cache)
            embed_raw = np.load(embed_cache)
        elif (softmax_cache_a is not None and softmax_cache_a.exists()
              and embed_cache_a.exists() and _cache_size_ok(softmax_cache_a)):
            log(f'      Loading cached Mamba outputs from Strategy A...')
            softmax = np.load(softmax_cache_a)
            embed_raw = np.load(embed_cache_a)
            np.save(softmax_cache, softmax)
            np.save(embed_cache, embed_raw)
        else:
            # Try several known locations for the Mamba checkpoint.
            ckpt_candidates = [
                ds_cfg['data_dir'].parent.parent / 'stage3_results' / dataset_key / link_name / f'{link_name}_Mamba_Only.pt',
                ds_cfg['data_dir'] / f'{link_name}_Mamba_Only.pt',
                PROJECT / 'experiments' / 'results' / dataset_key / 'checkpoints' / f'{link_name}_Mamba_Only.pt',
                PROJECT / 'experiments' / 'results' / 'checkpoints' / 'stage3' / f'{link_name}_Mamba_Only.pt',
            ]
            ckpt_path = next((p for p in ckpt_candidates if p.exists()), None)
            if ckpt_path is None:
                raise FileNotFoundError(
                    f'No Mamba checkpoint found for {link_name}. Tried: '
                    + ', '.join(str(p) for p in ckpt_candidates))
            log(f'      Mamba ckpt: {ckpt_path}')
            extractor = MambaEmbeddingExtractor(ckpt_path, n_features=len(feature_order))
            softmax, embed_raw = extractor.extract_all(X_w)
            np.save(softmax_cache, softmax)
            np.save(embed_cache, embed_raw)
            del extractor; gc.collect()

        extractor_pca = MambaEmbeddingExtractor.__new__(MambaEmbeddingExtractor)
        extractor_pca.pca = PCA(n_components=16, random_state=SEED)
        extractor_pca.pca.fit(embed_raw)
        embed_pca = extractor_pca.pca.transform(embed_raw).astype(np.float32)
        log(f'      PCA fitted: 128D -> 16D (explained: {extractor_pca.pca.explained_variance_ratio_.sum():.3f})')

        all_X.append(X)
        all_y.append(y)
        all_softmax.append(softmax)
        all_embed_pca.append(embed_pca)

        del X, y, embed_raw; gc.collect()

    total_windows = sum(len(s) for s in all_softmax)
    log(f'  All 5 links loaded. Total windows: {fmt(total_windows)}')

    # ================================================================
    # Step 2: Phase B0 — AFRAS-FL Pipeline
    # ================================================================
    phase_b0_key = f'{dataset_key}::phase_b0'
    afras_packages = {}

    if phase_b0_key not in progress['completed']:
        log(f'\n{"="*60}')
        log(f'  Step 2: Phase B0 — AFRAS-FL Pipeline (7 phases)')
        log(f'{"="*60}')

        # Select top 2 features per link (use XAI importance or first 2)
        # For simplicity, use the 2 most variant features per link
        for i, link_name in enumerate(LINK_NAMES):
            afras_dir = afras_base / link_name
            X_w = all_X_windows[i]
            y_w = np.array([all_softmax[i][j].argmax() for j in range(len(all_softmax[i]))], dtype=np.int64)
            # Use window-level mean for 2D features
            X_mean = X_w.mean(axis=1)  # (N, 14)

            # Select top 2 features by variance
            variances = X_mean.var(axis=0)
            top2_idx = np.argsort(-variances)[:2]
            top2_names = [feature_order[idx] for idx in top2_idx]
            X_2d = X_mean[:, top2_idx].astype(np.float32)
            log(f'\n    {link_name}: Top 2 features = {top2_names}')

            pipeline = AFRASFLPipeline(X_2d, y_w, top2_names, link_name, afras_dir)
            package = pipeline.run_all()
            afras_packages[link_name] = package

        progress['completed'].append(phase_b0_key)
        save_progress(progress, progress_file)
        log(f'  Phase B0: COMPLETED')
    else:
        log(f'  Phase B0: COMPLETED (loading from cache)')
        for i, link_name in enumerate(LINK_NAMES):
            deploy_file = afras_base / link_name / 'phase7_deployed.json'
            if deploy_file.exists():
                with open(deploy_file, 'r') as f:
                    afras_packages[link_name] = json.load(f)

    # ================================================================
    # Build Fuzzy Inference Engines
    # ================================================================
    fuzzy_engines = {}
    for link_name in LINK_NAMES:
        pkg = afras_packages.get(link_name, {})
        if pkg.get('rules') and pkg.get('membership_params'):
            fuzzy_engines[link_name] = FuzzyInferenceEngine(
                rules=pkg['rules'],
                membership_params=pkg['membership_params'],
                feature_names=pkg.get('feature_names', []),
            )
        else:
            fuzzy_engines[link_name] = FuzzyInferenceEngine([], {}, [])

    # ================================================================
    # Create Environments (with MOSAR + Fuzzy)
    # ================================================================
    envs = []
    agents = [DQNAgent() for _ in range(5)]
    for i, link_name in enumerate(LINK_NAMES):
        env = SDNEnvironmentB(
            X_windows=all_X_windows[i],
            y_windows=np.array([all_softmax[i][j].argmax() for j in range(len(all_softmax[i]))], dtype=np.int64),
            mamba_softmax=all_softmax[i],
            mamba_embed_pca=all_embed_pca[i],
            link_idx=i,
            feature_order=feature_order,
            fuzzy_engine=fuzzy_engines[link_name],
            use_mosar=True,
            use_fuzzy_reward=False,  # Will enable in Phase B4
            max_steps=MAX_STEPS_PER_EPISODE,
        )
        envs.append(env)
        log(f'      Env {link_name}: {env.n_windows} windows, DQN: {agents[i].n_params} params')

    # Free windowed data
    del all_X_windows; gc.collect()

    # ================================================================
    # Step 3: Phase B1 — Independent DQN + MOSAR
    # ================================================================
    phase_b1_key = f'{dataset_key}::phase_b1'
    phase_b1_dir = ds_dir / 'mosar_independent'

    if phase_b1_key not in progress['completed']:
        log(f'\n{"="*60}')
        log(f'  Step 3: Phase B1 — Independent DQN + MOSAR')
        log(f'{"="*60}')
        results_b1 = train_independent_mosar(agents, envs, INDEPENDENT_EPISODES, phase_b1_dir)
        progress['completed'].append(phase_b1_key)
        progress['results'][phase_b1_key] = results_b1.get('global', {})
        save_progress(progress, progress_file)
    else:
        log(f'  Phase B1: COMPLETED (loading)')
        res_file = phase_b1_dir / 'results.json'
        if res_file.exists():
            with open(res_file) as f:
                results_b1 = json.load(f)
        else:
            results_b1 = {'global': progress['results'].get(phase_b1_key, {})}
        # Load agent weights
        for i in range(5):
            ckpt_path = phase_b1_dir / LINK_NAMES[i] / 'best_dqn.pt'
            if ckpt_path.exists():
                agents[i].q_net.load_state_dict(
                    torch.load(ckpt_path, weights_only=True, map_location=DEVICE))
                agents[i].target_net.load_state_dict(agents[i].q_net.state_dict())

    # ================================================================
    # Step 4: Phase B2 — FedAvg + MOSAR
    # ================================================================
    phase_b2_key = f'{dataset_key}::phase_b2'
    phase_b2_dir = ds_dir / 'mosar_fedavg'

    if phase_b2_key not in progress['completed']:
        log(f'\n{"="*60}')
        log(f'  Step 4: Phase B2 — FedAvg + MOSAR')
        log(f'{"="*60}')
        results_b2 = train_fl_mosar(agents, envs, FL_ROUNDS, LOCAL_EPISODES_PER_ROUND,
                                     algo='fedavg', phase_dir=phase_b2_dir)
        progress['completed'].append(phase_b2_key)
        progress['results'][phase_b2_key] = results_b2.get('global', {})
        save_progress(progress, progress_file)
    else:
        log(f'  Phase B2: COMPLETED (loading)')
        res_file = phase_b2_dir / 'results.json'
        if res_file.exists():
            with open(res_file) as f:
                results_b2 = json.load(f)
        else:
            results_b2 = {'global': progress['results'].get(phase_b2_key, {})}
        for i in range(5):
            ckpt_path = phase_b2_dir / LINK_NAMES[i] / 'best_dqn.pt'
            if ckpt_path.exists():
                agents[i].q_net.load_state_dict(
                    torch.load(ckpt_path, weights_only=True, map_location=DEVICE))
                agents[i].target_net.load_state_dict(agents[i].q_net.state_dict())

    # ================================================================
    # Step 5: Phase B3 — FedProx + MOSAR
    # ================================================================
    phase_b3_key = f'{dataset_key}::phase_b3'
    phase_b3_dir = ds_dir / 'mosar_fedprox'

    if phase_b3_key not in progress['completed']:
        log(f'\n{"="*60}')
        log(f'  Step 5: Phase B3 — FedProx + MOSAR')
        log(f'{"="*60}')
        results_b3 = train_fl_mosar(agents, envs, FL_ROUNDS, LOCAL_EPISODES_PER_ROUND,
                                     algo='fedprox', mu=FEDPROX_MU, phase_dir=phase_b3_dir)
        progress['completed'].append(phase_b3_key)
        progress['results'][phase_b3_key] = results_b3.get('global', {})
        save_progress(progress, progress_file)
    else:
        log(f'  Phase B3: COMPLETED (loading)')
        res_file = phase_b3_dir / 'results.json'
        if res_file.exists():
            with open(res_file) as f:
                results_b3 = json.load(f)
        else:
            results_b3 = {'global': progress['results'].get(phase_b3_key, {})}
        for i in range(5):
            ckpt_path = phase_b3_dir / LINK_NAMES[i] / 'best_dqn.pt'
            if ckpt_path.exists():
                agents[i].q_net.load_state_dict(
                    torch.load(ckpt_path, weights_only=True, map_location=DEVICE))
                agents[i].target_net.load_state_dict(agents[i].q_net.state_dict())

    # ================================================================
    # Step 6: Phase B4 — Fuzzy-Augmented Re-training
    # ================================================================
    phase_b4_key = f'{dataset_key}::phase_b4'
    phase_b4_dir = ds_dir / 'fuzzy_retrain'

    if phase_b4_key not in progress['completed']:
        log(f'\n{"="*60}')
        if USE_PBRS:
            log(f'  Step 6: Phase B4 — PBRS Re-training  (lambda_PBRS={LAMBDA_PBRS}, gamma={PBRS_GAMMA})')
        else:
            log(f'  Step 6: Phase B4 — Fuzzy-Augmented Re-training')
        log(f'{"="*60}')
        # Enable shaping (fuzzy or PBRS)
        for env in envs:
            env.use_fuzzy_reward = True
            env.use_pbrs_reward = USE_PBRS
        results_b4 = train_fuzzy_augmented(agents, envs, FL_ROUNDS, LOCAL_EPISODES_PER_ROUND,
                                            mu=FEDPROX_MU, phase_dir=phase_b4_dir)
        progress['completed'].append(phase_b4_key)
        progress['results'][phase_b4_key] = results_b4.get('global', {})
        save_progress(progress, progress_file)
    else:
        log(f'  Phase B4: COMPLETED (loading)')
        res_file = phase_b4_dir / 'results.json'
        if res_file.exists():
            with open(res_file) as f:
                results_b4 = json.load(f)
        else:
            results_b4 = {'global': progress['results'].get(phase_b4_key, {})}
        for i in range(5):
            ckpt_path = phase_b4_dir / LINK_NAMES[i] / 'best_dqn.pt'
            if ckpt_path.exists():
                agents[i].q_net.load_state_dict(
                    torch.load(ckpt_path, weights_only=True, map_location=DEVICE))
                agents[i].target_net.load_state_dict(agents[i].q_net.state_dict())

    # ================================================================
    # Step 7: Phase B5 — Master Agent
    # ================================================================
    phase_b5_key = f'{dataset_key}::phase_b5'
    master_dir = ds_dir / 'master_agent'

    if phase_b5_key not in progress['completed']:
        log(f'\n{"="*60}')
        log(f'  Step 7: Phase B5 — Master Agent Load Balancing')
        log(f'{"="*60}')
        os.makedirs(master_dir, exist_ok=True)

        # Load best phase weights (B4 fuzzy-augmented)
        for i in range(5):
            ckpt_path = phase_b4_dir / LINK_NAMES[i] / 'best_dqn.pt'
            if ckpt_path.exists():
                agents[i].q_net.load_state_dict(
                    torch.load(ckpt_path, weights_only=True, map_location=DEVICE))
                agents[i].target_net.load_state_dict(agents[i].q_net.state_dict())

        master = MasterAgent()
        master_stats = master.evaluate_over_episodes(agents, envs, n_episodes=50)
        log(f'\n  Master Agent Results:')
        log(f'    Health: {[f"{h:.4f}" for h in master_stats["health_mean"]]}')
        log(f'    JFI Before: {master_stats["jfi_before_mean"]:.4f}')
        log(f'    JFI After:  {master_stats["jfi_after_mean"]:.4f}')
        log(f'    Best Link:  {master_stats["best_link_frequency"]}')

        with open(master_dir / 'results.json', 'w') as f:
            json.dump(master_stats, f, indent=2)

        progress['completed'].append(phase_b5_key)
        progress['results'][phase_b5_key] = {
            'health_mean': master_stats['health_mean'],
            'jfi_before': master_stats['jfi_before_mean'],
            'jfi_after': master_stats['jfi_after_mean'],
        }
        save_progress(progress, progress_file)
    else:
        log(f'  Phase B5: COMPLETED (loading)')
        res_file = master_dir / 'results.json'
        if res_file.exists():
            with open(res_file) as f:
                master_stats = json.load(f)
        else:
            master_stats = progress['results'].get(phase_b5_key, {})

    # ================================================================
    # Save Combined Results
    # ================================================================
    combined = {
        'dataset': dataset_key,
        'label': ds_label,
        'phase_b0': {ln: afras_packages.get(ln, {}).get('summary', {}) for ln in LINK_NAMES},
        'phase_b1': results_b1.get('global', {}),
        'phase_b2': results_b2.get('global', {}),
        'phase_b3': results_b3.get('global', {}),
        'phase_b4': results_b4.get('global', {}),
        'master_agent': master_stats if isinstance(master_stats, dict) else {},
    }
    with open(ds_dir / 'results.json', 'w') as f:
        json.dump(combined, f, indent=2)

    log(f'\n{"="*60}')
    log(f'  Strategy B COMPLETED for {ds_label}')
    log(f'  Results saved to: {ds_dir}')
    log(f'{"="*60}')

    return combined


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['ciciot2023', 'diad2024', 'all'],
                        default='all')
    parser.add_argument('--lambda-fuzzy', type=float, default=None,
                        help='Override LAMBDA_FUZZY (fuzzy shaping weight). '
                             'Default: 0.25 (or env LAMBDA_FUZZY).')
    parser.add_argument('--mu-link', type=float, default=None,
                        help='Override MU_LINK (inter-agent link reward weight). '
                             'Default: 0.10 (or env MU_LINK).')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix appended to results_StrategyB folder '
                             '(e.g. "_lam0.10"). Use a leading underscore to '
                             'keep the default folder untouched.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override the training seed (default 42). Used by '
                             'the multi-seed replication study.')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Override MAX_STEPS_PER_EPISODE (default 5000). Used to speed up multi-seed runs.')
    parser.add_argument('--use-rls', action='store_true',
                        help='Load features from the record-level split '
                             '(leakage-free preprocessing; reviewer R2-6). '
                             'Requires scripts/data/record_level_split.py to '
                             'have been run beforehand.')
    parser.add_argument('--rls-root', type=str, default=None,
                        help='Override the RLS root folder. Defaults to '
                             'data/unified_14features_5links_rls.')
    parser.add_argument('--rls-split', choices=['train', 'val', 'test'],
                        default='train',
                        help='Which split feeds the RL agent when --use-rls '
                             'is on. Default: train.')
    parser.add_argument('--scalar-reward', action='store_true',
                        help='Ablation cell (a): replace MOSAR five-component '
                             'reward with a single R_detection scalar '
                             '(+1 correct / -0.5 wrong). Also forces '
                             '--lambda-fuzzy 0 and --mu-link 0.')
    parser.add_argument('--use-pbrs', action='store_true',
                        help='PBRS baseline (reviewer R13/L3): replace the '
                             'AFRAS-FL fuzzy shaping term in Phase B4 with a '
                             'standard potential-based shaping term '
                             'F(s,s\') = gamma * Phi(s\') - Phi(s), where Phi is '
                             'the Mamba-softmax attack probability. Everything '
                             'else in the pipeline is unchanged.')
    parser.add_argument('--lambda-pbrs', type=float, default=None,
                        help='Override LAMBDA_PBRS (defaults to LAMBDA_FUZZY).')
    args = parser.parse_args()

    # CLI overrides take precedence over environment variables
    global LAMBDA_FUZZY, MU_LINK, RESULTS_BASE, SEED, MAX_STEPS_PER_EPISODE
    global USE_RLS, RLS_ROOT, RLS_SPLIT, SCALAR_REWARD
    global USE_PBRS, LAMBDA_PBRS
    if args.scalar_reward:
        SCALAR_REWARD = True
        LAMBDA_FUZZY = 0.0
        MU_LINK = 0.0
        log('  SCALAR_REWARD mode ON: MOSAR replaced by R_detection scalar; '
            'LAMBDA_FUZZY and MU_LINK forced to 0.')
    if args.use_pbrs:
        USE_PBRS = True
        if args.lambda_pbrs is not None:
            LAMBDA_PBRS = args.lambda_pbrs
        elif args.lambda_fuzzy is not None:
            LAMBDA_PBRS = args.lambda_fuzzy
        log(f'  USE_PBRS mode ON: Phase B4 fuzzy shaping replaced by '
            f'PBRS F(s,s\') = gamma * Phi(s\') - Phi(s) with '
            f'LAMBDA_PBRS={LAMBDA_PBRS}, PBRS_GAMMA={PBRS_GAMMA}, '
            f'attack_class={PBRS_ATTACK_CLASS}.')
    if args.use_rls:
        USE_RLS = True
        RLS_SPLIT = args.rls_split
        RLS_ROOT = (Path(args.rls_root) if args.rls_root else
                    DATA_DIR / 'unified_14features_5links_rls')
        if not RLS_ROOT.exists():
            raise FileNotFoundError(
                f'--use-rls requested but RLS root does not exist: {RLS_ROOT}. '
                f'Run: python scripts/data/record_level_split.py '
                f'--input-root data/unified_14features_5links '
                f'--output-root {RLS_ROOT} --layout 14feat')
        log(f'  RLS mode ON  root={RLS_ROOT}  split={RLS_SPLIT}')
    if args.lambda_fuzzy is not None:
        LAMBDA_FUZZY = float(args.lambda_fuzzy)
    if args.mu_link is not None:
        MU_LINK = float(args.mu_link)
    if args.max_steps is not None:
        MAX_STEPS_PER_EPISODE = int(args.max_steps)
        log(f'  MAX_STEPS_PER_EPISODE overridden to {MAX_STEPS_PER_EPISODE}')
    if args.seed is not None:
        SEED = int(args.seed)
        # override the imported constant everywhere
        import train_isolated as _ti
        _ti.SEED = SEED
        import random as _rnd, numpy as _np
        _rnd.seed(SEED); _np.random.seed(SEED)
        import torch as _t
        _t.manual_seed(SEED)
        if _t.cuda.is_available(): _t.cuda.manual_seed_all(SEED)
    if args.suffix:
        RESULTS_BASE = PROJECT / 'experiments' / f'results_StrategyB{args.suffix}'

    log('=' * 70)
    log(f'  Ablation config: LAMBDA_FUZZY={LAMBDA_FUZZY}, MU_LINK={MU_LINK}')
    log(f'  Results dir    : {RESULTS_BASE}')
    log('=' * 70)

    if args.dataset == 'all':
        for ds_key in ['ciciot2023', 'diad2024']:
            train_dataset(ds_key)
    else:
        train_dataset(args.dataset)


if __name__ == '__main__':
    main()
