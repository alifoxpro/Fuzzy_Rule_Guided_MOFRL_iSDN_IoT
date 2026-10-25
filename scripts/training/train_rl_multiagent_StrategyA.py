# -*- coding: utf-8 -*-
"""
============================================================================
FL-RLMAMS Phase 4 — Strategy A: Multi-Agent DQN + FL + Master Agent
============================================================================
5 SDN Links = 5 DQN Agents with:
  - Frozen Mamba-Only (detection) → severity + 128D embedding
  - Trainable DQN → action: ALLOW / RATE_LIMIT / BLOCK
  - FL server aggregates DQN weights across 5 agents
  - Master Agent: Best Link Ranking + Load Balancing (post-training)

Training Phases:
  Phase A: Independent DQN (no FL, no neighbors)
  Phase B: FL-DQN with FedAvg
  Phase C: FL-DQN with FedProx
  Phase D: Master Agent Load Balancing (inference only)

Usage:
  python train_rl_multiagent_StrategyA.py --dataset ciciot2023
  python train_rl_multiagent_StrategyA.py --dataset diad2024
  python train_rl_multiagent_StrategyA.py --dataset all
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
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

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
RLS_ROOT = None
RLS_SPLIT = 'train'

# ============================================================================
# Configuration
# ============================================================================
PROJECT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT / 'data'
RESULTS_BASE = PROJECT / 'experiments' / 'results_StrategyA'

MAX_ROWS = 3_000_000  # Per link CSV limit

# DQN Hyperparameters
STATE_DIM = 28           # 4 link_metrics + 4 lfa_state + 16 mamba_embed + 4 neighbor
N_ACTIONS = 3            # ALLOW=0, RATE_LIMIT=1, BLOCK=2
DQN_HIDDEN = [128, 64, 32]
DQN_LR = 1e-3
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BUFFER_SIZE = 10000
DQN_BATCH = 64
TAU = 0.005              # Soft target update
MAX_STEPS_PER_EPISODE = 5000

# Training
INDEPENDENT_EPISODES = 100
FL_ROUNDS = 30
LOCAL_EPISODES_PER_ROUND = 5
FEDPROX_MU = 0.01
BLEND_ALPHA = 0.7        # 0.7 local + 0.3 global

# Reward weights
W_CORRECT = 0.4
W_SECURITY = 0.3
W_SERVICE = 0.2
W_NEIGHBOR = 0.1

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
PHASE_COLORS = {'Independent': '#E74C3C', 'FedAvg': '#3498DB', 'FedProx': '#27AE60'}
PHASE_HATCHES = {'Independent': '///', 'FedAvg': '...', 'FedProx': 'xxx'}
LINK_COLORS = ['#E74C3C', '#3498DB', '#27AE60', '#F39C12', '#9B59B6']
ACTION_COLORS = ['#27AE60', '#F39C12', '#E74C3C']  # ALLOW, LIMIT, BLOCK

# IEEE Academic Style
plt.rcParams.update({
    'font.size': 10, 'font.family': 'serif',
    'axes.labelsize': 11, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
})

IEEE_WIDTH_SINGLE = 3.5  # inches
IEEE_WIDTH_DOUBLE = 7.16  # inches


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
# Data Loading (from CSV)
# ============================================================================
def load_link_csv(csv_path, feature_order, dataset_key=None, link_name=None):
    """Load link CSV, subsample if needed, return X (float32), y (int64).

    When USE_RLS is on, dispatches to load_link_rls (leakage-free
    preprocessing per reviewer R2-6). Otherwise falls back to the legacy
    flow-level loader that later fits its scaler on the concatenated data.
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
    del chunks
    gc.collect()
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
    del df
    gc.collect()
    log(f'      Class dist: H={np.sum(y==0):,} M={np.sum(y==1):,} L={np.sum(y==2):,}')
    return X, y


# ============================================================================
# Mamba Embedding Extractor (frozen)
# ============================================================================
class MambaEmbeddingExtractor:
    """Loads frozen Mamba-Only, pre-computes softmax + 128D embeddings."""

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
        all_softmax = []
        all_embeds = []
        batch_size = 512
        for i in range(0, len(X_windows), batch_size):
            batch = torch.FloatTensor(X_windows[i:i+batch_size]).to(self.device)
            x = self.model.input_proj(batch)
            x = self.model.mamba(x)
            embed = x[:, -1, :]
            logits = self.model.fc(embed)
            probs = F.softmax(logits, dim=-1)
            all_softmax.append(probs.cpu().numpy())
            all_embeds.append(embed.cpu().numpy())
        softmax = np.concatenate(all_softmax, axis=0)
        embeddings = np.concatenate(all_embeds, axis=0)
        return softmax, embeddings

    def fit_pca(self, embeddings, n_components=16):
        self.pca = PCA(n_components=n_components, random_state=SEED)
        self.pca.fit(embeddings)
        log(f'      PCA fitted: 128D -> {n_components}D (explained: {self.pca.explained_variance_ratio_.sum():.3f})')

    def transform_pca(self, embeddings):
        return self.pca.transform(embeddings).astype(np.float32)


# ============================================================================
# SDN Environment (offline Gym-like)
# ============================================================================
class SDNEnvironment:
    """Offline RL environment for one SDN link."""

    def __init__(self, X_windows, y_windows, mamba_softmax, mamba_embed_pca,
                 link_idx=0, feature_order=None, max_steps=MAX_STEPS_PER_EPISODE):
        self.X_windows = X_windows
        self.y_windows = y_windows
        self.mamba_softmax = mamba_softmax
        self.mamba_embed = mamba_embed_pca
        self.link_idx = link_idx
        self.n_windows = len(y_windows)
        self.max_steps = min(max_steps, self.n_windows)
        self.feature_order = feature_order

        self._precompute_link_metrics()

        self.step_idx = 0
        self.episode_indices = None
        self.recent_severities = deque(maxlen=10)
        self.neighbor_envs = []

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

    def reset(self):
        indices = np.random.permutation(self.n_windows)[:self.max_steps]
        self.episode_indices = indices
        self.step_idx = 0
        self.recent_severities.clear()
        return self._get_state()

    def step(self, action):
        idx = self.episode_indices[self.step_idx]
        true_severity = self.y_windows[idx]
        pred_severity = self.mamba_softmax[idx].argmax()
        self.recent_severities.append(pred_severity)
        neighbor_actions = self._get_neighbor_actions()
        reward = self._compute_reward(action, true_severity, neighbor_actions)
        self.step_idx += 1
        done = self.step_idx >= len(self.episode_indices)
        info = {
            'true_severity': int(true_severity),
            'pred_severity': int(pred_severity),
            'action': int(action),
            'correct': int(action == self._optimal_action(true_severity)),
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

    @staticmethod
    def _optimal_action(severity):
        return {0: 2, 1: 1, 2: 0}[severity]

    @staticmethod
    def _compute_reward(action, true_severity, neighbor_actions=None):
        optimal = SDNEnvironment._optimal_action(true_severity)
        R_correct = 1.0 if action == optimal else -0.5
        R_security = -1.0 if (action == 0 and true_severity == 0) else 0.0
        R_service = -1.0 if (action == 2 and true_severity == 2) else 0.0
        R_neighbor = 0.0
        if neighbor_actions is not None and len(neighbor_actions) > 0:
            consensus = sum(1 for a in neighbor_actions if a == action) / len(neighbor_actions)
            R_neighbor = 0.2 * (consensus - 0.5)
        return W_CORRECT * R_correct + W_SECURITY * R_security + \
               W_SERVICE * R_service + W_NEIGHBOR * R_neighbor


# ============================================================================
# DQN Network
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


# ============================================================================
# Experience Replay Buffer
# ============================================================================
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


# ============================================================================
# DQN Agent
# ============================================================================
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
        """Get Q-values for a state (for Master Agent analysis)."""
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
    fp_count = 0
    fn_count = 0
    benign_count = 0
    attack_count = 0

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, eval_mode=True)
            next_state, reward, done, info = env.step(action)
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
            state = next_state

    msr = total_correct / max(total_steps, 1)
    fpr = fp_count / max(benign_count, 1)
    fnr = fn_count / max(attack_count, 1)
    avg_reward = total_reward / max(total_steps, 1)
    service_avail = 1.0 - fpr

    return {
        'msr': float(msr),
        'fpr': float(fpr),
        'fnr': float(fnr),
        'avg_reward': float(avg_reward),
        'service_availability': float(service_avail),
        'total_steps': int(total_steps),
        'action_distribution': {ACTION_NAMES[k]: int(v) for k, v in action_counts.items()},
        'severity_action_matrix': severity_action_matrix.tolist(),
    }


def compute_jain_fairness(msr_list):
    n = len(msr_list)
    sum_x = sum(msr_list)
    sum_x2 = sum(x**2 for x in msr_list)
    if sum_x2 == 0:
        return 1.0
    return (sum_x ** 2) / (n * sum_x2)


# ============================================================================
# >>>>>>>>>> NEW: Master Agent — Load Balancing (6-Step Algorithm) <<<<<<<<<<
# ============================================================================
class MasterAgent:
    """
    Master-Driven Load Balancing for Multi-Agent SDN System.
    Collects metrics from 5 agents, ranks links, computes routing decisions.
    """

    def __init__(self, n_links=5, capacity_mbps=1000.0):
        self.n_links = n_links
        self.capacity = capacity_mbps
        # Health score weights
        self.w_capacity = 0.40
        self.w_latency = 0.25
        self.w_pktloss = 0.20
        self.w_safety = 0.15
        # Thresholds
        self.overload_thresh = 0.85
        self.healthy_h_thresh = 0.60
        self.healthy_u_thresh = 0.70
        self.jfi_thresh = 0.90

    # -- Step 1: Health Scores --
    def compute_health_scores(self, U, L, PL, A):
        """
        H_i = 0.40 * (1-U_i) + 0.25 * max(0,1-L_i/100) + 0.20 * (1-PL_i) + 0.15 * (1-A_i)
        """
        H = np.zeros(self.n_links)
        for i in range(self.n_links):
            C_i = 1.0 - U[i]
            L_i_norm = max(0.0, 1.0 - L[i] / 100.0)
            P_i = 1.0 - PL[i]
            A_i_safe = 1.0 - A[i]
            H[i] = (self.w_capacity * C_i + self.w_latency * L_i_norm +
                     self.w_pktloss * P_i + self.w_safety * A_i_safe)
        return H

    # -- Step 2: Identify Overloaded and Healthy --
    def identify_links(self, H, U):
        overloaded = [i for i in range(self.n_links) if U[i] > self.overload_thresh]
        healthy = [i for i in range(self.n_links)
                    if H[i] > self.healthy_h_thresh and U[i] < self.healthy_u_thresh]
        healthy.sort(key=lambda i: H[i], reverse=True)
        return overloaded, healthy

    # -- Step 3: Redistribution Plan --
    def compute_redistribution(self, U, H, overloaded, healthy):
        reroute_commands = []
        U_sim = U.copy()  # Simulated utilization for planning
        healthy_pool = list(healthy)

        for i in overloaded:
            excess_i = (U_sim[i] - self.overload_thresh) * self.capacity
            if excess_i <= 0:
                continue
            if not healthy_pool:
                continue

            available_target = (1.0 - U_sim[healthy_pool[0]]) * self.capacity
            if available_target >= excess_i:
                target = healthy_pool[0]
                reroute_commands.append({
                    'from_link': LINK_SHORT[i],
                    'to_link': LINK_SHORT[target],
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
                        'from_link': LINK_SHORT[i],
                        'to_link': LINK_SHORT[j],
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

    # -- Step 4: Routing Probabilities --
    def compute_routing_probabilities(self, H):
        total_h = H.sum()
        if total_h == 0:
            return np.ones(self.n_links) / self.n_links
        return H / total_h

    # -- Step 5: Fairness --
    def compute_fairness(self, U):
        sum_u = U.sum()
        sum_u2 = (U ** 2).sum()
        if sum_u2 == 0:
            return 1.0
        return (sum_u ** 2) / (self.n_links * sum_u2)

    # -- Step 6: Full Pipeline --
    def rank_links(self, U, L, PL, A):
        """
        Full 6-step pipeline.
        Returns: ranking dict with all results.
        """
        H = self.compute_health_scores(U, L, PL, A)
        overloaded, healthy = self.identify_links(H, U)
        reroute_commands, U_after = self.compute_redistribution(U.copy(), H, overloaded, healthy)
        P = self.compute_routing_probabilities(H)
        jfi_before = self.compute_fairness(U)
        jfi_after = self.compute_fairness(U_after)

        ranking_idx = np.argsort(-H)

        return {
            'health_scores': H.tolist(),
            'health_components': {
                LINK_SHORT[i]: {
                    'capacity': float(1.0 - U[i]),
                    'latency': float(max(0, 1.0 - L[i]/100)),
                    'pkt_loss': float(1.0 - PL[i]),
                    'safety': float(1.0 - A[i]),
                    'total': float(H[i]),
                } for i in range(self.n_links)
            },
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
        """Run Master ranking over many episodes, collect statistics."""
        all_rankings = []
        all_health = []
        all_jfi_before = []
        all_jfi_after = []
        all_best_links = []
        all_reroute_counts = []
        all_routing_probs = []

        for ep in range(n_episodes):
            # Run one episode per agent to collect metrics
            U = np.zeros(5)
            L = np.zeros(5)
            PL = np.zeros(5)
            A = np.zeros(5)

            for i in range(5):
                state = envs[i].reset()
                done = False
                fp, benign, total = 0, 0, 0
                traffic_sum, latency_sum, attack_sum = 0.0, 0.0, 0.0

                while not done:
                    action = agents[i].select_action(state, eval_mode=True)
                    next_state, reward, done, info = envs[i].step(action)
                    total += 1
                    if info['true_severity'] == 2:
                        benign += 1
                        if action == 2:
                            fp += 1
                    traffic_sum += state[0]
                    latency_sum += state[1]
                    attack_sum += state[4]  # attack_detected (1=attack, 0=benign)
                    state = next_state

                U[i] = traffic_sum / max(total, 1)
                L[i] = (latency_sum / max(total, 1)) * 100.0
                PL[i] = fp / max(benign, 1)
                A[i] = attack_sum / max(total, 1)

            result = self.rank_links(U, L, PL, A)
            all_rankings.append(result['ranking'])
            all_health.append(result['health_scores'])
            all_jfi_before.append(result['jfi_before'])
            all_jfi_after.append(result['jfi_after'])
            all_best_links.append(result['best_link'])
            all_reroute_counts.append(len(result['reroute_commands']))
            all_routing_probs.append(list(result['routing_probabilities'].values()))

        # Aggregate stats
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
            'all_rankings': all_rankings,
            'all_health': all_health,
            'all_jfi_before': all_jfi_before,
            'all_jfi_after': all_jfi_after,
            'last_result': result,
        }


# ============================================================================
# Mid-Phase Checkpoint Helpers
# ============================================================================
CKPT_EVERY_EP = 10       # Phase A: checkpoint every 10 episodes
CKPT_EVERY_ROUND = 5     # Phase B/C: checkpoint every 5 FL rounds


def save_mid_checkpoint(agents, history, ep_or_round, phase_dir, prefix='phase'):
    """Save agent weights + epsilon + history for mid-phase resume."""
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
    path = phase_dir / f'{prefix}_resume.pt'
    torch.save(ckpt, path)


def load_mid_checkpoint(agents, phase_dir, prefix='phase'):
    """Load mid-phase checkpoint. Returns (start_ep, history) or None."""
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
    """Remove mid-phase checkpoint after successful completion."""
    path = phase_dir / f'{prefix}_resume.pt'
    if path.exists():
        os.remove(path)


# ============================================================================
# Training: Phase A (Independent DQN, no FL) — with mid-phase resume
# ============================================================================
def train_independent(agents, envs, n_episodes, phase_dir):
    log(f'\n  Phase A: Independent DQN Training ({n_episodes} episodes)')
    os.makedirs(phase_dir, exist_ok=True)
    history = {str(i): {'rewards': [], 'msr': [], 'loss': [], 'epsilon': []} for i in range(5)}
    start_ep = 0

    # Try resume from mid-phase checkpoint
    resumed = load_mid_checkpoint(agents, phase_dir, 'phaseA')
    if resumed:
        start_ep, history = resumed[0] + 1, resumed[1]
        log(f'      Resuming Phase A from episode {start_ep}')

    for ep in range(start_ep, n_episodes):
        ep_rewards = []
        for i in range(5):
            state = envs[i].reset()
            done = False
            ep_reward, ep_loss, steps = 0.0, 0.0, 0
            while not done:
                action = agents[i].select_action(state)
                next_state, reward, done, info = envs[i].step(action)
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
            mean_msr = np.mean(msr_list)
            jfi = compute_jain_fairness(msr_list)
            log(f'    Ep {ep+1:3d}/{n_episodes}: Reward={mean_reward:.4f} '
                f'MSR={mean_msr:.4f} JFI={jfi:.4f} eps={agents[0].epsilon:.3f}')

        # Mid-phase checkpoint
        if (ep + 1) % CKPT_EVERY_EP == 0 and (ep + 1) < n_episodes:
            save_mid_checkpoint(agents, history, ep, phase_dir, 'phaseA')

    # Final evaluation
    log(f'\n  [Final Eval] Independent DQN:')
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
    remove_mid_checkpoint(phase_dir, 'phaseA')
    return results


# ============================================================================
# Training: Phase B/C (FL-DQN with FedAvg or FedProx)
# ============================================================================
def train_fl_dqn(agents, envs, fl_rounds, local_episodes, algo='fedavg',
                 mu=0.0, phase_dir=None):
    algo_label = 'FedAvg' if algo == 'fedavg' else f'FedProx(mu={mu})'
    phase_letter = 'B' if algo == 'fedavg' else 'C'
    ckpt_prefix = f'phase{phase_letter}'
    log(f'\n  Phase {phase_letter}: FL-DQN with {algo_label} '
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
        'global_msr': [],
        'global_fpr': [],       # A5: round-drift detection quality (false alarm)
        'global_fnr': [],       # A5: round-drift detection quality (missed attack)
        'mean_loss': [],        # A5: convergence signal, to decouple from detection
        'jain_fairness': [],
    }
    start_rd = 0

    # Try resume from mid-phase checkpoint
    if phase_dir:
        resumed = load_mid_checkpoint(agents, phase_dir, ckpt_prefix)
        if resumed:
            start_rd, history = resumed[0] + 1, resumed[1]
            # Reload global params from checkpoint
            gp_path = phase_dir / 'global_dqn_mid.pt'
            if gp_path.exists():
                global_params = torch.load(gp_path, weights_only=False, map_location=DEVICE)
            log(f'      Resuming Phase {phase_letter} from round {start_rd}')

    # A5 round-drift: backfill new history keys for runs resumed from older checkpoints
    for _k in ('global_fpr', 'global_fnr', 'mean_loss'):
        history.setdefault(_k, [])

    # A5 round-drift: directory for per-round global checkpoints
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
                    next_state, reward, done, info = envs[i].step(action)
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
        history['global_fpr'].append(float(np.nanmean(fpr_list)))
        history['global_fnr'].append(float(np.nanmean(fnr_list)))
        history['mean_loss'].append(float(np.mean(round_losses)) if round_losses else float('nan'))
        history['jain_fairness'].append(float(jfi))

        if (rd + 1) % 5 == 0 or rd == 0:
            log(f'    Round {rd+1:2d}/{fl_rounds}: MSR={mean_msr:.4f} '
                f'JFI={jfi:.4f} eps={agents[0].epsilon:.3f}')

        # Mid-phase checkpoint
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
        # Clean up mid global params
        mid_gp = phase_dir / 'global_dqn_mid.pt'
        if mid_gp.exists():
            os.remove(mid_gp)

    return results


# ============================================================================
# >>>>>>>>>> IEEE Academic Charts (12 Figures) <<<<<<<<<<
# ============================================================================

def generate_all_charts(results_a, results_b, results_c, master_stats, charts_dir, ds_label):
    """Generate all IEEE-quality figures for academic publication."""
    os.makedirs(charts_dir, exist_ok=True)
    log(f'\n  Generating IEEE charts...')

    phase_results = {
        'Independent': results_a,
        'FedAvg': results_b,
        'FedProx': results_c,
    }

    # --- fig1: Training Curves (Phase A) — 2x2 (reward, msr, loss, epsilon) ---
    fig, axes = plt.subplots(2, 2, figsize=(IEEE_WIDTH_DOUBLE, 5.5))
    hist_a = results_a.get('history', {})
    for i in range(5):
        key = str(i)
        if key in hist_a and 'rewards' in hist_a[key]:
            axes[0, 0].plot(hist_a[key]['rewards'], label=LINK_SHORT[i],
                            color=LINK_COLORS[i], linewidth=1.0, alpha=0.8)
            if 'loss' in hist_a[key]:
                axes[0, 1].plot(hist_a[key]['loss'], label=LINK_SHORT[i],
                                color=LINK_COLORS[i], linewidth=1.0, alpha=0.8)
            if 'msr' in hist_a[key]:
                msr_data = hist_a[key]['msr']
                x_msr = list(range(len(msr_data)))
                axes[1, 0].plot(x_msr, msr_data, 'o-', label=LINK_SHORT[i],
                                color=LINK_COLORS[i], linewidth=1.0, markersize=3)
            if 'epsilon' in hist_a[key]:
                if i == 0:  # Same for all agents
                    axes[1, 1].plot(hist_a[key]['epsilon'], '-', color='#E74C3C', linewidth=1.5)
    axes[0, 0].set_xlabel('Episode'); axes[0, 0].set_ylabel('Avg Reward')
    axes[0, 0].set_title('(a) Reward Convergence')
    axes[0, 0].legend(fontsize=6, ncol=3)
    axes[0, 1].set_xlabel('Episode'); axes[0, 1].set_ylabel('Avg Loss')
    axes[0, 1].set_title('(b) TD Loss')
    axes[0, 1].legend(fontsize=6, ncol=3)
    axes[1, 0].set_xlabel('Episode'); axes[1, 0].set_ylabel('MSR')
    axes[1, 0].set_title('(c) Mitigation Success Rate')
    axes[1, 0].legend(fontsize=6, ncol=3)
    axes[1, 1].set_xlabel('Episode'); axes[1, 1].set_ylabel('Epsilon')
    axes[1, 1].set_title('(d) Exploration Decay')
    axes[1, 1].axhline(y=EPSILON_END, color='gray', linestyle='--', linewidth=0.8)
    plt.suptitle(f'Phase A: Independent DQN Training \u2014 {ds_label}', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(charts_dir / 'fig1_training_curves.png', dpi=300, bbox_inches='tight'); plt.close()
    log(f'    Saved: fig1_training_curves.png')

    # --- fig2: Phase Comparison (A vs B vs C) ---
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH_DOUBLE, 3.5))
    metrics = ['mean_msr', 'mean_fpr', 'mean_fnr', 'jain_fairness']
    metric_labels = ['MSR', 'FPR', 'FNR', 'JFI']
    x = np.arange(len(metrics))
    width = 0.25
    for pi, (pname, pres) in enumerate(phase_results.items()):
        glob = pres.get('global', {})
        vals = [glob.get(m, 0) for m in metrics]
        bars = ax.bar(x + pi * width, vals, width, label=pname,
                      color=PHASE_COLORS[pname], hatch=PHASE_HATCHES[pname],
                      edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel('Score')
    ax.set_title(f'Phase Comparison: Independent vs FedAvg vs FedProx \u2014 {ds_label}')
    ax.legend()
    ax.set_ylim(0, 1.15)
    plt.tight_layout()
    plt.savefig(charts_dir / 'fig2_phase_comparison.png', dpi=300, bbox_inches='tight'); plt.close()
    log(f'    Saved: fig2_phase_comparison.png')

    # --- fig3: Per-Link MSR across 3 phases ---
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH_DOUBLE, 3.5))
    x = np.arange(5)
    width = 0.25
    for pi, (pname, pres) in enumerate(phase_results.items()):
        msr_vals = [pres.get(ln, {}).get('msr', 0) for ln in LINK_NAMES]
        ax.bar(x + pi * width, msr_vals, width, label=pname,
               color=PHASE_COLORS[pname], hatch=PHASE_HATCHES[pname],
               edgecolor='black', linewidth=0.5)
    ax.set_xticks(x + width)
    ax.set_xticklabels(LINK_SHORT, fontsize=9)
    ax.set_ylabel('MSR')
    ax.set_title(f'Per-Link Mitigation Success Rate \u2014 {ds_label}')
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(charts_dir / 'fig3_per_link_msr.png', dpi=300, bbox_inches='tight'); plt.close()
    log(f'    Saved: fig3_per_link_msr.png')

    # --- fig4: FL Convergence (FedAvg vs FedProx global MSR) ---
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH_DOUBLE, 3.0))
    hist_b = results_b.get('history', {})
    hist_c = results_c.get('history', {})
    if 'global_msr' in hist_b:
        ax.plot(hist_b.get('rounds', []), hist_b['global_msr'], 'o-',
                label='FedAvg', color=PHASE_COLORS['FedAvg'], linewidth=1.5, markersize=3)
    if 'global_msr' in hist_c:
        ax.plot(hist_c.get('rounds', []), hist_c['global_msr'], 's-',
                label='FedProx', color=PHASE_COLORS['FedProx'], linewidth=1.5, markersize=3)
    ax.set_xlabel('FL Round'); ax.set_ylabel('Global MSR')
    ax.set_title(f'Federated Learning Convergence \u2014 {ds_label}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(charts_dir / 'fig4_fl_convergence.png', dpi=300, bbox_inches='tight'); plt.close()
    log(f'    Saved: fig4_fl_convergence.png')

    # --- fig5: JFI Convergence ---
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH_DOUBLE, 3.0))
    if 'jain_fairness' in hist_b:
        ax.plot(hist_b.get('rounds', []), hist_b['jain_fairness'], 'o-',
                label='FedAvg', color=PHASE_COLORS['FedAvg'], linewidth=1.5, markersize=3)
    if 'jain_fairness' in hist_c:
        ax.plot(hist_c.get('rounds', []), hist_c['jain_fairness'], 's-',
                label='FedProx', color=PHASE_COLORS['FedProx'], linewidth=1.5, markersize=3)
    ax.axhline(y=0.9, color='gray', linestyle='--', linewidth=1, label='Threshold (0.9)')
    ax.set_xlabel('FL Round'); ax.set_ylabel("Jain's Fairness Index")
    ax.set_title(f'Fairness Convergence \u2014 {ds_label}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(charts_dir / 'fig5_jfi_convergence.png', dpi=300, bbox_inches='tight'); plt.close()
    log(f'    Saved: fig5_jfi_convergence.png')

    # --- fig6: Action Distribution per Link (best phase = FedProx) ---
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH_DOUBLE, 3.5))
    best_phase = results_c  # FedProx
    x = np.arange(5)
    bottom = np.zeros(5)
    for ai, aname in enumerate(ACTION_NAMES):
        vals = []
        for ln in LINK_NAMES:
            ad = best_phase.get(ln, {}).get('action_distribution', {})
            total_a = sum(ad.values()) if ad else 1
            vals.append(ad.get(aname, 0) / max(total_a, 1))
        ax.bar(x, vals, 0.5, bottom=bottom, label=aname, color=ACTION_COLORS[ai],
               edgecolor='black', linewidth=0.5)
        bottom += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(LINK_SHORT, fontsize=9)
    ax.set_ylabel('Action Proportion')
    ax.set_title(f'Action Distribution per Link (FedProx) \u2014 {ds_label}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(charts_dir / 'fig6_action_distribution.png', dpi=300, bbox_inches='tight'); plt.close()
    log(f'    Saved: fig6_action_distribution.png')

    # --- fig7: Severity-Action Heatmap (best phase) ---
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH_SINGLE + 1, 3.5))
    total_sam = np.zeros((3, 3), dtype=np.float64)
    for ln in LINK_NAMES:
        sam = best_phase.get(ln, {}).get('severity_action_matrix', [[0]*3]*3)
        total_sam += np.array(sam, dtype=np.float64)
    row_sums = total_sam.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    total_sam_norm = total_sam / row_sums
    sns.heatmap(total_sam_norm, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=ACTION_NAMES, yticklabels=SEV_NAMES,
                ax=ax, vmin=0, vmax=1, linewidths=0.5)
    ax.set_xlabel('Agent Action'); ax.set_ylabel('True Severity')
    ax.set_title(f'Severity vs Action Mapping (FedProx) \u2014 {ds_label}')
    plt.tight_layout()
    plt.savefig(charts_dir / 'fig7_severity_action_heatmap.png', dpi=300, bbox_inches='tight'); plt.close()
    log(f'    Saved: fig7_severity_action_heatmap.png')

    # --- fig8: Master Agent Health Scores (stacked bar) ---
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH_DOUBLE, 3.5))
    last_result = master_stats.get('last_result', {})
    hc = last_result.get('health_components', {})
    x = np.arange(5)
    comp_names = ['capacity', 'latency', 'pkt_loss', 'safety']
    comp_weights = [0.40, 0.25, 0.20, 0.15]
    comp_colors = ['#3498DB', '#F39C12', '#E74C3C', '#27AE60']
    bottom = np.zeros(5)
    for ci, (cname, cw) in enumerate(zip(comp_names, comp_weights)):
        vals = [hc.get(LINK_SHORT[i], {}).get(cname, 0) * cw for i in range(5)]
        ax.bar(x, vals, 0.5, bottom=bottom, label=f'{cname} (w={cw})',
               color=comp_colors[ci], edgecolor='black', linewidth=0.5)
        bottom += np.array(vals)
    # Total health score labels
    for i in range(5):
        total_h = master_stats['health_mean'][i]
        ax.text(i, bottom[i] + 0.01, f'H={total_h:.3f}', ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(LINK_SHORT, fontsize=9)
    ax.set_ylabel('Weighted Health Score')
    ax.set_title(f'Master Agent: Link Health Score Breakdown \u2014 {ds_label}')
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(charts_dir / 'fig8_master_health_scores.png', dpi=300, bbox_inches='tight'); plt.close()
    log(f'    Saved: fig8_master_health_scores.png')

    # --- fig9: Master Routing Probabilities (pie + bar) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_WIDTH_DOUBLE, 3.5))
    rp = master_stats.get('routing_prob_mean', {})
    rp_vals = [rp.get(LINK_SHORT[i], 0.2) for i in range(5)]
    ax1.pie(rp_vals, labels=LINK_SHORT, autopct='%1.1f%%', colors=LINK_COLORS,
            startangle=90, textprops={'fontsize': 8})
    ax1.set_title('(a) Routing Probabilities', fontsize=10, fontweight='bold')

    # Best link frequency
    blf = master_stats.get('best_link_frequency', {})
    bl_names = list(blf.keys())
    bl_vals = [blf[n] for n in bl_names]
    bl_colors = [LINK_COLORS[LINK_SHORT.index(n)] if n in LINK_SHORT else '#999' for n in bl_names]
    ax2.barh(range(len(bl_names)), bl_vals, color=bl_colors, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(range(len(bl_names)))
    ax2.set_yticklabels(bl_names, fontsize=9)
    ax2.set_xlabel('Times Selected as Best')
    ax2.set_title('(b) Best Link Frequency', fontsize=10, fontweight='bold')
    ax2.invert_yaxis()
    plt.suptitle(f'Master Agent: Routing Decisions \u2014 {ds_label}', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(charts_dir / 'fig9_master_routing.png', dpi=300, bbox_inches='tight'); plt.close()
    log(f'    Saved: fig9_master_routing.png')

    # --- fig10: Link Ranking Over Episodes ---
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH_DOUBLE, 3.5))
    all_rankings = master_stats.get('all_rankings', [])
    if all_rankings:
        n_eps = len(all_rankings)
        for li in range(5):
            ranks = []
            for ep_ranking in all_rankings:
                if LINK_SHORT[li] in ep_ranking:
                    ranks.append(ep_ranking.index(LINK_SHORT[li]) + 1)
                else:
                    ranks.append(5)
            ax.plot(range(n_eps), ranks, 'o-', label=LINK_SHORT[li],
                    color=LINK_COLORS[li], linewidth=1.2, markersize=3, alpha=0.8)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Link Rank (1=Best)')
        ax.set_title(f'Master Agent: Link Ranking Over Time \u2014 {ds_label}')
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.invert_yaxis()
        ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(charts_dir / 'fig10_master_ranking_over_time.png', dpi=300, bbox_inches='tight'); plt.close()
    log(f'    Saved: fig10_master_ranking_over_time.png')

    # --- fig11: Comparison Dashboard (2x2) ---
    fig, axes = plt.subplots(2, 2, figsize=(IEEE_WIDTH_DOUBLE, 6.0))

    # (a) MSR improvement across phases
    ax = axes[0, 0]
    phase_names = list(phase_results.keys())
    msr_vals = [phase_results[p].get('global', {}).get('mean_msr', 0) for p in phase_names]
    bars = ax.bar(phase_names, msr_vals, color=[PHASE_COLORS[p] for p in phase_names],
                  edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, msr_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', fontsize=8, fontweight='bold')
    ax.set_ylabel('MSR'); ax.set_title('(a) MSR by Phase')
    ax.set_ylim(0, 1.1)

    # (b) JFI before vs after Master
    ax = axes[0, 1]
    jfi_b = master_stats.get('jfi_before_mean', 0)
    jfi_a = master_stats.get('jfi_after_mean', 0)
    bars = ax.bar(['Before Master', 'After Master'], [jfi_b, jfi_a],
                  color=['#E74C3C', '#27AE60'], edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, [jfi_b, jfi_a]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', fontsize=8, fontweight='bold')
    ax.axhline(y=0.9, color='gray', linestyle='--', linewidth=1)
    ax.set_ylabel('JFI'); ax.set_title('(b) Fairness: Before vs After Master')
    ax.set_ylim(0, 1.1)

    # (c) Utilization before vs after
    ax = axes[1, 0]
    u_before = last_result.get('utilization_before', [0]*5)
    u_after = last_result.get('utilization_after', [0]*5)
    x = np.arange(5)
    ax.bar(x - 0.15, u_before, 0.3, label='Before', color='#E74C3C', alpha=0.8,
           edgecolor='black', linewidth=0.5)
    ax.bar(x + 0.15, u_after, 0.3, label='After', color='#27AE60', alpha=0.8,
           edgecolor='black', linewidth=0.5)
    ax.axhline(y=0.85, color='gray', linestyle='--', linewidth=1, label='Overload (85%)')
    ax.set_xticks(x); ax.set_xticklabels(LINK_SHORT, fontsize=8)
    ax.set_ylabel('Utilization'); ax.set_title('(c) Link Utilization Redistribution')
    ax.legend(fontsize=7)

    # (d) Health Score radar
    ax = axes[1, 1]
    ax.set_axis_off()
    health_mean = master_stats.get('health_mean', [0]*5)
    routing_mean = master_stats.get('routing_prob_mean', {})
    text_lines = ['MASTER AGENT SUMMARY', '-' * 30]
    for i in range(5):
        rp_val = routing_mean.get(LINK_SHORT[i], 0)
        text_lines.append(f'{LINK_SHORT[i]}: H={health_mean[i]:.3f}  P={rp_val:.1%}')
    text_lines.append(f'\nJFI: {jfi_b:.4f} -> {jfi_a:.4f}')
    text_lines.append(f'Avg Reroutes: {master_stats.get("avg_reroutes_per_episode", 0):.1f}/ep')
    best_freq = master_stats.get('best_link_frequency', {})
    if best_freq:
        top_link = max(best_freq, key=best_freq.get)
        text_lines.append(f'Best Link: {top_link} ({best_freq[top_link]} times)')
    ax.text(0.1, 0.95, '\n'.join(text_lines), transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('(d) Master Agent Summary')

    plt.suptitle(f'Strategy A: Comprehensive Dashboard \u2014 {ds_label}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(charts_dir / 'fig11_comparison_dashboard.png', dpi=300, bbox_inches='tight'); plt.close()
    log(f'    Saved: fig11_comparison_dashboard.png')

    # --- fig12: Epsilon Decay ---
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH_SINGLE + 1, 3.0))
    hist_a = results_a.get('history', {})
    if '0' in hist_a and 'epsilon' in hist_a['0']:
        eps_vals = hist_a['0']['epsilon']
        ax.plot(range(len(eps_vals)), eps_vals, '-', color='#E74C3C', linewidth=1.5)
        ax.axhline(y=EPSILON_END, color='gray', linestyle='--', linewidth=1,
                   label=f'eps_end={EPSILON_END}')
        ax.set_xlabel('Episode'); ax.set_ylabel('Epsilon')
        ax.set_title(f'Exploration Decay \u2014 {ds_label}')
        ax.legend()
    plt.tight_layout()
    plt.savefig(charts_dir / 'fig12_epsilon_decay.png', dpi=300, bbox_inches='tight'); plt.close()
    log(f'    Saved: fig12_epsilon_decay.png')

    # --- fig13: Radar Chart — Multi-Metric Phase Comparison ---
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    radar_metrics = ['MSR', 'Service\nAvail.', '1-FPR', '1-FNR', 'JFI']
    n_metrics = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]
    phase_colors_list = ['#E74C3C', '#3498DB', '#27AE60']
    phase_markers = ['o', 's', 'D']
    for pi, (pname, pres) in enumerate(phase_results.items()):
        g = pres.get('global', {})
        vals = [
            g.get('mean_msr', 0),
            1.0 - g.get('mean_fpr', 0),  # Service Availability
            1.0 - g.get('mean_fpr', 0),
            1.0 - g.get('mean_fnr', 0),
            g.get('jain_fairness', 0),
        ]
        vals += vals[:1]
        ax.plot(angles, vals, f'{phase_markers[pi]}-', linewidth=2, markersize=5,
                label=pname, color=phase_colors_list[pi])
        ax.fill(angles, vals, alpha=0.1, color=phase_colors_list[pi])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_metrics, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title(f'Multi-Metric Phase Comparison \u2014 {ds_label}',
                 fontsize=11, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(charts_dir / 'fig13_radar_comparison.png', dpi=300, bbox_inches='tight'); plt.close()
    log(f'    Saved: fig13_radar_comparison.png')

    # --- fig14: Box Plot — MSR Distribution Across Links per Phase ---
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH_DOUBLE, 3.5))
    box_data = []
    box_labels = []
    box_colors = []
    for pname, pres in phase_results.items():
        msr_vals = [pres.get(ln, {}).get('msr', 0) for ln in LINK_NAMES]
        box_data.append(msr_vals)
        box_labels.append(pname)
        box_colors.append(PHASE_COLORS[pname])
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.5,
                    showmeans=True, meanprops=dict(marker='D', markerfacecolor='black', markersize=5))
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('MSR')
    ax.set_title(f'MSR Distribution Across 5 Links \u2014 {ds_label}')
    # Add individual data points
    for xi, (data, color) in enumerate(zip(box_data, box_colors)):
        jitter = np.random.normal(0, 0.04, size=len(data))
        ax.scatter([xi + 1 + j for j in jitter], data, color=color,
                   edgecolor='black', s=40, zorder=5, alpha=0.8)
    plt.tight_layout()
    plt.savefig(charts_dir / 'fig14_msr_boxplot.png', dpi=300, bbox_inches='tight'); plt.close()
    log(f'    Saved: fig14_msr_boxplot.png')

    # --- fig15: Per-Agent FL Convergence (FedAvg vs FedProx per link) ---
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_WIDTH_DOUBLE, 3.5))
    for ax_i, (algo_name, hist_fl) in enumerate([('FedAvg', hist_b), ('FedProx', hist_c)]):
        ax = axes[ax_i]
        pa = hist_fl.get('per_agent', {})
        rounds = hist_fl.get('rounds', [])
        for i in range(5):
            key = str(i)
            if key in pa and 'msr' in pa[key]:
                ax.plot(rounds[:len(pa[key]['msr'])], pa[key]['msr'], 'o-',
                        label=LINK_SHORT[i], color=LINK_COLORS[i],
                        linewidth=1.0, markersize=2, alpha=0.8)
        ax.set_xlabel('FL Round'); ax.set_ylabel('MSR')
        ax.set_title(f'({chr(97+ax_i)}) {algo_name}: Per-Link MSR')
        ax.legend(fontsize=6, ncol=2)
    plt.suptitle(f'Per-Agent FL Convergence \u2014 {ds_label}', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(charts_dir / 'fig15_per_agent_fl_convergence.png', dpi=300, bbox_inches='tight'); plt.close()
    log(f'    Saved: fig15_per_agent_fl_convergence.png')

    # --- fig16: FPR + FNR Comparison (grouped bar per phase) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_WIDTH_DOUBLE, 3.5))
    x = np.arange(5)
    width = 0.25
    for pi, (pname, pres) in enumerate(phase_results.items()):
        fpr_vals = [pres.get(ln, {}).get('fpr', 0) for ln in LINK_NAMES]
        fnr_vals = [pres.get(ln, {}).get('fnr', 0) for ln in LINK_NAMES]
        ax1.bar(x + pi * width, fpr_vals, width, label=pname,
                color=PHASE_COLORS[pname], hatch=PHASE_HATCHES[pname],
                edgecolor='black', linewidth=0.5)
        ax2.bar(x + pi * width, fnr_vals, width, label=pname,
                color=PHASE_COLORS[pname], hatch=PHASE_HATCHES[pname],
                edgecolor='black', linewidth=0.5)
    ax1.set_xticks(x + width); ax1.set_xticklabels(LINK_SHORT, fontsize=8)
    ax1.set_ylabel('FPR'); ax1.set_title('(a) False Positive Rate')
    ax1.legend(fontsize=7)
    ax2.set_xticks(x + width); ax2.set_xticklabels(LINK_SHORT, fontsize=8)
    ax2.set_ylabel('FNR'); ax2.set_title('(b) False Negative Rate')
    ax2.legend(fontsize=7)
    plt.suptitle(f'Error Rate Analysis \u2014 {ds_label}', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(charts_dir / 'fig16_fpr_fnr_comparison.png', dpi=300, bbox_inches='tight'); plt.close()
    log(f'    Saved: fig16_fpr_fnr_comparison.png')

    # --- fig17: Master Agent — Health Score Heatmap Over Episodes ---
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH_DOUBLE, 3.0))
    all_health = master_stats.get('all_health', [])
    if all_health and len(all_health) > 1:
        health_arr = np.array(all_health).T  # (5, n_episodes)
        sns.heatmap(health_arr, ax=ax, cmap='RdYlGn', vmin=0, vmax=1,
                    yticklabels=LINK_SHORT, xticklabels=False,
                    cbar_kws={'label': 'Health Score'}, linewidths=0.3)
        ax.set_xlabel('Episode'); ax.set_ylabel('Link')
        ax.set_title(f'Master Agent: Health Score Heatmap \u2014 {ds_label}')
    else:
        ax.text(0.5, 0.5, 'No episode data (loaded from cache)',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title(f'Master Agent: Health Score Heatmap \u2014 {ds_label}')
    plt.tight_layout()
    plt.savefig(charts_dir / 'fig17_health_heatmap.png', dpi=300, bbox_inches='tight'); plt.close()
    log(f'    Saved: fig17_health_heatmap.png')

    # --- fig18: Comprehensive Summary Table as Figure ---
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH_DOUBLE, 4.0))
    ax.set_axis_off()
    col_labels = ['Phase', 'MSR', 'FPR', 'FNR', 'JFI', 'Reward']
    table_data = []
    for pname, pres in phase_results.items():
        g = pres.get('global', {})
        table_data.append([
            pname,
            f'{g.get("mean_msr", 0):.4f}',
            f'{g.get("mean_fpr", 0):.4f}',
            f'{g.get("mean_fnr", 0):.4f}',
            f'{g.get("jain_fairness", 0):.4f}',
            f'{g.get("mean_reward", 0):.4f}',
        ])
    # Add Master row
    jfi_b = master_stats.get('jfi_before_mean', 0)
    jfi_a = master_stats.get('jfi_after_mean', 0)
    best_freq = master_stats.get('best_link_frequency', {})
    best_link = max(best_freq, key=best_freq.get) if best_freq else 'N/A'
    table_data.append([f'Master Agent', f'Best: {best_link}', '-', '-',
                       f'{jfi_a:.4f}', f'JFI+{jfi_a-jfi_b:+.4f}'])

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)
    # Color header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#2C3E50')
        table[0, j].set_text_props(color='white', fontweight='bold')
    # Highlight best MSR row
    best_msr_row = 0
    best_msr_val = 0
    for ri, row in enumerate(table_data[:-1]):  # Exclude Master row
        val = float(row[1])
        if val > best_msr_val:
            best_msr_val = val
            best_msr_row = ri
    for j in range(len(col_labels)):
        table[best_msr_row + 1, j].set_facecolor('#D5F5E3')
    # Master row color
    for j in range(len(col_labels)):
        table[len(table_data), j].set_facecolor('#FCF3CF')

    ax.set_title(f'Strategy A: Complete Results Summary \u2014 {ds_label}',
                 fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(charts_dir / 'fig18_summary_table.png', dpi=300, bbox_inches='tight'); plt.close()
    log(f'    Saved: fig18_summary_table.png')

    log(f'  All 18 charts saved to {charts_dir}')


# ============================================================================
# Report + LaTeX Tables
# ============================================================================
def generate_report(results_a, results_b, results_c, master_stats, ds_dir, ds_label):
    """Generate comprehensive text report + LaTeX tables."""
    lines = []
    lines.append('=' * 80)
    lines.append(f'  FL-RLMAMS Phase 4 Strategy A: Multi-Agent DQN + FL + Master Agent')
    lines.append(f'  Dataset: {ds_label}')
    lines.append('=' * 80)

    phase_data = {
        'Phase A (Independent)': results_a,
        'Phase B (FedAvg)': results_b,
        'Phase C (FedProx)': results_c,
    }

    # 1. Phase Comparison
    lines.append('\n1. PHASE COMPARISON (Global Metrics)')
    lines.append('-' * 80)
    lines.append(f'{"Phase":30s} {"MSR":>8s} {"FPR":>8s} {"FNR":>8s} {"JFI":>8s} {"Reward":>8s}')
    for pname, pres in phase_data.items():
        g = pres.get('global', {})
        lines.append(f'{pname:30s} {g.get("mean_msr",0):8.4f} {g.get("mean_fpr",0):8.4f} '
                     f'{g.get("mean_fnr",0):8.4f} {g.get("jain_fairness",0):8.4f} '
                     f'{g.get("mean_reward",0):8.4f}')

    # 2. Per-Link Results (best phase)
    lines.append('\n2. PER-LINK RESULTS (FedProx)')
    lines.append('-' * 80)
    lines.append(f'{"Link":15s} {"MSR":>8s} {"FPR":>8s} {"FNR":>8s} {"Reward":>8s}')
    for ln, ls in zip(LINK_NAMES, LINK_SHORT):
        m = results_c.get(ln, {})
        lines.append(f'{ls:15s} {m.get("msr",0):8.4f} {m.get("fpr",0):8.4f} '
                     f'{m.get("fnr",0):8.4f} {m.get("avg_reward",0):8.4f}')

    # 3. Master Agent Results
    lines.append('\n3. MASTER AGENT: LOAD BALANCING RESULTS')
    lines.append('-' * 80)
    lines.append(f'  Health Scores (Mean over {len(master_stats.get("all_health",[[]]))} episodes):')
    for i in range(5):
        h_mean = master_stats['health_mean'][i]
        h_std = master_stats['health_std'][i]
        lines.append(f'    {LINK_SHORT[i]:15s}: H = {h_mean:.4f} +/- {h_std:.4f}')

    lines.append(f'\n  JFI Before Master: {master_stats["jfi_before_mean"]:.4f}')
    lines.append(f'  JFI After Master:  {master_stats["jfi_after_mean"]:.4f}')
    lines.append(f'  JFI Improvement:   {master_stats["jfi_improvement"]:+.4f}')

    lines.append(f'\n  Best Link Frequency:')
    for link, count in master_stats.get('best_link_frequency', {}).items():
        lines.append(f'    {link:15s}: {count} times')

    lines.append(f'\n  Routing Probabilities (Mean):')
    for link, prob in master_stats.get('routing_prob_mean', {}).items():
        lines.append(f'    {link:15s}: {prob:.1%}')

    lines.append(f'\n  Avg Reroutes per Episode: {master_stats.get("avg_reroutes_per_episode", 0):.1f}')

    # 4. Last Episode Reroute Commands
    last = master_stats.get('last_result', {})
    commands = last.get('reroute_commands', [])
    if commands:
        lines.append(f'\n  Last Episode Reroute Commands:')
        for cmd in commands:
            lines.append(f'    {cmd["from_link"]} --> {cmd["to_link"]}: {cmd["amount_mbps"]:.1f} Mbps')

    # 5. Master Ranking
    lines.append(f'\n  Final Link Ranking:')
    ranking = last.get('ranking', LINK_SHORT)
    for ri, link in enumerate(ranking):
        marker = ' *** PRIMARY ***' if ri == 0 else (' (BACKUP)' if ri == 1 else '')
        lines.append(f'    #{ri+1} {link}{marker}')

    lines.append('\n' + '=' * 80)

    # LaTeX Table
    lines.append('\n% === LaTeX Table (for IEEE Paper) ===')
    lines.append('\\begin{table}[htbp]')
    lines.append('\\caption{Strategy A: Phase Comparison}')
    lines.append('\\label{tab:strategy_a}')
    lines.append('\\centering')
    lines.append('\\begin{tabular}{lcccc}')
    lines.append('\\toprule')
    lines.append('Phase & MSR & FPR & FNR & JFI \\\\')
    lines.append('\\midrule')
    for pname, pres in phase_data.items():
        g = pres.get('global', {})
        short_name = pname.split('(')[1].rstrip(')')
        lines.append(f'{short_name} & {g.get("mean_msr",0):.4f} & {g.get("mean_fpr",0):.4f} & '
                     f'{g.get("mean_fnr",0):.4f} & {g.get("jain_fairness",0):.4f} \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    # Master Agent Table
    lines.append('')
    lines.append('\\begin{table}[htbp]')
    lines.append('\\caption{Master Agent: Link Health Scores and Routing}')
    lines.append('\\label{tab:master_agent}')
    lines.append('\\centering')
    lines.append('\\begin{tabular}{lccc}')
    lines.append('\\toprule')
    lines.append('Link & Health Score & Routing Prob. & Rank \\\\')
    lines.append('\\midrule')
    for ri, link in enumerate(ranking):
        li = LINK_SHORT.index(link)
        h = master_stats['health_mean'][li]
        rp = master_stats['routing_prob_mean'].get(link, 0)
        lines.append(f'{link} & {h:.4f} & {rp:.1%} & {ri+1} \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    lines.append('\n' + '=' * 80)
    lines.append('  END OF REPORT')
    lines.append('=' * 80)

    report = '\n'.join(lines)
    report_file = ds_dir / 'report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    log(f'  Report saved: {report_file}')
    return report


# ============================================================================
# Main: Train one dataset
# ============================================================================
def train_dataset(dataset_key):
    global _log_file
    ds_label = DS_LABELS[dataset_key]
    ds_dir = RESULTS_BASE / dataset_key
    _log_file = ds_dir / 'training.log'
    progress_file = ds_dir / 'progress_state.json'
    charts_dir = ds_dir / 'charts'

    os.makedirs(ds_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)
    seed_everything(SEED)

    log('=' * 70)
    log(f'  FL-RLMAMS Phase 4 Strategy A: Multi-Agent DQN + FL + Master Agent')
    log(f'  Dataset: {ds_label}')
    log(f'  Device: {DEVICE}')
    if torch.cuda.is_available():
        log(f'  GPU: {torch.cuda.get_device_name(0)}')
    log(f'  State dim: {STATE_DIM}, Actions: {N_ACTIONS}')
    log(f'  Phases: A({INDEPENDENT_EPISODES} eps) + B(FedAvg {FL_ROUNDS}r) + C(FedProx {FL_ROUNDS}r) + D(Master)')
    log('=' * 70)

    progress = load_progress(progress_file)

    # Load feature order
    with open(DATASETS[dataset_key]['json_file'], 'r') as f:
        feat_info = json.load(f)
    feature_order = feat_info['selected_features']
    log(f'  Features: {len(feature_order)}')

    data_dir = DATASETS[dataset_key]['data_dir']
    link_files = sorted(data_dir.glob('Link_*.csv'))
    ckpt_dir = PROJECT / 'experiments' / 'results' / dataset_key / 'checkpoints'

    # ================================================================
    # Step 1: Load data + pre-compute Mamba outputs
    # ================================================================
    log(f'\n{"="*60}')
    log(f'  Step 1: Loading Data & Mamba Pre-computation')
    log(f'{"="*60}')

    envs = []
    agents = []

    for li, link_file in enumerate(link_files):
        link_name = link_file.stem
        log(f'\n  [{li+1}/5] {link_name}')

        X_raw, y_raw = load_link_csv(link_file, feature_order,
                                     dataset_key=ds_key, link_name=link_name)
        log(f'    Creating windows (W={WINDOW_SIZE}, S={STRIDE})...')
        X_win, y_win = create_windows(X_raw, y_raw, WINDOW_SIZE, STRIDE)
        del X_raw, y_raw
        gc.collect()
        log(f'      Windows: {fmt(len(y_win))}')

        n_feat = X_win.shape[2]
        if USE_RLS:
            # RLS mode: already scaled by scripts/data/record_level_split.py
            # on TRAIN partition only. Skip the in-pipeline scaler to avoid
            # re-fitting a redundant (though not leaky) second scaler.
            log(f'      RLS mode: skipping in-pipeline StandardScaler.')
        else:
            scaler = StandardScaler()
            X_flat = X_win.reshape(-1, n_feat)
            X_flat = scaler.fit_transform(X_flat)
            X_win = X_flat.reshape(X_win.shape).astype(np.float32)
            del X_flat
            gc.collect()

        mamba_ckpt = ckpt_dir / f'{link_name}_Mamba_Only.pt'
        if not mamba_ckpt.exists():
            log(f'      WARNING: No Mamba checkpoint at {mamba_ckpt}')
            continue

        cache_dir = ds_dir / 'mamba_cache'
        os.makedirs(cache_dir, exist_ok=True)
        cache_softmax = cache_dir / f'{link_name}_softmax.npy'
        cache_embed = cache_dir / f'{link_name}_embed.npy'

        if cache_softmax.exists() and cache_embed.exists():
            log(f'      Loading cached Mamba outputs...')
            softmax = np.load(cache_softmax)
            embeddings = np.load(cache_embed)
            extractor = MambaEmbeddingExtractor(mamba_ckpt, n_features=n_feat)
            extractor.fit_pca(embeddings[:min(50000, len(embeddings))], n_components=16)
        else:
            log(f'      Computing Mamba outputs...')
            extractor = MambaEmbeddingExtractor(mamba_ckpt, n_features=n_feat)
            softmax, embeddings = extractor.extract_all(X_win)
            np.save(cache_softmax, softmax)
            np.save(cache_embed, embeddings)
            log(f'      Cached to {cache_dir.name}/')
            extractor.fit_pca(embeddings[:min(50000, len(embeddings))], n_components=16)

        embed_pca = extractor.transform_pca(embeddings)
        del embeddings
        gc.collect()

        env = SDNEnvironment(X_win, y_win, softmax, embed_pca,
                             link_idx=li, feature_order=feature_order)
        envs.append(env)

        agent = DQNAgent(state_dim=STATE_DIM, n_actions=N_ACTIONS, device=DEVICE)
        agents.append(agent)
        log(f'      Env: {env.n_windows} windows, DQN: {agent.n_params} params')

    log(f'\n  All 5 links loaded. Total windows: {sum(e.n_windows for e in envs):,}')

    # ================================================================
    # Step 2: Phase A
    # ================================================================
    phase_a_key = f'{dataset_key}::independent'
    if phase_a_key not in progress['completed']:
        log(f'\n{"="*60}')
        log(f'  Step 2: Phase A -- Independent DQN')
        log(f'{"="*60}')
        seed_everything(SEED)
        for agent in agents:
            agent.reset()
        phase_a_dir = ds_dir / 'independent'
        results_a = train_independent(agents, envs, INDEPENDENT_EPISODES, phase_a_dir)
        progress['completed'].append(phase_a_key)
        progress['results'][phase_a_key] = results_a.get('global', {})
        save_progress(progress, progress_file)
    else:
        log(f'\n  Phase A: COMPLETED (loading)')
        with open(ds_dir / 'independent' / 'results.json', 'r') as f:
            results_a = json.load(f)

    # ================================================================
    # Step 3: Phase B
    # ================================================================
    phase_b_key = f'{dataset_key}::fedavg'
    if phase_b_key not in progress['completed']:
        log(f'\n{"="*60}')
        log(f'  Step 3: Phase B -- FL-DQN (FedAvg)')
        log(f'{"="*60}')
        seed_everything(SEED)
        for agent in agents:
            agent.reset()
        phase_b_dir = ds_dir / 'fedavg'
        results_b = train_fl_dqn(agents, envs, FL_ROUNDS, LOCAL_EPISODES_PER_ROUND,
                                  algo='fedavg', phase_dir=phase_b_dir)
        progress['completed'].append(phase_b_key)
        progress['results'][phase_b_key] = results_b.get('global', {})
        save_progress(progress, progress_file)
    else:
        log(f'\n  Phase B: COMPLETED (loading)')
        with open(ds_dir / 'fedavg' / 'results.json', 'r') as f:
            results_b = json.load(f)

    # ================================================================
    # Step 4: Phase C
    # ================================================================
    phase_c_key = f'{dataset_key}::fedprox'
    if phase_c_key not in progress['completed']:
        log(f'\n{"="*60}')
        log(f'  Step 4: Phase C -- FL-DQN (FedProx)')
        log(f'{"="*60}')
        seed_everything(SEED)
        for agent in agents:
            agent.reset()
        phase_c_dir = ds_dir / 'fedprox'
        results_c = train_fl_dqn(agents, envs, FL_ROUNDS, LOCAL_EPISODES_PER_ROUND,
                                  algo='fedprox', mu=FEDPROX_MU, phase_dir=phase_c_dir)
        progress['completed'].append(phase_c_key)
        progress['results'][phase_c_key] = results_c.get('global', {})
        save_progress(progress, progress_file)
    else:
        log(f'\n  Phase C: COMPLETED (loading)')
        with open(ds_dir / 'fedprox' / 'results.json', 'r') as f:
            results_c = json.load(f)

    # ================================================================
    # Step 5: Phase D — Master Agent (post-training)
    # ================================================================
    master_key = f'{dataset_key}::master'
    if master_key not in progress['completed']:
        log(f'\n{"="*60}')
        log(f'  Step 5: Phase D -- Master Agent Load Balancing')
        log(f'{"="*60}')

        # Load best agents (FedProx checkpoints)
        fedprox_dir = ds_dir / 'fedprox'
        for i in range(5):
            ckpt_path = fedprox_dir / LINK_NAMES[i] / 'best_dqn.pt'
            if ckpt_path.exists():
                agents[i].q_net.load_state_dict(
                    torch.load(ckpt_path, weights_only=True, map_location=DEVICE))
                agents[i].target_net.load_state_dict(agents[i].q_net.state_dict())
                log(f'    Loaded FedProx checkpoint: {LINK_NAMES[i]}')

        master = MasterAgent(n_links=5, capacity_mbps=1000.0)
        log(f'    Running Master evaluation over 50 episodes...')
        master_stats = master.evaluate_over_episodes(agents, envs, n_episodes=50)

        # Print summary
        log(f'\n    === MASTER AGENT RESULTS ===')
        log(f'    Health Scores: {[f"{h:.3f}" for h in master_stats["health_mean"]]}')
        log(f'    JFI Before: {master_stats["jfi_before_mean"]:.4f}')
        log(f'    JFI After:  {master_stats["jfi_after_mean"]:.4f} ({master_stats["jfi_improvement"]:+.4f})')
        log(f'    Best Link:  {max(master_stats["best_link_frequency"], key=master_stats["best_link_frequency"].get)}')

        # Save master results
        master_dir = ds_dir / 'master_agent'
        os.makedirs(master_dir, exist_ok=True)
        # Convert numpy arrays for JSON serialization
        master_save = {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                       for k, v in master_stats.items()
                       if k not in ['all_rankings']}
        # Convert nested numpy arrays in all_health
        if 'all_health' in master_save:
            master_save['all_health'] = [[float(x) for x in row] for row in master_save['all_health']]
        if 'all_jfi_before' in master_save:
            master_save['all_jfi_before'] = [float(x) for x in master_save['all_jfi_before']]
        if 'all_jfi_after' in master_save:
            master_save['all_jfi_after'] = [float(x) for x in master_save['all_jfi_after']]
        master_save['last_result'] = master_stats['last_result']
        with open(master_dir / 'results.json', 'w') as f:
            json.dump(master_save, f, indent=2)

        progress['completed'].append(master_key)
        progress['results'][master_key] = {
            'jfi_before': master_save.get('jfi_before_mean', 0),
            'jfi_after': master_save.get('jfi_after_mean', 0),
        }
        save_progress(progress, progress_file)
    else:
        log(f'\n  Phase D: COMPLETED (loading)')
        with open(ds_dir / 'master_agent' / 'results.json', 'r') as f:
            master_stats = json.load(f)
        # Provide empty lists for chart generation
        master_stats.setdefault('all_rankings', [])
        master_stats.setdefault('all_health', [])
        master_stats.setdefault('all_jfi_before', [])
        master_stats.setdefault('all_jfi_after', [])

    # ================================================================
    # Step 6: Generate Charts
    # ================================================================
    log(f'\n{"="*60}')
    log(f'  Step 6: Generating IEEE Charts')
    log(f'{"="*60}')
    generate_all_charts(results_a, results_b, results_c, master_stats, charts_dir, ds_label)

    # ================================================================
    # Step 7: Generate Report
    # ================================================================
    log(f'\n{"="*60}')
    log(f'  Step 7: Generating Report')
    log(f'{"="*60}')
    report = generate_report(results_a, results_b, results_c, master_stats, ds_dir, ds_label)
    print('\n' + report)

    # Save combined results JSON
    combined = {
        'dataset': dataset_key,
        'label': ds_label,
        'phase_a': results_a.get('global', {}),
        'phase_b': results_b.get('global', {}),
        'phase_c': results_c.get('global', {}),
        'master_agent': {k: v for k, v in master_stats.items()
                         if k not in ['all_rankings', 'all_health', 'all_jfi_before', 'all_jfi_after']},
    }
    with open(ds_dir / 'results.json', 'w') as f:
        json.dump(combined, f, indent=2)

    # Clean up progress
    if progress_file.exists():
        os.remove(progress_file)
    log(f'\n  {ds_label} Strategy A Complete!')
    log('=' * 70)


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='FL-RLMAMS Phase 4 Strategy A')
    parser.add_argument('--dataset', type=str, default='all',  choices=['ciciot2023', 'diad2024', 'all'])
    parser.add_argument('--use-rls', action='store_true',
                        help='Load features from the record-level split '
                             '(leakage-free preprocessing; reviewer R2-6).')
    parser.add_argument('--rls-root', type=str, default=None,
                        help='Override the RLS root folder. Defaults to '
                             'data/unified_14features_5links_rls.')
    parser.add_argument('--rls-split', choices=['train', 'val', 'test'],
                        default='train',
                        help='Which split feeds the RL agent (default: train).')
    args = parser.parse_args()

    global USE_RLS, RLS_ROOT, RLS_SPLIT
    if args.use_rls:
        USE_RLS = True
        RLS_SPLIT = args.rls_split
        RLS_ROOT = (Path(args.rls_root) if args.rls_root else
                    PROJECT / 'data' / 'unified_14features_5links_rls')
        if not RLS_ROOT.exists():
            raise FileNotFoundError(
                f'--use-rls requested but RLS root does not exist: {RLS_ROOT}. '
                f'Run scripts/data/record_level_split.py first.')
        log(f'  RLS mode ON  root={RLS_ROOT}  split={RLS_SPLIT}')

    datasets_to_run = ['ciciot2023', 'diad2024'] if args.dataset == 'all' else [args.dataset]

    for ds_key in datasets_to_run:
        train_dataset(ds_key)


if __name__ == '__main__':
    main()
