# -*- coding: utf-8 -*-
"""
============================================================================
FL-RLMAMS Phase 4: Multi-Agent DQN + Federated Learning
============================================================================
Each SDN Link = Agent with:
  - Frozen Mamba-Only (detection) → severity + 128D embedding
  - Trainable DQN → action: ALLOW / RATE_LIMIT / BLOCK
  - FL server aggregates DQN weights across 5 agents

Usage:
  python train_rl_multiagent.py --dataset ciciot2023
  python train_rl_multiagent.py --dataset diad2024
  python train_rl_multiagent.py --dataset all
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

# ============================================================================
# Configuration
# ============================================================================
PROJECT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT / 'data'
RESULTS_BASE = PROJECT / 'experiments' / 'results' / 'phase4_rl'

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
INDEPENDENT_EPISODES = 10#100
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
def load_link_csv(csv_path, feature_order):
    """Load link CSV, subsample if needed, return X (float32), y (int64)."""
    log(f'    Loading {csv_path.name}...')
    needed_cols = list(feature_order) + ['severity']
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    usecols = [c for c in needed_cols if c in header]
    # Use chunked reading for large files to avoid OOM
    chunks = []
    total_rows = 0
    for chunk in pd.read_csv(csv_path, usecols=usecols, low_memory=True,
                             chunksize=500_000):
        chunks.append(chunk)
        total_rows += len(chunk)
        if total_rows >= MAX_ROWS * 2:
            break  # No need to read more than we'll sample
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
        """Extract softmax probs + 128D embeddings for all windows.
        Args: X_windows (np.ndarray): shape (N, W, F)
        Returns: softmax (N, 3), embeddings (N, 128)
        """
        dataset = torch.FloatTensor(X_windows).to(self.device)
        all_softmax = []
        all_embeds = []
        batch_size = 512
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            # Forward through mamba to get 128D embedding
            x = self.model.input_proj(batch)   # (B, W, 128)
            x = self.model.mamba(x)             # (B, W, 128)
            embed = x[:, -1, :]                 # (B, 128) — last timestep
            # Forward through FC for softmax
            logits = self.model.fc(embed)        # (B, 3)
            probs = F.softmax(logits, dim=-1)
            all_softmax.append(probs.cpu().numpy())
            all_embeds.append(embed.cpu().numpy())
        softmax = np.concatenate(all_softmax, axis=0)
        embeddings = np.concatenate(all_embeds, axis=0)
        return softmax, embeddings

    def fit_pca(self, embeddings, n_components=16):
        """Fit PCA on train embeddings."""
        self.pca = PCA(n_components=n_components, random_state=SEED)
        self.pca.fit(embeddings)
        log(f'      PCA fitted: 128D → {n_components}D (explained: {self.pca.explained_variance_ratio_.sum():.3f})')

    def transform_pca(self, embeddings):
        """Apply PCA to reduce 128D → 16D."""
        return self.pca.transform(embeddings).astype(np.float32)


# ============================================================================
# SDN Environment (offline Gym-like)
# ============================================================================
class SDNEnvironment:
    """Offline RL environment for one SDN link."""

    def __init__(self, X_windows, y_windows, mamba_softmax, mamba_embed_pca,
                 link_idx=0, feature_order=None, max_steps=MAX_STEPS_PER_EPISODE):
        self.X_windows = X_windows        # (N, W, F)
        self.y_windows = y_windows        # (N,) true severity
        self.mamba_softmax = mamba_softmax # (N, 3) softmax probs
        self.mamba_embed = mamba_embed_pca # (N, 16) PCA embeddings
        self.link_idx = link_idx
        self.n_windows = len(y_windows)
        self.max_steps = min(max_steps, self.n_windows)
        self.feature_order = feature_order

        # Pre-compute link metrics from raw features
        self._precompute_link_metrics()

        # Episode state
        self.step_idx = 0
        self.episode_indices = None
        self.recent_severities = deque(maxlen=10)

        # Neighbor references (set externally during FL)
        self.neighbor_envs = []  # list of other SDNEnvironment instances

    def _precompute_link_metrics(self):
        """Derive link metrics from raw 14 features per window."""
        # Mean across timesteps in each window
        X_mean = self.X_windows.mean(axis=1)  # (N, 14)

        # Feature indices (from feature_order)
        fo = self.feature_order if self.feature_order else []
        def _idx(name):
            try:
                return fo.index(name)
            except ValueError:
                return -1

        idx_flow_packets = _idx('flow_packets_per_s')
        idx_flow_duration = _idx('flow_duration')
        idx_pkt_size = _idx('avg_pkt_size')
        idx_flow_iat = _idx('flow_iat_mean')

        # traffic_load: normalized flow_packets_per_s
        if idx_flow_packets >= 0:
            traffic = X_mean[:, idx_flow_packets]
            self.traffic_load = (traffic - traffic.min()) / (traffic.max() - traffic.min() + 1e-8)
        else:
            self.traffic_load = np.zeros(self.n_windows, dtype=np.float32)

        # latency_est: normalized flow_duration
        if idx_flow_duration >= 0:
            lat = X_mean[:, idx_flow_duration]
            self.latency_est = (lat - lat.min()) / (lat.max() - lat.min() + 1e-8)
        else:
            self.latency_est = np.zeros(self.n_windows, dtype=np.float32)

        # packet_rate: normalized avg_pkt_size
        if idx_pkt_size >= 0:
            pkt = X_mean[:, idx_pkt_size]
            self.packet_rate = (pkt - pkt.min()) / (pkt.max() - pkt.min() + 1e-8)
        else:
            self.packet_rate = np.zeros(self.n_windows, dtype=np.float32)

        # entropy: will be computed dynamically from recent severity predictions
        self.entropy_vals = np.zeros(self.n_windows, dtype=np.float32)

    def reset(self):
        """Start new episode with shuffled window subset."""
        indices = np.random.permutation(self.n_windows)[:self.max_steps]
        self.episode_indices = indices
        self.step_idx = 0
        self.recent_severities.clear()
        return self._get_state()

    def step(self, action):
        """Execute action, return (next_state, reward, done, info)."""
        idx = self.episode_indices[self.step_idx]
        true_severity = self.y_windows[idx]
        pred_severity = self.mamba_softmax[idx].argmax()

        # Track recent predictions for entropy
        self.recent_severities.append(pred_severity)

        # Compute reward
        neighbor_actions = self._get_neighbor_actions()
        reward = self._compute_reward(action, true_severity, neighbor_actions)

        # Advance
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
        """Construct 28D state vector for current window."""
        if self.step_idx >= len(self.episode_indices):
            return np.zeros(STATE_DIM, dtype=np.float32)

        idx = self.episode_indices[self.step_idx]
        state = np.zeros(STATE_DIM, dtype=np.float32)

        # 1. Link Metrics (4D)
        state[0] = self.traffic_load[idx]
        state[1] = self.latency_est[idx]
        state[2] = self._compute_entropy()
        state[3] = self.packet_rate[idx]

        # 2. LFA State (4D)
        softmax = self.mamba_softmax[idx]
        pred = softmax.argmax()
        state[4] = float(pred != 2)            # attack_detected (not Low)
        state[5] = float(pred) / 2.0           # severity_pred normalized
        state[6] = float(softmax.max())        # attack_confidence
        state[7] = self._severity_ratio()      # ratio of High in recent

        # 3. Mamba Embedding (16D)
        state[8:24] = self.mamba_embed[idx]

        # 4. Neighbor Flags (4D)
        n_flags = self._get_neighbor_flags()
        state[24:28] = n_flags

        return state

    def _compute_entropy(self):
        """Shannon entropy of recent severity predictions."""
        if len(self.recent_severities) < 2:
            return 0.5
        counts = Counter(self.recent_severities)
        total = len(self.recent_severities)
        entropy = 0.0
        for c in counts.values():
            p = c / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy / math.log2(3)  # normalized to [0, 1]

    def _severity_ratio(self):
        """Fraction of High (0) predictions in recent history."""
        if len(self.recent_severities) == 0:
            return 0.0
        return sum(1 for s in self.recent_severities if s == 0) / len(self.recent_severities)

    def _get_neighbor_flags(self):
        """Get severity info from neighboring agents."""
        flags = np.zeros(4, dtype=np.float32)
        if not self.neighbor_envs:
            return flags
        # Left neighbor
        if self.link_idx > 0 and len(self.neighbor_envs) > self.link_idx - 1:
            env_l = self.neighbor_envs[self.link_idx - 1]
            if len(env_l.recent_severities) > 0:
                flags[0] = env_l.recent_severities[-1] / 2.0  # normalized
        # Right neighbor
        if self.link_idx < 4 and len(self.neighbor_envs) > self.link_idx + 1:
            env_r = self.neighbor_envs[self.link_idx + 1]
            if len(env_r.recent_severities) > 0:
                flags[1] = env_r.recent_severities[-1] / 2.0
        # Alert count: how many neighbors have High severity
        alert = 0
        for env in self.neighbor_envs:
            if env is not self and len(env.recent_severities) > 0:
                if env.recent_severities[-1] == 0:
                    alert += 1
        flags[2] = alert / max(len(self.neighbor_envs) - 1, 1)
        # Coordination signal: mean severity across all neighbors
        sev_sum = 0
        sev_count = 0
        for env in self.neighbor_envs:
            if env is not self and len(env.recent_severities) > 0:
                sev_sum += env.recent_severities[-1]
                sev_count += 1
        flags[3] = (sev_sum / max(sev_count, 1)) / 2.0
        return flags

    def _get_neighbor_actions(self):
        """Get most recent actions from neighbors (for reward calc)."""
        # During independent training, no neighbor info
        return None

    @staticmethod
    def _optimal_action(severity):
        """Map severity → optimal action."""
        return {0: 2, 1: 1, 2: 0}[severity]  # High→BLOCK, Med→LIMIT, Low→ALLOW

    @staticmethod
    def _compute_reward(action, true_severity, neighbor_actions=None):
        """Multi-objective reward function."""
        optimal = SDNEnvironment._optimal_action(true_severity)

        # R_correct
        R_correct = 1.0 if action == optimal else -0.5

        # R_security: penalty for allowing High attacks
        R_security = -1.0 if (action == 0 and true_severity == 0) else 0.0

        # R_service: penalty for blocking benign
        R_service = -1.0 if (action == 2 and true_severity == 2) else 0.0

        # R_neighbor: consensus bonus
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
    """MLP Q-network: state → Q-values for each action."""
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

    def train_step(self, global_params=None, mu=0.0):
        """One training step from replay buffer. Optional FedProx proximal term."""
        if len(self.buffer) < DQN_BATCH:
            return 0.0
        states, actions, rewards, next_states, dones = self.buffer.sample(DQN_BATCH)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values (from target network)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            target_q = rewards_t + GAMMA * next_q * (1 - dones_t)

        # TD loss
        loss = F.mse_loss(q_values, target_q)

        # FedProx proximal term
        if mu > 0 and global_params is not None:
            prox_term = 0.0
            for p, gp in zip(self.q_net.parameters(), global_params):
                prox_term += ((p - gp.to(self.device)) ** 2).sum()
            loss = loss + (mu / 2.0) * prox_term

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Soft target update
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
        """Reset agent for new experiment phase."""
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
    """Weighted average of DQN parameters."""
    n_params = len(client_params_list[0])
    aggregated = []
    for i in range(n_params):
        weighted_sum = torch.zeros_like(client_params_list[0][i])
        for c, w in enumerate(client_weights):
            weighted_sum += w * client_params_list[c][i]
        aggregated.append(weighted_sum)
    return aggregated


def blend_params(local_params, global_params, alpha=BLEND_ALPHA):
    """Blend local and global: alpha*local + (1-alpha)*global."""
    blended = []
    for lp, gp in zip(local_params, global_params):
        blended.append(alpha * lp + (1.0 - alpha) * gp)
    return blended


# ============================================================================
# Evaluation
# ============================================================================
def evaluate_agent(agent, env, n_episodes=5):
    """Evaluate agent over multiple episodes. Returns metrics dict."""
    total_correct = 0
    total_steps = 0
    total_reward = 0.0
    action_counts = Counter()
    severity_action_matrix = np.zeros((3, 3), dtype=np.int64)  # severity x action
    fp_count = 0  # blocking benign
    fn_count = 0  # allowing attacks
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

            if info['true_severity'] == 2:  # Low/benign
                benign_count += 1
                if action == 2:  # BLOCK
                    fp_count += 1
            elif info['true_severity'] == 0:  # High
                attack_count += 1
                if action == 0:  # ALLOW
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
    """Jain's Fairness Index."""
    n = len(msr_list)
    sum_x = sum(msr_list)
    sum_x2 = sum(x**2 for x in msr_list)
    if sum_x2 == 0:
        return 1.0
    return (sum_x ** 2) / (n * sum_x2)


# ============================================================================
# Training: Phase A (Independent DQN, no FL)
# ============================================================================
def train_independent(agents, envs, n_episodes, phase_dir):
    """Train each agent independently (no FL, no neighbors)."""
    log(f'\n  Phase A: Independent DQN Training ({n_episodes} episodes)')
    history = {i: {'rewards': [], 'msr': [], 'loss': []} for i in range(5)}

    for ep in range(n_episodes):
        ep_rewards = []
        ep_losses = []
        for i in range(5):
            state = envs[i].reset()
            done = False
            ep_reward = 0.0
            ep_loss = 0.0
            steps = 0
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
            history[i]['rewards'].append(ep_reward / max(steps, 1))
            history[i]['loss'].append(ep_loss / max(steps, 1))
            ep_rewards.append(ep_reward / max(steps, 1))

        # Log every 10 episodes
        if (ep + 1) % 10 == 0 or ep == 0:
            mean_reward = np.mean(ep_rewards)
            # Quick eval
            msr_list = []
            for i in range(5):
                m = evaluate_agent(agents[i], envs[i], n_episodes=1)
                msr_list.append(m['msr'])
                history[i]['msr'].append(m['msr'])
            mean_msr = np.mean(msr_list)
            jfi = compute_jain_fairness(msr_list)
            log(f'    Ep {ep+1:3d}/{n_episodes}: Reward={mean_reward:.4f} '
                f'MSR={mean_msr:.4f} JFI={jfi:.4f} ε={agents[0].epsilon:.3f}')

    # Final evaluation
    log(f'\n  [Final Eval] Independent DQN:')
    results = {}
    for i in range(5):
        m = evaluate_agent(agents[i], envs[i], n_episodes=10)
        log(f'    {LINK_SHORT[i]}: MSR={m["msr"]:.4f} FPR={m["fpr"]:.4f} '
            f'FNR={m["fnr"]:.4f} Reward={m["avg_reward"]:.4f}')
        results[LINK_NAMES[i]] = m
        # Save checkpoint
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
    results['history'] = {str(k): v for k, v in history.items()}
    log(f'    Global: MSR={results["global"]["mean_msr"]:.4f} '
        f'JFI={results["global"]["jain_fairness"]:.4f}')

    # Save results
    with open(phase_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ============================================================================
# Training: Phase B/C (FL-DQN with FedAvg or FedProx)
# ============================================================================
def train_fl_dqn(agents, envs, fl_rounds, local_episodes, algo='fedavg',
                 mu=0.0, phase_dir=None):
    """Train agents with Federated Learning."""
    algo_label = 'FedAvg' if algo == 'fedavg' else f'FedProx(μ={mu})'
    log(f'\n  Phase {"B" if algo == "fedavg" else "C"}: FL-DQN with {algo_label} '
        f'({fl_rounds} rounds × {local_episodes} local eps)')

    # Enable neighbor communication
    for env in envs:
        env.neighbor_envs = envs

    # Initialize global params
    global_params = agents[0].get_params()
    client_weights = [env.n_windows for env in envs]
    total_w = sum(client_weights)
    client_weights = [w / total_w for w in client_weights]

    history = {
        'rounds': [],
        'per_agent': {i: {'msr': [], 'reward': [], 'fpr': [], 'fnr': []} for i in range(5)},
        'global_msr': [],
        'global_fpr': [],       # A5: round-drift detection quality (false alarm)
        'global_fnr': [],       # A5: round-drift detection quality (missed attack)
        'mean_loss': [],        # A5: convergence signal, to decouple from detection
        'jain_fairness': [],
    }

    # A5 round-drift: directory for per-round global checkpoints
    round_ckpt_dir = None
    if phase_dir:
        round_ckpt_dir = phase_dir / 'round_ckpts'
        os.makedirs(round_ckpt_dir, exist_ok=True)

    for rd in range(fl_rounds):
        # Local training
        client_params_list = []
        round_rewards = []
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

        # Server aggregation
        global_params = fedavg_aggregate(client_params_list, client_weights)

        # A5 round-drift: persist the aggregated GLOBAL model for this round.
        # Enables offline re-evaluation of the pure global detector per round
        # (model convergence vs detection convergence) without retraining.
        if round_ckpt_dir is not None:
            torch.save([p.clone() for p in global_params],
                       round_ckpt_dir / f'global_round_{rd + 1:02d}.pt')

        # Distribute with blending
        for i in range(5):
            local_p = agents[i].get_params()
            blended = blend_params(local_p, global_params, BLEND_ALPHA)
            agents[i].set_params(blended)
            # Also update target net
            agents[i].target_net.load_state_dict(agents[i].q_net.state_dict())

        # Evaluate
        msr_list = []
        fpr_list = []
        fnr_list = []
        for i in range(5):
            m = evaluate_agent(agents[i], envs[i], n_episodes=3)
            msr_list.append(m['msr'])
            fpr_list.append(m.get('fpr', float('nan')))
            fnr_list.append(m.get('fnr', float('nan')))
            history['per_agent'][i]['msr'].append(m['msr'])
            history['per_agent'][i]['reward'].append(m['avg_reward'])
            history['per_agent'][i]['fpr'].append(m.get('fpr', float('nan')))
            history['per_agent'][i]['fnr'].append(m.get('fnr', float('nan')))

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
                f'JFI={jfi:.4f} ε={agents[0].epsilon:.3f}')

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

    return results


# ============================================================================
# Main: Train one dataset
# ============================================================================
def train_dataset(dataset_key):
    global _log_file
    ds_label = DS_LABELS[dataset_key]
    ds_dir = RESULTS_BASE / dataset_key
    _log_file = ds_dir / 'training.log'
    progress_file = ds_dir / 'progress_state.json'

    os.makedirs(ds_dir, exist_ok=True)
    seed_everything(SEED)

    log('=' * 70)
    log(f'  FL-RLMAMS Phase 4: Multi-Agent DQN + FL — {ds_label}')
    log(f'  Device: {DEVICE}')
    if torch.cuda.is_available():
        log(f'  GPU: {torch.cuda.get_device_name(0)}')
    log(f'  State dim: {STATE_DIM}, Actions: {N_ACTIONS}, DQN params: ~14K')
    log('=' * 70)

    # Load progress
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
    # Step 1: Load data + pre-compute Mamba outputs for all 5 links
    # ================================================================
    log(f'\n{"="*60}')
    log(f'  Step 1: Loading Data & Mamba Pre-computation')
    log(f'{"="*60}')

    envs = []
    agents = []

    for li, link_file in enumerate(link_files):
        link_name = link_file.stem
        log(f'\n  [{li+1}/5] {link_name}')

        # Load CSV
        X_raw, y_raw = load_link_csv(link_file, feature_order)

        # Create windows
        log(f'    Creating windows (W={WINDOW_SIZE}, S={STRIDE})...')
        X_win, y_win = create_windows(X_raw, y_raw, WINDOW_SIZE, STRIDE)
        del X_raw, y_raw
        gc.collect()
        log(f'      Windows: {fmt(len(y_win))}')

        # Scale
        n_feat = X_win.shape[2]
        scaler = StandardScaler()
        X_flat = X_win.reshape(-1, n_feat)
        X_flat = scaler.fit_transform(X_flat)
        X_win = X_flat.reshape(X_win.shape).astype(np.float32)
        del X_flat
        gc.collect()

        # Load Mamba checkpoint
        mamba_ckpt = ckpt_dir / f'{link_name}_Mamba_Only.pt'
        if not mamba_ckpt.exists():
            log(f'      WARNING: No Mamba checkpoint at {mamba_ckpt}')
            continue

        # Cache file for mamba outputs
        cache_dir = ds_dir / 'mamba_cache'
        os.makedirs(cache_dir, exist_ok=True)
        cache_softmax = cache_dir / f'{link_name}_softmax.npy'
        cache_embed = cache_dir / f'{link_name}_embed.npy'

        if cache_softmax.exists() and cache_embed.exists():
            log(f'      Loading cached Mamba outputs...')
            softmax = np.load(cache_softmax)
            embeddings = np.load(cache_embed)
            extractor = MambaEmbeddingExtractor(mamba_ckpt, n_features=n_feat)
            # Need PCA — fit on cached embeddings
            extractor.fit_pca(embeddings[:min(50000, len(embeddings))], n_components=16)
        else:
            log(f'      Computing Mamba outputs (this may take a minute)...')
            extractor = MambaEmbeddingExtractor(mamba_ckpt, n_features=n_feat)
            softmax, embeddings = extractor.extract_all(X_win)
            np.save(cache_softmax, softmax)
            np.save(cache_embed, embeddings)
            log(f'      Cached to {cache_dir.name}/')
            extractor.fit_pca(embeddings[:min(50000, len(embeddings))], n_components=16)

        # PCA compress embeddings
        embed_pca = extractor.transform_pca(embeddings)
        del embeddings
        gc.collect()

        # Create environment
        env = SDNEnvironment(X_win, y_win, softmax, embed_pca,
                             link_idx=li, feature_order=feature_order)
        envs.append(env)

        # Create DQN agent
        agent = DQNAgent(state_dim=STATE_DIM, n_actions=N_ACTIONS, device=DEVICE)
        agents.append(agent)
        log(f'      Env: {env.n_windows} windows, DQN: {agent.n_params} params')

    log(f'\n  All 5 links loaded. Total windows: {sum(e.n_windows for e in envs):,}')

    # ================================================================
    # Step 2: Phase A — Independent DQN
    # ================================================================
    phase_a_key = f'{dataset_key}::independent'
    if phase_a_key not in progress['completed']:
        log(f'\n{"="*60}')
        log(f'  Step 2: Phase A — Independent DQN')
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
        log(f'\n  Phase A: COMPLETED (skipping)')

    # ================================================================
    # Step 3: Phase B — FL-DQN with FedAvg
    # ================================================================
    phase_b_key = f'{dataset_key}::fedavg'
    if phase_b_key not in progress['completed']:
        log(f'\n{"="*60}')
        log(f'  Step 3: Phase B — FL-DQN (FedAvg)')
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
        log(f'\n  Phase B: COMPLETED (skipping)')

    # ================================================================
    # Step 4: Phase C — FL-DQN with FedProx
    # ================================================================
    phase_c_key = f'{dataset_key}::fedprox'
    if phase_c_key not in progress['completed']:
        log(f'\n{"="*60}')
        log(f'  Step 4: Phase C — FL-DQN (FedProx)')
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
        log(f'\n  Phase C: COMPLETED (skipping)')

    # ================================================================
    # Summary
    # ================================================================
    log(f'\n{"="*60}')
    log(f'  Summary — {ds_label}')
    log(f'{"="*60}')
    for key in [phase_a_key, phase_b_key, phase_c_key]:
        r = progress['results'].get(key, {})
        phase = key.split('::')[1]
        log(f'  {phase:12s}: MSR={r.get("mean_msr", 0):.4f} '
            f'JFI={r.get("jain_fairness", 0):.4f} '
            f'FPR={r.get("mean_fpr", 0):.4f} '
            f'FNR={r.get("mean_fnr", 0):.4f}')

    # Clean up progress
    if progress_file.exists():
        os.remove(progress_file)
    log(f'\n  {ds_label} Phase 4 Complete!')
    log('=' * 70)

    return progress['results']


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='FL-RLMAMS Phase 4: Multi-Agent DQN + FL')
    parser.add_argument('--dataset', type=str, default='all',choices=['ciciot2023', 'diad2024', 'all'])
    args = parser.parse_args()
    datasets_to_run = ['ciciot2023', 'diad2024'] if args.dataset == 'all' else [args.dataset]
    for ds_key in datasets_to_run:
        train_dataset(ds_key)


if __name__ == '__main__':
    main()
