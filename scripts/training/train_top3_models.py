# -*- coding: utf-8 -*-
"""
============================================================================
FL-RLMAMS Stage 3: Cross-Dataset Generalization Study
============================================================================
Train on CICIoT2023, evaluate on DIAD2024 (unseen) — Top 3 models from
Fuzzy-TOPSIS selection: LSTM-Only, Mamba-Only, CNN-Transformer.

Per-link local training on 5 SDN links with academic charts.
============================================================================
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys, io, json, time, warnings, math
from pathlib import Path
from collections import Counter

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

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# Configuration
# ============================================================================
PROJECT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT / 'data'
LOG_FILE = PROJECT / "experiments" / "results" / "stage3_progress.log"
CHARTS_DIR = PROJECT / "experiments" / "results" / "charts" / "stage3"
CKPT_DIR = PROJECT / "experiments" / "results" / "checkpoints" / "stage3"
RESULTS_JSON = PROJECT / "experiments" / "results" / "stage3_results.json"
RECOMMEND_FILE = PROJECT / "experiments" / "results" / "stage3_recommendation.txt"
PROGRESS_FILE = PROJECT / "experiments" / "results" / "stage3_progress_state.json"

SEED = 42
WINDOW_SIZE = 20
STRIDE = 20
BATCH_SIZE = 256
MAX_EPOCHS = 50
LR = 1e-3
NUM_CLASSES = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEV_MAP = {'High': 0, 'Medium': 1, 'Low': 2}
SEV_NAMES = ['High', 'Medium', 'Low']
SEV_COLORS = {'High': '#E74C3C', 'Medium': '#F39C12', 'Low': '#27AE60'}

# Train on CICIoT2023, Unseen test on DIAD2024
TRAIN_DATASET = 'ciciot2023'
UNSEEN_DATASET = 'diad2024'

DATASETS = {
    'ciciot2023': {
        'data_dir': DATA_DIR / 'unified_14features_5links' / 'ciciot2023',
        'json_file': DATA_DIR / 'unified_selected_14.json',
    },
    'diad2024': {
        'data_dir': DATA_DIR / 'unified_14features_5links' / 'diad2024',
        'json_file': DATA_DIR / 'unified_selected_14.json',  # SAME unified JSON
    },
}

# Top 3 models from Fuzzy-TOPSIS (Combined CC ranking)
VARIANT_NAMES = ['LSTM-Only', 'Mamba-Only', 'CNN-Transformer']

VARIANT_COLORS = {
    'LSTM-Only': '#3498DB',       # Blue (RNN)
    'Mamba-Only': '#27AE60',      # Green (SSM)
    'CNN-Transformer': '#9B59B6', # Purple (Transformer)
}

VARIANT_FAMILIES = {
    'LSTM-Only': 'RNN',
    'Mamba-Only': 'SSM',
    'CNN-Transformer': 'Transformer',
}

FAMILY_COLORS = {'RNN': '#3498DB', 'SSM': '#27AE60', 'Transformer': '#9B59B6'}

# Matplotlib academic style
plt.rcParams.update({
    'font.size': 12, 'font.family': 'serif',
    'axes.labelsize': 14, 'axes.titlesize': 15, 'axes.titleweight': 'bold',
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ============================================================================
# Logging
# ============================================================================
def log(msg=''):
    sys.stdout.write(str(msg) + '\n')
    try:
        sys.stdout.flush()
    except Exception:
        pass
    os.makedirs(LOG_FILE.parent, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(str(msg) + '\n')

def fmt(n):
    return f'{n:,}'


# ============================================================================
# Progress Tracking (Resume Support)
# ============================================================================
def load_progress():
    """Load progress state from disk. Returns dict with completed pairs and partial results."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {
        'completed': [],          # list of "link_name::variant_name" strings
        'seen_link_results': {},  # {link_name: {variant: metrics}}
        'unseen_link_results': {},
        'all_histories': {},      # {variant: history} (first link only)
        'params_dict': {},
    }


def save_progress(state):
    """Save progress state to disk atomically."""
    os.makedirs(PROGRESS_FILE.parent, exist_ok=True)
    tmp = str(PROGRESS_FILE) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    # Atomic rename (Windows: need to remove first)
    if PROGRESS_FILE.exists():
        os.remove(PROGRESS_FILE)
    os.rename(tmp, PROGRESS_FILE)


def is_completed(state, link_name, variant_name):
    """Check if a (link, model) pair is already completed."""
    key = f'{link_name}::{variant_name}'
    return key in state['completed']



# ============================================================================
# Reproducibility
# ============================================================================
def seed_everything(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# Data Pipeline
# ============================================================================
class WindowedFlowDataset(Dataset):
    def __init__(self, X_windows, y_windows):
        self.X = torch.FloatTensor(X_windows)
        self.y = torch.LongTensor(y_windows)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_link_data(csv_path, feature_order):
    log(f'    Loading {csv_path.name}...')
    df = pd.read_csv(csv_path, low_memory=False)
    log(f'      Total rows: {fmt(len(df))}')
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    available = [f for f in feature_order if f in df.columns]
    if len(available) < len(feature_order):
        missing = set(feature_order) - set(available)
        log(f'      WARNING: Missing features: {missing}')
    X = df[available].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y_str = df['severity'].values
    y = np.array([SEV_MAP.get(s, 2) for s in y_str], dtype=np.int64)
    log(f'      Class dist: H={np.sum(y==0):,} M={np.sum(y==1):,} L={np.sum(y==2):,}')
    return X, y


def create_windows(X, y, window_size=WINDOW_SIZE, stride=STRIDE):
    n_samples = X.shape[0]
    if n_samples < window_size:
        return np.empty((0, window_size, X.shape[1])), np.empty(0, dtype=np.int64)
    n_windows = (n_samples - window_size) // stride + 1
    X_win = np.empty((n_windows, window_size, X.shape[1]), dtype=np.float32)
    y_win = np.empty(n_windows, dtype=np.int64)
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        X_win[i] = X[start:end]
        y_win[i] = y[end - 1]
    return X_win, y_win


def prepare_train_datasets(csv_path, feature_order):
    """Prepare CICIoT2023 training data with 70/15/15 split. Returns scaler."""
    X_raw, y_raw = load_link_data(csv_path, feature_order)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    log(f'    Creating windows (W={WINDOW_SIZE}, S={STRIDE})...')
    X_win, y_win = create_windows(X_scaled, y_raw)
    log(f'      Total windows: {fmt(len(y_win))}')
    if len(y_win) == 0:
        return None, None, None, None, None, None

    win_counts = Counter(y_win.tolist())
    can_stratify = all(c >= 4 for c in win_counts.values())
    strat_arg = y_win if can_stratify else None
    log(f'      {"Stratified" if can_stratify else "Random"} split')
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_win, y_win, test_size=0.30, random_state=SEED, stratify=strat_arg)
    strat_temp = y_temp if can_stratify else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=strat_temp)
    log(f'      Train: {fmt(len(y_train))}  Val: {fmt(len(y_val))}  Test: {fmt(len(y_test))}')

    # Re-fit scaler on train only (flatten windows -> scale -> reshape)
    n_feat = X_train.shape[2]
    scaler_final = StandardScaler()
    X_train = scaler_final.fit_transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape)
    X_val = scaler_final.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape)
    X_test = scaler_final.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)

    # Class weights
    counts = Counter(y_train.tolist())
    total = len(y_train)
    class_weights = torch.FloatTensor([
        total / (NUM_CLASSES * max(counts.get(c, 1), 1)) for c in range(NUM_CLASSES)
    ]).to(DEVICE)
    log(f'      Class weights: {[f"{w:.3f}" for w in class_weights.cpu().numpy()]}')

    train_dl = DataLoader(WindowedFlowDataset(X_train, y_train), batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0, pin_memory=True)
    val_dl = DataLoader(WindowedFlowDataset(X_val, y_val), batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0, pin_memory=True)
    test_dl = DataLoader(WindowedFlowDataset(X_test, y_test), batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=0, pin_memory=True)
    return train_dl, val_dl, test_dl, class_weights, scaler_final, n_feat


def prepare_unseen_datasets(csv_path, feature_order, scaler, n_feat):
    """Prepare DIAD2024 unseen data using scaler from CICIoT2023 training."""
    X_raw, y_raw = load_link_data(csv_path, feature_order)
    # Apply the SAME scaler fitted on CICIoT2023 train data
    X_scaled = scaler.transform(X_raw)
    log(f'    Creating unseen windows (W={WINDOW_SIZE}, S={STRIDE})...')
    X_win, y_win = create_windows(X_scaled, y_raw)
    log(f'      Unseen windows: {fmt(len(y_win))}')
    if len(y_win) == 0:
        return None
    # Scale windows using same scaler
    X_win_flat = X_win.reshape(-1, n_feat)
    X_win_scaled = scaler.transform(X_win_flat).reshape(X_win.shape)
    unseen_dl = DataLoader(WindowedFlowDataset(X_win_scaled, y_win), batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=0, pin_memory=True)
    return unseen_dl


# ============================================================================
# Shared Modules
# ============================================================================
class CNNBlock(nn.Module):
    """Shared CNN feature extractor: (B, W, F) -> (B, W//4, 64)"""
    def __init__(self, n_features=14):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, F, W)
        x = self.conv(x)         # (B, 64, W//4)
        x = x.permute(0, 2, 1)   # (B, W//4, 64)
        return x


# ============================================================================
# Mamba Block (Pure PyTorch — Selective State Space Model)
# ============================================================================
class SelectiveSSM(nn.Module):
    """Selective Scan (S6) — core of Mamba. Pure PyTorch, no CUDA kernels."""
    def __init__(self, d_inner, d_state=16, dt_rank=8):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank ** -0.5 * 1.0
        nn.init.uniform_(self.dt_proj.bias, -dt_init_std, dt_init_std)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x):
        batch, seq_len, d_inner = x.shape
        x_dbl = self.x_proj(x)
        delta, B, C = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        A = -torch.exp(self.A_log)
        y = self._selective_scan(x, delta, A, B, C)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        return y

    def _selective_scan(self, x, delta, A, B, C):
        batch, seq_len, d_inner = x.shape
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB_x = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)
        h = torch.zeros(batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(seq_len):
            h = deltaA[:, t] * h + deltaB_x[:, t]
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1)
            ys.append(y_t)
        return torch.stack(ys, dim=1)


class MambaBlock(nn.Module):
    """Full Mamba block: projection + causal conv + SSM + gating."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand_factor
        dt_rank = max(1, math.ceil(d_model / 16))
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )
        self.ssm = SelectiveSSM(self.d_inner, d_state=d_state, dt_rank=dt_rank)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)
        x_ssm = x_ssm.permute(0, 2, 1)
        x_ssm = self.conv1d(x_ssm)[:, :, :x.shape[1]]
        x_ssm = x_ssm.permute(0, 2, 1)
        x_ssm = F.silu(x_ssm)
        x_ssm = self.ssm(x_ssm)
        z = F.silu(z)
        x_out = x_ssm * z
        x_out = self.out_proj(x_out)
        return x_out + residual


# ============================================================================
# Model Definitions — Top 3 from Fuzzy-TOPSIS
# ============================================================================
class LSTMOnlyModel(nn.Module):
    """LSTM-Only: 2-layer LSTM (128->64) + FC"""
    def __init__(self, n_features=14, n_classes=NUM_CLASSES, **kw):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, 128, batch_first=True, dropout=0.2, num_layers=1)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True, num_layers=1)
        self.fc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3),
                                nn.Linear(64, n_classes))
    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return self.fc(x[:, -1, :])


class MambaOnlyModel(nn.Module):
    """Mamba-Only: Linear -> MambaBlock -> FC"""
    def __init__(self, n_features=14, n_classes=NUM_CLASSES, **kw):
        super().__init__()
        self.input_proj = nn.Linear(n_features, 128)
        self.mamba = MambaBlock(d_model=128, d_state=16, d_conv=4, expand_factor=2)
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
                                nn.Linear(64, n_classes))
    def forward(self, x):
        x = self.input_proj(x)
        x = self.mamba(x)
        x = x[:, -1, :]
        return self.fc(x)


class CNNTransformerModel(nn.Module):
    """CNN-Transformer: CNN -> Transformer Encoder (2 layers, 4 heads) -> mean pool -> FC"""
    def __init__(self, n_features=14, n_classes=NUM_CLASSES, **kw):
        super().__init__()
        self.cnn = CNNBlock(n_features)
        d_model = 64
        max_len = WINDOW_SIZE // 4 + 1
        self.pos_enc = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=256,
            dropout=0.1, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(0.3),
                                nn.Linear(64, n_classes))
    def forward(self, x):
        x = self.cnn(x)
        seq_len = x.size(1)
        x = x + self.pos_enc[:, :seq_len, :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)


VARIANT_CLASSES = {
    'LSTM-Only': LSTMOnlyModel,
    'Mamba-Only': MambaOnlyModel,
    'CNN-Transformer': CNNTransformerModel,
}

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Training & Evaluation
# ============================================================================
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += X_batch.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += X_batch.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def train_model(model, train_dl, val_dl, class_weights, variant_name,
                ckpt_path, max_epochs=MAX_EPOCHS):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    start_epoch = 0

    # --- Resume from mid-training checkpoint if exists ---
    resume_path = Path(str(ckpt_path).replace('.pt', '_resume.pt'))
    if resume_path.exists():
        log(f'      RESUME: Loading mid-training checkpoint (continuing training)...')
        resume_state = torch.load(resume_path, weights_only=False)
        model.load_state_dict(resume_state['model_state_dict'])
        optimizer.load_state_dict(resume_state['optimizer_state_dict'])
        scheduler.load_state_dict(resume_state['scheduler_state_dict'])
        start_epoch = resume_state['epoch'] + 1
        history = resume_state['history']
        best_val_loss = resume_state['best_val_loss']
        log(f'      Resuming from epoch {start_epoch + 1}/{max_epochs}')

    for epoch in range(start_epoch, max_epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_dl, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_dl, criterion)
        scheduler.step(val_loss)
        elapsed = time.time() - t0
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            log(f'      Epoch {epoch+1:3d}/{max_epochs}: '
                f'TrLoss={train_loss:.4f} TrAcc={train_acc:.4f} | '
                f'VaLoss={val_loss:.4f} VaAcc={val_acc:.4f} | {elapsed:.1f}s')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(ckpt_path.parent, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)

        # Save mid-training checkpoint every 5 epochs for resume
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history,
                'best_val_loss': best_val_loss,
            }, resume_path)

    # Training complete — load best model and clean up resume file
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    if resume_path.exists():
        os.remove(resume_path)
    return history


def full_evaluation(model, test_dl, n_features=14):
    """Evaluate model on a test DataLoader. Returns metrics dict."""
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, preds, labels = evaluate(model, test_dl, criterion)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    prec_macro = precision_score(labels, preds, average='macro', zero_division=0)
    rec_macro = recall_score(labels, preds, average='macro', zero_division=0)
    f1_per = f1_score(labels, preds, average=None, zero_division=0)
    prec_per = precision_score(labels, preds, average=None, zero_division=0)
    rec_per = recall_score(labels, preds, average=None, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    # Inference time
    model.eval()
    dummy = torch.randn(BATCH_SIZE, WINDOW_SIZE, n_features).to(DEVICE)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inf_time_ms = (time.time() - t0) / 10 * 1000
    return {
        'test_loss': float(test_loss), 'accuracy': float(test_acc),
        'f1_macro': float(f1_macro), 'precision_macro': float(prec_macro),
        'recall_macro': float(rec_macro),
        'f1_per_class': [float(x) for x in f1_per],
        'precision_per_class': [float(x) for x in prec_per],
        'recall_per_class': [float(x) for x in rec_per],
        'confusion_matrix': cm.tolist(),
        'inference_time_ms': float(inf_time_ms),
    }


# ============================================================================
# Chart Generation (fig1–fig12)
# ============================================================================

def plot_training_curves(all_histories, fig_id='fig1'):
    """fig1: Training curves for 3 models (2x3 subplots)."""
    n_models = len(all_histories)
    fig, axes = plt.subplots(2, n_models, figsize=(6 * n_models, 10))
    for i, (vname, hist) in enumerate(all_histories.items()):
        c = VARIANT_COLORS[vname]
        # Loss
        axes[0, i].plot(hist['train_loss'], color=c, linestyle='-', label='Train', linewidth=2)
        axes[0, i].plot(hist['val_loss'], color=c, linestyle='--', label='Val', linewidth=2)
        axes[0, i].set_xlabel('Epoch'); axes[0, i].set_ylabel('Loss')
        axes[0, i].set_title(f'{vname} — Loss')
        axes[0, i].legend()
        # Accuracy
        axes[1, i].plot(hist['train_acc'], color=c, linestyle='-', label='Train', linewidth=2)
        axes[1, i].plot(hist['val_acc'], color=c, linestyle='--', label='Val', linewidth=2)
        axes[1, i].set_xlabel('Epoch'); axes[1, i].set_ylabel('Accuracy')
        axes[1, i].set_title(f'{vname} — Accuracy')
        axes[1, i].legend()
    plt.suptitle('CICIoT2023 Training — Top 3 Models', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = CHARTS_DIR / f'{fig_id}_training_curves.png'
    plt.savefig(out); plt.close()
    log(f'    Saved: {out.name}')


def plot_performance_bars(aggregated, title, fig_id):
    """Bar chart: 4 metrics x 3 models."""
    variants = list(aggregated.keys())
    metrics_names = ['Accuracy', 'F1-Macro', 'Precision', 'Recall']
    values = np.array([[aggregated[v]['accuracy'], aggregated[v]['f1_macro'],
                        aggregated[v]['precision_macro'], aggregated[v]['recall_macro']]
                       for v in variants])
    x = np.arange(len(variants))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 7))
    metric_colors = ['#3498DB', '#E74C3C', '#F39C12', '#27AE60']
    for j, mn in enumerate(metrics_names):
        bars = ax.bar(x + j * width, values[:, j], width, label=mn,
                      color=metric_colors[j], alpha=0.85)
        for bar, val in zip(bars, values[:, j]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, rotation=45)
    ax.set_xlabel('Model'); ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(variants, fontsize=12)
    ax.legend(); ax.set_ylim(0, 1.15)
    plt.tight_layout()
    out = CHARTS_DIR / f'{fig_id}.png'
    plt.savefig(out); plt.close()
    log(f'    Saved: {out.name}')


def plot_generalization_gap(seen_agg, unseen_agg, fig_id='fig4'):
    """fig4: Grouped bar chart: seen vs unseen F1/accuracy per model."""
    variants = VARIANT_NAMES
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(variants))
    width = 0.30

    for ax_idx, metric, metric_label in [(0, 'accuracy', 'Accuracy'), (1, 'f1_macro', 'F1-Macro')]:
        seen_vals = [seen_agg[v][metric] for v in variants]
        unseen_vals = [unseen_agg[v][metric] for v in variants]
        axes[ax_idx].bar(x - width/2, seen_vals, width, label='Seen (CICIoT2023)',
                         color='#3498DB', alpha=0.85)
        axes[ax_idx].bar(x + width/2, unseen_vals, width, label='Unseen (DIAD2024)',
                         color='#E74C3C', alpha=0.85)
        for i in range(len(variants)):
            gap = unseen_vals[i] - seen_vals[i]
            sign = '+' if gap >= 0 else ''
            axes[ax_idx].text(x[i] + width/2, unseen_vals[i] + 0.005,
                              f'{sign}{gap:.3f}', ha='center', va='bottom',
                              fontsize=9, color='red' if gap < 0 else 'green', fontweight='bold')
        axes[ax_idx].set_xlabel('Model'); axes[ax_idx].set_ylabel(metric_label)
        axes[ax_idx].set_title(f'{metric_label}: Seen vs Unseen')
        axes[ax_idx].set_xticks(x)
        axes[ax_idx].set_xticklabels(variants, fontsize=10)
        axes[ax_idx].legend()
        axes[ax_idx].set_ylim(0, 1.15)

    plt.suptitle('Generalization Gap Analysis', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = CHARTS_DIR / f'{fig_id}_generalization_gap.png'
    plt.savefig(out); plt.close()
    log(f'    Saved: {out.name}')


def plot_confusion_matrices(all_metrics, title, fig_id):
    """1x3 confusion matrices."""
    n_variants = len(all_metrics)
    fig, axes = plt.subplots(1, n_variants, figsize=(6 * n_variants, 5))
    if n_variants == 1:
        axes = [axes]
    for idx, (vname, m) in enumerate(all_metrics.items()):
        cm = np.array(m['confusion_matrix'])
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=SEV_NAMES, yticklabels=SEV_NAMES,
                    ax=axes[idx], cbar=False)
        fam = VARIANT_FAMILIES.get(vname, '')
        axes[idx].set_title(f'{vname} [{fam}]\nAcc={m["accuracy"]:.3f} F1={m["f1_macro"]:.3f}',
                            fontsize=11)
        axes[idx].set_xlabel('Predicted'); axes[idx].set_ylabel('Actual')
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = CHARTS_DIR / f'{fig_id}.png'
    plt.savefig(out); plt.close()
    log(f'    Saved: {out.name}')


def plot_per_link_f1(link_results, title, fig_id):
    """Per-link F1 comparison — grouped bars."""
    link_names = list(link_results.keys())
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(link_names))
    width = 0.25
    for vi, vname in enumerate(VARIANT_NAMES):
        f1s = [link_results[ln].get(vname, {}).get('f1_macro', 0) for ln in link_names]
        bars = ax.bar(x + vi * width, f1s, width, label=vname,
                      color=VARIANT_COLORS[vname], alpha=0.85)
        for bar, val in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)
    ax.set_xlabel('Network Link'); ax.set_ylabel('F1-Macro')
    ax.set_title(title)
    ax.set_xticks(x + width)
    ax.set_xticklabels([ln.replace('_', '\n') for ln in link_names], fontsize=9)
    ax.legend(); ax.set_ylim(0, 1.15)
    plt.tight_layout()
    out = CHARTS_DIR / f'{fig_id}.png'
    plt.savefig(out); plt.close()
    log(f'    Saved: {out.name}')


def plot_per_link_gap_heatmap(seen_links, unseen_links, fig_id='fig9'):
    """fig9: Per-link generalization gap heatmap (model x link)."""
    link_names = sorted(set(seen_links.keys()) & set(unseen_links.keys()))
    gap_data = []
    for vname in VARIANT_NAMES:
        row = []
        for ln in link_names:
            seen_f1 = seen_links.get(ln, {}).get(vname, {}).get('f1_macro', 0)
            unseen_f1 = unseen_links.get(ln, {}).get(vname, {}).get('f1_macro', 0)
            row.append(unseen_f1 - seen_f1)
        gap_data.append(row)

    gap_matrix = np.array(gap_data)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(gap_matrix, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                xticklabels=[ln.replace('_', '\n') for ln in link_names],
                yticklabels=VARIANT_NAMES, ax=ax, linewidths=0.5)
    ax.set_title('Per-Link Generalization Gap (Unseen F1 - Seen F1)', fontsize=14, fontweight='bold')
    ax.set_xlabel('SDN Link'); ax.set_ylabel('Model')
    plt.tight_layout()
    out = CHARTS_DIR / f'{fig_id}_per_link_gap_heatmap.png'
    plt.savefig(out); plt.close()
    log(f'    Saved: {out.name}')


def plot_radar_seen_unseen(seen_agg, unseen_agg, params_dict, fig_id='fig10'):
    """fig10: Radar chart — seen vs unseen for each model."""
    categories = ['F1-Macro\n(Seen)', 'F1-Macro\n(Unseen)', 'Accuracy\n(Seen)',
                  'Accuracy\n(Unseen)', 'Speed\n(1/InfTime)', 'Efficiency\n(1/Params)']
    N = len(categories)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    max_params = max(params_dict.values()) if params_dict else 1
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for vi, vname in enumerate(VARIANT_NAMES):
        sm = seen_agg[vname]
        um = unseen_agg[vname]
        n_params = params_dict.get(vname, max_params)
        values = [
            sm['f1_macro'], um['f1_macro'],
            sm['accuracy'], um['accuracy'],
            min(1.0, 1.0 / (sm['inference_time_ms'] + 0.1)),
            1.0 - (n_params / max_params),
        ]
        values += values[:1]
        color = VARIANT_COLORS[vname]
        ax.plot(angles, values, 'o-', linewidth=2.5, color=color, label=vname, markersize=6)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Top 3 Models: Seen vs Unseen Performance', fontsize=14, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
    plt.tight_layout()
    out = CHARTS_DIR / f'{fig_id}_radar_seen_unseen.png'
    plt.savefig(out); plt.close()
    log(f'    Saved: {out.name}')


def plot_complexity_scatter(seen_agg, unseen_agg, params_dict, fig_id='fig11'):
    """fig11: Bubble chart — params vs F1 (seen & unseen), size=inference time."""
    fig, ax = plt.subplots(figsize=(10, 7))
    for vname in VARIANT_NAMES:
        n_params = params_dict.get(vname, 0)
        c = VARIANT_COLORS[vname]
        # Seen
        sm = seen_agg[vname]
        ax.scatter(n_params / 1000, sm['f1_macro'],
                   s=sm['inference_time_ms'] * 30 + 80,
                   c=c, marker='o', alpha=0.7, edgecolors='black', linewidths=1)
        ax.annotate(f'{vname}\n(Seen)', (n_params / 1000, sm['f1_macro']),
                    fontsize=8, ha='center', va='bottom')
        # Unseen
        um = unseen_agg[vname]
        ax.scatter(n_params / 1000, um['f1_macro'],
                   s=um['inference_time_ms'] * 30 + 80,
                   c=c, marker='s', alpha=0.7, edgecolors='black', linewidths=1)
        ax.annotate(f'{vname}\n(Unseen)', (n_params / 1000, um['f1_macro']),
                    fontsize=8, ha='center', va='top')
    ax.set_xlabel('Parameters (K)'); ax.set_ylabel('F1-Macro')
    ax.set_title('Model Complexity vs F1 (Seen & Unseen)\n(bubble size = inference time)')
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                                   markersize=8, label='Seen (CICIoT2023)'),
                       plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                                   markersize=8, label='Unseen (DIAD2024)')]
    ax.legend(handles=legend_elements, fontsize=10)
    plt.tight_layout()
    out = CHARTS_DIR / f'{fig_id}_complexity_scatter.png'
    plt.savefig(out); plt.close()
    log(f'    Saved: {out.name}')


def plot_class_f1_comparison(seen_agg, unseen_agg, fig_id='fig12'):
    """fig12: Per-class F1 comparison (seen vs unseen) for each model."""
    fig, axes = plt.subplots(1, len(VARIANT_NAMES), figsize=(6 * len(VARIANT_NAMES), 6))
    if len(VARIANT_NAMES) == 1:
        axes = [axes]
    for i, vname in enumerate(VARIANT_NAMES):
        x = np.arange(NUM_CLASSES)
        width = 0.30
        seen_f1 = seen_agg[vname].get('f1_per_class', [0]*NUM_CLASSES)
        unseen_f1 = unseen_agg[vname].get('f1_per_class', [0]*NUM_CLASSES)
        axes[i].bar(x - width/2, seen_f1, width, label='Seen', color='#3498DB', alpha=0.85)
        axes[i].bar(x + width/2, unseen_f1, width, label='Unseen', color='#E74C3C', alpha=0.85)
        for j in range(NUM_CLASSES):
            gap = unseen_f1[j] - seen_f1[j]
            sign = '+' if gap >= 0 else ''
            axes[i].text(x[j] + width/2, unseen_f1[j] + 0.01,
                         f'{sign}{gap:.3f}', ha='center', fontsize=8,
                         color='green' if gap >= 0 else 'red', fontweight='bold')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(SEV_NAMES)
        axes[i].set_ylim(0, 1.15)
        axes[i].set_title(f'{vname}')
        axes[i].set_ylabel('F1-Score')
        axes[i].legend(fontsize=9)
    plt.suptitle('Per-Class F1: Seen vs Unseen', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = CHARTS_DIR / f'{fig_id}_class_f1_comparison.png'
    plt.savefig(out); plt.close()
    log(f'    Saved: {out.name}')


# ============================================================================
# Recommendation Report
# ============================================================================
def generate_recommendation(seen_results, unseen_results, params_dict, gen_gap):
    lines = []
    lines.append('=' * 80)
    lines.append('  FL-RLMAMS Stage 3: Cross-Dataset Generalization Report')
    lines.append('  Train: CICIoT2023 | Unseen Test: DIAD2024')
    lines.append('=' * 80)
    lines.append('')

    # 1. Seen Performance
    lines.append('1. SEEN PERFORMANCE (CICIoT2023 Test Set)')
    lines.append('-' * 80)
    lines.append(f'{"Model":<20s} {"Acc":>8s} {"F1":>8s} {"Prec":>8s} {"Rec":>8s} {"Inf(ms)":>8s} {"Params":>10s}')
    for vname in VARIANT_NAMES:
        s = seen_results['aggregated'][vname]
        p = params_dict.get(vname, 0)
        lines.append(f'{vname:<20s} {s["accuracy"]:8.4f} {s["f1_macro"]:8.4f} '
                     f'{s["precision_macro"]:8.4f} {s["recall_macro"]:8.4f} '
                     f'{s["inference_time_ms"]:8.2f} {fmt(p):>10s}')
    lines.append('')

    # 2. Unseen Performance
    lines.append('2. UNSEEN PERFORMANCE (DIAD2024 — Full Dataset)')
    lines.append('-' * 80)
    lines.append(f'{"Model":<20s} {"Acc":>8s} {"F1":>8s} {"Prec":>8s} {"Rec":>8s} {"Inf(ms)":>8s}')
    for vname in VARIANT_NAMES:
        u = unseen_results['aggregated'][vname]
        lines.append(f'{vname:<20s} {u["accuracy"]:8.4f} {u["f1_macro"]:8.4f} '
                     f'{u["precision_macro"]:8.4f} {u["recall_macro"]:8.4f} '
                     f'{u["inference_time_ms"]:8.2f}')
    lines.append('')

    # 3. Generalization Gap
    lines.append('3. GENERALIZATION GAP (Unseen - Seen)')
    lines.append('-' * 80)
    lines.append(f'{"Model":<20s} {"Acc Gap":>10s} {"F1 Gap":>10s} {"Interpretation":>20s}')
    for vname in VARIANT_NAMES:
        g = gen_gap[vname]
        interp = 'Improves' if g['f1_drop'] >= 0 else ('Moderate' if g['f1_drop'] > -0.1 else 'Significant')
        lines.append(f'{vname:<20s} {g["accuracy_drop"]:+10.4f} {g["f1_drop"]:+10.4f} {interp:>20s}')
    lines.append('')

    # 4. Ranking by unseen F1
    lines.append('4. RANKING BY UNSEEN F1-MACRO')
    lines.append('-' * 80)
    ranked = sorted(VARIANT_NAMES,
                    key=lambda v: unseen_results['aggregated'][v]['f1_macro'], reverse=True)
    for rank, vname in enumerate(ranked, 1):
        u = unseen_results['aggregated'][vname]
        star = ' *** RECOMMENDED ***' if rank == 1 else ''
        lines.append(f'  #{rank} {vname:<20s} F1={u["f1_macro"]:.4f} Acc={u["accuracy"]:.4f}{star}')
    lines.append('')

    # 5. Per-Link Robustness
    lines.append('5. PER-LINK ROBUSTNESS ON UNSEEN DATA')
    lines.append('-' * 80)
    for vname in VARIANT_NAMES:
        link_f1s = []
        for ln, lm in unseen_results['link_results'].items():
            if vname in lm:
                link_f1s.append(lm[vname]['f1_macro'])
        if link_f1s:
            cv = np.std(link_f1s) / (np.mean(link_f1s) + 1e-8)
            lines.append(f'  {vname:<20s} Mean F1={np.mean(link_f1s):.4f} '
                         f'Std={np.std(link_f1s):.4f} CV={cv:.4f}')
    lines.append('')

    lines.append('=' * 80)
    lines.append(f'  Report generated: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('=' * 80)

    report = '\n'.join(lines)
    os.makedirs(RECOMMEND_FILE.parent, exist_ok=True)
    with open(RECOMMEND_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    log(f'    Saved recommendation: {RECOMMEND_FILE.name}')
    log('')
    for line in lines:
        log(line)
    return report


# ============================================================================
# Aggregation Helpers
# ============================================================================
def aggregate_results(link_results, params_dict):
    """Aggregate per-link metrics into mean +/- std."""
    aggregated = {}
    for vname in VARIANT_NAMES:
        accs, f1s, precs, recs, inf_times, cms = [], [], [], [], [], []
        f1_per_class_all = []
        for link_name, link_metrics in link_results.items():
            if vname in link_metrics:
                m = link_metrics[vname]
                accs.append(m['accuracy']); f1s.append(m['f1_macro'])
                precs.append(m['precision_macro']); recs.append(m['recall_macro'])
                inf_times.append(m['inference_time_ms'])
                cms.append(np.array(m['confusion_matrix']))
                if 'f1_per_class' in m:
                    f1_per_class_all.append(m['f1_per_class'])
        if accs:
            agg = {
                'accuracy': float(np.mean(accs)), 'accuracy_std': float(np.std(accs)),
                'f1_macro': float(np.mean(f1s)), 'f1_macro_std': float(np.std(f1s)),
                'precision_macro': float(np.mean(precs)), 'recall_macro': float(np.mean(recs)),
                'inference_time_ms': float(np.mean(inf_times)),
                'confusion_matrix': np.sum(cms, axis=0).tolist(),
                'n_links': len(accs),
            }
            if f1_per_class_all:
                agg['f1_per_class'] = [float(np.mean([f[c] for f in f1_per_class_all]))
                                       for c in range(NUM_CLASSES)]
            aggregated[vname] = agg
            log(f'    {vname:20s}: Acc={np.mean(accs):.4f}+/-{np.std(accs):.4f}  '
                f'F1={np.mean(f1s):.4f}+/-{np.std(f1s):.4f}')
    return aggregated


# ============================================================================
# Main
# ============================================================================
def main():
    seed_everything(SEED)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    log('=' * 70)
    log('  FL-RLMAMS Stage 3: Cross-Dataset Generalization Study')
    log(f'  Train: CICIoT2023 | Unseen Test: DIAD2024')
    log(f'  Device: {DEVICE}')
    if torch.cuda.is_available():
        log(f'  GPU: {torch.cuda.get_device_name(0)}')
    log(f'  Window: W={WINDOW_SIZE}, Stride={STRIDE}')
    log(f'  Top 3 Models: {VARIANT_NAMES}')
    log('=' * 70)

    # Load feature order (UNIFIED — same for both datasets)
    with open(DATASETS[TRAIN_DATASET]['json_file'], 'r') as f:
        train_feat_info = json.load(f)
    train_feature_order = train_feat_info['selected_features']
    unseen_feature_order = train_feature_order  # Same unified features

    train_data_dir = DATASETS[TRAIN_DATASET]['data_dir']
    unseen_data_dir = DATASETS[UNSEEN_DATASET]['data_dir']
    train_link_files = sorted(train_data_dir.glob('Link_*.csv'))
    unseen_link_files = sorted(unseen_data_dir.glob('Link_*.csv'))
    log(f'  Train links: {len(train_link_files)} | Unseen links: {len(unseen_link_files)}')

    # ---- Load progress state (resume support) ----
    progress = load_progress()
    n_completed = len(progress['completed'])
    if n_completed > 0:
        log(f'\n  RESUME MODE: {n_completed} model(s) already completed, skipping them.')
        for key in progress['completed']:
            log(f'    - {key}')

    seen_link_results = progress['seen_link_results']
    unseen_link_results = progress['unseen_link_results']
    params_dict = progress['params_dict']
    all_histories = progress['all_histories']

    t_start = time.time()

    for lf_idx, train_lf in enumerate(train_link_files):
        link_name = train_lf.stem

        # Check if ALL models for this link are completed
        all_done = all(is_completed(progress, link_name, v) for v in VARIANT_NAMES)
        if all_done:
            log(f'\n  [{lf_idx+1}/{len(train_link_files)}] {link_name} — ALL COMPLETED (skipping)')
            continue

        log(f'\n{"="*60}')
        log(f'  [{lf_idx+1}/{len(train_link_files)}] {link_name}')
        log(f'{"="*60}')

        # ---- Prepare training data (CICIoT2023) ----
        log(f'\n  [Train] Preparing CICIoT2023 data for {link_name}...')
        result = prepare_train_datasets(train_lf, train_feature_order)
        if result[0] is None:
            log(f'    SKIP: No training windows for {link_name}')
            continue
        train_dl, val_dl, test_dl, class_weights, scaler, n_feat = result

        # ---- Find matching unseen link file (DIAD2024) ----
        link_num = link_name.split('_')[1]
        unseen_lf = None
        for ulf in unseen_link_files:
            if ulf.stem.split('_')[1] == link_num:
                unseen_lf = ulf
                break
        if unseen_lf is None:
            log(f'    WARNING: No matching unseen link for {link_name}, skipping unseen eval')

        # ---- Prepare unseen data (DIAD2024) ----
        unseen_dl = None
        unseen_link_name = None
        if unseen_lf is not None:
            unseen_link_name = unseen_lf.stem
            log(f'\n  [Unseen] Preparing DIAD2024 data for {unseen_link_name}...')
            unseen_dl = prepare_unseen_datasets(unseen_lf, unseen_feature_order, scaler, n_feat)

        if link_name not in seen_link_results:
            seen_link_results[link_name] = {}
        if unseen_link_name and unseen_link_name not in unseen_link_results:
            unseen_link_results[unseen_link_name] = {}

        # ---- Train & Evaluate each model ----
        for vi, vname in enumerate(VARIANT_NAMES):
            # Skip if already completed
            if is_completed(progress, link_name, vname):
                log(f'\n    [{vi+1}/{len(VARIANT_NAMES)}] {vname} — COMPLETED (skipping)')
                continue

            ModelClass = VARIANT_CLASSES[vname]
            log(f'\n    [{vi+1}/{len(VARIANT_NAMES)}] Training {vname} [{VARIANT_FAMILIES[vname]}]...')
            seed_everything(SEED)
            model = ModelClass(n_features=n_feat, n_classes=NUM_CLASSES).to(DEVICE)
            n_params = count_params(model)
            params_dict[vname] = n_params
            log(f'      Parameters: {fmt(n_params)}')

            ckpt_path = CKPT_DIR / f'{link_name}_{vname.replace("-", "_")}.pt'
            resume_path = Path(str(ckpt_path).replace('.pt', '_resume.pt'))

            # If fully trained checkpoint exists (no resume file), skip training
            if ckpt_path.exists() and not resume_path.exists():
                log(f'      CACHED: Loading fully trained checkpoint (eval only)')
                model.load_state_dict(torch.load(ckpt_path, weights_only=True))
                history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
            else:
                # Train (with epoch-level resume built into train_model)
                history = train_model(model, train_dl, val_dl, class_weights, vname, ckpt_path)

            # Save history for first link only (for chart)
            if lf_idx == 0 and history.get('train_loss'):
                all_histories[vname] = history

            # Evaluate on SEEN test (CICIoT2023)
            log(f'      [Seen Test] Evaluating on CICIoT2023 test split...')
            seen_metrics = full_evaluation(model, test_dl, n_feat)
            log(f'        Acc={seen_metrics["accuracy"]:.4f} F1={seen_metrics["f1_macro"]:.4f}')
            seen_link_results[link_name][vname] = seen_metrics

            # Evaluate on UNSEEN (DIAD2024)
            unseen_metrics = None
            if unseen_dl is not None and unseen_link_name is not None:
                log(f'      [Unseen Test] Evaluating on DIAD2024 (full)...')
                unseen_metrics = full_evaluation(model, unseen_dl, n_feat)
                log(f'        Acc={unseen_metrics["accuracy"]:.4f} F1={unseen_metrics["f1_macro"]:.4f}')
                unseen_link_results[unseen_link_name][vname] = unseen_metrics

            # ---- Mark as completed & save progress ----
            key = f'{link_name}::{vname}'
            progress['completed'].append(key)
            progress['seen_link_results'] = seen_link_results
            progress['unseen_link_results'] = unseen_link_results
            progress['all_histories'] = all_histories
            progress['params_dict'] = params_dict
            save_progress(progress)
            log(f'      [Progress saved: {len(progress["completed"])}/15 completed]')

            del model
            torch.cuda.empty_cache()

    # ====================================================================
    # Aggregate Results
    # ====================================================================
    log(f'\n{"="*60}')
    log('  Aggregating Results')
    log(f'{"="*60}')

    log('\n  [Seen - CICIoT2023]')
    seen_agg = aggregate_results(seen_link_results, params_dict)
    log('\n  [Unseen - DIAD2024]')
    unseen_agg = aggregate_results(unseen_link_results, params_dict)

    # Generalization gap
    gen_gap = {}
    for vname in VARIANT_NAMES:
        if vname in seen_agg and vname in unseen_agg:
            gen_gap[vname] = {
                'f1_drop': unseen_agg[vname]['f1_macro'] - seen_agg[vname]['f1_macro'],
                'accuracy_drop': unseen_agg[vname]['accuracy'] - seen_agg[vname]['accuracy'],
            }
            log(f'    {vname:20s}: F1 gap={gen_gap[vname]["f1_drop"]:+.4f}  '
                f'Acc gap={gen_gap[vname]["accuracy_drop"]:+.4f}')

    # ====================================================================
    # Save JSON Results
    # ====================================================================
    log('\n  [Save] Writing results JSON...')
    json_results = {
        'seen_ciciot2023': {
            'aggregated': seen_agg,
            'link_results': seen_link_results,
            'params': params_dict,
        },
        'unseen_diad2024': {
            'aggregated': unseen_agg,
            'link_results': unseen_link_results,
        },
        'generalization_gap': gen_gap,
    }
    os.makedirs(RESULTS_JSON.parent, exist_ok=True)
    with open(RESULTS_JSON, 'w') as f:
        json.dump(json_results, f, indent=2)
    log(f'    Saved: {RESULTS_JSON.name}')

    # ====================================================================
    # Generate Charts (fig1–fig12)
    # ====================================================================
    log(f'\n{"="*60}')
    log('  Generating Charts')
    log(f'{"="*60}')

    # fig1: Training curves
    if all_histories:
        plot_training_curves(all_histories, 'fig1')

    # fig2: Seen performance bars
    plot_performance_bars(seen_agg,
                         'CICIoT2023 (Seen Test) — Top 3 Models Performance', 'fig2_seen_performance')

    # fig3: Unseen performance bars
    plot_performance_bars(unseen_agg,
                         'DIAD2024 (Unseen Test) — Top 3 Models Performance', 'fig3_unseen_performance')

    # fig4: Generalization gap
    plot_generalization_gap(seen_agg, unseen_agg, 'fig4')

    # fig5: Confusion matrices (seen)
    plot_confusion_matrices(seen_agg,
                           'CICIoT2023 (Seen) — Confusion Matrices', 'fig5_confusion_seen')

    # fig6: Confusion matrices (unseen)
    plot_confusion_matrices(unseen_agg,
                           'DIAD2024 (Unseen) — Confusion Matrices', 'fig6_confusion_unseen')

    # fig7: Per-link F1 (seen)
    plot_per_link_f1(seen_link_results,
                     'CICIoT2023 (Seen) — Per-Link F1-Macro', 'fig7_per_link_seen')

    # fig8: Per-link F1 (unseen)
    plot_per_link_f1(unseen_link_results,
                     'DIAD2024 (Unseen) — Per-Link F1-Macro', 'fig8_per_link_unseen')

    # fig9: Per-link gap heatmap
    plot_per_link_gap_heatmap(seen_link_results, unseen_link_results, 'fig9')

    # fig10: Radar chart
    plot_radar_seen_unseen(seen_agg, unseen_agg, params_dict, 'fig10')

    # fig11: Complexity scatter
    plot_complexity_scatter(seen_agg, unseen_agg, params_dict, 'fig11')

    # fig12: Per-class F1
    plot_class_f1_comparison(seen_agg, unseen_agg, 'fig12')

    # ====================================================================
    # Recommendation Report
    # ====================================================================
    log(f'\n{"="*60}')
    log('  Generating Recommendation Report')
    log(f'{"="*60}')
    generate_recommendation(
        {'aggregated': seen_agg, 'link_results': seen_link_results},
        {'aggregated': unseen_agg, 'link_results': unseen_link_results},
        params_dict, gen_gap)

    # Clean up progress file (all done!)
    if PROGRESS_FILE.exists():
        os.remove(PROGRESS_FILE)
        log('  [Cleanup] Removed progress state file (all training complete)')

    elapsed = time.time() - t_start
    log(f'\n  Total time: {elapsed/60:.1f} minutes')
    log('=' * 70)
    log('  Stage 3 Complete!')
    log('=' * 70)


if __name__ == '__main__':
    main()
