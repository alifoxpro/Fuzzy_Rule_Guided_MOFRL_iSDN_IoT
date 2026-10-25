# -*- coding: utf-8 -*-
"""
============================================================================
FL-RLMAMS Stage 3: Isolated Per-Dataset Training
============================================================================
Each dataset (CICIoT2023, DIAD2024) is trained and tested independently.
Top 3 models from Fuzzy-TOPSIS: LSTM-Only, Mamba-Only, CNN-Transformer.
Per-link local training on 5 SDN links with academic charts.

Usage:
  python train_isolated.py --dataset ciciot2023
  python train_isolated.py --dataset diad2024
  python train_isolated.py --dataset all        # both sequentially
============================================================================
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys, io, json, time, warnings, math, argparse, gc
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

DATASET_LABELS = {
    'ciciot2023': 'CICIoT2023',
    'diad2024': 'DIAD2024',
}

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

VARIANT_NAMES = ['LSTM-Only', 'Mamba-Only', 'CNN-Transformer']

VARIANT_COLORS = {
    'LSTM-Only': '#3498DB',
    'Mamba-Only': '#27AE60',
    'CNN-Transformer': '#9B59B6',
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
# Per-dataset paths
# ============================================================================
def get_paths(dataset_key):
    base = PROJECT / 'experiments' / 'results' / dataset_key
    return {
        'log_file': base / 'training.log',
        'charts_dir': base / 'charts',
        'ckpt_dir': base / 'checkpoints',
        'results_json': base / 'results.json',
        'report_file': base / 'report.txt',
        'progress_file': base / 'progress_state.json',
    }


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
# Progress Tracking (Resume Support)
# ============================================================================
def load_progress(progress_file):
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {
        'completed': [],
        'link_results': {},
        'all_histories': {},
        'params_dict': {},
    }


def save_progress(state, progress_file):
    os.makedirs(progress_file.parent, exist_ok=True)
    tmp = str(progress_file) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    if progress_file.exists():
        os.remove(progress_file)
    os.rename(tmp, progress_file)


def is_completed(state, link_name, variant_name):
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


MAX_ROWS = 3_000_000  # Limit per link to prevent OOM

def load_link_data(csv_path, feature_order):
    log(f'    Loading {csv_path.name}...')
    needed_cols = list(feature_order) + ['severity']
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    usecols = [c for c in needed_cols if c in header]
    df = pd.read_csv(csv_path, usecols=usecols, low_memory=True)
    total_rows = len(df)
    log(f'      Total rows: {fmt(total_rows)}')
    # Subsample if too large (stratified by severity)
    if total_rows > MAX_ROWS:
        log(f'      Subsampling to {fmt(MAX_ROWS)} rows (stratified)...')
        from sklearn.model_selection import train_test_split as tts
        df, _ = tts(df, train_size=MAX_ROWS, random_state=SEED,
                     stratify=df['severity'])
        df = df.reset_index(drop=True)
        log(f'      After subsample: {fmt(len(df))} rows')
    available = [f for f in feature_order if f in df.columns]
    if len(available) < len(feature_order):
        missing = set(feature_order) - set(available)
        log(f'      WARNING: Missing features: {missing}')
    X = df[available].values.astype(np.float32)
    y_str = df['severity'].values
    del df
    gc.collect()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array([SEV_MAP.get(s, 2) for s in y_str], dtype=np.int64)
    del y_str
    gc.collect()
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


def prepare_datasets(csv_path, feature_order):
    """Prepare data with 70/15/15 split (train/val/test). Returns scaler."""
    X_raw, y_raw = load_link_data(csv_path, feature_order)
    log(f'    Creating windows (W={WINDOW_SIZE}, S={STRIDE})...')
    X_win, y_win = create_windows(X_raw, y_raw)
    del X_raw, y_raw
    gc.collect()
    log(f'      Total windows: {fmt(len(y_win))}')
    if len(y_win) == 0:
        return None

    win_counts = Counter(y_win.tolist())
    can_stratify = all(c >= 4 for c in win_counts.values())
    strat_arg = y_win if can_stratify else None
    log(f'      {"Stratified" if can_stratify else "Random"} split')
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_win, y_win, test_size=0.30, random_state=SEED, stratify=strat_arg)
    del X_win, y_win
    gc.collect()
    strat_temp = y_temp if can_stratify else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=strat_temp)
    del X_temp, y_temp
    gc.collect()
    log(f'      Train: {fmt(len(y_train))}  Val: {fmt(len(y_val))}  Test: {fmt(len(y_test))}')

    # Fit scaler on train windows only
    n_feat = X_train.shape[2]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)

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
    return train_dl, val_dl, test_dl, class_weights, scaler, n_feat


# ============================================================================
# Shared Modules
# ============================================================================
class CNNBlock(nn.Module):
    def __init__(self, n_features=14):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        return x


# ============================================================================
# Mamba Block (Pure PyTorch)
# ============================================================================
class SelectiveSSM(nn.Module):
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
# Model Definitions
# ============================================================================
class LSTMOnlyModel(nn.Module):
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

    # Resume from mid-training checkpoint
    resume_path = Path(str(ckpt_path).replace('.pt', '_resume.pt'))
    if resume_path.exists():
        log(f'      RESUME: Loading mid-training checkpoint...')
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

        # Save resume checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history,
                'best_val_loss': best_val_loss,
            }, resume_path)

    # Load best model, clean up resume file
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    if resume_path.exists():
        os.remove(resume_path)
    return history


def full_evaluation(model, test_dl, n_features=14):
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
# Chart Generation (fig1–fig8)
# ============================================================================

def plot_training_curves(all_histories, charts_dir, ds_label, fig_id='fig1'):
    """fig1: Training curves for 3 models (2x3 subplots)."""
    n_models = len(all_histories)
    fig, axes = plt.subplots(2, n_models, figsize=(6 * n_models, 10))
    if n_models == 1:
        axes = axes.reshape(2, 1)
    for i, (vname, hist) in enumerate(all_histories.items()):
        c = VARIANT_COLORS[vname]
        axes[0, i].plot(hist['train_loss'], color=c, linestyle='-', label='Train', linewidth=2)
        axes[0, i].plot(hist['val_loss'], color=c, linestyle='--', label='Val', linewidth=2)
        axes[0, i].set_xlabel('Epoch'); axes[0, i].set_ylabel('Loss')
        axes[0, i].set_title(f'{vname} — Loss')
        axes[0, i].legend()
        axes[1, i].plot(hist['train_acc'], color=c, linestyle='-', label='Train', linewidth=2)
        axes[1, i].plot(hist['val_acc'], color=c, linestyle='--', label='Val', linewidth=2)
        axes[1, i].set_xlabel('Epoch'); axes[1, i].set_ylabel('Accuracy')
        axes[1, i].set_title(f'{vname} — Accuracy')
        axes[1, i].legend()
    plt.suptitle(f'{ds_label} Training — Top 3 Models', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = charts_dir / f'{fig_id}_training_curves.png'
    plt.savefig(out); plt.close()
    log(f'    Saved: {out.name}')


def plot_performance_bars(aggregated, charts_dir, title, fig_id):
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
    out = charts_dir / f'{fig_id}.png'
    plt.savefig(out); plt.close()
    log(f'    Saved: {out.name}')


def plot_confusion_matrices(all_metrics, charts_dir, title, fig_id):
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
    out = charts_dir / f'{fig_id}.png'
    plt.savefig(out); plt.close()
    log(f'    Saved: {out.name}')


def plot_per_link_f1(link_results, charts_dir, title, fig_id):
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
    out = charts_dir / f'{fig_id}.png'
    plt.savefig(out); plt.close()
    log(f'    Saved: {out.name}')


def plot_per_class_f1(aggregated, charts_dir, ds_label, fig_id):
    """Per-class F1 for each model."""
    fig, axes = plt.subplots(1, len(VARIANT_NAMES), figsize=(6 * len(VARIANT_NAMES), 6))
    if len(VARIANT_NAMES) == 1:
        axes = [axes]
    for i, vname in enumerate(VARIANT_NAMES):
        x = np.arange(NUM_CLASSES)
        f1_vals = aggregated[vname].get('f1_per_class', [0]*NUM_CLASSES)
        colors = [SEV_COLORS[s] for s in SEV_NAMES]
        bars = axes[i].bar(x, f1_vals, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, f1_vals):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                         f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(SEV_NAMES)
        axes[i].set_ylim(0, 1.15)
        axes[i].set_title(f'{vname} [{VARIANT_FAMILIES[vname]}]')
        axes[i].set_ylabel('F1-Score')
    plt.suptitle(f'{ds_label} — Per-Class F1 Score', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = charts_dir / f'{fig_id}.png'
    plt.savefig(out); plt.close()
    log(f'    Saved: {out.name}')


def plot_radar(aggregated, params_dict, charts_dir, ds_label, fig_id):
    """Radar chart — multi-metric comparison."""
    categories = ['F1-Macro', 'Accuracy', 'Precision', 'Recall',
                  'Speed\n(1/InfTime)', 'Efficiency\n(1/Params)']
    N = len(categories)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    max_params = max(params_dict.values()) if params_dict else 1
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for vname in VARIANT_NAMES:
        if vname not in aggregated:
            continue
        m = aggregated[vname]
        n_params = params_dict.get(vname, max_params)
        values = [
            m['f1_macro'], m['accuracy'],
            m['precision_macro'], m['recall_macro'],
            min(1.0, 1.0 / (m['inference_time_ms'] + 0.1)),
            1.0 - (n_params / max_params),
        ]
        values += values[:1]
        color = VARIANT_COLORS[vname]
        ax.plot(angles, values, 'o-', linewidth=2.5, color=color, label=vname, markersize=6)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(f'{ds_label} — Top 3 Models Multi-Metric Radar', fontsize=14, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
    plt.tight_layout()
    out = charts_dir / f'{fig_id}_radar.png'
    plt.savefig(out); plt.close()
    log(f'    Saved: {out.name}')


def plot_complexity_scatter(aggregated, params_dict, charts_dir, ds_label, fig_id):
    """Bubble chart — params vs F1, size=inference time."""
    fig, ax = plt.subplots(figsize=(10, 7))
    for vname in VARIANT_NAMES:
        if vname not in aggregated:
            continue
        n_params = params_dict.get(vname, 0)
        c = VARIANT_COLORS[vname]
        m = aggregated[vname]
        ax.scatter(n_params / 1000, m['f1_macro'],
                   s=m['inference_time_ms'] * 30 + 80,
                   c=c, marker='o', alpha=0.7, edgecolors='black', linewidths=1)
        ax.annotate(f'{vname}\nF1={m["f1_macro"]:.3f}', (n_params / 1000, m['f1_macro']),
                    fontsize=9, ha='center', va='bottom',
                    xytext=(0, 10), textcoords='offset points')
    ax.set_xlabel('Parameters (K)'); ax.set_ylabel('F1-Macro')
    ax.set_title(f'{ds_label} — Model Complexity vs F1\n(bubble size = inference time)')
    plt.tight_layout()
    out = charts_dir / f'{fig_id}_complexity.png'
    plt.savefig(out); plt.close()
    log(f'    Saved: {out.name}')


def plot_per_link_heatmap(link_results, charts_dir, ds_label, fig_id):
    """Heatmap: model x link F1 scores."""
    link_names = sorted(link_results.keys())
    data = []
    for vname in VARIANT_NAMES:
        row = [link_results.get(ln, {}).get(vname, {}).get('f1_macro', 0) for ln in link_names]
        data.append(row)
    matrix = np.array(data)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlGnBu',
                xticklabels=[ln.replace('_', '\n') for ln in link_names],
                yticklabels=VARIANT_NAMES, ax=ax, linewidths=0.5,
                vmin=0, vmax=1)
    ax.set_title(f'{ds_label} — Per-Link F1-Macro Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('SDN Link'); ax.set_ylabel('Model')
    plt.tight_layout()
    out = charts_dir / f'{fig_id}_link_heatmap.png'
    plt.savefig(out); plt.close()
    log(f'    Saved: {out.name}')


# ============================================================================
# Aggregation
# ============================================================================
def aggregate_results(link_results, params_dict):
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
# Report
# ============================================================================
def generate_report(aggregated, link_results, params_dict, ds_label, report_file):
    lines = []
    lines.append('=' * 80)
    lines.append(f'  FL-RLMAMS Stage 3: Isolated Training Report — {ds_label}')
    lines.append(f'  Train & Test: {ds_label} (70/15/15 split)')
    lines.append('=' * 80)
    lines.append('')

    lines.append('1. OVERALL PERFORMANCE (Mean across 5 Links)')
    lines.append('-' * 80)
    lines.append(f'{"Model":<20s} {"Acc":>8s} {"F1":>8s} {"Prec":>8s} {"Rec":>8s} {"Inf(ms)":>8s} {"Params":>10s}')
    for vname in VARIANT_NAMES:
        if vname not in aggregated:
            continue
        s = aggregated[vname]
        p = params_dict.get(vname, 0)
        lines.append(f'{vname:<20s} {s["accuracy"]:8.4f} {s["f1_macro"]:8.4f} '
                     f'{s["precision_macro"]:8.4f} {s["recall_macro"]:8.4f} '
                     f'{s["inference_time_ms"]:8.2f} {fmt(p):>10s}')
    lines.append('')

    lines.append('2. PER-LINK F1-MACRO')
    lines.append('-' * 80)
    header = f'{"Model":<20s}'
    link_names = sorted(link_results.keys())
    for ln in link_names:
        short = ln.replace('Link_', 'L').replace('_', '')
        header += f' {short:>10s}'
    header += f' {"Std":>8s}'
    lines.append(header)
    for vname in VARIANT_NAMES:
        row = f'{vname:<20s}'
        f1s = []
        for ln in link_names:
            f1 = link_results.get(ln, {}).get(vname, {}).get('f1_macro', 0)
            f1s.append(f1)
            row += f' {f1:10.4f}'
        row += f' {np.std(f1s):8.4f}'
        lines.append(row)
    lines.append('')

    lines.append('3. RANKING BY F1-MACRO')
    lines.append('-' * 80)
    ranked = sorted([v for v in VARIANT_NAMES if v in aggregated],
                    key=lambda v: aggregated[v]['f1_macro'], reverse=True)
    for rank, vname in enumerate(ranked, 1):
        s = aggregated[vname]
        star = ' *** BEST ***' if rank == 1 else ''
        lines.append(f'  #{rank} {vname:<20s} F1={s["f1_macro"]:.4f} Acc={s["accuracy"]:.4f}{star}')
    lines.append('')

    lines.append('4. PER-LINK ROBUSTNESS')
    lines.append('-' * 80)
    for vname in VARIANT_NAMES:
        link_f1s = []
        for ln, lm in link_results.items():
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
    os.makedirs(report_file.parent, exist_ok=True)
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    log(f'    Saved report: {report_file.name}')
    log('')
    for line in lines:
        log(line)
    return report


# ============================================================================
# Train one dataset
# ============================================================================
def train_dataset(dataset_key):
    global _log_file
    ds_label = DATASET_LABELS[dataset_key]
    paths = get_paths(dataset_key)
    _log_file = paths['log_file']
    charts_dir = paths['charts_dir']
    ckpt_dir = paths['ckpt_dir']

    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    seed_everything(SEED)

    log('=' * 70)
    log(f'  FL-RLMAMS Stage 3: Isolated Training — {ds_label}')
    log(f'  Train & Test: {ds_label} (70/15/15 split)')
    log(f'  Device: {DEVICE}')
    if torch.cuda.is_available():
        log(f'  GPU: {torch.cuda.get_device_name(0)}')
    log(f'  Window: W={WINDOW_SIZE}, Stride={STRIDE}')
    log(f'  Top 3 Models: {VARIANT_NAMES}')
    log('=' * 70)

    # Load feature order
    with open(DATASETS[dataset_key]['json_file'], 'r') as f:
        feat_info = json.load(f)
    feature_order = feat_info['selected_features']
    log(f'  Features: {len(feature_order)} unified features')

    data_dir = DATASETS[dataset_key]['data_dir']
    link_files = sorted(data_dir.glob('Link_*.csv'))
    log(f'  Links: {len(link_files)}')

    # Load progress (resume)
    progress = load_progress(paths['progress_file'])
    n_completed = len(progress['completed'])
    total_tasks = len(link_files) * len(VARIANT_NAMES)
    if n_completed > 0:
        log(f'\n  RESUME: {n_completed}/{total_tasks} completed, skipping them.')

    link_results = progress['link_results']
    all_histories = progress['all_histories']
    params_dict = progress['params_dict']

    t_start = time.time()

    for lf_idx, link_file in enumerate(link_files):
        link_name = link_file.stem

        all_done = all(is_completed(progress, link_name, v) for v in VARIANT_NAMES)
        if all_done:
            log(f'\n  [{lf_idx+1}/{len(link_files)}] {link_name} — ALL COMPLETED (skipping)')
            continue

        log(f'\n{"="*60}')
        log(f'  [{lf_idx+1}/{len(link_files)}] {link_name}')
        log(f'{"="*60}')

        log(f'\n  Preparing {ds_label} data for {link_name}...')
        result = prepare_datasets(link_file, feature_order)
        if result is None:
            log(f'    SKIP: No windows for {link_name}')
            continue
        train_dl, val_dl, test_dl, class_weights, scaler, n_feat = result

        if link_name not in link_results:
            link_results[link_name] = {}

        for vi, vname in enumerate(VARIANT_NAMES):
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

            ckpt_path = ckpt_dir / f'{link_name}_{vname.replace("-", "_")}.pt'
            resume_path = Path(str(ckpt_path).replace('.pt', '_resume.pt'))

            if ckpt_path.exists() and not resume_path.exists():
                log(f'      CACHED: Loading fully trained checkpoint')
                model.load_state_dict(torch.load(ckpt_path, weights_only=True))
                history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
            else:
                history = train_model(model, train_dl, val_dl, class_weights, vname, ckpt_path)

            if lf_idx == 0 and history.get('train_loss'):
                all_histories[vname] = history

            log(f'      [Test] Evaluating on {ds_label} test split...')
            metrics = full_evaluation(model, test_dl, n_feat)
            log(f'        Acc={metrics["accuracy"]:.4f} F1={metrics["f1_macro"]:.4f}')
            link_results[link_name][vname] = metrics

            # Save progress
            key = f'{link_name}::{vname}'
            progress['completed'].append(key)
            progress['link_results'] = link_results
            progress['all_histories'] = all_histories
            progress['params_dict'] = params_dict
            save_progress(progress, paths['progress_file'])
            n_done = len(progress['completed'])
            log(f'      [Progress: {n_done}/{total_tasks} completed]')

            del model
            torch.cuda.empty_cache()
            gc.collect()

        # Free data after each link
        del train_dl, val_dl, test_dl, class_weights
        gc.collect()

    # ====================================================================
    # Aggregate
    # ====================================================================
    log(f'\n{"="*60}')
    log(f'  Aggregating Results — {ds_label}')
    log(f'{"="*60}')

    aggregated = aggregate_results(link_results, params_dict)

    # ====================================================================
    # Save JSON
    # ====================================================================
    json_results = {
        'dataset': dataset_key,
        'label': ds_label,
        'aggregated': aggregated,
        'link_results': link_results,
        'params': params_dict,
    }
    os.makedirs(paths['results_json'].parent, exist_ok=True)
    with open(paths['results_json'], 'w') as f:
        json.dump(json_results, f, indent=2)
    log(f'  Saved: {paths["results_json"].name}')

    # ====================================================================
    # Charts (fig1–fig8)
    # ====================================================================
    log(f'\n{"="*60}')
    log(f'  Generating Charts — {ds_label}')
    log(f'{"="*60}')

    if all_histories:
        plot_training_curves(all_histories, charts_dir, ds_label, 'fig1')

    plot_performance_bars(aggregated, charts_dir,
                          f'{ds_label} — Top 3 Models Performance', 'fig2_performance')

    plot_confusion_matrices(aggregated, charts_dir,
                            f'{ds_label} — Confusion Matrices', 'fig3_confusion')

    plot_per_link_f1(link_results, charts_dir,
                     f'{ds_label} — Per-Link F1-Macro', 'fig4_per_link_f1')

    plot_per_class_f1(aggregated, charts_dir, ds_label, 'fig5_per_class_f1')

    plot_radar(aggregated, params_dict, charts_dir, ds_label, 'fig6')

    plot_complexity_scatter(aggregated, params_dict, charts_dir, ds_label, 'fig7')

    plot_per_link_heatmap(link_results, charts_dir, ds_label, 'fig8')

    # ====================================================================
    # Report
    # ====================================================================
    log(f'\n{"="*60}')
    log(f'  Generating Report — {ds_label}')
    log(f'{"="*60}')
    generate_report(aggregated, link_results, params_dict, ds_label, paths['report_file'])

    # Clean up progress
    if paths['progress_file'].exists():
        os.remove(paths['progress_file'])
        log(f'  [Cleanup] Removed progress state file')

    elapsed = time.time() - t_start
    log(f'\n  {ds_label} Total time: {elapsed/60:.1f} minutes')
    log('=' * 70)
    log(f'  {ds_label} Training Complete!')
    log('=' * 70)

    return json_results


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='FL-RLMAMS Stage 3: Isolated Per-Dataset Training')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['ciciot2023', 'diad2024', 'all'],
                        help='Dataset to train on (default: all)')
    args = parser.parse_args()

    if args.dataset == 'all':
        datasets_to_run = ['ciciot2023', 'diad2024']
    else:
        datasets_to_run = [args.dataset]

    all_results = {}
    for ds_key in datasets_to_run:
        result = train_dataset(ds_key)
        all_results[ds_key] = result

    # If both datasets done, generate cross-dataset comparison
    if len(all_results) == 2:
        generate_cross_dataset_comparison(all_results)


def generate_cross_dataset_comparison(all_results):
    """Generate comparison charts between CICIoT2023 and DIAD2024."""
    global _log_file
    compare_dir = PROJECT / 'experiments' / 'results' / 'comparison'
    os.makedirs(compare_dir, exist_ok=True)
    _log_file = compare_dir / 'comparison.log'

    log('\n' + '=' * 70)
    log('  Cross-Dataset Comparison: CICIoT2023 vs DIAD2024')
    log('=' * 70)

    ds_keys = ['ciciot2023', 'diad2024']
    ds_labels = [DATASET_LABELS[k] for k in ds_keys]

    # fig_cmp1: Side-by-side F1 comparison
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(VARIANT_NAMES))
    width = 0.30
    for di, ds_key in enumerate(ds_keys):
        agg = all_results[ds_key]['aggregated']
        f1s = [agg.get(v, {}).get('f1_macro', 0) for v in VARIANT_NAMES]
        color = '#3498DB' if di == 0 else '#E74C3C'
        bars = ax.bar(x + di * width - width/2, f1s, width,
                      label=DATASET_LABELS[ds_key], color=color, alpha=0.85)
        for bar, val in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xlabel('Model'); ax.set_ylabel('F1-Macro')
    ax.set_title('Cross-Dataset F1-Macro Comparison\n(Each dataset trained & tested independently)')
    ax.set_xticks(x); ax.set_xticklabels(VARIANT_NAMES, fontsize=12)
    ax.legend(fontsize=12); ax.set_ylim(0, 1.15)
    plt.tight_layout()
    out = compare_dir / 'fig_cmp1_f1_comparison.png'
    plt.savefig(out); plt.close()
    log(f'  Saved: {out.name}')

    # fig_cmp2: Per-link heatmap side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(24, 6))
    for di, ds_key in enumerate(ds_keys):
        lr = all_results[ds_key]['link_results']
        link_names = sorted(lr.keys())
        data = []
        for vname in VARIANT_NAMES:
            row = [lr.get(ln, {}).get(vname, {}).get('f1_macro', 0) for ln in link_names]
            data.append(row)
        matrix = np.array(data)
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlGnBu',
                    xticklabels=[ln.replace('_', '\n') for ln in link_names],
                    yticklabels=VARIANT_NAMES, ax=axes[di], linewidths=0.5,
                    vmin=0, vmax=1)
        axes[di].set_title(f'{DATASET_LABELS[ds_key]} — Per-Link F1', fontsize=13, fontweight='bold')
        axes[di].set_xlabel('SDN Link'); axes[di].set_ylabel('Model')
    plt.suptitle('Cross-Dataset Per-Link F1 Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = compare_dir / 'fig_cmp2_per_link_heatmaps.png'
    plt.savefig(out); plt.close()
    log(f'  Saved: {out.name}')

    # fig_cmp3: Radar overlay
    categories = ['F1-Macro', 'Accuracy', 'Precision', 'Recall']
    N = len(categories)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    fig, axes = plt.subplots(1, len(VARIANT_NAMES), figsize=(7 * len(VARIANT_NAMES), 7),
                             subplot_kw=dict(polar=True))
    if len(VARIANT_NAMES) == 1:
        axes = [axes]
    for vi, vname in enumerate(VARIANT_NAMES):
        for di, ds_key in enumerate(ds_keys):
            agg = all_results[ds_key]['aggregated']
            if vname not in agg:
                continue
            m = agg[vname]
            values = [m['f1_macro'], m['accuracy'], m['precision_macro'], m['recall_macro']]
            values += values[:1]
            color = '#3498DB' if di == 0 else '#E74C3C'
            axes[vi].plot(angles, values, 'o-', linewidth=2.5, color=color,
                          label=DATASET_LABELS[ds_key], markersize=6)
            axes[vi].fill(angles, values, alpha=0.1, color=color)
        axes[vi].set_xticks(angles[:-1])
        axes[vi].set_xticklabels(categories, fontsize=9)
        axes[vi].set_ylim(0, 1)
        axes[vi].set_title(f'{vname}', fontsize=13, y=1.08)
        axes[vi].legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.suptitle('Cross-Dataset Radar Comparison', fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    out = compare_dir / 'fig_cmp3_radar_comparison.png'
    plt.savefig(out); plt.close()
    log(f'  Saved: {out.name}')

    # Summary table
    lines = []
    lines.append('=' * 80)
    lines.append('  Cross-Dataset Comparison Summary')
    lines.append('=' * 80)
    lines.append('')
    lines.append(f'{"Model":<20s} {"CICIoT2023 F1":>15s} {"DIAD2024 F1":>15s} {"Delta":>10s}')
    lines.append('-' * 65)
    for vname in VARIANT_NAMES:
        f1_c = all_results['ciciot2023']['aggregated'].get(vname, {}).get('f1_macro', 0)
        f1_d = all_results['diad2024']['aggregated'].get(vname, {}).get('f1_macro', 0)
        delta = f1_d - f1_c
        lines.append(f'{vname:<20s} {f1_c:15.4f} {f1_d:15.4f} {delta:+10.4f}')
    lines.append('')

    report = '\n'.join(lines)
    with open(compare_dir / 'comparison_summary.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    log('')
    for line in lines:
        log(line)


if __name__ == '__main__':
    main()
