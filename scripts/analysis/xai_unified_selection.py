# -*- coding: utf-8 -*-
"""
============================================================================
FL-RLMAMS Stage 3: Unified XAI Feature Selection (SHAP + LIME)
============================================================================
Cross-dataset feature selection on the ~20 shared features between
CICIoT2023 and DIAD2024 using unified column names.

Steps:
  1. Load stratified samples from unified processed 5-link CSVs
  2. Remove constant features in EITHER dataset
  3. Train XGBoost baseline per dataset
  4. Compute SHAP global feature importance per dataset
  5. Compute LIME local explanations per dataset
  6. Combined ranking: avg normalized SHAP across both datasets
  7. Select top 14 shared features
  8. Generate academic charts
  9. Save results JSON + reduced-feature CSVs
============================================================================
"""

import os, sys, io, json, time, warnings
from pathlib import Path
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

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats

import xgboost as xgb
import shap
import lime
import lime.lime_tabular

# ============================================================================
# Config
# ============================================================================
PROJECT = Path(__file__).resolve().parent.parent.parent
UNIFIED_DIR = PROJECT / 'data' / 'unified_processed'
OUTPUT_DIR = PROJECT / 'data'
CHARTS_DIR = PROJECT / 'experiments' / 'results' / 'charts' / 'xai_unified'
LOG_FILE = PROJECT / 'experiments' / 'results' / 'xai_unified_progress.log'

SEED = 42
N_PER_LINK = 10000
N_LIME_PER_CLASS = 100
N_SELECTED = 14

SEVERITY_MAP = {'High': 0, 'Medium': 1, 'Low': 2}
SEVERITY_NAMES = ['High', 'Medium', 'Low']
META_COLS = ['severity', 'link_id']

DATASETS = {
    'ciciot2023': UNIFIED_DIR / 'ciciot2023_5links',
    'diad2024':   UNIFIED_DIR / 'diad2024_5links',
}

DS_LABELS = {'ciciot2023': 'CICIoT2023', 'diad2024': 'DIAD2024'}
DS_COLORS = {'ciciot2023': '#3498DB', 'diad2024': '#E74C3C'}


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

def setup_style():
    plt.rcParams.update({
        'font.size': 12, 'font.family': 'serif',
        'axes.labelsize': 14, 'axes.titlesize': 15, 'axes.titleweight': 'bold',
        'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'axes.grid': True, 'grid.alpha': 0.3,
        'axes.spines.top': False, 'axes.spines.right': False,
    })


# ============================================================================
# Step 1: Load Stratified Sample
# ============================================================================
def load_stratified_sample(ds_name, data_dir):
    log(f'\n  [Step 1] Loading stratified sample from 5 links...')
    all_samples = []
    link_files = sorted(data_dir.glob('Link_*.csv'))

    for lf in link_files:
        link_name = lf.stem
        log(f'    Reading {link_name}...')

        with open(lf, 'rb') as f:
            total_lines = sum(1 for _ in f) - 1
        log(f'      Total rows: {fmt(total_lines)}')

        if total_lines <= N_PER_LINK:
            df = pd.read_csv(lf, low_memory=False)
        else:
            np.random.seed(SEED)
            all_indices = np.arange(1, total_lines + 1)
            np.random.shuffle(all_indices)
            keep_indices = sorted(all_indices[:N_PER_LINK * 3])
            skip_rows = set(range(1, total_lines + 1)) - set(keep_indices)
            df = pd.read_csv(lf, skiprows=skip_rows, low_memory=False)

        log(f'      Read {len(df)} rows, severity: '
            f'H:{(df["severity"]=="High").sum()} '
            f'M:{(df["severity"]=="Medium").sum()} '
            f'L:{(df["severity"]=="Low").sum()}')

        sampled = df.groupby('severity', group_keys=False).apply(
            lambda x: x.sample(
                n=min(len(x), max(1, int(N_PER_LINK * len(x) / len(df)))),
                random_state=SEED
            )
        )
        all_samples.append(sampled)
        log(f'      Sampled {len(sampled)} rows')

    df_all = pd.concat(all_samples, ignore_index=True)
    log(f'    Total sample: {len(df_all)} rows')

    y_str = df_all['severity'].values
    y = np.array([SEVERITY_MAP[s] for s in y_str])

    X = df_all.drop(columns=META_COLS, errors='ignore')
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

    feature_names = list(X.columns)
    log(f'    Features: {len(feature_names)} numeric columns')
    log(f'    Class dist: H={np.sum(y==0)}, M={np.sum(y==1)}, L={np.sum(y==2)}')

    return X, y, feature_names


# ============================================================================
# Step 2: Remove Constant Features Across Both Datasets
# ============================================================================
def find_constant_features(datasets_data):
    """Find features that are constant (zero variance) in ANY dataset."""
    log(f'\n  [Step 2] Identifying constant features across datasets...')
    constant_in_any = set()

    for ds_name, (X, y, fnames) in datasets_data.items():
        for col in X.columns:
            if X[col].std() < 1e-10:
                constant_in_any.add(col)
                log(f'    {col} is constant in {ds_name}')

    log(f'    Total constant features: {len(constant_in_any)}')
    return constant_in_any


# ============================================================================
# Step 3: Train XGBoost Baseline
# ============================================================================
def train_baseline(X, y, feature_names, ds_name):
    log(f'\n  [Step 3] Training XGBoost baseline for {ds_name} (5-fold CV)...')

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective='multi:softprob', num_class=3,
        tree_method='hist', random_state=SEED,
        n_jobs=-1, eval_metric='mlogloss', verbosity=0
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_acc, cv_f1 = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model.fit(X_train, y_train, verbose=False)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')
        cv_acc.append(acc)
        cv_f1.append(f1)
        log(f'    Fold {fold}: Acc={acc:.4f}, F1={f1:.4f}')

    cv_results = {
        'cv_accuracy_mean': float(np.mean(cv_acc)),
        'cv_accuracy_std': float(np.std(cv_acc)),
        'cv_f1_macro_mean': float(np.mean(cv_f1)),
        'cv_f1_macro_std': float(np.std(cv_f1)),
    }
    log(f'    CV Mean: Acc={cv_results["cv_accuracy_mean"]:.4f}+-{cv_results["cv_accuracy_std"]:.4f}, '
        f'F1={cv_results["cv_f1_macro_mean"]:.4f}+-{cv_results["cv_f1_macro_std"]:.4f}')

    log(f'    Retraining on full {len(X)} samples...')
    model.fit(X, y, verbose=False)
    return model, cv_results


# ============================================================================
# Step 4: SHAP Analysis
# ============================================================================
def compute_shap(model, X, feature_names, ds_name):
    log(f'\n  [Step 4] Computing SHAP values for {ds_name}...')
    t0 = time.time()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    elapsed = time.time() - t0
    log(f'    SHAP computed in {elapsed:.1f}s')

    if isinstance(shap_values, list):
        shap_arr = np.array(shap_values)
        global_imp = np.mean(np.abs(shap_arr), axis=(0, 1))
        per_class_imp = {}
        for c in range(3):
            per_class_imp[SEVERITY_NAMES[c]] = np.mean(np.abs(shap_arr[c]), axis=0)
    else:
        global_imp = np.mean(np.abs(shap_values), axis=(0, 2))
        per_class_imp = {}
        for c in range(3):
            per_class_imp[SEVERITY_NAMES[c]] = np.mean(np.abs(shap_values[:, :, c]), axis=0)

    ranking = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': global_imp
    }).sort_values('shap_importance', ascending=False).reset_index(drop=True)
    ranking['rank'] = range(1, len(ranking) + 1)

    log(f'    Top 10 by SHAP:')
    for _, row in ranking.head(10).iterrows():
        log(f'      #{int(row["rank"]):2d} {row["feature"]:<25s} SHAP={row["shap_importance"]:.6f}')

    return shap_values, ranking, per_class_imp


# ============================================================================
# Step 5: LIME Verification
# ============================================================================
def compute_lime(model, X, y, feature_names, ds_name):
    log(f'\n  [Step 5] Computing LIME explanations for {ds_name} '
        f'({N_LIME_PER_CLASS} per class)...')
    t0 = time.time()

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X.values, feature_names=feature_names,
        class_names=SEVERITY_NAMES, mode='classification',
        random_state=SEED
    )

    np.random.seed(SEED)
    lime_weights = {fn: [] for fn in feature_names}

    for c in range(3):
        class_name = SEVERITY_NAMES[c]
        class_idx = np.where(y == c)[0]
        if len(class_idx) == 0:
            continue
        n_sample = min(N_LIME_PER_CLASS, len(class_idx))
        sample_idx = np.random.choice(class_idx, size=n_sample, replace=False)
        log(f'    Processing {class_name} ({n_sample} instances)...')

        for i, idx in enumerate(sample_idx):
            exp = explainer.explain_instance(
                X.values[idx], model.predict_proba,
                num_features=min(20, len(feature_names)),
                num_samples=1000, labels=(c,)
            )
            feat_w = dict(exp.as_list(label=c))
            for fn in feature_names:
                if fn in feat_w:
                    lime_weights[fn].append(abs(feat_w[fn]))
            if (i + 1) % 25 == 0:
                log(f'      ... {i+1}/{n_sample} done')

    lime_importance = {fn: float(np.mean(ws)) if ws else 0.0
                       for fn, ws in lime_weights.items()}

    lime_ranking = pd.DataFrame({
        'feature': list(lime_importance.keys()),
        'lime_importance': list(lime_importance.values())
    }).sort_values('lime_importance', ascending=False).reset_index(drop=True)
    lime_ranking['lime_rank'] = range(1, len(lime_ranking) + 1)

    elapsed = time.time() - t0
    log(f'    LIME computed in {elapsed:.1f}s')

    log(f'    Top 10 by LIME:')
    for _, row in lime_ranking.head(10).iterrows():
        log(f'      #{int(row["lime_rank"]):2d} {row["feature"]:<25s} LIME={row["lime_importance"]:.6f}')

    return lime_ranking


# ============================================================================
# Step 6: Agreement Metrics
# ============================================================================
def compute_agreement(shap_ranking, lime_ranking, top_k=20):
    log(f'\n  [Step 6] Computing SHAP vs LIME agreement...')

    merged = shap_ranking.merge(
        lime_ranking[['feature', 'lime_rank', 'lime_importance']],
        on='feature', how='inner')

    top_feats = merged.head(top_k)
    rho, p_value = stats.spearmanr(top_feats['rank'], top_feats['lime_rank'])

    shap_top14 = set(shap_ranking.head(N_SELECTED)['feature'])
    lime_top14 = set(lime_ranking.head(N_SELECTED)['feature'])
    jaccard = len(shap_top14 & lime_top14) / len(shap_top14 | lime_top14)

    agreement = {
        'spearman_rho': float(rho) if not np.isnan(rho) else 0.0,
        'spearman_p_value': float(p_value) if not np.isnan(p_value) else 1.0,
        'jaccard_top14': float(jaccard),
        'common_in_top14': sorted(list(shap_top14 & lime_top14)),
        'shap_only_top14': sorted(list(shap_top14 - lime_top14)),
        'lime_only_top14': sorted(list(lime_top14 - shap_top14)),
    }

    log(f'    Spearman rho (top {top_k}): {agreement["spearman_rho"]:.4f}')
    log(f'    Jaccard (top 14): {jaccard:.4f}')
    return agreement, merged


# ============================================================================
# Step 7: Combined Ranking & Selection
# ============================================================================
def combined_selection(per_ds_rankings):
    """Combine SHAP rankings from both datasets using normalized averaging."""
    log(f'\n  [Step 7] Computing combined cross-dataset ranking...')

    combined_scores = {}

    for ds_name, ranking in per_ds_rankings.items():
        max_imp = ranking['shap_importance'].max()
        min_imp = ranking['shap_importance'].min()
        range_imp = max_imp - min_imp if max_imp > min_imp else 1.0

        for _, row in ranking.iterrows():
            feat = row['feature']
            norm_score = (row['shap_importance'] - min_imp) / range_imp
            if feat not in combined_scores:
                combined_scores[feat] = {}
            combined_scores[feat][ds_name] = norm_score

    # Average across datasets
    combined = []
    for feat, scores in combined_scores.items():
        avg_score = np.mean(list(scores.values()))
        combined.append({
            'feature': feat,
            'combined_score': avg_score,
            **{f'norm_shap_{ds}': scores.get(ds, 0.0) for ds in per_ds_rankings}
        })

    combined_df = pd.DataFrame(combined).sort_values(
        'combined_score', ascending=False).reset_index(drop=True)
    combined_df['combined_rank'] = range(1, len(combined_df) + 1)

    selected = combined_df.head(N_SELECTED)['feature'].tolist()

    log(f'    Combined ranking (top {N_SELECTED}):')
    for _, row in combined_df.head(N_SELECTED).iterrows():
        log(f'      #{int(row["combined_rank"]):2d} {row["feature"]:<25s} '
            f'Combined={row["combined_score"]:.4f} '
            f'(CICIoT={row.get("norm_shap_ciciot2023", 0):.4f} '
            f'DIAD={row.get("norm_shap_diad2024", 0):.4f})')

    # Cumulative importance (based on combined scores)
    total = combined_df['combined_score'].sum()
    cum_imp = combined_df['combined_score'].cumsum() / total * 100
    cum_at_14 = float(cum_imp.iloc[N_SELECTED - 1])
    log(f'    Cumulative combined importance at {N_SELECTED}: {cum_at_14:.1f}%')

    return selected, combined_df, cum_imp, cum_at_14


# ============================================================================
# Step 8: Charts
# ============================================================================
def generate_charts(per_ds_data, per_ds_rankings, per_ds_agreements,
                    combined_df, selected, cum_imp):
    log(f'\n  [Step 8] Generating academic charts...')
    os.makedirs(CHARTS_DIR, exist_ok=True)
    setup_style()

    # --- Fig 1: Combined SHAP importance (side-by-side bars) ---
    fig, ax = plt.subplots(figsize=(14, 8))
    top_feats = combined_df.head(N_SELECTED + 3)
    x = np.arange(len(top_feats))
    width = 0.35

    ciciot_vals = top_feats.get('norm_shap_ciciot2023', pd.Series([0]*len(top_feats))).values
    diad_vals = top_feats.get('norm_shap_diad2024', pd.Series([0]*len(top_feats))).values

    ax.bar(x - width/2, ciciot_vals, width, label='CICIoT2023',
           color='#3498DB', alpha=0.85)
    ax.bar(x + width/2, diad_vals, width, label='DIAD2024',
           color='#E74C3C', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(top_feats['feature'].values, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Normalized SHAP Importance')
    ax.set_title('Fig.1: Cross-Dataset SHAP Feature Importance (Unified Features)')
    ax.legend(fontsize=12)

    # Mark selected
    for i, feat in enumerate(top_feats['feature']):
        if feat in selected:
            ax.get_xticklabels()[i].set_fontweight('bold')

    ax.axvline(x=N_SELECTED - 0.5, color='gray', linestyle='--', alpha=0.5,
               label=f'Top {N_SELECTED} cutoff')

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'fig1_combined_shap_importance.png')
    plt.close()
    log(f'    Saved fig1_combined_shap_importance.png')

    # --- Fig 2: Combined ranking bar chart ---
    fig, ax = plt.subplots(figsize=(12, 8))
    all_feats = combined_df.sort_values('combined_score', ascending=True)
    colors = ['#E74C3C' if f in selected else '#BDC3C7'
              for f in all_feats['feature']]
    ax.barh(range(len(all_feats)), all_feats['combined_score'].values,
            color=colors, edgecolor='white', height=0.7)
    ax.set_yticks(range(len(all_feats)))
    ax.set_yticklabels(all_feats['feature'].values, fontsize=9)
    ax.set_xlabel('Combined Normalized SHAP Score')
    ax.set_title('Fig.2: Complete Unified Feature Ranking')
    ax.axhline(y=len(all_feats) - N_SELECTED - 0.5, color='red',
               linestyle='--', alpha=0.7)
    ax.legend(handles=[
        mpatches.Patch(color='#E74C3C', label=f'Selected (Top {N_SELECTED})'),
        mpatches.Patch(color='#BDC3C7', label='Removed')
    ], loc='lower right')
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'fig2_complete_ranking.png')
    plt.close()
    log(f'    Saved fig2_complete_ranking.png')

    # --- Fig 3: Per-dataset SHAP ranking comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for i, (ds_name, ranking) in enumerate(per_ds_rankings.items()):
        ax = axes[i]
        r = ranking.sort_values('shap_importance', ascending=True)
        colors = ['#E74C3C' if f in selected else '#BDC3C7' for f in r['feature']]
        ax.barh(range(len(r)), r['shap_importance'].values,
                color=colors, edgecolor='white', height=0.7)
        ax.set_yticks(range(len(r)))
        ax.set_yticklabels(r['feature'].values, fontsize=9)
        ax.set_xlabel('Mean |SHAP Value|')
        ax.set_title(f'({chr(97+i)}) {DS_LABELS[ds_name]}')
    fig.suptitle('Fig.3: Per-Dataset SHAP Feature Importance', fontsize=16,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'fig3_per_dataset_shap.png')
    plt.close()
    log(f'    Saved fig3_per_dataset_shap.png')

    # --- Fig 4: SHAP vs LIME agreement per dataset ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for i, (ds_name, agr) in enumerate(per_ds_agreements.items()):
        ax = axes[i]
        ranking = per_ds_rankings[ds_name]
        lime_data = per_ds_data[ds_name]['lime_ranking']
        merged = ranking.merge(lime_data[['feature', 'lime_rank']], on='feature', how='inner')
        top_m = merged.head(min(17, len(merged)))
        x_pos = range(len(top_m))
        ax.bar([x - 0.2 for x in x_pos], top_m['rank'].values, 0.35,
               label='SHAP Rank', color='#3498DB')
        ax.bar([x + 0.2 for x in x_pos], top_m['lime_rank'].values, 0.35,
               label='LIME Rank', color='#F39C12')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(top_m['feature'].values, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Rank')
        ax.set_title(f'({chr(97+i)}) {DS_LABELS[ds_name]}\n'
                     f'rho={agr["spearman_rho"]:.3f}, Jaccard={agr["jaccard_top14"]:.3f}')
        ax.legend(fontsize=9)
        ax.invert_yaxis()
    fig.suptitle('Fig.4: SHAP vs LIME Ranking Agreement', fontsize=16,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'fig4_shap_vs_lime_agreement.png')
    plt.close()
    log(f'    Saved fig4_shap_vs_lime_agreement.png')

    # --- Fig 5: Cumulative importance elbow ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(cum_imp) + 1), cum_imp.values, 'b-o', markersize=5, linewidth=2)
    ax.axvline(x=N_SELECTED, color='red', linestyle='--', linewidth=2,
               label=f'Selected: {N_SELECTED} features')
    cum_at_14 = cum_imp.iloc[N_SELECTED - 1]
    ax.axhline(y=cum_at_14, color='red', linestyle=':', alpha=0.5)
    ax.annotate(f'{cum_at_14:.1f}%', xy=(N_SELECTED, cum_at_14),
                xytext=(N_SELECTED + 1, cum_at_14 - 5),
                fontsize=13, fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red'))
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Cumulative Combined Importance (%)')
    ax.set_title('Fig.5: Feature Selection Elbow Curve (Combined)')
    ax.legend(fontsize=12)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'fig5_cumulative_importance.png')
    plt.close()
    log(f'    Saved fig5_cumulative_importance.png')

    # --- Fig 6: Correlation heatmap (selected 14) per dataset ---
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    for i, (ds_name, data) in enumerate(per_ds_data.items()):
        ax = axes[i]
        X = data['X']
        sel_cols = [c for c in selected if c in X.columns]
        corr = X[sel_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                    ax=ax, cbar_kws={'shrink': 0.8}, annot_kws={'size': 7})
        ax.set_title(f'({chr(97+i)}) {DS_LABELS[ds_name]}', fontsize=13)
    fig.suptitle(f'Fig.6: Feature Correlation Heatmap (Selected {N_SELECTED})',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'fig6_correlation_heatmap.png')
    plt.close()
    log(f'    Saved fig6_correlation_heatmap.png')

    # --- Fig 7: Feature reduction summary ---
    fig, ax = plt.subplots(figsize=(8, 6))
    stages = ['Shared Features\n(Semantic Match)', 'After Removing\nConstant', 'XAI Selected\n(SHAP+LIME)']
    n_shared = len(combined_df)
    n_after_const = n_shared  # already removed
    counts = [20, n_shared, N_SELECTED]
    colors_bar = ['#BDC3C7', '#F39C12', '#E74C3C']
    bars = ax.bar(stages, counts, color=colors_bar, edgecolor='white', width=0.5)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(val), ha='center', va='bottom', fontsize=16, fontweight='bold')
    ax.set_ylabel('Number of Features')
    ax.set_title('Fig.7: Unified Feature Selection Pipeline')
    for i in range(len(counts) - 1):
        ax.annotate('', xy=(i+1, counts[i+1] + 0.5),
                    xytext=(i, counts[i] - 0.5),
                    arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2.5))
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'fig7_feature_reduction_pipeline.png')
    plt.close()
    log(f'    Saved fig7_feature_reduction_pipeline.png')


# ============================================================================
# Step 9: Save Results & Generate Reduced CSVs
# ============================================================================
def save_results(selected, combined_df, per_ds_rankings, per_ds_agreements,
                 per_ds_cv, cum_at_14):
    log(f'\n  [Step 9a] Saving results JSON...')

    # Build feature ranking
    feature_ranking = []
    for _, row in combined_df.iterrows():
        entry = {
            'combined_rank': int(row['combined_rank']),
            'feature': row['feature'],
            'combined_score': round(float(row['combined_score']), 6),
        }
        for ds_name in per_ds_rankings:
            ds_ranking = per_ds_rankings[ds_name]
            ds_row = ds_ranking[ds_ranking['feature'] == row['feature']]
            if len(ds_row) > 0:
                entry[f'shap_{ds_name}'] = round(float(ds_row['shap_importance'].values[0]), 6)
                entry[f'rank_{ds_name}'] = int(ds_row['rank'].values[0])
        feature_ranking.append(entry)

    report = {
        'dataset': 'unified',
        'n_original_features': 20,
        'n_usable_features': len(combined_df),
        'n_selected_features': N_SELECTED,
        'selected_features': selected,
        'feature_ranking': feature_ranking,
        'cumulative_importance_at_14': round(cum_at_14, 2),
        'per_dataset_agreement': per_ds_agreements,
        'per_dataset_baseline': per_ds_cv,
        'seed': SEED,
        'method': 'Combined normalized SHAP (0.5*CICIoT + 0.5*DIAD)',
    }

    out_file = OUTPUT_DIR / 'unified_selected_14.json'
    os.makedirs(out_file.parent, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log(f'    Saved: {out_file}')
    return report


def generate_reduced_datasets(selected):
    log(f'\n  [Step 9b] Generating reduced-feature CSVs (streaming)...')
    keep_cols = selected + META_COLS

    for ds_name, data_dir in DATASETS.items():
        out_dir = OUTPUT_DIR / 'unified_14features_5links' / ds_name
        os.makedirs(out_dir, exist_ok=True)
        log(f'    [{ds_name}]')

        link_files = sorted(data_dir.glob('Link_*.csv'))
        for lf in link_files:
            link_name = lf.stem
            out_file = out_dir / f'{link_name}.csv'

            first = True
            total_rows = 0
            for chunk in pd.read_csv(lf, chunksize=100000, low_memory=False,
                                      usecols=lambda c: c in keep_cols):
                # Ensure column order
                chunk = chunk[[c for c in keep_cols if c in chunk.columns]]
                chunk.to_csv(out_file, mode='w' if first else 'a',
                            header=first, index=False)
                first = False
                total_rows += len(chunk)

            log(f'      {link_name}: {fmt(total_rows)} rows, {len(keep_cols)} cols')

    log(f'    Output: {OUTPUT_DIR / "unified_14features_5links"}/')


# ============================================================================
# Main
# ============================================================================
def main():
    os.makedirs(LOG_FILE.parent, exist_ok=True)
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write('')

    t_global = time.time()
    log('=' * 70)
    log('  FL-RLMAMS Stage 3: Unified XAI Feature Selection')
    log('  Cross-Dataset: CICIoT2023 + DIAD2024')
    log('  Method: Combined Normalized SHAP + LIME Verification')
    log('=' * 70)

    # ---- Step 1: Load data from both datasets ----
    datasets_data = {}
    for ds_name, data_dir in DATASETS.items():
        log(f'\n{"="*60}')
        log(f'  Dataset: {ds_name.upper()}')
        log(f'{"="*60}')
        X, y, feature_names = load_stratified_sample(ds_name, data_dir)
        datasets_data[ds_name] = {'X': X, 'y': y, 'features': feature_names}

    # ---- Step 2: Remove constant features ----
    constant_feats = find_constant_features(
        {ds: (d['X'], d['y'], d['features']) for ds, d in datasets_data.items()})

    for ds_name, data in datasets_data.items():
        drop = [c for c in constant_feats if c in data['X'].columns]
        if drop:
            data['X'] = data['X'].drop(columns=drop)
            data['features'] = [f for f in data['features'] if f not in constant_feats]
            log(f'  Dropped {len(drop)} constant features from {ds_name}: {drop}')

    # ---- Steps 3-6: Per-dataset XAI ----
    per_ds_rankings = {}
    per_ds_agreements = {}
    per_ds_cv = {}

    for ds_name, data in datasets_data.items():
        log(f'\n{"="*60}')
        log(f'  XAI Analysis: {ds_name.upper()}')
        log(f'{"="*60}')

        X, y, fnames = data['X'], data['y'], data['features']

        # Step 3: Baseline
        model, cv_results = train_baseline(X, y, fnames, ds_name)
        per_ds_cv[ds_name] = cv_results

        # Step 4: SHAP
        shap_values, shap_ranking, per_class_imp = compute_shap(model, X, fnames, ds_name)

        # Step 5: LIME
        lime_ranking = compute_lime(model, X, y, fnames, ds_name)
        data['lime_ranking'] = lime_ranking

        # Step 6: Agreement
        agreement, merged = compute_agreement(shap_ranking, lime_ranking)
        per_ds_rankings[ds_name] = shap_ranking
        per_ds_agreements[ds_name] = agreement

    # ---- Step 7: Combined selection ----
    selected, combined_df, cum_imp, cum_at_14 = combined_selection(per_ds_rankings)

    # ---- Step 8: Charts ----
    generate_charts(datasets_data, per_ds_rankings, per_ds_agreements,
                    combined_df, selected, cum_imp)

    # ---- Step 9: Save results & reduced CSVs ----
    report = save_results(selected, combined_df, per_ds_rankings,
                          per_ds_agreements, per_ds_cv, cum_at_14)
    generate_reduced_datasets(selected)

    # ---- Summary ----
    elapsed = time.time() - t_global
    log(f'\n{"="*70}')
    log(f'  Unified XAI Feature Selection Complete!')
    log(f'  Time: {elapsed/60:.1f} minutes')
    log(f'{"="*70}')
    log(f'\n  Selected {N_SELECTED} unified features:')
    for i, f in enumerate(selected, 1):
        log(f'    {i:2d}. {f}')
    log(f'\n  Output: {OUTPUT_DIR / "unified_14features_5links"} /')
    log(f'  JSON:   {OUTPUT_DIR / "unified_selected_14.json"}')
    log(f'  Charts: {CHARTS_DIR}/')
    log('=' * 70)


if __name__ == '__main__':
    main()
