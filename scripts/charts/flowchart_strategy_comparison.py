#!/usr/bin/env python3
"""
Flowchart: Strategy A vs Strategy B — Side-by-Side Comparison
IEEE-quality academic figure (600 DPI, Times New Roman)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ============================================================================
# IEEE Style
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

PROJECT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT / 'experiments' / 'comparison_charts' / 'flowcharts'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Colors
# ============================================================================
COL_A = '#3498DB'       # Strategy A blue
COL_A_LIGHT = '#D6EAF8'
COL_B = '#E74C3C'       # Strategy B red
COL_B_LIGHT = '#FADBD8'
COL_SHARED = '#27AE60'  # Shared steps green
COL_SHARED_LIGHT = '#D5F5E3'
COL_INNOVATION = '#9B59B6'  # Innovation purple
COL_INNOVATION_LIGHT = '#E8DAEF'
COL_MASTER = '#F39C12'  # Master Agent orange
COL_MASTER_LIGHT = '#FEF9E7'
COL_DATA = '#7F8C8D'
COL_DATA_LIGHT = '#F2F3F4'
COL_ARROW = '#2C3E50'

# ============================================================================
# Helper: Draw a rounded box with text
# ============================================================================
def draw_box(ax, x, y, w, h, text, facecolor, edgecolor, fontsize=8,
             fontweight='bold', style='round', text_color='#2C3E50',
             linewidth=1.5, subtext=None):
    """Draw a rounded rectangle with centered text."""
    if style == 'round':
        fancy = mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle=mpatches.BoxStyle("Round", pad=0.08),
            facecolor=facecolor, edgecolor=edgecolor,
            linewidth=linewidth, zorder=3
        )
    elif style == 'diamond':
        # Use a rotated square via polygon
        from matplotlib.patches import Polygon
        dx, dy = w/2, h/2
        pts = [(x, y+dy), (x+dx, y), (x, y-dy), (x-dx, y)]
        fancy = Polygon(pts, closed=True, facecolor=facecolor,
                       edgecolor=edgecolor, linewidth=linewidth, zorder=3)
    else:
        fancy = mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle=mpatches.BoxStyle("Round", pad=0.02),
            facecolor=facecolor, edgecolor=edgecolor,
            linewidth=linewidth, zorder=3
        )
    ax.add_patch(fancy)

    if subtext:
        ax.text(x, y + 0.15, text, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight, color=text_color, zorder=4)
        ax.text(x, y - 0.2, subtext, ha='center', va='center',
                fontsize=fontsize - 1.5, fontstyle='italic', color='#555555', zorder=4)
    else:
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight, color=text_color, zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, color=COL_ARROW, style='->', lw=1.2):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                               connectionstyle='arc3,rad=0'),
                zorder=2)


def draw_dashed_arrow(ax, x1, y1, x2, y2, color=COL_ARROW, lw=1.0):
    """Draw a dashed arrow."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                               linestyle='dashed',
                               connectionstyle='arc3,rad=0'),
                zorder=2)


def draw_brace_label(ax, x, y1, y2, text, color='#555555', side='right'):
    """Draw a bracket label on the side."""
    mid = (y1 + y2) / 2
    offset = 0.3 if side == 'right' else -0.3
    ax.annotate('', xy=(x + offset, y1), xytext=(x + offset, y2),
                arrowprops=dict(arrowstyle='-', color=color, lw=0.8,
                               connectionstyle='arc3,rad=0'))
    ax.plot([x + offset - 0.05, x + offset + 0.05], [y1, y1], color=color, lw=0.8)
    ax.plot([x + offset - 0.05, x + offset + 0.05], [y2, y2], color=color, lw=0.8)
    txt_offset = offset + (0.15 if side == 'right' else -0.15)
    ax.text(x + txt_offset, mid, text, ha='left' if side == 'right' else 'right',
            va='center', fontsize=6.5, color=color, fontstyle='italic', rotation=90)


# ============================================================================
# Main Flowchart
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 14))
ax.set_xlim(-5.5, 5.5)
ax.set_ylim(-16.5, 2.5)
ax.set_aspect('equal')
ax.axis('off')

# Title
ax.text(0, 2.0, 'Strategy A vs Strategy B: Pipeline Comparison',
        ha='center', va='center', fontsize=14, fontweight='bold',
        color='#2C3E50', family='serif')
ax.text(0, 1.5, 'FL-RLMAMS: Federated Learning — Reinforcement Learning Multi-Agent System',
        ha='center', va='center', fontsize=9, color='#7F8C8D', family='serif')

# Column headers
bw, bh = 3.2, 0.7  # box width, height

ax.text(-2.8, 0.7, 'Strategy A', ha='center', va='center',
        fontsize=12, fontweight='bold', color=COL_A, family='serif')
ax.text(-2.8, 0.3, '(Baseline Multi-Agent)', ha='center', va='center',
        fontsize=8, color=COL_A, family='serif')

ax.text(2.8, 0.7, 'Strategy B', ha='center', va='center',
        fontsize=12, fontweight='bold', color=COL_B, family='serif')
ax.text(2.8, 0.3, '(Full Innovation Pipeline)', ha='center', va='center',
        fontsize=8, color=COL_B, family='serif')

# Separator line
ax.plot([0, 0], [-0.2, -16], color='#BDC3C7', lw=1, linestyle=':', zorder=1)

# ============================================================================
# Strategy A (Left Column) — x = -2.8
# ============================================================================
xa = -2.8
y0 = -0.5

# Data Input
draw_box(ax, xa, y0, bw, bh, 'Raw CSV Data', COL_DATA_LIGHT, COL_DATA,
         subtext='14 Features × 5 Links')
ya = y0

# Phase A
ya -= 1.4
draw_box(ax, xa, ya, bw, bh, 'Phase A: Independent DQN', COL_A_LIGHT, COL_A,
         subtext='100 Episodes × 5 Agents')
draw_arrow(ax, xa, y0 - bh/2, xa, ya + bh/2, COL_A)
ya_phaseA = ya

# Phase B
ya -= 1.4
draw_box(ax, xa, ya, bw, bh, 'Phase B: FL-DQN (FedAvg)', COL_A_LIGHT, COL_A,
         subtext='30 Rounds × 5 Local Eps')
draw_arrow(ax, xa, ya + 1.4 - bh/2, xa, ya + bh/2, COL_A)

# Phase C
ya -= 1.4
draw_box(ax, xa, ya, bw, bh, 'Phase C: FL-DQN (FedProx)', COL_A_LIGHT, COL_A,
         subtext='30 Rounds, \u03bc=0.01')
draw_arrow(ax, xa, ya + 1.4 - bh/2, xa, ya + bh/2, COL_A)

# Master Agent
ya -= 1.6
draw_box(ax, xa, ya, bw, 0.9, 'Master Agent', COL_MASTER_LIGHT, COL_MASTER,
         fontsize=9, subtext='Health Score → Ranking → Routing')
draw_arrow(ax, xa, ya + 1.6 - bh/2, xa, ya + 0.9/2, COL_MASTER, lw=1.5)
ya_master_a = ya

# Output A
ya -= 1.5
draw_box(ax, xa, ya, bw, 0.7, 'Best Link Selection', COL_SHARED_LIGHT, COL_SHARED,
         subtext='Routing Probability + JFI')
draw_arrow(ax, xa, ya + 1.5 - 0.9/2, xa, ya + 0.7/2, COL_SHARED, lw=1.5)

# Side annotation for A
draw_brace_label(ax, xa - bw/2 - 0.1, ya_phaseA + bh/2, ya_phaseA - 3*1.4 + 1.4 - bh/2,
                 'Standard DQN\n+ Standard FL', COL_A, side='left')

# ============================================================================
# Strategy B (Right Column) — x = 2.8
# ============================================================================
xb = 2.8
yb = -0.5

# Data Input
draw_box(ax, xb, yb, bw, bh, 'Raw CSV Data', COL_DATA_LIGHT, COL_DATA,
         subtext='14 Features × 5 Links')

# Phase B0: AFRAS-FL
yb -= 1.4
draw_box(ax, xb, yb, bw, 0.9, 'Phase B0: AFRAS-FL', COL_INNOVATION_LIGHT, COL_INNOVATION,
         fontsize=9, subtext='7-Phase Preprocessing')
draw_arrow(ax, xb, y0 - bh/2, xb, yb + 0.9/2, COL_INNOVATION)
yb_afras = yb

# AFRAS sub-steps (small boxes)
afras_steps = [
    ('EKF + Wavelet Denoise', '#F5EEF8'),
    ('NURBS-SMOTE Balance', '#F5EEF8'),
    ('K-D Tree + R-Tree Boundary', '#F5EEF8'),
]
yb_sub = yb - 0.75
for i, (step, col) in enumerate(afras_steps):
    draw_box(ax, xb, yb_sub, bw - 0.4, 0.4, step, col, COL_INNOVATION,
             fontsize=6.5, fontweight='normal', linewidth=0.8)
    if i == 0:
        draw_arrow(ax, xb, yb - 0.9/2, xb, yb_sub + 0.2, COL_INNOVATION, lw=0.8)
    else:
        draw_arrow(ax, xb, yb_sub + 0.55, xb, yb_sub + 0.2, COL_INNOVATION, lw=0.8)
    yb_sub -= 0.55

# Phase B1: MOSAR Independent
yb = yb_sub - 0.5
draw_box(ax, xb, yb, bw, bh, 'Phase B1: MOSAR-DQN', COL_B_LIGHT, COL_B,
         subtext='5-Obj Reward × 100 Eps')
draw_arrow(ax, xb, yb + 0.5 + 0.55 - 0.2, xb, yb + bh/2, COL_B)

# Phase B2: MOSAR FedAvg
yb -= 1.4
draw_box(ax, xb, yb, bw, bh, 'Phase B2: MOSAR-FedAvg', COL_B_LIGHT, COL_B,
         subtext='30 Rounds × 5 Local Eps')
draw_arrow(ax, xb, yb + 1.4 - bh/2, xb, yb + bh/2, COL_B)

# Phase B3: MOSAR FedProx
yb -= 1.4
draw_box(ax, xb, yb, bw, bh, 'Phase B3: MOSAR-FedProx', COL_B_LIGHT, COL_B,
         subtext='30 Rounds, \u03bc=0.01')
draw_arrow(ax, xb, yb + 1.4 - bh/2, xb, yb + bh/2, COL_B)

# Phase B4: Fuzzy Augmented
yb -= 1.4
draw_box(ax, xb, yb, bw, bh, 'Phase B4: Fuzzy-Augmented RL', COL_INNOVATION_LIGHT, COL_INNOVATION,
         subtext='Fuzzy Rules + Retrain')
draw_arrow(ax, xb, yb + 1.4 - bh/2, xb, yb + bh/2, COL_INNOVATION)

# Master Agent B
yb -= 1.6
draw_box(ax, xb, yb, bw, 0.9, 'Master Agent', COL_MASTER_LIGHT, COL_MASTER,
         fontsize=9, subtext='Health Score → Ranking → Routing')
draw_arrow(ax, xb, yb + 1.6 - bh/2, xb, yb + 0.9/2, COL_MASTER, lw=1.5)

# Output B
yb -= 1.5
draw_box(ax, xb, yb, bw, 0.7, 'Best Link Selection', COL_SHARED_LIGHT, COL_SHARED,
         subtext='Routing Probability + JFI')
draw_arrow(ax, xb, yb + 1.5 - 0.9/2, xb, yb + 0.7/2, COL_SHARED, lw=1.5)

# Side annotations for B
draw_brace_label(ax, xb + bw/2 + 0.1, yb_afras + 0.9/2, yb_afras - 0.75 - 2*0.55 - 0.2,
                 'AFRAS-FL\nInnovation', COL_INNOVATION, side='right')

# ============================================================================
# Legend
# ============================================================================
legend_y = -15.8
legend_items = [
    (COL_A_LIGHT, COL_A, 'Strategy A (Standard DQN + FL)'),
    (COL_B_LIGHT, COL_B, 'Strategy B (MOSAR + AFRAS + Fuzzy)'),
    (COL_INNOVATION_LIGHT, COL_INNOVATION, 'Innovation Components (B only)'),
    (COL_MASTER_LIGHT, COL_MASTER, 'Master Agent (Shared)'),
    (COL_SHARED_LIGHT, COL_SHARED, 'Output (Shared)'),
]
lx_start = -4.0
for i, (fc, ec, label) in enumerate(legend_items):
    lx = lx_start + i * 2.2
    rect = mpatches.FancyBboxPatch(
        (lx, legend_y - 0.12), 0.3, 0.24,
        boxstyle=mpatches.BoxStyle("Round", pad=0.02),
        facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=3
    )
    ax.add_patch(rect)
    ax.text(lx + 0.4, legend_y, label, ha='left', va='center',
            fontsize=6, color='#2C3E50')

# ============================================================================
# Key Differences annotation
# ============================================================================
ax.text(0, -16.3,
        'Key: Strategy B adds AFRAS-FL preprocessing (7 phases), MOSAR multi-objective reward, and Fuzzy-Augmented RL retraining',
        ha='center', va='center', fontsize=7, color='#7F8C8D', fontstyle='italic')

# Save
out_path = OUT_DIR / 'flowchart_StrategyA_vs_StrategyB.png'
fig.savefig(out_path, dpi=600, bbox_inches='tight', pad_inches=0.1,
            facecolor='white', edgecolor='none')
plt.close(fig)
print(f'Saved: {out_path}')
print('Done!')
