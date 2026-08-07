import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import numpy as np

plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'text.color': '#c9d1d9',
    'axes.labelcolor': '#8b949e',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.alpha': 0.8,
    'font.family': 'DejaVu Sans',
    'font.size': 10,
})

# ══════════════════════════════════════════════════════════════════════
# Data: Round 1 key milestones (simplified from ~100 experiments)
# ══════════════════════════════════════════════════════════════════════
r1_labels = [
    'Baseline', 'Chunk\n2048', 'BF16 KV', 'Chunk\n512', 'Block\n64',
    'MoE\nblock128', 'Skip\nweight', 'NKI\nrouter', 'Scale\nfuse Q',
    'Remove\nmask/NaN', 'Remove\nscale wrap', '6-way\nSBUF'
]
r1_toks = [570.6, 1078.1, 1082.9, 1139.3, 1148.9, 1173.1, 1219.2, 1227.3, 1247.9, 1450.6, 1452.4, 1454.0]
r1_phases = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3]
r1_phase_colors = {1: '#58a6ff', 2: '#3fb950', 3: '#f0883e'}

# ══════════════════════════════════════════════════════════════════════
# Data: Round 2 key milestones
# ══════════════════════════════════════════════════════════════════════
r2_labels = [
    'Baseline\n(new bench)', 'Batch1024\n+block128', 'GQA\nbroadcast',
    'BF16\nQK/PV', 'BF16\nsoftmax', 'Pre-scale\nQ', 'BF16\nRMSNorm',
    'NKI flash\nattention', 'ModelLen\n+O3'
]
r2_toks = [845, 899.5, 1031.5, 1124.5, 1555.8, 1611.8, 1626.0, 4071.1, 4269.0]
r2_gains = ['', '+6%', '+15%', '+9%', '+38%', '+4%', '+1%', '+150%', '+5%']

# ══════════════════════════════════════════════════════════════════════
# Create figure with two subplots
# ══════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 1.2]})
fig.subplots_adjust(hspace=0.38, top=0.89, bottom=0.07, left=0.08, right=0.95)

fig.suptitle('Tongyi-30B-A3B (Qwen3MoE) Prefill Throughput Optimization',
             fontsize=16, fontweight='bold', color='#f0f6fc', y=0.97)
fig.text(0.5, 0.935, 'AWS Trainium 2  |  TP=4  |  100% correctness  |  Two rounds, different benchmarks',
         ha='center', fontsize=10, color='#8b949e')

# ══════════════════════════════════════════════════════════════════════
# Round 1 chart
# ══════════════════════════════════════════════════════════════════════
ax1.set_title('Round 1: Short-Context (16K)  —  571 → 1,454 tok/s  (2.5× speedup)',
              fontsize=12, fontweight='bold', color='#c9d1d9', pad=14)

x1 = np.arange(len(r1_labels))
colors1 = [r1_phase_colors[p] for p in r1_phases]

ax1.plot(x1, r1_toks, color='#3fb950', linewidth=2, zorder=2, alpha=0.6)
ax1.scatter(x1, r1_toks, c=colors1, s=70, zorder=3, edgecolors='white', linewidths=0.5)

# Fill area
ax1.fill_between(x1, r1_toks, alpha=0.08, color='#3fb950')

# Annotate the big jump
ax1.annotate('+16.2%\n(remove\nredundant ops)', xy=(9, 1450.6), xytext=(9, 1250),
             fontsize=8, color='#3fb950', fontweight='bold', ha='center',
             arrowprops=dict(arrowstyle='->', color='#3fb950', lw=1.2))

# Phase separators
ax1.axvline(4.5, color='#30363d', linestyle='--', linewidth=1, alpha=0.7)
ax1.axvline(10.5, color='#30363d', linestyle='--', linewidth=1, alpha=0.7)
ax1.text(2, 1540, 'Phase 1: Params', fontsize=9, color='#58a6ff', ha='center', alpha=0.8)
ax1.text(7.5, 1540, 'Phase 2: Model Code', fontsize=9, color='#3fb950', ha='center', alpha=0.8)
ax1.text(11, 1540, 'P3', fontsize=9, color='#f0883e', ha='center', alpha=0.8)

# Note about failed experiments
ax1.text(11, 1100, '~40 MoE kernel\nexperiments: all <1%', fontsize=8,
         color='#f85149', ha='center', style='italic',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8514910', edgecolor='#f8514940'))

ax1.set_xticks(x1)
ax1.set_xticklabels(r1_labels, fontsize=8)
ax1.set_ylabel('tok/s')
ax1.set_ylim(400, 1620)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{int(v):,}'))
ax1.grid(True, axis='y', alpha=0.5)

# ══════════════════════════════════════════════════════════════════════
# Round 2 chart
# ══════════════════════════════════════════════════════════════════════
ax2.set_title('Round 2: Long-Context (32K, 10×3000 tok)  —  845 → 4,269 tok/s  (5.1× speedup)',
              fontsize=12, fontweight='bold', color='#c9d1d9', pad=14)

x2 = np.arange(len(r2_labels))

# Gradient-like color for the line
ax2.plot(x2, r2_toks, color='#a371f7', linewidth=2.5, zorder=2)
ax2.scatter(x2, r2_toks, c='#a371f7', s=80, zorder=3, edgecolors='white', linewidths=0.8)
ax2.fill_between(x2, r2_toks, alpha=0.06, color='#a371f7')

# Annotate gains
for i, (tok, gain) in enumerate(zip(r2_toks, r2_gains)):
    if gain and gain in ('+38%', '+150%'):
        color = '#f0883e' if gain == '+150%' else '#a371f7'
        weight = 'bold'
        fontsize = 11
    elif gain:
        color = '#8b949e'
        weight = 'normal'
        fontsize = 8
    else:
        continue
    yoff = 200 if tok > 2000 else 120
    ax2.annotate(gain, xy=(i, tok), xytext=(i, tok + yoff),
                 fontsize=fontsize, color=color, fontweight=weight, ha='center',
                 arrowprops=dict(arrowstyle='-', color=color, lw=0.8, alpha=0.5) if tok > 2000 else None)

# Highlight the two big jumps
ax2.axhspan(1500, 1620, xmin=0.22, xmax=0.42, alpha=0.04, color='#a371f7')
ax2.axhspan(4000, 4300, xmin=0.75, xmax=0.95, alpha=0.04, color='#f0883e')

# Star annotation for flash attention
ax2.annotate('★ Largest single gain\n(nkilib kernel integration)',
             xy=(7, 4071), xytext=(5.5, 3400),
             fontsize=9, color='#f0883e', fontweight='bold', ha='center',
             arrowprops=dict(arrowstyle='->', color='#f0883e', lw=1.5))

ax2.set_xticks(x2)
ax2.set_xticklabels(r2_labels, fontsize=8.5)
ax2.set_ylabel('tok/s')
ax2.set_ylim(400, 4800)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v/1000:.1f}K' if v >= 1000 else f'{int(v)}'))
ax2.grid(True, axis='y', alpha=0.5)

# Final result badge
ax2.text(8.3, 4269, '4,269 tok/s', fontsize=11, fontweight='bold', color='#f0f6fc',
         ha='center', va='bottom',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#a371f7', edgecolor='none', alpha=0.9))

# ══════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════
plt.savefig('/dev3/zigeng/bc/opt/optimization_timeline.png', dpi=180, bbox_inches='tight')
print("Saved: /dev3/zigeng/bc/opt/optimization_timeline.png")
