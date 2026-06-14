#!/usr/bin/env python3
"""Side-by-side bar chart: FlashAttention vs Matmul speedup vs PPL.

Left subplot (FlashAttention) uses a broken y-axis (1..2 cut, 11..14.5 top)
so trend among 4 TL configs is visible without compressing the bottom.
Right subplot (Matmul) is a single-axis bar plot from 1.0..1.20.

`gap_coef` only controls how far the *colored bar fill* extends into the
inter-panel gap. Axes layout, ylim and tick locations stay fixed regardless
of its value.
  - gap_coef = 0   no extension beyond the spine (clear // break)
  - gap_coef = 10  bars fully bridge the gap (no visual break)
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

plt.rcParams.update({
    'font.size': 14,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.labelsize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
})

data = {
    'TL (bare)':      {'FlashAttention': 11.99, 'Matmul': 1.03},
    'TL + addr':      {'FlashAttention': 12.97, 'Matmul': 1.08},
    'TL + pipe':      {'FlashAttention': 13.20, 'Matmul': 1.09},
    'TL + addr+pipe': {'FlashAttention': 13.65, 'Matmul': 1.15},
}
configs = list(data.keys())
xticklabels = ['TL\n(bare)', 'TL\n+addr', 'TL\n+pipe', 'TL\n+addr+pipe']
colors = ["#428cc1", "#e99041", "#5bb85b", "#d15555"]

gap_coef = 6  # 0..10, see module docstring

# layout: left has broken axis (2 stacked subplots), right is single
fig = plt.figure(figsize=(11, 5))
outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30)
left = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=outer[0], height_ratios=[4, 1], hspace=0.18)
ax_top = fig.add_subplot(left[0])
ax_bot = fig.add_subplot(left[1])
ax_mm = fig.add_subplot(outer[1])

x = np.arange(len(configs))
width = 0.55

# --- FlashAttention (broken axis) ---
# Fixed axes layout: top 11..14.5, bottom 1..2
ax_top.set_ylim(11, 14.5)
ax_bot.set_ylim(1.0, 2.0)
ax_top.set_yticks(np.arange(11, 14.6, 1))
ax_bot.set_yticks([1.0, 2.0])
ax_top.spines['bottom'].set_visible(False)
ax_bot.spines['top'].set_visible(False)
ax_top.tick_params(labelbottom=False, bottom=False)

# Plot the normal bars (clipped to their own axis range)
for xi, cfg, col in zip(x, configs, colors):
    v = data[cfg]['FlashAttention']
    ax_top.bar(xi, v, width, color=col, edgecolor='none')
    ax_bot.bar(xi, v, width, color=col, edgecolor='none')

# annotate values
for ax in (ax_top, ax_bot):
    y_lo, y_hi = ax.get_ylim()
    for xi, cfg in zip(x, configs):
        v = data[cfg]['FlashAttention']
        if y_lo <= v <= y_hi:
            ax.text(xi, v, f'{v:.2f}×', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')

ax_bot.set_xticks(x)
ax_bot.set_xticklabels(xticklabels)
ax_bot.set_xlabel('(a) FlashAttention', labelpad=10, fontsize=15)
ax_top.grid(axis='y', alpha=0.3)
ax_bot.grid(axis='y', alpha=0.3)
ax_top.set_ylabel('')
ax_bot.set_ylabel('')

# --- y-axis stubs + diagonal break marks (in figure coords) ---
fig.canvas.draw()
bbox_top = ax_top.get_position()
bbox_bot = ax_bot.get_position()
gap_fig = bbox_top.y0 - bbox_bot.y1            # full gap height in fig coords
stub_len = gap_fig * 0.30
for xpos in (bbox_top.x0, bbox_top.x1):
    fig.add_artist(plt.Line2D(
        [xpos, xpos], [bbox_top.y0, bbox_top.y0 - stub_len],
        color='k', lw=1, transform=fig.transFigure, clip_on=False))
    fig.add_artist(plt.Line2D(
        [xpos, xpos], [bbox_bot.y1, bbox_bot.y1 + stub_len],
        color='k', lw=1, transform=fig.transFigure, clip_on=False))

dx_fig = 0.005
dy_fig = 0.008
for xpos in (bbox_top.x0, bbox_top.x1):
    y_end_top = bbox_top.y0 - stub_len
    fig.add_artist(plt.Line2D(
        [xpos - dx_fig, xpos + dx_fig],
        [y_end_top - dy_fig, y_end_top + dy_fig],
        color='k', lw=1, transform=fig.transFigure, clip_on=False))
    y_end_bot = bbox_bot.y1 + stub_len
    fig.add_artist(plt.Line2D(
        [xpos - dx_fig, xpos + dx_fig],
        [y_end_bot - dy_fig, y_end_bot + dy_fig],
        color='k', lw=1, transform=fig.transFigure, clip_on=False))

# --- Colored "bridge" rectangles for the gap, controlled solely by gap_coef.
# Each rectangle starts at the spine and grows toward the center of the gap.
# At gap_coef=10 the two rectangles meet -> no visible break.
from matplotlib.patches import Rectangle

# x positions for bars in figure coords
x_data_to_fig = lambda x_data, bbox: (
    bbox.x0 + (x_data - ax_top.get_xlim()[0])
    / (ax_top.get_xlim()[1] - ax_top.get_xlim()[0]) * bbox.width
)
half_w_fig = (
    bbox_top.width * width
    / (ax_top.get_xlim()[1] - ax_top.get_xlim()[0]) / 2
)
half_extend = gap_fig / 2 * (gap_coef / 10.0)
for xi, cfg, col in zip(x, configs, colors):
    xc_fig = x_data_to_fig(xi, bbox_top)
    fig.add_artist(Rectangle(
        (xc_fig - half_w_fig, bbox_top.y0 - half_extend),
        2 * half_w_fig, half_extend,
        facecolor=col, edgecolor='none', transform=fig.transFigure,
        zorder=0.5, clip_on=False))
    fig.add_artist(Rectangle(
        (xc_fig - half_w_fig, bbox_bot.y1),
        2 * half_w_fig, half_extend,
        facecolor=col, edgecolor='none', transform=fig.transFigure,
        zorder=0.5, clip_on=False))

# centered y-label for the broken left panel
y_mid = (bbox_top.y1 + bbox_bot.y0) / 2
x_label = bbox_top.x0 - 0.06
fig.text(x_label, y_mid, 'Speedup over PPL',
         rotation=90, ha='center', va='center',
         fontsize=15, fontweight='bold')

# --- Matmul (single axis) ---
for xi, cfg, col in zip(x, configs, colors):
    v = data[cfg]['Matmul']
    ax_mm.bar(xi, v, width, color=col, edgecolor='none')
    ax_mm.text(xi, v, f'{v:.2f}×', ha='center', va='bottom',
               fontsize=11, fontweight='bold')
ax_mm.set_xticks(x)
ax_mm.set_xticklabels(xticklabels)
ax_mm.set_ylim(1.0, 1.20)
ax_mm.set_yticks(np.arange(1.0, 1.21, 0.05))
ax_mm.grid(axis='y', alpha=0.3)
ax_mm.set_ylabel('Speedup over PPL')
ax_mm.set_xlabel('(b) Matmul', labelpad=10, fontsize=15)

out_png = '/mnt2/users/tilelanguser-xxw/tilelang-tpu/tpu_benchmark/abl/speedup_vs_ppl_split.png'
out_pdf = '/mnt2/users/tilelanguser-xxw/tilelang-tpu/tpu_benchmark/abl/speedup_vs_ppl_split.pdf'
fig.savefig(out_png, dpi=140, bbox_inches='tight')
fig.savefig(out_pdf, bbox_inches='tight')
print(f"saved: {out_png}")
print(f"saved: {out_pdf}")
