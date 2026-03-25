"""
experiments/plot_results.py

Generates all figures for the report:
    fig1_net2vec_by_layer.pdf     - Single vs multi-filter IoU across layers
    fig2_net2vec_per_concept.pdf  - Per-concept IoU at conv5
    fig3_tcav_heatmap.pdf         - TCAV scores across concepts x layers
    fig4_comparison_scatter.pdf   - Net2Vec IoU vs TCAV score (key figure)

Usage (in Colab):
    !python net2vec-tcav/experiments/plot_results.py \
        --results_dir net2vec-tcav/results \
        --output_dir net2vec-tcav/figures
"""

import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

CONCEPTS = ['sky', 'grass', 'building', 'person',
            'car', 'wheel', 'water', 'wood']
LAYERS   = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default='results')
    p.add_argument('--output_dir',  default='figures')
    return p.parse_args()


def load_results(results_dir):
    with open(os.path.join(results_dir, 'net2vec_results.pkl'), 'rb') as f:
        n2v = pickle.load(f)
    with open(os.path.join(results_dir, 'tcav_results.pkl'), 'rb') as f:
        tcav = pickle.load(f)
    return n2v, tcav


# ----------------------------------------------------------------
# Figure 1: Single vs multi-filter IoU across layers
# ----------------------------------------------------------------
def plot_net2vec_by_layer(n2v, output_dir):
    """
    Bar chart showing mean IoU across concepts per layer,
    for single-filter and multi-filter side by side.
    Replicates the style of Figure 2 in the paper.
    """
    single_means, single_errs = [], []
    multi_means,  multi_errs  = [], []

    for layer in LAYERS:
        single_vals, multi_vals = [], []
        for concept in CONCEPTS:
            if concept in n2v and layer in n2v[concept]:
                single_vals.append(n2v[concept][layer]['single_val_iou'])
                multi_vals.append(n2v[concept][layer]['multi_val_iou'])
        single_means.append(np.mean(single_vals))
        single_errs.append(np.std(single_vals) / np.sqrt(len(single_vals)))
        multi_means.append(np.mean(multi_vals))
        multi_errs.append(np.std(multi_vals) / np.sqrt(len(multi_vals)))

    x = np.arange(len(LAYERS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width/2, single_means, width, yerr=single_errs,
           capsize=4, label='Single filter', color='#4C72B0',
           alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, multi_means, width, yerr=multi_errs,
           capsize=4, label='Multi-filter', color='#C44E52',
           alpha=0.85, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(LAYERS)
    ax.set_xlabel('AlexNet layer')
    ax.set_ylabel('Mean set IoU (val)')
    ax.set_title('Net2Vec: single-filter vs multi-filter concept segmentation')
    ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig1_net2vec_by_layer.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ----------------------------------------------------------------
# Figure 2: Per-concept IoU at conv5
# ----------------------------------------------------------------
def plot_net2vec_per_concept(n2v, output_dir):
    """
    Grouped bar chart showing single vs multi-filter IoU
    for each concept at conv5.
    """
    concepts_present = [c for c in CONCEPTS
                        if c in n2v and 'conv5' in n2v[c]]
    single_ious = [n2v[c]['conv5']['single_val_iou']
                   for c in concepts_present]
    multi_ious  = [n2v[c]['conv5']['multi_val_iou']
                   for c in concepts_present]

    x = np.arange(len(concepts_present))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width/2, single_ious, width,
           label='Single filter', color='#4C72B0',
           alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, multi_ious, width,
           label='Multi-filter', color='#C44E52',
           alpha=0.85, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(concepts_present, rotation=30, ha='right')
    ax.set_ylabel('Set IoU (val)')
    ax.set_title('Net2Vec concept segmentation IoU at conv5')
    ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig2_net2vec_per_concept.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ----------------------------------------------------------------
# Figure 3: TCAV heatmap
# ----------------------------------------------------------------
def plot_tcav_heatmap(tcav, output_dir):
    """
    Heatmap of TCAV scores for each concept x layer.
    Diverging colormap centred at 0.5 (the random baseline).
    """
    concepts_present = [c for c in CONCEPTS if c in tcav]
    matrix = np.zeros((len(concepts_present), len(LAYERS)))

    for i, concept in enumerate(concepts_present):
        for j, layer in enumerate(LAYERS):
            if layer in tcav[concept]:
                matrix[i, j] = tcav[concept][layer]['mean_tcav_score']
            else:
                matrix[i, j] = 0.5

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap='RdBu', vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(range(len(LAYERS)))
    ax.set_xticklabels(LAYERS)
    ax.set_yticks(range(len(concepts_present)))
    ax.set_yticklabels(concepts_present)
    ax.set_title('TCAV scores — target class: car (ImageNet 817)')

    plt.colorbar(im, ax=ax, label='TCAV score (0.5 = random baseline)')

    # Annotate each cell
    for i in range(len(concepts_present)):
        for j in range(len(LAYERS)):
            ax.text(j, i, f'{matrix[i,j]:.2f}',
                    ha='center', va='center', fontsize=8,
                    color='black')

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig3_tcav_heatmap.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ----------------------------------------------------------------
# Figure 4: Net2Vec IoU vs TCAV score scatter (KEY FIGURE)
# ----------------------------------------------------------------
def plot_comparison_scatter(n2v, tcav, output_dir):
    """
    Scatter plot: Net2Vec best-layer IoU (x) vs TCAV conv5 score (y).

    This is the main novel contribution figure.
    Each point is a concept. Quadrants reveal the relationship
    between spatial encoding and causal influence.
    """
    palette = sns.color_palette('tab10', len(CONCEPTS))
    concepts_present = [c for c in CONCEPTS
                        if c in n2v and c in tcav
                        and 'conv5' in tcav[c]]

    # Best-layer IoU = max single-filter IoU across all layers
    best_ious = []
    for c in concepts_present:
        ious = [n2v[c][l]['single_val_iou']
                for l in LAYERS if l in n2v[c]]
        best_ious.append(max(ious))

    # TCAV score at conv5
    tcav_scores = [tcav[c]['conv5']['mean_tcav_score']
                   for c in concepts_present]

    fig, ax = plt.subplots(figsize=(7, 6))

    for i, concept in enumerate(concepts_present):
        ax.scatter(best_ious[i], tcav_scores[i],
                   s=100, color=palette[i],
                   zorder=3, label=concept)
        ax.annotate(concept,
                    (best_ious[i], tcav_scores[i]),
                    textcoords='offset points',
                    xytext=(8, 4), fontsize=9)

    # Random baseline
    ax.axhline(0.5, color='gray', linestyle='--',
               linewidth=0.9, alpha=0.7,
               label='TCAV null (0.5)')

    # Linear regression
    if len(best_ious) > 2:
        slope, intercept, r, p, _ = stats.linregress(
            best_ious, tcav_scores
        )
        xs = np.linspace(min(best_ious), max(best_ious), 100)
        ax.plot(xs, slope * xs + intercept,
                'k-', linewidth=1.5, alpha=0.6,
                label=f'Linear fit (r={r:.2f}, p={p:.3f})')
        print(f"\nCorrelation: r={r:.3f}, p={p:.4f}")

    # Quadrant labels
    iou_mid = np.median(best_ious)
    ax.axvline(iou_mid, color='lightgray',
               linestyle=':', linewidth=0.8)
    ax.text(min(best_ious), 1.03,
            'low IoU / high TCAV',
            fontsize=8, color='gray', style='italic')
    ax.text(iou_mid * 1.05, -0.02,
            'high IoU / low TCAV',
            fontsize=8, color='gray', style='italic',
            va='top')

    ax.set_xlabel('Net2Vec best-layer IoU (single filter)', fontsize=12)
    ax.set_ylabel('TCAV score at conv5 (target: car)', fontsize=12)
    ax.set_title('Spatial encoding vs causal influence\n'
                 'Net2Vec IoU vs TCAV score per concept',
                 fontsize=12)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=8, framealpha=0.7, loc='upper left')

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig4_comparison_scatter.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading results...")
    n2v, tcav = load_results(args.results_dir)

    print("\nGenerating figures...")
    plot_net2vec_by_layer(n2v,       args.output_dir)
    plot_net2vec_per_concept(n2v,    args.output_dir)
    plot_tcav_heatmap(tcav,          args.output_dir)
    plot_comparison_scatter(n2v, tcav, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == '__main__':
    main()