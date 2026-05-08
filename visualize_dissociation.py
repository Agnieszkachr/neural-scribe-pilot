"""
Neural Scribe: Dissociation Visualization (v2.0)

Generates a four-panel comparative figure:
  (a) Classic Stylometry PCA   - overlap in word-frequency space
  (b) Koine-BERT distances     - primary neural probe
  (c) Ancient-Greek-BERT       - robustness check
  (d) Model comparison overlay
"""

import sys, os
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from pathlib import Path

RESULTS_DIR = Path('results')
CLASSIC_JSON = RESULTS_DIR / 'classic_results.json'
KOINE_JSON = RESULTS_DIR / 'neural_results_koine.json'
ANCIENT_JSON = RESULTS_DIR / 'neural_results_ancient.json'


def plot_distance_bars(ax, neural_data, title):
    """Plot distance bar chart on a given axis."""
    results = neural_data.get('distance_results', [])

    texts = ['Paul (ref)'] + [r['text'] for r in results]
    means = [0.0] + [float(r['mean_sigma']) for r in results]

    errors = [[0, 0]]
    for r in results:
        ci_lower, ci_upper = r['ci_95']
        mean = float(r['mean_sigma'])
        errors.append([mean - ci_lower, ci_upper - mean])

    y_pos = np.arange(len(texts))
    errors = np.array(errors).T

    bar_colors = ['#1f77b4']
    for r in results:
        interp = r.get('interpretation', '')
        if 'Significant' in interp:
            bar_colors.append('#d62728')
        elif 'Moderate' in interp or 'Marginal' in interp:
            bar_colors.append('#ff7f0e')
        else:
            bar_colors.append('#2ca02c')

    ax.bar(y_pos, means, yerr=errors, align='center', alpha=0.8,
           color=bar_colors, capsize=8)
    ax.set_xticks(y_pos)
    ax.set_xticklabels(texts, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Distance (sigma units)', fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    gradient = neural_data.get('gradient', {})
    rho = gradient.get('spearman_rho', 0)
    p_val = gradient.get('p_value', 1)
    n = gradient.get('n_texts', 0)
    status = "CONFIRMED" if rho > 0.7 and p_val < 0.05 else "trend"
    gradient_text = f"rho={rho:.2f}, p={p_val:.3f} ({status})"
    ax.text(0.95, 0.95, gradient_text, transform=ax.transAxes,
            ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def main():
    print("Neural Scribe: Generating 4-Panel Visualization (v2.0)")
    print("=" * 55)

    if not CLASSIC_JSON.exists():
        print(f"Error: {CLASSIC_JSON} not found.")
        return
    if not KOINE_JSON.exists():
        print(f"Error: {KOINE_JSON} not found.")
        return

    with open(CLASSIC_JSON, 'r') as f:
        classic_data = json.load(f)
    with open(KOINE_JSON, 'r') as f:
        koine_data = json.load(f)

    has_ancient = ANCIENT_JSON.exists()
    if has_ancient:
        with open(ANCIENT_JSON, 'r') as f:
            ancient_data = json.load(f)

    classic_df = pd.DataFrame(classic_data['chunks'])

    # Setup: 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    sns.set_style("whitegrid")

    # ── (a) Classic Stylometry PCA ────────────────────────
    print("  (a) Classic PCA...")
    ax1 = axes[0, 0]

    palette = {
        'anchor': '#1f77b4', 'colossians': '#ff7f0e',
        '2thessalonians': '#9467bd', 'ephesians': '#8c564b',
        '1timothy': '#2ca02c', '2timothy': '#17becf',
        'titus': '#bcbd22', 'hebrews': '#d62728'
    }

    sns.scatterplot(data=classic_df, x='PC1', y='PC2', hue='group',
                    palette=palette, alpha=0.6, s=80, edgecolor='w', ax=ax1)

    var_exp = classic_data.get('variance_explained', {})
    ax1.set_title('(a) Classic Stylometry (PCA on MFW z-scores)', fontsize=11, fontweight='bold')
    ax1.set_xlabel(f"PC1 ({var_exp.get('PC1', '?')}%)", fontsize=9)
    ax1.set_ylabel(f"PC2 ({var_exp.get('PC2', '?')}%)", fontsize=9)

    # ── (b) Koine-BERT Distances (primary) ────────────────
    print("  (b) Koine-BERT distances...")
    plot_distance_bars(axes[0, 1], koine_data,
                       '(b) Koine-BERT v0.1 (Primary)')

    # ── (c) Ancient-Greek-BERT Distances (robustness) ─────
    print("  (c) Ancient-Greek-BERT distances...")
    if has_ancient:
        plot_distance_bars(axes[1, 0], ancient_data,
                           '(c) Ancient-Greek-BERT (Robustness)')
    else:
        axes[1, 0].text(0.5, 0.5, 'Ancient-Greek-BERT\nnot available',
                        ha='center', va='center', fontsize=14, color='gray',
                        transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('(c) Ancient-Greek-BERT (not run)', fontsize=11)

    # ── (d) Model Comparison ──────────────────────────────
    print("  (d) Model comparison...")
    ax4 = axes[1, 1]

    k_results = {r['text']: float(r['mean_sigma']) for r in koine_data['distance_results']}
    texts = list(k_results.keys())
    k_vals = [k_results[t] for t in texts]

    x = np.arange(len(texts))
    width = 0.35

    ax4.bar(x - width/2, k_vals, width, label='Koine-BERT', color='#2b5797', alpha=0.8)

    if has_ancient:
        a_results = {r['text']: float(r['mean_sigma']) for r in ancient_data['distance_results']}
        a_vals = [a_results.get(t, 0) for t in texts]
        ax4.bar(x + width/2, a_vals, width, label='Ancient-Greek-BERT', color='#b8860b', alpha=0.8)

    ax4.set_xticks(x)
    ax4.set_xticklabels(texts, rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel('Distance (sigma units)', fontsize=9)
    ax4.set_title('(d) Cross-Model Comparison', fontsize=11, fontweight='bold')
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.legend(fontsize=9)

    plt.tight_layout()
    output_path = RESULTS_DIR / 'dissociation.png'
    plt.savefig(output_path, dpi=300)
    print(f"\nVisualization saved to {output_path}")


if __name__ == "__main__":
    main()
