"""
Neural Scribe: Dissociation Visualization

Generates a comparative figure:
- Left: Classic Stylometry PCA (Showing overlap)
- Right: Neural Probe Cosine Distances (Showing gradient)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Configuration
RESULTS_DIR = Path('results')
CLASSIC_JSON = RESULTS_DIR / 'classic_results.json'
NEURAL_JSON = RESULTS_DIR / 'neural_results.json'


def main():
    print("Neural Scribe: Generating Visualization")
    print("=" * 50)
    
    if not CLASSIC_JSON.exists() or not NEURAL_JSON.exists():
        print("Error: Results not found. Run analysis scripts first.")
        return

    # Load Data (both JSON now)
    with open(CLASSIC_JSON, 'r') as f:
        classic_data = json.load(f)
    with open(NEURAL_JSON, 'r') as f:
        neural_data = json.load(f)
    
    # Convert classic chunks to DataFrame
    classic_df = pd.DataFrame(classic_data['chunks'])
        
    # Setup Figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.set_style("whitegrid")
    
    # ---------------------------------------------------------
    # Plot 1: Classic Stylometry (PCA)
    # ---------------------------------------------------------
    print("Plotting Classic PCA...")
    ax1 = axes[0]
    
    palette = {
        'anchor': '#1f77b4',        # Blue
        'colossians': '#ff7f0e',    # Orange
        '2thessalonians': '#9467bd',# Purple
        'ephesians': '#8c564b',     # Brown
        '1timothy': '#2ca02c',      # Green
        '2timothy': '#17becf',      # Cyan
        'titus': '#bcbd22',         # Yellow-green
        'hebrews': '#d62728'        # Red
    }
    
    sns.scatterplot(
        data=classic_df,
        x='PC1', y='PC2',
        hue='group',
        palette=palette,
        alpha=0.6,
        s=80,
        edgecolor='w',
        ax=ax1
    )
    
    # Add variance explained to title
    var_exp = classic_data.get('variance_explained', {})
    pc1_var = var_exp.get('PC1', 0)
    pc2_var = var_exp.get('PC2', 0)
    
    ax1.set_title('Classic Stylometry (MFW 100)', fontsize=14, fontweight='bold')
    ax1.set_xlabel(f'Principal Component 1 ({pc1_var}%)')
    ax1.set_ylabel(f'Principal Component 2 ({pc2_var}%)')
    
    # ---------------------------------------------------------
    # Plot 2: Neural Probe (Distances)
    # ---------------------------------------------------------
    print("Plotting Neural Distances...")
    ax2 = axes[1]
    
    results = neural_data.get('distance_results', [])
    
    texts = ['Paul (ref)'] + [r['text'] for r in results]
    means = [0.0] + [float(r['mean_sigma']) for r in results]
    
    # Calculate Error bars
    errors = [[0,0]]
    for r in results:
        ci_lower, ci_upper = r['ci_95']
        mean = float(r['mean_sigma'])
        err_low = mean - ci_lower
        err_high = ci_upper - mean
        errors.append([err_low, err_high])
        
    y_pos = np.arange(len(texts))
    errors = np.array(errors).T
    
    # Colors based on interpretation
    bar_colors = ['#1f77b4']  # Paul = blue
    for r in results:
        interp = r.get('interpretation', '')
        if 'Significant' in interp:
            bar_colors.append('#d62728')  # Red
        elif 'Moderate' in interp or 'Marginal' in interp:
            bar_colors.append('#ff7f0e')  # Orange
        else:
            bar_colors.append('#2ca02c')  # Green
        
    ax2.bar(y_pos, means, yerr=errors, align='center', alpha=0.8, color=bar_colors, capsize=10)
    ax2.set_xticks(y_pos)
    ax2.set_xticklabels(texts, rotation=45, ha='right')
    ax2.set_ylabel('Distance from Pauline Center (sigma units)')
    ax2.set_title('Neural Probe (Cosine + Mean Pooling)', fontsize=14, fontweight='bold')
    
    ax2.axhline(y=0, color='gray', linestyle='--', label='Pauline Center')
    ax2.legend()
    
    # Add gradient info
    gradient = neural_data.get('gradient', {})
    rho = gradient.get('spearman_rho', 0)
    p_val = gradient.get('p_value', 1)
    n_texts = gradient.get('n_texts', 0)
    
    gradient_text = f"Gradient (n={n_texts}): rho = {rho:.2f}, p = {p_val:.3f}"
    if rho > 0.7 and p_val < 0.05:
        gradient_text += " (CONFIRMED)"
    ax2.text(0.95, 0.95, gradient_text, transform=ax2.transAxes, 
             ha='right', va='top', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = RESULTS_DIR / 'double_dissociation.png'
    plt.savefig(output_path, dpi=300)
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    main()
