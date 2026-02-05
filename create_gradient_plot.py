"""
Neural Scribe: Gradient Scatter Plot

Creates a scatter plot showing the correlation between 
scholarly rejection rate and neural semantic distance.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from scipy import stats

RESULTS_DIR = Path('results')

def main():
    print("Generating Gradient Scatter Plot...")
    
    # Load results
    with open(RESULTS_DIR / 'neural_results.json', 'r') as f:
        data = json.load(f)
    
    # Rejection rates from scholarly consensus
    REJECTION_RATES = {
        'colossians': 0.40,
        '2thessalonians': 0.50,
        'ephesians': 0.60,
        '1timothy': 0.80,
        '2timothy': 0.80,
        'titus': 0.80,
        'hebrews': 1.00,
    }
    
    # Extract data points
    texts = []
    rates = []
    distances = []
    
    for r in data['distance_results']:
        text_lower = r['text'].lower()
        if text_lower in REJECTION_RATES:
            texts.append(r['text'])
            rates.append(REJECTION_RATES[text_lower])
            distances.append(float(r['mean_sigma']))
    
    # Calculate correlation
    rho, p_val = stats.spearmanr(rates, distances)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Scatter points with colors based on significance
    colors = []
    for r in data['distance_results']:
        text_lower = r['text'].lower()
        if text_lower not in REJECTION_RATES:
            continue
        if r['p_value'] < 0.05:
            colors.append('#d62728')  # Red for significant
        elif r['p_value'] < 0.10:
            colors.append('#ff7f0e')  # Orange for marginal
        else:
            colors.append('#1f77b4')  # Blue for indistinguishable
    
    ax.scatter(rates, distances, c=colors, s=150, alpha=0.8, edgecolors='white', linewidths=2)
    
    # Add text labels
    for i, txt in enumerate(texts):
        offset_y = 0.08 if distances[i] >= 0 else -0.12
        ax.annotate(txt, (rates[i], distances[i] + offset_y), 
                    ha='center', fontsize=10, fontweight='bold')
    
    # Add trend line
    z = np.polyfit(rates, distances, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0.35, 1.05, 100)
    ax.plot(x_line, p(x_line), '--', color='gray', alpha=0.7, linewidth=2)
    
    # Reference lines
    ax.axhline(y=0, color='green', linestyle='-', alpha=0.3, linewidth=2, label='Pauline Center')
    
    # Labels and title
    ax.set_xlabel('Scholarly Rejection Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Semantic Distance from Pauline Center (sigma)', fontsize=12, fontweight='bold')
    ax.set_title(f'Authorship Gradient Analysis\nSpearman rho = {rho:.3f}, p = {p_val:.4f}', 
                 fontsize=14, fontweight='bold')
    
    # Legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', label='Significant (p < 0.05)'),
        Patch(facecolor='#ff7f0e', label='Marginal (p < 0.10)'),
        Patch(facecolor='#1f77b4', label='Indistinguishable'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    ax.set_xlim(0.3, 1.1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = RESULTS_DIR / 'gradient_scatter.png'
    plt.savefig(output_path, dpi=300)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
