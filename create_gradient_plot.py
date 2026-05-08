"""
Neural Scribe: Gradient Scatter Plot (v2.0)

Creates a dual-model scatter plot showing the correlation between
scholarly rejection rate and neural semantic distance for both
Koine-BERT (primary) and Ancient-Greek-BERT (robustness check).
"""

import sys, os
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from scipy import stats

RESULTS_DIR = Path('results')

def main():
    print("Generating Dual-Model Gradient Scatter Plot...")

    koine_path = RESULTS_DIR / 'neural_results_koine.json'
    ancient_path = RESULTS_DIR / 'neural_results_ancient.json'

    if not koine_path.exists():
        print(f"Error: {koine_path} not found. Run: python run_neural_probe.py --model both")
        return

    REJECTION_RATES = {
        'colossians': 0.40, '2thessalonians': 0.50, 'ephesians': 0.60,
        '1timothy': 0.80, '2timothy': 0.80, 'titus': 0.80, 'hebrews': 1.00,
    }

    def extract_points(data):
        texts, rates, distances = [], [], []
        for r in data['distance_results']:
            text_lower = r['text'].lower()
            if text_lower in REJECTION_RATES:
                texts.append(r['text'])
                rates.append(REJECTION_RATES[text_lower])
                distances.append(float(r['mean_sigma']))
        return texts, rates, distances

    with open(koine_path, 'r') as f:
        koine_data = json.load(f)
    k_texts, k_rates, k_dists = extract_points(koine_data)
    k_rho, k_p = stats.spearmanr(k_rates, k_dists)

    has_ancient = ancient_path.exists()
    if has_ancient:
        with open(ancient_path, 'r') as f:
            ancient_data = json.load(f)
        a_texts, a_rates, a_dists = extract_points(ancient_data)
        a_rho, a_p = stats.spearmanr(a_rates, a_dists)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Koine-BERT (primary, filled)
    ax.scatter(k_rates, k_dists, c='#2b5797', s=160, alpha=0.85,
               edgecolors='white', linewidths=2, zorder=3,
               label=f'Koine-BERT (rho={k_rho:.3f}, p={k_p:.3f})')

    for i, txt in enumerate(k_texts):
        offset_y = 0.08 if k_dists[i] >= 0 else -0.12
        ax.annotate(txt, (k_rates[i], k_dists[i] + offset_y),
                    ha='center', fontsize=10, fontweight='bold', color='#2b5797')

    # Koine trend line
    z_k = np.polyfit(k_rates, k_dists, 1)
    p_k = np.poly1d(z_k)
    x_line = np.linspace(0.35, 1.05, 100)
    ax.plot(x_line, p_k(x_line), '-', color='#2b5797', alpha=0.5, linewidth=2)

    if has_ancient:
        # Ancient-Greek-BERT (secondary, open markers)
        ax.scatter(a_rates, a_dists, facecolors='none', edgecolors='#b8860b',
                   s=120, alpha=0.7, linewidths=2, zorder=2,
                   label=f'Ancient-Greek-BERT (rho={a_rho:.3f}, p={a_p:.3f})')

        z_a = np.polyfit(a_rates, a_dists, 1)
        p_a = np.poly1d(z_a)
        ax.plot(x_line, p_a(x_line), '--', color='#b8860b', alpha=0.5, linewidth=2)

    ax.axhline(y=0, color='green', linestyle='-', alpha=0.3, linewidth=2)

    ax.set_xlabel('Scholarly Rejection Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Semantic Distance from Pauline Center (sigma)', fontsize=12, fontweight='bold')
    ax.set_title('Authorship Gradient Analysis (Dual-Model)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(0.3, 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = RESULTS_DIR / 'gradient_scatter.png'
    plt.savefig(output_path, dpi=300)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
