"""
Neural Scribe: Classic Stylometry Baseline

Computes Burrows' Delta using Top 100 Most Frequent Words,
with Z-score normalization and PCA for visualization.
"""

import json
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from data_loader import load_corpus

RESULTS_DIR = Path('results')


def get_mfw_table(corpus, n_mfw=100):
    """
    Generate frequency table for top N Most Frequent Words.
    Returns: matrix (list of lists), labels (list of dicts)
    """
    all_text = []
    doc_map = []
    
    for group, windows in corpus.items():
        for i, win in enumerate(windows):
            all_text.append(win)
            doc_map.append({'group': group, 'label': f"{group}_{i}"})
            
    # Tokenize and count
    tokenized_docs = [doc.split() for doc in all_text]
    all_tokens = [token for doc in tokenized_docs for token in doc]
    
    # Identify MFW
    freqs = Counter(all_tokens)
    mfw = [w for w, _ in freqs.most_common(n_mfw)]
    
    # Build Matrix
    matrix = []
    for doc in tokenized_docs:
        doc_len = len(doc)
        if doc_len == 0:
            row = [0.0] * n_mfw
        else:
            doc_counts = Counter(doc)
            row = [doc_counts[w] / doc_len for w in mfw]
        matrix.append(row)
        
    return matrix, doc_map, mfw


def main():
    print("Neural Scribe: Classic Baseline Analysis")
    print("=" * 50)
    
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Load corpus
    corpus = load_corpus(window_size=150, stride=75)
    
    # MFW Analysis
    print("Calculating MFW frequencies (Top 100)...")
    matrix, doc_map, mfw_list = get_mfw_table(corpus, n_mfw=100)
    
    # Z-Score Normalization (Burrows' Delta standard)
    print("Applying Z-score normalization...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(matrix)
    
    # PCA for visualization
    print("Performing PCA...")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled_data)
    
    # Build results
    results = {
        'methodology': "Burrows' Delta (100 MFW, Z-score, PCA)",
        'n_mfw': 100,
        'variance_explained': {
            'PC1': round(pca.explained_variance_ratio_[0] * 100, 2),
            'PC2': round(pca.explained_variance_ratio_[1] * 100, 2)
        },
        'chunks': []
    }
    
    for i, doc in enumerate(doc_map):
        results['chunks'].append({
            'group': doc['group'],
            'label': doc['label'],
            'PC1': round(float(coords[i, 0]), 4),
            'PC2': round(float(coords[i, 1]), 4)
        })
    
    output_path = RESULTS_DIR / 'classic_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
