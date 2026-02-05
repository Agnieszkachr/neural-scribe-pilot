"""
Neural Scribe: Neural Embedding Probe

One-class classification using Ancient Greek BERT embeddings.
Uses mean pooling + cosine distance + chunk-level analysis.
"""

import numpy as np
from scipy.spatial.distance import cosine
from scipy import stats
from transformers import AutoTokenizer, AutoModel
import torch
import json
from pathlib import Path
from sklearn.decomposition import PCA

from data_loader import load_corpus

BERT_MODEL = 'pranaydeeps/Ancient-Greek-BERT'
RESULTS_DIR = Path('results')


def extract_embeddings_mean_pooling(texts, tokenizer, model):
    """Extract mean-pooled embeddings (better for semantic similarity)."""
    embeddings = []
    if not texts:
        return np.array([])
    
    model.eval()
    
    with torch.no_grad():
        for i, text in enumerate(texts):
            inputs = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )
            outputs = model(**inputs)
            
            # Mean pooling with attention mask
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_embedding = (sum_embeddings / sum_mask).squeeze().numpy()
            
            embeddings.append(mean_embedding)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(texts)} chunks...", end='\r')
    
    print("")
    return np.array(embeddings)


def compute_baseline_cosine(anchor_embeddings):
    """Compute Pauline baseline using cosine distance."""
    centroid = np.mean(anchor_embeddings, axis=0)
    
    anchor_distances = np.array([
        cosine(emb, centroid) for emb in anchor_embeddings
    ])
    
    baseline_mean = np.mean(anchor_distances)
    baseline_std = np.std(anchor_distances)
    
    # Normalize to sigma units
    anchor_dist_normalized = (anchor_distances - baseline_mean) / baseline_std
    
    return centroid, baseline_mean, baseline_std, anchor_dist_normalized


def compute_distances_cosine(embeddings, centroid, baseline_mean, baseline_std):
    """Compute normalized cosine distances in sigma units."""
    if len(embeddings) == 0:
        return np.array([])
    
    raw_distances = np.array([
        cosine(emb, centroid) for emb in embeddings
    ])
    
    # Normalize to sigma units (how many SDs from Paul's mean)
    normalized = (raw_distances - baseline_mean) / baseline_std
    
    return normalized


def chunk_distribution_analysis(anchor_dist, target_dist, name):
    """Analyze what proportion of chunks fall outside Pauline distribution."""
    
    # Percentiles of anchor distribution
    p75 = np.percentile(anchor_dist, 75)
    p90 = np.percentile(anchor_dist, 90)
    p95 = np.percentile(anchor_dist, 95)
    
    # What proportion of target exceeds these?
    outside_75 = np.mean(target_dist > p75) * 100
    outside_90 = np.mean(target_dist > p90) * 100
    outside_95 = np.mean(target_dist > p95) * 100
    
    return {
        'text': name,
        'pct_outside_p75': round(outside_75, 1),
        'pct_outside_p90': round(outside_90, 1),
        'pct_outside_p95': round(outside_95, 1),
    }


def statistical_analysis(anchor_dist, target_dist, name):
    """Statistical comparison with proper sigma interpretation."""
    if len(target_dist) == 0:
        return {'text': name, 'error': 'No data'}
    
    mean_d = np.mean(target_dist)
    std_d = np.std(target_dist)
    n = len(target_dist)
    se = stats.sem(target_dist)
    ci = stats.t.interval(0.95, n-1, loc=mean_d, scale=se)
    
    # One-sample t-test: is target mean significantly different from 0 (Paul's center)?
    t_stat, p_val = stats.ttest_1samp(target_dist, 0)
    
    # Effect size: how far is target mean from Paul's center in pooled SD units
    pooled_std = np.sqrt((np.var(anchor_dist, ddof=1) + np.var(target_dist, ddof=1)) / 2)
    cohens_d = mean_d / pooled_std
    
    # Interpretation
    if p_val < 0.001 and cohens_d > 0.8:
        interpretation = "Significant divergence"
    elif p_val < 0.05 and cohens_d > 0.5:
        interpretation = "Moderate divergence"
    elif p_val < 0.05:
        interpretation = "Marginal divergence"
    else:
        interpretation = "Indistinguishable"
    
    return {
        'text': name,
        'n_chunks': n,
        'mean_sigma': round(mean_d, 2),
        'std_sigma': round(std_d, 2),
        'ci_95': (round(ci[0], 2), round(ci[1], 2)),
        't_statistic': round(t_stat, 2),
        'p_value': p_val,
        'cohens_d': round(cohens_d, 2),
        'interpretation': interpretation
    }


def main():
    """Run neural probe analysis with revised methodology."""
    print("Neural Scribe: One-Class Classification (Cosine + Mean Pooling)")
    print("=" * 60)
    
    RESULTS_DIR.mkdir(exist_ok=True)
    
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
    
    # Load corpus
    print("\nLoading corpus...")
    corpus = load_corpus()
    
    # Load model
    print("Loading Ancient Greek BERT...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = AutoModel.from_pretrained(BERT_MODEL)
    
    # Extract embeddings with mean pooling
    print("\nExtracting embeddings (mean pooling)...")
    
    # Anchor embeddings
    print("  Anchor (Paul)...")
    anchor_emb = extract_embeddings_mean_pooling(corpus['anchor'], tokenizer, model)
    
    # Extract embeddings for all disputed texts
    embeddings = {}
    for group in REJECTION_RATES.keys():
        if group in corpus and len(corpus[group]) > 0:
            print(f"  {group.capitalize()}...")
            embeddings[group] = extract_embeddings_mean_pooling(corpus[group], tokenizer, model)
    
    print(f"\nEmbedding dimension: {anchor_emb.shape[1]}")
    
    # Compute baseline from anchor ONLY
    print("\nComputing Pauline baseline (cosine distance)...")
    centroid, baseline_mean, baseline_std, anchor_dist = compute_baseline_cosine(anchor_emb)
    
    print(f"  Anchor chunks: {len(anchor_emb)}")
    print(f"  Baseline mean distance: {baseline_mean:.6f}")
    print(f"  Baseline std: {baseline_std:.6f}")
    
    # Compute distances for all texts
    print("\nComputing distances...")
    distances = {}
    for group, emb in embeddings.items():
        distances[group] = compute_distances_cosine(emb, centroid, baseline_mean, baseline_std)
    
    # Statistical analysis
    results = []
    chunk_results = []
    for group in REJECTION_RATES.keys():
        if group in distances:
            results.append(statistical_analysis(anchor_dist, distances[group], group.capitalize()))
            chunk_results.append(chunk_distribution_analysis(anchor_dist, distances[group], group.capitalize()))
    
    # Print results
    print("\n" + "=" * 90)
    print("DISTANCE ANALYSIS (in sigma units from Pauline center)")
    print("=" * 90)
    print(f"{'Text':<18} {'Mean sigma':<12} {'95% CI':<20} {'p-value':<12} {'Cohen d':<10} {'Status':<20}")
    print("-" * 90)
    print(f"{'Paul (ref)':<18} {'0.00':<12} {'-':<20} {'-':<12} {'-':<10} {'Baseline':<20}")
    
    for r in results:
        if 'error' in r:
            print(f"{r['text']:<18} ERROR")
            continue
        ci_str = f"[{r['ci_95'][0]}, {r['ci_95'][1]}]"
        p_str = f"{r['p_value']:.4f}" if r['p_value'] >= 0.0001 else "<0.0001"
        print(f"{r['text']:<18} {r['mean_sigma']:<12} {ci_str:<20} {p_str:<12} {r['cohens_d']:<10} {r['interpretation']:<20}")
    
    print("\n" + "=" * 90)
    print("CHUNK DISTRIBUTION ANALYSIS")
    print("=" * 90)
    print(f"{'Text':<18} {'% > P75':<12} {'% > P90':<12} {'% > P95':<12}")
    print("-" * 90)
    print(f"{'Paul (ref)':<18} {'25.0%':<12} {'10.0%':<12} {'5.0%':<12}")
    
    for c in chunk_results:
        print(f"{c['text']:<18} {c['pct_outside_p75']:<12} {c['pct_outside_p90']:<12} {c['pct_outside_p95']:<12}")
    
    # Gradient test (n=7 texts now)
    print("\n" + "=" * 90)
    
    # Build gradient data: match rejection rates to distances
    grad_rates = []
    grad_dists = []
    for r in results:
        if 'mean_sigma' in r:
            text_lower = r['text'].lower()
            if text_lower in REJECTION_RATES:
                grad_rates.append(REJECTION_RATES[text_lower])
                grad_dists.append(float(r['mean_sigma']))
    
    rho, p_grad = 0.0, 1.0
    n_texts = len(grad_rates)
    if n_texts >= 3:
        rho, p_grad = stats.spearmanr(grad_rates, grad_dists)
        print(f"GRADIENT TEST (n={n_texts} texts): Spearman rho = {rho:.3f}, p = {p_grad:.4f}")
        
        if rho > 0.7 and p_grad < 0.05:
            print("[+] GRADIENT CONFIRMED: Distance correlates with scholarly consensus")
        elif rho > 0.5:
            print("[~] Moderate positive trend detected")
        elif rho > 0:
            print("[~] Weak positive trend, not statistically significant")
        else:
            print("[-] No gradient detected")
    
    # Save results
    output = {
        'methodology': 'One-Class Cosine Distance with Mean Pooling',
        'anchor_chunks': len(anchor_emb),
        'baseline_mean': float(baseline_mean),
        'baseline_std': float(baseline_std),
        'distance_results': results,
        'chunk_distribution': chunk_results,
        'gradient': {
            'n_texts': n_texts,
            'spearman_rho': round(rho, 3),
            'p_value': round(p_grad, 4)
        }
    }
    
    with open(RESULTS_DIR / 'neural_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    # Save embeddings for visualization
    emb_data = {'anchor': anchor_emb, 'anchor_dist': anchor_dist}
    for group, emb in embeddings.items():
        emb_data[group] = emb
        emb_data[f'{group}_dist'] = distances[group]
    
    np.savez(RESULTS_DIR / 'embeddings.npz', **emb_data)
    
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
