"""
Neural Scribe: Corpus Loader

Fetches and segments the SBLGNT Greek New Testament corpus
for authorship attribution analysis.
"""

import requests
import re
from pathlib import Path

# Configuration
SBLGNT_BASE_URL = "https://raw.githubusercontent.com/morphgnt/sblgnt/master"
DATA_DIR = Path('data')

# Corpus group definitions
# Anchor: Undisputed Pauline letters
# Target: Disputed letters (varying rejection rates)
# Control: Non-Pauline (Hebrews)
CORPUS_GROUPS = {
    'anchor': [
        '66-Ro-morphgnt.txt',   # Romans
        '67-1Co-morphgnt.txt',  # 1 Corinthians
        '68-2Co-morphgnt.txt',  # 2 Corinthians
        '69-Ga-morphgnt.txt',   # Galatians
        '71-Php-morphgnt.txt',  # Philippians
        '73-1Th-morphgnt.txt',  # 1 Thessalonians
        '78-Phm-morphgnt.txt',  # Philemon
    ],
    'colossians': ['72-Col-morphgnt.txt'],
    '2thessalonians': ['74-2Th-morphgnt.txt'],
    'ephesians': ['70-Eph-morphgnt.txt'],
    '1timothy': ['75-1Ti-morphgnt.txt'],
    '2timothy': ['76-2Ti-morphgnt.txt'],
    'titus': ['77-Tit-morphgnt.txt'],
    'hebrews': ['79-Heb-morphgnt.txt'],
}


def download_file(filename):
    """Download file from SBLGNT repository if not cached locally."""
    DATA_DIR.mkdir(exist_ok=True)
    local_path = DATA_DIR / filename
    
    if local_path.exists():
        return local_path
        
    url = f"{SBLGNT_BASE_URL}/{filename}"
    response = requests.get(url)
    response.raise_for_status()
    
    with open(local_path, 'w', encoding='utf-8') as f:
        f.write(response.text)
        
    return local_path


def parse_sblgnt(file_path):
    """Extract surface Greek text from SBLGNT format."""
    text = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            if len(parts) > 3:
                # SBLGNT format: ref pos lemma surface ...
                text.append(parts[3])
    return clean_text(" ".join(text))


def clean_text(text):
    """Normalize Greek text."""
    text = re.sub(r'[.,;·\(\)\[\]«»]', '', text)
    return text.lower()


def segment_text(text, window_size=150, stride=75):
    """Segment text into overlapping windows."""
    words = text.split()
    windows = []
    for i in range(0, len(words), stride):
        chunk = words[i:i + window_size]
        if len(chunk) >= window_size * 0.8:
            windows.append(" ".join(chunk))
    return windows


def load_corpus(window_size=150, stride=75):
    """
    Load, parse, and segment the corpus.
    
    Returns dictionary with keys for each corpus group.
    """
    corpus = {group: [] for group in CORPUS_GROUPS.keys()}
    
    print("Fetching and processing corpus groups...")
    
    for group, filenames in CORPUS_GROUPS.items():
        for filename in filenames:
            try:
                path = download_file(filename)
                full_text = parse_sblgnt(path)
                windows = segment_text(full_text, window_size, stride)
                corpus[group].extend(windows)
                print(f"  Loaded {filename}: {len(windows)} windows")
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
                
    return corpus


if __name__ == "__main__":
    c = load_corpus()
    for k, v in c.items():
        print(f"{k}: {len(v)} windows")
