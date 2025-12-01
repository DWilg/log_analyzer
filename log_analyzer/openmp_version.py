from joblib import Parallel, delayed
from collections import Counter
from utils import count_words, filter_lines, stats_per_hour
import time

def process_chunk(chunk, words):
    return count_words(chunk, words)

def analyze_log_openmp(log_path, words, level=None, date_range=None, n_jobs=-1):
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if level or date_range:
        lines = filter_lines(lines, level, date_range)
    chunk_size = max(1, len(lines) // (n_jobs if n_jobs > 0 else 4))
    chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]
    start = time.time()
    results = Parallel(n_jobs=n_jobs)(delayed(process_chunk)(chunk, words) for chunk in chunks)
    total = Counter()
    for r in results:
        total.update(r)
    elapsed = time.time() - start
    return total, elapsed

def analyze_stats_per_hour_openmp(log_path, word, n_jobs=-1):
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    chunk_size = max(1, len(lines) // (n_jobs if n_jobs > 0 else 4))
    chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]
    results = Parallel(n_jobs=n_jobs)(delayed(stats_per_hour)(chunk, word) for chunk in chunks)
    from collections import defaultdict
    stats = defaultdict(int)
    for d in results:
        for k, v in d.items():
            stats[k] += v
    return stats
