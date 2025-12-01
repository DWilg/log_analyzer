from mpi4py import MPI
from collections import Counter, defaultdict
from utils import count_words, filter_lines, stats_per_hour
import time

def analyze_log_mpi(log_path, words, level=None, date_range=None):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    lines = None
    if rank == 0:
        with open(log_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        if level or date_range:
            all_lines = filter_lines(all_lines, level, date_range)
        chunk_size = (len(all_lines) + size - 1) // size
        chunks = [all_lines[i*chunk_size:(i+1)*chunk_size] for i in range(size)]
    else:
        chunks = None
    chunk = comm.scatter(chunks, root=0)
    start = time.time()
    local_counter = count_words(chunk, words)
    local_elapsed = time.time() - start
    total_counter = comm.reduce(local_counter, op=MPI.SUM, root=0)
    max_elapsed = comm.reduce(local_elapsed, op=MPI.MAX, root=0)
    return total_counter, max_elapsed if rank == 0 else (None, None)

def analyze_stats_per_hour_mpi(log_path, word):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    lines = None
    if rank == 0:
        with open(log_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        chunk_size = (len(all_lines) + size - 1) // size
        chunks = [all_lines[i*chunk_size:(i+1)*chunk_size] for i in range(size)]
    else:
        chunks = None
    chunk = comm.scatter(chunks, root=0)
    local_stats = stats_per_hour(chunk, word)
    all_stats = comm.gather(local_stats, root=0)
    if rank == 0:
        stats = defaultdict(int)
        for d in all_stats:
            for k, v in d.items():
                stats[k] += v
        return stats
    return None
