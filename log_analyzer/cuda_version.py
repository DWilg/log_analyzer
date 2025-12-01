import numpy as np
from numba import cuda
from utils import parse_log_line
import time

def prepare_lines(lines, words):
    arr = np.array([line.encode('utf-8') for line in lines])
    word_arr = np.array([w.encode('utf-8') for w in words])
    return arr, word_arr

@cuda.jit
def count_words_kernel(lines, words, counts):
    i = cuda.grid(1)
    if i < lines.shape[0]:
        for j in range(words.shape[0]):
            if words[j] in lines[i]:
                cuda.atomic.add(counts, j, 1)

def analyze_log_cuda(log_path, words):
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    arr, word_arr = prepare_lines(lines, words)
    d_lines = cuda.to_device(arr)
    d_words = cuda.to_device(word_arr)
    counts = np.zeros(len(words), dtype=np.int32)
    d_counts = cuda.to_device(counts)
    threadsperblock = 128
    blockspergrid = (arr.shape[0] + (threadsperblock - 1)) // threadsperblock
    start = time.time()
    count_words_kernel[blockspergrid, threadsperblock](d_lines, d_words, d_counts)
    cuda.synchronize()
    elapsed = time.time() - start
    result = d_counts.copy_to_host()
    return dict(zip(words, result)), elapsed
