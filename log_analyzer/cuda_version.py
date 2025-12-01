import numpy as np
from numba import cuda
from utils import parse_log_line
import time


def prepare_lines(lines, words, max_line_len=256, max_word_len=32):
    arr = np.zeros((len(lines), max_line_len), dtype=np.uint8)
    line_lengths = np.zeros(len(lines), dtype=np.int32)
    for i, line in enumerate(lines):
        b = line.encode('utf-8')[:max_line_len]
        arr[i, :len(b)] = np.frombuffer(b, dtype=np.uint8)
        line_lengths[i] = len(b)
    word_arr = np.zeros((len(words), max_word_len), dtype=np.uint8)
    word_lengths = np.zeros(len(words), dtype=np.int32)
    for i, w in enumerate(words):
        b = w.encode('utf-8')[:max_word_len]
        word_arr[i, :len(b)] = np.frombuffer(b, dtype=np.uint8)
        word_lengths[i] = len(b)
    return arr, line_lengths, word_arr, word_lengths


@cuda.jit
def count_words_kernel(lines, line_lengths, words, word_lengths, counts):
    i = cuda.grid(1)
    if i < lines.shape[0]:
        for j in range(words.shape[0]):
            found = False
            for k in range(line_lengths[i] - word_lengths[j] + 1):
                match = True
                for l in range(word_lengths[j]):
                    if lines[i, k + l] != words[j, l]:
                        match = False
                        break
                if match:
                    found = True
                    break
            if found:
                cuda.atomic.add(counts, j, 1)


def analyze_log_cuda(log_path, words):
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    arr, line_lengths, word_arr, word_lengths = prepare_lines(lines, words)
    d_lines = cuda.to_device(arr)
    d_line_lengths = cuda.to_device(line_lengths)
    d_words = cuda.to_device(word_arr)
    d_word_lengths = cuda.to_device(word_lengths)
    counts = np.zeros(len(words), dtype=np.int32)
    d_counts = cuda.to_device(counts)
    threadsperblock = 128
    blockspergrid = (arr.shape[0] + (threadsperblock - 1)) // threadsperblock
    start = time.time()
    count_words_kernel[blockspergrid, threadsperblock](d_lines, d_line_lengths, d_words, d_word_lengths, d_counts)
    cuda.synchronize()
    elapsed = time.time() - start
    result = d_counts.copy_to_host()
    return dict(zip(words, result)), elapsed
