import re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np

def parse_log_line(line):
    match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\w+) (.*)", line)
    if match:
        return {
            'datetime': match.group(1),
            'level': match.group(2),
            'message': match.group(3)
        }
    return None

def filter_lines(lines, level=None, date_range=None):
    result = []
    for line in lines:
        parsed = parse_log_line(line)
        if not parsed:
            continue
        if level and parsed['level'] != level:
            continue
        if date_range:
            if not (date_range[0] <= parsed['datetime'] <= date_range[1]):
                continue
        result.append(line)
    return result

def count_words(lines, words):
    counter = Counter()
    for line in lines:
        for word in words:
            if word in line:
                counter[word] += 1
    return counter

def stats_per_hour(lines, word):
    stats = defaultdict(int)
    for line in lines:
        parsed = parse_log_line(line)
        if parsed and word in line:
            hour = parsed['datetime'][:13]  # YYYY-MM-DD HH
            stats[hour] += 1
    return stats

def plot_top_words(counter, title="Top words", save_path=None):
    words, counts = zip(*counter.most_common())
    plt.figure(figsize=(8,4))
    plt.bar(words, counts)
    plt.title(title)
    plt.ylabel('Count')
    plt.xlabel('Word')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_stats_per_hour(stats, title="Events per hour", save_path=None):
    hours = sorted(stats.keys())
    counts = [stats[h] for h in hours]
    plt.figure(figsize=(10,4))
    plt.plot(hours, counts, marker='o')
    plt.title(title)
    plt.ylabel('Count')
    plt.xlabel('Hour')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
