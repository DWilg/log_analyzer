import matplotlib.pyplot as plt
from typing import Dict, Iterable

def compute_speedup(times: Dict[int, float]) -> Dict[int, float]:
    if 1 not in times:
        baseline_p = min(times.keys())
    else:
        baseline_p = 1
    t1 = times[baseline_p]
    speedup = {p: t1 / t for p, t in sorted(times.items())}
    return speedup

def compute_efficiency(speedup: Dict[int, float]) -> Dict[int, float]:
    return {p: s / p for p, s in speedup.items()}

def plot_speedup_and_efficiency(times: Dict[int, float], title_prefix: str = "Scalaing", save_path: str = None):
    speedup = compute_speedup(times)
    efficiency = compute_efficiency(speedup)
    ps = list(speedup.keys())
    Ss = [speedup[p] for p in ps]
    Es = [efficiency[p] for p in ps]

    fig, ax1 = plt.subplots(figsize=(8,4))
    ax1.plot(ps, Ss, marker='o', label='Przyspieszenie (S)')
    ax1.set_xlabel('Liczba wątków/procesów')
    ax1.set_ylabel('Przyspieszenie [S]')
    ax1.set_xticks(ps)
    ax1.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(ps, Es, marker='s', color='orange', label='Efektywność (E)')
    ax2.set_ylabel('Efektywność [E]')
    ax2.set_ylim(0, max(1.0, max(Es) * 1.1))

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    plt.title(f"{title_prefix}: Przyspieszenie i Efektywność")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def make_stats_per_hour_plot(lines: Iterable[str], word: str, save_path: str = None):
    from utils import stats_per_hour
    stats = stats_per_hour(lines, word)
    hours = sorted(stats.keys())
    counts = [stats[h] for h in hours]
    plt.figure(figsize=(10,4))
    plt.plot(hours, counts, marker='o')
    plt.title(f'Liczba zdarzeń "{word}" na godzinę')
    plt.xlabel('Godzina')
    plt.ylabel('Liczba zdarzeń')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

