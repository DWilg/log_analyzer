import argparse
from utils import plot_top_words, plot_stats_per_hour
from openmp_version import analyze_log_openmp, analyze_stats_per_hour_openmp
from mpi_version import analyze_log_mpi, analyze_stats_per_hour_mpi
from cuda_version import analyze_log_cuda
import sys

def main():
    parser = argparse.ArgumentParser(description="Log Analyzer - OpenMP/MPI/CUDA/ALL")
    parser.add_argument('--mode', choices=['openmp', 'mpi', 'cuda', 'all'], required=True)
    parser.add_argument('--log', required=True)
    parser.add_argument('--words', nargs='+', required=True)
    parser.add_argument('--level', help='Log level to filter (e.g. ERROR)')
    parser.add_argument('--date_from', help='Start date (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--date_to', help='End date (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    date_range = None
    if args.date_from and args.date_to:
        date_range = (args.date_from, args.date_to)

    results = {}
    if args.mode == 'openmp' or args.mode == 'all':
        print("[OpenMP] Counting words...")
        counter, elapsed = analyze_log_openmp(args.log, args.words, args.level, date_range)
        print(f"[OpenMP] Results: {dict(counter)}")
        print(f"[OpenMP] Elapsed: {elapsed:.4f}s")
        from collections import Counter
        plot_top_words(Counter(counter), title="Top words (OpenMP)", save_path="top_words_openmp.png")
        results['OpenMP'] = (elapsed, counter)
    if args.mode == 'mpi' or args.mode == 'all':
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            print("[MPI] Counting words...")
        counter, elapsed = analyze_log_mpi(args.log, args.words, args.level, date_range)
        if rank == 0:
            print(f"[MPI] Results: {dict(counter)}")
            print(f"[MPI] Elapsed: {elapsed:.4f}s")
            from collections import Counter
            plot_top_words(Counter(counter), title="Top words (MPI)", save_path="top_words_mpi.png")
            results['MPI'] = (elapsed, counter)
    if args.mode == 'cuda' or args.mode == 'all':
        print("[CUDA] Counting words...")
        counter, elapsed = analyze_log_cuda(args.log, args.words)
        print(f"[CUDA] Results: {dict(counter)}")
        print(f"[CUDA] Elapsed: {elapsed:.4f}s")
        from collections import Counter
        plot_top_words(Counter(counter), title="Top words (CUDA)", save_path="top_words_cuda.png")
        results['CUDA'] = (elapsed, counter)

    if args.mode == 'all':
        import os
        from utils import plot_throughput, plot_cpu_gpu_comparison
        import matplotlib.pyplot as plt
        techs = []
        times = []
        for tech in ['OpenMP', 'MPI', 'CUDA']:
            if tech in results:
                techs.append(tech)
                times.append(results[tech][0])
        if times:
            plt.figure(figsize=(8,4))
            plt.bar(techs, times)
            plt.title('Porównanie czasu wykonania')
            plt.ylabel('Czas [s]')
            plt.xlabel('Technologia')
            plt.savefig('execution_time_comparison.png')
            print("[ALL] Wygenerowano wykres porównawczy: execution_time_comparison.png")

            file_size = os.path.getsize(args.log)
            tech_times = {tech: results[tech][0] for tech in techs}
            plot_throughput(tech_times, file_size, title="Przepustowość GB/s", save_path="throughput_comparison.png")
            print("[ALL] Wygenerowano wykres przepustowości: throughput_comparison.png")

            if 'OpenMP' in results and 'CUDA' in results:
                plot_cpu_gpu_comparison(results['OpenMP'][0], results['CUDA'][0], title="Porównanie CPU vs GPU", save_path="cpu_vs_gpu.png")
                print("[ALL] Wygenerowano wykres CPU vs GPU: cpu_vs_gpu.png")

if __name__ == "__main__":
    main()