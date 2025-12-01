import argparse
from utils import plot_top_words, plot_stats_per_hour
from openmp_version import analyze_log_openmp, analyze_stats_per_hour_openmp
from mpi_version import analyze_log_mpi, analyze_stats_per_hour_mpi
from cuda_version import analyze_log_cuda
import sys

def main():
    parser = argparse.ArgumentParser(description="Log Analyzer - OpenMP/MPI/CUDA")
    parser.add_argument('--mode', choices=['openmp', 'mpi', 'cuda'], required=True)
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

    if args.mode == 'openmp':
        print("[OpenMP] Counting words...")
        counter, elapsed = analyze_log_openmp(args.log, args.words, args.level, date_range)
        print(f"Results: {dict(counter)}")
        print(f"Elapsed: {elapsed:.4f}s")
        if args.plot:
            plot_top_words(counter, title="Top words (OpenMP)")
    elif args.mode == 'mpi':
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            print("[MPI] Counting words...")
        counter, elapsed = analyze_log_mpi(args.log, args.words, args.level, date_range)
        if rank == 0:
            print(f"Results: {dict(counter)}")
            print(f"Elapsed: {elapsed:.4f}s")
            if args.plot:
                plot_top_words(counter, title="Top words (MPI)")
    elif args.mode == 'cuda':
        print("[CUDA] Counting words...")
        counter, elapsed = analyze_log_cuda(args.log, args.words)
        print(f"Results: {dict(counter)}")
        print(f"Elapsed: {elapsed:.4f}s")
        if args.plot:
            from collections import Counter
            plot_top_words(Counter(counter), title="Top words (CUDA)")
    else:
        print("Unknown mode.")
        sys.exit(1)

if __name__ == "__main__":
    main()
