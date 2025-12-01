import argparse
from utils import plot_top_words, plot_stats_per_hour
from openmp_version import analyze_log_openmp, analyze_stats_per_hour_openmp
from mpi_version import analyze_log_mpi, analyze_stats_per_hour_mpi
from cuda_version import analyze_log_cuda
import sys

def main():
    parser = argparse.ArgumentParser(description="Log Analyzer - OpenMP/MPI/CUDA")
    parser.add_argument('--mode', choices=['openmp', 'mpi', 'cuda'], required=True)

