# Log Analyzer

Projekt: Analiza logów systemowych (zliczanie i filtrowanie)

## Wersje
- OpenMP (Python multiprocessing/joblib)
- MPI (mpi4py)
- CUDA (numba)

## Uruchomienie

1. Zainstaluj wymagane pakiety:
   ```bash
   pip install -r requirements.txt
   ```
2. Przykładowe uruchomienie (wersja OpenMP):
   ```bash
   python main.py --mode openmp --log sample_logs/example.log --words ERROR WARNING INFO
   ```
3. Przykładowe uruchomienie (wersja MPI):
   ```bash
   mpiexec -n 4 python main.py --mode mpi --log sample_logs/example.log --words ERROR WARNING INFO
   ```
4. Przykładowe uruchomienie (wersja CUDA):
   ```bash
   python main.py --mode cuda --log sample_logs/example.log --words ERROR WARNING INFO
   ```

## Parametry
- `--mode` - tryb działania: openmp, mpi, cuda
- `--log` - ścieżka do pliku logów
- `--words` - lista słów/fraz do zliczania
- `--filter` - filtr (np. ERROR, zakres dat)
- `--plot` - generuj wykresy

## Wyniki
- Wykresy top-N słów
- Przepustowość GB/s
- Porównanie CPU/GPU

## Autorzy
- Imię i nazwisko 1
- Imię i nazwisko 2
