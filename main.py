import subprocess
import sys
import time
import os

STEPS = [
    {
        "name": "EDA - Eksploracja danych i ujednolicenie",
        "file": "eda.py",
        "output_check": "dataset_out",
    },
    {
        "name": "PCA - Redukcja wymiarowosci",
        "file": "pca.py",
        "output_check": "pca/pca_data.npz",
    },
    {
        "name": "Clustering - Klasteryzacja (K-Means + DBSCAN)",
        "file": "clustering.py",
        "output_check": "clustering/metryki_klasteryzacji.txt",
    },
    {
        "name": "Classification - Klasyfikatory (KNN, SVM, Random Forest)",
        "file": "classification.py",
        "output_check": "classification/porownanie_dokladnosci.png",
    },
]

def separator(char="=", width=60):
    print(char * width)

def run_step(step, index, total):
    separator()
    print(f"[{index}/{total}] {step['name']}")
    separator()

    if not os.path.isfile(step["file"]):
        print(f"[BLAD] Nie znaleziono pliku: {step['file']}")
        return False

    start = time.time()

    result = subprocess.run(
        [sys.executable, step["file"]],
        capture_output=False
    )

    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n[BLAD] Krok '{step['name']}' zakonczyl sie bledem (kod: {result.returncode}).")
        return False

    if not os.path.exists(step["output_check"]):
        print(f"\n[OSTRZEZENIE] Oczekiwany wynik nie powstal: {step['output_check']}")
    else:
        print(f"\n[OK] Wynik zapisany: {step['output_check']}")

    print(f"[CZAS] {elapsed:.1f}s")
    return True


def main():
    separator("=")
    print("  KLASYFIKACJA NOMINALOW MONET - POTOK PROJEKTU")
    print("  Autorzy: Krzysztof, Kacper, Julia")
    separator("=")
    print()

    if not os.path.isdir("dataset"):
        print("[BLAD] Nie znaleziono folderu 'dataset'.")
        sys.exit(1)

    total = len(STEPS)
    start_total = time.time()
    completed = 0

    for i, step in enumerate(STEPS, start=1):
        success = run_step(step, i, total)
        if not success:
            print(f"\nPotok zatrzymany na kroku {i}/{total}.")
            sys.exit(1)
        completed += 1
        print()

    total_time = time.time() - start_total
    separator("=")
    print(f"  GOTOWE! Wszystkie {completed}/{total} krokow wykonane.")
    print(f"  Laczny czas: {total_time:.1f}s ({total_time/60:.1f} min)")
    separator("=")
    print()
    print("Wyniki zapisane w folderach:")
    print("  dataset_out/    - ujednolicone obrazy")
    print("  eda/            - wykresy eksploracji danych")
    print("  pca/            - redukcja wymiarowosci i eigencoins")
    print("  clustering/     - klasteryzacja K-Means i DBSCAN")
    print("  classification/ - klasyfikatory i macierze pomylek")


if __name__ == "__main__":
    main()
