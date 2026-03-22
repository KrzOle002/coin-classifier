import subprocess
import sys
import time
import os

# =====[ GŁÓWNY POTOK PROJEKTU ]=====
# Uruchamia wszystkie etapy projektu po kolei:
# 1. EDA        — eksploracja danych i ujednolicenie
# 2. PCA        — redukcja wymiarowości
# 3. Clustering — klasteryzacja (uczenie bez nadzoru)
# 4. Classification — klasyfikatory bazowe (surowe piksele)
# 5. Improvement    — HOG + augmentacja + ulepszone klasyfikatory

STEPS = [
    {
        "name": "EDA — Eksploracja danych i ujednolicenie",
        "file": "eda.py",
        "output_check": "dataset_out",  # folder który powinien powstać
    },
    {
        "name": "PCA — Redukcja wymiarowości",
        "file": "pca.py",
        "output_check": "pca/pca_data.npz",
    },
    {
        "name": "Clustering — Klasteryzacja (K-Means + DBSCAN)",
        "file": "clustering.py",
        "output_check": "clustering/metryki_klasteryzacji.txt",
    },
    {
        "name": "Classification — Klasyfikatory bazowe (KNN, SVM, Random Forest)",
        "file": "classification.py",
        "output_check": "classification/porownanie_dokladnosci.png",
    },
    {
        "name": "Improvement — HOG + augmentacja + ulepszone klasyfikatory",
        "file": "improvement.py",
        "output_check": "improvement/porownanie_przed_po.png",
    },
]

def separator(char="=", width=60):
    print(char * width)

def run_step(step, index, total):
    separator()
    print(f"[{index}/{total}] {step['name']}")
    separator()

    # Sprawdzamy czy plik skryptu istnieje
    if not os.path.isfile(step["file"]):
        print(f"[BŁĄD] Nie znaleziono pliku: {step['file']}")
        print("Upewnij się że wszystkie skrypty są w tym samym folderze co main.py.")
        return False

    start = time.time()

    # Uruchamiamy skrypt tym samym interpreterem Pythona co main.py
    result = subprocess.run(
        [sys.executable, step["file"]],
        capture_output=False  # wyświetlamy output na bieżąco w konsoli
    )

    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n[BŁĄD] Krok '{step['name']}' zakończył się błędem (kod: {result.returncode}).")
        print("Sprawdź komunikaty powyżej i popraw problem przed ponownym uruchomieniem.")
        return False

    # Sprawdzamy czy oczekiwany plik/folder powstał
    if not os.path.exists(step["output_check"]):
        print(f"\n[OSTRZEŻENIE] Oczekiwany wynik nie powstał: {step['output_check']}")
        print("Krok mógł się nie wykonać poprawnie.")
    else:
        print(f"\n[OK] Wynik zapisany: {step['output_check']}")

    print(f"[CZAS] {elapsed:.1f}s")
    return True


def main():
    separator("=")
    print("  KLASYFIKACJA NOMINAŁÓW MONET — POTOK PROJEKTU")
    print("  Autorzy: Krzysztof, Kacper, Julia")
    separator("=")
    print()

    # Sprawdzamy czy dataset istnieje przed startem
    if not os.path.isdir("dataset"):
        print("[BŁĄD] Nie znaleziono folderu 'dataset'.")
        print("Upewnij się że folder z danymi (dataset/) jest w tym samym miejscu co main.py.")
        sys.exit(1)

    total = len(STEPS)
    start_total = time.time()
    completed = 0

    for i, step in enumerate(STEPS, start=1):
        success = run_step(step, i, total)
        if not success:
            print(f"\nPotok zatrzymany na kroku {i}/{total}.")
            print("Napraw błąd i uruchom main.py ponownie — ukończone kroki zostaną pominięte.")
            sys.exit(1)
        completed += 1
        print()

    total_time = time.time() - start_total
    separator("=")
    print(f"  GOTOWE! Wszystkie {completed}/{total} kroków wykonane.")
    print(f"  Łączny czas: {total_time:.1f}s ({total_time/60:.1f} min)")
    separator("=")
    print()
    print("Wyniki zapisane w folderach:")
    print("  dataset_out/   — ujednolicone obrazy")
    print("  eda/           — wykresy eksploracji danych")
    print("  pca/           — redukcja wymiarowości i eigencoins")
    print("  clustering/    — klasteryzacja K-Means i DBSCAN")
    print("  classification/ — klasyfikatory bazowe i macierze pomyłek")
    print("  improvement/   — ulepszone klasyfikatory (HOG + augmentacja)")


if __name__ == "__main__":
    main()
