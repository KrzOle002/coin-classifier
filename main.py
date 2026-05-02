# -*- coding: utf-8 -*-
"""
main.py - punkt wejscia calego pipeline'u
==========================================
Uruchamia kolejno:
  1. EDA          -> eda_output/
  2. Wizualizacja krawedzi -> edge_viz_output/
  3. PCA          -> pca_output/
  4. Klasyfikacja -> classification_output/

Kazdy modul mozna tez uruchomic osobno:
  python eda.py
  python edge_visualization.py
  python pca.py
  python classification.py
"""

import sys
import io
import traceback
import importlib.util

# Ustaw UTF-8 raz, przed uruchomieniem modulow.
# Zabezpieczenie: nie nadpisuj jesli stdout nie ma .buffer (np. redirect do pliku).
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def run_module(module_path: str, label: str) -> bool:
    """Uruchom modul jako skrypt; zwroc True przy sukcesie."""
    print("\n" + "#" * 60)
    print("#  " + label)
    print("#" * 60 + "\n")
    try:
        spec = importlib.util.spec_from_file_location("_mod", module_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return True
    except Exception:
        print("\n[BLAD] Modul '" + module_path + "' zakonczyl sie z bledem:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    steps = [
        ("eda.py",                "KROK 1 - EDA (analiza eksploracyjna)"),
        ("edge_visualization.py", "KROK 2 - Wizualizacja krawedzi i cech"),
        ("pca.py",                "KROK 3 - Redukcja wymiarowosci (PCA)"),
        ("classification.py",     "KROK 4 - Klasyfikacja i porownanie modeli"),
    ]

    # Filtr opcjonalny: python main.py eda pca
    step_filter = set(sys.argv[1:])

    ok_count  = 0
    err_count = 0

    for script, label in steps:
        if step_filter:
            key = script.replace(".py", "")
            if not any(k in key for k in step_filter):
                print("[pominieto] " + script)
                continue

        success = run_module(script, label)
        if success:
            ok_count += 1
        else:
            err_count += 1

    print("\n" + "=" * 60)
    print("  Pipeline zakonczony: " + str(ok_count) + " OK, " + str(err_count) + " BLAD(ow)")
    print("=" * 60)
    print("""
Wyniki w katalogach:
  eda_output/            - wykresy EDA
  edge_viz_output/       - panele krawedzi klas
  pca_output/            - wykresy PCA
  classification_output/ - macierze pomylek, porownanie modeli, modele .joblib
""")
