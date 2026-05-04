# -*- coding: utf-8 -*-
"""
Preprocessing i ekstrakcja cech
================================
Pipeline:
  1. Wczytaj obraz -> skala szarosci -> resize 128×128
  2. Gaussian Blur -> redukcja szumu
  3. Canny edge detection
  4. Ekstrakcja trzech grup cech WYLACZNIE z mapy krawedzi:
       a) centroid krawedzi (2 cechy)
       b) lokalna gestosc krawedzi -- siatka 4×4 (16 cech)
       c) projekcje poziome i pionowe -- po 16 wartosci (32 cechy)
     Lacznie: 50 cech na obraz.

Dlaczego te cechy?
------------------
• Centroid  -> gdzie na monecie jest cyfra / glowny element graficzny.
              Monety z roznymi cyframi maja rozne rozklady krawedzi
              (np. „1" jest wezsza niz „50").
• Siatka 4×4 -> lokalny rozklad krawedzi: cyfry roznia sie gestoscia
               w okreslonych kwadrantach (np. „2" ma krawedzie w lewej
               gornej czesci, „1" niemal tylko w centrum).
• Projekcje -> 1D sumy wierszy i kolumn po binarnej mapie krawedzi.
              Redukuja obraz do profilu horyzontalnego i wertykalnego --
              lekka, czytelna reprezentacja ksztaltu cyfry.
"""

import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -- Parametry -----------------------------------------------------------------
DATASET_DIR  = "dataset"
IMG_SIZE     = 128
BLUR_K       = 3
CANNY_LOW    = 30
CANNY_HIGH   = 100
GRID_N       = 4      # siatka GRID_N × GRID_N dla lokalnej gestosci
PROJ_BINS    = 16     # liczba kubelkow dla projekcji (po redukcji)
TEST_SIZE    = 0.2
RANDOM_STATE = 42

CLASSES = sorted(os.listdir(DATASET_DIR))
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}


# -- Pipeline obrazu ------------------------------------------------------------
def preprocess(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Wczytaj obraz i zwroc (obraz_szarosci, mapa_krawedzi).
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Nie mozna wczytac: {path}")
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray  = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    blur  = cv2.GaussianBlur(gray, (BLUR_K, BLUR_K), 0)
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
    return gray, edges


# -- Ekstrakcja cech ------------------------------------------------------------
def extract_features(edges: np.ndarray) -> np.ndarray:
    """
    Wyciagnij 50-wymiarowy wektor cech z binarnej mapy krawedzi.

    Zwraca:
        np.ndarray ksztaltu (50,)
    """
    H, W = edges.shape
    binary = (edges > 0).astype(np.float32)  # binarna mapa {0, 1}

    # -- (a) Centroid krawedzi (2 cechy) ---------------------------------------
    # Srednia pozycja (row, col) pikseli krawedzi, znormalizowana do [0, 1].
    # Jesli brak krawedzi -- srodek obrazu (0.5, 0.5).
    ys, xs = np.where(binary)
    if len(ys) > 0:
        cy = ys.mean() / H   # znormalizowana pozycja pionowa
        cx = xs.mean() / W   # znormalizowana pozycja pozioma
    else:
        cy, cx = 0.5, 0.5
    centroid = np.array([cy, cx], dtype=np.float32)

    # -- (b) Siatka 4×4 -- lokalna gestosc krawedzi (16 cech) -----------------
    # Dzielimy obraz na GRID_N × GRID_N kafelkow i liczymy odsetek pikseli
    # krawedzi w kazdym kafelku.  Daje to „mape aktywnosci" krawedzi.
    cell_h = H // GRID_N
    cell_w = W // GRID_N
    grid_feats = []
    for r in range(GRID_N):
        for c in range(GRID_N):
            cell = binary[r * cell_h:(r + 1) * cell_h,
                          c * cell_w:(c + 1) * cell_w]
            grid_feats.append(cell.mean())
    grid_density = np.array(grid_feats, dtype=np.float32)  # (16,)

    # -- (c) Projekcje poziome i pionowe (32 cechy) ----------------------------
    # Suma pikseli krawedzi w kazdym wierszu -> profil pionowy (128 wartosci),
    # nastepnie redukujemy do PROJ_BINS za pomoca usredniania.
    # Analogicznie dla kolumn (profil poziomy).
    # Wynik: 2 × PROJ_BINS = 32 wartosci.

    row_proj = binary.sum(axis=1)          # (128,)
    col_proj = binary.sum(axis=0)          # (128,)

    # Redukcja: dzielimy na PROJ_BINS rownych blokow i usredniamy.
    row_proj = row_proj.reshape(PROJ_BINS, -1).mean(axis=1)  # (16,)
    col_proj = col_proj.reshape(PROJ_BINS, -1).mean(axis=1)  # (16,)

    # Normalizacja do [0, 1] (maksymalna suma = szerokosc lub wysokosc okna)
    row_proj = row_proj / (W * (IMG_SIZE // PROJ_BINS))
    col_proj = col_proj / (H * (IMG_SIZE // PROJ_BINS))

    projections = np.concatenate([row_proj, col_proj])       # (32,)

    # -- Zlozenie ---------------------------------------------------------------
    return np.concatenate([centroid, grid_density, projections])  # (50,)


def feature_names() -> list[str]:
    """Zwroc czytelne nazwy wszystkich 50 cech."""
    names = ["centroid_y", "centroid_x"]
    for r in range(GRID_N):
        for c in range(GRID_N):
            names.append(f"grid_{r}{c}_density")
    for i in range(PROJ_BINS):
        names.append(f"row_proj_{i}")
    for i in range(PROJ_BINS):
        names.append(f"col_proj_{i}")
    return names


# -- Wczytaj caly zbior danych -------------------------------------------------
def load_dataset(verbose: bool = True) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Przejdz przez wszystkie klasy i zwroc (X, y, sciezki).

    X: macierz cech (N, 50)
    y: etykiety numeryczne (N,)
    paths: lista sciezek (pomocna przy analizie bledow)
    """
    X, y, paths = [], [], []
    for cls in CLASSES:
        cls_dir = os.path.join(DATASET_DIR, cls)
        files = sorted([f for f in os.listdir(cls_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
        label = CLASS_TO_IDX[cls]
        for fname in files:
            path = os.path.join(cls_dir, fname)
            try:
                _, edges = preprocess(path)
                feats = extract_features(edges)
                X.append(feats)
                y.append(label)
                paths.append(path)
            except Exception as e:
                if verbose:
                    print(f"  [ostrzezenie] Pominieto {path}: {e}")
    return np.array(X, dtype=np.float32), np.array(y), paths


def prepare_data(verbose: bool = True):
    """
    Zaladuj dane, podziel na train/test, zastosuj StandardScaler.

    Skalowanie TYLKO na zbiorze treningowym, aby uniknac wycieku danych.
    Zwraca: X_train, X_test, y_train, y_test, scaler, feature_names_list
    """
    if verbose:
        print("Wczytywanie danych i ekstrakcja cech...")
    X, y, paths = load_dataset(verbose)
    if verbose:
        print(f"  Zaladowano {len(y)} obrazow, {X.shape[1]} cech na obraz")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    if verbose:
        print(f"  Trening: {len(y_train)}, Test: {len(y_test)}")

    return X_train, X_test, y_train, y_test, scaler, feature_names()


# -- Podglad (uruchom bezposrednio) --------------------------------------------
if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, scaler, fnames = prepare_data()
    print("\nNazwy cech:")
    for i, n in enumerate(fnames):
        print(f"  [{i:2d}] {n}")
    print(f"\nKsztalt X_train: {X_tr.shape}")
    print(f"Ksztalt X_test:  {X_te.shape}")
