# -*- coding: utf-8 -*-
"""
EDA - Exploratory Data Analysis
================================
Cel: zrozumienie struktury zbioru danych przed przetwarzaniem.
Sprawdzamy licznosc klas, wyswietlamy przykladowe obrazy oraz
mapy krawedzi (Canny) dla kazdej klasy.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# -- Konfiguracja --------------------------------------------------------------
DATASET_DIR = "dataset"
IMG_SIZE    = 128
CLASSES     = sorted(os.listdir(DATASET_DIR))
OUTPUT_DIR  = "eda_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- 1. Licznosc klas ----------------------------------------------------------
print("=" * 55)
print("  EDA -- klasyfikacja monet euro")
print("=" * 55)

counts = {}
for cls in CLASSES:
    files = [f for f in os.listdir(os.path.join(DATASET_DIR, cls))
             if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    counts[cls] = len(files)

print(f"\nLiczba klas: {len(CLASSES)}")
print(f"{'Klasa':<12} {'Liczba obrazow':>15}")
print("-" * 28)
for cls, n in counts.items():
    print(f"  {cls:<10} {n:>12}")
print(f"\n  RAZEM: {sum(counts.values())} obrazow")

# Wykres licznosci
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(counts.keys(), counts.values(), color="steelblue", edgecolor="white")
ax.bar_label(bars, padding=3, fontsize=9)
ax.set_title("Licznosc klas -- monety euro", fontsize=13, fontweight="bold")
ax.set_xlabel("Klasa (nominal)")
ax.set_ylabel("Liczba obrazow")
ax.set_ylim(0, max(counts.values()) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_class_distribution.png"), dpi=120)
plt.close()
print(f"\n[zapisano] {OUTPUT_DIR}/01_class_distribution.png")


# -- Pomocnicza: wczytaj i standaryzuj obraz -----------------------------------
def load_gray(path: str) -> np.ndarray:
    """Wczytaj obraz jako skala szarosci 128x128."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Nie mozna wczytac: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (IMG_SIZE, IMG_SIZE))


def get_sample_path(cls: str, idx: int = 0) -> str:
    d = os.path.join(DATASET_DIR, cls)
    files = sorted([f for f in os.listdir(d)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
    return os.path.join(d, files[idx])


# -- 2. Przykladowe obrazy (po 1 na klase) ------------------------------------
n = len(CLASSES)
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.flatten()

for i, cls in enumerate(CLASSES):
    img = load_gray(get_sample_path(cls))
    axes[i].imshow(img, cmap="gray")
    axes[i].set_title(cls, fontsize=11, fontweight="bold")
    axes[i].axis("off")

fig.suptitle("Przykladowe obrazy (skala szarosci, 128×128)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_sample_images.png"), dpi=120)
plt.close()
print(f"[zapisano] {OUTPUT_DIR}/02_sample_images.png")


# -- 3. Mapy krawedzi Canny ----------------------------------------------------
# Canny wymaga dwoch progow (low, high). Stosujemy rozmycie Gaussowskie
# przed detekcja, by zredukowac szum i uzyskac czystsze krawedzie.
BLUR_K      = 3    # jadro rozmycia (piksele)
CANNY_LOW   = 30
CANNY_HIGH  = 100

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.flatten()

for i, cls in enumerate(CLASSES):
    img    = load_gray(get_sample_path(cls))
    blur   = cv2.GaussianBlur(img, (BLUR_K, BLUR_K), 0)
    edges  = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
    axes[i].imshow(edges, cmap="gray")
    axes[i].set_title(cls, fontsize=11, fontweight="bold")
    axes[i].axis("off")

fig.suptitle("Mapy krawedzi Canny (blur->Canny)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_edge_maps.png"), dpi=120)
plt.close()
print(f"[zapisano] {OUTPUT_DIR}/03_edge_maps.png")


# -- 4. Porownanie oryginal / krawedzie (siatka 8×2) --------------------------
fig = plt.figure(figsize=(14, 5))
gs  = gridspec.GridSpec(2, 8, hspace=0.05, wspace=0.05)

for i, cls in enumerate(CLASSES):
    img   = load_gray(get_sample_path(cls))
    blur  = cv2.GaussianBlur(img, (BLUR_K, BLUR_K), 0)
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    ax_img  = fig.add_subplot(gs[0, i])
    ax_edge = fig.add_subplot(gs[1, i])

    ax_img.imshow(img, cmap="gray")
    ax_img.axis("off")
    if i == 0:
        ax_img.set_ylabel("oryginal", fontsize=9)

    ax_edge.imshow(edges, cmap="gray")
    ax_edge.axis("off")
    ax_edge.set_xlabel(cls, fontsize=9, labelpad=2)
    if i == 0:
        ax_edge.set_ylabel("Canny", fontsize=9)

fig.suptitle("Oryginal vs. krawedzie Canny", fontsize=13, fontweight="bold", y=1.01)
plt.savefig(os.path.join(OUTPUT_DIR, "04_original_vs_edges.png"), dpi=120, bbox_inches="tight")
plt.close()
print(f"[zapisano] {OUTPUT_DIR}/04_original_vs_edges.png")


# -- 5. Podstawowe statystyki pikseli krawedzi --------------------------------
# Dla kazdej klasy liczymy srednia gestosc krawedzi (odsetek bialych pikseli).
print("\n-- Statystyki gestosci krawedzi (Canny) -------------------------")
print(f"{'Klasa':<10} {'sr. gestosc':>12} {'std':>10} {'min':>10} {'max':>10}")
print("-" * 55)

stats_list = []
for cls in CLASSES:
    d = os.path.join(DATASET_DIR, cls)
    files = sorted([f for f in os.listdir(d)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
    densities = []
    for fname in files:
        img   = load_gray(os.path.join(d, fname))
        blur  = cv2.GaussianBlur(img, (BLUR_K, BLUR_K), 0)
        edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
        # gestosc = odsetek pikseli krawedzi
        densities.append(np.mean(edges > 0))
    densities = np.array(densities)
    stats_list.append({
        "klasa":  cls,
        "mean":   densities.mean(),
        "std":    densities.std(),
        "min":    densities.min(),
        "max":    densities.max(),
    })
    print(f"  {cls:<8} {densities.mean():>12.4f} {densities.std():>10.4f} "
          f"{densities.min():>10.4f} {densities.max():>10.4f}")

# Wykres gestosci krawedzi
fig, ax = plt.subplots(figsize=(8, 4))
means = [s["mean"] for s in stats_list]
stds  = [s["std"]  for s in stats_list]
labels = [s["klasa"] for s in stats_list]
ax.bar(labels, means, yerr=stds, capsize=4, color="coral", edgecolor="white")
ax.set_title("Srednia gestosc krawedzi Canny per klasa\n(± std)", fontsize=12)
ax.set_ylabel("Odsetek pikseli krawedzi")
ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.2)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_edge_density_stats.png"), dpi=120)
plt.close()
print(f"\n[zapisano] {OUTPUT_DIR}/05_edge_density_stats.png")

print("\n[EDA zakonczona -- wyniki w katalogu 'eda_output/']")
