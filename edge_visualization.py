# -*- coding: utf-8 -*-
"""
Wizualizacja krawedzi i cech
==============================
Dla kazdej klasy generuje panel 3-elementowy:
  1. Obraz oryginalny (szarosc)
  2. Mapa krawedzi Canny
  3. Lokalna gestosc krawedzi -- siatka 4×4 (heatmapa)
  4. Projekcje pozioma i pionowa

Pozwala ocenic, jak bardzo krawedzie roznia sie miedzy klasami.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from preprocessing import (DATASET_DIR, CLASSES, IMG_SIZE,
                            BLUR_K, CANNY_LOW, CANNY_HIGH, GRID_N, PROJ_BINS,
                            preprocess, extract_features, feature_names)

OUTPUT_DIR = "edge_viz_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_first_image(cls: str) -> str:
    d = os.path.join(DATASET_DIR, cls)
    files = sorted([f for f in os.listdir(d)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
    return os.path.join(d, files[0])


# -- Panel dla kazdej klasy ----------------------------------------------------
for cls in CLASSES:
    path       = get_first_image(cls)
    gray, edges = preprocess(path)
    binary     = (edges > 0).astype(np.float32)

    # Siatka gestosci
    cell_h = IMG_SIZE // GRID_N
    cell_w = IMG_SIZE // GRID_N
    grid = np.zeros((GRID_N, GRID_N), dtype=np.float32)
    for r in range(GRID_N):
        for c in range(GRID_N):
            cell = binary[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            grid[r, c] = cell.mean()

    # Projekcje
    row_proj = binary.sum(axis=1)
    col_proj = binary.sum(axis=0)

    # -- Rysuj -----------------------------------------------------------------
    fig = plt.figure(figsize=(14, 4))
    gs  = gridspec.GridSpec(1, 5, figure=fig, wspace=0.35)

    # 1. Oryginal
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(gray, cmap="gray")
    ax1.set_title("Szarosc", fontsize=9)
    ax1.axis("off")

    # 2. Krawedzie Canny
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(edges, cmap="gray")
    ax2.set_title("Krawedzie Canny", fontsize=9)
    ax2.axis("off")

    # 3. Heatmapa siatki 4×4
    ax3 = fig.add_subplot(gs[2])
    im = ax3.imshow(grid, cmap="YlOrRd", vmin=0, vmax=0.3)
    ax3.set_title(f"Gestosc {GRID_N}×{GRID_N}", fontsize=9)
    ax3.set_xticks([])
    ax3.set_yticks([])
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    # 4. Projekcja pozioma (wiersze)
    ax4 = fig.add_subplot(gs[3])
    ax4.barh(range(len(row_proj)), row_proj, color="steelblue", height=1.0)
    ax4.invert_yaxis()
    ax4.set_title("Proj. pionowa\n(sumy wierszy)", fontsize=9)
    ax4.set_xlabel("suma krawedzi")
    ax4.set_ylabel("wiersz")

    # 5. Projekcja pionowa (kolumny)
    ax5 = fig.add_subplot(gs[4])
    ax5.bar(range(len(col_proj)), col_proj, color="coral", width=1.0)
    ax5.set_title("Proj. pozioma\n(sumy kolumn)", fontsize=9)
    ax5.set_xlabel("kolumna")
    ax5.set_ylabel("suma krawedzi")

    fig.suptitle(f"Klasa: {cls}", fontsize=13, fontweight="bold")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{cls}_edge_panel.png"),
                dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[zapisano] {OUTPUT_DIR}/{cls}_edge_panel.png")


# -- Zbiorczy wykres: siatki gestosci dla wszystkich klas ---------------------
fig, axes = plt.subplots(2, 4, figsize=(13, 7))
axes = axes.flatten()

all_grids = {}
for i, cls in enumerate(CLASSES):
    path       = get_first_image(cls)
    gray, edges = preprocess(path)
    binary     = (edges > 0).astype(np.float32)
    cell_h = IMG_SIZE // GRID_N
    cell_w = IMG_SIZE // GRID_N
    grid = np.zeros((GRID_N, GRID_N), dtype=np.float32)
    for r in range(GRID_N):
        for c in range(GRID_N):
            cell = binary[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            grid[r, c] = cell.mean()
    all_grids[cls] = grid

vmax = max(g.max() for g in all_grids.values())
for i, cls in enumerate(CLASSES):
    im = axes[i].imshow(all_grids[cls], cmap="YlOrRd", vmin=0, vmax=vmax)
    axes[i].set_title(cls, fontsize=11, fontweight="bold")
    axes[i].set_xticks([])
    axes[i].set_yticks([])

fig.suptitle(f"Siatka gestosci krawedzi {GRID_N}×{GRID_N} -- porownanie klas",
             fontsize=13, fontweight="bold")
plt.colorbar(im, ax=axes, shrink=0.6, label="Gestosc krawedzi")
plt.savefig(os.path.join(OUTPUT_DIR, "all_classes_grid_density.png"),
            dpi=120, bbox_inches="tight")
plt.close()
print(f"[zapisano] {OUTPUT_DIR}/all_classes_grid_density.png")

print(f"\n[Wizualizacja zakonczona -- wyniki w '{OUTPUT_DIR}/']")
