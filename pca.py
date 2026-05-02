# -*- coding: utf-8 -*-
"""
Redukcja wymiarowosci - PCA
============================
Analizujemy wplyw liczby glownych skladowych na wariancje wyjasniona.
Wizualizujemy dane w 2D po redukcji PCA, zeby ocenic separowalnosc klas.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from preprocessing import prepare_data, CLASSES

OUTPUT_DIR = "pca_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- Dane ----------------------------------------------------------------------
X_train, X_test, y_train, y_test, scaler, fnames = prepare_data()
X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])

n_features = X_train.shape[1]

# -- PCA pelne (wszystkie komponenty) ------------------------------------------
pca_full = PCA(n_components=n_features, random_state=42)
pca_full.fit(X_train)

cumvar = np.cumsum(pca_full.explained_variance_ratio_)

# Ile komponentow potrzeba na 90%, 95%, 99% wariancji?
for threshold in [0.90, 0.95, 0.99]:
    n_comp = int(np.searchsorted(cumvar, threshold)) + 1
    print(f"  {int(threshold*100)}% wariancji -> {n_comp} komponentow")

# -- Wykres: skumulowana wariancja wyjasniona ----------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(cumvar) + 1), cumvar, marker=".", color="steelblue", linewidth=1.5)
for th, col in [(0.90, "orange"), (0.95, "red"), (0.99, "darkred")]:
    n = int(np.searchsorted(cumvar, th)) + 1
    ax.axhline(th, linestyle="--", color=col, linewidth=0.9,
               label=f"{int(th*100)}% -> {n} komp.")
    ax.axvline(n, linestyle=":", color=col, linewidth=0.9)
ax.set_xlabel("Liczba komponentow PCA")
ax.set_ylabel("Skumulowana wariancja wyjasniona")
ax.set_title("PCA -- skumulowana wariancja wyjasniona", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.set_xlim(1, n_features)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_pca_cumvar.png"), dpi=120)
plt.close()
print(f"[zapisano] {OUTPUT_DIR}/01_pca_cumvar.png")

# -- Indywidualna wariancja komponentow (pierwsze 20) -------------------------
fig, ax = plt.subplots(figsize=(8, 4))
n_show = min(20, n_features)
ax.bar(range(1, n_show + 1), pca_full.explained_variance_ratio_[:n_show],
       color="coral", edgecolor="white")
ax.set_xlabel("Komponent PCA")
ax.set_ylabel("Wariancja wyjasniona")
ax.set_title(f"PCA -- wariancja wyjasniona (pierwsze {n_show} komponentow)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_pca_individual_variance.png"), dpi=120)
plt.close()
print(f"[zapisano] {OUTPUT_DIR}/02_pca_individual_variance.png")

# -- Wizualizacja 2D -----------------------------------------------------------
pca_2d = PCA(n_components=2, random_state=42)
Z_train = pca_2d.fit_transform(X_train)
Z_test  = pca_2d.transform(X_test)
Z_all   = np.vstack([Z_train, Z_test])

colors = plt.cm.tab10(np.linspace(0, 1, len(CLASSES)))

fig, ax = plt.subplots(figsize=(9, 7))
for i, cls in enumerate(CLASSES):
    mask = y_all == i
    ax.scatter(Z_all[mask, 0], Z_all[mask, 1],
               color=colors[i], label=cls, alpha=0.65, s=30, edgecolors="none")

ax.set_xlabel(f"PC1 ({pca_full.explained_variance_ratio_[0]*100:.1f}% wariancji)")
ax.set_ylabel(f"PC2 ({pca_full.explained_variance_ratio_[1]*100:.1f}% wariancji)")
ax.set_title("PCA 2D -- separacja klas monet", fontsize=12, fontweight="bold")
ax.legend(fontsize=9, markerscale=1.5, framealpha=0.8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_pca_2d.png"), dpi=120)
plt.close()
print(f"[zapisano] {OUTPUT_DIR}/03_pca_2d.png")

# -- Analiza: 3 najwazniejsze cechy dla PC1 i PC2 -----------------------------
print("\n-- Najwazniejsze cechy dla PC1 i PC2 ---------------------------")
for pc_idx, pc_name in enumerate(["PC1", "PC2"]):
    loadings = pca_2d.components_[pc_idx]
    top3_idx = np.argsort(np.abs(loadings))[::-1][:3]
    top3 = [(fnames[j], loadings[j]) for j in top3_idx]
    print(f"  {pc_name}: " + ", ".join(f"{n} ({v:+.3f})" for n, v in top3))

print(f"\n[PCA zakonczona -- wyniki w '{OUTPUT_DIR}/']")
