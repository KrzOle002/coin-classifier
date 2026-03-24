import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

os.makedirs("clustering", exist_ok=True)

# =====[ Wczytywanie danych po PCA ]=====

# Używamy tych samych danych co klasyfikatory — cechy po PCA (50 komponentów).
# Klasteryzacja jest uczeniem BEZ nadzoru: algorytm nie zna etykiet klas,
# sam grupuje próbki na podstawie podobieństwa w przestrzeni cech.
print("Wczytywanie danych po PCA...")
data = np.load("pca/pca_data.npz", allow_pickle=True)
X = data["X"]
y = data["y"]

classes = sorted(np.unique(y))
n_classes = len(classes)

# Kodujemy etykiety tekstowe na liczby (potrzebne do metryk)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"Załadowano {len(X)} próbek, {n_classes} klas.")



# =====[ 1. K-MEANS ]=====

# K-Means dzieli dane na k klastrów, minimalizując odległość próbek
# od centroidu (środka) ich klastra. Liczba klastrów k musi być podana z góry.
# Uruchamiamy z k = liczba klas (17), żeby sprawdzić czy algorytm
# sam "odkryje" podział zbliżony do prawdziwego.

print("\n=== K-MEANS ===")

# --- Metoda łokcia (Elbow method) ---
# Sprawdzamy inercję (suma kwadratów odległości do centroidów) dla różnych k.
# Szukamy "łokcia" — miejsca gdzie dodanie kolejnego klastra
# przestaje znacząco zmniejszać inercję.
print("Obliczanie metody łokcia...")

inertias = []
k_range = range(2, 30)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(k_range, inertias, marker="o", markersize=4)
plt.axvline(x=n_classes, color="r", linestyle="--", label=f"k={n_classes} (liczba klas)")
plt.xlabel("Liczba klastrów k")
plt.ylabel("Inercja")
plt.title("Metoda łokcia — K-Means")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("clustering/kmeans_elbow.png", dpi=150)
plt.close()
print("Zapisano wykres łokcia!")

# --- K-Means z k = liczba klas ---
print(f"\nUruchamianie K-Means z k={n_classes}...")
kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X)

# Metryki jakości klasteryzacji:
# - ARI (Adjusted Rand Index): 1.0 = idealne dopasowanie do etykiet, 0.0 = losowe
# - NMI (Normalized Mutual Info): miara wzajemnej informacji między klastrami a klasami
# - Silhouette: jak dobrze próbki pasują do swojego klastra vs. sąsiednich (-1 do 1)
ari_km = adjusted_rand_score(y_encoded, kmeans_labels)
nmi_km = normalized_mutual_info_score(y_encoded, kmeans_labels)
sil_km = silhouette_score(X, kmeans_labels)

print(f"K-Means (k={n_classes}):")
print(f"  Adjusted Rand Index (ARI):    {ari_km:.4f}  (im bliżej 1, tym lepiej)")
print(f"  Normalized Mutual Info (NMI): {nmi_km:.4f}  (im bliżej 1, tym lepiej)")
print(f"  Silhouette Score:             {sil_km:.4f}  (im bliżej 1, tym lepiej)")

# Wizualizacja klastrów K-Means w 2D (PC1 vs PC2)
cmap = plt.get_cmap("tab20")
color_map = {c: cmap(i / n_classes) for i, c in enumerate(classes)}

plt.figure(figsize=(13, 8))
scatter = plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap="tab20", alpha=0.6, s=40)
plt.colorbar(scatter, label="Klaster")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"K-Means (k={n_classes}) — wizualizacja klastrów w przestrzeni PCA")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("clustering/kmeans_clusters_2d.png", dpi=150)
plt.close()

# Wykres prawdziwych klas dla porównania
plt.figure(figsize=(13, 8))
for c in classes:
    mask = y == c
    plt.scatter(X[mask, 0], X[mask, 1], label=c, alpha=0.6, s=40, color=color_map[c])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Prawdziwe klasy — wizualizacja w przestrzeni PCA (dla porównania)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("clustering/true_classes_2d.png", dpi=150)
plt.close()
print("Zapisano wizualizacje K-Means!")



# =====[ 2. DBSCAN ]=====

# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
# grupuje próbki na podstawie gęstości — nie wymaga podania liczby klastrów.
# Próbki w rzadkich obszarach są oznaczane jako szum (etykieta -1).
# Parametry: eps = promień sąsiedztwa, min_samples = min. próbek w klastrze.

print("\n=== DBSCAN ===")

# --- Dobór parametru eps (k-distance graph) ---
# Obliczamy odległość każdej próbki do jej k-tego sąsiada.
# Posortowany wykres tych odległości pomaga wybrać eps — szukamy "łokcia".
print("Szacowanie parametru eps (k-distance graph)...")

k_neighbors = 5
nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(X)
distances, _ = nbrs.kneighbors(X)
k_distances = np.sort(distances[:, -1])[::-1]

plt.figure(figsize=(10, 5))
plt.plot(k_distances)
plt.xlabel("Próbki (posortowane malejąco)")
plt.ylabel(f"Odległość do {k_neighbors}. sąsiada")
plt.title("K-distance graph — szacowanie eps dla DBSCAN")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("clustering/dbscan_kdistance.png", dpi=150)
plt.close()
print("Zapisano k-distance graph!")

# --- DBSCAN z kilkoma wartościami eps ---
# Dobieramy zakres eps automatycznie na podstawie rzeczywistych odległości w danych.
# Bierzemy percentyle odległości do k-tego sąsiada jako punkty startowe.
eps_auto = np.percentile(k_distances, [10, 20, 35, 50, 65, 80])
eps_values = sorted(set([round(float(e), 1) for e in eps_auto]))
print(f"Automatycznie dobrane wartości eps: {eps_values}")

best_eps    = None
best_sil    = -1
best_labels = None

print("\nTestowanie różnych wartości eps...")
for eps in eps_values:
    db = DBSCAN(eps=eps, min_samples=5)
    labels = db.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = np.sum(labels == -1)

    # Silhouette wymaga min. 2 klastrów i przynajmniej jednej nie-szumowej próbki
    if n_clusters >= 2 and np.sum(labels != -1) > 0:
        sil = silhouette_score(X[labels != -1], labels[labels != -1])
    else:
        sil = -1

    print(f"  eps={eps:5.1f} → klastrów: {n_clusters:3d}, szum: {n_noise:3d}, silhouette: {sil:.4f}")

    if sil > best_sil:
        best_sil    = sil
        best_eps    = eps
        best_labels = labels

print(f"\nNajlepsze eps={best_eps} (silhouette={best_sil:.4f})")

# Jeśli żaden eps nie dał klastrów, zwiększamy eps do 2x maksimum z k-distance
# i zmniejszamy min_samples — DBSCAN jest zbyt restrykcyjny dla tych danych.
if best_labels is None:
    print("DBSCAN nie znalazł klastrów przy domyślnych eps — próbuję z większym eps i min_samples=3...")
    for eps_fallback in [k_distances.max() * 0.5, k_distances.max() * 1.0, k_distances.max() * 2.0]:
        db = DBSCAN(eps=eps_fallback, min_samples=3)
        labels = db.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise    = np.sum(labels == -1)
        print(f"  eps={eps_fallback:.1f}, min_samples=3 → klastrów: {n_clusters}, szum: {n_noise}")
        if n_clusters >= 2:
            best_eps    = round(eps_fallback, 1)
            best_labels = labels
            best_sil    = silhouette_score(X[labels != -1], labels[labels != -1]) if np.sum(labels != -1) > 1 else -1
            break

# Ostateczny fallback: jeden wielki klaster (wszystko jako jeden klaster)
if best_labels is None:
    print("DBSCAN nie znalazł struktury w danych — dane są zbyt jednorodne dla DBSCAN.")
    best_eps    = "N/A"
    best_labels = np.zeros(len(X), dtype=int)
    best_sil    = 0.0

# Metryki dla najlepszego DBSCAN
# Dla ARI i NMI wykluczamy próbki szumowe
mask_no_noise = best_labels != -1
n_clusters_best = len(set(best_labels)) - (1 if -1 in best_labels else 0)
n_noise_best    = np.sum(best_labels == -1)

if np.sum(mask_no_noise) > 0 and n_clusters_best >= 2:
    ari_db = adjusted_rand_score(y_encoded[mask_no_noise], best_labels[mask_no_noise])
    nmi_db = normalized_mutual_info_score(y_encoded[mask_no_noise], best_labels[mask_no_noise])
else:
    ari_db = 0.0
    nmi_db = 0.0

print(f"\nDBSCAN (eps={best_eps}, min_samples=5):")
print(f"  Liczba klastrów:              {n_clusters_best}")
print(f"  Próbki szumowe:               {n_noise_best} ({n_noise_best/len(X)*100:.1f}%)")
print(f"  Adjusted Rand Index (ARI):    {ari_db:.4f}")
print(f"  Normalized Mutual Info (NMI): {nmi_db:.4f}")
print(f"  Silhouette Score:             {best_sil:.4f}")

# Wizualizacja DBSCAN w 2D
plt.figure(figsize=(13, 8))
unique_labels = sorted(set(best_labels))
cmap_db = plt.get_cmap("tab20")

for label in unique_labels:
    mask = best_labels == label
    if label == -1:
        plt.scatter(X[mask, 0], X[mask, 1], c="black", alpha=0.3, s=15, label="Szum")
    else:
        color = cmap_db(label / max(1, n_clusters_best))
        plt.scatter(X[mask, 0], X[mask, 1], color=color, alpha=0.6, s=40, label=f"Klaster {label}")

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"DBSCAN (eps={best_eps}) — {n_clusters_best} klastrów, {n_noise_best} próbek szumowych")
if n_clusters_best <= 20:
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("clustering/dbscan_clusters_2d.png", dpi=150)
plt.close()
print("Zapisano wizualizację DBSCAN!")



# =====[ 3. TABELA PORÓWNAWCZA METRYK ]=====

print("\n=== PODSUMOWANIE METRYK ===")

metrics = {
    f"K-Means (k={n_classes})": {"ARI": ari_km, "NMI": nmi_km, "Silhouette": sil_km},
    f"DBSCAN (eps={best_eps})":  {"ARI": ari_db, "NMI": nmi_db, "Silhouette": best_sil},
}

print(f"\n{'Algorytm':<25} {'ARI':>8} {'NMI':>8} {'Silhouette':>12}")
print("-" * 55)
for name, m in metrics.items():
    print(f"{name:<25} {m['ARI']:>8.4f} {m['NMI']:>8.4f} {m['Silhouette']:>12.4f}")

# Zapis tabeli do pliku
with open("clustering/metryki_klasteryzacji.txt", "w", encoding="utf-8") as f:
    f.write("Porównanie algorytmów klasteryzacji\n")
    f.write("=" * 55 + "\n")
    f.write(f"{'Algorytm':<25} {'ARI':>8} {'NMI':>8} {'Silhouette':>12}\n")
    f.write("-" * 55 + "\n")
    for name, m in metrics.items():
        f.write(f"{name:<25} {m['ARI']:>8.4f} {m['NMI']:>8.4f} {m['Silhouette']:>12.4f}\n")
    f.write("\n")
    f.write("Objaśnienie metryk:\n")
    f.write("  ARI (Adjusted Rand Index): zgodność z prawdziwymi etykietami, 1.0 = ideał\n")
    f.write("  NMI (Normalized Mutual Info): wzajemna informacja, 1.0 = ideał\n")
    f.write("  Silhouette: spójność klastrów, zakres [-1, 1], bliżej 1 = lepiej\n")

# Wykres słupkowy metryk
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
metric_names = ["ARI", "NMI", "Silhouette"]
algo_names   = list(metrics.keys())
colors       = ["steelblue", "darkorange"]

for i, metric in enumerate(metric_names):
    vals = [metrics[a][metric] for a in algo_names]
    bars = axes[i].bar(algo_names, vals, color=colors)
    for bar in bars:
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)
    axes[i].set_title(metric)
    axes[i].set_ylim(0, max(max(vals) * 1.3, 0.2))
    axes[i].tick_params(axis="x", rotation=15)
    axes[i].grid(axis="y", alpha=0.3)

plt.suptitle("Porównanie metryk klasteryzacji: K-Means vs DBSCAN", fontsize=13)
plt.tight_layout()
plt.savefig("clustering/porownanie_metryk.png", dpi=150)
plt.close()

print("\nWszystkie pliki zapisane w folderze clustering/")
print("Zakończono klasteryzację.")
