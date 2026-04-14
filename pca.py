import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Folder z ujednoliconymi obrazami (wynik eda.py)
dir_out = "dataset_out"

# Folder na wyniki PCA
os.makedirs("pca", exist_ok=True)

classes = sorted(os.listdir(dir_out))

print("Wczytywanie obrazów...")

#Wczytywanie i spłaszczanie obrazów

# Każdy obraz 128x128 RGB spłaszczamy do wektora 128*128*3 = 49152 wartości.
# PCA potrzebuje macierzy (n_samples, n_features).
X = []  # Lista wektorów obrazów
y = []  # Lista etykiet (nazwa klasy)

for c in classes:
    class_path = os.path.join(dir_out, c)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Konwersja BGR -> RGB i spłaszczenie do 1D
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X.append(img.flatten())
        y.append(c)

X = np.array(X)
y = np.array(y)

print(f"Załadowano {len(X)} obrazów z {len(classes)} klas.")
print(f"Rozmiar macierzy wejściowej: {X.shape}")  # (n_samples, 49152)



#Standaryzacja

# Przed PCA standaryzujemy dane (mean=0, std=1), żeby żaden kanał koloru nie dominował tylko przez skalę wartości.
print("\nStandaryzacja danych...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



#PCA - wyjaśniona wariancja

# Najpierw uruchamiamy PCA z większą liczbą komponentów, żeby sprawdzić ile komponentów wystarczy do zachowania np. 95% wariancji.
print("\nObliczanie PCA (analiza wariancji)...")

n_components_analysis = min(100, len(X) - 1, X_scaled.shape[1])
pca_full = PCA(n_components=n_components_analysis)
pca_full.fit(X_scaled)

# Skumulowana wyjaśniona wariancja
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Szukamy ile komponentów potrzeba na 95% wariancji.
# Jeśli nawet 100 komponentów nie osiąga 95%, bierzemy maksymalną dostępną liczbę.
indices_above_95 = np.where(cumulative_variance >= 0.95)[0]
if len(indices_above_95) > 0:
    n_95 = indices_above_95[0] + 1
    label_95 = f"{n_95} komponentów (95% wariancji)"
else:
    n_95 = n_components_analysis
    label_95 = f"95% nie osiągnięte w {n_components_analysis} komponentach"

print(f"Liczba komponentów potrzebnych do wyjaśnienia 95% wariancji: {label_95}")

# Wykres skumulowanej wyjaśnionej wariancji
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o", markersize=3)
plt.axhline(y=0.95, color="r", linestyle="--", label="95% wariancji")

# Rysujemy zieloną linię tylko jeśli próg 95% został osiągnięty
if len(indices_above_95) > 0:
    plt.axvline(x=n_95, color="g", linestyle="--", label=f"{n_95} komponentów")
else:
    plt.text(
        n_components_analysis * 0.6,
        cumulative_variance[-1] - 0.03,
        f"Max wariancja przy {n_components_analysis} komponentach: {cumulative_variance[-1]:.2%}",
        color="g", fontsize=9
    )

plt.xlabel("Liczba komponentów PCA")
plt.ylabel("Skumulowana wyjaśniona wariancja")
plt.title("Wyjaśniona wariancja vs. liczba komponentów PCA")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pca/wariancja_skumulowana.png")
plt.close()

print("Zapisano wykres wariancji skumulowanej!")


#PCA do 50 komponentów (redukcja docelowa)

# Używamy 50 komponentów jako rozsądny kompromis między zachowaniem informacji a redukcją wymiarowości (z 49152 do 50).
N_COMPONENTS = 50
print(f"\nRedukcja do {N_COMPONENTS} komponentów PCA...")

pca = PCA(n_components=N_COMPONENTS)
X_pca = pca.fit_transform(X_scaled)

print(f"Rozmiar po PCA: {X_pca.shape}")  # (n_samples, 50)
print(f"Wyjaśniona wariancja ({N_COMPONENTS} komponentów): {cumulative_variance[N_COMPONENTS-1]:.2%}")



#Wizualizacja 2D (PC1 vs PC2)

# Pierwsze dwa komponenty główne zawierają najwięcej informacji.
# Wizualizujemy próbki na płaszczyźnie PC1 x PC2.
print("\nTworzenie wizualizacji 2D...")

# Generujemy unikalny kolor dla każdej klasy
cmap = plt.get_cmap("tab20")
color_map = {c: cmap(i / len(classes)) for i, c in enumerate(classes)}

plt.figure(figsize=(14, 9))

for c in classes:
    mask = y == c
    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=c,
        alpha=0.6,
        s=40,
        color=color_map[c]
    )

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} wariancji)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} wariancji)")
plt.title("PCA - wizualizacja 2D (PC1 vs PC2)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("pca/pca_2d.png", dpi=150)
plt.close()

print("Zapisano wizualizację 2D!")



#Wizualizacja 3D (PC1, PC2, PC3)

# Dodanie trzeciego komponentu często ujawnia dodatkową strukturę danych.
print("Tworzenie wizualizacji 3D...")

fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111, projection="3d")

for c in classes:
    mask = y == c
    ax.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        X_pca[mask, 2],
        label=c,
        alpha=0.6,
        s=30,
        color=color_map[c]
    )

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.2%})")
ax.set_title("PCA - wizualizacja 3D (PC1, PC2, PC3)")
ax.legend(bbox_to_anchor=(1.1, 1), loc="upper left", fontsize=7)
plt.tight_layout()
plt.savefig("pca/pca_3d.png", dpi=150)
plt.close()

print("Zapisano wizualizację 3D!")



#Eigenfaces - wizualizacja komponentów głównych

# Pierwsze komponenty PCA (tzw. "eigenfaces" / "eigencoins") pokazują kierunki największej zmienności w danych — czyli cechy, które PCA uznało za najbardziej informacyjne.
print("Tworzenie wizualizacji komponentów głównych (eigencoins)...")

n_show = min(16, N_COMPONENTS)
fig, axes = plt.subplots(4, 4, figsize=(12, 12))

for i, ax in enumerate(axes.flatten()):
    if i < n_show:
        # Składowa PCA ma taki sam rozmiar jak spłaszczony obraz
        component = pca.components_[i].reshape(128, 128, 3)

        # Normalizacja do zakresu [0, 1] dla wyświetlenia
        component -= component.min()
        component /= component.max() + 1e-8

        ax.imshow(component)
        ax.set_title(f"PC{i+1} ({pca.explained_variance_ratio_[i]:.2%})", fontsize=8)
        ax.axis("off")
    else:
        ax.axis("off")

plt.suptitle("Pierwsze 16 komponentów głównych PCA (eigencoins)", fontsize=13)
plt.tight_layout()
plt.savefig("pca/eigencoins.png", dpi=150)
plt.close()

print("Zapisano eigencoins!")



#Zapis przetworzonego datasetu

# Zapisujemy X_pca i y do pliku .npz — będą używane przez klasyfikatory w kolejnym kroku (classification.py), żeby nie przeliczać PCA od nowa.
print("\nZapisywanie danych po PCA...")
np.savez("pca/pca_data.npz", X=X_pca, y=y)
np.save("pca/pca_components.npy", pca.components_)

print("Dane zapisane do pca/pca_data.npz")
print("\nZakończono PCA.")