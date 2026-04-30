import os
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import random

#Folder z klasami
dir = "dataset"

#Folder po ujednoliceniu
dir_out = "dataset_out"
os.makedirs(dir_out, exist_ok=True)

#Podfoldery
classes = sorted(os.listdir(dir))

#Folder, do którego będziemy zapisywać wykresy wynikowe
os.makedirs("eda", exist_ok=True)



#=====[ Ujednolicenie ]=====
print("Ujednolicanie klas...")

#Tworzymy foldery dla ujednoliconych klas
for c in classes:
    c_in = os.path.join(dir, c)
    c_out = os.path.join(dir_out, c)
    os.makedirs(c_out, exist_ok=True)

    #Zmieniamy rozmiary obrazów na 128x128
    for img_name in os.listdir(c_in):
        img_path = os.path.join(c_in, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #Standaryzacja nazwy + wymuszenie formatu PNG
        name = os.path.splitext(img_name)[0]
        name = name.lower().replace(" ", "_")

        output_path = os.path.join(c_out, name + ".png")
        cv2.imwrite(output_path, img)

print("Ujednolicanie zakończone!")



print("\nEDA")

#=====[ Liczność klas ]=====
print("Obliczanie liczności klas...")

#Klucz - klasa
#Wartość - liczność klasy
counts = {}

#Dla każdej klasy liczymy ilość obrazów
for c in classes:
    class_path = os.path.join(dir_out, c)
    images = [
        f for f in os.listdir(class_path)
        if os.path.isfile(os.path.join(class_path, f))
    ]
    counts[c] = len(images)

#Zapis do wykresu
plt.figure(figsize=(12, 5))
colors = ["steelblue" if c.startswith("ct_") else "darkorange" for c in counts.keys()]
bars = plt.bar(counts.keys(), counts.values(), color=colors)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             str(int(bar.get_height())), ha="center", va="bottom", fontsize=8)
plt.xticks(rotation=90)
plt.title("Licznosci klas (niebieski = centy, pomaranczowy = euro)")
plt.ylabel("Liczba obrazow")
plt.tight_layout()
plt.savefig("eda/licznosc_klas.png")
plt.close()

print("Zapisano wykres liczności klas!")

#=====[ Histogramy — grayscale ]=====

print("Tworzenie histogramów pikseli...")

#Dla każdej z klas tworzymy histogram grayscale
for c in classes:
    class_path = os.path.join(dir_out, c)
    all_pixels_gray = []

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        all_pixels_gray.extend(gray.flatten())

    plt.figure()
    plt.hist(all_pixels_gray, bins=256, color="gray")
    plt.title(f"Histogram pikseli (grayscale) - {c}")
    plt.xlabel("Intensywnosc")
    plt.ylabel("Liczba pikseli")
    plt.savefig(f"eda/histogram_{c}.png")
    plt.close()

print("Histogramy zapisane!")

#=====[ Średni obraz dla klasy ]=====

print("Tworzenie średnich obrazów...")

#Dla każdej klasy tworzymy listę wszystkich obrazów.
for c in classes:
    class_path = os.path.join(dir_out, c)
    imgs = []
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (128, 128))
        imgs.append(img)

    #Mając pełną listę generujemy "średnią grafikę". Jest to średnia wartość pikseli wszystkich obrazów.
    mean_img = np.mean(imgs, axis=0).astype("uint8")
    mean_gray = cv2.cvtColor(mean_img, cv2.COLOR_BGR2GRAY)

    plt.figure()

    plt.imshow(mean_gray, cmap="gray")
    plt.axis("off")
    plt.title(f"Mean image - {c}")

    plt.savefig(f"eda/mean_{c}.png")
    plt.close()

print("Średnie obrazy zapisane!")



#=====[ Analiza rozmiarów ]=====

print("Analizowanie rozmiarów obrazów...")

#Tworzymy listy długości i wysokości obrazów. Tym razem bierzemy pod uwagę wszystkie obrazy (bez podziału na klasy)
widths = []
heights = []

#Badamy rozmiary obrazów z każdej klasy
for c in classes:
    class_path = os.path.join(dir_out, c)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        widths.append(w)
        heights.append(h)

plt.figure()
plt.scatter(widths, heights, alpha=0.5)

plt.xlabel("Szerokosc")
plt.ylabel("Wysokosc")
plt.title("Rozmiary obrazow")

plt.savefig("eda/image_sizes.png")

#Histogram szerokości
plt.figure()
plt.hist(widths, bins=20)
plt.xlabel("Szerokosc")
plt.ylabel("Liczba obrazow")
plt.title("Histogram szerokosci obrazow")
plt.savefig("eda/image_widths_hist.png")
plt.close()

#Histogram wysokości
plt.figure()
plt.hist(heights, bins=20)
plt.xlabel("Wysokosc")
plt.ylabel("Liczba obrazow")
plt.title("Histogram wysokosci obrazow")
plt.savefig("eda/image_heights_hist.png")
plt.close()

plt.close()

print("Analiza zakończona!")



#=====[ Przykładowe monety ]=====

print("Zapisywanie przykładowych obrazów...")

#Dla każdej klasy tworzymy losowo zestawienie 5 przykładowych obrazów.
for c in classes:
    class_path = os.path.join(dir_out, c)
    images = os.listdir(class_path)

    #Losowo wybieramy próbkę zestawu obrazów.
    sample = random.sample(images, min(5, len(images)))
    plt.figure(figsize=(10, 2))

    #Składamy zestawienie obrazów z tej próbki
    for i, img_name in enumerate(sample):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #Wybrane obrazy są rozmieszczane "w rzędzie".
        plt.subplot(1, 5, i+1)
        plt.imshow(gray, cmap="gray")
        plt.axis("off")

    plt.suptitle(c)

    plt.savefig(f"eda/samples_{c}.png")
    plt.close()

print("Przykładowe obrazy zapisane")



#=====[ Wizualizacja HoughCircles ]=====

print("Tworzenie wizualizacji wykrytych okręgów...")

#Dla każdej klasy bierzemy pierwsze dostępne zdjęcie i rysujemy wykryte okręgi
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for ax, c in zip(axes.flatten(), classes):
    class_path = os.path.join(dir_out, c)
    img_name = os.listdir(class_path)[0]
    img = cv2.imread(os.path.join(class_path, img_name))
    if img is None:
        ax.axis("off")
        continue

    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=15,
        param1=60, param2=30,
        minRadius=10, maxRadius=60,
    )

    # Rysujemy wykryte okręgi na kopii obrazu
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    n_circles = 0
    if circles is not None:
        circles_int = np.round(circles[0]).astype(int)
        n_circles = len(circles_int)
        for (x, y, r) in circles_int:
            cv2.circle(vis, (x, y), r, (0, 255, 0), 2)
            cv2.circle(vis, (x, y), 2, (0, 0, 255), 3)

    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{c}\n{n_circles} okręgów", fontsize=9)
    ax.axis("off")

plt.suptitle("Wykryte okręgi HoughCircles dla każdej klasy", fontsize=13)
plt.tight_layout()
plt.savefig("eda/hough_circles.png", dpi=150)
plt.close()

print("Zapisano wizualizację HoughCircles!")


# =====[ EDA 12 cech konturowych — tylko klasy Euro ]=====
print("\nEDA 12 cech konturowych (HoughCircles + profil radialny + hierarchia, tylko Euro)...")

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import extract_contour_features, EURO_CLASSES, FEATURE_NAMES, N_FEATURES

euro_classes = sorted(c for c in classes if c in EURO_CLASSES)

# Zbieramy cechy ze wszystkich obrazów per klasa
class_features = {}
for c in euro_classes:
    class_path = os.path.join(dir_out, c)
    feats_list = []
    for img_name in sorted(os.listdir(class_path)):
        img = cv2.imread(os.path.join(class_path, img_name))
        if img is None:
            continue
        img = cv2.resize(img, (128, 128))
        feats_list.append(extract_contour_features(img))
    if feats_list:
        class_features[c] = np.array(feats_list, dtype=np.float32)

n_classes = len(class_features)
palette   = plt.cm.tab10(np.linspace(0, 1, n_classes))
color_map = {c: col for c, col in zip(class_features.keys(), palette)}

# -----[ 1. Boxploty 12 cech per klasa ]-------------------------------
print("  1/6 Boxploty 12 cech per klasa...")

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
fig.suptitle(
    "Rozkład 12 cech konturowych per klasa Euro (boxploty)\n"
    "Grupa 1 [0–1]: HoughCircles | Grupa 2 [2–7]: Profil radialny (6 stref) | Grupa 3 [8–9]: Hierarchia | Grupa 4 [10–11]: Kątowe",
    fontsize=12
)
group_colors = ["#d0e8ff"] * 2 + ["#d0ffd0"] * 6 + ["#ffd0d0"] * 2 + ["#fff0d0"] * 2

for fi, feat_name in enumerate(FEATURE_NAMES):
    ax = axes[fi // 4][fi % 4]
    data = [class_features[c][:, fi] for c in class_features]
    bp = ax.boxplot(data, patch_artist=True, medianprops={"color": "black", "linewidth": 2})
    for patch, c in zip(bp["boxes"], class_features):
        patch.set_facecolor(color_map[c])
        patch.set_alpha(0.75)
    ax.set_facecolor(group_colors[fi])
    ax.set_title(feat_name, fontsize=9, fontweight="bold")
    ax.set_xticks(range(1, n_classes + 1))
    ax.set_xticklabels(list(class_features.keys()), rotation=45, fontsize=7, ha="right")
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("eda/kontury_boxploty_cech.png", dpi=150)
plt.close()

# -----[ 2. Heatmapa korelacji 12×12 ]----------------------------------
print("  2/6 Heatmapa korelacji 12×12...")

all_feats = np.vstack(list(class_features.values()))
corr = np.corrcoef(all_feats.T)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
plt.colorbar(im, ax=ax, label="Współczynnik Pearsona")
ax.set_xticks(range(N_FEATURES))
ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(N_FEATURES))
ax.set_yticklabels(FEATURE_NAMES, fontsize=9)
for i in range(N_FEATURES):
    for j in range(N_FEATURES):
        ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                fontsize=7, color="white" if abs(corr[i,j]) > 0.6 else "black")
for b in [2, 8, 10]:
    ax.axhline(b - 0.5, color="white", linewidth=2)
    ax.axvline(b - 0.5, color="white", linewidth=2)
ax.set_title("Macierz korelacji Pearsona — 12 cech konturowych (Euro)\n"
             "linie = granice grup (Hough | Profil 6 stref | Hierarchia | Kątowe)", fontsize=12)
plt.tight_layout()
plt.savefig("eda/kontury_korelacja_cech.png", dpi=150)
plt.close()

# -----[ 3. Heatmapa klasy × 12 cech (znorm.) ]------------------------
print("  3/6 Heatmapa klasy × cechy...")

mean_matrix = np.array([class_features[c].mean(axis=0) for c in class_features])
col_range = mean_matrix.max(axis=0) - mean_matrix.min(axis=0)
col_range[col_range == 0] = 1.0
mean_norm = (mean_matrix - mean_matrix.min(axis=0)) / col_range

fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(mean_norm, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label="Wartość (znorm. per cecha)")
ax.set_xticks(range(N_FEATURES))
ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(n_classes))
ax.set_yticklabels(list(class_features.keys()), fontsize=10)
for i in range(n_classes):
    for j in range(N_FEATURES):
        ax.text(j, i, f"{mean_norm[i,j]:.2f}", ha="center", va="center",
                fontsize=7, color="white" if mean_norm[i,j] > 0.65 else "black")
for b in [2, 8, 10]:
    ax.axvline(b - 0.5, color="dodgerblue", linewidth=2)
ax.set_title("Heatmapa: klasy Euro × 12 cech konturowych (srednie znormalizowane)\n"
             "niebieski pasek = granica grup cech", fontsize=12)
plt.tight_layout()
plt.savefig("eda/kontury_heatmapa_klas.png", dpi=150)
plt.close()

# -----[ 4. Scatter kluczowych par cech ]-------------------------------
print("  4/6 Scatter plot kluczowych par cech...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Rozrzut kluczowych par cech — monety Euro", fontsize=13)

pairs = [
    ("n_circles",    "radius_ratio",
     "HoughCircles: n_circles vs radius_ratio\ne_1/e_2 → 2 okręgi, radius_ratio > 0"),
    ("edge_z1",      "edge_z6",
     "Profil radialny: centrum vs obrzeże\nryfle → wysoka edge_z6 dla ct_10/20/50/e_1/e_2"),
    ("n_holes",      "edge_ring_std",
     "Hierarchia vs krawędź kątowa\nryfle regularne → niska edge_ring_std"),
]

for ax, (xname, yname, title) in zip(axes, pairs):
    xi = FEATURE_NAMES.index(xname)
    yi = FEATURE_NAMES.index(yname)
    for c, feats in class_features.items():
        ax.scatter(feats[:, xi], feats[:, yi],
                   color=color_map[c], alpha=0.5, s=25, label=c)
    ax.set_xlabel(xname, fontsize=10)
    ax.set_ylabel(yname, fontsize=10)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("eda/kontury_scatter_cechy.png", dpi=150)
plt.close()

# -----[ 5. Srednie mapy krawędzi Canny per klasa ]--------------------
print("  5/6 Srednie mapy krawędzi Canny...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Srednie mapy krawędzi Canny per klasa Euro\n"
             "(uśrednione mapy — podstawa do findContours)", fontsize=13)

for ax, c in zip(axes.flatten(), euro_classes):
    class_path = os.path.join(dir_out, c)
    edge_imgs = []
    for img_name in sorted(os.listdir(class_path)):
        img = cv2.imread(os.path.join(class_path, img_name))
        if img is None:
            continue
        img = cv2.resize(img, (128, 128))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        blur = cv2.GaussianBlur(gray, (7, 7), 1.5)
        edge_imgs.append(cv2.Canny(blur, 40, 120).astype(np.float32))
    if edge_imgs:
        ax.imshow(np.mean(edge_imgs, axis=0), cmap="hot")
    ax.set_title(c, fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig("eda/kontury_srednie_canny.png", dpi=150)
plt.close()

# -----[ 6. Grupowany wykres srednich cech ]----------------------------
print("  6/6 Grupowany wykres srednich cech...")

x     = np.arange(N_FEATURES)
bar_w = 0.8 / n_classes

fig, ax = plt.subplots(figsize=(16, 5))
for i, (c, row) in enumerate(zip(class_features.keys(), mean_norm)):
    offset = (i - n_classes / 2 + 0.5) * bar_w
    ax.bar(x + offset, row, width=bar_w, label=c, color=color_map[c], alpha=0.85)

for start, end, col in [(0, 2, "#eef6ff"), (2, 8, "#eeffee"), (8, 10, "#ffeeee"), (10, 12, "#fff8ee")]:
    ax.axvspan(start - 0.5, end - 0.5, color=col, zorder=0)

ax.set_xticks(x)
ax.set_xticklabels(FEATURE_NAMES, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Srednia wartość cechy (znorm. per cecha)")
ax.set_title("Porównanie klas — srednie 12 cech konturowych\n"
             "tło = grupy: Hough / Profil 6 stref / Hierarchia / Kątowe")
ax.legend(fontsize=8, ncol=4)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("eda/kontury_srednie_cechy.png", dpi=150)
plt.close()

print("EDA zapisane!")
print("  eda/kontury_boxploty_cech.png   — boxploty 12 cech (3 grupy z kolorowym tłem)")
print("  eda/kontury_korelacja_cech.png  — heatmapa korelacji 12×12")
print("  eda/kontury_heatmapa_klas.png   — heatmapa klasy × cechy")
print("  eda/kontury_scatter_cechy.png   — scatter: Hough / profil radialny / hierarchia")
print("  eda/kontury_srednie_canny.png   — srednie mapy Canny per klasa")
print("  eda/kontury_srednie_cechy.png   — grupowany wykres srednich")

print("\nZakończono generowanie EDA.")