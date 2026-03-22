import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from skimage.feature import hog

# Folder z ujednoliconymi obrazami (wynik eda.py)
dir_out = "dataset_out"

os.makedirs("improvement", exist_ok=True)

classes = sorted(os.listdir(dir_out))
print(f"Klasy: {classes}")



# =====[ 1. DATA AUGMENTATION ]=====

# Zwiększamy dataset przez tworzenie zmodyfikowanych kopii każdego obrazu.
# Dla każdego oryginalnego zdjęcia generujemy dodatkowe warianty:
# obrót o 90/180/270 stopni, odbicie poziome i pionowe, zmiana jasności.
# Dzięki temu model uczy się rozpoznawać monety niezależnie od orientacji i oświetlenia.

print("\n=== 1. DATA AUGMENTATION ===")

def augment_image(img):
    """Zwraca listę augmentowanych wersji obrazu."""
    augmented = []

    # Obroty co 90 stopni
    for angle in [90, 180, 270]:
        M = cv2.getRotationMatrix2D((64, 64), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (128, 128))
        augmented.append(rotated)

    # Odbicie poziome i pionowe
    augmented.append(cv2.flip(img, 1))
    augmented.append(cv2.flip(img, 0))

    # Rozjaśnienie (+30) i przyciemnienie (-30)
    brighter = np.clip(img.astype(np.int16) + 30, 0, 255).astype(np.uint8)
    darker   = np.clip(img.astype(np.int16) - 30, 0, 255).astype(np.uint8)
    augmented.append(brighter)
    augmented.append(darker)

    return augmented

X_aug = []
y_aug = []

for c in classes:
    class_path = os.path.join(dir_out, c)
    count_orig = 0
    count_aug  = 0

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Dodajemy oryginalny obraz
        X_aug.append(img_rgb)
        y_aug.append(c)
        count_orig += 1

        # Dodajemy augmentowane wersje
        for aug in augment_image(img_rgb):
            X_aug.append(aug)
            y_aug.append(c)
            count_aug += 1

    print(f"  {c}: {count_orig} oryginałów → {count_orig + count_aug} łącznie")

X_aug = np.array(X_aug)
y_aug = np.array(y_aug)
print(f"\nDataset po augmentacji: {len(X_aug)} obrazów")



# =====[ 2. EKSTRAKCJA CECH HOG ]=====

# HOG (Histogram of Oriented Gradients) opisuje kształt obiektu przez
# analizę gradientów jasności w lokalnych obszarach obrazu.
# W przeciwieństwie do surowych pikseli, HOG jest odporny na zmiany
# jasności i koloru — skupia się na krawędziach i kształtach.

print("\n=== 2. EKSTRAKCJA CECH HOG ===")

def extract_hog(img):
    """Ekstrahuje cechy HOG z obrazu RGB."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    features = hog(
        gray,
        orientations=9,        # 9 kierunków gradientu
        pixels_per_cell=(8, 8),  # rozmiar komórki
        cells_per_block=(2, 2),  # normalizacja po blokach
        block_norm="L2-Hys",
        feature_vector=True
    )
    return features

print("Ekstrakcja HOG z augmentowanego datasetu...")
X_hog = np.array([extract_hog(img) for img in X_aug])
print(f"Rozmiar wektora HOG: {X_hog.shape[1]} cech (z {128*128} pikseli)")



# =====[ 3. STANDARYZACJA I PCA NA CECHACH HOG ]=====

# Standaryzujemy cechy HOG i redukujemy wymiarowość przez PCA,
# tak samo jak robiliśmy na surowych pikselach.
print("\n=== 3. PCA NA CECHACH HOG ===")

scaler = StandardScaler()
X_hog_scaled = scaler.fit_transform(X_hog)

# Redukujemy do 50 komponentów
n_components = min(50, X_hog_scaled.shape[1], len(X_hog_scaled) - 1)
pca = PCA(n_components=n_components)
X_hog_pca = pca.fit_transform(X_hog_scaled)

explained = np.sum(pca.explained_variance_ratio_)
print(f"PCA: {n_components} komponentów, wyjaśniona wariancja: {explained:.2%}")

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X_hog_pca, y_aug, test_size=0.2, random_state=42, stratify=y_aug
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")



# =====[ 4. GRIDSEARCH — DOBÓR PARAMETRÓW ]=====

# Parametry zgodne z planem projektu (z dokumentu Plan-działania)

print("\n=== 4. TRENING KLASYFIKATORÓW ===")

best_clfs = {
    "KNN": KNeighborsClassifier(
        n_neighbors=7, metric="euclidean", weights="distance"
    ),
    "SVM": SVC(
        kernel="rbf", C=10, gamma="scale",
        decision_function_shape="ovr", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=None,
        min_samples_leaf=2, random_state=42
    )
}

for name, clf in best_clfs.items():
    print(f"Trening: {name}...")
    clf.fit(X_train, y_train)
    print(f"  Gotowe!")



# =====[ 5. EWALUACJA NAJLEPSZYCH MODELI ]=====

print("\n=== 5. EWALUACJA ===")

results_improved = {}

for name, clf in best_clfs.items():
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(clf, X_hog_pca, y_aug, cv=5, scoring="accuracy")

    print(f"\n{name}:")
    print(f"  Test accuracy: {acc:.4f}")
    print(f"  CV accuracy:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(classification_report(y_test, y_pred, target_names=classes))

    results_improved[name] = {
        "accuracy": acc,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std()
    }

    # Macierz pomyłek
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(14, 12))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True)
    ax.set_title(f"Macierz pomyłek (ulepszona) — {name}")
    plt.tight_layout()
    plt.savefig(f"improvement/confusion_{name.replace(' ', '_')}_improved.png", dpi=150)
    plt.close()

    # Raport tekstowy
    with open(f"improvement/report_{name.replace(' ', '_')}_improved.txt", "w", encoding="utf-8") as f:
        f.write(f"Klasyfikator: {name} (ulepszony)\n")
        f.write(f"Test accuracy: {acc:.4f}\n")
        f.write(f"CV accuracy:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")
        f.write(classification_report(y_test, y_pred, target_names=classes))



# =====[ 6. WYKRES PORÓWNAWCZY: PRZED vs PO ]=====

# Porównujemy dokładność oryginalnych modeli (z classification.py)
# z ulepszonymi modelami (HOG + augmentacja + GridSearch).

print("\n=== 6. PORÓWNANIE PRZED vs PO ===")

# Wyniki z classification.py (wpisane ręcznie na podstawie raportów)
results_original = {
    "KNN":          {"accuracy": 0.3915, "cv_mean": 0.2901},
    "SVM":          {"accuracy": 0.5556, "cv_mean": 0.4240},
    "Random Forest":{"accuracy": 0.5291, "cv_mean": 0.4208}
}

names = ["KNN", "SVM", "Random Forest"]
x = np.arange(len(names))
width = 0.2

fig, ax = plt.subplots(figsize=(13, 7))

# Oryginalne wyniki
orig_test = [results_original[n]["accuracy"] for n in names]
orig_cv   = [results_original[n]["cv_mean"]  for n in names]

# Ulepszone wyniki
impr_test = [results_improved[n]["accuracy"] for n in names]
impr_cv   = [results_improved[n]["cv_mean"]  for n in names]
impr_std  = [results_improved[n]["cv_std"]   for n in names]

b1 = ax.bar(x - 1.5*width, orig_test, width, label="Oryginał — test",      color="steelblue",   alpha=0.85)
b2 = ax.bar(x - 0.5*width, orig_cv,   width, label="Oryginał — CV",        color="steelblue",   alpha=0.45)
b3 = ax.bar(x + 0.5*width, impr_test, width, label="Ulepszony — test",     color="darkorange",  alpha=0.85)
b4 = ax.bar(x + 1.5*width, impr_cv,   width,
            yerr=impr_std, label="Ulepszony — CV (±std)", color="darkorange", alpha=0.45, capsize=5)

for bars in [b1, b2, b3, b4]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

ax.set_xlabel("Klasyfikator")
ax.set_ylabel("Dokładność")
ax.set_title("Porównanie: oryginalne modele vs. ulepszone\n(HOG + augmentacja + GridSearch)")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylim(0, 1.1)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("improvement/porownanie_przed_po.png", dpi=150)
plt.close()

print("Zapisano wykres porównawczy przed/po!")
print("\nZakończono poprawę modeli.")
