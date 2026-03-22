import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

# Folder na wyniki klasyfikacji
os.makedirs("classification", exist_ok=True)

# =====[ Wczytywanie danych po PCA ]=====

# Ładujemy dane przygotowane przez pca.py — nie musimy ponownie
# przetwarzać obrazów ani przeliczać PCA.
print("Wczytywanie danych po PCA...")
data = np.load("pca/pca_data.npz", allow_pickle=True)
X = data["X"]
y = data["y"]

classes = sorted(np.unique(y))
print(f"Załadowano {len(X)} próbek, {len(classes)} klas.")
print(f"Klasy: {classes}")



# =====[ Podział na zbiór treningowy i testowy ]=====

# Podział 80/20 — 80% danych do trenowania, 20% do testowania.
# stratify=y gwarantuje że każda klasa jest proporcjonalnie reprezentowana
# w obu zbiorach (ważne przy niezbalansowanych danych).
print("\nPodział danych (80% train / 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} próbek | Test: {len(X_test)} próbek")



# =====[ Definicja klasyfikatorów ]=====

# Porównujemy 3 klasyczne algorytmy uczenia z nadzorem:
# - KNN: klasyfikuje na podstawie k najbliższych sąsiadów w przestrzeni PCA
# - SVM: szuka optymalnej hiperpłaszczyzny rozdzielającej klasy
# - Random Forest: zespół drzew decyzyjnych głosujących na wynik
classifiers = {
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF)": SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}



# =====[ Trening, ewaluacja i cross-walidacja ]=====

results = {}

for name, clf in classifiers.items():
    print(f"\n--- {name} ---")

    # Trening na zbiorze treningowym
    print("Trening...")
    clf.fit(X_train, y_train)

    # Predykcja na zbiorze testowym
    y_pred = clf.predict(X_test)

    # Dokładność na zbiorze testowym
    acc = accuracy_score(y_test, y_pred)
    print(f"Dokładność (test): {acc:.4f} ({acc*100:.2f}%)")

    # Cross-walidacja 5-krotna na całym zbiorze — bardziej wiarygodna ocena
    # niż pojedynczy podział, bo uśrednia wyniki z 5 różnych podziałów.
    print("Cross-walidacja (5-fold)...")
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(f"CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Raport klasyfikacji (precision, recall, f1 dla każdej klasy)
    report = classification_report(y_test, y_pred, target_names=classes)
    print(f"\nRaport klasyfikacji:\n{report}")

    # Zapis raportu do pliku tekstowego
    report_path = f"classification/report_{name.replace(' ', '_').replace('(', '').replace(')', '')}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Klasyfikator: {name}\n")
        f.write(f"Dokładność (test): {acc:.4f}\n")
        f.write(f"CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")
        f.write(report)
    print(f"Raport zapisany: {report_path}")

    # Macierz pomyłek
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    fig, ax = plt.subplots(figsize=(14, 12))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True)
    ax.set_title(f"Macierz pomyłek — {name}")
    plt.tight_layout()
    cm_path = f"classification/confusion_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Macierz pomyłek zapisana: {cm_path}")

    # Zapisujemy wyniki do słownika dla wykresów porównawczych
    results[name] = {
        "accuracy": acc,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "y_pred": y_pred
    }



# =====[ Wykres porównawczy — dokładność ]=====

print("\nTworzenie wykresów porównawczych...")

names = list(results.keys())
accuracies = [results[n]["accuracy"] for n in names]
cv_means = [results[n]["cv_mean"] for n in names]
cv_stds = [results[n]["cv_std"] for n in names]

x = np.arange(len(names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, accuracies, width, label="Test accuracy", color="steelblue")
bars2 = ax.bar(x + width/2, cv_means, width, yerr=cv_stds, label="CV accuracy (±std)", color="darkorange", capsize=5)

# Wartości nad słupkami
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

ax.set_xlabel("Klasyfikator")
ax.set_ylabel("Dokładność")
ax.set_title("Porównanie klasyfikatorów — dokładność na zbiorze testowym vs. cross-walidacja")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("classification/porownanie_dokladnosci.png", dpi=150)
plt.close()

print("Zapisano wykres porównawczy!")
print("\nZakończono klasyfikację.")
