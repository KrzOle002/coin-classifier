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

os.makedirs("classification", exist_ok=True)

print("Wczytywanie danych po PCA...")
data = np.load("pca/pca_data.npz", allow_pickle=True)
X = data["X"]
y = data["y"]

classes = sorted(np.unique(y))
print(f"Zaladowano {len(X)} probek, {len(classes)} klas.")
print(f"Klasy: {classes}")

print("\nPodzial danych (80% train / 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} probek | Test: {len(X_test)} probek")

classifiers = {
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF)": SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

results = {}

for name, clf in classifiers.items():
    print(f"\n--- {name} ---")

    print("Trening...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Dokladnosc (test): {acc:.4f} ({acc*100:.2f}%)")

    print("Cross-walidacja (5-fold)...")
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(f"CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    report = classification_report(y_test, y_pred, target_names=classes)
    print(f"\nRaport klasyfikacji:\n{report}")

    report_path = f"classification/report_{name.replace(' ', '_').replace('(', '').replace(')', '')}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Klasyfikator: {name}\n")
        f.write(f"Dokladnosc (test): {acc:.4f}\n")
        f.write(f"CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}\n\n")
        f.write(report)
    print(f"Raport zapisany: {report_path}")

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    fig, ax = plt.subplots(figsize=(14, 12))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True)
    ax.set_title(f"Macierz pomylek - {name}")
    plt.tight_layout()
    cm_path = f"classification/confusion_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Macierz pomylek zapisana: {cm_path}")

    results[name] = {
        "accuracy": acc,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "y_pred": y_pred
    }

print("\nTworzenie wykresow porownawczych...")

names = list(results.keys())
accuracies = [results[n]["accuracy"] for n in names]
cv_means = [results[n]["cv_mean"] for n in names]
cv_stds = [results[n]["cv_std"] for n in names]

x = np.arange(len(names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, accuracies, width, label="Test accuracy", color="steelblue")
bars2 = ax.bar(x + width/2, cv_means, width, yerr=cv_stds, label="CV accuracy (+/-std)", color="darkorange", capsize=5)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

ax.set_xlabel("Klasyfikator")
ax.set_ylabel("Dokladnosc")
ax.set_title("Porownanie klasyfikatorow - dokladnosc na zbiorze testowym vs. cross-walidacja")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("classification/porownanie_dokladnosci.png", dpi=150)
plt.close()

print("Zapisano wykres porownawczy!")
print("\nZakonczona klasyfikacja.")
