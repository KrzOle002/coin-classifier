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
    accuracy_score,
    f1_score
)

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import prepare_dataset

os.makedirs("classification", exist_ok=True)

print("Wczytywanie i przetwarzanie danych...")
X, y = prepare_dataset()

classes = sorted(np.unique(y))
print(f"Zaladowano {len(X)} probek, {len(classes)} klas.")
print(f"Klasy: {classes}")

print("\nPodzial danych (80% train / 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} probek | Test: {len(X_test)} probek")

#Parametry zgodne z planem projektu
classifiers = {
    "KNN (k=7)": KNeighborsClassifier(
        n_neighbors=7, metric="euclidean", weights="distance"
    ),
    "SVM (RBF)": SVC(
        kernel="rbf", C=10, gamma="scale",
        decision_function_shape="ovr", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=None,
        min_samples_leaf=2, random_state=42
    ),
}

results = {}

for name, clf in classifiers.items():
    print(f"\n--- {name} ---")

    print("Trening...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="macro")

    print(f"Dokladnosc (test): {acc:.4f} ({acc*100:.2f}%)")
    print(f"F1 macro (test):   {f1:.4f}")

    print("Cross-walidacja (5-fold)...")
    cv_acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    cv_f1  = cross_val_score(clf, X, y, cv=5, scoring="f1_macro")
    print(f"CV accuracy: {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}")
    print(f"CV F1 macro: {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")

    report = classification_report(y_test, y_pred, target_names=classes)
    print(f"\nRaport klasyfikacji:\n{report}")

    report_path = f"classification/report_{name.replace(' ', '_').replace('(', '').replace(')', '')}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Klasyfikator: {name}\n")
        f.write(f"Dokladnosc (test): {acc:.4f}\n")
        f.write(f"F1 macro (test):   {f1:.4f}\n")
        f.write(f"CV accuracy: {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}\n")
        f.write(f"CV F1 macro: {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}\n\n")
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
        "accuracy":  acc,
        "f1":        f1,
        "cv_mean":   cv_acc.mean(),
        "cv_std":    cv_acc.std(),
        "cv_f1":     cv_f1.mean(),
    }

#=====[ Tabela zbiorcza wyników ]=====

print("\nTabela zbiorcza wynikow:")
print(f"{'Klasyfikator':<20} {'Accuracy':>10} {'F1 macro':>10} {'CV Accuracy':>12} {'CV F1':>10}")
print("-" * 65)
for name, r in results.items():
    print(f"{name:<20} {r['accuracy']:>10.4f} {r['f1']:>10.4f} {r['cv_mean']:>12.4f} {r['cv_f1']:>10.4f}")

with open("classification/tabela_zbiorcza.txt", "w", encoding="utf-8") as f:
    f.write(f"{'Klasyfikator':<20} {'Accuracy':>10} {'F1 macro':>10} {'CV Accuracy':>12} {'CV F1':>10}\n")
    f.write("-" * 65 + "\n")
    for name, r in results.items():
        f.write(f"{name:<20} {r['accuracy']:>10.4f} {r['f1']:>10.4f} {r['cv_mean']:>12.4f} {r['cv_f1']:>10.4f}\n")
print("Tabela zapisana: classification/tabela_zbiorcza.txt")

#=====[ Wykres porównawczy — accuracy i F1 ]=====

print("\nTworzenie wykresow porownawczych...")

names = list(results.keys())
x = np.arange(len(names))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))
b1 = ax.bar(x - 1.5*width, [results[n]["accuracy"] for n in names], width, label="Test accuracy",  color="steelblue")
b2 = ax.bar(x - 0.5*width, [results[n]["f1"]       for n in names], width, label="Test F1 macro",  color="cornflowerblue")
b3 = ax.bar(x + 0.5*width, [results[n]["cv_mean"]  for n in names], width, label="CV accuracy",    color="darkorange")
b4 = ax.bar(x + 1.5*width, [results[n]["cv_f1"]    for n in names], width, label="CV F1 macro",    color="sandybrown")

for bars in [b1, b2, b3, b4]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

ax.set_xlabel("Klasyfikator")
ax.set_ylabel("Wynik")
ax.set_title("Porownanie klasyfikatorow — Accuracy i F1 macro")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("classification/porownanie_dokladnosci.png", dpi=150)
plt.close()

print("Zapisano wykres porownawczy!")
print("\nZakonczona klasyfikacja.")
