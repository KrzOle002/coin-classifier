import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import prepare_data, CLASSES

os.makedirs("classification", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("Wczytywanie i przetwarzanie danych...")
X_train, X_test, y_train, y_test, scaler, fnames = prepare_data()

classes = CLASSES
print(f"Zaladowano dane. Klasy: {classes}")
print(f"Train: {len(X_train)} probek | Test: {len(X_test)} probek")

# Parametry zgodne z planem projektu
classifiers = {
    "SVM": SVC(
        kernel="rbf", C=10, gamma="scale", random_state=42
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1
    ),
}

results = {}
trained_models = {}

for name, clf in classifiers.items():
    print(f"\n--- {name} ---")

    print("Trening...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="macro")

    print(f"Dokladnosc (test): {acc:.4f} ({acc*100:.2f}%)")
    print(f"F1 macro (test):   {f1:.4f}")

    label_indices = list(range(len(classes)))
    report = classification_report(y_test, y_pred, labels=label_indices, target_names=classes)
    print(f"\nRaport klasyfikacji:\n{report}")

    report_path = f"classification/report_{name.replace(' ', '_')}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Klasyfikator: {name}\n")
        f.write(f"Dokladnosc (test): {acc:.4f}\n")
        f.write(f"F1 macro (test):   {f1:.4f}\n\n")
        f.write(report)
    print(f"Raport zapisany: {report_path}")

    cm = confusion_matrix(y_test, y_pred, labels=label_indices)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    fig, ax = plt.subplots(figsize=(14, 12))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True)
    ax.set_title(f"Macierz pomylek - {name}")
    plt.tight_layout()
    cm_path = f"classification/confusion_{name.replace(' ', '_')}.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Macierz pomylek zapisana: {cm_path}")

    results[name] = {
        "accuracy": acc,
        "f1":       f1,
        "model":    clf,
    }
    trained_models[name] = clf

    # Zapis modelu do pliku
    model_filename = f"models/{name.replace(' ', '_')}.joblib"
    joblib.dump(clf, model_filename)
    print(f"Model zapisany: {model_filename}")

# Zapis scalera (potrzebny do predykcji na nowych danych)
joblib.dump(scaler, "models/scaler.joblib")
print("\nScaler zapisany: models/scaler.joblib")

#=====[ Tabela zbiorcza wyników ]=====

print("\nTabela zbiorcza wynikow:")
print(f"{'Klasyfikator':<20} {'Accuracy':>10} {'F1 macro':>10}")
print("-" * 43)
for name, r in results.items():
    print(f"{name:<20} {r['accuracy']:>10.4f} {r['f1']:>10.4f}")

with open("classification/tabela_zbiorcza.txt", "w", encoding="utf-8") as f:
    f.write(f"{'Klasyfikator':<20} {'Accuracy':>10} {'F1 macro':>10}\n")
    f.write("-" * 43 + "\n")
    for name, r in results.items():
        f.write(f"{name:<20} {r['accuracy']:>10.4f} {r['f1']:>10.4f}\n")
print("Tabela zapisana: classification/tabela_zbiorcza.txt")

#=====[ Wykres porównawczy — accuracy i F1 ]=====

print("\nTworzenie wykresow porownawczych...")

names = list(results.keys())
x = np.arange(len(names))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))
b1 = ax.bar(x - width/2, [results[n]["accuracy"] for n in names], width, label="Test accuracy", color="steelblue")
b2 = ax.bar(x + width/2, [results[n]["f1"]       for n in names], width, label="Test F1 macro", color="cornflowerblue")

for bars in [b1, b2]:
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
print(f"Modele zapisane w folderze: models/")