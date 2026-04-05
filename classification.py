import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import prepare_dataset

os.makedirs("classification", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("Wczytywanie i przetwarzanie danych...")
# prepare_dataset() robi podział PRZED augmentacją i fituje scaler tylko na train
X_train, X_test, y_train, y_test, scaler = prepare_dataset()

classes = sorted(np.unique(y_train))
print(f"Zaladowano dane. Klasy: {classes}")
print(f"Train: {len(X_train)} probek | Test: {len(X_test)} probek")

# Parametry zgodne z planem projektu
classifiers = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42
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

    # Cross-walidacja na zbiorze treningowym (nie całym — bez leakage)
    print("Cross-walidacja (3-fold na danych treningowych)...")
    cv_acc = cross_val_score(clf, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1)
    cv_f1  = cross_val_score(clf, X_train, y_train, cv=3, scoring="f1_macro",  n_jobs=-1)
    print(f"CV accuracy: {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}")
    print(f"CV F1 macro: {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")

    report = classification_report(y_test, y_pred, target_names=classes)
    print(f"\nRaport klasyfikacji:\n{report}")

    report_path = f"classification/report_{name.replace(' ', '_')}.txt"
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
    cm_path = f"classification/confusion_{name.replace(' ', '_')}.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Macierz pomylek zapisana: {cm_path}")

    results[name] = {
        "accuracy":  acc,
        "f1":        f1,
        "cv_mean":   cv_acc.mean(),
        "cv_std":    cv_acc.std(),
        "cv_f1":     cv_f1.mean(),
        "model":     clf,
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
print(f"Modele zapisane w folderze: models/")
