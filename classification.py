# -*- coding: utf-8 -*-
"""
Klasyfikacja - porownanie modeli
==================================
Trenujemy i porownujemy trzy modele klasyczne na cechach krawedziowych:
  • Logistic Regression
  • Random Forest
  • Extra Trees

Dla kazdego modelu: raport klasyfikacji, macierz pomylek, kroswalidacja.
Na koncu: porownanie zbiorcze i analiza waznosci cech (drzewa).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics         import (classification_report, confusion_matrix,
                                     accuracy_score, f1_score)
from sklearn.model_selection import cross_val_score
from sklearn.decomposition   import PCA

import joblib

from preprocessing import prepare_data, CLASSES

OUTPUT_DIR = "classification_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- Dane ----------------------------------------------------------------------
X_train, X_test, y_train, y_test, scaler, fnames = prepare_data()

# -- Definicja modeli ----------------------------------------------------------
MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, C=1.0, solver="lbfgs",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    ),
}

# -- Pomocnicza: rysuj macierz pomylek ----------------------------------------
def plot_confusion_matrix(cm: np.ndarray, title: str, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
                linewidths=0.4, linecolor="white")
    ax.set_xlabel("Predykcja", fontsize=11)
    ax.set_ylabel("Prawdziwa klasa", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=120)
    plt.close()


# -- Trening i ocena kazdego modelu -------------------------------------------
results = {}

for name, model in MODELS.items():
    print(f"\n{'='*55}")
    print(f"  MODEL: {name}")
    print(f"{'='*55}")

    # Trening
    model.fit(X_train, y_train)

    # Predykcja na zbiorze testowym
    y_pred = model.predict(X_test)

    # Metryki
    acc   = accuracy_score(y_test, y_pred)
    f1    = f1_score(y_test, y_pred, average="macro")
    cv_sc = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro", n_jobs=-1)

    results[name] = {"acc": acc, "f1": f1, "cv_mean": cv_sc.mean(), "cv_std": cv_sc.std()}

    print(f"\nAccuracy (test):   {acc:.4f}")
    print(f"F1 macro (test):   {f1:.4f}")
    print(f"CV F1 macro (5-fold train): {cv_sc.mean():.4f} ± {cv_sc.std():.4f}")
    print(f"\nRaport klasyfikacji:\n")
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    # Macierz pomylek
    cm = confusion_matrix(y_test, y_pred)
    safe_name = name.replace(" ", "_").lower()
    plot_confusion_matrix(
        cm,
        title=f"Macierz pomylek -- {name}\n(accuracy={acc:.3f}, F1={f1:.3f})",
        fname=f"cm_{safe_name}.png"
    )
    print(f"[zapisano] {OUTPUT_DIR}/cm_{safe_name}.png")

    # Zapis modelu
    joblib.dump(model, os.path.join(OUTPUT_DIR, f"{safe_name}.joblib"))
    print(f"[zapisano] {OUTPUT_DIR}/{safe_name}.joblib")


# -- Porownanie zbiorcze -------------------------------------------------------
print(f"\n{'='*55}")
print("  POROWNANIE MODELI")
print(f"{'='*55}")
print(f"{'Model':<25} {'Acc':>8} {'F1 macro':>10} {'CV F1':>12}")
print("-" * 58)
for name, r in results.items():
    print(f"  {name:<23} {r['acc']:>8.4f} {r['f1']:>10.4f} "
          f"  {r['cv_mean']:.4f}±{r['cv_std']:.3f}")

# Wykres porownania
fig, ax = plt.subplots(figsize=(8, 4))
model_names = list(results.keys())
accs = [results[n]["acc"] for n in model_names]
f1s  = [results[n]["f1"]  for n in model_names]
x = np.arange(len(model_names))
width = 0.35

bars1 = ax.bar(x - width/2, accs, width, label="Accuracy", color="steelblue", edgecolor="white")
bars2 = ax.bar(x + width/2, f1s,  width, label="F1 macro",  color="coral",     edgecolor="white")
ax.bar_label(bars1, fmt="%.3f", padding=2, fontsize=8)
ax.bar_label(bars2, fmt="%.3f", padding=2, fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=9)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Wynik")
ax.set_title("Porownanie modeli -- Accuracy i F1 macro", fontsize=12, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=120)
plt.close()
print(f"\n[zapisano] {OUTPUT_DIR}/model_comparison.png")


# -- Waznosc cech (Random Forest i Extra Trees) --------------------------------
for name in ["Random Forest", "Extra Trees"]:
    model = MODELS[name]
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Top 20
    top_k = min(20, len(fnames))
    top_idx   = indices[:top_k]
    top_names = [fnames[i] for i in top_idx]
    top_vals  = importances[top_idx]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(top_k), top_vals, color="mediumseagreen", edgecolor="white")
    ax.set_xticks(range(top_k))
    ax.set_xticklabels(top_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Waznosc cechy")
    ax.set_title(f"Waznosc cech -- {name} (top {top_k})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    safe_name = name.replace(" ", "_").lower()
    plt.savefig(os.path.join(OUTPUT_DIR, f"feature_importance_{safe_name}.png"), dpi=120)
    plt.close()
    print(f"[zapisano] {OUTPUT_DIR}/feature_importance_{safe_name}.png")

    print(f"\n-- Top 10 cech ({name}) ------------------------------")
    for rank, i in enumerate(indices[:10], 1):
        print(f"  {rank:2d}. {fnames[i]:<25} {importances[i]:.4f}")


# -- Analiza par trudnych (z macierzy pomylek ET) ------------------------------
# Pokazujemy ktore klasy sa najczesciej mylone przez najlepszy model.
print("\n-- Analiza pomylek (Extra Trees) --------------------------------")
et = MODELS["Extra Trees"]
y_pred_et = et.predict(X_test)
cm_et = confusion_matrix(y_test, y_pred_et)

# Wylacz diagonal (poprawne klasyfikacje)
cm_off = cm_et.copy()
np.fill_diagonal(cm_off, 0)

# Top 5 par z najwieksza liczba pomylek
pairs = []
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        if i != j and cm_off[i, j] > 0:
            pairs.append((cm_off[i, j], CLASSES[i], CLASSES[j]))
pairs.sort(reverse=True)

print(f"  {'Prawdziwa':<12} {'Predykcja':<12} {'Liczba':>8}")
print(f"  {'-'*35}")
for cnt, true_cls, pred_cls in pairs[:10]:
    print(f"  {true_cls:<12} {pred_cls:<12} {cnt:>8}")

print(f"\n[Klasyfikacja zakonczona -- wyniki w '{OUTPUT_DIR}/']")
