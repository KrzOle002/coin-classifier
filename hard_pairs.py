import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import prepare_dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score
)

os.makedirs("hard_pairs", exist_ok=True)

print("Wczytywanie i przetwarzanie danych...")
X_all, y_all = prepare_dataset()

#=====[ Definicja trudnych par ]=====
#
# Pary klas które najczesciej sie mylily w classification.py.
# Kazda para to dwie klasy ktore sa podobne wizualnie —
# ten sam kolor lub podobny rozmiar.

hard_pairs = [
    ("ct_1",  "ct_2"),    # male miedziane eurocenty
    ("ct_10", "ct_20"),   # zolte eurocenty
    ("ct_1",  "ct_5"),    # male miedziane eurocenty
    ("ct_20", "ct_50"),   # zolte eurocenty
    ("e_1",   "e_2"),     # srebrno-zolte euro
]

classifiers = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42
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

#=====[ Wyniki dla kazdej pary ]=====

all_results = []

for cls_a, cls_b in hard_pairs:
    print(f"\n{'='*50}")
    print(f"Para: {cls_a} vs {cls_b}")
    print(f"{'='*50}")

    # Filtrujemy tylko probki nalezace do tej pary klas
    mask = (y_all == cls_a) | (y_all == cls_b)
    X = X_all[mask]
    y = y_all[mask]

    n_a = np.sum(y == cls_a)
    n_b = np.sum(y == cls_b)
    print(f"Liczba probek: {cls_a}={n_a}, {cls_b}={n_b}, lacznie={len(y)}")

    # Podział 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pair_results = {"para": f"{cls_a} vs {cls_b}"}

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="macro")
        pair_results[name] = {"acc": acc, "f1": f1}
        print(f"  {name:20s}: accuracy={acc:.3f}  F1={f1:.3f}")

    all_results.append(pair_results)

#=====[ Wykres porownawczy wszystkich par ]=====

print("\nTworzenie wykresu porownawczego...")

pairs_labels = [r["para"] for r in all_results]
clf_names = list(classifiers.keys())
colors = ["steelblue", "darkorange", "seagreen"]

x = np.arange(len(pairs_labels))
width = 0.25

fig, ax = plt.subplots(figsize=(16, 7))

for i, (name, color) in enumerate(zip(clf_names, colors)):
    accs = [r[name]["acc"] for r in all_results]
    bars = ax.bar(x + i * width, accs, width, label=name, color=color, alpha=0.85)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)

# Linia bazowa — losowe zgadywanie w problemie binarnym = 50%
ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Losowe (50%)")

ax.set_xlabel("Para klas")
ax.set_ylabel("Accuracy")
ax.set_title("Porownanie trudnych par klas — accuracy klasyfikatorow")
ax.set_xticks(x + width)
ax.set_xticklabels(pairs_labels, rotation=30, ha="right", fontsize=9)
ax.set_ylim(0, 1.15)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("hard_pairs/porownanie_par.png", dpi=150)
plt.close()
print("Zapisano: hard_pairs/porownanie_par.png")

#=====[ Macierze pomylek dla WSZYSTKICH par i WSZYSTKICH klasyfikatorow ]=====

print("\nTworzenie macierzy pomylek dla wszystkich par...")

for cls_a, cls_b in hard_pairs:
    mask = (y_all == cls_a) | (y_all == cls_b)
    X = X_all[mask]
    y = y_all[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Jeden rzad — 3 macierze obok siebie (po jednej na klasyfikator)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"Macierze pomylek: {cls_a} vs {cls_b}", fontsize=13)

    for ax, (name, clf) in zip(axes, classifiers.items()):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred, labels=[cls_a, cls_b])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[cls_a, cls_b])
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(f"{name}\nacc={acc:.3f}")

    plt.tight_layout()
    out_path = f"hard_pairs/confusion_{cls_a}_vs_{cls_b}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Zapisano: {out_path}")

#=====[ Tabela zbiorcza ]=====

print("\nTabela zbiorcza trudnych par:")
header = f"{'Para':<20} {'LR acc':>9} {'SVM acc':>9} {'RF acc':>9}"
print(header)
print("-" * 50)

with open("hard_pairs/tabela_trudnych_par.txt", "w", encoding="utf-8") as f:
    f.write(header + "\n")
    f.write("-" * 50 + "\n")
    for r in all_results:
        line = (f"{r['para']:<20} "
                f"{r['Logistic Regression']['acc']:>9.3f} "
                f"{r['SVM (RBF)']['acc']:>9.3f} "
                f"{r['Random Forest']['acc']:>9.3f}")
        print(line)
        f.write(line + "\n")

print("\nZapisano: hard_pairs/tabela_trudnych_par.txt")
print("\nZakonczona analiza trudnych par.")
