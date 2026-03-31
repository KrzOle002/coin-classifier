# Coin Classifier

Klasyfikacja nominałów monet PLN i EUR przy użyciu algorytmów uczenia maszynowego.

## Wersja Pythona: 3.12.3

## Instalacja

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Uruchomienie

```
python main.py
```

Lub poszczególne kroki osobno:

```
python eda.py
python pca.py
python clustering.py
python classification.py
```

## Struktura projektu

```
coin-classifier/
├── dataset/          - zdjęcia monet (17 klas: gr_1, gr_2, ..., e_1, e_2, ...)
├── dataset_out/      - ujednolicone obrazy 128x128 (generowane przez eda.py)
├── eda/              - wykresy EDA (generowane przez eda.py)
├── pca/              - wyniki PCA (generowane przez pca.py)
├── clustering/       - wyniki klasteryzacji (generowane przez clustering.py)
├── classification/   - wyniki klasyfikacji (generowane przez classification.py)
├── eda.py            - eksploracja danych i ujednolicenie
├── pca.py            - redukcja wymiarowosci
├── clustering.py     - klasteryzacja K-Means i DBSCAN
├── classification.py - klasyfikacja KNN, SVM, Random Forest
├── main.py           - uruchamia wszystkie kroki po kolei
└── requirements.txt
```

## Klasy (17)

PLN: gr_1, gr_2, gr_5, gr_10, gr_20, gr_50, zl_1, zl_2, zl_5

EUR: ct_1, ct_2, ct_5, ct_10, ct_20, ct_50, e_1, e_2
