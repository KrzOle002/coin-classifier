import os
import cv2
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from skimage.feature import hog

# =====[ ZAPIS MODELU DO PLIKU ]=====
# Ten skrypt trenuje najlepszy model (SVM + HOG + augmentacja)
# i zapisuje wszystkie potrzebne komponenty do folderu model/
# Zapisane pliki są następnie używane przez camera.py do klasyfikacji na żywo.

os.makedirs("model", exist_ok=True)

dir_out = "dataset_out"
classes = sorted(os.listdir(dir_out))
print(f"Klasy ({len(classes)}): {classes}")



# =====[ Wczytywanie i augmentacja ]=====

print("\nWczytywanie i augmentacja danych...")

def augment_image(img):
    augmented = []
    for angle in [90, 180, 270]:
        M = cv2.getRotationMatrix2D((64, 64), angle, 1.0)
        augmented.append(cv2.warpAffine(img, M, (128, 128)))
    augmented.append(cv2.flip(img, 1))
    augmented.append(cv2.flip(img, 0))
    augmented.append(np.clip(img.astype(np.int16) + 30, 0, 255).astype(np.uint8))
    augmented.append(np.clip(img.astype(np.int16) - 30, 0, 255).astype(np.uint8))
    return augmented

X_raw, y = [], []
for c in classes:
    class_path = os.path.join(dir_out, c)
    for img_name in os.listdir(class_path):
        img = cv2.imread(os.path.join(class_path, img_name))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X_raw.append(img_rgb)
        y.append(c)
        for aug in augment_image(img_rgb):
            X_raw.append(aug)
            y.append(c)

print(f"Łącznie {len(X_raw)} próbek po augmentacji.")



# =====[ Ekstrakcja cech HOG ]=====

print("Ekstrakcja cech HOG...")

def extract_hog(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return hog(gray, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)

X_hog = np.array([extract_hog(img) for img in X_raw])
y = np.array(y)



# =====[ Standaryzacja + PCA ]=====

print("Standaryzacja i PCA...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_hog)

n_components = min(50, X_scaled.shape[1], len(X_scaled) - 1)
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)



# =====[ Trening SVM ]=====

print("Trening SVM (RBF, C=10)...")

clf = SVC(kernel="rbf", C=10, gamma="scale",
          decision_function_shape="ovr", random_state=42, probability=True)
clf.fit(X_pca, y)

# Szybka weryfikacja na losowym podziale
X_tr, X_te, y_tr, y_te = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)
clf_check = SVC(kernel="rbf", C=10, gamma="scale", decision_function_shape="ovr",
                random_state=42, probability=True)
clf_check.fit(X_tr, y_tr)
acc = clf_check.score(X_te, y_te)
print(f"Weryfikacja (80/20): {acc:.4f} ({acc*100:.2f}%)")



# =====[ ZAPIS WSZYSTKICH KOMPONENTÓW ]=====
# Zapisujemy 4 pliki które są potrzebne do predykcji na nowych obrazach:
# - model/svm_model.pkl   — wytrenowany klasyfikator SVM
# - model/scaler.pkl      — standaryzacja (musi być ta sama co przy treningu)
# - model/pca.pkl         — redukcja wymiarowości (ta sama co przy treningu)
# - model/classes.npy     — lista nazw klas w odpowiedniej kolejności

print("\nZapis modelu...")
joblib.dump(clf,    "model/svm_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(pca,    "model/pca.pkl")
np.save("model/classes.npy", np.array(classes))

print("Zapisano:")
print("  model/svm_model.pkl")
print("  model/scaler.pkl")
print("  model/pca.pkl")
print("  model/classes.npy")
print("\nModel gotowy do użycia w camera.py!")
