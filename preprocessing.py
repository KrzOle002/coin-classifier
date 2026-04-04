import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

dir = "dataset_out"
N_PCA_COMPONENTS = 50


def load_data():
    images = []
    labels = []

    classes = sorted(os.listdir(dir))

    for c in classes:
        class_path = os.path.join(dir, c)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            images.append(img)
            labels.append(c)

    return images, labels


def extract_features(img):
    """
    Wyciąga cechy krawędziowe monety:
    - obraz krawędzi Canny spłaszczony do wektora (główna informacja dla klasyfikatora)
    - cechy geometryczne z HoughCircles i konturów jako dodatkowy kontekst
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

    # =====[ Krawędzie Canny — główna cecha ]=====
    edges = cv2.Canny(blurred, 40, 120)
    edges_small = cv2.resize(edges, (32, 32))
    edges_flat = (edges_small / 255.0).flatten()  # wektor 32*32 = 1024 wartości

    # =====[ Okręgi — HoughCircles ]=====
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=15,
        param1=60,
        param2=30,
        minRadius=10,
        maxRadius=60,
    )

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        radii = sorted([c[2] for c in circles], reverse=True)
        num_circles = len(radii)
        max_radius = radii[0]
        radius_ratio = radii[-1] / radii[0] if radii[0] > 0 else 0
    else:
        num_circles = 0
        max_radius = 0
        radius_ratio = 0

    # =====[ Kontury ]=====
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0
    else:
        area = 0
        perimeter = 0
        circularity = 0

    geometric = np.array([
        num_circles, max_radius, radius_ratio,
        area, perimeter, circularity,
    ], dtype=np.float32)

    # Łączymy obraz krawędzi (16384) + cechy geometryczne (6)
    return np.concatenate([edges_flat, geometric])


def prepare_dataset():
    images, labels = load_data()

    X = []
    y = []

    for img, label in zip(images, labels):
        img = cv2.resize(img, (128, 128))
        feats = extract_features(img)
        X.append(feats)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    # =====[ Standaryzacja ]=====
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =====[ PCA — redukcja wymiarowości ]=====
    n_components = min(N_PCA_COMPONENTS, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    explained = np.sum(pca.explained_variance_ratio_) * 100
    print(f"PCA: {X.shape[1]} cech → {n_components} komponentów "
          f"({explained:.1f}% wyjasnionej wariancji)")

    return X_pca, y


def main():
    X, y = prepare_dataset()
    print("Shape X po PCA:", X.shape)
    print("Shape y:", y.shape)


if __name__ == "__main__":
    main()