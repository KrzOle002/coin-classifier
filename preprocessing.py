import os
import cv2
import numpy as np

dir = "dataset_out"


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
    Wyciąga cechy geometryczne monety z obrazu:
    - liczba wykrytych okręgów (HoughCircles) — kluczowa dla odróżnienia np. 2ct vs 2 euro
    - promień największego okręgu (rozmiar monety)
    - rozpiętość promieni (różnica max-min) — jak bardzo okręgi są różnej wielkości
    - stosunek najmniejszego do największego promienia — czy okręgi są koncentryczne
    - liczba konturów zewnętrznych — złożoność krawędzi
    - pole największego konturu
    - obwód największego konturu
    - współczynnik kolistości (4π*pole/obwód²) — jak bardzo kontur przypomina okrąg
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

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
        radius_spread = radii[0] - radii[-1]
        radius_ratio = radii[-1] / radii[0] if radii[0] > 0 else 0
    else:
        num_circles = 0
        max_radius = 0
        radius_spread = 0
        radius_ratio = 0

    # =====[ Kontury ]=====
    edges = cv2.Canny(blurred, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    num_contours = len(contours)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        # Współczynnik kolistości: 1.0 = idealny okrąg, im mniej tym bardziej nieregularny
        circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0
    else:
        area = 0
        perimeter = 0
        circularity = 0

    return np.array([
        num_circles,
        max_radius,
        radius_spread,
        radius_ratio,
        num_contours,
        area,
        perimeter,
        circularity,
    ], dtype=np.float32)


def prepare_dataset():
    images, labels = load_data()

    X = []
    y = []

    for img, label in zip(images, labels):
        img = cv2.resize(img, (128, 128))
        feats = extract_features(img)
        X.append(feats)
        y.append(label)

    return np.array(X), np.array(y)


def main():
    X, y = prepare_dataset()
    print("Shape X:", X.shape)
    print("Shape y:", y.shape)
    print("Przyklad cech (pierwsza probka):", X[0])
    print("Nazwy cech: num_circles, max_radius, radius_spread, radius_ratio, "
          "num_contours, area, perimeter, circularity")


if __name__ == "__main__":
    main()