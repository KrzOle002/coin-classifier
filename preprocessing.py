import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

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


def augment(img):
    """
    Generuje 7 wersji obrazu: oryginał + obroty co 60° + odbicie poziome.
    Moneta jest okrągła więc każdy obrót jest poprawną próbką tej samej klasy.
    """
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    variants = [img]

    for angle in [60, 120, 180, 240, 300]:
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        variants.append(rotated)

    variants.append(cv2.flip(img, 1))
    return variants


def extract_digit_features(gray, cx, cy, inner_radius):
    """
    Wyciąga cechy cyfry ze środka monety używając tylko OpenCV.
    Analizuje kontury wewnątrz najmniejszego okręgu (tam jest cyfra nominału).

    Zwraca 6 cech:
    - num_inner_contours : liczba konturów wewnątrz ROI (cyfra "8" ma więcej niż "1")
    - digit_area_ratio   : stosunek pola konturów do pola ROI (jak "pełna" jest cyfra)
    - digit_aspect_ratio : stosunek szerokości do wysokości bounding box cyfry
    - num_holes          : liczba dziur w konturach (0=1/2/5/7, 1=0/4/6/9, 2=8)
    - mean_intensity     : średnia jasność ROI (tło vs cyfra)
    - std_intensity      : odchylenie jasności ROI (kontrast cyfry)
    """
    gray = gray.astype(np.uint8)

    r = max(inner_radius - 10, 5)
    x1 = max(cx - r, 0)
    y1 = max(cy - r, 0)
    x2 = min(cx + r, gray.shape[1])
    y2 = min(cy + r, gray.shape[0])

    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros(6, dtype=np.float32)

    # Preprocessing ROI
    roi_blur = cv2.GaussianBlur(roi, (3, 3), 0)
    _, roi_thresh = cv2.threshold(roi_blur, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    roi_morph = cv2.morphologyEx(roi_thresh, cv2.MORPH_CLOSE, k3)

    # Kontury zewnętrzne (cyfra jako całość)
    contours_ext, _ = cv2.findContours(roi_morph, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
    # Kontury z dziurami (wykrywa wnętrze liter jak "0", "4", "8")
    contours_all, hierarchy = cv2.findContours(roi_morph, cv2.RETR_CCOMP,
                                                cv2.CHAIN_APPROX_SIMPLE)

    roi_area = roi.shape[0] * roi.shape[1]

    # Liczba konturów zewnętrznych
    num_inner_contours = len(contours_ext)

    # Stosunek pola cyfry do pola ROI
    digit_area = sum(cv2.contourArea(c) for c in contours_ext)
    digit_area_ratio = digit_area / roi_area if roi_area > 0 else 0

    # Aspect ratio bounding box największego konturu
    if contours_ext:
        largest = max(contours_ext, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        digit_aspect_ratio = w / h if h > 0 else 0
    else:
        digit_aspect_ratio = 0

    # Liczba dziur (kontury wewnętrzne = dziury w cyfrach jak "0", "8")
    num_holes = 0
    if hierarchy is not None and len(hierarchy) > 0:
        for h in hierarchy[0]:
            if h[3] != -1:  # ma rodzica = jest dziurą
                num_holes += 1

    # Statystyki jasności ROI
    mean_intensity = float(np.mean(roi))
    std_intensity  = float(np.std(roi))

    return np.array([
        num_inner_contours,
        digit_area_ratio,
        digit_aspect_ratio,
        num_holes,
        mean_intensity,
        std_intensity,
    ], dtype=np.float32)


def extract_features(img):
    """
    Wyciąga cechy monety:
    - HOG (Histogram of Oriented Gradients) — ~1764 cech zamiast 16384 raw Canny.
      HOG koduje krawędzie i gradienty efektywniej, jest mniej wrażliwy na szum
      i nie wymaga tak dużo pamięci ani czasu treningu.
    - cechy geometryczne z HoughCircles i konturów (6 wartości)
    - cechy cyfry ze środka monety przez OpenCV (6 wartości)
    Razem: ~1776 cech (zamiast wcześniejszych 16396)
    """
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Kopia do OpenCV (uint8)
    gray_uint8 = gray.copy()

    #Normalizacja tylko do HOG (float)
    gray_float = gray.astype(np.float32) / 255.0

    #Blur robimy na uint8 (dla HoughCircles)
    blurred = cv2.GaussianBlur(gray_uint8, (7, 7), 1.5)

    #HOG - Histogram of Oriented Gradients
    #pixels_per_cell=(16,16), cells_per_block=(2,2), 9 orientacji
    #dla obrazu 128x128: (128/16)=8 komórek na oś → 8x8=64 komórki
    #po normalizacji blokowej (2x2): 7x7=49 bloków → 49*4*9 = 1764 cech
    hog_feats = hog(
        gray_float,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    #Okręgi - HoughCircles
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=15,
        param1=60, param2=30,
        minRadius=10, maxRadius=60,
    )
    if circles is not None:
        circles_sorted = sorted(circles[0], key=lambda c: c[2])  # rosnąco po r
        num_circles  = len(circles_sorted)
        max_radius   = int(circles_sorted[-1][2])
        radius_ratio = circles_sorted[0][2] / circles_sorted[-1][2] if circles_sorted[-1][2] > 0 else 0
        #Środek i promień najmniejszego okręgu - tam jest cyfra
        cx_inner = int(circles_sorted[0][0])
        cy_inner = int(circles_sorted[0][1])
        r_inner  = int(circles_sorted[0][2])
    else:
        num_circles, max_radius, radius_ratio = 0, 0, 0
        cx_inner = cy_inner = gray.shape[0] // 2
        r_inner  = gray.shape[0] // 4

    #Kontury zewnętrzne monety (z krawędzi Canny)
    edges = cv2.Canny(blurred, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest    = max(contours, key=cv2.contourArea)
        area       = cv2.contourArea(largest)
        perimeter  = cv2.arcLength(largest, True)
        circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0
    else:
        area, perimeter, circularity = 0, 0, 0

    geometric = np.array(
        [num_circles, max_radius, radius_ratio, area, perimeter, circularity],
        dtype=np.float32
    )

    #Cechy cyfry ze środka monety
    digit_feats = extract_digit_features(gray, cx_inner, cy_inner, r_inner)

    return np.concatenate([hog_feats, geometric, digit_feats])


def prepare_dataset(augment_train_only=True):
    """
    Wczytuje dane, dzieli na train/test, augmentuje TYLKO zbiór treningowy,
    a scaler fituje TYLKO na danych treningowych — bez data leakage.

    Zwraca: X_train, X_test, y_train, y_test, scaler
    """
    from sklearn.model_selection import train_test_split

    images, labels = load_data()
    labels = np.array(labels)

    print(f"Wczytano {len(images)} obrazow oryginalnych.")

    #Podział przed augmentacją - zapobiega data leakage
    idx = np.arange(len(images))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=labels)

    print(f"Podzial: {len(idx_train)} train / {len(idx_test)} test (przed augmentacja)")

    #Zbiór treningowy - z augmentacją
    X_train_list, y_train_list = [], []
    for i in idx_train:
        img = cv2.resize(images[i], (128, 128))
        for variant in augment(img):
            X_train_list.append(extract_features(variant))
            y_train_list.append(labels[i])

    #Zbiór testowy - BEZ augmentacji, tylko oryginały
    X_test_list, y_test_list = [], []
    for i in idx_test:
        img = cv2.resize(images[i], (128, 128))
        X_test_list.append(extract_features(img))
        y_test_list.append(labels[i])

    X_train = np.array(X_train_list, dtype=np.float32)
    X_test  = np.array(X_test_list,  dtype=np.float32)
    y_train = np.array(y_train_list)
    y_test  = np.array(y_test_list)

    print(f"Po augmentacji: {len(X_train)} probek treningowych (7x), {len(X_test)} testowych")
    print(f"Wymiar cech: {X_train.shape[1]}")

    #Scaler fitowany TYLKO na danych treningowych
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def main():
    X_train, X_test, y_train, y_test, _ = prepare_dataset()
    print("Shape X_train:", X_train.shape)
    print("Shape X_test:",  X_test.shape)


if __name__ == "__main__":
    main()