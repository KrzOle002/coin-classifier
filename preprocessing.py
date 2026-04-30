import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

dir = "dataset_out"

# Tylko monety Euro (zgodnie z wytycznymi)
EURO_CLASSES = {"ct_1", "ct_2", "ct_5", "ct_10", "ct_20", "ct_50", "e_1", "e_2"}

# 12 cech w 4 grupach:
#   [0–1]   HoughCircles        — bimetalik e_1/e_2
#   [2–7]   Krawędziowy profil  — gęstość pikseli Canny w 6 strefach radialnych
#   [8–9]   Hierarchia RETR_TREE — dziury w kształtach
#   [10–11] Krawędź + złożoność — odchylenie kątowe + łączna liczba konturów
N_FEATURES = 12

FEATURE_NAMES = [
    # --- Grupa 1: HoughCircles ---
    "n_circles",        # 0  liczba okręgów (1 dla centów, 2 dla e_1/e_2)
    "radius_ratio",     # 1  min_r/max_r (0 gdy 1 okrąg; ~0.6 dla bimetaliku)
    # --- Grupa 2: Krawędziowy profil radialny (6 stref) ---
    "edge_z1",          # 2  gęstość krawędzi  r < 0.17  (centrum)
    "edge_z2",          # 3  gęstość krawędzi  0.17–0.33
    "edge_z3",          # 4  gęstość krawędzi  0.33–0.50
    "edge_z4",          # 5  gęstość krawędzi  0.50–0.67
    "edge_z5",          # 6  gęstość krawędzi  0.67–0.83
    "edge_z6",          # 7  gęstość krawędzi  0.83–0.97  (ryfle krawędzi)
    # --- Grupa 3: Hierarchia konturów (RETR_TREE) ---
    "n_holes",          # 8  liczba konturów z rodzicem (dziury w cyfrach i wzorach)
    "hole_area_ratio",  # 9  suma pól dziur / pole największego konturu
    # --- Grupa 4: Krawędź kątowa + złożoność ---
    "edge_ring_std",    # 10 odch. std gęstości krawędzi w 16 sektorach kątowych
                        #    niskie = regularne ryfle (ct_10/20/50), wysokie = wzór nieregularny
    "n_total_norm",     # 11 łączna liczba konturów (RETR_LIST) / 100 — złożoność wzoru
]

# Granice 6 stref radialnych (ułamek promienia obrazu)
ZONE_BOUNDS = [
    (0.00, 0.17),
    (0.17, 0.33),
    (0.33, 0.50),
    (0.50, 0.67),
    (0.67, 0.83),
    (0.83, 0.97),
]

N_ANGULAR_SECTORS = 16   # sektory kątowe do edge_ring_std


def load_data():
    images = []
    labels = []
    classes = sorted(c for c in os.listdir(dir) if c in EURO_CLASSES)
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
    Generuje 7 wersji: oryginał + obroty co 60° + odbicie poziome.
    Moneta jest okrągła — każdy obrót to poprawna próbka tej samej klasy.
    """
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    variants = [img]
    for angle in [60, 120, 180, 240, 300]:
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        variants.append(cv2.warpAffine(img, M, (w, h)))
    variants.append(cv2.flip(img, 1))
    return variants


def extract_contour_features(img):
    """
    Wyciąga 12 cech konturowych w 4 grupach:

    Grupa 1 — HoughCircles (2 cechy):
        Wykrywa koncentryczne okręgi. Klucz do odróżnienia e_1/e_2
        (2 okręgi, bimetalik) od 6 centów (1 okrąg).

    Grupa 2 — Krawędziowy profil radialny (6 cech):
        Bezpośrednie zliczanie pikseli Canny w 6 strefach koncentrycznych.
        Bardziej odporne na szum niż liczenie centroidów konturów.
        Strefa 6 (r=83-97%) wykrywa ryfle krawędzi (ct_10/20/50/e_1/e_2).
        Strefy centralne (1-3) różnicują wzory nominałów.

    Grupa 3 — Hierarchia RETR_TREE (2 cechy):
        n_holes: dziury w konturach (cyfry z otworami).
        hole_area_ratio: jak duże są te dziury.

    Grupa 4 — Krawędź kątowa + złożoność (2 cechy):
        edge_ring_std: odch. std gęstości krawędzi w 16 sektorach kątowych
            w zewnętrznej strefie (r=67-97%). Monety z ryflamidzeniami mają
            regularne rozłożenie krawędzi → niższe std.
        n_total_norm: złożoność wzoru (łączna liczba konturów / 100).

    Pipeline: GaussianBlur → Canny → HoughCircles + bezpośrednie zliczanie
              pikseli w strefach + findContours(RETR_TREE)
    """
    gray    = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_u8 = gray if gray.dtype == np.uint8 else (gray * 255).astype(np.uint8)

    h, w   = gray_u8.shape
    half_w = w / 2.0
    cx, cy = w / 2.0, h / 2.0

    blurred = cv2.GaussianBlur(gray_u8, (7, 7), 1.5)
    edges   = cv2.Canny(blurred, 40, 120)

    # Mapa odległości radialnej od środka (normalizowana do [0,1])
    Y, X      = np.mgrid[0:h, 0:w]
    dist_norm = np.sqrt((X - cx)**2 + (Y - cy)**2) / half_w

    # ====================================================================
    # Grupa 1: HoughCircles
    # ====================================================================
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=15,
        param1=60, param2=30,
        minRadius=10, maxRadius=60,
    )

    if circles is not None:
        circ   = sorted(circles[0], key=lambda c: c[2])
        n_c    = float(len(circ))
        max_r  = circ[-1][2]
        min_r  = circ[0][2]
        radius_ratio = float(min_r / (max_r + 1e-6)) if n_c > 1 else 0.0
    else:
        n_c          = 0.0
        radius_ratio = 0.0

    hough_feats = np.array([n_c, radius_ratio], dtype=np.float32)

    # ====================================================================
    # Grupa 2: Krawędziowy profil radialny — bezpośrednie piksele Canny
    # ====================================================================
    edge_zone = np.zeros(6, dtype=np.float32)
    for zi, (r_lo, r_hi) in enumerate(ZONE_BOUNDS):
        mask = (dist_norm >= r_lo) & (dist_norm < r_hi)
        n_px = float(mask.sum())
        if n_px > 0:
            edge_zone[zi] = float(edges[mask].sum()) / (255.0 * n_px)

    # ====================================================================
    # Grupa 3: Hierarchia konturów (RETR_TREE)
    # ====================================================================
    tree_contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    n_holes       = 0.0
    hole_area_sum = 0.0
    largest_ext   = 1.0

    if hierarchy is not None and len(tree_contours) > 0:
        hier      = hierarchy[0]
        ext_areas = [cv2.contourArea(c) for c, hr in zip(tree_contours, hier) if hr[3] == -1]
        if ext_areas:
            largest_ext = max(ext_areas)
        for cnt, hr in zip(tree_contours, hier):
            if hr[3] != -1:
                n_holes       += 1
                hole_area_sum += cv2.contourArea(cnt)

    hole_area_ratio = float(hole_area_sum / (largest_ext + 1e-6))
    tree_feats = np.array([n_holes, hole_area_ratio], dtype=np.float32)

    # ====================================================================
    # Grupa 4: Krawędź kątowa + złożoność konturów
    # ====================================================================
    # edge_ring_std — odch. std gęstości krawędzi w 16 sektorach kątowych
    # liczone w zewnętrznej strefie (r=0.67–0.97) gdzie są ryfle
    angle_map = np.arctan2(Y - cy, X - cx)   # [-π, π]
    outer_mask = (dist_norm >= 0.67) & (dist_norm < 0.97)

    sector_density = np.zeros(N_ANGULAR_SECTORS, dtype=np.float32)
    sector_edges   = np.deg2rad(360.0 / N_ANGULAR_SECTORS)
    for s in range(N_ANGULAR_SECTORS):
        a_lo = -np.pi + s * sector_edges
        a_hi = a_lo + sector_edges
        sec_mask = outer_mask & (angle_map >= a_lo) & (angle_map < a_hi)
        n_px = float(sec_mask.sum())
        if n_px > 0:
            sector_density[s] = float(edges[sec_mask].sum()) / (255.0 * n_px)

    edge_ring_std = float(np.std(sector_density))

    # n_total_norm — łączna liczba konturów RETR_LIST / 100
    all_contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    n_total_norm = float(len(all_contours)) / 100.0

    extra_feats = np.array([edge_ring_std, n_total_norm], dtype=np.float32)

    return np.concatenate([hough_feats, edge_zone, tree_feats, extra_feats])


# Alias dla zgodności z resztą pipeline'u
extract_features = extract_contour_features


def prepare_dataset(augment_train_only=True):
    """
    Wczytuje dane (tylko Euro), dzieli na train/test,
    augmentuje TYLKO zbiór treningowy, scaler fituje TYLKO na train.

    Cechy: 12 cech (2 HoughCircles + 6 profil radialny + 2 hierarchia + 2 kątowe).
    Zwraca: X_train, X_test, y_train, y_test, scaler
    """
    from sklearn.model_selection import train_test_split

    images, labels = load_data()
    labels = np.array(labels)

    print(f"Wczytano {len(images)} obrazow oryginalnych (tylko Euro).")
    print(f"Klasy: {sorted(set(labels))}")

    idx = np.arange(len(images))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Podzial: {len(idx_train)} train / {len(idx_test)} test (przed augmentacja)")

    X_train_list, y_train_list = [], []
    for i in idx_train:
        img = cv2.resize(images[i], (128, 128))
        for variant in augment(img):
            X_train_list.append(extract_features(variant))
            y_train_list.append(labels[i])

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
    print(f"Wymiar cech: {X_train.shape[1]}  (12 — spelnia wymog)")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def main():
    X_train, X_test, y_train, y_test, _ = prepare_dataset()
    print("Shape X_train:", X_train.shape)
    print("Shape X_test:",  X_test.shape)


if __name__ == "__main__":
    main()
