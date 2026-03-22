import cv2
import numpy as np
import joblib
import sys
import os
from skimage.feature import hog
from tkinter import Tk, filedialog

# =====[ KLASYFIKACJA MONET ZE ZDJĘCIA ]=====
# Uruchom: python predict_image.py
# Otworzy się okno wyboru pliku — wybierz zdjęcie z monetami.



# =====[ Wczytywanie modelu ]=====

print("Wczytywanie modelu...")
try:
    clf     = joblib.load("model/svm_model.pkl")
    scaler  = joblib.load("model/scaler.pkl")
    pca     = joblib.load("model/pca.pkl")
    classes = np.load("model/classes.npy", allow_pickle=True).tolist()
    print(f"Model wczytany! Klasy: {classes}")
except FileNotFoundError:
    print("[BŁĄD] Nie znaleziono modelu w folderze model/")
    print("Najpierw uruchom: python save_model.py")
    sys.exit(1)



# =====[ Okno wyboru pliku ]=====

# Ukrywamy główne okno tkinter — chcemy tylko dialog wyboru pliku
root = Tk()
root.withdraw()
root.attributes("-topmost", True)  # dialog pojawia się na wierzchu

print("\nOtwieranie okna wyboru pliku...")
img_path = filedialog.askopenfilename(
    title="Wybierz zdjęcie z monetami",
    filetypes=[
        ("Obrazy", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
        ("Wszystkie pliki", "*.*")
    ]
)
root.destroy()

# Jeśli użytkownik zamknął dialog bez wyboru pliku
if not img_path:
    print("Nie wybrano pliku. Zamykanie.")
    sys.exit(0)

print(f"Wybrano: {img_path}")

frame = cv2.imread(img_path)
if frame is None:
    print(f"[BŁĄD] Nie można wczytać obrazu: {img_path}")
    sys.exit(1)

print(f"Wczytano zdjęcie: {os.path.basename(img_path)} ({frame.shape[1]}x{frame.shape[0]}px)")



# =====[ Ładne nazwy ]=====

LABELS = {
    "gr_1": "1 grosz",   "gr_2": "2 grosze",  "gr_5": "5 groszy",
    "gr_10": "10 groszy", "gr_20": "20 groszy", "gr_50": "50 groszy",
    "zl_1": "1 zloty",   "zl_2": "2 zlote",   "zl_5": "5 zlotych",
    "ct_1": "1 cent",    "ct_2": "2 centy",   "ct_5": "5 centow",
    "ct_10": "10 centow", "ct_20": "20 centow", "ct_50": "50 centow",
    "e_1": "1 euro",     "e_2": "2 euro",
}

WARTOSCI_PLN = {
    "gr_1": 0.01, "gr_2": 0.02, "gr_5": 0.05,
    "gr_10": 0.10, "gr_20": 0.20, "gr_50": 0.50,
    "zl_1": 1.00, "zl_2": 2.00, "zl_5": 5.00,
}
WARTOSCI_EUR = {
    "ct_1": 0.01, "ct_2": 0.02, "ct_5": 0.05,
    "ct_10": 0.10, "ct_20": 0.20, "ct_50": 0.50,
    "e_1": 1.00,  "e_2": 2.00,
}

def label(cls):
    return LABELS.get(cls, cls)

def box_color(cls):
    if cls.startswith(("gr_", "zl_")):
        return (219, 112, 30)
    else:
        return (30, 180, 219)



# =====[ Funkcja predykcji ]=====

def predict_coin(img_bgr):
    img     = cv2.resize(img_bgr, (128, 128))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)
    features_scaled = scaler.transform([features])
    features_pca    = pca.transform(features_scaled)
    pred  = clf.predict(features_pca)[0]
    proba = clf.predict_proba(features_pca)[0]
    conf  = np.max(proba)
    return pred, conf



# =====[ Detekcja monet ]=====

# Skalujemy zdjęcie jeśli jest bardzo duże — Hough działa lepiej na mniejszych obrazach
MAX_DIM = 1600
h, w = frame.shape[:2]
scale = 1.0
if max(h, w) > MAX_DIM:
    scale = MAX_DIM / max(h, w)
    frame_small = cv2.resize(frame, (int(w * scale), int(h * scale)))
else:
    frame_small = frame.copy()

display = frame_small.copy()

gray  = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

# CLAHE — poprawia kontrast krawędzi (szczególnie ważne dla monet euro)
clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray_eq  = clahe.apply(gray)
blurred  = cv2.GaussianBlur(gray_eq, (9, 9), 2)

print("Wykrywanie monet...")

# param2=60 — wyższy próg = mniej fałszywych detekcji
# minDist=80 — monety muszą być oddalone od siebie o min. 80px
circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=80,
    param1=100,
    param2=60,
    minRadius=25,
    maxRadius=300
)

# Minimalny próg pewności — predykcje poniżej tego progu są odrzucane
MIN_CONF = 0.50



# =====[ Klasyfikacja i rysowanie wyników ]=====

results = []   # lista (x, y, r, pred, conf) do podsumowania
total_pln = 0.0
total_eur = 0.0

if circles is None:
    print("[UWAGA] Nie wykryto żadnych monet na zdjęciu.")
    print("Spróbuj zdjęcie z jednolitym tłem i dobrym oświetleniem.")
else:
    circles = np.round(circles[0, :]).astype(int)
    print(f"Wykryto {len(circles)} okrągłych obiektów (przed filtrem pewnosci).")

    for (x, y, r) in circles:
        margin = int(r * 0.15)
        x1 = max(0, x - r - margin)
        y1 = max(0, y - r - margin)
        x2 = min(frame_small.shape[1], x + r + margin)
        y2 = min(frame_small.shape[0], y + r + margin)

        roi = frame_small[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        pred, conf = predict_coin(roi)

        # Odrzucamy słabe predykcje — to główna przyczyna fałszywych "5 groszy"
        if conf < MIN_CONF:
            continue

        results.append((x, y, r, pred, conf))

        color = box_color(pred)

        # Okrąg wokół monety
        cv2.circle(display, (x, y), r, color, 3)
        cv2.circle(display, (x, y), 4, color, -1)

        # Etykieta nad monetą
        lbl_text  = label(pred)
        conf_text = f"{conf*100:.0f}%"

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, r / 80)   # rozmiar czcionki proporcjonalny do monety
        thickness  = 2

        (tw, th), _ = cv2.getTextSize(lbl_text, font, font_scale, thickness)
        (cw, _),  _ = cv2.getTextSize(conf_text, font, font_scale * 0.7, 1)

        box_w = max(tw, cw) + 16
        box_h = th + int(th * 1.4) + 12
        bx1 = x - box_w // 2
        bx2 = bx1 + box_w
        by1 = y - r - box_h - 6
        by2 = by1 + box_h

        # Tło etykiety z przezroczystością
        overlay = display.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, -1)
        cv2.addWeighted(overlay, 0.85, display, 0.15, 0, display)
        cv2.rectangle(display, (bx1, by1), (bx2, by2), (255, 255, 255), 1)

        cv2.putText(display, lbl_text,
                    (bx1 + 8, by1 + th + 4),
                    font, font_scale, (255, 255, 255), thickness)
        cv2.putText(display, conf_text,
                    (bx1 + 8, by2 - 4),
                    font, font_scale * 0.7, (220, 220, 220), 1)

        # Zliczamy wartość
        if pred in WARTOSCI_PLN:
            total_pln += WARTOSCI_PLN[pred]
        elif pred in WARTOSCI_EUR:
            total_eur += WARTOSCI_EUR[pred]



# =====[ Pasek podsumowania na dole ]=====

bar_h = 50
bar = np.zeros((bar_h, display.shape[1], 3), dtype=np.uint8)
bar[:] = (30, 30, 30)

n = len(results)
# cv2.putText nie obsługuje polskich znaków — używamy ASCII
summary = f"Wykryto: {n} monet"
if total_pln > 0:
    summary += f"  |  PLN: {total_pln:.2f} zl"
if total_eur > 0:
    summary += f"  |  EUR: {total_eur:.2f} EUR"

cv2.putText(bar, summary, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
display_final = np.vstack([display, bar])



# =====[ Zapis i wyświetlenie wyniku ]=====

# Nazwa pliku wyjściowego — dodajemy "_wynik" do nazwy oryginału
os.makedirs("wyniki", exist_ok=True)
base, ext = os.path.splitext(os.path.basename(img_path))
out_path = os.path.join("wyniki", base + "_wynik" + ext)
cv2.imwrite(out_path, display_final)
print(f"\nWynik zapisany: {out_path}")

# Podsumowanie w konsoli
print("\n===== PODSUMOWANIE =====")
for i, (x, y, r, pred, conf) in enumerate(results, 1):
    print(f"  Moneta {i:2d}: {label(pred):12s} ({conf*100:.1f}%)")
if total_pln > 0:
    print(f"\n  Łącznie PLN: {total_pln:.2f} zł")
if total_eur > 0:
    print(f"  Łącznie EUR: {total_eur:.2f} €")

# Wyświetlamy okno z wynikiem
cv2.imshow("Wynik klasyfikacji — dowolny klawisz aby zamknąć", display_final)
cv2.waitKey(0)
cv2.destroyAllWindows()
