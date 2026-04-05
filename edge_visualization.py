import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Folder z ujednoliconymi obrazami
dir_out = "dataset_out"
os.makedirs("edges", exist_ok=True)

classes = sorted(os.listdir(dir_out))

print("Generowanie wizualizacji krawędzi...")

for c in classes:
    class_path = os.path.join(dir_out, c)
    img_names = os.listdir(class_path)
    if not img_names:
        continue

    # Bierzemy pierwsze dostępne zdjęcie z klasy
    img_path = os.path.join(class_path, img_names[0])
    img = cv2.imread(img_path)
    if img is None:
        continue

    img = cv2.resize(img, (128, 128))

    # =====[ Krok 1: Obraz wejściowy (grayscale) ]=====
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # =====[ Krok 2: Rozmycie Gaussian ]=====
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

    # =====[ Krok 3: Krawędzie Canny ]=====
    edges = cv2.Canny(blurred, 40, 120)

    # =====[ Krok 4: Kontury na obrazie ]=====
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 1)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(contour_vis, [largest], -1, (0, 0, 255), 2)

    # =====[ Krok 5: Okręgi HoughCircles ]=====
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=15,
        param1=60, param2=30,
        minRadius=10, maxRadius=60,
    )
    circle_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    n_circles = 0
    if circles is not None:
        circles_int = np.round(circles[0]).astype(int)
        n_circles = len(circles_int)
        for (x, y, r) in circles_int:
            cv2.circle(circle_vis, (x, y), r, (0, 255, 0), 2)
            cv2.circle(circle_vis, (x, y), 2, (0, 0, 255), 3)

    # =====[ Wykres — 5 kroków obok siebie ]=====
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle(f"Pipeline wykrywania krawędzi — klasa: {c}", fontsize=13)

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("1. Obraz wejściowy\n(grayscale)")

    axes[1].imshow(blurred, cmap="gray")
    axes[1].set_title("2. Gaussian Blur\n(redukcja szumu)")

    axes[2].imshow(edges, cmap="gray")
    axes[2].set_title("3. Canny\n(krawędzie)")

    axes[3].imshow(cv2.cvtColor(contour_vis, cv2.COLOR_BGR2RGB))
    axes[3].set_title(f"4. Kontury\n(zielony=wszystkie, czerwony=największy)")

    axes[4].imshow(cv2.cvtColor(circle_vis, cv2.COLOR_BGR2RGB))
    axes[4].set_title(f"5. HoughCircles\n({n_circles} okręgów)")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"edges/pipeline_{c}.png", dpi=150)
    plt.close()
    print(f"Zapisano: edges/pipeline_{c}.png")

# =====[ Zbiorczy wykres wszystkich klas — samo Canny ]=====
print("\nGenerowanie zbiorczego wykresu Canny...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Krawędzie Canny dla wszystkich klas", fontsize=14)

for ax, c in zip(axes.flatten(), classes):
    class_path = os.path.join(dir_out, c)
    img_names = os.listdir(class_path)
    if not img_names:
        ax.axis("off")
        continue

    img = cv2.imread(os.path.join(class_path, img_names[0]))
    if img is None:
        ax.axis("off")
        continue

    img = cv2.resize(img, (128, 128))
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
    edges = cv2.Canny(blurred, 40, 120)

    ax.imshow(edges, cmap="gray")
    ax.set_title(c, fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig("edges/canny_wszystkie_klasy.png", dpi=150)
plt.close()
print("Zapisano: edges/canny_wszystkie_klasy.png")

print("\nZakończono generowanie wizualizacji krawędzi.")
print("Wyniki w folderze: edges/")
