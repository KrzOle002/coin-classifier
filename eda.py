import os
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import random

#Folder z klasami
dir = "dataset"

#Folder po ujednoliceniu
dir_out = "dataset_out"
os.makedirs(dir_out, exist_ok=True)

#Podfoldery
classes = sorted(os.listdir(dir))

#Folder, do którego będziemy zapisywać wykresy wynikowe
os.makedirs("eda", exist_ok=True)



#=====[ Ujednolicenie ]=====
print("Ujednolicanie klas...")

#Tworzymy foldery dla ujednoliconych klas
for c in classes:
    c_in = os.path.join(dir, c)
    c_out = os.path.join(dir_out, c)
    os.makedirs(c_out, exist_ok=True)

    #Zmieniamy rozmiary obrazów na 128x128
    for img_name in os.listdir(c_in):
        img_path = os.path.join(c_in, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #Standaryzacja nazwy + wymuszenie formatu PNG
        name = os.path.splitext(img_name)[0]
        name = name.lower().replace(" ", "_")

        output_path = os.path.join(c_out, name + ".png")
        cv2.imwrite(output_path, img)

print("Ujednolicanie zakończone!")



print("\nEDA")

#=====[ Liczność klas ]=====
print("Obliczanie liczności klas...")

#Klucz - klasa
#Wartość - liczność klasy
counts = {}

#Dla każdej klasy liczymy ilość obrazów
for c in classes:
    class_path = os.path.join(dir_out, c)
    images = [
        f for f in os.listdir(class_path)
        if os.path.isfile(os.path.join(class_path, f))
    ]
    counts[c] = len(images)

#Zapis do wykresu
plt.figure(figsize=(12, 5))
colors = ["steelblue" if c.startswith("ct_") else "darkorange" for c in counts.keys()]
bars = plt.bar(counts.keys(), counts.values(), color=colors)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             str(int(bar.get_height())), ha="center", va="bottom", fontsize=8)
plt.xticks(rotation=90)
plt.title("Licznosci klas (niebieski = centy, pomaranczowy = euro)")
plt.ylabel("Liczba obrazow")
plt.tight_layout()
plt.savefig("eda/licznosc_klas.png")
plt.close()

print("Zapisano wykres liczności klas!")



#=====[ Histogramy — grayscale ]=====

print("Tworzenie histogramów pikseli...")

#Dla każdej z klas tworzymy histogram grayscale
for c in classes:
    class_path = os.path.join(dir_out, c)
    all_pixels_gray = []

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        all_pixels_gray.extend(gray.flatten())

    plt.figure()
    plt.hist(all_pixels_gray, bins=256, color="gray")
    plt.title(f"Histogram pikseli (grayscale) - {c}")
    plt.xlabel("Intensywnosc")
    plt.ylabel("Liczba pikseli")
    plt.savefig(f"eda/histogram_{c}.png")
    plt.close()

print("Histogramy zapisane!")

#Histogramy RGB

# print("Tworzenie histogramów RGB...")

# for c in classes:
#     class_path = os.path.join(dir_out, c)

#     r_vals, g_vals, b_vals = [], [], []

#     for img_name in os.listdir(class_path):
#         img_path = os.path.join(class_path, img_name)
#         img = cv2.imread(img_path)
#         if img is None:
#             continue

#         b, g, r = cv2.split(img)
#         r_vals.extend(r.flatten())
#         g_vals.extend(g.flatten())
#         b_vals.extend(b.flatten())

#     plt.figure()
#     plt.hist(r_vals, bins=256, color='r', alpha=0.5, label='R')
#     plt.hist(g_vals, bins=256, color='g', alpha=0.5, label='G')
#     plt.hist(b_vals, bins=256, color='b', alpha=0.5, label='B')
#     plt.legend()

#     plt.title(f"Histogram RGB - {c}")
#     plt.xlabel("Intensywnosc")
#     plt.ylabel("Liczba pikseli")

#     plt.savefig(f"eda/hist_rgb_{c}.png")
#     plt.close()

# print("Histogramy RGB zapisane!")



#=====[ Średni obraz dla klasy ]=====

print("Tworzenie średnich obrazów...")

#Dla każdej klasy tworzymy listę wszystkich obrazów.
for c in classes:
    class_path = os.path.join(dir_out, c)
    imgs = []
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (128, 128))
        imgs.append(img)

    #Mając pełną listę generujemy "średnią grafikę". Jest to średnia wartość pikseli wszystkich obrazów.
    mean_img = np.mean(imgs, axis=0).astype("uint8")
    mean_gray = cv2.cvtColor(mean_img, cv2.COLOR_BGR2GRAY)

    #Mean RGB (opcjonalnie)
    plt.figure()
    plt.imshow(cv2.cvtColor(mean_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Mean RGB - {c}")
    plt.savefig(f"eda/mean_rgb_{c}.png")
    plt.close()

    plt.figure()

    plt.imshow(mean_gray, cmap="gray")
    plt.axis("off")
    plt.title(f"Mean image - {c}")

    plt.savefig(f"eda/mean_{c}.png")
    plt.close()

print("Średnie obrazy zapisane!")



# =====[ Analiza rozmiarów ]=====

# print("Analizowanie rozmiarów obrazów...")

# #Tworzymy listy długości i wysokości obrazów. Tym razem bierzemy pod uwagę wszystkie obrazy (bez podziału na klasy)
# widths = []
# heights = []

# #Badamy rozmiary obrazów z każdej klasy
# for c in classes:
#     class_path = os.path.join(dir_out, c)
#     for img_name in os.listdir(class_path):
#         img_path = os.path.join(class_path, img_name)
#         img = cv2.imread(img_path)
#         if img is None:
#             continue

#         h, w = img.shape[:2]
#         widths.append(w)
#         heights.append(h)

# plt.figure()
# plt.scatter(widths, heights, alpha=0.5)

# plt.xlabel("Szerokosc")
# plt.ylabel("Wysokosc")
# plt.title("Rozmiary obrazow")

# plt.savefig("eda/image_sizes.png")

# #Histogram szerokości
# plt.figure()
# plt.hist(widths, bins=20)
# plt.xlabel("Szerokosc")
# plt.ylabel("Liczba obrazow")
# plt.title("Histogram szerokosci obrazow")
# plt.savefig("eda/image_widths_hist.png")
# plt.close()

# #Histogram wysokości
# plt.figure()
# plt.hist(heights, bins=20)
# plt.xlabel("Wysokosc")
# plt.ylabel("Liczba obrazow")
# plt.title("Histogram wysokosci obrazow")
# plt.savefig("eda/image_heights_hist.png")
# plt.close()

# plt.close()

# print("Analiza zakończona!")



#=====[ Przykładowe monety ]=====

print("Zapisywanie przykładowych obrazów...")

#Dla każdej klasy tworzymy losowo zestawienie 5 przykładowych obrazów.
for c in classes:
    class_path = os.path.join(dir_out, c)
    images = os.listdir(class_path)

    #Losowo wybieramy próbkę zestawu obrazów.
    sample = random.sample(images, min(5, len(images)))
    plt.figure(figsize=(10, 2))

    #Składamy zestawienie obrazów z tej próbki
    for i, img_name in enumerate(sample):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #Wybrane obrazy są rozmieszczane "w rzędzie".
        plt.subplot(1, 5, i+1)
        plt.imshow(gray, cmap="gray")
        plt.axis("off")

    plt.suptitle(c)

    plt.savefig(f"eda/samples_{c}.png")
    plt.close()

print("Przykładowe obrazy zapisane")



#=====[ Wizualizacja HoughCircles ]=====

print("Tworzenie wizualizacji wykrytych okręgów...")

#Dla każdej klasy bierzemy pierwsze dostępne zdjęcie i rysujemy wykryte okręgi
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for ax, c in zip(axes.flatten(), classes):
    class_path = os.path.join(dir_out, c)
    img_name = os.listdir(class_path)[0]
    img = cv2.imread(os.path.join(class_path, img_name))
    if img is None:
        ax.axis("off")
        continue

    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=15,
        param1=60, param2=30,
        minRadius=10, maxRadius=60,
    )

    # Rysujemy wykryte okręgi na kopii obrazu
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    n_circles = 0
    if circles is not None:
        circles_int = np.round(circles[0]).astype(int)
        n_circles = len(circles_int)
        for (x, y, r) in circles_int:
            cv2.circle(vis, (x, y), r, (0, 255, 0), 2)
            cv2.circle(vis, (x, y), 2, (0, 0, 255), 3)

    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{c}\n{n_circles} okręgów", fontsize=9)
    ax.axis("off")

plt.suptitle("Wykryte okręgi HoughCircles dla każdej klasy", fontsize=13)
plt.tight_layout()
plt.savefig("eda/hough_circles.png", dpi=150)
plt.close()

print("Zapisano wizualizację HoughCircles!")

print("Zakończono generowanie EDA.")