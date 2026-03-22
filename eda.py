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
classes = os.listdir(dir)

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
        output_path = os.path.join(c_out, img_name)
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
plt.figure(figsize=(10,5))
plt.bar(counts.keys(), counts.values())
plt.xticks(rotation=90)
plt.title("Liczności klas")

plt.tight_layout()
plt.savefig("eda/licznosc_klas.png")
plt.close()

print("Zapisano wykres liczności klas!")



#=====[ Histogramy ]===== 

print("Tworzenie histogramów pikseli...")

#Dla każdej z klas tworzymy listę pikseli z obrazów
for c in classes:
    class_path = os.path.join(dir_out, c)
    all_pixels = []
    #Każdy obraz konwertujemy do grayscale i dodajemy jego piksele do listy
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        all_pixels.extend(gray.flatten())

    #Zapisujemy histogram na podstawie intensywności pikseli obrazów z danej klasy.
    plt.figure()

    plt.hist(all_pixels, bins=256)
    plt.title(f"Histogram pikseli - {c}")

    plt.xlabel("Intensywność")
    plt.ylabel("Liczba pikseli")

    plt.savefig(f"eda/histogram_{c}.png")
    plt.close()

print("Histogramy zapisane!")



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

        img = cv2.resize(img,(128,128))
        imgs.append(img)

    #Mając pełną listę generujemy "średnią grafikę". Jest to średnia wartość pikseli wszystkich obrazów.
    mean_img = np.mean(imgs, axis=0).astype("uint8")

    plt.figure()

    plt.imshow(cv2.cvtColor(mean_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Mean image - {c}")

    plt.savefig(f"eda/mean_{c}.png")
    plt.close()

print("Średnie obrazy zapisane!")



#=====[ Analiza rozmiarów ]===== 

print("Analizowanie rozmiarów obrazów...")

#Tworzymy listy długości i wysokości obrazów. Tym razem bierzemy pod uwagę wszystkie obrazy (bez podziału na klasy)
widths = []
heights = []

#Badamy rozmiary obrazów z każdej klasy
for c in classes:
    class_path = os.path.join(dir_out, c)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        #Zmienna "_" reprezentuje liczbę kanałów, która w tym wypadku nie jest nam potrzebna
        h, w, _ = img.shape
        widths.append(w)
        heights.append(h)

plt.figure()
plt.scatter(widths, heights, alpha=0.5)

plt.xlabel("Szerokość")
plt.ylabel("Wysokość")
plt.title("Rozmiary obrazów")

plt.savefig("eda/image_sizes.png")
plt.close()

print("Analiza zakończona!")



#=====[ Przykładowe monety ]===== 

print("Zapisywanie przykładowych obrazów...")

#Dla każdej klasy tworzymy losowo zestawienie 5 przykładowych obrazów.
for c in classes:
    class_path = os.path.join(dir_out, c)
    images = os.listdir(class_path)

    #Losowo wybieramy próbkę zestawu obrazów.
    sample = random.sample(images, min(5,len(images)))
    plt.figure(figsize=(10,2))

    #Składamy zestawienie obrazów z tej próbki
    for i,img_name in enumerate(sample):
        img_path = os.path.join(class_path,img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #Wybrane obrazy są rozmieszczane "w rzędzie".
        plt.subplot(1,5,i+1)
        plt.imshow(img)
        plt.axis("off")

    plt.suptitle(c)

    plt.savefig(f"eda/samples_{c}.png")
    plt.close()

print("Przykładowe obrazy zapisane")

print("Zakończono generowanie EDA.")