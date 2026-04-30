import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import extract_contour_features, EURO_CLASSES, FEATURE_NAMES, ZONE_BOUNDS

dir_out = "dataset_out"
os.makedirs("edges", exist_ok=True)

classes = sorted(c for c in os.listdir(dir_out) if c in EURO_CLASSES)

print("Generowanie wizualizacji pipeline cech konturowych (tylko Euro)...")

# Kolory dla 6 stref
ZONE_COLORS_BGR = [
    (255,  50,  50),   # z1 centrum — ciemnoniebieski
    (255, 130,   0),   # z2
    (  0, 200,  80),   # z3
    (  0, 180, 200),   # z4
    (  0,  80, 255),   # z5
    (180,   0, 255),   # z6 obrzeże — fioletowy
]
ZONE_COLORS_RGB = [(r, g, b) for (b, g, r) in ZONE_COLORS_BGR]

for c in classes:
    class_path = os.path.join(dir_out, c)
    img_names  = os.listdir(class_path)
    if not img_names:
        continue

    img = cv2.imread(os.path.join(class_path, img_names[0]))
    if img is None:
        continue
    img = cv2.resize(img, (128, 128))

    gray    = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_u8 = gray if gray.dtype == np.uint8 else (gray * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(gray_u8, (7, 7), 1.5)
    edges   = cv2.Canny(blurred, 40, 120)

    h, w   = gray.shape
    cx, cy = w / 2.0, h / 2.0
    half_w = w / 2.0
    Y, X   = np.mgrid[0:h, 0:w]
    dist_norm = np.sqrt((X - cx)**2 + (Y - cy)**2) / half_w

    # --- Panel 3: HoughCircles ---
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=15,
        param1=60, param2=30, minRadius=10, maxRadius=60,
    )
    hough_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    n_c = 0
    if circles is not None:
        n_c = len(circles[0])
        for (x, y, r) in np.round(circles[0]).astype(int):
            cv2.circle(hough_vis, (x, y), r, (0, 255, 0), 2)
            cv2.circle(hough_vis, (x, y), 2,  (0, 0, 255), 3)

    # --- Panel 4: 6 stref radialnych (kolorowe piksele krawędzi) ---
    zone_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for zi, (r_lo, r_hi) in enumerate(ZONE_BOUNDS):
        mask = (dist_norm >= r_lo) & (dist_norm < r_hi) & (edges > 0)
        zone_vis[mask] = ZONE_COLORS_BGR[zi]
    for zi, (r_lo, r_hi) in enumerate(ZONE_BOUNDS):
        cv2.circle(zone_vis, (int(cx), int(cy)), int(r_hi * half_w), ZONE_COLORS_BGR[zi], 1)

    # --- Panel 5: Hierarchia RETR_TREE ---
    tree_c, hier = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    tree_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    n_holes = 0
    if hier is not None:
        for cnt, h_row in zip(tree_c, hier[0]):
            if h_row[3] != -1:
                cv2.drawContours(tree_vis, [cnt], -1, (0, 80, 255), 1)
                n_holes += 1
            else:
                cv2.drawContours(tree_vis, [cnt], -1, (0, 200, 80), 1)

    # --- Wykres ---
    fig, axes = plt.subplots(1, 5, figsize=(25, 4))
    fig.suptitle(f"Pipeline cech konturowych — klasa: {c}", fontsize=13)

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("1. Obraz wejściowy")

    axes[1].imshow(edges, cmap="gray")
    axes[1].set_title("2. Canny edges")

    axes[2].imshow(cv2.cvtColor(hough_vis, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"3. HoughCircles\n({n_c} okrągów)")

    axes[3].imshow(cv2.cvtColor(zone_vis, cv2.COLOR_BGR2RGB))
    axes[3].set_title("4. 6 stref radialnych\n(piksele Canny kolorowane wg strefy)")

    axes[4].imshow(cv2.cvtColor(tree_vis, cv2.COLOR_BGR2RGB))
    axes[4].set_title(f"5. RETR_TREE\nzielony=zewn., czerwony=dziury ({n_holes})")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"edges/pipeline_{c}.png", dpi=150)
    plt.close()
    print(f"  Zapisano: edges/pipeline_{c}.png")


# =====[ Zbiorczy: HoughCircles ]=====
print("\nZbiorczy: HoughCircles...")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("HoughCircles — monety Euro\n"
             "e_1/e_2 mają 2 okręgi (bimetalik), centy — 1", fontsize=13)

for ax, c in zip(axes.flatten(), classes):
    class_path = os.path.join(dir_out, c)
    img_names  = os.listdir(class_path)
    if not img_names:
        ax.axis("off"); continue
    img = cv2.imread(os.path.join(class_path, img_names[0]))
    if img is None:
        ax.axis("off"); continue
    img     = cv2.resize(img, (128, 128))
    gray    = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=15,
                                param1=60, param2=30, minRadius=10, maxRadius=60)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    n_c = 0
    if circles is not None:
        n_c = len(circles[0])
        for (x, y, r) in np.round(circles[0]).astype(int):
            cv2.circle(vis, (x, y), r, (0, 255, 0), 2)
            cv2.circle(vis, (x, y), 2,  (0, 0, 255), 3)
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{c}  ({n_c} okrągów)", fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.savefig("edges/hough_circles_wszystkie_klasy.png", dpi=150)
shutil.copy("edges/hough_circles_wszystkie_klasy.png", "edges/canny_hog_wszystkie_klasy.png")
plt.close()
print("  Zapisano: edges/hough_circles_wszystkie_klasy.png")


# =====[ Zbiorczy: 6 stref radialnych ]=====
print("Zbiorczy: strefy radialne...")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Profil krawędziowy — 6 stref radialnych\n"
             "Piksele Canny kolorowane wg strefy (centrum → obrzeże)", fontsize=13)

for ax, c in zip(axes.flatten(), classes):
    class_path = os.path.join(dir_out, c)
    img_names  = os.listdir(class_path)
    if not img_names:
        ax.axis("off"); continue
    img = cv2.imread(os.path.join(class_path, img_names[0]))
    if img is None:
        ax.axis("off"); continue
    img     = cv2.resize(img, (128, 128))
    gray    = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_u8 = gray if gray.dtype == np.uint8 else (gray * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(gray_u8, (7, 7), 1.5)
    edges   = cv2.Canny(blurred, 40, 120)
    h2, w2  = gray.shape
    cx2, cy2 = w2 / 2.0, h2 / 2.0
    hw2 = w2 / 2.0
    Y2, X2 = np.mgrid[0:h2, 0:w2]
    dn = np.sqrt((X2 - cx2)**2 + (Y2 - cy2)**2) / hw2

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for zi, (r_lo, r_hi) in enumerate(ZONE_BOUNDS):
        mask = (dn >= r_lo) & (dn < r_hi) & (edges > 0)
        vis[mask] = ZONE_COLORS_BGR[zi]
    for zi, (r_lo, r_hi) in enumerate(ZONE_BOUNDS):
        cv2.circle(vis, (int(cx2), int(cy2)), int(r_hi * hw2), ZONE_COLORS_BGR[zi], 1)

    feats = extract_contour_features(img)
    edge_vals = feats[2:8]
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{c}\nz1={edge_vals[0]:.2f} z6={edge_vals[5]:.2f}", fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.savefig("edges/strefy_radialne_wszystkie_klasy.png", dpi=150)
shutil.copy("edges/strefy_radialne_wszystkie_klasy.png", "edges/canny_wszystkie_klasy.png")
plt.close()
print("  Zapisano: edges/strefy_radialne_wszystkie_klasy.png")


# =====[ Srednie wartości 12 cech per klasa ]=====
print("Srednie cechy per klasa...")
class_feats_mean = {}
for c in classes:
    class_path = os.path.join(dir_out, c)
    flist = []
    for img_name in os.listdir(class_path):
        img = cv2.imread(os.path.join(class_path, img_name))
        if img is None:
            continue
        img = cv2.resize(img, (128, 128))
        flist.append(extract_contour_features(img))
    if flist:
        class_feats_mean[c] = np.mean(flist, axis=0)

colors_cls = plt.cm.tab10(np.linspace(0, 1, len(classes)))
color_map  = {c: col for c, col in zip(classes, colors_cls)}

fig, axes = plt.subplots(3, 4, figsize=(16, 10))
fig.suptitle("Srednie wartości 12 cech konturowych — porównanie klas Euro", fontsize=13)
group_bg = ["#d0e8ff"] * 2 + ["#d0ffd0"] * 6 + ["#ffd0d0"] * 2 + ["#fff0d0"] * 2

for fi, feat_name in enumerate(FEATURE_NAMES):
    ax = axes[fi // 4][fi % 4]
    vals = [class_feats_mean[c][fi] for c in class_feats_mean]
    ax.bar(list(class_feats_mean.keys()), vals,
           color=[color_map[c] for c in class_feats_mean], alpha=0.85)
    ax.set_facecolor(group_bg[fi])
    ax.set_title(feat_name, fontsize=9)
    ax.set_xticklabels(list(class_feats_mean.keys()), rotation=45, fontsize=7, ha="right")
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("edges/srednie_cechy_konturowe.png", dpi=150)
plt.close()
print("  Zapisano: edges/srednie_cechy_konturowe.png")

print("\nZakończono generowanie wizualizacji.")
