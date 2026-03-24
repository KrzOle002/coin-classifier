import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

dir = "dataset_out"

def load_data():
    images = []
    labels = []

    classes = os.listdir(dir)

    for label, c in enumerate(classes):
        class_path = os.path.join(dir, c)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            images.append(img)
            labels.append(label)

    return images, labels


def preprocess(img):
    img = cv2.resize(img, (128,128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = gray / 255.0
    return img

def augment(img):
    rows, cols = img.shape[:2]

    # rotacja
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 30, 1)
    rotated = cv2.warpAffine(img, M, (cols, rows))

    # flip
    flipped = cv2.flip(img, 1)

    return [img, rotated, flipped]

def extract_hog(img):
    features, _ = hog(img,
                      pixels_per_cell=(8,8),
                      cells_per_block=(2,2),
                      visualize=True)
    return features

def extract_lbp(img):
    lbp = local_binary_pattern(img, P=8, R=1)
    return lbp.flatten()

def extract_raw(img):
    return img.flatten()

def prepare_dataset():
    images, labels = load_data()

    X = []
    y = []

    for img, label in zip(images, labels):
        img = preprocess(img)

        feats = extract_hog(img)

        X.append(feats)
        y.append(label)

    return np.array(X), np.array(y)

def main():
    X, y = prepare_dataset()

    print("Shape X:", X.shape)
    print("Shape y:", y.shape)


if __name__ == "__main__":
    main()