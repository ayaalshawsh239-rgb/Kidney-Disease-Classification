import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog

def extract_image_features(image):
    mean_b, mean_g, mean_r = cv2.mean(image)[:3]
    std_b, std_g, std_r = cv2.meanStdDev(image)[1].flatten()[:3]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, (8, 8, 8), [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    hist_features = hist.flatten()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2), block_norm="L2-Hys",
                       feature_vector=True)
    features = [mean_b, mean_g, mean_r, std_b, std_g, std_r]
    features += list(hist_features)
    features += list(hog_features)
    return features

dataset_path = "dataset"
categories = {"cats": 0, "dogs": 1}
records = []
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

for category, label in categories.items():
    folder_path = os.path.join(dataset_path, category)
    if not os.path.exists(folder_path):
        continue
    count = 0
    for img_name in os.listdir(folder_path):
        if not img_name.lower().endswith(VALID_EXTENSIONS):
            continue
        img_path = os.path.join(folder_path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.resize(image, (128, 128))
        feats = extract_image_features(image)
        row = {
            "Image_Name": img_name, "Category": category, "Label": label,
            "Mean_Blue": round(feats[0], 2), "Mean_Green": round(feats[1], 2),
            "Mean_Red": round(feats[2], 2), "Std_Blue": round(feats[3], 2),
            "Std_Green": round(feats[4], 2), "Std_Red": round(feats[5], 2),
        }
        for idx, v in enumerate(feats[6:518]):
            row[f"Hist_Bin_{idx+1}"] = round(v, 5)
        for idx, v in enumerate(feats[518:]):
            row[f"HOG_{idx+1}"] = round(float(v), 5)
        records.append(row)
        count += 1
    print(f"Done: {category} = {count} images")

df = pd.DataFrame(records)
df.to_excel("extracted_features.xlsx", index=False)
print(f"Total: {len(df)} images x {df.shape[1]} columns")
