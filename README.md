# 🐱🐶 Cat vs. Dog Image Classification & Model Benchmarking

A machine-learning project that uses **OpenCV** and **Scikit-Learn** to classify cat and dog images based on handcrafted visual features. The project extracts color-based and shape-based features, applies **StandardScaler** and **Principal Component Analysis (PCA)** for dimensionality reduction, and trains multiple classical machine-learning classifiers for comparison.

The complete pipeline covers **feature extraction → preprocessing → dimensionality reduction → model training → evaluation → visual prediction**.

---

## 📌 Features

- **Feature Extraction:** Extracts color statistics (Mean BGR, Standard Deviation, and 3D HSV Color Histograms) combined with **HOG (Histogram of Oriented Gradients)** descriptors for shape and edge information.

- **Preprocessing & Dimensionality Reduction:** Applies `StandardScaler` followed by **PCA (Principal Component Analysis)** to reduce feature dimensionality and mitigate the curse of dimensionality.

- **Multi-Model Benchmarking:** Trains and compares four classical machine-learning classifiers:
  - Random Forest Classifier
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Decision Tree Classifier
  - **Voting Classifier (soft voting)** — combines the strongest base models to improve overall accuracy and stability.

- **Model Evaluation:** Evaluates classifiers using **Accuracy, Precision, Recall, and F1-Score** via **Stratified K-Fold Cross-Validation (5-fold)** for more reliable results than a single train/test split.

- **Visual Inference:** Uses OpenCV to display predicted classes and predicted class probabilities directly on test images.

---

## 🛠️ Technologies Used

- Python 3
- OpenCV (cv2)
- NumPy
- Pandas
- Scikit-Learn
- Scikit-Image (skimage) — for HOG features
- Matplotlib

---

## 📁 Project Structure
