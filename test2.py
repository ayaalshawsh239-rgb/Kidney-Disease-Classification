import cv2
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# =========================================================
# 1. قراءة البيانات التدريبية من ملف Excel المعالج
# =========================================================
excel_file = "extracted_features.xlsx"

try:
  df = pd.read_excel(excel_file)
except Exception as e:
  print(f"❌ لم يتم العثور على ملف الإكسل: {excel_file}")
  exit()

X = df.drop(columns=["Image_Name", "Category", "Label"]).values
y = df["Label"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================================================
# 2. إعداد وتدريب قائمة الخوارزميات
# =========================================================
models = {
    "1. Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    "2. Support Vector Machine (SVM)": SVC(kernel="rbf", probability=True),
    "3. K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
    "4. Decision Tree": DecisionTreeClassifier(random_state=42),
    "5. Quadratic Discriminant Analysis (QDA)": QuadraticDiscriminantAnalysis(),
}

print("جاري تدريب الخوارزميات على سمات الإكسل...")
for name, model in models.items():
  model.fit(X_scaled, y)


# =========================================================
# 3. دالة استخراج سمات الصورة الجديدة
# =========================================================
def extract_single_image_features(image):
  mean_b, mean_g, mean_r = cv2.mean(image)[:3]
  std_b, std_g, std_r = cv2.meanStdDev(image)[1].flatten()[:3]

  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  hist = cv2.calcHist([hsv], [0, 1, 2], None, (8, 8, 8), [0, 180, 0, 256, 0, 256])
  cv2.normalize(hist, hist)

  feats = [mean_b, mean_g, mean_r, std_b, std_g, std_r] + list(hist.flatten())
  return np.array(feats).reshape(1, -1)


# =========================================================
# 4. الاختبار التتابعي (خوارزمية تلو الأخرى)
# =========================================================
test_image_path = "/home/abdullah/Documents/0aame/COURSE-AI/4-image processing books/1-PRACTICAL/Dr.1/dog2.jpg"  # ضع مسار صورة الاختبار هنا
original_img = cv2.imread(test_image_path)

if original_img is None:
  print(f"❌ تعذر العثور على الصورة: {test_image_path}")
else:
  resized_img = cv2.resize(original_img, (128, 128))
  raw_features = extract_single_image_features(resized_img)
  scaled_features = scaler.transform(raw_features)

  labels_map = {0: "Cat (قطة)", 1: "Dog (كلب)"}
  total_models = len(models)

  print(
      "\n💡 اضغط على أي زر في لوحة المفاتيح للانتقال إلى الخوارزمية التالية...\n"
  )

  for index, (name, model) in enumerate(models.items(), 1):
    # التنبؤ
    pred = model.predict(scaled_features)[0]
    result_text = labels_map[pred]

    # حساب نسبة الثقة إن أمكن
    if hasattr(model, "predict_proba"):
      prob = model.predict_proba(scaled_features)[0][pred] * 100
      confidence_text = f"Confidence: {prob:.1f}%"
    else:
      confidence_text = "Confidence: N/A"

    # نسخ الصورة لإضافة الكتابة عليها
    display_img = original_img.copy()

    # شريط خلفية علوي لتسهيل قراءة النص
    cv2.rectangle(
        display_img, (0, 0), (display_img.shape[1], 90), (30, 30, 30), -1
    )

    # كتابة بيانات الخوارزمية والنتيجة
    cv2.putText(
        display_img,
        f"Model [{index}/{total_models}]: {name}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        display_img,
        f"Result: {result_text} | {confidence_text}",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
    )

    print(
        f"[{index}/{total_models}] {name} -> {result_text} ({confidence_text})"
    )

    # عرض نافذة الخوارزمية الحالية
    cv2.imshow("Multi-Model Sequential Test", display_img)

    # الانتظار حتى يضغط الطالب على أي زر لإغلاق النافذة والانتقال للتالية
    cv2.waitKey(0)
    cv2.destroyAllWindows()

  print("\n✅ اكتمل اختبار الصورة على جميع الخوارزميات!")