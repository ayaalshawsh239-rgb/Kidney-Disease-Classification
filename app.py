import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, request, render_template_string
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = tf.keras.models.load_model('kidney_disease_model.keras')

HTML_PAGE = """
<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <title>Kidney Disease Classification</title>
</head>
<body style="font-family: Arial; text-align: center; padding: 40px;">
    <h1>Kidney Disease Classification</h1>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <br><br>
        <button type="submit">Analyze Image</button>
    </form>
    {% if result %}
        <h2>Result: {{ result }} (Confidence: {{ confidence }}%)</h2>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return render_template_string(HTML_PAGE)
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize((224, 224))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        pred = model.predict(arr, verbose=0)[0][0]
        if pred > 0.5:
            result = "Tumor"
            confidence = f"{pred*100:.2f}"
        else:
            result = "Normal"
            confidence = f"{(1-pred)*100:.2f}"
        return render_template_string(HTML_PAGE, result=result, confidence=confidence)
    return render_template_string(HTML_PAGE)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
