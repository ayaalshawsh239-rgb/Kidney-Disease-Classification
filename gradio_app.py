"""
Kidney Disease Classification - Gradio Web Interface
Run with: python gradio_app.py
"""

import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load model
model = load_model('kidney_disease_model.keras')
print("✅ Model loaded successfully")

def predict(img):
    if img is None:
        return "⚠️ Please upload an image first"
    
    img = img.convert('RGB')
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    
    pred = model.predict(arr, verbose=0)[0][0]
    
    if pred > 0.5:
        return f"⚠️ Tumor\n\nConfidence: {pred*100:.2f}%"
    else:
        return f"✅ Normal\n\nConfidence: {(1-pred)*100:.2f}%"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="🖼️ Upload CT Scan Image"),
    outputs=gr.Textbox(label="📊 Result", lines=3),
    title="🩺 Kidney Disease Classification",
    description="Upload a kidney CT scan image to get instant diagnosis.",
)

if __name__ == "__main__":
    demo.launch(share=True)
