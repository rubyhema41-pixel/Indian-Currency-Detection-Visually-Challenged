import tensorflow as tf
from ultralytics import YOLO
import numpy as np
import cv2
import gradio as gr

# -----------------------------
# Load Models
# -----------------------------

fake_model = tf.keras.models.load_model(
    "D:/Dataset/Currency_detection/fake_currency_detection.h5"
)

value_model = YOLO(
    "D:/Dataset/Currency_detection/runs/currency_model/weights/best.pt"
)

# -----------------------------
# Prediction Function
# -----------------------------

def predict_currency(image):

    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Prepare image for fake model
    img_fake = cv2.resize(img, (224,224))
    img_fake = img_fake / 255.0
    img_fake = np.expand_dims(img_fake, axis=0)

    # -----------------------------
    # Fake Detection
    # -----------------------------

    fake_pred = fake_model.predict(img_fake)

    if fake_pred[0][0] > 0.5:
        return "❌ FAKE Currency"

    # -----------------------------
    # Real Currency → Value Detection
    # -----------------------------

    results = value_model(img)

    if len(results[0].boxes) == 0:
        return "⚠ Currency detected but value unclear"

    box = results[0].boxes[0]

    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    value = value_model.names[class_id]

    return f"✅ REAL Currency\n💰 Value: ₹{value}\nConfidence: {confidence:.2f}"


# -----------------------------
# Gradio Interface
# -----------------------------

interface = gr.Interface(
    fn=predict_currency,
    inputs=gr.Image(type="numpy", label="Upload Currency Image"),
    outputs=gr.Textbox(label="Detection Result"),
    title="Fake Currency Detection & Value Reader",
    description="Detect Fake/Real currency and read the note value."
)

# -----------------------------
# Launch UI
# -----------------------------

interface.launch()