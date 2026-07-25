import tensorflow as tf
import numpy as np
import cv2

# -----------------------------
# Load Models
# -----------------------------

fake_model = tf.keras.models.load_model("fake_real_currency_model.h5")
value_model = tf.keras.models.load_model("currency_value_model.h5")

# Class order printed during training
classes = ['10','100','20','200','2000','50','500']

# -----------------------------
# Load Image
# -----------------------------

image_path = "test1.jpg"   # change image here
img = cv2.imread(image_path)

if img is None:
    print("Image not found")
    exit()

# Resize image
img = cv2.resize(img,(224,224))

# Normalize
img = img / 255.0

# Expand dimension
img = np.expand_dims(img,axis=0)

# -----------------------------
# Fake Detection
# -----------------------------

fake_pred = fake_model.predict(img)

print("Fake Prediction Score:", fake_pred)

if fake_pred[0][0] < 0.5:

    print("✅ Currency is REAL")

    # -----------------------------
    # Value Prediction
    # -----------------------------

    value_pred = value_model.predict(img)[0]

    print("\nPrediction Scores:")

    for i in range(len(classes)):
        print(classes[i],":",value_pred[i])

    value_index = np.argmax(value_pred)

    currency_value = classes[value_index]

    print("\n💰 Currency Value: ₹",currency_value)

else:

    print("❌ Currency is FAKE")