import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

MODEL_PATH = 'models/model.h5'
CLASS_INDICES_PATH = 'models/class_indices.json'
IMG_SIZE = 224


if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
else:
    print(f"❌ Model file not found at {MODEL_PATH}")
    exit()


if os.path.exists(CLASS_INDICES_PATH):
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    label_map = {v: k for k, v in class_indices.items()}
else:
    print(f"❌ class_indices.json not found at: {CLASS_INDICES_PATH}")
    exit()


def predict_disease(img_path):
    if not os.path.exists(img_path):
        print(f"❌ Image not found at {img_path}")
        return

    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    pred_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    predicted_class = label_map.get(pred_index, "Unknown")


    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.2%}")
    plt.show()

    print("\n🎯 Prediction Result")
    print("----------------------------")
    print(f"🌿 Disease      : {predicted_class}")
    print(f"📈 Confidence   : {confidence:.2%}")
    print("----------------------------")


if __name__ == "__main__":
    print("🌿 Crop Disease Prediction 🌿")
    img_path = input("📸 Enter path to image: ").strip()
    predict_disease(img_path)
