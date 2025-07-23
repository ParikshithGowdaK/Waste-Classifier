import numpy as np
import json
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class WasteClassifier:
    def __init__(self, model_path, labels_path):
        self.model = load_model(model_path)

        with open(labels_path, "r") as f:
            self.class_indices = json.load(f)

        # Create reverse mapping: 0 -> "Recyclable", etc.
        self.index_to_label = {v: k for k, v in self.class_indices.items()}

    def preprocess(self, img):
        img_resized = cv2.resize(img, (224, 224))  # Resize to model input
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
        return img_array

    def predict(self, img):
        processed = self.preprocess(img)
        preds = self.model.predict(processed)
        class_idx = np.argmax(preds, axis=1)[0]
        label = self.index_to_label[class_idx]
        confidence = float(np.max(preds))
        return label, confidence
