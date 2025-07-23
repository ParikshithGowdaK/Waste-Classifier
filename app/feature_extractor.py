import os
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tqdm import tqdm

# Load your trained model
model = load_model("app/waste_classifier.h5")

# Remove final classification layer to get feature extractor
feature_model = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

# Path to dataset
dataset_dir = "dataset"
features = []

for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_dir):
        continue

    for img_file in tqdm(os.listdir(class_dir), desc=f"Processing {class_name}"):
        img_path = os.path.join(class_dir, img_file)
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            feature_vector = feature_model.predict(img_array)[0]
            features.append((feature_vector, img_path, class_name))
        except:
            print(f"Skipped: {img_path}")

# Save features to a pickle file
with open("app/features.pkl", "wb") as f:
    pickle.dump(features, f)

print("[✅] Feature vectors saved to app/features.pkl")
