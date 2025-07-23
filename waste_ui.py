import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- Load model and feature extractor ---
model = load_model("app/waste_classifier.h5")
feature_extractor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

# --- Training class order used ---
class_order = ['Harmful', 'Organic', 'Recyclable', 'Residual']

# --- Load feature vectors ---
with open("app/features.pkl", "rb") as f:
    features = pickle.load(f)
feature_vectors = np.array([f[0] for f in features])
image_paths = [f[1] for f in features]
class_names = [f[2] for f in features]

# --- Load bin images ---
bin_mapping = {
    "recyclable": "bin_images/recyclable_bin.png",
    "organic": "bin_images/organic_bin.png",
    "residual": "bin_images/residual_bin.png",
    "harmful": "bin_images/harmful_bin.png"
}
bin_images = {}
for label, path in bin_mapping.items():
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Warning: Could not load bin image for {label}")
    bin_images[label] = img

# --- Load overlay images ---
background = cv2.imread("background.png")
arrow = cv2.imread("arrow.png", cv2.IMREAD_UNCHANGED)

# --- Preprocessing function ---
def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# --- Prediction and similarity logic ---
def get_prediction_and_similar(frame):
    processed = preprocess(frame)
    pred = model.predict(processed)
    class_index = np.argmax(pred)
    label = class_order[class_index].lower()  # align with bin_mapping keys

    # Feature similarity
    feat = feature_extractor.predict(processed)[0].reshape(1, -1)
    sims = cosine_similarity(feat, feature_vectors)
    best_idx = np.argmax(sims)
    similar_img = cv2.imread(image_paths[best_idx])

    return label, bin_images[label], similar_img

# --- Webcam and display loop ---
cap = cv2.VideoCapture(0)

# Constants for placement
cam_x, cam_y = 159, 148
cam_w, cam_h = 454, 340

sim_x, sim_y = 909, 127
bin_x, bin_y = 895, 374
icon_size = 100  # for bin/similar/arrow image

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        label, bin_img, sim_img = get_prediction_and_similar(frame)
    except Exception as e:
        print(f"[ERROR] {e}")
        continue

    display = background.copy()

    # Place webcam feed
    cam_frame = cv2.resize(frame, (cam_w, cam_h))
    display[cam_y:cam_y+cam_h, cam_x:cam_x+cam_w] = cam_frame

    # Place similar image
    if sim_img is not None:
        sim_resized = cv2.resize(sim_img, (icon_size, icon_size))
        display[sim_y:sim_y+icon_size, sim_x:sim_x+icon_size] = sim_resized

    # Place bin image
    if bin_img is not None:
        bin_resized = cv2.resize(bin_img, (icon_size, icon_size))
        if bin_resized.shape[2] == 4:
            alpha = bin_resized[:, :, 3] / 255.0
            for c in range(3):
                display[bin_y:bin_y+icon_size, bin_x:bin_x+icon_size, c] = (
                    alpha * bin_resized[:, :, c] +
                    (1 - alpha) * display[bin_y:bin_y+icon_size, bin_x:bin_x+icon_size, c]
                )
        else:
            display[bin_y:bin_y+icon_size, bin_x:bin_x+icon_size] = bin_resized

    # Place arrow exactly between bin and similar
    center_x = (sim_x + bin_x) // 2
    center_y = (sim_y + bin_y) // 2
    if arrow is not None:
        arrow_resized = cv2.resize(arrow, (icon_size, icon_size))
        if arrow_resized.shape[2] == 4:
            alpha = arrow_resized[:, :, 3] / 255.0
            for c in range(3):
                display[center_y:center_y+icon_size, center_x:center_x+icon_size, c] = (
                    alpha * arrow_resized[:, :, c] +
                    (1 - alpha) * display[center_y:center_y+icon_size, center_x:center_x+icon_size, c]
                )
        else:
            display[center_y:center_y+icon_size, center_x:center_x+icon_size] = arrow_resized

    # Show display
    cv2.imshow("Waste Classifier UI", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
