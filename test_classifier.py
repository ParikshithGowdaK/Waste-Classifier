import cv2
from app.classifier import WasteClassifier

# Define paths
model_path = "app/waste_classifier.h5"         # Path to the trained model
labels_path = "app/labels.json"     # Path to the label mapping
test_image_path = "test_samples/recyclable_sample.jpg"  # Your test image

# Initialize classifier
classifier = WasteClassifier(model_path, labels_path)

# Load test image
img = cv2.imread(test_image_path)

if img is None:
    print("Failed to load test image. Check the path.")
else:
    predicted_label, confidence = classifier.predict(img)
    print(f"Prediction: {predicted_label} ({confidence:.2f})")
