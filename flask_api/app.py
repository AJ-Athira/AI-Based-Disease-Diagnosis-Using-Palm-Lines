import os
import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__, template_folder="../frontend/templates", static_folder="../frontend/static")

# Load trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/disease_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (Ensure this matches model's output layer size)
class_labels = [
    "Stroke Risk", "High Blood Pressure", "Cardiovascular Issues", 
    "Migraine", "Mental Health Disorder", "Cognitive Decline", "Dementia"
]  # ⚠️ Check if these are correct!

# Route: Home Page
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# Route: Predict Disease from Palm Image
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files["image"]
    if img_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read image as BytesIO
        img_bytes = io.BytesIO(img_file.read())

        # Load and preprocess the image
        img = image.load_img(img_bytes, target_size=(224, 224))  # Match model input size
        img_array = image.img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Expand for batch

        # Predict disease
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Ensure predicted_class is within range of class_labels
        if predicted_class >= len(class_labels):
            return jsonify({"error": "Predicted class index out of range"}), 500

        disease_name = class_labels[predicted_class]
        confidence = float(predictions[0][predicted_class])  # Convert confidence score

        return jsonify({"predicted_disease": disease_name, "confidence": confidence})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
