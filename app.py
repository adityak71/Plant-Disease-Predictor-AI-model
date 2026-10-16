import json
import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "plant_disease_model.keras"
CLASS_INDICES_PATH = BASE_DIR / "models" / "class_indices.json"
IMG_HEIGHT, IMG_WIDTH = 224, 224

model = None
if MODEL_PATH.exists():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        app.logger.info("Model loaded successfully from %s", MODEL_PATH)
    except Exception:
        app.logger.exception("Unable to load model from %s", MODEL_PATH)

class_names = {}
if CLASS_INDICES_PATH.exists():
    try:
        with CLASS_INDICES_PATH.open(encoding="utf-8") as file:
            indices = json.load(file)
        class_names = {int(index): name for name, index in indices.items()}
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        app.logger.exception("Unable to load class indices from %s", CLASS_INDICES_PATH)


def prepare_image(img_file):
    """Validate, resize, and normalize an uploaded image."""
    with Image.open(img_file) as source:
        source.verify()
    img_file.seek(0)
    with Image.open(img_file) as source:
        image = source.convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
    return np.expand_dims(np.array(image) / 255.0, axis=0)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No selected file"}), 400

    if model is None or not class_names:
        return jsonify({
            "error": "Prediction service is not ready. Train the model and ensure class_indices.json exists."
        }), 503

    try:
        predictions = model.predict(prepare_image(file), verbose=0)
        predicted_class_index = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions))
        predicted_disease_name = class_names.get(predicted_class_index)

        if predicted_disease_name is None:
            return jsonify({"error": "The model output does not match the class mapping."}), 500

        if confidence < 0.50:
            return jsonify({
                "disease": "Not Recognized or Low Confidence",
                "confidence": f"{confidence * 100:.2f}%",
                "message": "The model is not confident enough to classify this correctly.",
            })

        return jsonify({
            "disease": predicted_disease_name.replace("___", " - ").replace("_", " "),
            "confidence": f"{confidence * 100:.2f}%",
        })
    except (OSError, ValueError, SyntaxError):
        app.logger.exception("Unable to process uploaded image")
        return jsonify({"error": "The uploaded file is not a valid image."}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Please try again."}), 500


@app.errorhandler(413)
def request_entity_too_large(_error):
    return jsonify({"error": "Image is too large. Maximum size is 10 MB."}), 413


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "").lower() == "true")
