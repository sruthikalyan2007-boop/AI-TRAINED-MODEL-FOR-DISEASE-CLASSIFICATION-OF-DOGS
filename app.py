from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "dog_disease_model.h5"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

CLASS_NAMES = [
    "Dermatitis",
    "Fungal infections",
    "Healthy",
    "Hypersensitivity",
    "Demodicosis",
    "Ringworm",
]

PRECAUTIONS = {
    "Dermatitis": [
        "Keep the affected area clean and dry.",
        "Avoid new shampoos, foods, or irritants until a vet advises you.",
        "Arrange a veterinary examination to identify the underlying trigger.",
    ],
    "Fungal infections": [
        "Limit contact with other pets until a veterinarian confirms the cause.",
        "Wash bedding and frequently touched surfaces regularly.",
        "Book a veterinary visit for the correct antifungal treatment.",
    ],
    "Hypersensitivity": [
        "Note recent changes in food, treats, medication, or environment.",
        "Prevent scratching where possible and keep the skin clean.",
        "Ask a veterinarian about allergy testing and a treatment plan.",
    ],
    "Demodicosis": [
        "Schedule a veterinary examination promptly for skin testing.",
        "Do not use human or over-the-counter mite treatments without advice.",
        "Follow the complete treatment plan and monitor for secondary infection.",
    ],
    "Ringworm": [
        "Wash hands after handling your dog and limit close contact temporarily.",
        "Clean bedding and vacuum areas your dog uses frequently.",
        "See a veterinarian because ringworm can spread to people and pets.",
    ],
    "Healthy": [
        "Continue regular grooming and routine veterinary checkups.",
        "Watch for changes in skin, coat, appetite, or behavior.",
        "Seek veterinary advice if symptoms appear despite this result.",
    ],
}

model = tf.keras.models.load_model(MODEL_PATH)


def predict_image(file_storage):
    with Image.open(file_storage) as uploaded_image:
        prepared_image = uploaded_image.convert("RGB").resize((224, 224))
        image_array = np.asarray(prepared_image, dtype=np.float32) / 255.0

    predictions = model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0]
    winning_index = int(np.argmax(predictions))
    diagnosis = CLASS_NAMES[winning_index]
    return {
        "diagnosis": diagnosis,
        "is_healthy": diagnosis == "Healthy",
        "confidence": round(float(predictions[winning_index]) * 100, 1),
        "precautions": PRECAUTIONS[diagnosis],
        "scores": [
            {"name": name, "score": round(float(score) * 100, 1)}
            for name, score in zip(CLASS_NAMES, predictions)
        ],
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    filename = None

    if request.method == "POST":
        uploaded_file = request.files.get("image")
        if not uploaded_file or not uploaded_file.filename:
            error = "Choose a dog image before starting the analysis."
        else:
            try:
                result = predict_image(uploaded_file)
                filename = uploaded_file.filename
            except (OSError, ValueError) as exc:
                error = f"That file could not be analyzed: {exc}"

    return render_template("index.html", result=result, error=error, filename=filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
