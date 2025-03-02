import os
import logging
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)

# Load Model
MODEL_PATH = "models/eye_disease_model.h5"
logging.info(f"Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)
logging.info("Model loaded successfully!")
CATEGORIES = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # Process Image
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Make Prediction
            prediction = model.predict(img_array)
            result = CATEGORIES[np.argmax(prediction)]

            logging.info(f"Prediction: {result}")

            return render_template("result.html", result=result, img_path=filepath)

    return render_template("index.html")

if __name__ == "__main__":
    logging.info("Starting Flask app...")
    app.run(debug=True)
