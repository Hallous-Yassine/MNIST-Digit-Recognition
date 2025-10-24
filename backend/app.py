"""
app.py
------
Flask API for MNIST handwritten digit recognition.
Works with a frontend drawing canvas or image upload by expecting 
a JSON payload containing base64 Data URL image data.
"""

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import base64 
from flask_cors import CORS
import sys 

# -----------------------------
# Flask setup
# -----------------------------
app = Flask(__name__)
# Enable CORS to allow your frontend (running on a different port) to access the API
CORS(app) 

# -----------------------------
# Load trained model
# -----------------------------
# Define model path relative to this script's location
MODEL_FILENAME = 'mnist_cnn_enhanced.h5'
# Assumes the structure: /app.py -> /model/mnist_cnn.h5
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', MODEL_FILENAME)

# Check if the model exists before trying to load it
if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: Model file not found at {MODEL_PATH}")
    print("Please ensure you have run 'python model/train_model.py' to generate the model file.")
    sys.exit(1)

try:
    # Load the model using the standard tf.keras path
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model successfully loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    sys.exit(1)
    
# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_image(img):
    """
    Convert uploaded or drawn image to 28x28 grayscale, invert colors, and normalize.
    """
    # Convert to grayscale (Luminance)
    img = img.convert('L')

    # Resize to 28x28 (standard MNIST input size)
    img = img.resize((28, 28), Image.Resampling.LANCZOS) 

    # Convert to numpy array
    img_array = np.array(img)

    # Invert colors (Canvas drawings are usually black stroke on white background. 
    # MNIST models are trained on white digit on black background.)
    img_array = 255 - img_array

    # Normalize pixel values to 0-1
    img_array = img_array.astype('float32') / 255.0

    # Reshape to (1, 28, 28, 1) - Batch size 1, 28x28 image, 1 channel (grayscale)
    img_array = np.expand_dims(img_array, axis=(0, -1))
    return img_array

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict_digit():
    """
    Receives base64 image data (Data URL) in a JSON payload and returns the predicted digit.
    """
    try:
        # 1. Get JSON data from the request body
        data = request.get_json()
        if not data or 'imageData' not in data:
             return jsonify({'error': 'Invalid JSON payload. Expected {"imageData": "..."}'}), 400

        base64_data_url = data['imageData']

        # 2. Extract the raw base64 part from the Data URL (stripping prefix)
        if ',' in base64_data_url:
            _, base64_data = base64_data_url.split(',', 1)
        else:
            base64_data = base64_data_url

        # 3. Decode the base64 string into bytes
        image_bytes = base64.b64decode(base64_data)

        # 4. Load image from bytes stream using PIL
        img = Image.open(io.BytesIO(image_bytes))

        # 5. Preprocess
        processed = preprocess_image(img)

        # 6. Predict
        prediction = model.predict(processed)
        
        # Get the predicted digit (index of the highest probability)
        predicted_digit = int(np.argmax(prediction))
        
        # Get the confidence level
        confidence = float(np.max(prediction))

        # 7. Return result
        return jsonify({
            'prediction': predicted_digit,
            'confidence': confidence
        })

    except Exception as e:
        # Catch any unexpected errors during processing or prediction
        print(f"Prediction Error: {e}")
        return jsonify({'error': f'Internal server error during prediction: {e}'}), 500

# -----------------------------
# Run server
# -----------------------------
if __name__ == '__main__':
    # Setting host to 0.0.0.0 allows access from external sources in containerized environments.
    # use_reloader=False prevents the model from loading twice in debug mode.
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
