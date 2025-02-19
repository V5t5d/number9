from image_service import process_image
import numpy as np
import logging
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from io import BytesIO
import os
import matplotlib
import json
from flask import Flask, request
from flask_cors import CORS
matplotlib.use('Agg')  # Use a non-interactive backend for thread safety

# Constants
PORT = 5000  # Port on which the Flask server will run
HOST = '0.0.0.0'  # Host address for the Flask server
DEBUG_MODE = True  # Enable/disable debug mode for Flask

# Set up logging
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s - %(levelname)s - %(message)s')

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename="./logs/server.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


app = Flask(__name__)
CORS(app)


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Endpoint to handle image uploads.
    Reads the image from the request, processes it, and returns the results.
    """
    if not request.files:
        logging.error("No file part in the request")
        return jsonify({"error": "No file part"}), 400

    # Get the first file from the request
    image = next(iter(request.files.values()))

    if image.filename == '':
        logging.error("No file selected")
        return jsonify({"error": "No selected file"}), 400

    logging.info(f"Processing image: {image.filename}")

    try:
        # Read the image bytes
        image_bytes = image.read()

        results = process_image(image_bytes)

        results_serializable = get_results_serializable(results)

        # Convert the response to a JSON string
        response_json = json.dumps(results_serializable)

        return response_json, 200

    except Exception as e:
        logging.error(
            f"An error occurred while processing the image in upload_file(): {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


def get_results_serializable(results):
    if results is None:
        logging.error("No results provided to serialize.")
        return None

    try:
        # Use dictionary comprehension for conversion with a safer check
        return {
            key: (results[key].tolist() if isinstance(
                results[key], np.ndarray) else results[key])
            for key in results  # Iterate over all keys in the results
        }

    except Exception as e:
        logging.error(f"Error serializing results: {str(e)}")
        return None


if __name__ == '__main__':
    logging.info(f"Starting Flask server on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=DEBUG_MODE)
