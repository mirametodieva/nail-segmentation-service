from flask import Flask, request, jsonify, send_file, after_this_request
from roboflow import Roboflow
import cv2
import numpy as np
import tempfile
import os
from io import BytesIO

app = Flask(__name__)

# Initialize Roboflow model
rf = Roboflow(api_key="iZLItqoQ2uKAiATHnqdd")
project = rf.workspace().project("nails_segmentation-vhnmw-p6sip")
model = project.version(3).model

@app.route('/segment-nails', methods=['POST'])
def segment_nails():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    tmp_file_path = None
    output_path = None

    try:
        # Save uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            image_file.save(tmp_file)
            tmp_file_path = tmp_file.name

        # Run prediction
        result = model.predict(tmp_file_path, confidence=40).json()
        original = cv2.imread(tmp_file_path)
        height, width = original.shape[:2]

        # Create transparent RGBA image
        transparent_output = np.zeros((height, width, 4), dtype=np.uint8)

        for pred in result["predictions"]:
            if "points" in pred:
                # Convert polygon points
                points = np.array([[pt["x"], pt["y"]] for pt in pred["points"]], dtype=np.int32)

                # Create binary mask
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [points], 255)

                # Extract the nail region from original image
                nail_rgb = cv2.bitwise_and(original, original, mask=mask)

                # Convert to RGBA and apply mask to alpha channel
                nail_rgba = cv2.cvtColor(nail_rgb, cv2.COLOR_BGR2BGRA)
                nail_rgba[:, :, 3] = mask  # set alpha to the mask

                # Composite onto transparent background
                transparent_output = cv2.add(transparent_output, nail_rgba)

        # Save result as PNG
        with tempfile.NamedTemporaryFile(delete=False, suffix="_nails.png") as out_file:
            output_path = out_file.name
            cv2.imwrite(output_path, transparent_output)

        # Load to memory and return
        with open(output_path, 'rb') as f:
            image_bytes = BytesIO(f.read())

        @after_this_request
        def cleanup(response):
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
            if output_path and os.path.exists(output_path):
                os.remove(output_path)
            return response

        return send_file(image_bytes, mimetype='image/png', download_name='nails_only.png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
