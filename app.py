from flask import Flask, request, send_file, jsonify, after_this_request
import requests
import tempfile
import os
import cv2
import numpy as np
from io import BytesIO
import traceback

app = Flask(__name__)

ROBOFLOW_API_KEY = "iZLItqoQ2uKAiATHnqdd"
ROBOFLOW_URL = "https://detect.roboflow.com/nails_segmentation-vhnmw-p6sip-mejkj/2?api_key=" + ROBOFLOW_API_KEY

@app.route('/segment-nails', methods=['POST'])
def segment_nails():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']

    try:
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            image_file.save(tmp_file)
            tmp_file_path = tmp_file.name

        # Send multipart/form-data POST to Roboflow directly
        with open(tmp_file_path, 'rb') as img:
            roboflow_response = requests.post(
                ROBOFLOW_URL,
                files={'file': img}
            )

        if roboflow_response.status_code != 200:
            return jsonify({'error': 'Roboflow error', 'details': roboflow_response.text}), 500

        result = roboflow_response.json()

        # Process predictions
        original = cv2.imread(tmp_file_path)
        height, width = original.shape[:2]
        transparent_output = np.zeros((height, width, 4), dtype=np.uint8)

        predictions = result.get("predictions", [])

        for pred in predictions:
            if "points" in pred:
                points = np.array([[pt["x"], pt["y"]] for pt in pred["points"]], dtype=np.int32)

                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [points], 255)

                nail_rgb = cv2.bitwise_and(original, original, mask=mask)
                nail_rgba = cv2.cvtColor(nail_rgb, cv2.COLOR_BGR2BGRA)
                nail_rgba[:, :, 3] = mask

                transparent_output = cv2.add(transparent_output, nail_rgba)

        # Save and return result
        with tempfile.NamedTemporaryFile(delete=False, suffix="_nails.png") as out_file:
            output_path = out_file.name
            cv2.imwrite(output_path, transparent_output)

        @after_this_request
        def cleanup(response):
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
            if output_path and os.path.exists(output_path):
                os.remove(output_path)
            return response

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
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
