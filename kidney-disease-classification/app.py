from flask import Flask, request, jsonify, render_template
import os
import sys
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.predict import PredictionPipeline
from cnnClassifier import logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


clApp = ClientApp()


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    try:
        logger.info("Triggering training pipeline execution...")
        exit_code = os.system("python main.py")
        if exit_code == 0:
            return jsonify({
                "status": "success",
                "message": "Model training completed successfully! Model weights saved."
            })
        else:
            return jsonify({
                "status": "warning",
                "message": f"Training completed with code {exit_code}. Check logs/running_logs.log for details."
            }), 500
    except Exception as e:
        logger.exception(e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        data = request.get_json(silent=True)
        if not data or 'image' not in data:
            return jsonify({"status": "error", "message": "No image data provided"}), 400

        image_base64 = data['image']
        # Handle data URL prefix if present
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        decodeImage(image_base64, clApp.filename)
        
        try:
            result = clApp.classifier.predict()
            return jsonify(result)
        except FileNotFoundError as fnf:
            logger.warning(f"Model not found: {fnf}")
            # Friendly response explaining model training is needed
            return jsonify([{
                "image": "Model Not Trained",
                "confidence": "N/A",
                "details": "Model file not found. Please click 'Run Training Pipeline' first."
            }])
        except Exception as pred_err:
            logger.exception(pred_err)
            return jsonify({"status": "error", "message": str(pred_err)}), 500

    except Exception as e:
        logger.exception(e)
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
