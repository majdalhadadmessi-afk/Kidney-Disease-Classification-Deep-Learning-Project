import numpy as np
import os
import tensorflow as tf


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        model_paths = [
            os.path.join("model", "model.h5"),
            os.path.join("artifacts", "training", "model.h5")
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path is None:
            raise FileNotFoundError(
                "Trained model not found in 'model/model.h5' or 'artifacts/training/model.h5'. "
                "Please run training first (python main.py or dvc repro)."
            )

        model = tf.keras.models.load_model(model_path)

        imagename = self.filename
        test_image = tf.keras.preprocessing.image.load_img(imagename, target_size=(224, 224))
        test_image = tf.keras.preprocessing.image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0
        
        raw_predictions = model.predict(test_image)[0]
        predicted_class_index = int(np.argmax(raw_predictions))
        confidence = float(np.max(raw_predictions)) * 100

        # Alphabetical sorting from flow_from_directory for 4 classes
        class_mapping = {
            0: "Cyst",
            1: "Normal",
            2: "Stone",
            3: "Tumor"
        }

        # Fallback if old 2-class model is loaded
        if len(raw_predictions) == 2:
            prediction = "Tumor" if predicted_class_index == 1 else "Normal"
        else:
            prediction = class_mapping.get(predicted_class_index, "Unknown")

        # Class probability breakdown
        probabilities = {}
        for idx, prob in enumerate(raw_predictions):
            class_name = class_mapping.get(idx, f"Class_{idx}")
            probabilities[class_name] = f"{float(prob) * 100:.2f}%"

        return [{
            "image": prediction,
            "confidence": f"{confidence:.2f}%",
            "class_index": predicted_class_index,
            "probabilities": probabilities
        }]
