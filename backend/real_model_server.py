#!/usr/bin/env python3
"""
Real Model Server for Fetal Hypoxia Detection
Uses actual trained ML models with .hea/.dat converter
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import shutil
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processor import DataProcessor

app = Flask(__name__)
CORS(app)

# Initialize components
data_processor = DataProcessor()

# Import required methods components
try:
    from methods.model_builder import ModelBuilder
    from methods.trainer import ModelTrainer
    from methods.data_handler import DataHandler
    from methods.visualizer import Visualizer
    print("✅ Methods imported successfully")
except Exception as e:
    print(f"❌ Failed to import methods: {e}")

class SimpleHypoxiaDetector:
    """Simplified version of MultimodalHypoxiaDetector for web use"""

    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        self.models_path = self.base_path / 'models'
        self.trainer = None
        self.model_builder = None

        try:
            # Initialize components needed for prediction
            self.data_handler = DataHandler(str(self.base_path))
            self.model_builder = ModelBuilder()
            self.visualizer = Visualizer(str(self.base_path), self.model_builder)
            self.trainer = ModelTrainer(str(self.base_path), self.data_handler, self.model_builder, self.visualizer)
            print("✅ SimpleHypoxiaDetector initialized")
        except Exception as e:
            print(f"❌ Failed to initialize detector: {e}")
            self.trainer = None

    def get_available_methods(self):
        """Get available trained methods"""
        if self.trainer:
            return self.trainer.get_available_methods()
        return []

    def predict_with_converted_data(self, clinical_features, signal_features, method='mdnn'):
        """Make prediction using converted .hea/.dat data"""
        if not self.trainer:
            raise Exception("Trainer not initialized")

        try:
            # Create temporary dataset with converted data
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_processed = Path(temp_dir) / "processed_data"
                temp_processed.mkdir()

                # Create signals directory
                signals_dir = temp_processed / "signals"
                signals_dir.mkdir()

                # Create temporary record ID
                record_id = "web_upload_001"

                # Save clinical data as CSV
                clinical_df = pd.DataFrame([clinical_features],
                                        columns=data_processor.clinical_params[:len(clinical_features)])
                clinical_df.index = [record_id]
                clinical_csv = temp_processed / "clinical_dataset.csv"
                clinical_df.to_csv(clinical_csv)

                # Save signal data as .npy
                signal_file = signals_dir / f"{record_id}.npy"
                np.save(signal_file, np.array(signal_features))

                # Temporarily redirect data handler
                original_path = self.data_handler.processed_data_path
                self.data_handler.processed_data_path = temp_processed

                try:
                    # Generate dataset with converted data
                    self.data_handler.generate_multimodal_dataset()

                    # Load model
                    self.trainer.load_model_from_pkl(method)

                    # Get prediction data
                    (X_signals_train, X_clinical_train, y_train,
                     X_signals_val, X_clinical_val, y_val,
                     X_signals_test, X_clinical_test, y_test) = self.data_handler.prepare_data_for_training()

                    # Use test data for prediction (it contains our uploaded data)
                    if len(X_signals_test) > 0 and len(X_clinical_test) > 0:
                        # Take first sample (our uploaded data)
                        prediction = self.trainer.model.predict([X_signals_test[:1], X_clinical_test[:1]], verbose=0)

                        predicted_class = np.argmax(prediction[0])
                        confidence = float(prediction[0][predicted_class])
                        probabilities = {
                            self.data_handler.label_names[i]: float(prediction[0][i])
                            for i in range(len(prediction[0]))
                        }

                        return {
                            "prediction": self.data_handler.label_names[predicted_class],
                            "confidence": confidence,
                            "probabilities": probabilities,
                            "method_used": self.model_builder.get_method_display_name(method)
                        }
                    else:
                        raise Exception("No test data available for prediction")

                finally:
                    # Restore original path
                    self.data_handler.processed_data_path = original_path

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            raise e

# Global detector instance
detector = None

@app.before_first_request
def initialize():
    """Initialize detector on startup"""
    global detector
    print("🚀 Initializing Real Model Server...")
    try:
        detector = SimpleHypoxiaDetector()
        print("✅ Real Model Server initialized!")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")

@app.route('/')
def root():
    """API health check"""
    available_methods = []
    if detector:
        available_methods = detector.get_available_methods()

    return jsonify({
        "message": "Fetal Hypoxia Detection Real Model API is running!",
        "status": "healthy",
        "available_methods": available_methods,
        "features": ["Real Trained ML Models", ".hea/.dat File Processing", "Multimodal Analysis"]
    })

@app.route('/models')
def get_available_models():
    """Get list of available prediction methods"""
    available_methods = []
    descriptions = {}

    if detector:
        available_methods = detector.get_available_methods()

        # Method descriptions
        method_info = {
            'mdnn': {'name': 'MDNN', 'description': 'Multimodal Dense Neural Network', 'accuracy': '80%+'},
            'gan': {'name': 'GAN', 'description': 'GAN-Enhanced Feature Extraction', 'accuracy': '60%+'},
            'mobilenet': {'name': 'MobileNet', 'description': 'Lightweight CNN Architecture', 'accuracy': '75%+'},
            'resnet': {'name': 'ResNet', 'description': 'Deep Residual Neural Network', 'accuracy': '70%+'}
        }

        descriptions = {method: method_info.get(method, {
            'name': method.upper(),
            'description': 'Neural Network Model',
            'accuracy': 'Available'
        }) for method in available_methods}

    return jsonify({
        "methods": available_methods,
        "descriptions": descriptions
    })

@app.route('/predict_complete', methods=['POST'])
def predict_complete_workflow():
    """Complete workflow: upload → convert → predict with real ML models"""

    if not detector:
        return jsonify({"error": "System not initialized"}), 500

    try:
        # Check if files are present
        if 'hea_file' not in request.files or 'dat_file' not in request.files:
            return jsonify({"error": "Both hea_file and dat_file are required"}), 400

        hea_file = request.files['hea_file']
        dat_file = request.files['dat_file']
        method = request.form.get('method', 'mdnn')

        print(f"📁 Processing: {hea_file.filename}, {dat_file.filename}")
        print(f"🔧 Method: {method}")

        # Validate files
        if not hea_file.filename.endswith('.hea') or not dat_file.filename.endswith('.dat'):
            return jsonify({"error": "Invalid file extensions"}), 400

        hea_name = Path(hea_file.filename).stem
        dat_name = Path(dat_file.filename).stem

        if hea_name != dat_name:
            return jsonify({"error": "File names must match"}), 400

        # Check if method is available
        available_methods = detector.get_available_methods()
        if method not in available_methods:
            return jsonify({
                "error": f"Method '{method}' not available. Available methods: {available_methods}"
            }), 400

        # Process files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            hea_path = temp_path / hea_file.filename
            dat_path = temp_path / dat_file.filename

            hea_file.save(str(hea_path))
            dat_file.save(str(dat_path))

            # Step 1: Convert .hea/.dat to features
            print("🔄 Converting files...")
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            conversion_result = loop.run_until_complete(
                data_processor.process_files(hea_path, dat_path)
            )
            print("✅ Files converted")

            # Step 2: Real ML prediction
            print(f"🧠 Making real ML prediction with {method.upper()}...")

            try:
                prediction_result = detector.predict_with_converted_data(
                    conversion_result["clinical_features"],
                    conversion_result["signal_features"],
                    method
                )
                print("✅ Real ML prediction successful")

                # Generate clinical interpretation
                prediction = prediction_result["prediction"]
                confidence = prediction_result["confidence"]

                if prediction == "Normal":
                    risk_level = "Low Risk"
                    recommendation = "Continue routine monitoring"
                    urgency = "routine"
                elif prediction == "Suspect":
                    risk_level = "Moderate Risk"
                    recommendation = "Increased monitoring recommended"
                    urgency = "moderate"
                else:  # Hypoxia
                    risk_level = "High Risk"
                    recommendation = "Immediate medical attention required"
                    urgency = "urgent"

                # Create comprehensive response
                response_data = {
                    "status": "success",
                    "record_id": conversion_result["processing_info"]["record_id"],
                    "method": method,
                    "processing_info": conversion_result["processing_info"],
                    "prediction": prediction,
                    "confidence": confidence,
                    "probabilities": prediction_result["probabilities"],
                    "interpretation": {
                        "risk_level": risk_level,
                        "recommendation": recommendation,
                        "urgency": urgency,
                        "method_used": prediction_result["method_used"],
                        "confidence_level": "High" if confidence > 0.8 else "Moderate" if confidence > 0.6 else "Low",
                        "insights": [
                            f"Real ML prediction using trained {method.upper()} model",
                            f"Analysis based on {len(conversion_result['clinical_features'])} clinical parameters",
                            f"FHR signal processing: {len(conversion_result['signal_features'])} samples analyzed"
                        ],
                        "clinical_note": f"Prediction made using trained {prediction_result['method_used']} model with {confidence:.1%} confidence."
                    },
                    "note": f"Real ML prediction using trained {method.upper()} model completed successfully"
                }

                return jsonify(response_data)

            except Exception as model_error:
                print(f"❌ Real ML prediction failed: {model_error}")

                # Fallback to intelligent analysis
                sys.path.append('.')
                from minimal_server import generate_intelligent_prediction

                fallback_result = generate_intelligent_prediction(
                    conversion_result["clinical_features"],
                    method
                )

                fallback_result["status"] = "fallback"
                fallback_result["note"] = f"Real ML model unavailable ({str(model_error)}). Showing clinical parameter analysis."
                fallback_result["record_id"] = conversion_result["processing_info"]["record_id"]
                fallback_result["processing_info"] = conversion_result["processing_info"]

                return jsonify(fallback_result)

    except Exception as e:
        print(f"❌ Complete workflow error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Complete workflow error: {str(e)}"}), 500

if __name__ == "__main__":
    print("🌐 Starting Real Model Server for Fetal Hypoxia Detection...")
    print("🔗 API will be available at: http://localhost:8000")
    print("🎯 Features: Trained ML Models + .hea/.dat Converter")
    print("⚡ Press Ctrl+C to stop the server\n")

    app.run(host="0.0.0.0", port=8000, debug=True)