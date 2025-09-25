#!/usr/bin/env python3
"""
Model Loader for Fetal Hypoxia Detection
Loads trained models and handles prediction
"""

import pickle
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    # Fix for keras.src import error
    import tensorflow.keras as keras
except ImportError:
    print("Warning: TensorFlow or sklearn not available")

class ModelLoader:
    """Load and manage trained hypoxia detection models"""

    def __init__(self, models_path: Optional[Path] = None):
        self.models_path = models_path or Path(__file__).parent.parent / 'models'
        self.loaded_models = {}
        self.label_names = ['Normal', 'Suspect', 'Hypoxia']

        # Method information
        self.method_info = {
            'mdnn': {
                'name': 'MDNN',
                'description': 'Multimodal Dense Neural Network',
                'accuracy': '80%+'
            },
            'gan': {
                'name': 'GAN',
                'description': 'GAN-Enhanced Feature Extraction',
                'accuracy': '60%+'
            },
            'mobilenet': {
                'name': 'MobileNet',
                'description': 'Lightweight CNN Architecture',
                'accuracy': '75%+'
            },
            'resnet': {
                'name': 'ResNet',
                'description': 'Deep Residual Neural Network',
                'accuracy': '70%+'
            }
        }

    async def load_models(self):
        """Load all available trained models"""
        print("🔄 Loading trained models...")

        methods = ['mdnn', 'gan', 'mobilenet', 'resnet']
        loaded_count = 0

        for method in methods:
            try:
                model_path = self.models_path / f'{method}_multimodal_hypoxia_detector.pkl'

                if model_path.exists():
                    print(f"📦 Loading {method.upper()} model...")

                    # Load with custom handling for keras compatibility
                    try:
                        with open(model_path, 'rb') as f:
                            model_data = pickle.load(f)

                        self.loaded_models[method] = {
                            'model': model_data['model'],
                            'scaler': model_data.get('clinical_scaler'),
                            'label_names': model_data.get('label_names', self.label_names),
                            'signal_length': model_data.get('signal_length', 5000),
                            'method': method
                        }

                        loaded_count += 1
                        print(f"✅ {method.upper()} model loaded successfully")

                    except Exception as load_error:
                        if "keras.src" in str(load_error) or "No module named" in str(load_error):
                            print(f"   ⚠️  Model compatibility issue, skipping {method.upper()}")
                        else:
                            print(f"   ❌ Load error: {load_error}")

                else:
                    print(f"⚠️ {method.upper()} model not found: {model_path}")

            except Exception as e:
                print(f"❌ Failed to process {method.upper()} model: {e}")

        if loaded_count > 0:
            print(f"🎉 Successfully loaded {loaded_count} models!")
        else:
            print("⚠️ No models loaded. Please train models first.")

    def get_available_methods(self) -> List[str]:
        """Get list of loaded methods"""
        return list(self.loaded_models.keys())

    def get_method_descriptions(self) -> Dict[str, Dict[str, str]]:
        """Get method information"""
        available_methods = self.get_available_methods()
        return {
            method: self.method_info[method]
            for method in available_methods
            if method in self.method_info
        }

    async def predict(
        self,
        clinical_features: List[float],
        signal_features: List[float],
        method: str = 'mdnn'
    ) -> Dict[str, Any]:
        """Make prediction using specified method"""

        if method not in self.loaded_models:
            available = ', '.join(self.get_available_methods())
            raise ValueError(f"Method '{method}' not loaded. Available: {available}")

        try:
            model_data = self.loaded_models[method]
            model = model_data['model']
            scaler = model_data['scaler']

            # Prepare clinical features
            clinical_array = np.array(clinical_features).reshape(1, -1)

            # Scale clinical features if scaler is available
            if scaler is not None:
                try:
                    clinical_array = scaler.transform(clinical_array)
                except Exception as e:
                    print(f"Warning: Could not apply scaler: {e}")

            # Prepare signal features
            signal_length = model_data['signal_length']
            signal_array = np.array(signal_features)

            # Ensure correct signal length
            if len(signal_array) != signal_length:
                if len(signal_array) > signal_length:
                    # Truncate
                    signal_array = signal_array[:signal_length]
                else:
                    # Pad with zeros
                    padded = np.zeros(signal_length)
                    padded[:len(signal_array)] = signal_array
                    signal_array = padded

            signal_array = signal_array.reshape(1, -1)

            # Make prediction
            probabilities = model.predict([signal_array, clinical_array], verbose=0)[0]
            prediction_idx = np.argmax(probabilities)
            confidence = float(probabilities[prediction_idx])

            # Get label names
            label_names = model_data['label_names']
            prediction_label = label_names[prediction_idx]

            # Generate interpretation
            interpretation = self._generate_interpretation(
                prediction_label, confidence, probabilities, method
            )

            return {
                'prediction': prediction_label,
                'confidence': confidence,
                'probabilities': {
                    label_names[i]: float(prob)
                    for i, prob in enumerate(probabilities)
                },
                'interpretation': interpretation
            }

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def _generate_interpretation(
        self,
        prediction: str,
        confidence: float,
        probabilities: np.ndarray,
        method: str
    ) -> Dict[str, Any]:
        """Generate clinical interpretation of prediction"""

        method_display = self.method_info.get(method, {}).get('name', method.upper())

        # Risk level based on prediction and confidence
        if prediction == 'Normal' and confidence > 0.8:
            risk_level = 'Low Risk'
            recommendation = 'Continue routine monitoring'
            urgency = 'routine'
        elif prediction == 'Normal' and confidence > 0.6:
            risk_level = 'Low Risk'
            recommendation = 'Continue monitoring with increased attention'
            urgency = 'routine'
        elif prediction == 'Suspect':
            risk_level = 'Moderate Risk'
            recommendation = 'Increased monitoring recommended, consider intervention if pattern continues'
            urgency = 'moderate'
        elif prediction == 'Hypoxia' and confidence > 0.7:
            risk_level = 'High Risk'
            recommendation = 'Immediate medical attention required, consider urgent delivery'
            urgency = 'urgent'
        else:
            risk_level = 'Uncertain'
            recommendation = 'Uncertain classification, clinical judgment required'
            urgency = 'review'

        # Additional insights
        insights = []
        normal_prob = probabilities[0]
        suspect_prob = probabilities[1]
        hypoxia_prob = probabilities[2]

        if hypoxia_prob > 0.3:
            insights.append(f"Significant hypoxia probability ({hypoxia_prob:.1%})")

        if suspect_prob > 0.4:
            insights.append(f"Moderate suspect classification probability ({suspect_prob:.1%})")

        if confidence < 0.6:
            insights.append("Low confidence prediction - additional monitoring recommended")

        return {
            'risk_level': risk_level,
            'recommendation': recommendation,
            'urgency': urgency,
            'method_used': method_display,
            'confidence_level': 'High' if confidence > 0.8 else 'Moderate' if confidence > 0.6 else 'Low',
            'insights': insights,
            'clinical_note': f'Prediction made using {method_display} with {confidence:.1%} confidence'
        }

    def get_model_status(self) -> Dict[str, Any]:
        """Get status of loaded models"""
        return {
            'loaded_methods': self.get_available_methods(),
            'total_methods': len(self.method_info),
            'models_path': str(self.models_path),
            'status': 'ready' if self.loaded_models else 'no_models'
        }