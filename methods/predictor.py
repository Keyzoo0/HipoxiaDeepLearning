#!/usr/bin/env python3
"""
Prediction Module
Handles single and batch predictions using trained models
"""

import numpy as np

class ModelPredictor:
    def __init__(self, data_handler, trainer, visualizer, model_builder):
        self.data_handler = data_handler
        self.trainer = trainer
        self.visualizer = visualizer
        self.model_builder = model_builder

    def predict_single_record(self, record_id, method='mdnn'):
        """Predict hypoxia for a single record"""
        method_display = self.model_builder.get_method_display_name(method)
        print(f"🔮 Predicting for record {record_id} using {method_display} method...")

        # Load model if not loaded
        if self.trainer.model is None:
            self.trainer.load_model_from_pkl(method)

        # Load and preprocess signal
        fhr_signal = self.data_handler.load_signal_data(record_id)
        if fhr_signal is None:
            print(f"❌ Signal data not found for record {record_id}")
            return None

        processed_signal = self.data_handler.preprocess_signal(fhr_signal)
        if processed_signal is None:
            print(f"❌ Failed to process signal for record {record_id}")
            return None

        # Get clinical features for this record
        clinical_features, available_features = self.data_handler.get_record_clinical_features(record_id)
        if clinical_features is None:
            print(f"❌ Clinical data not found for record {record_id}")
            return None

        # Scale clinical features using the stored scaler
        if self.data_handler.clinical_scaler is None:
            print(f"❌ Clinical scaler not available. Please train a model first.")
            return None

        clinical_features_scaled = self.data_handler.clinical_scaler.transform(clinical_features)

        # Make prediction
        signal_input = processed_signal.reshape(1, -1)
        prediction_probs = self.trainer.model.predict([signal_input, clinical_features_scaled], verbose=0)
        predicted_class = np.argmax(prediction_probs[0])
        confidence = np.max(prediction_probs[0])

        result = {
            'record_id': record_id,
            'method': method,
            'predicted_class': predicted_class,
            'predicted_label': self.data_handler.label_names[predicted_class],
            'confidence': float(confidence),
            'class_probabilities': {
                self.data_handler.label_names[i]: float(prediction_probs[0][i])
                for i in range(len(self.data_handler.label_names))
            }
        }

        method_display = self.model_builder.get_method_display_name(method)
        print(f"✅ {method_display} Prediction Results:")
        print(f"   Record: {record_id}")
        print(f"   Prediction: {result['predicted_label']}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Probabilities:")
        for label, prob in result['class_probabilities'].items():
            print(f"     {label}: {prob:.3f}")

        # Generate detailed prediction analysis for all methods
        self.visualizer.generate_detailed_prediction_analysis(
            record_id, result, method, self.data_handler
        )

        return result

    def predict_batch_records(self, record_ids, method='mdnn'):
        """Predict hypoxia for multiple records"""
        method_display = self.model_builder.get_method_display_name(method)
        print(f"📊 Batch prediction for {len(record_ids)} records using {method_display}...")

        results = []
        for record_id in record_ids:
            result = self.predict_single_record(record_id, method)
            if result:
                results.append(result)

        if results:
            print(f"\n📊 {method_display} Batch Prediction Summary:")
            for result in results:
                print(f"   Record {result['record_id']}: {result['predicted_label']} ({result['confidence']:.3f})")

        return results

    def compare_all_methods(self, record_id):
        """Compare predictions from all available methods"""
        available_methods = self.trainer.get_available_methods()
        if len(available_methods) < 2:
            print("❌ Need at least 2 trained models for comparison")
            return

        print(f"\n🆚 COMPARING ALL METHODS FOR RECORD {record_id}")
        print("="*60)

        results = {}
        for method in available_methods:
            try:
                # Reset model to load correct method
                self.trainer.model = None
                result = self.predict_single_record(record_id, method)
                if result:
                    results[method] = result
                print()  # Add spacing between methods
            except Exception as e:
                method_display = self.model_builder.get_method_display_name(method)
                print(f"❌ {method_display} error: {e}")

        if len(results) > 1:
            print(f"📊 COMPARISON SUMMARY:")
            print("="*50)
            print(f"{'Method':<12} {'Prediction':<10} {'Confidence':<12}")
            print("-" * 34)
            for method, result in results.items():
                method_display = self.model_builder.get_method_display_name(method)
                print(f"{method_display:<12} {result['predicted_label']:<10} {result['confidence']:<12.3f}")

            # Check consensus
            predictions = [r['predicted_label'] for r in results.values()]
            if len(set(predictions)) == 1:
                print(f"\n✅ CONSENSUS: All methods predict {predictions[0]}")
            else:
                print(f"\n⚠️ DISAGREEMENT: Methods have different predictions")

        return results