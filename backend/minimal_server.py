#!/usr/bin/env python3
"""
Minimal Flask Server for Testing File Upload
No model loading - just test file processing
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data_processor import DataProcessor

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Initialize components
data_processor = DataProcessor()

def generate_intelligent_prediction(clinical_features, method="test"):
    """
    Generate intelligent prediction based on actual clinical data
    Uses clinical parameter rules similar to real medical assessment
    """
    import random

    # Extract key clinical parameters (matching data_processor.clinical_params order)
    pH = clinical_features[0] if len(clinical_features) > 0 else 7.35
    BDecf = clinical_features[1] if len(clinical_features) > 1 else 0.0
    pCO2 = clinical_features[2] if len(clinical_features) > 2 else 5.0
    BE = clinical_features[3] if len(clinical_features) > 3 else 0.0
    Apgar1 = clinical_features[4] if len(clinical_features) > 4 else 8
    Apgar5 = clinical_features[5] if len(clinical_features) > 5 else 9

    print(f"📊 Analyzing: pH={pH:.2f}, BDecf={BDecf:.2f}, pCO2={pCO2:.2f}, Apgar1={Apgar1}")

    # Clinical decision logic based on medical guidelines
    risk_score = 0
    insights = []

    # pH analysis (most important indicator)
    if pH < 7.20:
        risk_score += 30
        insights.append(f"Acidosis detected (pH {pH:.2f} < 7.20)")
    elif pH < 7.25:
        risk_score += 15
        insights.append(f"Mild acidosis (pH {pH:.2f} between 7.20-7.25)")
    elif pH > 7.45:
        risk_score += 5
        insights.append(f"Alkalosis noted (pH {pH:.2f} > 7.45)")

    # Base Deficit analysis
    if BDecf > 8:
        risk_score += 20
        insights.append(f"Significant base deficit ({BDecf:.1f} mmol/L)")
    elif BDecf > 12:
        risk_score += 35
        insights.append(f"Severe base deficit ({BDecf:.1f} mmol/L)")

    # pCO2 analysis
    if pCO2 > 8.0:
        risk_score += 15
        insights.append(f"Elevated pCO2 ({pCO2:.1f} kPa) suggesting respiratory acidosis")
    elif pCO2 < 4.0:
        risk_score += 8
        insights.append(f"Low pCO2 ({pCO2:.1f} kPa) possible hyperventilation")

    # Apgar score analysis
    if Apgar1 < 4:
        risk_score += 25
        insights.append(f"Low Apgar1 score ({int(Apgar1)}) indicates birth complications")
    elif Apgar1 < 7:
        risk_score += 10
        insights.append(f"Moderate Apgar1 score ({int(Apgar1)})")

    if Apgar5 < 7:
        risk_score += 15
        insights.append(f"Persistent low Apgar5 score ({int(Apgar5)})")

    # Base Excess analysis
    if BE < -8:
        risk_score += 20
        insights.append(f"Significant metabolic acidosis (BE {BE:.1f})")
    elif BE < -12:
        risk_score += 30
        insights.append(f"Severe metabolic acidosis (BE {BE:.1f})")

    # Add method-specific variation (simulate different AI approaches)
    method_modifier = {
        'mdnn': 1.0,     # Baseline
        'gan': 0.85,     # Slightly lower confidence
        'mobilenet': 1.1,  # Slightly higher sensitivity
        'resnet': 0.95,   # Conservative
        'test': 1.0      # Same as MDNN
    }.get(method, 1.0)

    risk_score = int(risk_score * method_modifier)

    # Add some randomness for signal analysis (simulate FHR patterns)
    signal_risk = random.randint(-5, 15)  # Random component for FHR analysis
    risk_score += signal_risk

    if signal_risk > 10:
        insights.append("Concerning FHR patterns detected")
    elif signal_risk > 5:
        insights.append("Mild FHR variability noted")

    # Determine prediction based on risk score
    if risk_score >= 40:
        prediction = "Hypoxia"
        confidence = min(0.95, 0.6 + (risk_score - 40) * 0.01)
        normal_prob = max(0.05, 0.3 - (risk_score - 40) * 0.005)
        suspect_prob = max(0.05, 0.4 - (risk_score - 40) * 0.008)
        hypoxia_prob = 1.0 - normal_prob - suspect_prob
        risk_level = "High Risk"
        recommendation = "Immediate medical attention required. Consider urgent delivery."
        urgency = "urgent"
    elif risk_score >= 20:
        prediction = "Suspect"
        confidence = min(0.85, 0.5 + (risk_score - 20) * 0.02)
        normal_prob = max(0.15, 0.6 - (risk_score - 20) * 0.01)
        suspect_prob = min(0.75, 0.3 + (risk_score - 20) * 0.02)
        hypoxia_prob = 1.0 - normal_prob - suspect_prob
        risk_level = "Moderate Risk"
        recommendation = "Increased monitoring recommended. Consider intervention if pattern continues."
        urgency = "moderate"
    else:
        prediction = "Normal"
        confidence = max(0.65, 0.9 - risk_score * 0.01)
        normal_prob = max(0.6, 0.85 - risk_score * 0.01)
        suspect_prob = min(0.3, 0.1 + risk_score * 0.01)
        hypoxia_prob = 1.0 - normal_prob - suspect_prob
        risk_level = "Low Risk"
        recommendation = "Continue routine monitoring. Current parameters within normal limits."
        urgency = "routine"

    # Normalize probabilities
    total_prob = normal_prob + suspect_prob + hypoxia_prob
    normal_prob /= total_prob
    suspect_prob /= total_prob
    hypoxia_prob /= total_prob

    # Add method-specific insights
    method_names = {
        'mdnn': 'Multimodal Dense Neural Network',
        'gan': 'GAN-Enhanced Feature Extraction',
        'mobilenet': 'MobileNet Architecture',
        'resnet': 'Residual Neural Network'
    }

    method_display = method_names.get(method, 'Test Mode')

    if not insights:
        insights.append("All major clinical parameters within acceptable ranges")

    print(f"🎯 Prediction: {prediction} (confidence: {confidence:.1%}, risk_score: {risk_score})")

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": {
            "Normal": normal_prob,
            "Suspect": suspect_prob,
            "Hypoxia": hypoxia_prob
        },
        "interpretation": {
            "risk_level": risk_level,
            "recommendation": recommendation,
            "urgency": urgency,
            "method_used": method_display,
            "confidence_level": "High" if confidence > 0.8 else "Moderate" if confidence > 0.6 else "Low",
            "insights": insights,
            "clinical_note": f"Analysis based on clinical parameters: pH={pH:.2f}, BDecf={BDecf:.1f}, Apgar1={int(Apgar1)}. Risk score: {risk_score}/100."
        }
    }

@app.route('/')
def root():
    """API health check"""
    return jsonify({
        "message": "Minimal Fetal Hypoxia Detection API is running!",
        "status": "healthy",
        "available_methods": ["test"]
    })

@app.route('/models')
def get_available_models():
    """Get list of available prediction methods"""
    return jsonify({
        "methods": ["test"],
        "descriptions": {
            "test": {
                "name": "Test Mode",
                "description": "File processing test without AI prediction",
                "accuracy": "Test only"
            }
        }
    })

@app.route('/predict_complete', methods=['POST'])
def predict_complete_workflow():
    """Complete workflow: upload → process → mock predict"""
    try:
        # Check if files are present
        if 'hea_file' not in request.files or 'dat_file' not in request.files:
            return jsonify({"error": "Both hea_file and dat_file are required"}), 400

        hea_file = request.files['hea_file']
        dat_file = request.files['dat_file']
        method = request.form.get('method', 'test')

        print(f"📁 Received files: {hea_file.filename}, {dat_file.filename}")
        print(f"🔧 Method: {method}")

        # Validate files
        if not hea_file.filename.endswith('.hea') or not dat_file.filename.endswith('.dat'):
            return jsonify({"error": "Invalid file extensions"}), 400

        hea_name = Path(hea_file.filename).stem
        dat_name = Path(dat_file.filename).stem

        if hea_name != dat_name:
            return jsonify({"error": "File names must match"}), 400

        # Process files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            hea_path = temp_path / hea_file.filename
            dat_path = temp_path / dat_file.filename

            hea_file.save(str(hea_path))
            dat_file.save(str(dat_path))

            print(f"💾 Files saved to: {temp_path}")

            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Step 1: Process files
            print("🔄 Processing files...")
            upload_result = loop.run_until_complete(data_processor.process_files(hea_path, dat_path))
            print("✅ File processing complete")

            # Step 2: Intelligent prediction based on clinical data
            print("🧠 Making intelligent prediction based on clinical data...")
            mock_prediction_result = generate_intelligent_prediction(upload_result["clinical_features"], method)

            return jsonify({
                "status": "success",
                "record_id": upload_result["processing_info"]["record_id"],
                "method": method,
                "processing_info": upload_result["processing_info"],
                "prediction": mock_prediction_result["prediction"],
                "confidence": mock_prediction_result["confidence"],
                "probabilities": mock_prediction_result["probabilities"],
                "interpretation": mock_prediction_result["interpretation"],
                "note": "This is a test mode response. Train models for actual predictions."
            })

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Complete workflow error: {str(e)}"}), 500

@app.route('/upload', methods=['POST'])
def upload_files():
    """Test file upload and processing only"""
    try:
        if 'hea_file' not in request.files or 'dat_file' not in request.files:
            return jsonify({"error": "Both hea_file and dat_file are required"}), 400

        hea_file = request.files['hea_file']
        dat_file = request.files['dat_file']

        # Quick validation
        if not hea_file.filename.endswith('.hea'):
            return jsonify({"error": "First file must be a .hea file"}), 400
        if not dat_file.filename.endswith('.dat'):
            return jsonify({"error": "Second file must be a .dat file"}), 400

        hea_name = Path(hea_file.filename).stem
        dat_name = Path(dat_file.filename).stem

        if hea_name != dat_name:
            return jsonify({
                "error": f"File names must match: {hea_name}.hea and {dat_name}.dat"
            }), 400

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            hea_path = temp_path / hea_file.filename
            dat_path = temp_path / dat_file.filename

            hea_file.save(str(hea_path))
            dat_file.save(str(dat_path))

            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(data_processor.process_files(hea_path, dat_path))

            return jsonify({
                "status": "success",
                "message": "Files processed successfully",
                "record_id": hea_name,
                "clinical_features": result["clinical_features"][:5],  # Show only first 5
                "signal_length": len(result["signal_features"]),
                "processing_info": result["processing_info"]
            })

    except Exception as e:
        print(f"❌ Upload error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

if __name__ == "__main__":
    print("🧪 Starting Minimal Test Server (No ML Models)...")
    print("🔗 API will be available at: http://localhost:8000")
    print("🎯 This server only tests file processing, not AI prediction")
    print("⚡ Press Ctrl+C to stop the server\n")

    app.run(host="0.0.0.0", port=8000, debug=True)