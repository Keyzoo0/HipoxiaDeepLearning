import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy import signal
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from generateDataset import DatasetGenerator

class MobileNetPredictor:
    def __init__(self, base_path='/home/zainul/joki/HipoxiaDeepLearning'):
        self.base_path = Path(base_path)
        self.models_path = self.base_path / 'models' / 'mobilenet_models'
        self.results_path = self.base_path / 'results' / 'prediction_results' / 'mobilenet_predictions'
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Model parameters (should match training)
        self.signal_length = 5000
        self.spectrogram_shape = (224, 224)
        self.label_names = ['Normal', 'Suspect', 'Hypoxia']
        self.label_colors = ['green', 'orange', 'red']
        
        # Load model
        self.model = None
        self.dataset_generator = DatasetGenerator(base_path)
        
    def load_model(self):
        """Load trained MobileNet model"""
        try:
            model_path = self.models_path / 'mobilenet_classifier.h5'
            
            if not model_path.exists():
                raise FileNotFoundError(f"MobileNet model not found: {model_path}")
                
            self.model = keras.models.load_model(model_path)
            print("✅ MobileNet model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load MobileNet model: {e}")
    
    def signal_to_spectrogram(self, fhr_signal, nperseg=256, noverlap=128):
        """Convert FHR signal to spectrogram (same as training)"""
        try:
            # Remove trend and normalize
            fhr_detrended = signal.detrend(fhr_signal)
            fhr_normalized = (fhr_detrended - np.mean(fhr_detrended)) / (np.std(fhr_detrended) + 1e-8)
            
            # Compute spectrogram
            frequencies, times, Sxx = signal.spectrogram(
                fhr_normalized, 
                fs=4,  # 4 Hz sampling frequency
                nperseg=nperseg,
                noverlap=noverlap,
                window='hann'
            )
            
            # Convert to dB scale
            Sxx_db = 10 * np.log10(Sxx + 1e-10)
            
            # Normalize to [0, 1]
            Sxx_norm = (Sxx_db - np.min(Sxx_db)) / (np.max(Sxx_db) - np.min(Sxx_db) + 1e-8)
            
            # Resize to MobileNet input size (224, 224)
            from tensorflow.image import resize
            Sxx_resized = resize(
                np.expand_dims(Sxx_norm, axis=-1),
                self.spectrogram_shape
            ).numpy()
            
            # Convert to 3-channel RGB (repeat grayscale)
            spectrogram_rgb = np.repeat(Sxx_resized, 3, axis=-1)
            
            return spectrogram_rgb, frequencies, times, Sxx_norm
            
        except Exception as e:
            print(f"⚠️ Error creating spectrogram: {e}")
            return None, None, None, None
    
    def preprocess_signal(self, fhr_signal):
        """Preprocess signal for prediction (same as training)"""
        # Handle variable length signals
        if len(fhr_signal) < self.signal_length:
            # Pad with interpolation
            signal_padded = np.interp(
                np.linspace(0, len(fhr_signal)-1, self.signal_length),
                np.arange(len(fhr_signal)),
                fhr_signal
            )
        else:
            # Truncate or downsample
            step = len(fhr_signal) // self.signal_length
            signal_padded = fhr_signal[::step][:self.signal_length]
        
        return signal_padded
    
    def predict_record(self, record_id, show_visualizations=True):
        """Predict hypoxia for specific record using MobileNet"""
        print(f"🔍 Predicting record {record_id} using MobileNet method...")
        
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Get record info
        record_info = self.dataset_generator.get_record_info(record_id)
        if record_info is None:
            raise ValueError(f"Record {record_id} not found in dataset")
        
        # Load signal data
        try:
            signal_data = self.dataset_generator.load_signal_data(record_id)
        except FileNotFoundError:
            raise ValueError(f"Signal file not found for record {record_id}")
        
        # Preprocess signal
        fhr_signal = signal_data['FHR']
        uc_signal = signal_data['UC']
        
        # Preprocess FHR for prediction
        fhr_processed = self.preprocess_signal(fhr_signal)
        
        # Convert to spectrogram
        spectrogram, frequencies, times, spectrogram_raw = self.signal_to_spectrogram(fhr_processed)
        
        if spectrogram is None:
            raise RuntimeError("Failed to create spectrogram from FHR signal")
        
        # Make prediction
        prediction_probs = self.model.predict(
            np.expand_dims(spectrogram, axis=0), 
            verbose=0
        )[0]
        
        predicted_class = np.argmax(prediction_probs)
        predicted_label = self.label_names[predicted_class]
        confidence = prediction_probs[predicted_class]
        
        # Get true label for comparison
        true_label = record_info['label']
        true_class = self.label_names.index(true_label) if true_label in self.label_names else -1
        
        # Create prediction result
        result = {
            'record_id': record_id,
            'predicted_class': predicted_class,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'probabilities': prediction_probs.tolist(),
            'true_label': true_label,
            'correct_prediction': predicted_label.lower() == true_label.lower(),
            'signal_length': len(fhr_signal),
            'sampling_frequency': signal_data['sampling_frequency'],
            'spectrogram_shape': spectrogram.shape
        }
        
        # Print results
        print(f"\n📊 MobileNet Prediction Results for Record {record_id}:")
        print(f"   Predicted: {predicted_label} ({confidence:.1%} confidence)")
        print(f"   True Label: {true_label}")
        print(f"   Correct: {'✅' if result['correct_prediction'] else '❌'}")
        print(f"   Probabilities:")
        for i, (label, prob) in enumerate(zip(self.label_names, prediction_probs)):
            print(f"     {label}: {prob:.1%}")
        
        if show_visualizations:
            self.create_prediction_visualizations(
                result, fhr_signal, uc_signal, fhr_processed, 
                spectrogram, frequencies, times, spectrogram_raw
            )
        
        return result
    
    def create_prediction_visualizations(self, result, fhr_original, uc_original, 
                                       fhr_processed, spectrogram, frequencies, times, spectrogram_raw):
        """Create comprehensive prediction visualizations for MobileNet"""
        record_id = result['record_id']
        
        # Create figure with subplots
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
        
        # 1. Original FHR Signal
        ax1 = fig.add_subplot(gs[0, 0])
        time_axis = np.arange(len(fhr_original)) / result['sampling_frequency'] / 60
        ax1.plot(time_axis, fhr_original, 'b-', alpha=0.7, linewidth=0.5)
        ax1.set_title(f'Original FHR Signal - Record {record_id}')
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('FHR (bpm)')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(50, 200)
        
        # 2. Original UC Signal
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time_axis, uc_original, 'r-', alpha=0.7, linewidth=0.5)
        ax2.set_title(f'Uterine Contractions - Record {record_id}')
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('UC (mmHg)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Processed FHR Signal
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(fhr_processed, 'g-', alpha=0.8, linewidth=0.8)
        ax3.set_title('Preprocessed FHR Signal')
        ax3.set_xlabel('Sample Points')
        ax3.set_ylabel('FHR (bpm)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Raw Spectrogram (before RGB conversion)
        ax4 = fig.add_subplot(gs[1, 0])
        im1 = ax4.imshow(
            spectrogram_raw, 
            cmap='viridis', 
            aspect='auto', 
            origin='lower',
            extent=[times[0], times[-1], frequencies[0], frequencies[-1]]
        )
        ax4.set_title('Spectrogram (Raw)')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Frequency (Hz)')
        plt.colorbar(im1, ax=ax4, label='Power (dB)')
        
        # 5. RGB Spectrogram (Model Input)
        ax5 = fig.add_subplot(gs[1, 1])
        # Display first channel of RGB spectrogram
        im2 = ax5.imshow(
            spectrogram[:, :, 0], 
            cmap='viridis', 
            aspect='auto',
            origin='lower'
        )
        ax5.set_title(f'Model Input Spectrogram\n{self.spectrogram_shape}px RGB')
        ax5.set_xlabel('Time Bins')
        ax5.set_ylabel('Frequency Bins')
        plt.colorbar(im2, ax=ax5, label='Intensity')
        
        # 6. Feature Map Visualization (if possible)
        ax6 = fig.add_subplot(gs[1, 2])
        try:
            # Get intermediate layer outputs for visualization
            feature_model = keras.Model(
                inputs=self.model.input,
                outputs=self.model.layers[-4].output  # Feature dense layer
            )
            features = feature_model.predict(
                np.expand_dims(spectrogram, axis=0), 
                verbose=0
            )[0]
            
            # Visualize feature vector as bar plot
            ax6.bar(range(len(features)), features, alpha=0.7)
            ax6.set_title('Extracted Features\n(Dense Layer)')
            ax6.set_xlabel('Feature Index')
            ax6.set_ylabel('Activation Value')
            ax6.grid(True, alpha=0.3)
            
        except Exception as e:
            ax6.text(0.5, 0.5, f'Feature Visualization\nNot Available\n({str(e)[:30]}...)', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Feature Visualization')
        
        # 7. Prediction Probabilities (Bar Chart)
        ax7 = fig.add_subplot(gs[2, 0])
        bars = ax7.bar(self.label_names, result['probabilities'], 
                      color=self.label_colors, alpha=0.7)
        ax7.set_title('Prediction Probabilities')
        ax7.set_ylabel('Probability')
        ax7.set_ylim(0, 1)
        
        # Highlight predicted class
        predicted_idx = result['predicted_class']
        bars[predicted_idx].set_alpha(1.0)
        bars[predicted_idx].set_edgecolor('black')
        bars[predicted_idx].set_linewidth(2)
        
        # Add probability values on bars
        for i, (bar, prob) in enumerate(zip(bars, result['probabilities'])):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
        
        ax7.grid(True, alpha=0.3, axis='y')
        
        # 8. Confidence Score Gauge
        ax8 = fig.add_subplot(gs[2, 1])
        
        # Create a simple gauge chart
        confidence = result['confidence']
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        wedges = [0.2, 0.4, 0.6, 0.8, 1.0]
        
        for i, (wedge, color) in enumerate(zip(wedges, colors)):
            start_angle = 180 - wedge * 180
            end_angle = 180
            if i == 0:
                start_angle = 0
            else:
                start_angle = 180 - wedges[i-1] * 180
            
            theta = np.linspace(np.radians(start_angle), np.radians(180 - wedge * 180), 20)
            x = np.cos(theta)
            y = np.sin(theta)
            ax8.fill_between(x, 0, y, alpha=0.3, color=color)
        
        # Add confidence needle
        conf_angle = np.radians(180 - confidence * 180)
        ax8.arrow(0, 0, 0.8 * np.cos(conf_angle), 0.8 * np.sin(conf_angle),
                 head_width=0.05, head_length=0.05, fc='black', ec='black')
        
        ax8.set_xlim(-1, 1)
        ax8.set_ylim(0, 1)
        ax8.set_aspect('equal')
        ax8.axis('off')
        ax8.set_title(f'Confidence: {confidence:.1%}')
        
        # 9. Model Architecture Info
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        arch_info = f"""
MobileNet Architecture
{'='*20}
Base: MobileNetV2
Input: {self.spectrogram_shape}×3
Parameters: {self.model.count_params():,}

Preprocessing:
1. Signal standardization
2. STFT spectrogram
3. dB scaling & normalization
4. Resize to 224×224
5. RGB conversion

Transfer Learning:
✓ ImageNet pretrained
✓ Fine-tuned on FHR data
        """
        
        ax9.text(0.05, 0.95, arch_info, transform=ax9.transAxes,
                fontsize=8, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcyan', alpha=0.8))
        
        # 10. Prediction Summary
        ax10 = fig.add_subplot(gs[3, :])
        ax10.axis('off')
        
        summary_text = f"""
MOBILENET PREDICTION SUMMARY - Record {record_id}
{'='*80}
Predicted Class: {result['predicted_label']} ({result['confidence']:.1%} confidence)
True Label: {result['true_label']}
Prediction Status: {'✅ CORRECT' if result['correct_prediction'] else '❌ INCORRECT'}

Signal Information:
• Original Signal Length: {result['signal_length']:,} samples ({result['signal_length']/result['sampling_frequency']/60:.1f} minutes)
• Sampling Frequency: {result['sampling_frequency']} Hz
• Spectrogram Shape: {result['spectrogram_shape']}

Method: MobileNet with Transfer Learning (ImageNet → FHR Spectrograms)
Processing: Time-Frequency Analysis → Deep CNN Feature Extraction → Classification
        """
        
        ax10.text(0.05, 0.95, summary_text, transform=ax10.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        
        plt.suptitle(f'MobileNet Method Prediction - Record {record_id}', 
                    fontsize=16, fontweight='bold')
        
        # Save plot
        plot_filename = f'mobilenet_prediction_record_{record_id}.png'
        plt.savefig(self.results_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ MobileNet visualization saved: {self.results_path / plot_filename}")
    
    def batch_predict(self, record_ids, save_summary=True):
        """Predict multiple records and create summary"""
        print(f"🔍 Running MobileNet batch prediction on {len(record_ids)} records...")
        
        results = []
        correct_predictions = 0
        
        for record_id in record_ids:
            try:
                result = self.predict_record(record_id, show_visualizations=False)
                results.append(result)
                if result['correct_prediction']:
                    correct_predictions += 1
                    
            except Exception as e:
                print(f"⚠️ Error predicting record {record_id}: {e}")
                continue
        
        # Calculate metrics
        accuracy = correct_predictions / len(results) if results else 0
        
        print(f"\n📊 MobileNet Batch Prediction Summary:")
        print(f"   Records processed: {len(results)}")
        print(f"   Accuracy: {accuracy:.1%}")
        
        if save_summary and results:
            self.save_batch_summary(results)
        
        return results
    
    def save_batch_summary(self, results):
        """Save batch prediction summary for MobileNet"""
        # Create summary DataFrame
        df_results = pd.DataFrame(results)
        df_results.to_csv(self.results_path / 'mobilenet_batch_predictions.csv', index=False)
        
        # Create summary visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('MobileNet Method - Batch Prediction Summary', fontsize=14)
        
        # 1. Confusion Matrix
        true_labels = [r['true_label'] for r in results]
        predicted_labels = [r['predicted_label'] for r in results]
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_labels, predicted_labels, 
                            labels=[l.lower() for l in self.label_names])
        
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=self.label_names, yticklabels=self.label_names)
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        
        # 2. Confidence Distribution
        confidences = [r['confidence'] for r in results]
        ax2.hist(confidences, bins=20, alpha=0.7, edgecolor='black', color='orange')
        ax2.set_title('Prediction Confidence Distribution')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3)
        
        # 3. Accuracy by Class
        class_accuracies = []
        class_counts = []
        for label in self.label_names:
            class_results = [r for r in results if r['true_label'] == label.lower()]
            if class_results:
                class_acc = sum(r['correct_prediction'] for r in class_results) / len(class_results)
                class_accuracies.append(class_acc)
                class_counts.append(len(class_results))
            else:
                class_accuracies.append(0)
                class_counts.append(0)
        
        bars = ax3.bar(self.label_names, class_accuracies, 
                      color=self.label_colors, alpha=0.7)
        ax3.set_title('Accuracy by Class')
        ax3.set_ylabel('Accuracy')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, acc, count in zip(bars, class_accuracies, class_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.1%}\n(n={count})', ha='center', va='bottom', 
                    fontweight='bold', fontsize=8)
        
        # 4. Sample Distribution
        true_dist = pd.Series(true_labels).value_counts()
        colors_mapped = [self.label_colors[self.label_names.index(label.title())] 
                        for label in true_dist.index]
        ax4.pie(true_dist.values, labels=[label.title() for label in true_dist.index], 
               autopct='%1.1f%%', colors=colors_mapped)
        ax4.set_title('True Label Distribution')
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'mobilenet_batch_summary.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ MobileNet batch summary saved: {self.results_path / 'mobilenet_batch_summary.png'}")

def main():
    """Main prediction function for testing"""
    predictor = MobileNetPredictor()
    
    # Get available records
    dataset_gen = DatasetGenerator()
    available_records = dataset_gen.get_available_records()
    
    if not available_records:
        print("❌ No records available for prediction")
        return
    
    # Test prediction on first few records
    test_records = available_records[:3]
    print(f"Testing MobileNet prediction on records: {test_records}")
    
    # Single prediction test
    result = predictor.predict_record(test_records[0])
    
    # Batch prediction test
    batch_results = predictor.batch_predict(test_records)
    
    print("✅ MobileNet prediction testing completed!")

if __name__ == "__main__":
    main()