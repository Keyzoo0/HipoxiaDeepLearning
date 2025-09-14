import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from generateDataset import DatasetGenerator

class ResNetPredictor:
    def __init__(self, base_path='/home/zainul/joki/HipoxiaDeepLearning'):
        self.base_path = Path(base_path)
        self.models_path = self.base_path / 'models' / 'resnet_models'
        self.results_path = self.base_path / 'results' / 'prediction_results' / 'resnet_predictions'
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Model parameters (should match training)
        self.signal_length = 5000
        self.label_names = ['Normal', 'Suspect', 'Hypoxia']
        self.label_colors = ['green', 'orange', 'red']
        
        # Load model
        self.model = None
        self.dataset_generator = DatasetGenerator(base_path)
        
    def load_model(self):
        """Load trained ResNet model"""
        try:
            model_path = self.models_path / 'resnet_classifier.h5'
            
            if not model_path.exists():
                raise FileNotFoundError(f"ResNet model not found: {model_path}")
                
            self.model = keras.models.load_model(model_path)
            print("✅ ResNet model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load ResNet model: {e}")
    
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
        
        # Normalize FHR signal (same as training)
        signal_normalized = (signal_padded - np.mean(signal_padded)) / (np.std(signal_padded) + 1e-8)
        
        # Z-score normalization
        signal_standardized = (signal_normalized - np.mean(signal_normalized)) / (np.std(signal_normalized) + 1e-8)
        
        return signal_standardized
    
    def extract_features(self, signal):
        """Extract intermediate features from ResNet layers"""
        try:
            # Get layer outputs for feature visualization
            layer_outputs = []
            layer_names = []
            
            for layer in self.model.layers:
                if isinstance(layer, keras.layers.Conv1D) and 'conv' in layer.name.lower():
                    layer_outputs.append(layer.output)
                    layer_names.append(layer.name)
                elif isinstance(layer, keras.layers.Dense) and 'fc' in layer.name.lower():
                    layer_outputs.append(layer.output)
                    layer_names.append(layer.name)
            
            if not layer_outputs:
                return None, None
            
            # Create feature extraction model
            feature_model = keras.Model(self.model.input, layer_outputs)
            
            # Extract features
            features = feature_model.predict(signal.reshape(1, -1), verbose=0)
            
            return features, layer_names
            
        except Exception as e:
            print(f"⚠️ Could not extract features: {e}")
            return None, None
    
    def predict_record(self, record_id, show_visualizations=True):
        """Predict hypoxia for specific record using ResNet"""
        print(f"🔍 Predicting record {record_id} using ResNet method...")
        
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
        
        # Make prediction
        prediction_probs = self.model.predict(
            fhr_processed.reshape(1, -1), 
            verbose=0
        )[0]
        
        predicted_class = np.argmax(prediction_probs)
        predicted_label = self.label_names[predicted_class]
        confidence = prediction_probs[predicted_class]
        
        # Get true label for comparison
        true_label = record_info['label']
        true_class = self.label_names.index(true_label) if true_label in self.label_names else -1
        
        # Extract features for visualization
        features, layer_names = self.extract_features(fhr_processed)
        
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
            'features': features,
            'layer_names': layer_names
        }
        
        # Print results
        print(f"\n📊 ResNet Prediction Results for Record {record_id}:")
        print(f"   Predicted: {predicted_label} ({confidence:.1%} confidence)")
        print(f"   True Label: {true_label}")
        print(f"   Correct: {'✅' if result['correct_prediction'] else '❌'}")
        print(f"   Probabilities:")
        for i, (label, prob) in enumerate(zip(self.label_names, prediction_probs)):
            print(f"     {label}: {prob:.1%}")
        
        if show_visualizations:
            self.create_prediction_visualizations(
                result, fhr_signal, uc_signal, fhr_processed
            )
        
        return result
    
    def create_prediction_visualizations(self, result, fhr_original, uc_original, fhr_processed):
        """Create comprehensive prediction visualizations for ResNet"""
        record_id = result['record_id']
        
        # Create figure with subplots
        fig = plt.figure(figsize=(18, 16))
        gs = fig.add_gridspec(5, 3, hspace=0.4, wspace=0.3)
        
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
        ax3.set_title('Preprocessed FHR Signal (Model Input)')
        ax3.set_xlabel('Sample Points')
        ax3.set_ylabel('Standardized FHR')
        ax3.grid(True, alpha=0.3)
        
        # 4. Feature Maps Visualization (if available)
        features = result.get('features')
        layer_names = result.get('layer_names')
        
        if features and layer_names and len(features) > 0:
            # First Conv Layer Feature Map
            ax4 = fig.add_subplot(gs[1, 0])
            first_conv_features = features[0][0]  # First sample, first layer
            
            # Show first few filters
            n_filters_to_show = min(8, first_conv_features.shape[-1])
            for i in range(n_filters_to_show):
                ax4.plot(first_conv_features[:500, i], alpha=0.6, linewidth=0.5)
            
            ax4.set_title(f'Conv Layer Features\n({layer_names[0]})')
            ax4.set_xlabel('Time Steps')
            ax4.set_ylabel('Feature Value')
            ax4.grid(True, alpha=0.3)
            
            # Middle Conv Layer Feature Map
            if len(features) > 3:
                ax5 = fig.add_subplot(gs[1, 1])
                mid_conv_features = features[len(features)//2][0]
                
                # Average across filters for visualization
                avg_features = np.mean(mid_conv_features, axis=-1)
                ax5.plot(avg_features, 'purple', alpha=0.8)
                ax5.set_title(f'Mid-Layer Features\n({layer_names[len(features)//2]})')
                ax5.set_xlabel('Time Steps')
                ax5.set_ylabel('Average Feature Value')
                ax5.grid(True, alpha=0.3)
            else:
                ax5 = fig.add_subplot(gs[1, 1])
                ax5.text(0.5, 0.5, 'Mid-Layer Features\nNot Available', 
                        ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('Mid-Layer Features')
            
            # Dense Layer Features
            dense_features = None
            for i, (feature, name) in enumerate(zip(features, layer_names)):
                if 'fc' in name.lower() or 'dense' in name.lower():
                    dense_features = feature[0]  # First sample
                    break
            
            ax6 = fig.add_subplot(gs[1, 2])
            if dense_features is not None and len(dense_features) > 0:
                ax6.bar(range(min(len(dense_features), 50)), dense_features[:50], alpha=0.7)
                ax6.set_title('Dense Layer Features (First 50)')
                ax6.set_xlabel('Feature Index')
                ax6.set_ylabel('Activation Value')
                ax6.grid(True, alpha=0.3)
            else:
                ax6.text(0.5, 0.5, 'Dense Features\nNot Available', 
                        ha='center', va='center', transform=ax6.transAxes)
                ax6.set_title('Dense Features')
        else:
            # No features available
            for i, ax in enumerate([fig.add_subplot(gs[1, j]) for j in range(3)]):
                ax.text(0.5, 0.5, 'Feature Extraction\nNot Available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(['Conv Features', 'Mid Features', 'Dense Features'][i])
        
        # 5. Prediction Probabilities (Bar Chart)
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
        
        # 6. Confidence Gauge
        ax8 = fig.add_subplot(gs[2, 1])
        
        # Simple confidence visualization
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
        
        # 7. ResNet Architecture Overview
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        arch_info = f"""
1D ResNet Architecture
{'='*20}
Input: {self.signal_length} samples
Blocks: Residual connections
Filters: 64→128→256→512
Parameters: ~{self.model.count_params()//1000}K

Key Components:
• 1D Convolutions
• Skip Connections
• Batch Normalization  
• Global Avg Pooling
• Dense Classification

Preprocessing:
• Length standardization
• Z-score normalization
• Temporal augmentation
        """
        
        ax9.text(0.05, 0.95, arch_info, transform=ax9.transAxes,
                fontsize=8, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8))
        
        # 8. Signal Analysis (Frequency Domain)
        ax10 = fig.add_subplot(gs[3, 0])
        try:
            from scipy.fft import fft, fftfreq
            
            # Compute FFT of processed signal
            fft_vals = np.abs(fft(fhr_processed))
            freqs = fftfreq(len(fhr_processed), d=1/4)  # 4 Hz sampling
            
            # Plot only positive frequencies
            pos_mask = freqs > 0
            ax10.plot(freqs[pos_mask][:200], fft_vals[pos_mask][:200])
            ax10.set_title('Frequency Domain Analysis')
            ax10.set_xlabel('Frequency (Hz)')
            ax10.set_ylabel('Magnitude')
            ax10.grid(True, alpha=0.3)
            
        except ImportError:
            ax10.text(0.5, 0.5, 'Frequency Analysis\nRequires scipy', 
                     ha='center', va='center', transform=ax10.transAxes)
            ax10.set_title('Frequency Analysis')
        
        # 9. Signal Statistics
        ax11 = fig.add_subplot(gs[3, 1])
        ax11.axis('off')
        
        # Calculate signal statistics
        fhr_mean = np.mean(fhr_original)
        fhr_std = np.std(fhr_original)
        fhr_min = np.min(fhr_original)
        fhr_max = np.max(fhr_original)
        
        signal_stats = f"""
Signal Statistics
{'='*15}
Mean FHR: {fhr_mean:.1f} bpm
Std Dev:  {fhr_std:.1f} bpm
Min FHR:  {fhr_min:.1f} bpm  
Max FHR:  {fhr_max:.1f} bpm

Duration: {len(fhr_original)/result['sampling_frequency']/60:.1f} min
Samples:  {len(fhr_original):,}

Preprocessing:
• Standardized to {self.signal_length} samples
• Z-score normalized
• Ready for ResNet input
        """
        
        ax11.text(0.05, 0.95, signal_stats, transform=ax11.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
        
        # 10. Residual Block Concept
        ax12 = fig.add_subplot(gs[3, 2])
        ax12.axis('off')
        
        resnet_concept = f"""
Residual Learning Concept
{'='*23}
Traditional: H(x) = F(x)
ResNet: H(x) = F(x) + x

Benefits:
• Gradient flow improvement
• Deeper network training
• Feature reuse
• Better optimization

Skip Connections:
Input → Conv → BN → ReLU
    ↓     ↓
    → Add → ReLU → Output

Applied to 1D FHR signals
for temporal pattern learning
        """
        
        ax12.text(0.05, 0.95, resnet_concept, transform=ax12.transAxes,
                 fontsize=8, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
        
        # 11. Prediction Summary
        ax13 = fig.add_subplot(gs[4, :])
        ax13.axis('off')
        
        summary_text = f"""
RESNET PREDICTION SUMMARY - Record {record_id}
{'='*80}
Predicted Class: {result['predicted_label']} ({result['confidence']:.1%} confidence)
True Label: {result['true_label']}
Prediction Status: {'✅ CORRECT' if result['correct_prediction'] else '❌ INCORRECT'}

Signal Information:
• Original Signal Length: {result['signal_length']:,} samples ({result['signal_length']/result['sampling_frequency']/60:.1f} minutes)
• Processed Signal Length: {self.signal_length:,} samples
• Sampling Frequency: {result['sampling_frequency']} Hz

Method: 1D ResNet with Residual Connections
Architecture: Deep convolutional network optimized for temporal sequence analysis
Key Features: Skip connections, batch normalization, global average pooling
        """
        
        ax13.text(0.05, 0.95, summary_text, transform=ax13.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        
        plt.suptitle(f'ResNet Method Prediction - Record {record_id}', 
                    fontsize=16, fontweight='bold')
        
        # Save plot
        plot_filename = f'resnet_prediction_record_{record_id}.png'
        plt.savefig(self.results_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ ResNet visualization saved: {self.results_path / plot_filename}")
    
    def batch_predict(self, record_ids, save_summary=True):
        """Predict multiple records and create summary"""
        print(f"🔍 Running ResNet batch prediction on {len(record_ids)} records...")
        
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
        
        print(f"\n📊 ResNet Batch Prediction Summary:")
        print(f"   Records processed: {len(results)}")
        print(f"   Accuracy: {accuracy:.1%}")
        
        if save_summary and results:
            self.save_batch_summary(results)
        
        return results
    
    def save_batch_summary(self, results):
        """Save batch prediction summary for ResNet"""
        # Create summary DataFrame
        df_results = pd.DataFrame(results)
        df_results.to_csv(self.results_path / 'resnet_batch_predictions.csv', index=False)
        
        # Create summary visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('ResNet Method - Batch Prediction Summary', fontsize=14)
        
        # 1. Confusion Matrix
        true_labels = [r['true_label'] for r in results]
        predicted_labels = [r['predicted_label'] for r in results]
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_labels, predicted_labels, 
                            labels=[l.lower() for l in self.label_names])
        
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax1,
                   xticklabels=self.label_names, yticklabels=self.label_names)
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        
        # 2. Confidence Distribution
        confidences = [r['confidence'] for r in results]
        ax2.hist(confidences, bins=20, alpha=0.7, edgecolor='black', color='red')
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
        plt.savefig(self.results_path / 'resnet_batch_summary.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ ResNet batch summary saved: {self.results_path / 'resnet_batch_summary.png'}")

def main():
    """Main prediction function for testing"""
    predictor = ResNetPredictor()
    
    # Get available records
    dataset_gen = DatasetGenerator()
    available_records = dataset_gen.get_available_records()
    
    if not available_records:
        print("❌ No records available for prediction")
        return
    
    # Test prediction on first few records
    test_records = available_records[:3]
    print(f"Testing ResNet prediction on records: {test_records}")
    
    # Single prediction test
    result = predictor.predict_record(test_records[0])
    
    # Batch prediction test
    batch_results = predictor.batch_predict(test_records)
    
    print("✅ ResNet prediction testing completed!")

if __name__ == "__main__":
    main()