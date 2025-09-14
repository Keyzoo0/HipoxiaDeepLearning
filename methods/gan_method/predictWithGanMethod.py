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

class GANPredictor:
    def __init__(self, base_path='/home/zainul/joki/HipoxiaDeepLearning'):
        self.base_path = Path(base_path)
        self.models_path = self.base_path / 'models' / 'gan_models'
        self.results_path = self.base_path / 'results' / 'prediction_results' / 'gan_predictions'
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Model parameters (should match training)
        self.signal_length = 5000
        self.label_names = ['Normal', 'Suspect', 'Hypoxia']
        self.label_colors = ['green', 'orange', 'red']
        
        # Load models
        self.classifier = None
        self.generator = None
        self.dataset_generator = DatasetGenerator(base_path)
        
    def load_models(self):
        """Load trained GAN models"""
        try:
            classifier_path = self.models_path / 'gan_classifier.h5'
            generator_path = self.models_path / 'gan_generator.h5'
            
            if not classifier_path.exists():
                raise FileNotFoundError(f"Classifier model not found: {classifier_path}")
            if not generator_path.exists():
                print(f"⚠️ Generator model not found: {generator_path}")
                
            self.classifier = keras.models.load_model(classifier_path)
            
            if generator_path.exists():
                self.generator = keras.models.load_model(generator_path)
                print("✅ GAN models loaded successfully")
            else:
                print("✅ Classifier model loaded (Generator not available)")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load GAN models: {e}")
    
    def preprocess_signal(self, signal):
        """Preprocess signal for prediction (same as training)"""
        # Handle variable length signals
        if len(signal) < self.signal_length:
            # Pad with interpolation
            signal_padded = np.interp(
                np.linspace(0, len(signal)-1, self.signal_length),
                np.arange(len(signal)),
                signal
            )
        else:
            # Truncate or downsample
            step = len(signal) // self.signal_length
            signal_padded = signal[::step][:self.signal_length]
        
        # Normalize FHR signal to [0, 1] (same as training)
        signal_normalized = (signal_padded - 50) / 200
        signal_normalized = np.clip(signal_normalized, 0, 1)
        
        return signal_normalized
    
    def predict_record(self, record_id, show_visualizations=True):
        """Predict hypoxia for specific record"""
        print(f"🔍 Predicting record {record_id} using GAN method...")
        
        # Load models if not already loaded
        if self.classifier is None:
            self.load_models()
        
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
        prediction_probs = self.classifier.predict(fhr_processed.reshape(1, -1), verbose=0)[0]
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
            'sampling_frequency': signal_data['sampling_frequency']
        }
        
        # Print results
        print(f"\n📊 Prediction Results for Record {record_id}:")
        print(f"   Predicted: {predicted_label} ({confidence:.1%} confidence)")
        print(f"   True Label: {true_label}")
        print(f"   Correct: {'✅' if result['correct_prediction'] else '❌'}")
        print(f"   Probabilities:")
        for i, (label, prob) in enumerate(zip(self.label_names, prediction_probs)):
            print(f"     {label}: {prob:.1%}")
        
        if show_visualizations:
            self.create_prediction_visualizations(result, fhr_signal, uc_signal, fhr_processed)
        
        return result
    
    def create_prediction_visualizations(self, result, fhr_original, uc_original, fhr_processed):
        """Create comprehensive prediction visualizations"""
        record_id = result['record_id']
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Original FHR Signal
        ax1 = fig.add_subplot(gs[0, 0])
        time_axis = np.arange(len(fhr_original)) / result['sampling_frequency'] / 60  # Convert to minutes
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
        
        # 3. Processed FHR Signal (used for prediction)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(fhr_processed, 'g-', alpha=0.8, linewidth=0.8)
        ax3.set_title('Preprocessed FHR Signal (Model Input)')
        ax3.set_xlabel('Sample Points')
        ax3.set_ylabel('Normalized FHR')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # 4. Prediction Probabilities (Bar Chart)
        ax4 = fig.add_subplot(gs[1, 1])
        bars = ax4.bar(self.label_names, result['probabilities'], 
                      color=self.label_colors, alpha=0.7)
        ax4.set_title('Prediction Probabilities')
        ax4.set_ylabel('Probability')
        ax4.set_ylim(0, 1)
        
        # Highlight predicted class
        predicted_idx = result['predicted_class']
        bars[predicted_idx].set_alpha(1.0)
        bars[predicted_idx].set_edgecolor('black')
        bars[predicted_idx].set_linewidth(2)
        
        # Add probability values on bars
        for i, (bar, prob) in enumerate(zip(bars, result['probabilities'])):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
        
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Prediction Summary
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # Create summary text
        summary_text = f"""
PREDICTION SUMMARY - Record {record_id}
{'='*50}
Predicted Class: {result['predicted_label']} ({result['confidence']:.1%} confidence)
True Label: {result['true_label']}
Prediction Status: {'✅ CORRECT' if result['correct_prediction'] else '❌ INCORRECT'}

Signal Information:
• Original Signal Length: {result['signal_length']:,} samples
• Sampling Frequency: {result['sampling_frequency']} Hz
• Duration: {result['signal_length']/result['sampling_frequency']/60:.1f} minutes

Method: GAN-based Classification (CTGGAN + CNN-LSTM)
        """
        
        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle(f'GAN Method Prediction - Record {record_id}', 
                    fontsize=14, fontweight='bold')
        
        # Save plot
        plot_filename = f'gan_prediction_record_{record_id}.png'
        plt.savefig(self.results_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualization saved: {self.results_path / plot_filename}")
        
        # Generate synthetic signals if generator is available
        if self.generator is not None:
            self.generate_synthetic_comparison(record_id, result['predicted_class'])
    
    def generate_synthetic_comparison(self, record_id, predicted_class):
        """Generate synthetic signals for comparison"""
        try:
            # Generate synthetic signals for each class
            noise = np.random.normal(0, 1, (3, 100))  # noise_dim = 100
            labels = np.array([0, 1, 2])  # All classes
            
            synthetic_signals = self.generator.predict([noise, labels], verbose=0)
            
            # Create comparison plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f'Synthetic Signal Generation - Record {record_id}', fontsize=14)
            
            for i, (signal, label, color) in enumerate(zip(synthetic_signals, self.label_names, self.label_colors)):
                row = i // 2
                col = i % 2
                if i < 3:  # Only plot first 3
                    ax = axes[row, col] if i < 2 else axes[1, 1]
                    
                    # Convert back to FHR range for visualization
                    signal_fhr = signal * 200 + 50  # Reverse normalization
                    
                    ax.plot(signal_fhr[:1000], color=color, alpha=0.8, linewidth=1)
                    ax.set_title(f'Generated {label} FHR Signal')
                    ax.set_xlabel('Time Steps')
                    ax.set_ylabel('FHR (bpm)')
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(50, 200)
                    
                    # Highlight if this matches prediction
                    if i == predicted_class:
                        ax.set_title(f'Generated {label} FHR Signal ⭐ (Predicted Class)', 
                                   fontweight='bold')
            
            # Remove empty subplot
            if len(synthetic_signals) < 4:
                fig.delaxes(axes[1, 0]) if len(synthetic_signals) == 3 else None
            
            plt.tight_layout()
            
            # Save synthetic comparison
            synthetic_filename = f'gan_synthetic_comparison_record_{record_id}.png'
            plt.savefig(self.results_path / synthetic_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Synthetic comparison saved: {self.results_path / synthetic_filename}")
            
        except Exception as e:
            print(f"⚠️ Could not generate synthetic signals: {e}")
    
    def batch_predict(self, record_ids, save_summary=True):
        """Predict multiple records and create summary"""
        print(f"🔍 Running batch prediction on {len(record_ids)} records...")
        
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
        
        print(f"\n📊 Batch Prediction Summary:")
        print(f"   Records processed: {len(results)}")
        print(f"   Accuracy: {accuracy:.1%}")
        
        if save_summary and results:
            self.save_batch_summary(results)
        
        return results
    
    def save_batch_summary(self, results):
        """Save batch prediction summary"""
        # Create summary DataFrame
        df_results = pd.DataFrame(results)
        df_results.to_csv(self.results_path / 'gan_batch_predictions.csv', index=False)
        
        # Create summary visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('GAN Method - Batch Prediction Summary', fontsize=14)
        
        # 1. Accuracy by class
        true_labels = [r['true_label'] for r in results]
        predicted_labels = [r['predicted_label'] for r in results]
        
        # Confusion matrix data
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(true_labels, predicted_labels, labels=[l.lower() for l in self.label_names])
        
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=self.label_names, yticklabels=self.label_names)
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        
        # 2. Confidence distribution
        confidences = [r['confidence'] for r in results]
        ax2.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_title('Prediction Confidence Distribution')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3)
        
        # 3. Accuracy by class
        class_accuracies = []
        for label in self.label_names:
            class_results = [r for r in results if r['true_label'] == label.lower()]
            if class_results:
                class_acc = sum(r['correct_prediction'] for r in class_results) / len(class_results)
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0)
        
        bars = ax3.bar(self.label_names, class_accuracies, color=self.label_colors, alpha=0.7)
        ax3.set_title('Accuracy by Class')
        ax3.set_ylabel('Accuracy')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, acc in zip(bars, class_accuracies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Sample distribution
        true_dist = pd.Series(true_labels).value_counts()
        ax4.pie(true_dist.values, labels=true_dist.index, autopct='%1.1f%%', 
               colors=self.label_colors)
        ax4.set_title('True Label Distribution')
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'gan_batch_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Batch summary saved: {self.results_path / 'gan_batch_summary.png'}")

def main():
    """Main prediction function for testing"""
    predictor = GANPredictor()
    
    # Get available records
    dataset_gen = DatasetGenerator()
    available_records = dataset_gen.get_available_records()
    
    if not available_records:
        print("❌ No records available for prediction")
        return
    
    # Test prediction on first few records
    test_records = available_records[:5]
    print(f"Testing GAN prediction on records: {test_records}")
    
    # Single prediction test
    result = predictor.predict_record(test_records[0])
    
    # Batch prediction test
    batch_results = predictor.batch_predict(test_records)
    
    print("✅ GAN prediction testing completed!")

if __name__ == "__main__":
    main()