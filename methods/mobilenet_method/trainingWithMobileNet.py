import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

class MobileNetTrainer:
    def __init__(self, base_path='/home/zainul/joki/HipoxiaDeepLearning'):
        self.base_path = Path(base_path)
        self.models_path = self.base_path / 'models' / 'mobilenet_models'
        self.results_path = self.base_path / 'results' / 'training_plots'
        self.data_path = self.base_path / 'data' / 'mobilenet'
        
        # Ensure directories exist
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Model parameters
        self.signal_length = 5000  # Standardize signal length
        self.spectrogram_shape = (224, 224)  # MobileNet input size
        self.num_classes = 3
        self.epochs = 1
        self.batch_size = 16  # Smaller batch for memory efficiency
        
        self.label_names = ['Normal', 'Suspect', 'Hypoxia']
        
    def load_and_preprocess_data(self):
        """Load and preprocess data for MobileNet training"""
        print("🔄 Loading MobileNet training data...")
        
        X_data = np.load(self.data_path / 'X_data.npy', allow_pickle=True)
        y_data = np.load(self.data_path / 'y_data.npy')
        record_ids = np.load(self.data_path / 'record_ids.npy')
        
        # Preprocess signals to standard length
        X_processed = []
        y_processed = []
        
        for i, (fhr_signal, label) in enumerate(zip(X_data, y_data)):
            try:
                # Standardize signal length
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
                
                # Convert to spectrogram
                spectrogram = self.signal_to_spectrogram(signal_padded)
                
                if spectrogram is not None:
                    X_processed.append(spectrogram)
                    y_processed.append(label)
                    
            except Exception as e:
                print(f"⚠️ Error processing signal {i}: {e}")
                continue
        
        X_processed = np.array(X_processed)
        y_processed = np.array(y_processed)
        
        print(f"✅ Processed {len(X_processed)} spectrograms")
        print(f"   Spectrogram shape: {X_processed[0].shape}")
        print(f"   Label distribution: {np.bincount(y_processed)}")
        
        return X_processed, y_processed
    
    def signal_to_spectrogram(self, fhr_signal, nperseg=256, noverlap=128):
        """Convert FHR signal to spectrogram for MobileNet input"""
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
            
            return spectrogram_rgb
            
        except Exception as e:
            print(f"⚠️ Error creating spectrogram: {e}")
            return None
    
    def create_data_augmentation(self):
        """Create data augmentation for spectrograms"""
        return keras.Sequential([
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
            layers.RandomBrightness(0.1),
        ])
    
    def build_mobilenet_model(self):
        """Build MobileNet-based model for FHR classification"""
        print("🔧 Building MobileNet model...")
        
        # Input layer
        input_layer = layers.Input(shape=(*self.spectrogram_shape, 3))
        
        # Data augmentation (applied during training only)
        augmentation = self.create_data_augmentation()
        
        # MobileNetV2 base model
        base_model = MobileNetV2(
            input_shape=(*self.spectrogram_shape, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Build complete model
        x = augmentation(input_layer)
        x = base_model(x)
        
        # Custom classification head
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu', name='feature_dense')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu', name='classification_dense')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        output = layers.Dense(
            self.num_classes, 
            activation='softmax', 
            name='classification_output'
        )(x)
        
        # Create model
        model = keras.Model(input_layer, output, name='mobilenet_fhr_classifier')
        
        # Compile with initial lower learning rate
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        print(f"✅ MobileNet model built successfully")
        print(f"   Total parameters: {model.count_params():,}")
        print(f"   Trainable parameters: {sum(tf.size(p).numpy() for p in model.trainable_weights):,}")
        
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """Train MobileNet model with transfer learning"""
        print("🚀 Starting MobileNet training...")
        
        # Define callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                self.models_path / 'mobilenet_best_weights.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            )
        ]
        
        # Phase 1: Train only custom head
        print("\n📚 Phase 1: Training custom head (base model frozen)")
        
        history1 = self.model.fit(
            X_train, y_train,
            epochs=1,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Fine-tune top layers
        print("\n📚 Phase 2: Fine-tuning top layers")
        
        # Unfreeze top layers of base model
        base_model = self.model.layers[2]  # MobileNetV2 is 3rd layer (after input and augmentation)
        base_model.trainable = True
        
        # Freeze earlier layers
        for layer in base_model.layers[:-30]:  # Keep last 30 layers trainable
            layer.trainable = False
        
        # Recompile with lower learning rate for fine-tuning
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Continue training
        history2 = self.model.fit(
            X_train, y_train,
            epochs=1,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            # Skip initial_epoch to avoid issues
            verbose=1
        )
        
        # Combine histories safely
        combined_history = {}
        for key in history1.history.keys():
            if key in history2.history:
                combined_history[key] = history1.history[key] + history2.history[key]
            else:
                print(f"⚠️ Warning: Key '{key}' not found in phase 2 history")
                combined_history[key] = history1.history[key]
        
        print("✅ MobileNet training completed!")
        
        return combined_history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate trained model"""
        print("🔍 Evaluating MobileNet model...")
        
        # Make predictions
        y_pred_probs = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Classification report
        report = classification_report(
            y_test, y_pred,
            target_names=self.label_names,
            output_dict=True
        )

        # Get weighted average precision and recall
        test_precision = report['weighted avg']['precision']
        test_recall = report['weighted avg']['recall']

        print(f"📊 Test Results:")
        print(f"   Accuracy:  {test_accuracy:.4f}")
        print(f"   Precision: {test_precision:.4f}")
        print(f"   Recall:    {test_recall:.4f}")
        
        # Per-class results
        for i, label in enumerate(self.label_names):
            if label.lower() in report:
                metrics = report[label.lower()]
                print(f"   {label}: Precision={metrics['precision']:.3f}, "
                      f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        return {
            'test_accuracy': test_accuracy,
            'test_precision': test_precision, 
            'test_recall': test_recall,
            'classification_report': report,
            'y_pred': y_pred,
            'y_pred_probs': y_pred_probs
        }
    
    def save_training_plots(self, history, evaluation_results, X_test, y_test):
        """Save comprehensive training visualization plots"""
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Training History - Accuracy
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(history['accuracy'], label='Training Accuracy', color='blue')
        ax1.plot(history['val_accuracy'], label='Validation Accuracy', color='orange')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Training History - Loss
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(history['loss'], label='Training Loss', color='blue')
        ax2.plot(history['val_loss'], label='Validation Loss', color='orange')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Precision and Recall
        ax3 = fig.add_subplot(gs[0, 2])
        # Only plot if keys exist
        if 'precision' in history:
            ax3.plot(history['precision'], label='Training Precision', color='green')
            ax3.plot(history['val_precision'], label='Validation Precision', color='red')
        if 'recall' in history:
            ax3.plot(history['recall'], label='Training Recall', color='purple', linestyle='--')
            ax3.plot(history['val_recall'], label='Validation Recall', color='brown', linestyle='--')

        # If no precision/recall data, show test results as text
        if 'precision' not in history:
            ax3.text(0.5, 0.5, f'Test Results:\nPrecision: {evaluation_results["test_precision"]:.3f}\n'
                               f'Recall: {evaluation_results["test_recall"]:.3f}',
                    transform=ax3.transAxes, ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax3.set_title('Precision & Recall')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Metric Value')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Learning Rate (if available)
        ax4 = fig.add_subplot(gs[0, 3])
        if 'lr' in history:
            ax4.plot(history['lr'], color='red')
            ax4.set_title('Learning Rate Schedule')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.set_yscale('log')
        else:
            ax4.text(0.5, 0.5, 'Learning Rate\nData Not Available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Learning Rate')
        ax4.grid(True, alpha=0.3)
        
        # 5. Confusion Matrix
        ax5 = fig.add_subplot(gs[1, :2])
        y_pred = evaluation_results['y_pred']
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
                   xticklabels=self.label_names, yticklabels=self.label_names)
        ax5.set_title('Confusion Matrix')
        ax5.set_xlabel('Predicted')
        ax5.set_ylabel('True')
        
        # 6. Classification Report Heatmap
        ax6 = fig.add_subplot(gs[1, 2:])
        report = evaluation_results['classification_report']
        
        # Create metrics matrix for visualization
        metrics_data = []
        metrics_labels = []
        for label in self.label_names:
            if label.lower() in report:
                metrics = report[label.lower()]
                metrics_data.append([
                    metrics['precision'],
                    metrics['recall'], 
                    metrics['f1-score']
                ])
                metrics_labels.append(label)
        
        if metrics_data:
            metrics_df = pd.DataFrame(
                metrics_data, 
                index=metrics_labels,
                columns=['Precision', 'Recall', 'F1-Score']
            )
            sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax6,
                       vmin=0, vmax=1, cbar_kws={'label': 'Score'})
            ax6.set_title('Per-Class Metrics')
        
        # 7. Sample Spectrograms
        ax7 = fig.add_subplot(gs[2, 0])
        if len(X_test) > 0:
            sample_idx = 0
            sample_spec = X_test[sample_idx][:, :, 0]  # Show first channel
            im = ax7.imshow(sample_spec, cmap='viridis', aspect='auto')
            ax7.set_title(f'Sample Spectrogram\n(True: {self.label_names[y_test[sample_idx]]})')
            ax7.set_xlabel('Time')
            ax7.set_ylabel('Frequency')
            plt.colorbar(im, ax=ax7)
        
        # 8. Prediction Confidence Distribution
        ax8 = fig.add_subplot(gs[2, 1])
        y_pred_probs = evaluation_results['y_pred_probs']
        max_probs = np.max(y_pred_probs, axis=1)
        ax8.hist(max_probs, bins=20, alpha=0.7, edgecolor='black')
        ax8.set_title('Prediction Confidence Distribution')
        ax8.set_xlabel('Max Probability')
        ax8.set_ylabel('Count')
        ax8.grid(True, alpha=0.3)
        
        # 9. Model Architecture Summary
        ax9 = fig.add_subplot(gs[2, 2:])
        ax9.axis('off')
        
        # Create architecture summary text
        total_params = self.model.count_params()
        trainable_params = sum(tf.size(p).numpy() for p in self.model.trainable_weights)
        
        arch_summary = f"""
MobileNet Model Architecture Summary
{'='*40}
Base Model: MobileNetV2 (ImageNet pretrained)
Input Shape: {self.spectrogram_shape} × 3 channels
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}

Training Strategy:
1. Transfer Learning from ImageNet
2. Two-phase training:
   - Phase 1: Frozen base, train head (20 epochs)
   - Phase 2: Fine-tune top layers (30 epochs)
3. Data Augmentation: Rotation, Zoom, Contrast, Brightness

Final Test Results:
Accuracy:  {evaluation_results['test_accuracy']:.4f}
Precision: {evaluation_results['test_precision']:.4f}
Recall:    {evaluation_results['test_recall']:.4f}
        """
        
        ax9.text(0.05, 0.95, arch_summary, transform=ax9.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle('MobileNet Training Results - FHR Hypoxia Classification', 
                    fontsize=16, fontweight='bold')
        
        # Save plot
        plt.savefig(self.results_path / 'mobilenet_training_history.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Training plots saved to: {self.results_path / 'mobilenet_training_history.png'}")
    
    def save_model(self):
        """Save trained MobileNet model"""
        model_path = self.models_path / 'mobilenet_classifier.h5'
        self.model.save(model_path)
        
        # Also save weights separately
        weights_path = self.models_path / 'mobilenet_weights.h5'
        self.model.save_weights(weights_path)
        
        print(f"✅ MobileNet model saved to: {model_path}")
        print(f"✅ Model weights saved to: {weights_path}")

def main():
    """Main training function"""
    print("🚀 Starting MobileNet Method Training...")
    
    trainer = MobileNetTrainer()
    
    try:
        # Load and preprocess data
        X_data, y_data = trainer.load_and_preprocess_data()
        
        if len(X_data) == 0:
            print("❌ No valid data found for training")
            return False
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_data, y_data, test_size=0.3, random_state=42, stratify=y_data
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"📊 Data split:")
        print(f"   Train: {len(X_train)} samples")
        print(f"   Validation: {len(X_val)} samples") 
        print(f"   Test: {len(X_test)} samples")
        
        # Build and train model
        model = trainer.build_mobilenet_model()
        history = trainer.train_model(X_train, y_train, X_val, y_val)
        
        # Evaluate model
        evaluation_results = trainer.evaluate_model(X_test, y_test)
        
        # Save plots and model
        trainer.save_training_plots(history, evaluation_results, X_test, y_test)
        trainer.save_model()
        
        print("✅ MobileNet training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during MobileNet training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()