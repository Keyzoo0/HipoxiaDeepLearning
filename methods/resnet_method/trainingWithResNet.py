import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ResNetTrainer:
    def __init__(self, base_path='/home/zainul/joki/HipoxiaDeepLearning'):
        self.base_path = Path(base_path)
        self.models_path = self.base_path / 'models' / 'resnet_models'
        self.results_path = self.base_path / 'results' / 'training_plots'
        self.data_path = self.base_path / 'data' / 'resnet'
        
        # Ensure directories exist
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Model parameters
        self.signal_length = 5000  # Standardize signal length
        self.num_classes = 3
        self.epochs = 100
        self.batch_size = 32
        
        self.label_names = ['Normal', 'Suspect', 'Hypoxia']
        
    def load_and_preprocess_data(self):
        """Load and preprocess data for ResNet training"""
        print("🔄 Loading ResNet training data...")
        
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
                
                # Normalize FHR signal
                signal_normalized = (signal_padded - np.mean(signal_padded)) / (np.std(signal_padded) + 1e-8)
                
                # Z-score normalization
                signal_standardized = (signal_normalized - np.mean(signal_normalized)) / (np.std(signal_normalized) + 1e-8)
                
                X_processed.append(signal_standardized)
                y_processed.append(label)
                
            except Exception as e:
                print(f"⚠️ Error processing signal {i}: {e}")
                continue
        
        X_processed = np.array(X_processed)
        y_processed = np.array(y_processed)
        
        print(f"✅ Processed {len(X_processed)} signals")
        print(f"   Signal shape: {X_processed[0].shape}")
        print(f"   Label distribution: {np.bincount(y_processed)}")
        
        return X_processed, y_processed
    
    def residual_block_1d(self, x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
        """1D Residual block for ResNet"""
        
        if conv_shortcut:
            shortcut = layers.Conv1D(filters, 1, strides=stride, padding='valid', name=name + '_0_conv')(x)
            shortcut = layers.BatchNormalization(name=name + '_0_bn')(shortcut)
        else:
            if stride == 1:
                shortcut = x
            else:
                # Use average pooling to match dimensions
                shortcut = layers.AveragePooling1D(stride, padding='same')(x)
                # Zero-pad to match filter dimensions
                shortcut = layers.Lambda(
                    lambda x: tf.pad(x, [[0,0], [0,0], [0, filters - x.shape[-1]]])
                )(shortcut)
        
        # Main path
        x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same', 
                         name=name + '_1_conv')(x)
        x = layers.BatchNormalization(name=name + '_1_bn')(x)
        x = layers.Activation('relu', name=name + '_1_relu')(x)
        
        x = layers.Conv1D(filters, kernel_size, padding='same', 
                         name=name + '_2_conv')(x)
        x = layers.BatchNormalization(name=name + '_2_bn')(x)
        
        # Add shortcut
        x = layers.Add(name=name + '_add')([shortcut, x])
        x = layers.Activation('relu', name=name + '_out')(x)
        
        return x
    
    def resnet_stack_1d(self, x, filters, blocks, stride=1, name=None):
        """Stack of 1D residual blocks"""
        x = self.residual_block_1d(x, filters, stride=stride, conv_shortcut=True, name=name + '_block1')
        
        for i in range(2, blocks + 1):
            x = self.residual_block_1d(x, filters, name=name + f'_block{i}')
        
        return x
    
    def build_resnet1d_model(self):
        """Build 1D ResNet model for FHR classification"""
        print("🔧 Building 1D ResNet model...")
        
        # Input layer
        input_layer = layers.Input(shape=(self.signal_length,), name='input_signal')
        
        # Reshape for 1D convolutions
        x = layers.Reshape((self.signal_length, 1))(input_layer)
        
        # Initial convolution
        x = layers.Conv1D(64, 7, strides=2, padding='same', name='conv1_conv')(x)
        x = layers.BatchNormalization(name='conv1_bn')(x)
        x = layers.Activation('relu', name='conv1_relu')(x)
        x = layers.MaxPooling1D(3, strides=2, padding='same', name='pool1_pool')(x)
        
        # ResNet blocks
        x = self.resnet_stack_1d(x, 64, 3, stride=1, name='conv2')      # Output: (batch, 1250, 64)
        x = self.resnet_stack_1d(x, 128, 4, stride=2, name='conv3')     # Output: (batch, 625, 128)
        x = self.resnet_stack_1d(x, 256, 6, stride=2, name='conv4')     # Output: (batch, 313, 256)
        x = self.resnet_stack_1d(x, 512, 3, stride=2, name='conv5')     # Output: (batch, 157, 512)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D(name='avg_pool')(x)
        
        # Dropout for regularization
        x = layers.Dropout(0.5, name='dropout')(x)
        
        # Classification layers
        x = layers.Dense(256, activation='relu', name='fc1')(x)
        x = layers.BatchNormalization(name='fc1_bn')(x)
        x = layers.Dropout(0.3, name='fc1_dropout')(x)
        
        x = layers.Dense(128, activation='relu', name='fc2')(x)
        x = layers.BatchNormalization(name='fc2_bn')(x)
        x = layers.Dropout(0.3, name='fc2_dropout')(x)
        
        # Output layer
        output = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        # Create model
        model = keras.Model(input_layer, output, name='resnet1d_fhr_classifier')
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        
        print(f"✅ 1D ResNet model built successfully")
        print(f"   Total parameters: {model.count_params():,}")
        
        return model
    
    def create_data_augmentation(self, X):
        """Create augmented training data"""
        X_augmented = []
        
        for signal in X:
            # Original signal
            X_augmented.append(signal)
            
            # Add noise
            noise_factor = 0.05
            noisy_signal = signal + noise_factor * np.random.normal(0, 1, signal.shape)
            X_augmented.append(noisy_signal)
            
            # Time shift (circular shift)
            shift = np.random.randint(-100, 100)
            shifted_signal = np.roll(signal, shift)
            X_augmented.append(shifted_signal)
            
            # Scale variation
            scale_factor = np.random.uniform(0.8, 1.2)
            scaled_signal = signal * scale_factor
            X_augmented.append(scaled_signal)
        
        return np.array(X_augmented)
    
    def train_model(self, X_train, y_train, X_val, y_val, use_augmentation=True):
        """Train ResNet model"""
        print("🚀 Starting 1D ResNet training...")
        
        # Data augmentation for training set
        if use_augmentation:
            print("🔄 Applying data augmentation...")
            
            # Separate by class for balanced augmentation
            X_aug_list = []
            y_aug_list = []
            
            for class_idx in range(self.num_classes):
                class_mask = y_train == class_idx
                class_X = X_train[class_mask]
                class_y = y_train[class_mask]
                
                # Apply augmentation
                if len(class_X) > 0:
                    aug_X = self.create_data_augmentation(class_X)
                    aug_y = np.repeat(class_y, 4)  # Each signal becomes 4 (original + 3 augmented)
                    
                    X_aug_list.append(aug_X)
                    y_aug_list.append(aug_y)
            
            # Combine all augmented data
            X_train_aug = np.vstack(X_aug_list)
            y_train_aug = np.hstack(y_aug_list)
            
            # Shuffle
            indices = np.random.permutation(len(X_train_aug))
            X_train_aug = X_train_aug[indices]
            y_train_aug = y_train_aug[indices]
            
            print(f"   Training set size after augmentation: {len(X_train_aug)}")
        else:
            X_train_aug = X_train
            y_train_aug = y_train
        
        # Define callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                self.models_path / 'resnet_best_weights.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train_aug, y_train_aug,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ 1D ResNet training completed!")
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate trained ResNet model"""
        print("🔍 Evaluating 1D ResNet model...")
        
        # Make predictions
        y_pred_probs = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate metrics
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            X_test, y_test, verbose=0
        )
        
        # Classification report
        report = classification_report(
            y_test, y_pred, 
            target_names=self.label_names,
            output_dict=True
        )
        
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
    
    def visualize_feature_maps(self, X_sample):
        """Visualize intermediate feature maps"""
        # Get intermediate layer outputs
        layer_outputs = []
        layer_names = []
        
        for layer in self.model.layers:
            if isinstance(layer, layers.Conv1D) and 'conv' in layer.name:
                layer_outputs.append(layer.output)
                layer_names.append(layer.name)
        
        if not layer_outputs:
            return None, None
        
        # Create model to extract features
        feature_model = keras.Model(self.model.input, layer_outputs)
        
        # Get features for sample
        features = feature_model.predict(X_sample.reshape(1, -1), verbose=0)
        
        return features, layer_names
    
    def save_training_plots(self, history, evaluation_results, X_test, y_test):
        """Save comprehensive training visualization plots"""
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
        
        # 1. Training History - Accuracy
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Training History - Loss
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(history.history['loss'], label='Training Loss', color='blue')
        ax2.plot(history.history['val_loss'], label='Validation Loss', color='orange')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Precision and Recall
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(history.history['precision'], label='Training Precision', color='green')
        ax3.plot(history.history['val_precision'], label='Validation Precision', color='red')
        ax3.plot(history.history['recall'], label='Training Recall', color='purple', linestyle='--')
        ax3.plot(history.history['val_recall'], label='Validation Recall', color='brown', linestyle='--')
        ax3.set_title('Precision & Recall')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Metric Value')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Learning Rate Schedule
        ax4 = fig.add_subplot(gs[0, 3])
        if 'lr' in history.history:
            ax4.plot(history.history['lr'], color='red')
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
        
        # Create metrics matrix
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
        
        # 7. Sample Signal Analysis
        ax7 = fig.add_subplot(gs[2, 0])
        if len(X_test) > 0:
            sample_idx = 0
            sample_signal = X_test[sample_idx]
            ax7.plot(sample_signal[:1000], alpha=0.8)  # Plot first 1000 points
            ax7.set_title(f'Sample FHR Signal\n(True: {self.label_names[y_test[sample_idx]]})')
            ax7.set_xlabel('Time Steps')
            ax7.set_ylabel('Normalized FHR')
            ax7.grid(True, alpha=0.3)
        
        # 8. Feature Map Visualization
        ax8 = fig.add_subplot(gs[2, 1])
        try:
            if len(X_test) > 0:
                features, layer_names = self.visualize_feature_maps(X_test[0])
                if features:
                    # Visualize first conv layer features
                    first_conv_features = features[0][0]  # First sample, first layer
                    # Show average across all filters
                    avg_features = np.mean(first_conv_features, axis=-1)
                    ax8.plot(avg_features[:1000], alpha=0.8)
                    ax8.set_title('Conv1 Feature Map (Average)')
                    ax8.set_xlabel('Time Steps')
                    ax8.set_ylabel('Feature Value')
                else:
                    ax8.text(0.5, 0.5, 'Feature Maps\nNot Available', 
                            ha='center', va='center', transform=ax8.transAxes)
            ax8.grid(True, alpha=0.3)
        except:
            ax8.text(0.5, 0.5, 'Feature Visualization\nError', 
                    ha='center', va='center', transform=ax8.transAxes)
            ax8.set_title('Feature Maps')
        
        # 9. Prediction Confidence Distribution
        ax9 = fig.add_subplot(gs[2, 2])
        y_pred_probs = evaluation_results['y_pred_probs']
        max_probs = np.max(y_pred_probs, axis=1)
        ax9.hist(max_probs, bins=20, alpha=0.7, edgecolor='black', color='purple')
        ax9.set_title('Prediction Confidence Distribution')
        ax9.set_xlabel('Max Probability')
        ax9.set_ylabel('Count')
        ax9.grid(True, alpha=0.3)
        
        # 10. Model Architecture Summary
        ax10 = fig.add_subplot(gs[2, 3])
        ax10.axis('off')
        
        total_params = self.model.count_params()
        
        arch_summary = f"""
ResNet1D Architecture
{'='*18}
Blocks: 3+4+6+3 residual blocks
Filters: 64→128→256→512
Input: {self.signal_length} samples
Total Params: {total_params:,}

Key Features:
• 1D Convolutions
• Residual Connections  
• Batch Normalization
• Global Average Pooling
• Dense Classification Head

Data Augmentation:
• Noise injection
• Time shifting
• Amplitude scaling
        """
        
        ax10.text(0.05, 0.95, arch_summary, transform=ax10.transAxes,
                 fontsize=8, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8))
        
        # 11. Training Summary
        ax11 = fig.add_subplot(gs[3, :])
        ax11.axis('off')
        
        final_summary = f"""
1D RESNET TRAINING SUMMARY - FHR HYPOXIA CLASSIFICATION
{'='*80}
Final Test Results:
• Accuracy:  {evaluation_results['test_accuracy']:.4f}
• Precision: {evaluation_results['test_precision']:.4f}  
• Recall:    {evaluation_results['test_recall']:.4f}

Training Configuration:
• Epochs: {self.epochs} | Batch Size: {self.batch_size} | Signal Length: {self.signal_length}
• Data Augmentation: Noise + Time Shift + Amplitude Scaling
• Optimizer: Adam with ReduceLROnPlateau scheduling
• Early Stopping: Patience=15 on validation accuracy

Model: Custom 1D ResNet adapted for temporal FHR signal analysis
Architecture: Deep residual network with skip connections for gradient flow
        """
        
        ax11.text(0.05, 0.95, final_summary, transform=ax11.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        plt.suptitle('1D ResNet Training Results - FHR Hypoxia Classification', 
                    fontsize=16, fontweight='bold')
        
        # Save plot
        plt.savefig(self.results_path / 'resnet_training_history.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Training plots saved to: {self.results_path / 'resnet_training_history.png'}")
    
    def save_model(self):
        """Save trained ResNet model"""
        model_path = self.models_path / 'resnet_classifier.h5'
        self.model.save(model_path)
        
        # Also save weights separately
        weights_path = self.models_path / 'resnet_weights.h5'
        self.model.save_weights(weights_path)
        
        print(f"✅ ResNet model saved to: {model_path}")
        print(f"✅ Model weights saved to: {weights_path}")

def main():
    """Main training function"""
    print("🚀 Starting 1D ResNet Method Training...")
    
    trainer = ResNetTrainer()
    
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
        model = trainer.build_resnet1d_model()
        history = trainer.train_model(X_train, y_train, X_val, y_val)
        
        # Evaluate model
        evaluation_results = trainer.evaluate_model(X_test, y_test)
        
        # Save plots and model
        trainer.save_training_plots(history, evaluation_results, X_test, y_test)
        trainer.save_model()
        
        print("✅ 1D ResNet training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during ResNet training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()