import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class CTGGANTrainer:
    def __init__(self, base_path='/home/zainul/joki/HipoxiaDeepLearning'):
        self.base_path = Path(base_path)
        self.models_path = self.base_path / 'models' / 'gan_models'
        self.results_path = self.base_path / 'results' / 'training_plots'
        self.data_path = self.base_path / 'data' / 'gan'
        
        # Ensure directories exist
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Model parameters
        self.signal_length = 5000  # Standardize signal length
        self.noise_dim = 100
        self.num_classes = 3
        self.epochs = 1
        self.batch_size = 32
        
    def load_and_preprocess_data(self):
        """Load and preprocess data for GAN training"""
        print("🔄 Loading GAN training data...")
        
        X_data = np.load(self.data_path / 'X_data.npy', allow_pickle=True)
        y_data = np.load(self.data_path / 'y_data.npy')
        record_ids = np.load(self.data_path / 'record_ids.npy')
        
        # Preprocess signals to standard length
        X_processed = []
        y_processed = []
        
        for i, (signal, label) in enumerate(zip(X_data, y_data)):
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
            
            # Normalize FHR signal to [0, 1]
            signal_normalized = (signal_padded - 50) / 200  # FHR range ~50-250 bpm
            signal_normalized = np.clip(signal_normalized, 0, 1)
            
            X_processed.append(signal_normalized)
            y_processed.append(label)
        
        X_processed = np.array(X_processed)
        y_processed = np.array(y_processed)
        
        print(f"✅ Processed {len(X_processed)} signals")
        print(f"   Signal shape: {X_processed[0].shape}")
        print(f"   Label distribution: {np.bincount(y_processed)}")
        
        return X_processed, y_processed
    
    def build_generator(self):
        """Build CTGGAN Generator with Self-Attention and Residual Blocks"""
        
        # Input layers
        noise_input = layers.Input(shape=(self.noise_dim,), name='noise_input')
        label_input = layers.Input(shape=(1,), name='label_input')
        
        # Label embedding
        label_embedding = layers.Embedding(self.num_classes, 50)(label_input)
        label_embedding = layers.Flatten()(label_embedding)
        
        # Combine noise and label
        combined_input = layers.Concatenate()([noise_input, label_embedding])
        
        # Initial dense layer
        x = layers.Dense(256 * 125, activation='relu')(combined_input)
        x = layers.Reshape((125, 256))(x)
        x = layers.BatchNormalization()(x)
        
        # Residual Block 1
        x = self._residual_block(x, 128, kernel_size=4, strides=2)  # 250
        
        # Self-Attention Block
        x = self._self_attention_block(x)
        
        # Residual Block 2  
        x = self._residual_block(x, 64, kernel_size=4, strides=2)   # 500
        
        # Residual Block 3
        x = self._residual_block(x, 32, kernel_size=5, strides=2)   # 1000
        
        # Final layers to reach target length
        x = layers.Conv1DTranspose(16, kernel_size=5, strides=5, padding='same', activation='relu')(x)  # 5000
        x = layers.BatchNormalization()(x)
        
        # Output layer
        output = layers.Conv1D(1, kernel_size=7, padding='same', activation='sigmoid')(x)
        output = layers.Flatten()(output)
        
        model = keras.Model([noise_input, label_input], output, name='generator')
        return model
    
    def build_discriminator(self):
        """Build CTGGAN Discriminator"""
        
        # Signal input
        signal_input = layers.Input(shape=(self.signal_length,), name='signal_input')
        label_input = layers.Input(shape=(1,), name='label_input')
        
        # Reshape signal for conv layers
        x = layers.Reshape((self.signal_length, 1))(signal_input)
        
        # Label embedding and broadcasting
        label_embedding = layers.Embedding(self.num_classes, 50)(label_input)
        label_embedding = layers.Dense(self.signal_length)(label_embedding)
        label_embedding = layers.Reshape((self.signal_length, 1))(label_embedding)
        
        # Combine signal and label
        x = layers.Concatenate()([x, label_embedding])
        
        # Convolutional layers
        x = layers.Conv1D(32, kernel_size=5, strides=5, padding='same')(x)  # 1000
        x = layers.LeakyReLU(0.2)(x)
        
        # Residual Block 1
        x = self._discriminator_residual_block(x, 64, kernel_size=4, strides=2)  # 500
        
        # Self-Attention
        x = self._self_attention_block(x)
        
        # Residual Block 2
        x = self._discriminator_residual_block(x, 128, kernel_size=4, strides=2)  # 250
        
        # Residual Block 3
        x = self._discriminator_residual_block(x, 256, kernel_size=5, strides=5)  # 50
        
        # Global pooling and final layers
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.3)(x)
        
        # Real/fake output
        validity = layers.Dense(1, activation='sigmoid', name='validity')(x)
        
        model = keras.Model([signal_input, label_input], validity, name='discriminator')
        return model
    
    def _residual_block(self, x, filters, kernel_size=4, strides=2):
        """Residual block for generator"""
        shortcut = x
        
        # Main path
        x = layers.Conv1DTranspose(filters, kernel_size, strides=strides, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv1D(filters, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Shortcut connection
        if shortcut.shape[-1] != filters or strides != 1:
            shortcut = layers.Conv1DTranspose(filters, 1, strides=strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        
        return x
    
    def _discriminator_residual_block(self, x, filters, kernel_size=4, strides=2):
        """Residual block for discriminator"""
        shortcut = x
        
        # Main path
        x = layers.Conv1D(filters, kernel_size, strides=strides, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv1D(filters, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Shortcut connection
        if shortcut.shape[-1] != filters or strides != 1:
            shortcut = layers.Conv1D(filters, 1, strides=strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.Add()([x, shortcut])
        x = layers.LeakyReLU(0.2)(x)
        
        return x
    
    def _self_attention_block(self, x):
        """Self-attention mechanism"""
        batch_size, seq_len, channels = x.shape
        
        # Create query, key, value
        query = layers.Conv1D(channels // 8, 1)(x)  # [batch, seq_len, channels//8]
        key = layers.Conv1D(channels // 8, 1)(x)
        value = layers.Conv1D(channels, 1)(x)
        
        # Compute attention weights
        attention_weights = tf.matmul(query, key, transpose_b=True)  # [batch, seq_len, seq_len]
        attention_weights = tf.nn.softmax(attention_weights / tf.sqrt(tf.cast(channels // 8, tf.float32)))
        
        # Apply attention
        attended = tf.matmul(attention_weights, value)  # [batch, seq_len, channels]
        
        # Residual connection with learnable scaling using Dense layer
        gamma_layer = layers.Dense(1, use_bias=False, kernel_initializer='zeros', name='attention_gamma')
        gamma = gamma_layer(tf.reduce_mean(attended, axis=[1, 2], keepdims=True))
        gamma = tf.broadcast_to(gamma, tf.shape(attended))
        output = gamma * attended + x
        
        return output
    
    def build_combined_model(self):
        """Build combined GAN model"""
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        # Compile discriminator
        self.discriminator.compile(
            optimizer=keras.optimizers.Adam(0.0002, 0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Build combined model
        noise_input = layers.Input(shape=(self.noise_dim,))
        label_input = layers.Input(shape=(1,))
        
        generated_signal = self.generator([noise_input, label_input])
        
        # For combined model, discriminator is not trainable
        self.discriminator.trainable = False
        validity = self.discriminator([generated_signal, label_input])
        
        self.combined = keras.Model([noise_input, label_input], validity)
        self.combined.compile(
            optimizer=keras.optimizers.Adam(0.0002, 0.5),
            loss='binary_crossentropy'
        )
        
        print("✅ GAN models built successfully")
        print(f"   Generator parameters: {self.generator.count_params():,}")
        print(f"   Discriminator parameters: {self.discriminator.count_params():,}")
    
    def train(self, X_train, y_train):
        """Train CTGGAN"""
        print("🚀 Starting GAN training...")
        
        # Training history
        d_losses = []
        g_losses = []
        d_accuracies = []
        
        # Class balancing - oversample minority classes
        class_counts = np.bincount(y_train)
        max_count = np.max(class_counts)
        
        # Create balanced dataset
        X_balanced = []
        y_balanced = []
        
        for class_idx in range(self.num_classes):
            class_mask = y_train == class_idx
            class_samples = X_train[class_mask]
            class_labels = y_train[class_mask]
            
            # Oversample to match max class
            oversample_ratio = max_count // len(class_samples)
            remainder = max_count % len(class_samples)
            
            # Repeat samples
            oversampled_X = np.tile(class_samples, (oversample_ratio, 1))
            oversampled_y = np.tile(class_labels, oversample_ratio)
            
            # Add remainder
            if remainder > 0:
                indices = np.random.choice(len(class_samples), remainder, replace=False)
                oversampled_X = np.vstack([oversampled_X, class_samples[indices]])
                oversampled_y = np.hstack([oversampled_y, class_labels[indices]])
            
            X_balanced.append(oversampled_X)
            y_balanced.append(oversampled_y)
        
        X_balanced = np.vstack(X_balanced)
        y_balanced = np.hstack(y_balanced)
        
        # Shuffle
        indices = np.random.permutation(len(X_balanced))
        X_balanced = X_balanced[indices]
        y_balanced = y_balanced[indices]
        
        print(f"   Balanced dataset: {len(X_balanced)} samples")
        print(f"   New distribution: {np.bincount(y_balanced)}")
        
        for epoch in range(self.epochs):
            # Train discriminator
            self.discriminator.trainable = True
            
            # Select random batch
            idx = np.random.randint(0, X_balanced.shape[0], self.batch_size)
            real_signals = X_balanced[idx]
            real_labels = y_balanced[idx]
            
            # Generate fake signals
            noise = np.random.normal(0, 1, (self.batch_size, self.noise_dim))
            fake_labels = np.random.randint(0, self.num_classes, self.batch_size)
            fake_signals = self.generator.predict([noise, fake_labels], verbose=0)
            
            # Train discriminator on real and fake
            d_loss_real = self.discriminator.train_on_batch([real_signals, real_labels], np.ones((self.batch_size, 1)))
            d_loss_fake = self.discriminator.train_on_batch([fake_signals, fake_labels], np.zeros((self.batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train generator
            self.discriminator.trainable = False
            
            noise = np.random.normal(0, 1, (self.batch_size, self.noise_dim))
            gen_labels = np.random.randint(0, self.num_classes, self.batch_size)
            
            g_loss = self.combined.train_on_batch([noise, gen_labels], np.ones((self.batch_size, 1)))
            
            # Save losses
            d_losses.append(d_loss[0])
            g_losses.append(g_loss)
            d_accuracies.append(d_loss[1])
            
            # Print progress
            if epoch % 10 == 0:
                print(f"   Epoch {epoch:4d} | D Loss: {d_loss[0]:.4f} | D Acc: {d_loss[1]:.4f} | G Loss: {g_loss:.4f}")
        
        # Save training history
        history = {
            'discriminator_loss': d_losses,
            'generator_loss': g_losses,
            'discriminator_accuracy': d_accuracies
        }
        
        self.save_training_plots(history)
        
        print("✅ GAN training completed!")
        return history
    
    def save_training_plots(self, history):
        """Save training visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('CTGGAN Training History', fontsize=16)
        
        # Loss curves
        axes[0, 0].plot(history['discriminator_loss'], label='Discriminator Loss', color='blue')
        axes[0, 0].plot(history['generator_loss'], label='Generator Loss', color='red')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Discriminator accuracy
        axes[0, 1].plot(history['discriminator_accuracy'], color='green')
        axes[0, 1].set_title('Discriminator Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Generate sample signals for visualization
        noise = np.random.normal(0, 1, (3, self.noise_dim))
        sample_labels = np.array([0, 1, 2])  # One for each class
        sample_signals = self.generator.predict([noise, sample_labels], verbose=0)
        
        # Plot generated samples
        label_names = ['Normal', 'Suspect', 'Hypoxia']
        colors = ['green', 'orange', 'red']
        
        for i, (signal, label, color) in enumerate(zip(sample_signals, label_names, colors)):
            if i < 2:
                ax = axes[1, i]
                ax.plot(signal[:1000], color=color, alpha=0.7)  # Plot first 1000 points
                ax.set_title(f'Generated {label} FHR Signal (Sample)')
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Normalized FHR')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'gan_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Training plots saved to: {self.results_path / 'gan_training_history.png'}")
    
    def build_classifier(self):
        """Build classifier using generated and real data"""
        print("🔄 Building classifier...")
        
        # Load data
        X_train, y_train = self.load_and_preprocess_data()
        
        # Generate additional synthetic data for minority classes
        synthetic_X = []
        synthetic_y = []
        
        class_counts = np.bincount(y_train)
        max_count = np.max(class_counts)
        
        for class_idx in range(self.num_classes):
            current_count = class_counts[class_idx]
            needed = max_count - current_count
            
            if needed > 0:
                # Generate synthetic samples
                noise = np.random.normal(0, 1, (needed, self.noise_dim))
                labels = np.full(needed, class_idx)
                
                synthetic_signals = self.generator.predict([noise, labels], verbose=0)
                
                synthetic_X.append(synthetic_signals)
                synthetic_y.append(labels)
        
        if synthetic_X:
            synthetic_X = np.vstack(synthetic_X)
            synthetic_y = np.hstack(synthetic_y)
            
            # Combine real and synthetic data
            X_combined = np.vstack([X_train, synthetic_X])
            y_combined = np.hstack([y_train, synthetic_y])
            
            print(f"   Added {len(synthetic_X)} synthetic samples")
        else:
            X_combined = X_train
            y_combined = y_train
        
        # Build CNN-LSTM classifier
        input_layer = layers.Input(shape=(self.signal_length,))
        
        # Reshape for CNN
        x = layers.Reshape((self.signal_length, 1))(input_layer)
        
        # CNN layers
        x = layers.Conv1D(32, kernel_size=7, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        
        x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        
        x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        
        # LSTM layers
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(32)(x)
        x = layers.Dropout(0.3)(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.classifier = keras.Model(input_layer, output, name='fhr_classifier')
        
        self.classifier.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train classifier
        X_train_cls, X_val_cls, y_train_cls, y_val_cls = train_test_split(
            X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
        )
        
        history = self.classifier.fit(
            X_train_cls, y_train_cls,
            epochs=1,
            batch_size=32,
            validation_data=(X_val_cls, y_val_cls),
            verbose=1
        )
        
        # Save classifier training plot
        self.save_classifier_plots(history)
        
        print("✅ Classifier training completed!")
        return history
    
    def save_classifier_plots(self, history):
        """Save classifier training plots"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Classifier Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Classifier Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'gan_classifier_training.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Classifier plots saved to: {self.results_path / 'gan_classifier_training.png'}")
    
    def save_models(self):
        """Save trained models"""
        # Save generator
        self.generator.save(self.models_path / 'gan_generator.h5')
        
        # Save discriminator  
        self.discriminator.save(self.models_path / 'gan_discriminator.h5')
        
        # Save classifier
        self.classifier.save(self.models_path / 'gan_classifier.h5')
        
        print(f"✅ Models saved to: {self.models_path}")
    
def main():
    """Main training function"""
    print("🚀 Starting GAN Method Training...")
    
    trainer = CTGGANTrainer()
    
    try:
        # Load and preprocess data
        X_train, y_train = trainer.load_and_preprocess_data()
        
        # Build and train GAN
        trainer.build_combined_model()
        gan_history = trainer.train(X_train, y_train)
        
        # Build and train classifier
        classifier_history = trainer.build_classifier()
        
        # Save models
        trainer.save_models()
        
        print("✅ GAN method training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during GAN training: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()