#!/usr/bin/env python3
"""
Model Builder Module
Handles different neural network architectures for multimodal hypoxia detection
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

@keras.saving.register_keras_serializable()
def focal_loss_fixed(y_true, y_pred):
    """Standalone focal loss function for model serialization"""
    gamma = 2.
    alpha = 0.25
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

    # Convert sparse labels to categorical
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.one_hot(y_true, depth=3)  # 3 classes: Normal, Suspect, Hypoxia
    y_true = tf.cast(y_true, tf.float32)

    p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
    focal_weight = alpha_t * tf.pow((1 - p_t), gamma)

    loss = -focal_weight * tf.math.log(p_t)
    return tf.reduce_mean(tf.reduce_sum(loss, axis=1))

class ModelBuilder:
    def __init__(self, signal_length=5000, num_classes=3):
        self.signal_length = signal_length
        self.num_classes = num_classes

        # Method descriptions for scientific presentation
        self.method_names = {
            'mdnn': 'MDNN',  # Multimodal Dense Neural Network
            'gan': 'GAN',      # Generative Adversarial Network
            'mobilenet': 'MobileNet',  # MobileNet Architecture
            'resnet': 'ResNet'         # Residual Neural Network
        }

        self.method_descriptions = {
            'mdnn': 'Multimodal Dense Neural Network',
            'gan': 'GAN-Enhanced Feature Extraction',
            'mobilenet': 'MobileNet-Based CNN Architecture',
            'resnet': 'Deep Residual Neural Network'
        }

    def get_method_display_name(self, method):
        """Get the scientific display name for a method"""
        return self.method_names.get(method, method.upper())

    def get_method_description(self, method):
        """Get the description for a method"""
        return self.method_descriptions.get(method, "Unknown method")

    def build_multimodal_model(self, clinical_features_dim, method='mdnn'):
        """Build multimodal neural network with different architectures"""
        method_display = self.get_method_display_name(method)
        print(f"🔧 Building {method_display} multimodal model...")

        # Signal branch - processes FHR temporal patterns
        signal_input = layers.Input(shape=(self.signal_length,), name='signal_input')

        if method == 'mobilenet':
            # Simplified MobileNet for small dataset (552 samples)
            x_signal = layers.Reshape((self.signal_length, 1))(signal_input)

            # Simplified feature extraction with larger kernels
            x_signal = layers.Conv1D(64, 51, activation='relu', padding='same')(x_signal)  # Large kernel for temporal patterns
            x_signal = layers.BatchNormalization()(x_signal)
            x_signal = layers.MaxPooling1D(10)(x_signal)  # Less aggressive pooling
            x_signal = layers.Dropout(0.2)(x_signal)

            # Second conv layer with medium kernel
            x_signal = layers.Conv1D(128, 25, activation='relu', padding='same')(x_signal)
            x_signal = layers.BatchNormalization()(x_signal)
            x_signal = layers.MaxPooling1D(5)(x_signal)
            x_signal = layers.Dropout(0.2)(x_signal)

            # Final conv layer
            x_signal = layers.Conv1D(64, 11, activation='relu', padding='same')(x_signal)
            x_signal = layers.BatchNormalization()(x_signal)
            x_signal = layers.GlobalAveragePooling1D()(x_signal)
            x_signal = layers.Dropout(0.3)(x_signal)

            # Simplified dense layers following MDNN success pattern
            x_signal = layers.Dense(128, activation='relu')(x_signal)
            x_signal = layers.BatchNormalization()(x_signal)
            x_signal = layers.Dropout(0.3)(x_signal)
            x_signal = layers.Dense(64, activation='relu')(x_signal)
            x_signal = layers.Dropout(0.2)(x_signal)

        elif method == 'resnet':
            # Simplified ResNet architecture for small dataset
            x_signal = layers.Reshape((self.signal_length, 1))(signal_input)

            # Initial convolution with large kernel for temporal patterns
            x_signal = layers.Conv1D(64, 31, activation='relu', padding='same')(x_signal)
            x_signal = layers.BatchNormalization()(x_signal)
            x_signal = layers.MaxPooling1D(8)(x_signal)
            x_signal = layers.Dropout(0.2)(x_signal)

            # Single simplified residual block
            residual = x_signal
            x_signal = layers.Conv1D(128, 15, activation='relu', padding='same')(x_signal)
            x_signal = layers.BatchNormalization()(x_signal)
            x_signal = layers.Conv1D(128, 15, padding='same')(x_signal)
            x_signal = layers.BatchNormalization()(x_signal)

            # Adjust residual connection
            residual = layers.Conv1D(128, 1, padding='same')(residual)
            residual = layers.BatchNormalization()(residual)

            x_signal = layers.Add()([x_signal, residual])
            x_signal = layers.Activation('relu')(x_signal)
            x_signal = layers.MaxPooling1D(4)(x_signal)
            x_signal = layers.Dropout(0.2)(x_signal)

            x_signal = layers.GlobalAveragePooling1D()(x_signal)

            # Dense layers following MDNN success pattern
            x_signal = layers.Dense(128, activation='relu')(x_signal)
            x_signal = layers.BatchNormalization()(x_signal)
            x_signal = layers.Dropout(0.3)(x_signal)
            x_signal = layers.Dense(64, activation='relu')(x_signal)
            x_signal = layers.Dropout(0.2)(x_signal)

        elif method == 'gan':
            # Ultra-simplified GAN architecture to prevent overfitting
            x_signal = layers.Reshape((self.signal_length, 1))(signal_input)

            # Single CNN layer with very large kernel to capture temporal patterns
            x_signal = layers.Conv1D(64, 101, activation='relu', padding='same')(x_signal)  # Very large kernel
            x_signal = layers.BatchNormalization()(x_signal)
            x_signal = layers.MaxPooling1D(20)(x_signal)  # More aggressive pooling to reduce overfitting
            x_signal = layers.Dropout(0.4)(x_signal)  # Higher dropout

            x_signal = layers.GlobalAveragePooling1D()(x_signal)
            x_signal = layers.Dropout(0.5)(x_signal)  # Strong regularization

            # Very simple dense layers to match MDNN but with more regularization
            x_signal = layers.Dense(64, activation='relu')(x_signal)
            x_signal = layers.BatchNormalization()(x_signal)
            x_signal = layers.Dropout(0.5)(x_signal)
            x_signal = layers.Dense(32, activation='relu')(x_signal)  # Smaller layer
            x_signal = layers.Dropout(0.4)(x_signal)

        else:  # enhanced mdnn method
            x_signal = layers.Dense(256, activation='relu')(signal_input)
            x_signal = layers.BatchNormalization()(x_signal)
            x_signal = layers.Dropout(0.4)(x_signal)
            x_signal = layers.Dense(128, activation='relu')(x_signal)
            x_signal = layers.BatchNormalization()(x_signal)
            x_signal = layers.Dropout(0.3)(x_signal)
            x_signal = layers.Dense(64, activation='relu')(x_signal)
            x_signal = layers.BatchNormalization()(x_signal)
            x_signal = layers.Dropout(0.2)(x_signal)

        # Enhanced but practical clinical branch
        clinical_input = layers.Input(shape=(clinical_features_dim,), name='clinical_input')
        x_clinical = layers.Dense(48, activation='relu')(clinical_input)
        x_clinical = layers.BatchNormalization()(x_clinical)
        x_clinical = layers.Dropout(0.25)(x_clinical)
        x_clinical = layers.Dense(32, activation='relu')(x_clinical)
        x_clinical = layers.BatchNormalization()(x_clinical)
        x_clinical = layers.Dropout(0.2)(x_clinical)
        x_clinical = layers.Dense(16, activation='relu')(x_clinical)
        x_clinical = layers.BatchNormalization()(x_clinical)
        x_clinical = layers.Dropout(0.15)(x_clinical)

        # Simplified fusion for GAN method to prevent overfitting
        fusion = layers.Concatenate()([x_signal, x_clinical])

        if method == 'gan':
            # Ultra-simple fusion for GAN - no attention mechanism
            x = layers.Dense(48, activation='relu')(fusion)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.6)(x)  # Very high dropout
            x = layers.Dense(24, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.5)(x)
        else:
            # Standard fusion with attention for other methods
            attention_weights = layers.Dense(fusion.shape[-1], activation='softmax')(fusion)
            fusion_attended = layers.Multiply()([fusion, attention_weights])

            # Enhanced dense layers with proven dropout schedule
            x = layers.Dense(144, activation='relu')(fusion_attended)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.4)(x)
            x = layers.Dense(96, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(48, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.25)(x)
            x = layers.Dense(32, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)

        # Classification layer
        output = layers.Dense(self.num_classes, activation='softmax', name='classification')(x)

        # Create model
        model = keras.Model(inputs=[signal_input, clinical_input], outputs=output)

        # Optimized learning rates matching MDNN's successful pattern
        if method == 'gan':
            # Lower learning rate for GAN stability and better convergence
            optimizer = keras.optimizers.Adam(learning_rate=0.0003, beta_1=0.9, beta_2=0.999)
        elif method == 'mobilenet':
            # Slightly higher for simplified architecture
            optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        elif method == 'resnet':
            # Conservative learning rate for residual connections
            optimizer = keras.optimizers.Adam(learning_rate=0.0006, beta_1=0.9, beta_2=0.999)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=0.0008, beta_1=0.9, beta_2=0.999)

        # Use stable loss function for all methods
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        method_display = self.get_method_display_name(method)
        print(f"✅ {method_display} model built successfully:")
        print(f"   Total parameters: {model.count_params():,}")

        return model

    def focal_loss(self, gamma=2., alpha=0.25):
        """Focal Loss implementation for handling class imbalance"""
        def focal_loss_fixed_local(y_true, y_pred):
            epsilon = K.epsilon()
            y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

            # Convert sparse labels to categorical
            y_true = tf.cast(y_true, tf.int32)
            y_true = tf.one_hot(y_true, depth=3)  # Using hardcoded 3 for num_classes
            y_true = tf.cast(y_true, tf.float32)

            p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
            focal_weight = alpha_t * tf.pow((1 - p_t), gamma)

            loss = -focal_weight * tf.math.log(p_t)
            return tf.reduce_mean(tf.reduce_sum(loss, axis=1))

        return focal_loss_fixed_local