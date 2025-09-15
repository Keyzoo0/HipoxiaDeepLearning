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
            # Enhanced MobileNet-inspired architecture
            x_signal = layers.Reshape((self.signal_length, 1))(signal_input)

            # Multiple conv blocks for better feature extraction
            x_signal = layers.Conv1D(64, 7, activation='relu', padding='same')(x_signal)
            x_signal = layers.BatchNormalization()(x_signal)
            x_signal = layers.MaxPooling1D(2)(x_signal)

            x_signal = layers.DepthwiseConv1D(5, padding='same')(x_signal)
            x_signal = layers.Conv1D(128, 1, activation='relu')(x_signal)
            x_signal = layers.BatchNormalization()(x_signal)
            x_signal = layers.MaxPooling1D(2)(x_signal)

            x_signal = layers.DepthwiseConv1D(3, padding='same')(x_signal)
            x_signal = layers.Conv1D(256, 1, activation='relu')(x_signal)
            x_signal = layers.BatchNormalization()(x_signal)

            x_signal = layers.GlobalAveragePooling1D()(x_signal)
            x_signal = layers.Dense(256, activation='relu')(x_signal)
            x_signal = layers.BatchNormalization()(x_signal)
            x_signal = layers.Dropout(0.4)(x_signal)
            x_signal = layers.Dense(128, activation='relu')(x_signal)
            x_signal = layers.BatchNormalization()(x_signal)
            x_signal = layers.Dropout(0.3)(x_signal)
            x_signal = layers.Dense(64, activation='relu')(x_signal)
            x_signal = layers.Dropout(0.2)(x_signal)

        elif method == 'resnet':
            # Enhanced ResNet architecture with deeper residual connections
            x_signal = layers.Reshape((self.signal_length, 1))(signal_input)

            # Initial convolution with batch norm
            x_signal = layers.Conv1D(128, 15, activation='relu', padding='same')(x_signal)
            x_signal = layers.BatchNormalization()(x_signal)
            x_signal = layers.MaxPooling1D(4, strides=2)(x_signal)

            # Enhanced residual blocks with batch normalization
            for filters in [128, 256, 512, 256]:
                # Residual block with batch normalization
                residual = x_signal

                x_signal = layers.Conv1D(filters, 5, activation='relu', padding='same')(x_signal)
                x_signal = layers.BatchNormalization()(x_signal)
                x_signal = layers.Conv1D(filters, 5, activation='relu', padding='same')(x_signal)
                x_signal = layers.BatchNormalization()(x_signal)

                # Adjust residual connection if needed
                if residual.shape[-1] != filters:
                    residual = layers.Conv1D(filters, 1, padding='same')(residual)
                    residual = layers.BatchNormalization()(residual)

                x_signal = layers.Add()([x_signal, residual])
                x_signal = layers.Activation('relu')(x_signal)
                x_signal = layers.MaxPooling1D(2)(x_signal)

            x_signal = layers.GlobalAveragePooling1D()(x_signal)
            x_signal = layers.Dense(256, activation='relu')(x_signal)
            x_signal = layers.BatchNormalization()(x_signal)
            x_signal = layers.Dropout(0.4)(x_signal)
            x_signal = layers.Dense(128, activation='relu')(x_signal)
            x_signal = layers.BatchNormalization()(x_signal)
            x_signal = layers.Dropout(0.3)(x_signal)
            x_signal = layers.Dense(64, activation='relu')(x_signal)
            x_signal = layers.Dropout(0.2)(x_signal)

        elif method == 'gan':
            # Enhanced GAN-inspired feature extraction with convolutional layers
            x_signal = layers.Reshape((self.signal_length, 1))(signal_input)

            # Multi-scale feature extraction
            x_signal = layers.Conv1D(128, 11, activation='relu', padding='same')(x_signal)
            x_signal = layers.BatchNormalization()(x_signal)
            x_signal = layers.MaxPooling1D(3)(x_signal)

            x_signal = layers.Conv1D(256, 7, activation='relu', padding='same')(x_signal)
            x_signal = layers.BatchNormalization()(x_signal)
            x_signal = layers.MaxPooling1D(3)(x_signal)

            x_signal = layers.Conv1D(512, 5, activation='relu', padding='same')(x_signal)
            x_signal = layers.BatchNormalization()(x_signal)
            x_signal = layers.MaxPooling1D(2)(x_signal)

            x_signal = layers.Conv1D(256, 3, activation='relu', padding='same')(x_signal)
            x_signal = layers.BatchNormalization()(x_signal)

            # Global pooling and dense layers with more capacity
            x_signal = layers.GlobalAveragePooling1D()(x_signal)
            x_signal = layers.Dense(512, activation='relu')(x_signal)
            x_signal = layers.BatchNormalization()(x_signal)
            x_signal = layers.Dropout(0.5)(x_signal)
            x_signal = layers.Dense(256, activation='relu')(x_signal)
            x_signal = layers.BatchNormalization()(x_signal)
            x_signal = layers.Dropout(0.4)(x_signal)
            x_signal = layers.Dense(128, activation='relu')(x_signal)
            x_signal = layers.Dropout(0.3)(x_signal)
            x_signal = layers.Dense(64, activation='relu')(x_signal)
            x_signal = layers.Dropout(0.2)(x_signal)

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

        # Enhanced clinical branch - processes tabular medical data
        clinical_input = layers.Input(shape=(clinical_features_dim,), name='clinical_input')
        x_clinical = layers.Dense(64, activation='relu')(clinical_input)
        x_clinical = layers.BatchNormalization()(x_clinical)
        x_clinical = layers.Dropout(0.3)(x_clinical)
        x_clinical = layers.Dense(32, activation='relu')(x_clinical)
        x_clinical = layers.BatchNormalization()(x_clinical)
        x_clinical = layers.Dropout(0.2)(x_clinical)
        x_clinical = layers.Dense(16, activation='relu')(x_clinical)
        x_clinical = layers.BatchNormalization()(x_clinical)
        x_clinical = layers.Dropout(0.1)(x_clinical)

        # Enhanced fusion layer - combines both branches
        fusion = layers.Concatenate()([x_signal, x_clinical])
        x = layers.Dense(128, activation='relu')(fusion)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        # Classification layer
        output = layers.Dense(self.num_classes, activation='softmax', name='classification')(x)

        # Create model
        model = keras.Model(inputs=[signal_input, clinical_input], outputs=output)

        # Optimized learning rates for better convergence
        if method == 'gan':
            optimizer = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
        elif method == 'mobilenet':
            optimizer = keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999)
        elif method == 'resnet':
            optimizer = keras.optimizers.Adam(learning_rate=0.0003, beta_1=0.9, beta_2=0.999)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=0.0008, beta_1=0.9, beta_2=0.999)

        # Use weighted categorical crossentropy for better compatibility
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