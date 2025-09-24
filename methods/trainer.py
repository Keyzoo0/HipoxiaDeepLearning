#!/usr/bin/env python3
"""
Training Module
Handles model training with different methods and configurations
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from tensorflow import keras

class ModelTrainer:
    def __init__(self, base_path, data_handler, model_builder, visualizer):
        self.base_path = Path(base_path)
        self.models_path = self.base_path / 'models'
        self.data_handler = data_handler
        self.model_builder = model_builder
        self.visualizer = visualizer

        # Create directories
        self.models_path.mkdir(exist_ok=True)

        # Model storage
        self.model = None

    def get_training_parameters(self, method):
        """Get method-specific training parameters - OPTIMIZED FOR CONVERGENCE"""
        params = {
            'gan': {'epochs': 50, 'patience': 15, 'batch_size': 16},      # Reduced for faster training
            'mobilenet': {'epochs': 100, 'patience': 25, 'batch_size': 16},  # More epochs for stability
            'resnet': {'epochs': 120, 'patience': 30, 'batch_size': 16},     # More epochs for complex architecture
            'mdnn': {'epochs': 100, 'patience': 15, 'batch_size': 16}
        }
        return params.get(method, params['mdnn'])

    def calculate_class_weights(self, y_train):
        """Calculate enhanced class weights for imbalanced data"""
        # Get class distribution
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        print(f"📊 Training class distribution: {dict(zip(unique_classes, class_counts))}")

        if len(class_counts) == 0:
            return None

        min_count = class_counts.min()
        max_count = class_counts.max()

        if min_count == 0:
            print("⚠️ Detected empty class during training. Continuing with automatic balancing weights.")
        else:
            imbalance_ratio = max_count / min_count
            if imbalance_ratio <= 1.1:
                print("⚖️ Training labels are already balanced; skipping class weights.")
                return None

        # Compute balanced class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=y_train
        )

        # Enhanced weighting for hypoxia (most critical class)
        class_weight_dict = {}
        for i, weight in enumerate(class_weights):
            if unique_classes[i] == 2:  # Hypoxia class
                # Increase weight for critical hypoxia detection
                class_weight_dict[unique_classes[i]] = weight * 1.5
            else:
                class_weight_dict[unique_classes[i]] = weight

        print(f"🔧 Enhanced class weights: {class_weight_dict}")
        return class_weight_dict

    def setup_callbacks(self, method, patience):
        """Setup enhanced training callbacks for better convergence"""
        monitor = 'val_accuracy'
        mode = 'max'

        if method == 'gan':
            monitor = 'val_loss'
            mode = 'min'

        # Consistent early stopping strategies for all methods
        if method == 'mobilenet':
            early_stopping = keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=25,  # Increased patience for more epochs
                restore_best_weights=True,
                verbose=1,
                min_delta=0.002,
                mode=mode
            )
            lr_scheduler = keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                patience=10,
                factor=0.5,
                min_lr=1e-6,
                verbose=1,
                mode=mode
            )
        elif method == 'gan':
            early_stopping = keras.callbacks.EarlyStopping(
                monitor=monitor,  # Monitor loss instead of accuracy for stability
                patience=15,  # Reduced patience for faster training
                restore_best_weights=True,
                verbose=1,
                min_delta=0.002,  # Slightly higher threshold
                mode=mode
            )
            lr_scheduler = keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,  # Monitor loss instead of accuracy
                patience=8,  # Less patience before reducing LR
                factor=0.5,  # More aggressive reduction for faster convergence
                min_lr=1e-6,  # Lower min LR
                verbose=1,
                mode=mode
            )
        elif method == 'resnet':
            early_stopping = keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=30,  # More patience for complex architecture
                restore_best_weights=True,
                verbose=1,
                min_delta=0.002,
                mode=mode
            )
            lr_scheduler = keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                patience=12,
                factor=0.5,
                min_lr=1e-6,
                verbose=1,
                mode=mode
            )
        else:
            # MDNN maintains original settings
            early_stopping = keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=15,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.005,
                mode=mode
            )
            lr_scheduler = keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                patience=7,
                factor=0.5,
                min_lr=1e-6,
                verbose=1,
                mode=mode
            )

        # Unified smart callbacks with proper early stopping
        callbacks = [
            early_stopping,  # Will stop when model is good enough
            lr_scheduler,   # Will adjust learning rate dynamically
            keras.callbacks.ModelCheckpoint(
                str(self.models_path / f'{method}_multimodal_best_weights.h5'),
                save_best_only=True,
                monitor=monitor,
                verbose=1,
                mode=mode
            )
        ]
        return callbacks

    def train_model(self, method='mdnn'):
        """Train the multimodal model with specified method"""
        method_display = self.model_builder.get_method_display_name(method)
        print(f"🚀 Starting {method_display} multimodal training...")

        # Generate dataset if not available
        if self.data_handler.X_signals is None:
            self.data_handler.generate_multimodal_dataset()

        # Prepare data
        (X_signals_train, X_clinical_train, y_train,
         X_signals_val, X_clinical_val, y_val,
         X_signals_test, X_clinical_test, y_test) = self.data_handler.prepare_data_for_training()

        # Apply data augmentation to training data
        X_signals_train, X_clinical_train, y_train = self.data_handler.apply_data_augmentation(
            X_signals_train, X_clinical_train, y_train
        )

        # Build model
        self.model = self.model_builder.build_multimodal_model(X_clinical_train.shape[1], method)

        # Calculate class weights
        class_weight_dict = self.calculate_class_weights(y_train)

        # Get training parameters
        train_params = self.get_training_parameters(method)

        # Setup callbacks
        callbacks = self.setup_callbacks(method, train_params['patience'])

        # Train model
        print(f"🔄 Training {method_display} for {train_params['epochs']} epochs...")
        history = self.model.fit(
            [X_signals_train, X_clinical_train], y_train,
            validation_data=([X_signals_val, X_clinical_val], y_val),
            epochs=train_params['epochs'],
            batch_size=train_params['batch_size'],
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate model
        print("🔍 Evaluating model...")
        test_loss, test_accuracy = self.model.evaluate(
            [X_signals_test, X_clinical_test], y_test, verbose=0
        )

        # Predictions for detailed metrics
        y_pred_proba = self.model.predict([X_signals_test, X_clinical_test], verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Print results
        print(f"\n📊 {method_display} Training Results:")
        print(f"   Test Accuracy: {test_accuracy:.4f}")
        print(f"   Test Loss: {test_loss:.4f}")

        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.data_handler.label_names))

        # Generate visualizations
        self.visualizer.generate_comprehensive_training_analysis(
            method, history, y_test, y_pred, y_pred_proba, test_accuracy, test_loss
        )

        # Save model as PKL
        self.save_model_pkl(method)

        # Generate standard training plots
        self.visualizer.plot_training_results(history, y_test, y_pred, method)

        return history, test_accuracy

    def save_model_pkl(self, method):
        """Save model as PKL file"""
        model_data = {
            'model': self.model,
            'method': method,
            'clinical_scaler': self.data_handler.clinical_scaler,
            'signal_length': self.data_handler.signal_length,
            'num_classes': len(self.data_handler.label_names),
            'label_names': self.data_handler.label_names,
            'label_map': self.data_handler.label_map
        }

        pkl_path = self.models_path / f'{method}_multimodal_hypoxia_detector.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ Model saved as PKL: {pkl_path}")

    def load_model_from_pkl(self, method='mdnn'):
        """Load trained model from PKL file"""
        pkl_path = self.models_path / f'{method}_multimodal_hypoxia_detector.pkl'

        if not pkl_path.exists():
            # Fallback to old naming convention
            old_pkl_path = self.models_path / 'multimodal_hypoxia_detector.pkl'
            if old_pkl_path.exists():
                pkl_path = old_pkl_path
            else:
                raise FileNotFoundError(f"Model PKL not found: {pkl_path}")

        try:
            with open(pkl_path, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.data_handler.clinical_scaler = model_data['clinical_scaler']
            self.data_handler.signal_length = model_data['signal_length']
            self.data_handler.label_names = model_data['label_names']
            self.data_handler.label_map = model_data['label_map']
        except Exception as e:
            if "focal_loss_fixed" in str(e):
                print("⚠️  Model serialization issue detected. Attempting to rebuild model...")
                raise Exception("Model serialization issue requires retraining. Please retrain the model.")
            else:
                raise e

        method_display = self.model_builder.get_method_display_name(method)
        print(f"✅ {method_display} model loaded from PKL: {pkl_path}")
        return self.model

    def get_available_methods(self):
        """Get list of available trained methods"""
        available = []

        # Check new naming convention
        for method in ['mdnn', 'gan', 'mobilenet', 'resnet']:
            pkl_path = self.models_path / f'{method}_multimodal_hypoxia_detector.pkl'
            if pkl_path.exists():
                available.append(method)

        # Check old naming convention (for backward compatibility)
        old_pkl_path = self.models_path / 'multimodal_hypoxia_detector.pkl'
        if old_pkl_path.exists() and 'mdnn' not in available:
            available.append('mdnn')

        return available
