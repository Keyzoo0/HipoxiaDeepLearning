#!/usr/bin/env python3
"""
Unified Multimodal Hypoxia Detection System
Combines FHR signals with clinical parameters for enhanced prediction accuracy
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
warnings.filterwarnings('ignore')

class MultimodalHypoxiaDetector:
    def __init__(self, base_path='/home/zainul/joki/HipoxiaDeepLearning'):
        self.base_path = Path(base_path)
        self.processed_data_path = self.base_path / 'processed_data'
        self.signals_path = self.processed_data_path / 'signals'
        self.models_path = self.base_path / 'models'
        self.results_path = self.base_path / 'results'

        # Model parameters
        self.signal_length = 5000
        self.num_classes = 3
        self.label_names = ['Normal', 'Suspect', 'Hypoxia']
        self.label_map = {'normal': 0, 'suspect': 1, 'hypoxia': 2}

        # Initialize data
        self.X_signals = None
        self.X_clinical = None
        self.y_labels = None
        self.record_ids = None
        self.clinical_scaler = None
        self.model = None

        # Create directories
        self.models_path.mkdir(exist_ok=True)
        self.results_path.mkdir(exist_ok=True)
        (self.results_path / 'training_plots').mkdir(exist_ok=True)

    def load_clinical_data(self):
        """Load and clean clinical dataset"""
        print("🔄 Loading clinical dataset...")
        clinical_file = self.processed_data_path / 'clinical_dataset.csv'

        if not clinical_file.exists():
            raise FileNotFoundError(f"Clinical dataset not found: {clinical_file}")

        df = pd.read_csv(clinical_file)
        print(f"✅ Loaded clinical data: {len(df)} records")

        # Clean and select relevant clinical features
        clinical_features = [
            'pH', 'BDecf', 'pCO2', 'BE', 'Apgar1', 'Apgar5',
            'Age', 'Sex', 'Weight', 'Gravidity', 'Parity',
            'Diabetes', 'Hypertension', 'Preeclampsia',
            'Rec_time', 'Deliv_time', 'GA'
        ]

        # Add additional numeric features if available
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_columns:
            if col not in clinical_features and col not in ['record_id', 'label']:
                clinical_features.append(col)

        # Keep only available features
        available_features = [f for f in clinical_features if f in df.columns]
        print(f"📊 Using {len(available_features)} clinical features: {available_features[:5]}...")

        # Add sequential record_id mapping (row 0 -> record 1001, etc.)
        df['record_id'] = range(1001, 1001 + len(df))

        # Clean the data
        df_clean = df[['record_id', 'label'] + available_features].copy()

        # Convert categorical to numeric
        if 'Sex' in df_clean.columns:
            df_clean['Sex'] = pd.to_numeric(df_clean['Sex'], errors='coerce')

        # Fill missing values with median for numeric columns
        for col in available_features:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                df_clean[col].fillna(df_clean[col].median(), inplace=True)

        return df_clean, available_features

    def load_signal_data(self, record_id):
        """Load signal data for specific record"""
        signal_file = self.signals_path / f"{record_id}_signals.npy"
        if not signal_file.exists():
            return None

        try:
            data = np.load(signal_file, allow_pickle=True).item()
            return data.get('FHR', [])
        except:
            return None

    def preprocess_signal(self, fhr_signal):
        """Preprocess FHR signal to standard length and normalization"""
        if fhr_signal is None or len(fhr_signal) == 0:
            return None

        # Standardize length
        if len(fhr_signal) < self.signal_length:
            fhr_processed = np.interp(
                np.linspace(0, len(fhr_signal)-1, self.signal_length),
                np.arange(len(fhr_signal)),
                fhr_signal
            )
        else:
            step = len(fhr_signal) // self.signal_length
            fhr_processed = fhr_signal[::step][:self.signal_length]

        # Z-score normalization
        fhr_mean = np.mean(fhr_processed)
        fhr_std = np.std(fhr_processed) + 1e-8
        fhr_normalized = (fhr_processed - fhr_mean) / fhr_std

        # Clean outliers
        fhr_normalized = np.nan_to_num(fhr_normalized, nan=0.0, posinf=1.0, neginf=-1.0)
        fhr_normalized = np.clip(fhr_normalized, -5, 5)

        return fhr_normalized

    def generate_multimodal_dataset(self):
        """Generate multimodal dataset combining signals and clinical features"""
        print("🏗️ Generating multimodal dataset...")

        # Load clinical data
        clinical_df, available_features = self.load_clinical_data()

        signals_data = []
        clinical_data = []
        labels_data = []
        record_ids_data = []

        valid_count = 0
        total_count = len(clinical_df)

        print(f"🔄 Processing {total_count} records...")

        for idx, row in clinical_df.iterrows():
            if pd.isna(row['record_id']):
                continue

            record_id = int(row['record_id'])

            # Load signal data
            fhr_signal = self.load_signal_data(record_id)
            if fhr_signal is None:
                continue

            # Preprocess signal
            processed_signal = self.preprocess_signal(fhr_signal)
            if processed_signal is None:
                continue

            # Extract clinical features
            clinical_features = []
            for feature in available_features:
                value = row[feature] if feature in row else 0.0
                clinical_features.append(float(value))

            # Get label
            label = row['label']
            if label not in self.label_map:
                continue

            # Store data
            signals_data.append(processed_signal)
            clinical_data.append(clinical_features)
            labels_data.append(self.label_map[label])
            record_ids_data.append(record_id)
            valid_count += 1

            if valid_count % 50 == 0:
                print(f"   Processed {valid_count}/{total_count} records...")

        # Convert to numpy arrays
        self.X_signals = np.array(signals_data)
        self.X_clinical = np.array(clinical_data)
        self.y_labels = np.array(labels_data)
        self.record_ids = np.array(record_ids_data)

        print(f"✅ Generated multimodal dataset:")
        print(f"   Valid samples: {len(self.X_signals)}")
        print(f"   Signal shape: {self.X_signals.shape}")
        print(f"   Clinical features: {self.X_clinical.shape[1]}")
        print(f"   Label distribution: {np.bincount(self.y_labels)}")

        return self.X_signals, self.X_clinical, self.y_labels, self.record_ids

    def prepare_data_for_training(self):
        """Prepare and split data for training"""
        print("🔧 Preparing data for training...")

        # Normalize clinical features
        self.clinical_scaler = StandardScaler()
        X_clinical_scaled = self.clinical_scaler.fit_transform(self.X_clinical)

        # Split data
        X_signals_train, X_signals_temp, X_clinical_train, X_clinical_temp, y_train, y_temp = train_test_split(
            self.X_signals, X_clinical_scaled, self.y_labels,
            test_size=0.3, random_state=42, stratify=self.y_labels
        )

        X_signals_val, X_signals_test, X_clinical_val, X_clinical_test, y_val, y_test = train_test_split(
            X_signals_temp, X_clinical_temp, y_temp,
            test_size=0.5, random_state=42, stratify=y_temp
        )

        print(f"📊 Data split:")
        print(f"   Train: {len(X_signals_train)} samples")
        print(f"   Validation: {len(X_signals_val)} samples")
        print(f"   Test: {len(X_signals_test)} samples")

        return (X_signals_train, X_clinical_train, y_train,
                X_signals_val, X_clinical_val, y_val,
                X_signals_test, X_clinical_test, y_test)

    def build_multimodal_model(self, clinical_features_dim, method='simple'):
        """Build multimodal neural network with different architectures"""
        print(f"🔧 Building {method.upper()} multimodal model...")

        # Signal branch - processes FHR temporal patterns
        signal_input = layers.Input(shape=(self.signal_length,), name='signal_input')

        if method == 'mobilenet':
            # MobileNet-inspired architecture with depthwise separable convolutions
            x_signal = layers.Reshape((self.signal_length, 1))(signal_input)
            x_signal = layers.Conv1D(32, 3, activation='relu', padding='same')(x_signal)
            x_signal = layers.DepthwiseConv1D(3, padding='same')(x_signal)
            x_signal = layers.Conv1D(64, 1, activation='relu')(x_signal)
            x_signal = layers.GlobalAveragePooling1D()(x_signal)
            x_signal = layers.Dense(64, activation='relu')(x_signal)
            x_signal = layers.Dropout(0.3)(x_signal)

        elif method == 'resnet':
            # ResNet-inspired architecture with residual connections
            x_signal = layers.Reshape((self.signal_length, 1))(signal_input)

            # First conv block
            x_signal = layers.Conv1D(64, 7, padding='same', activation='relu')(x_signal)
            x_signal = layers.MaxPooling1D(2)(x_signal)

            # Residual blocks
            for i in range(3):
                residual = x_signal
                x_signal = layers.Conv1D(64, 3, padding='same', activation='relu')(x_signal)
                x_signal = layers.BatchNormalization()(x_signal)
                x_signal = layers.Conv1D(64, 3, padding='same', activation='relu')(x_signal)
                x_signal = layers.BatchNormalization()(x_signal)
                x_signal = layers.Add()([x_signal, residual])
                x_signal = layers.Activation('relu')(x_signal)

            x_signal = layers.GlobalAveragePooling1D()(x_signal)
            x_signal = layers.Dense(64, activation='relu')(x_signal)
            x_signal = layers.Dropout(0.3)(x_signal)

        elif method == 'gan':
            # GAN-enhanced feature extraction (discriminator-like)
            x_signal = layers.Dense(256, activation='relu')(signal_input)
            x_signal = layers.BatchNormalization()(x_signal)
            x_signal = layers.Dropout(0.4)(x_signal)
            x_signal = layers.Dense(128, activation='relu')(x_signal)
            x_signal = layers.BatchNormalization()(x_signal)
            x_signal = layers.Dropout(0.4)(x_signal)
            x_signal = layers.Dense(64, activation='relu')(x_signal)
            x_signal = layers.Dropout(0.3)(x_signal)

        else:  # simple method
            x_signal = layers.Dense(128, activation='relu')(signal_input)
            x_signal = layers.BatchNormalization()(x_signal)
            x_signal = layers.Dropout(0.3)(x_signal)
            x_signal = layers.Dense(64, activation='relu')(x_signal)
            x_signal = layers.BatchNormalization()(x_signal)
            x_signal = layers.Dropout(0.3)(x_signal)

        # Clinical branch - processes tabular medical data
        clinical_input = layers.Input(shape=(clinical_features_dim,), name='clinical_input')
        x_clinical = layers.Dense(32, activation='relu')(clinical_input)
        x_clinical = layers.BatchNormalization()(x_clinical)
        x_clinical = layers.Dropout(0.2)(x_clinical)
        x_clinical = layers.Dense(16, activation='relu')(x_clinical)
        x_clinical = layers.BatchNormalization()(x_clinical)
        x_clinical = layers.Dropout(0.2)(x_clinical)

        # Fusion layer - combines both branches
        fusion = layers.Concatenate()([x_signal, x_clinical])
        x = layers.Dense(32, activation='relu')(fusion)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(16, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        # Classification layer
        output = layers.Dense(self.num_classes, activation='softmax', name='classification')(x)

        # Create model
        model = keras.Model(inputs=[signal_input, clinical_input], outputs=output)

        # Compile model with method-specific parameters
        if method == 'gan':
            optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        elif method == 'mobilenet':
            optimizer = keras.optimizers.Adam(learning_rate=0.001)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=0.001)

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"✅ {method.upper()} model built successfully:")
        print(f"   Total parameters: {model.count_params():,}")

        return model

    def train_model(self, method='simple'):
        """Train the multimodal model with specified method"""
        print(f"🚀 Starting {method.upper()} multimodal training...")

        # Generate dataset
        if self.X_signals is None:
            self.generate_multimodal_dataset()

        # Prepare data
        (X_signals_train, X_clinical_train, y_train,
         X_signals_val, X_clinical_val, y_val,
         X_signals_test, X_clinical_test, y_test) = self.prepare_data_for_training()

        # Build model
        self.model = self.build_multimodal_model(X_clinical_train.shape[1], method)

        # Calculate class weights for imbalanced data
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        print(f"🔧 Class weights: {class_weight_dict}")

        # Method-specific training parameters
        if method == 'gan':
            epochs = 50
            patience = 15
            batch_size = 16
        elif method == 'mobilenet':
            epochs = 40
            patience = 12
            batch_size = 32
        elif method == 'resnet':
            epochs = 35
            patience = 10
            batch_size = 24
        else:  # simple
            epochs = 30
            patience = 10
            batch_size = 32

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=patience//2, factor=0.5, min_lr=1e-6),
            keras.callbacks.ModelCheckpoint(
                str(self.models_path / f'{method}_multimodal_best_weights.h5'),
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]

        # Train model
        history = self.model.fit(
            [X_signals_train, X_clinical_train], y_train,
            validation_data=([X_signals_val, X_clinical_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
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
        print(f"\n📊 {method.upper()} Training Results:")
        print(f"   Test Accuracy: {test_accuracy:.4f}")
        print(f"   Test Loss: {test_loss:.4f}")

        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_names))

        # Save model as PKL
        model_data = {
            'model': self.model,
            'method': method,
            'clinical_scaler': self.clinical_scaler,
            'signal_length': self.signal_length,
            'num_classes': self.num_classes,
            'label_names': self.label_names,
            'label_map': self.label_map
        }

        pkl_path = self.models_path / f'{method}_multimodal_hypoxia_detector.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ Model saved as PKL: {pkl_path}")

        # Generate plots
        self.plot_training_results(history, y_test, y_pred, method)

        return history, test_accuracy

    def plot_training_results(self, history, y_test, y_pred, method='simple'):
        """Generate and save training visualization plots"""
        print("📈 Generating training plots...")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{method.upper()} Multimodal Hypoxia Detection - Training Results', fontsize=16)

        # Training history plots
        axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2)
        axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_names, yticklabels=self.label_names, ax=axes[1, 0])
        axes[1, 0].set_title('Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')

        # Label Distribution
        unique, counts = np.unique(y_test, return_counts=True)
        axes[1, 1].bar([self.label_names[i] for i in unique], counts, alpha=0.7)
        axes[1, 1].set_title('Test Set Label Distribution')
        axes[1, 1].set_xlabel('Label')
        axes[1, 1].set_ylabel('Count')

        plt.tight_layout()
        plot_path = self.results_path / 'training_plots' / f'{method}_multimodal_training_results.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Training plots saved: {plot_path}")

    def load_model_from_pkl(self, method='simple'):
        """Load trained model from PKL file"""
        pkl_path = self.models_path / f'{method}_multimodal_hypoxia_detector.pkl'

        if not pkl_path.exists():
            # Fallback to old naming convention
            old_pkl_path = self.models_path / 'multimodal_hypoxia_detector.pkl'
            if old_pkl_path.exists():
                pkl_path = old_pkl_path
            else:
                raise FileNotFoundError(f"Model PKL not found: {pkl_path}")

        with open(pkl_path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.clinical_scaler = model_data['clinical_scaler']
        self.signal_length = model_data['signal_length']
        self.num_classes = model_data['num_classes']
        self.label_names = model_data['label_names']
        self.label_map = model_data['label_map']

        print(f"✅ {method.upper()} model loaded from PKL: {pkl_path}")
        return self.model

    def get_available_methods(self):
        """Get list of available trained methods"""
        available = []

        # Check new naming convention
        for method in ['simple', 'gan', 'mobilenet', 'resnet']:
            pkl_path = self.models_path / f'{method}_multimodal_hypoxia_detector.pkl'
            if pkl_path.exists():
                available.append(method)

        # Check old naming convention (for backward compatibility)
        old_pkl_path = self.models_path / 'multimodal_hypoxia_detector.pkl'
        if old_pkl_path.exists() and 'simple' not in available:
            available.append('simple')

        return available

    def predict_single_record(self, record_id, method='simple'):
        """Predict hypoxia for a single record"""
        print(f"🔮 Predicting for record {record_id} using {method.upper()} method...")

        # Load model if not loaded
        if self.model is None:
            self.load_model_from_pkl(method)

        # Load and preprocess signal
        fhr_signal = self.load_signal_data(record_id)
        if fhr_signal is None:
            print(f"❌ Signal data not found for record {record_id}")
            return None

        processed_signal = self.preprocess_signal(fhr_signal)
        if processed_signal is None:
            print(f"❌ Failed to process signal for record {record_id}")
            return None

        # Load clinical data
        clinical_df, available_features = self.load_clinical_data()

        # Map record_id to row index (record 1001 -> row 0, etc.)
        row_index = record_id - 1001
        if row_index < 0 or row_index >= len(clinical_df):
            print(f"❌ Clinical data not found for record {record_id}")
            return None

        record_row = clinical_df.iloc[row_index]

        # Extract clinical features
        clinical_features = []
        for feature in available_features:
            value = record_row[feature] if feature in record_row.index else 0.0
            clinical_features.append(float(value))

        clinical_features = np.array(clinical_features).reshape(1, -1)
        clinical_features_scaled = self.clinical_scaler.transform(clinical_features)

        # Make prediction
        signal_input = processed_signal.reshape(1, -1)
        prediction_probs = self.model.predict([signal_input, clinical_features_scaled], verbose=0)
        predicted_class = np.argmax(prediction_probs[0])
        confidence = np.max(prediction_probs[0])

        result = {
            'record_id': record_id,
            'method': method,
            'predicted_class': predicted_class,
            'predicted_label': self.label_names[predicted_class],
            'confidence': float(confidence),
            'class_probabilities': {
                self.label_names[i]: float(prediction_probs[0][i])
                for i in range(self.num_classes)
            }
        }

        print(f"✅ {method.upper()} Prediction Results:")
        print(f"   Record: {record_id}")
        print(f"   Prediction: {result['predicted_label']}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Probabilities:")
        for label, prob in result['class_probabilities'].items():
            print(f"     {label}: {prob:.3f}")

        return result

    def select_method(self, action_type="training"):
        """Method selection helper"""
        print(f"\n🔬 SELECT {action_type.upper()} METHOD")
        print("="*40)
        print("1. 🤖 GAN Method (Enhanced feature extraction)")
        print("2. 📱 MobileNet Method (Lightweight CNN)")
        print("3. 🏗️ ResNet Method (Deep residual network)")
        print("4. 🎯 Simple Method (Basic multimodal)")

        while True:
            choice = input("Select method (1-4): ").strip()
            if choice == '1':
                return 'gan'
            elif choice == '2':
                return 'mobilenet'
            elif choice == '3':
                return 'resnet'
            elif choice == '4':
                return 'simple'
            else:
                print("❌ Invalid choice. Please select 1-4.")

    def compare_all_methods(self, record_id):
        """Compare predictions from all available methods"""
        available_methods = self.get_available_methods()
        if len(available_methods) < 2:
            print("❌ Need at least 2 trained models for comparison")
            return

        print(f"\n🆚 COMPARING ALL METHODS FOR RECORD {record_id}")
        print("="*60)

        results = {}
        for method in available_methods:
            try:
                # Reset model to load correct method
                self.model = None
                result = self.predict_single_record(record_id, method)
                if result:
                    results[method] = result
                print()  # Add spacing between methods
            except Exception as e:
                print(f"❌ {method.upper()} error: {e}")

        if len(results) > 1:
            print(f"📊 COMPARISON SUMMARY:")
            print("="*50)
            print(f"{'Method':<12} {'Prediction':<10} {'Confidence':<12}")
            print("-" * 34)
            for method, result in results.items():
                print(f"{method.upper():<12} {result['predicted_label']:<10} {result['confidence']:<12.3f}")

            # Check consensus
            predictions = [r['predicted_label'] for r in results.values()]
            if len(set(predictions)) == 1:
                print(f"\n✅ CONSENSUS: All methods predict {predictions[0]}")
            else:
                print(f"\n⚠️ DISAGREEMENT: Methods have different predictions")

    def interactive_menu(self):
        """Interactive menu for training and prediction"""
        while True:
            print("\n" + "="*60)
            print("🧬 MULTIMODAL HYPOXIA DETECTION SYSTEM")
            print("="*60)
            print("1. 🎯 Train New Model (Signal + Clinical)")
            print("2. 🔮 Predict Single Record")
            print("3. 📊 Batch Prediction")
            print("4. 🆚 Compare All Methods")
            print("5. 📋 Show System Status")
            print("6. ❌ Exit")

            choice = input("\nSelect option (1-6): ").strip()

            if choice == '1':
                try:
                    method = self.select_method("training")
                    print(f"\n🚀 Starting {method.upper()} training...")
                    history, accuracy = self.train_model(method)
                    print(f"\n✅ {method.upper()} training completed! Final accuracy: {accuracy:.4f}")
                except Exception as e:
                    print(f"❌ Training error: {e}")

            elif choice == '2':
                try:
                    available_methods = self.get_available_methods()
                    if not available_methods:
                        print("❌ No trained models found. Please train a model first.")
                        continue

                    record_id = int(input("Enter record ID: "))

                    if len(available_methods) == 1:
                        method = available_methods[0]
                        print(f"Using available method: {method.upper()}")
                    else:
                        print(f"Available methods: {', '.join(m.upper() for m in available_methods)}")
                        method = self.select_method("prediction")

                    self.model = None  # Reset to load correct method
                    result = self.predict_single_record(record_id, method)
                except ValueError:
                    print("❌ Please enter a valid record ID")
                except Exception as e:
                    print(f"❌ Prediction error: {e}")

            elif choice == '3':
                try:
                    available_methods = self.get_available_methods()
                    if not available_methods:
                        print("❌ No trained models found. Please train a model first.")
                        continue

                    records_input = input("Enter record IDs (comma-separated): ")
                    record_ids = [int(x.strip()) for x in records_input.split(',')]

                    if len(available_methods) == 1:
                        method = available_methods[0]
                        print(f"Using available method: {method.upper()}")
                    else:
                        print(f"Available methods: {', '.join(m.upper() for m in available_methods)}")
                        method = self.select_method("batch prediction")

                    self.model = None  # Reset to load correct method
                    results = []
                    for record_id in record_ids:
                        result = self.predict_single_record(record_id, method)
                        if result:
                            results.append(result)

                    if results:
                        print(f"\n📊 {method.upper()} Batch Prediction Summary:")
                        for result in results:
                            print(f"   Record {result['record_id']}: {result['predicted_label']} ({result['confidence']:.3f})")

                except ValueError:
                    print("❌ Please enter valid record IDs")
                except Exception as e:
                    print(f"❌ Batch prediction error: {e}")

            elif choice == '4':
                try:
                    record_id = int(input("Enter record ID for comparison: "))
                    self.compare_all_methods(record_id)
                except ValueError:
                    print("❌ Please enter a valid record ID")
                except Exception as e:
                    print(f"❌ Comparison error: {e}")

            elif choice == '5':
                self.show_system_status()

            elif choice == '6':
                print("👋 Thank you for using the Multimodal Hypoxia Detection System!")
                break

            else:
                print("❌ Invalid choice. Please select 1-6.")

            input("\nPress Enter to continue...")

    def show_system_status(self):
        """Show system status and available data"""
        print("\n📋 SYSTEM STATUS")
        print("="*50)

        # Check data files
        clinical_file = self.processed_data_path / 'clinical_dataset.csv'
        signals_dir = self.signals_path

        print("📂 Data Files:")
        print(f"   Clinical Dataset: {'✅' if clinical_file.exists() else '❌'} {clinical_file}")
        print(f"   Signals Directory: {'✅' if signals_dir.exists() else '❌'} {signals_dir}")

        if signals_dir.exists():
            signal_files = list(signals_dir.glob("*_signals.npy"))
            print(f"   Signal Files: {len(signal_files)} files")

        # Check trained models
        available_methods = self.get_available_methods()
        print(f"\n🤖 Trained Models:")
        if available_methods:
            for method in ['simple', 'gan', 'mobilenet', 'resnet']:
                pkl_path = self.models_path / f'{method}_multimodal_hypoxia_detector.pkl'
                status = "✅" if method in available_methods else "❌"
                print(f"   {status} {method.upper()} Method")
                if pkl_path.exists():
                    size_mb = pkl_path.stat().st_size / (1024*1024)
                    print(f"      File: {pkl_path.name} ({size_mb:.1f} MB)")
        else:
            # Check old naming convention
            old_pkl_path = self.models_path / 'multimodal_hypoxia_detector.pkl'
            if old_pkl_path.exists():
                print(f"   ✅ Legacy Model (multimodal_hypoxia_detector.pkl)")
            else:
                print(f"   ❌ No trained models found")

        # Load and show clinical data stats
        if clinical_file.exists():
            try:
                df = pd.read_csv(clinical_file)
                print(f"\n📊 Clinical Dataset:")
                print(f"   Total records: {len(df)}")
                if 'label' in df.columns:
                    label_counts = df['label'].value_counts()
                    print(f"   Label distribution:")
                    for label, count in label_counts.items():
                        print(f"     {label}: {count}")
            except Exception as e:
                print(f"   Error reading clinical data: {e}")

        print(f"\n📁 Available Methods for Training:")
        methods_info = {
            'simple': 'Basic multimodal (fastest training)',
            'gan': 'GAN-enhanced feature extraction (best for complex patterns)',
            'mobilenet': 'Lightweight CNN (good for mobile deployment)',
            'resnet': 'Deep residual network (high capacity model)'
        }
        for method, description in methods_info.items():
            print(f"   🔬 {method.upper()}: {description}")

def main():
    """Main function"""
    detector = MultimodalHypoxiaDetector()
    detector.interactive_menu()

if __name__ == "__main__":
    main()