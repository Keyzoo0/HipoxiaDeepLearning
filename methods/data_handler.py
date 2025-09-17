#!/usr/bin/env python3
"""
Data Handler Module
Handles clinical data loading, signal processing, and dataset generation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek

class DataHandler:
    def __init__(self, base_path, signal_length=5000):
        self.base_path = Path(base_path)
        self.processed_data_path = self.base_path / 'processed_data'
        self.signals_path = self.processed_data_path / 'signals'
        self.signal_length = signal_length
        self.label_map = {'normal': 0, 'suspect': 1, 'hypoxia': 2}
        self.label_names = ['Normal', 'Suspect', 'Hypoxia']

        # Data storage
        self.X_signals = None
        self.X_clinical = None
        self.y_labels = None
        self.record_ids = None
        self.clinical_scaler = None

    def load_clinical_data(self):
        """Load and clean clinical dataset"""
        print("🔄 Loading clinical dataset...")
        clinical_file = self.processed_data_path / 'clinical_dataset.csv'

        if not clinical_file.exists():
            raise FileNotFoundError(f"Clinical dataset not found: {clinical_file}")

        df = pd.read_csv(clinical_file)
        print(f"✅ Loaded clinical data: {len(df)} records")

        # Clinical features to extract
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

        # Advanced feature preprocessing
        for col in available_features:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

                # Use label-specific imputation for better feature quality
                if col == 'pH':  # Critical feature - use label-specific median
                    for label in ['normal', 'suspect', 'hypoxia']:
                        mask = df_clean['label'] == label
                        if mask.any():
                            label_median = df_clean.loc[mask, col].median()
                            df_clean.loc[mask & df_clean[col].isna(), col] = label_median
                else:
                    # Regular median imputation for other features
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)

                # Remove extreme outliers (beyond 3 standard deviations)
                if col not in ['Sex', 'Diabetes', 'Hypertension', 'Preeclampsia']:
                    mean_val = df_clean[col].mean()
                    std_val = df_clean[col].std()
                    if std_val > 0:
                        lower_bound = mean_val - 3 * std_val
                        upper_bound = mean_val + 3 * std_val
                        df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)

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
        """Enhanced FHR signal preprocessing for better model performance"""
        if fhr_signal is None or len(fhr_signal) == 0:
            return None

        # Remove extreme outliers first (values outside physiological range)
        fhr_signal = np.array(fhr_signal)
        fhr_signal = np.clip(fhr_signal, 50, 200)  # Physiological FHR range

        # Standardize length with better interpolation
        if len(fhr_signal) < self.signal_length:
            fhr_processed = np.interp(
                np.linspace(0, len(fhr_signal)-1, self.signal_length),
                np.arange(len(fhr_signal)),
                fhr_signal
            )
        else:
            step = len(fhr_signal) // self.signal_length
            fhr_processed = fhr_signal[::step][:self.signal_length]

        # Simplified robust normalization for better model convergence
        # Use robust Z-score normalization only
        fhr_median = np.median(fhr_processed)
        fhr_mad = np.median(np.abs(fhr_processed - fhr_median))  # Median Absolute Deviation

        if fhr_mad > 0:
            # Robust Z-score using median and MAD
            fhr_normalized = (fhr_processed - fhr_median) / (fhr_mad * 1.4826)  # 1.4826 for normal distribution
        else:
            # Fallback to simple centering
            fhr_normalized = fhr_processed - fhr_median

        # Clean statistical outliers
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

    def apply_data_augmentation(self, X_signals, X_clinical, y_labels):
        """Advanced data augmentation for better minority class representation"""
        print("🔄 Applying advanced data augmentation...")

        # Apply signal-specific augmentation first
        X_signals_aug = []
        X_clinical_aug = []
        y_labels_aug = []

        for i, (signal, clinical, label) in enumerate(zip(X_signals, X_clinical, y_labels)):
            X_signals_aug.append(signal)
            X_clinical_aug.append(clinical)
            y_labels_aug.append(label)

            # Enhanced augmentation for 85% target
            if label == 2:  # Hypoxia - most critical
                # Multiple augmentation techniques
                # 1. Noise augmentation
                noise_factor = 0.02
                noise = np.random.normal(0, noise_factor, signal.shape)
                augmented_signal = signal + noise
                X_signals_aug.append(augmented_signal)
                X_clinical_aug.append(clinical)
                y_labels_aug.append(label)

                # 2. Time shifting
                shift = np.random.randint(-30, 30)
                shifted_signal = np.roll(signal, shift)
                X_signals_aug.append(shifted_signal)
                X_clinical_aug.append(clinical)
                y_labels_aug.append(label)

            elif label == 1:  # Suspect - always augment
                # Enhanced augmentation for Suspect class
                if np.random.random() < 0.7:  # 70% chance
                    # Time shifting
                    shift = np.random.randint(-35, 35)
                    shifted_signal = np.roll(signal, shift)
                    X_signals_aug.append(shifted_signal)
                    X_clinical_aug.append(clinical)
                    y_labels_aug.append(label)

                if np.random.random() < 0.3:  # 30% chance for noise
                    noise_factor = 0.018
                    noise = np.random.normal(0, noise_factor, signal.shape)
                    augmented_signal = signal + noise
                    X_signals_aug.append(augmented_signal)
                    X_clinical_aug.append(clinical)
                    y_labels_aug.append(label)

        X_signals_aug = np.array(X_signals_aug)
        X_clinical_aug = np.array(X_clinical_aug)
        y_labels_aug = np.array(y_labels_aug)

        # Apply SMOTE for remaining imbalance
        X_combined = np.hstack([X_signals_aug, X_clinical_aug])
        smote_enn = SMOTETomek(random_state=42)
        X_resampled, y_resampled = smote_enn.fit_resample(X_combined, y_labels_aug)

        # Split back into signals and clinical
        X_signals_aug = X_resampled[:, :self.signal_length]
        X_clinical_aug = X_resampled[:, self.signal_length:]

        print(f"📊 After augmentation: {len(X_resampled)} samples")
        unique, counts = np.unique(y_resampled, return_counts=True)
        print(f"📊 Class distribution: {dict(zip(unique, counts))}")

        return X_signals_aug, X_clinical_aug, y_resampled

    def get_record_clinical_features(self, record_id):
        """Get clinical features for a specific record"""
        clinical_df, available_features = self.load_clinical_data()

        # Map record_id to row index (record 1001 -> row 0, etc.)
        row_index = record_id - 1001
        if row_index < 0 or row_index >= len(clinical_df):
            return None, None

        record_row = clinical_df.iloc[row_index]

        # Extract clinical features
        clinical_features = []
        for feature in available_features:
            value = record_row[feature] if feature in record_row.index else 0.0
            clinical_features.append(float(value))

        clinical_features = np.array(clinical_features).reshape(1, -1)

        return clinical_features, available_features