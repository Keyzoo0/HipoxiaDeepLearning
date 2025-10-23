# Dataset Notes for MDNN Journal Preparation

## Source and Characteristics

### **CTU-UHB Intrapartum Cardiotocography Database**
- **Source**: PhysioNet PSYCONET 1.0.0 database
- **Total recordings**: 552 intrapartum CTG recordings
- **Duration**: 90 minutes per recording with 4 Hz sampling frequency
- **Data format**: Original `.dat/.hea` pairs converted to processed `.npy` arrays

### **Modalities Used in MDNN**
1. **Fetal Heart Rate (FHR) Signals**
   - Raw temporal data extracted from CTG recordings
   - Preprocessed to fixed length of 5000 points per record
   - Stored in `processed_data/signals/` as individual `.npy` files

2. **Clinical Parameters**
   - Core biochemical markers: pH, base deficit (BDecf), pCO2, base excess (BE)
   - Neonatal assessment: Apgar scores at 1 and 5 minutes
   - Maternal demographics: Age, weight, gravidity, parity
   - Obstetric history: Gestational age, diabetes, hypertension, preeclampsia
   - Delivery metadata: Recording time, delivery time

## Label Definition and Classification

### **Three-Class Target Labels**
- **Normal** (Label 0): pH ≥ 7.25, no signs of fetal distress
- **Suspect** (Label 1): pH 7.20-7.24, borderline acidosis indicators
- **Hypoxia** (Label 2): pH < 7.20, clear signs of fetal acidosis

### **Label Distribution**
Based on processed dataset analysis (`methods/data_handler.py:20`):
- Normal cases: ~60% of dataset (naturally occurring majority)
- Suspect cases: ~25% of dataset (intermediate risk category)
- Hypoxia cases: ~15% of dataset (critical minority requiring detection)

## Preprocessing Pipeline for MDNN

### **Signal Preprocessing** (`methods/data_handler.py:70`)
1. **Physiological Clipping**: FHR values constrained to 50-200 bpm range
2. **Length Standardization**: All signals resampled to exactly 5000 timesteps
3. **Robust Normalization**: Median-centered scaling using MAD (Median Absolute Deviation)
4. **Artifact Handling**: Outlier detection and smoothing for sensor noise reduction

### **Clinical Feature Processing** (`methods/data_handler.py:33`)
1. **Missing Value Imputation**:
   - pH values: Label-specific median imputation for clinical accuracy
   - Other parameters: Global median imputation to minimize data loss
2. **Outlier Management**: Values beyond ±3σ clipped to reduce measurement error impact
3. **Feature Standardization**: Z-score normalization using training set statistics
4. **Categorical Encoding**: Binary encoding for discrete clinical variables

### **Data Splitting Strategy** (`methods/data_handler.py:204`)
- **Training Set**: 70% stratified by class labels (354 samples)
- **Validation Set**: 15% for hyperparameter tuning (76 samples)
- **Test Set**: 15% held-out for final evaluation (76 samples)
- **Stratification**: Maintains class proportions across all splits

## Class Balancing for MDNN Training

### **Augmentation Strategy** (`methods/data_handler.py:118`)
- **Signal Augmentation**:
  - Gaussian noise injection (σ=0.018 for Suspect, σ=0.02 for Hypoxia)
  - Temporal shifts: Random circular shifts up to ±35 samples
- **Clinical Augmentation**: Small random perturbations within physiological bounds
- **Target Balancing**: Augment minority classes to achieve roughly equal representation

### **SMOTE-Tomek Resampling** (`methods/data_handler.py:220`)
- Applied to concatenated signal-clinical feature vectors
- Synthetic minority oversampling with Tomek link removal
- Refines class boundaries while maintaining realistic feature distributions
- Final balanced dataset: ~243 samples per class for training

### **Class Weight Enhancement** (`methods/trainer.py:54`)
- Hypoxia class weight increased by 1.5× during training
- Amplifies gradient contributions from critical minority cases
- Balances precision-recall trade-off for clinical sensitivity

## MDNN-Specific Data Characteristics

### **Signal Branch Input**
- **Dimensionality**: 5000-dimensional vectors (direct FHR timesteps)
- **Processing**: No convolution - signals fed directly to dense layers
- **Normalization**: Robust scaling preserves temporal relationships
- **Format**: Single-channel temporal data without explicit sequence modeling

### **Clinical Branch Input**
- **Feature Count**: 17+ numeric clinical parameters (exact count varies by record)
- **Preprocessing**: Standardized feature vectors with consistent scaling
- **Missing Handling**: Complete cases after imputation (no missing values in training)
- **Encoding**: Fully numeric representation suitable for dense layer processing

### **Multimodal Integration**
- **Concatenation**: Simple concatenation of signal and clinical embeddings
- **Attention Weighting**: Learned attention mechanism for adaptive modality fusion
- **Feature Alignment**: Both modalities processed to compatible embedding dimensions
- **Synchronization**: Signal and clinical data aligned by record ID for consistent pairing

## Data Quality Validation

### **Signal Quality Checks**
1. **Completeness**: Verify all 5000 timesteps present for each record
2. **Range Validation**: Confirm FHR values within physiological bounds
3. **Noise Assessment**: Evaluate signal-to-noise ratio and artifact presence
4. **Temporal Consistency**: Check for gaps or discontinuities in recordings

### **Clinical Data Validation**
1. **Parameter Completeness**: Document missing value patterns across features
2. **Physiological Ranges**: Validate clinical parameters within expected medical ranges
3. **Label Consistency**: Cross-check pH values with assigned classification labels
4. **Correlation Analysis**: Verify expected relationships between clinical parameters

### **Integration Verification**
1. **Record Alignment**: Ensure signal and clinical data match by record ID
2. **Batch Consistency**: Verify consistent preprocessing across train/val/test splits
3. **Scaling Validation**: Confirm normalization parameters computed only on training data
4. **Augmentation Tracking**: Document augmentation effects on data distribution

## Reproducibility Guidelines

### **Data Preparation Checklist**
1. **Source Verification**: Confirm CTU-UHB database version and completeness
2. **Preprocessing Consistency**: Apply identical preprocessing pipeline across all records
3. **Split Reproducibility**: Use fixed random seeds for consistent train/val/test splits
4. **Augmentation Control**: Document and version augmentation parameters
5. **Quality Metrics**: Maintain data quality logs throughout preprocessing

### **MDNN Training Data Requirements**
1. **Balanced Training Set**: Verify class balance after augmentation and resampling
2. **Validation Integrity**: Ensure validation set remains untouched during preprocessing
3. **Test Set Isolation**: Confirm test set isolation throughout development process
4. **Feature Scaling**: Maintain separate scalers for signal and clinical modalities
5. **Documentation**: Complete audit trail of all data transformations

### **Performance Baseline Establishment**
1. **Multiple Runs**: Conduct training across multiple random initializations
2. **Statistical Validation**: Report mean and standard deviation across runs
3. **Hyperparameter Sensitivity**: Document performance stability across parameter ranges
4. **Data Sensitivity**: Evaluate performance on different data splits
5. **Generalization Testing**: Assess performance on held-out validation data

## Future Data Enhancement Opportunities

### **Signal Processing Improvements**
- **Multi-scale Analysis**: Incorporate wavelet or frequency domain features
- **Temporal Modeling**: Add sequence information while preserving MDNN simplicity
- **Artifact Detection**: Enhanced automated artifact identification and removal
- **Signal Quality Metrics**: Quantitative assessment of recording quality

### **Clinical Feature Expansion**
- **Temporal Clinical Data**: Incorporate time-series clinical measurements
- **Demographic Stratification**: Analyze performance across population subgroups
- **Risk Factor Integration**: Include additional maternal and fetal risk factors
- **Outcome Prediction**: Extend beyond hypoxia to broader neonatal outcomes

### **Dataset Augmentation**
- **Multi-center Data**: Incorporate recordings from multiple hospitals
- **Prospective Collection**: Add contemporary recordings with modern equipment
- **External Validation**: Test on independent datasets for generalizability
- **Longitudinal Studies**: Track outcomes beyond immediate delivery period

This comprehensive dataset documentation ensures reproducible MDNN research and provides foundation for manuscript preparation and peer review.