# Dataset Notes for Journal Preparation

## Source
- **CTU-UHB Intrapartum Cardiotocography Database** (accessible via PhysioNet).
- 552 recordings, each lasting 90 minutes with 4 Hz sampling.

## Modalities Used
1. **FHR Signals**
   - Raw fetal heart rate extracted from preprocessed `.npy` files under `processed_data/signals/`.
   - Resampled to fixed length of 5000 points per record.
2. **Clinical Parameters**
   - Core variables: pH, BDecf, pCO2, Base Excess, Apgar scores, maternal demographics, obstetric history.
   - Additional numeric columns auto-included when present.

## Label Definition
- Labels (`Normal`, `Suspect`, `Hypoxia`) mapped to integers 0/1/2 (`methods/data_handler.py:20`).
- Derived from umbilical artery pH and documented clinical outcomes.

## Preprocessing Summary
- Missing values handled via median imputation; pH uses label-specific medians for clinical fidelity.
- Outliers beyond ±3σ clipped to reduce the influence of measurement errors.
- Signal normalisation leverages MAD-based robust scaling, ensuring resilience to transient spikes.

## Train/Validation/Test Split
- Stratified split: 70% train, 15% validation, 15% test (`methods/data_handler.py:204`).
- Clinical features standardised using statistics computed on the training fold.

## Augmentation Details
- **Noise Injection**: Gaussian noise with σ≈0.02 for hypoxia class, 0.018 for suspect.
- **Time Shift**: Random circular shifts up to ±35 samples.
- **SMOTETomek**: Applied to concatenated signal-clinical vectors to refine class boundaries.

## Reuse Checklist
- Verify availability of `processed_data/clinical_dataset.csv` and all `*_signals.npy` files.
- Recompute scalers and resampling whenever dataset updates occur.
- Document any deviations (e.g., alternative sampling rates) for reproducibility.
