# ResNet Robustness Journal

## 1. Abstract
This document analyses the ResNet-based branch of the multimodal fetal hypoxia detection system shipped in this repository. The goal is to explain why the residual architecture paired with the project’s training stack yields robust performance on noisy cardiotocography signals and heterogeneous clinical covariates. The journal consolidates implementation details, training configuration, and robustness arguments gathered from the codebase.

## 2. Background and Motivation
Fetal hypoxia manifests through subtle temporal patterns in fetal heart rate (FHR) signals and requires the integration of multiple clinical indicators. The Multimodal Hypoxia Detector orchestrates signal processing, feature fusion, and decision making across several deep learners (`main_modular.py`). Among these, the ResNet pathway attains the highest reported accuracy (>96%, see `results/README.md`). Understanding the design choices behind this outcome is essential for academic reporting and clinical translation.

## 3. Data Pipeline Overview
- **Signal preparation** (`methods/data_handler.py:70`): FHR traces are clipped to physiological bounds (50–200 bpm), resampled to a fixed length of 5000 points, and normalized via robust z-score (median/MAD). This suppresses sensor noise and preserves labour-phase trends.
- **Clinical features** (`methods/data_handler.py:33`): 17+ numeric attributes (e.g., pH, base deficit, Apgar scores) are cleaned with label-aware imputations and outlier clipping, then standardized using `StandardScaler` prior to training.
- **Balancing and augmentation** (`methods/data_handler.py:118` and `methods/data_handler.py:220`): Minority classes (Suspect, Hypoxia) receive targeted augmentations (Gaussian noise, time-shifts) followed by SMOTETomek resampling. The trainer further inflates the hypoxia class weight by 1.5× (`methods/trainer.py:54`). The combination mitigates class imbalance without oversmoothing physiological anomalies.

## 4. ResNet Architecture
Source: `methods/model_builder.py:67-123`.

1. **Temporal encoder**
   - Input reshaped to `(5000, 1)` for 1D convolutions.
   - Initial Conv1D layer (64 filters, kernel size 31) captures long-range heart rate oscillations before subsampling.
2. **Residual block**
   - Two Conv1D layers with 128 filters and kernel size 15 preserve mid-frequency patterns.
   - A 1×1 convolution aligns channel dimensions for the skip path, ensuring residual summation remains well-conditioned.
   - Batch normalisation in both paths stabilises training on varying signal amplitudes.
3. **Global summarisation**
   - Max pooling and dropout (0.2) aggressively remove noise while retaining salient beats.
   - GlobalAveragePooling1D reduces dimensionality and prevents overfitting to local spikes.
4. **Dense refinement**
   - Two dense layers (128 → 64) with batch norm and dropout finalise signal embeddings.
5. **Clinical fusion and attention**
   - Clinical branch: Dense layers 48 → 32 → 16 with progressive dropout (`methods/model_builder.py:141-154`).
   - Concatenated features pass through an attention-like reweighting layer (`Dense(..., activation='softmax')`) that emphasises modality contributions dynamically.
6. **Classifier head**
   - Dense stack (144 → 96 → 48 → 32) with calibrated dropout rates feeds a 3-way softmax output.

The architecture balances depth and parameter count by limiting residual stacks to a single well-regularised block—appropriate for the limited dataset size while still harnessing residual learning benefits.

## 5. Training Configuration
- **Hyperparameters** (`methods/trainer.py:26`): 120 epochs, batch size 16, patience 30 for early stopping.
- **Optimiser** (`methods/model_builder.py:172`): Adam with learning rate `6e-4`, β₁=0.9, β₂=0.999. A conservative rate keeps residual updates stable.
- **Callbacks** (`methods/trainer.py:79`): Early stopping on validation accuracy with restoration of best weights, ReduceLROnPlateau (patience 12, factor 0.5), and model checkpointing. This trio guards against overfitting while adapting the learning rate when validation gains saturate.
- **Evaluation** (`methods/trainer.py:121`): After training, the model is tested on a hold-out set; precision, recall, F1, and confusion matrix are reported. Visual analytics (loss/accuracy curves, ROC, per-class stats) are generated for documentation.

## 6. Prediction Workflow
When performing inference (`methods/predictor.py:15`):
1. Loads the stored ResNet model and scalers from `models/resnet_multimodal_hypoxia_detector.pkl`.
2. Retrieves raw FHR signal and clinical features for the requested record.
3. Applies identical preprocessing and scaling used during training.
4. Outputs class probabilities, dominant label, and confidence, then triggers detailed visual explanations.

The uniform preprocessing between training and prediction ensures consistent feature distributions, a prerequisite for robust deployment.

## 7. Robustness Rationale
1. **Residual learning for temporal stability**: Skip connections prevent vanishing gradients and allow the network to learn perturbation corrections rather than entire mappings, supporting generalisation on high-frequency artefacts.
2. **Robust preprocessing**: Median/MAD normalisation and outlier clipping reduce susceptibility to sensor spikes while retaining clinically meaningful variability.
3. **Targeted augmentation + SMOTETomek**: Augmenting minority classes before applying SMOTETomek yields a balanced yet diverse training set, improving resilience to class imbalance.
4. **Class weighting emphasis**: Boosting the hypoxia class weight amplifies rare-event gradients, combating bias toward the majority class.
5. **Adaptive regularisation**: Dropout, batch normalisation, and learning rate scheduling collectively dampen overfitting, especially important for limited data regimes.
6. **Multimodal attention fusion**: The attention gate allows the model to shift reliance between signal and clinical pathways, handling missingness or degraded signals gracefully.
7. **Model stewardship**: Automatic checkpointing preserves the best-performing weights, and comprehensive visual diagnostics support auditing and iterative refinement.

## 8. Experimental Summary
| Component | Setting |
|-----------|---------|
| Signal length | 5000 samples per record |
| Filters / kernels | 64@31 (stem), residual 128@15 |
| Pooling | MaxPooling1D (8, then 4), GlobalAveragePooling1D |
| Dense layers | Signal: 128→64, Fusion head: 144→96→48→32 |
| Dropout | 0.2–0.4 range across branches |
| Optimiser | Adam (lr 0.0006) |
| Epochs / batch | 120 epochs, batch size 16 |
| Early stopping | Patience 30 on validation accuracy |
| Class balance | Augmentation + SMOTETomek + weighted loss |

Reported performance from project documentation: Accuracy 96.1%, Precision 95.7%, Recall 95.9%, F1 95.8% on the internal test split (see `results/README.md`).

## 9. Reproduction Checklist
1. Generate multimodal dataset: run `python main_modular.py` and choose the dataset generation option, or call `MultimodalHypoxiaDetector().generate_dataset()`.
2. Train the ResNet branch: `multimodal_detector.train_model('resnet')` or via menu option 1.
3. Inspect outputs in `results/training_plots/trainingResultResNetMethod/`.
4. Evaluate single-record predictions: `multimodal_detector.predict_record(1001, 'resnet')`.
5. Archive trained weights: verify `models/resnet_multimodal_best_weights.h5` and corresponding PKL file.

## 10. Limitations and Future Work
- Current residual stack is shallow; experimenting with additional bottleneck blocks could further capture multi-scale temporal features if more data becomes available.
- Clinical feature imputation remains median-based. Incorporating probabilistic imputation or temporal clinical streams may enhance robustness to missing data.
- External validation on unseen hospital cohorts is necessary before clinical deployment.

## 11. Conclusion
The ResNet configuration achieves robust multimodal hypoxia detection by pairing residual temporal encoding with disciplined preprocessing, class balancing, and adaptive regularisation. The architecture maintains a favourable parameter-to-sample ratio while leveraging attention-based fusion to reconcile signal and clinical modalities. These design choices collectively explain its strong empirical performance and readiness for inclusion in academic reporting.
