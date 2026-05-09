#!/usr/bin/env bash
# set -euo pipefail

# Full pipeline commands for all supported models.
# Run each pipeline sequentially, or copy the blocks you need.
# Note: `xgboost` and `lightgbm` require optional dependencies.

# ============================================================================
# FEATURE VISUALIZATION (run once per feature kind, not per model)
# ============================================================================

# Visualize tabular features and dataset stats
python3 src/main.py --stage visualize-features --run-name visualize-features-tabular

# Visualize mel features and dataset stats
python3 src/main.py --stage visualize-features --run-name visualize-features-mel --feature-kind mel_from_audio

# ============================================================================
# Track A: Tabular CSV features (default features.kind=tabular_csv)
# The `feature` stage creates cached tabular features and splits (run once).
# ============================================================================

# Create features once (will be reused by all models)
python3 src/main.py --stage feature --run-name tabular-features

# Pipeline 1: MLP (PyTorch tabular)
python3 src/main.py --stage train --run-name tabular_mlp --model mlp
python3 src/main.py --stage evaluate --run-name tabular_mlp --model mlp
python3 src/main.py --stage visualize --run-name tabular_mlp

# Pipeline 2: kNN
python3 src/main.py --stage train --run-name tabular_knn --model knn
python3 src/main.py --stage evaluate --run-name tabular_knn --model knn
python3 src/main.py --stage visualize --run-name tabular_knn

# Pipeline 3: SVM with RBF kernel
python3 src/main.py --stage train --run-name tabular_svm_rbf --model svm_rbf
python3 src/main.py --stage evaluate --run-name tabular_svm_rbf --model svm_rbf
python3 src/main.py --stage visualize --run-name tabular_svm_rbf

# Pipeline 4: Linear SVM
python3 src/main.py --stage train --run-name tabular_linear_svm --model linear_svm
python3 src/main.py --stage evaluate --run-name tabular_linear_svm --model linear_svm
python3 src/main.py --stage visualize --run-name tabular_linear_svm

# Pipeline 5: Logistic Regression
python3 src/main.py --stage train --run-name tabular_logreg --model logreg
python3 src/main.py --stage evaluate --run-name tabular_logreg --model logreg
python3 src/main.py --stage visualize --run-name tabular_logreg

# Pipeline 6: Naive Bayes
python3 src/main.py --stage train --run-name tabular_naive_bayes --model naive_bayes
python3 src/main.py --stage evaluate --run-name tabular_naive_bayes --model naive_bayes
python3 src/main.py --stage visualize --run-name tabular_naive_bayes

# Pipeline 7: LDA
python3 src/main.py --stage train --run-name tabular_lda --model lda
python3 src/main.py --stage evaluate --run-name tabular_lda --model lda
python3 src/main.py --stage visualize --run-name tabular_lda

# Pipeline 8: QDA
python3 src/main.py --stage train --run-name tabular_qda --model qda
python3 src/main.py --stage evaluate --run-name tabular_qda --model qda
python3 src/main.py --stage visualize --run-name tabular_qda

# Pipeline 9: Random Forest
python3 src/main.py --stage train --run-name tabular_random_forest --model random_forest
python3 src/main.py --stage evaluate --run-name tabular_random_forest --model random_forest
python3 src/main.py --stage visualize --run-name tabular_random_forest

# Pipeline 10: Extra Trees
python3 src/main.py --stage train --run-name tabular_extra_trees --model extra_trees
python3 src/main.py --stage evaluate --run-name tabular_extra_trees --model extra_trees
python3 src/main.py --stage visualize --run-name tabular_extra_trees

# Pipeline 11: AdaBoost
python3 src/main.py --stage train --run-name tabular_ada_boost --model ada_boost
python3 src/main.py --stage evaluate --run-name tabular_ada_boost --model ada_boost
python3 src/main.py --stage visualize --run-name tabular_ada_boost

# Pipeline 12: GBDT
python3 src/main.py --stage train --run-name tabular_gbdt --model gbdt
python3 src/main.py --stage evaluate --run-name tabular_gbdt --model gbdt
python3 src/main.py --stage visualize --run-name tabular_gbdt

# Pipeline 13: XGBoost (optional dependency)
python3 src/main.py --stage train --run-name tabular_xgboost --model xgboost
python3 src/main.py --stage evaluate --run-name tabular_xgboost --model xgboost
python3 src/main.py --stage visualize --run-name tabular_xgboost

# Pipeline 14: LightGBM (optional dependency)
python3 src/main.py --stage train --run-name tabular_lightgbm --model lightgbm
python3 src/main.py --stage evaluate --run-name tabular_lightgbm --model lightgbm
python3 src/main.py --stage visualize --run-name tabular_lightgbm

# ============================================================================
# Track B: Mel spectrogram features from raw audio
# Create features once (will be reused by cnn_mel and lstm_mel).
# ============================================================================

python3 src/main.py --stage preprocess --run-name mel-preprocessing
python3 src/main.py --stage feature --run-name mel-features --feature-kind mel_from_audio

# Pipeline 15: CNN on mel spectrogram
python3 src/main.py --stage train --run-name mel_cnn --model cnn_mel --feature-kind mel_from_audio
python3 src/main.py --stage evaluate --run-name mel_cnn --model cnn_mel --feature-kind mel_from_audio
python3 src/main.py --stage visualize --run-name mel_cnn

# Pipeline 16: LSTM on mel spectrogram
python3 src/main.py --stage train --run-name mel_lstm --model lstm_mel --feature-kind mel_from_audio
python3 src/main.py --stage evaluate --run-name mel_lstm --model lstm_mel --feature-kind mel_from_audio
python3 src/main.py --stage visualize --run-name mel_lstm
