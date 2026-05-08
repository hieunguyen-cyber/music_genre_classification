#!/usr/bin/env bash
set -euo pipefail

# Full pipeline commands for all supported models.
# Run each pipeline sequentially, or copy the blocks you need.
# Note: `xgboost` and `lightgbm` require optional dependencies.

# Track A: Tabular CSV features (default features.kind=tabular_csv)
# The `feature` stage creates cached tabular features and splits.

# Pipeline 1: MLP (PyTorch tabular)
python src/main.py --stage feature --run-name tabular_mlp
python src/main.py --stage train --run-name tabular_mlp --model mlp
python src/main.py --stage evaluate --run-name tabular_mlp --model mlp
python src/main.py --stage visualize --run-name tabular_mlp

# Pipeline 2: kNN
python src/main.py --stage feature --run-name tabular_knn
python src/main.py --stage train --run-name tabular_knn --model knn
python src/main.py --stage evaluate --run-name tabular_knn --model knn
python src/main.py --stage visualize --run-name tabular_knn

# Pipeline 3: SVM with RBF kernel
python src/main.py --stage feature --run-name tabular_svm_rbf
python src/main.py --stage train --run-name tabular_svm_rbf --model svm_rbf
python src/main.py --stage evaluate --run-name tabular_svm_rbf --model svm_rbf
python src/main.py --stage visualize --run-name tabular_svm_rbf

# Pipeline 4: Linear SVM
python src/main.py --stage feature --run-name tabular_linear_svm
python src/main.py --stage train --run-name tabular_linear_svm --model linear_svm
python src/main.py --stage evaluate --run-name tabular_linear_svm --model linear_svm
python src/main.py --stage visualize --run-name tabular_linear_svm

# Pipeline 5: Logistic Regression
python src/main.py --stage feature --run-name tabular_logreg
python src/main.py --stage train --run-name tabular_logreg --model logreg
python src/main.py --stage evaluate --run-name tabular_logreg --model logreg
python src/main.py --stage visualize --run-name tabular_logreg

# Pipeline 6: Naive Bayes
python src/main.py --stage feature --run-name tabular_naive_bayes
python src/main.py --stage train --run-name tabular_naive_bayes --model naive_bayes
python src/main.py --stage evaluate --run-name tabular_naive_bayes --model naive_bayes
python src/main.py --stage visualize --run-name tabular_naive_bayes

# Pipeline 7: LDA
python src/main.py --stage feature --run-name tabular_lda
python src/main.py --stage train --run-name tabular_lda --model lda
python src/main.py --stage evaluate --run-name tabular_lda --model lda
python src/main.py --stage visualize --run-name tabular_lda

# Pipeline 8: QDA
python src/main.py --stage feature --run-name tabular_qda
python src/main.py --stage train --run-name tabular_qda --model qda
python src/main.py --stage evaluate --run-name tabular_qda --model qda
python src/main.py --stage visualize --run-name tabular_qda

# Pipeline 9: Random Forest
python src/main.py --stage feature --run-name tabular_random_forest
python src/main.py --stage train --run-name tabular_random_forest --model random_forest
python src/main.py --stage evaluate --run-name tabular_random_forest --model random_forest
python src/main.py --stage visualize --run-name tabular_random_forest

# Pipeline 10: Extra Trees
python src/main.py --stage feature --run-name tabular_extra_trees
python src/main.py --stage train --run-name tabular_extra_trees --model extra_trees
python src/main.py --stage evaluate --run-name tabular_extra_trees --model extra_trees
python src/main.py --stage visualize --run-name tabular_extra_trees

# Pipeline 11: AdaBoost
python src/main.py --stage feature --run-name tabular_ada_boost
python src/main.py --stage train --run-name tabular_ada_boost --model ada_boost
python src/main.py --stage evaluate --run-name tabular_ada_boost --model ada_boost
python src/main.py --stage visualize --run-name tabular_ada_boost

# Pipeline 12: GBDT
python src/main.py --stage feature --run-name tabular_gbdt
python src/main.py --stage train --run-name tabular_gbdt --model gbdt
python src/main.py --stage evaluate --run-name tabular_gbdt --model gbdt
python src/main.py --stage visualize --run-name tabular_gbdt

# Pipeline 13: XGBoost (optional dependency)
python src/main.py --stage feature --run-name tabular_xgboost
python src/main.py --stage train --run-name tabular_xgboost --model xgboost
python src/main.py --stage evaluate --run-name tabular_xgboost --model xgboost
python src/main.py --stage visualize --run-name tabular_xgboost

# Pipeline 14: LightGBM (optional dependency)
python src/main.py --stage feature --run-name tabular_lightgbm
python src/main.py --stage train --run-name tabular_lightgbm --model lightgbm
python src/main.py --stage evaluate --run-name tabular_lightgbm --model lightgbm
python src/main.py --stage visualize --run-name tabular_lightgbm

# Track B: Mel spectrogram features from raw audio
# Use --feature-kind mel_from_audio and run preprocess once for raw audio checks.

# Pipeline 15: CNN on mel spectrogram
python src/main.py --stage preprocess --run-name mel_cnn
python src/main.py --stage feature --run-name mel_cnn --feature-kind mel_from_audio 
python src/main.py --stage train --run-name mel_cnn --model cnn_mel --feature-kind mel_from_audio
python src/main.py --stage evaluate --run-name mel_cnn --model cnn_mel --feature-kind mel_from_audio
python src/main.py --stage visualize --run-name mel_cnn

# Pipeline 16: LSTM on mel spectrogram
python src/main.py --stage preprocess --run-name mel_lstm
python src/main.py --stage feature --run-name mel_lstm --feature-kind mel_from_audio
python src/main.py --stage train --run-name mel_lstm --model lstm_mel --feature-kind mel_from_audio
python src/main.py --stage evaluate --run-name mel_lstm --model lstm_mel --feature-kind mel_from_audio
python src/main.py --stage visualize --run-name mel_lstm
