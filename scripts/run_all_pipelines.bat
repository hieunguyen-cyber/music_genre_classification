@echo off
:: Equivalent of run_all_pipelines.sh
:: Full pipeline commands for all supported models.
:: Run each pipeline sequentially, or copy the blocks you need.
:: Note: xgboost and lightgbm require optional dependencies.

setlocal enabledelayedexpansion

:: Change to project root (parent of scripts/) so src/main.py resolves correctly
pushd "%~dp0\.."

:: Helper macro: run a command and abort on failure
:: Usage: call :run <command...>

:: ============================================================================
:: FEATURE VISUALIZATION (run once per feature kind, not per model)
:: ============================================================================

:: Visualize tabular features and dataset stats
python src/main.py --stage visualize-features --run-name visualize-features-tabular
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

:: Visualize mel features and dataset stats
python src/main.py --stage visualize-features --run-name visualize-features-mel --feature-kind mel_from_audio
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

:: ============================================================================
:: Track A: Tabular CSV features (default features.kind=tabular_csv)
:: The `feature` stage creates cached tabular features and splits (run once).
:: ============================================================================

:: Create features once (will be reused by all models)
python src/main.py --stage feature --run-name tabular-features
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

:: Pipeline 1: MLP (PyTorch tabular)
python src/main.py --stage train --run-name tabular_mlp --model mlp
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage evaluate --run-name tabular_mlp --model mlp
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage visualize --run-name tabular_mlp
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

:: Pipeline 2: kNN
python src/main.py --stage train --run-name tabular_knn --model knn
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage evaluate --run-name tabular_knn --model knn
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage visualize --run-name tabular_knn
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

:: Pipeline 3: SVM with RBF kernel
python src/main.py --stage train --run-name tabular_svm_rbf --model svm_rbf
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage evaluate --run-name tabular_svm_rbf --model svm_rbf
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage visualize --run-name tabular_svm_rbf
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

:: Pipeline 4: Linear SVM
python src/main.py --stage train --run-name tabular_linear_svm --model linear_svm
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage evaluate --run-name tabular_linear_svm --model linear_svm
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage visualize --run-name tabular_linear_svm
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

:: Pipeline 5: Logistic Regression
python src/main.py --stage train --run-name tabular_logreg --model logreg
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage evaluate --run-name tabular_logreg --model logreg
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage visualize --run-name tabular_logreg
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

:: Pipeline 6: Naive Bayes
python src/main.py --stage train --run-name tabular_naive_bayes --model naive_bayes
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage evaluate --run-name tabular_naive_bayes --model naive_bayes
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage visualize --run-name tabular_naive_bayes
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

:: Pipeline 7: LDA
python src/main.py --stage train --run-name tabular_lda --model lda
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage evaluate --run-name tabular_lda --model lda
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage visualize --run-name tabular_lda
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

:: Pipeline 8: QDA
python src/main.py --stage train --run-name tabular_qda --model qda
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage evaluate --run-name tabular_qda --model qda
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage visualize --run-name tabular_qda
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

:: Pipeline 9: Random Forest
python src/main.py --stage train --run-name tabular_random_forest --model random_forest
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage evaluate --run-name tabular_random_forest --model random_forest
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage visualize --run-name tabular_random_forest
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

:: Pipeline 10: Extra Trees
python src/main.py --stage train --run-name tabular_extra_trees --model extra_trees
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage evaluate --run-name tabular_extra_trees --model extra_trees
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage visualize --run-name tabular_extra_trees
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

:: Pipeline 11: AdaBoost
python src/main.py --stage train --run-name tabular_ada_boost --model ada_boost
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage evaluate --run-name tabular_ada_boost --model ada_boost
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage visualize --run-name tabular_ada_boost
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

:: Pipeline 12: GBDT
python src/main.py --stage train --run-name tabular_gbdt --model gbdt
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage evaluate --run-name tabular_gbdt --model gbdt
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage visualize --run-name tabular_gbdt
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

:: Pipeline 13: XGBoost (optional dependency)
python src/main.py --stage train --run-name tabular_xgboost --model xgboost
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage evaluate --run-name tabular_xgboost --model xgboost
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage visualize --run-name tabular_xgboost
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

:: Pipeline 14: LightGBM (optional dependency)
python src/main.py --stage train --run-name tabular_lightgbm --model lightgbm
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage evaluate --run-name tabular_lightgbm --model lightgbm
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage visualize --run-name tabular_lightgbm
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

:: ============================================================================
:: Track B: Mel spectrogram features from raw audio
:: Create features once (will be reused by cnn_mel and lstm_mel).
:: ============================================================================

python src/main.py --stage preprocess --run-name mel-preprocessing
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage feature --run-name mel-features --feature-kind mel_from_audio
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

:: Pipeline 15: CNN on mel spectrogram
python src/main.py --stage train --run-name mel_cnn --model cnn_mel --feature-kind mel_from_audio
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage evaluate --run-name mel_cnn --model cnn_mel --feature-kind mel_from_audio
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage visualize --run-name mel_cnn
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

:: Pipeline 16: LSTM on mel spectrogram
python src/main.py --stage train --run-name mel_lstm --model lstm_mel --feature-kind mel_from_audio
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage evaluate --run-name mel_lstm --model lstm_mel --feature-kind mel_from_audio
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python src/main.py --stage visualize --run-name mel_lstm
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

endlocal
