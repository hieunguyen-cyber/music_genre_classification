# Music Genre Classification (GTZAN) — Modular Research Pipeline

Repo này refactor notebook `music-genre-classification.ipynb` thành một pipeline modular, có thể chạy từng stage độc lập, có cache feature, logging, checkpointing, seed reproducibility và visualization chuyên sâu.

## Dataset

Repo hiện chứa GTZAN mirror trong `Data/`:
- `Data/genres_original/` — raw audio `.wav`
- `Data/features_3_sec.csv` — tabular features (3-sec segments, ~9990 samples)
- `Data/features_30_sec.csv` — tabular features (30-sec, 1000 samples)
- `Data/images_original/` — ảnh spectrogram theo file 30-sec (tuỳ mirror)

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Chạy pipeline

## Thứ tự chạy khuyến nghị

### Track A (giữ đúng logic notebook: dùng CSV features)

1) `feature` (tạo split + cache features)
2) `train`
3) `evaluate`
4) `visualize` (tuỳ chọn, để sinh plot)

```bash
python src/main.py --stage feature --run-name exp01
python src/main.py --stage train --run-name exp01
python src/main.py --stage evaluate --run-name exp01
python src/main.py --stage visualize --run-name exp01
```

### Track B (end-to-end từ raw audio: mel-spectrogram + CNN/LSTM)

1) `preprocess` (tuỳ chọn nhưng nên chạy để kiểm tra dataset)
2) chuyển `features.kind` sang `mel_from_audio` trong `configs/config.yaml`
3) `feature` (extract mel + group split theo track)
4) `train` với `--model cnn_mel` hoặc `--model lstm_mel`
5) `evaluate`
6) `visualize`

```bash
python src/main.py --stage preprocess --run-name mel01
python src/main.py --stage feature --run-name mel01
python src/main.py --stage train --run-name mel01 --model cnn_mel
python src/main.py --stage evaluate --run-name mel01 --model cnn_mel
python src/main.py --stage visualize --run-name mel01
```

### Chạy từng bước

```bash
python src/main.py --stage preprocess --run-name exp01
python src/main.py --stage feature --run-name exp01
python src/main.py --stage train --run-name exp01
python src/main.py --stage evaluate --run-name exp01
python src/main.py --stage visualize --run-name exp01
```

### Chọn mô hình

Model name nằm trong `configs/model.yaml` và chọn bằng `configs/config.yaml` hoặc `--model`.

Ví dụ chạy baseline classical ML:

```bash
python src/main.py --stage train --run-name knn01 --model knn
python src/main.py --stage evaluate --run-name knn01 --model knn
```

Các lựa chọn hiện có:
- `mlp` (PyTorch, tương đương logic Dense/Dropout của notebook)
- sklearn baselines: `knn`, `svm_rbf`, `linear_svm`, `logreg`, `naive_bayes`, `lda`, `qda`, `random_forest`, `extra_trees`, `ada_boost`, `gbdt`
- optional deps: `xgboost`, `lightgbm`
- mel-from-audio torch: `cnn_mel`, `lstm_mel` (yêu cầu `features.kind=mel_from_audio`)

### Chạy full pipeline

```bash
make full RUN_NAME=exp01
```

## Output structure

Mỗi `run` được lưu tách biệt:

```
outputs/<run_name>/
  checkpoints/
    best.pt
  logs/
    preprocess.log
    feature.log
    train.log
    evaluate.log
    visualize.log
  figures/
    *.png
  reports/
    config_snapshot.json
    history.json
    evaluation_report.md
    misclassified.csv
```

## Project structure

```
configs/
data/
docs/
notebooks/
scripts/
src/
outputs/
```

## Docs

- `docs/PIPELINE.md` — dependency và IO từng stage
- `docs/MATH.md` — Fourier/STFT/Mel/MFCC/Cross-Entropy/Adam/metrics
- `docs/VISUALIZATION.md` — cách đọc plots
- `docs/EXPERIMENTS.md` — template ghi lại thí nghiệm
