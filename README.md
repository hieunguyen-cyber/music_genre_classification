# Music Genre Classification (GTZAN) — Modular Research Pipeline

Repo này refactor notebook `music-genre-classification.ipynb` thành một pipeline modular, có thể chạy từng stage độc lập, có cache feature, logging, checkpointing, seed reproducibility và visualization chuyên sâu.

---

## 1) Bài toán là gì? (Problem statement)

**Music Genre Classification**: cho một đoạn audio (hoặc feature trích xuất từ audio), dự đoán nhãn thể loại nhạc (genre) trong tập hữu hạn \(C\) lớp. Với GTZAN (mirror phổ biến), \(C=10\):

- blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

Ta xem đây là bài toán **multi-class classification**:

- Input: \(x\) (audio waveform hoặc vector feature)
- Output: \(y \in \{0,1,\dots,C-1\}\)

### Vì sao bài toán này quan trọng?

- **Music Information Retrieval (MIR)**: tổ chức/thẻ hóa nội dung audio, hỗ trợ tìm kiếm/recommendation.
- **Feature engineering + signal processing**: bài toán “kinh điển” để học Fourier/STFT/Mel/MFCC.
- **Baseline pipeline**: dễ mở rộng sang mood, instrument, artist, tag prediction.

### Thách thức thực tế

- **Genre chồng lấp**: cùng nhịp/nhạc cụ có thể xuất hiện ở nhiều genre.
- **Domain shift**: dataset nhỏ, thu âm/production khác nhau → model dễ overfit.
- **Data leakage**: nếu cắt 1 bài thành nhiều đoạn rồi split sai, train/test có thể chứa segment của cùng bài.

---

## 2) Dataset GTZAN (mô tả chi tiết)

Repo hiện chứa GTZAN mirror trong `Data/`:
- `Data/genres_original/` — raw audio `.wav` (thường 100 tracks/genre, mỗi track ~30s → tổng ~1000 tracks)
- `Data/features_30_sec.csv` — tabular features cho mỗi track 30s (1000 rows)
- `Data/features_3_sec.csv` — tabular features cho các segment 3s (khoảng 9990 rows; mỗi track thường ~10 segment)
- `Data/images_original/` — ảnh spectrogram theo file 30-sec (tuỳ mirror)

### Hai “granularity” dữ liệu

1) **Track-level (30s)**
- Ít mẫu hơn (1000), nhưng mỗi mẫu đầy đủ bài.
- Dễ split đúng (theo bài), ít leakage hơn.

2) **Segment-level (3s)**
- Nhiều mẫu hơn (~9990), train nhanh và thường tăng accuracy.
- Nhưng **rất dễ leakage** nếu split theo row ngẫu nhiên (segment của cùng bài có thể vào nhiều split).

Trong pipeline:
- `features.kind=tabular_csv` (mặc định) giữ logic notebook: đọc `features_3_sec.csv` và split theo row.
- `features.kind=mel_from_audio` (end-to-end) sẽ cắt segment từ raw audio và **split theo track group** để tránh leakage.

---

## 3) Dữ liệu/feature trong `features_3_sec.csv` là gì?

File `Data/features_3_sec.csv` là dataset tabular đã trích xuất sẵn (thường bằng Librosa). Mỗi row gồm:

- `filename`: tên segment (ví dụ `blues.00000.0.wav`)
- `length`: độ dài tính theo frames/samples (tuỳ mirror)
- Nhóm feature **spectral/temporal** dạng *mean/variance*:
  - Chroma STFT: `chroma_stft_mean`, `chroma_stft_var`
  - RMS energy: `rms_mean`, `rms_var`
  - Spectral centroid/bandwidth/rolloff: `spectral_centroid_*`, `spectral_bandwidth_*`, `rolloff_*`
  - Zero crossing rate: `zero_crossing_rate_*`
  - Harmony/percussive (HPSS-related): `harmony_*`, `perceptr_*` (tuỳ mirror đặt tên)
  - `tempo`
  - MFCC 1..20: `mfcc{i}_mean`, `mfcc{i}_var`
- `label`: genre (string)

Ý nghĩa: biến 3 giây audio thành vector số thực \(x \in \mathbb{R}^d\) (với \(d \approx 58\) trong file này), giúp dùng được các model tabular: KNN/SVM/MLP/RandomForest…

---

## 4) Tiền xử lý (Preprocessing) — kèm toán học và ý nghĩa

Pipeline có 2 track preprocessing tương ứng 2 kiểu feature.

### 4.1 Track A: Tabular CSV (giống notebook)

#### (A) Label encoding

Chuyển `label` (string) → chỉ số lớp:
\[
f: \\{\\text{genre strings}\\} \\to \\{0,1,\\dots,C-1\\}
\]

Trong code dùng `LabelEncoder` fit trên train split để đảm bảo mapping ổn định.

#### (B) Drop cột không dùng cho model

`filename` không mang thông tin âm học trực tiếp (và có thể gây leakage theo id) nên loại bỏ khỏi \(X\).

#### (C) Standardization (z-score scaling)

Nhiều thuật toán (SVM, Logistic Regression, MLP) nhạy với thang đo feature. Ta chuẩn hóa từng feature \(j\):

\[
\\mu_j = \\frac{1}{N_{train}}\\sum_{i \\in train} x_{ij}, \\quad
\\sigma_j = \\sqrt{\\frac{1}{N_{train}}\\sum_{i \\in train} (x_{ij}-\\mu_j)^2}
\]
\[
\\tilde{x}_{ij} = \\frac{x_{ij}-\\mu_j}{\\sigma_j + \\epsilon}
\]

**Quan trọng**: \(\mu_j, \sigma_j\) chỉ được fit trên **train**, rồi áp dụng cho val/test để tránh “peek” vào test.

#### (D) Split train/val/test (stratified)

Giữ tỷ lệ lớp gần giống nhau giữa các split bằng stratification. Mục tiêu:
- train: học tham số
- val: chọn hyperparameter / early stopping
- test: ước lượng performance cuối cùng

Pipeline lưu split vào `data/splits/*.csv` để reproducible.

### 4.2 Track B: End-to-end raw audio → mel spectrogram

#### (A) Waveform

Audio rời rạc \(x[n]\) với sample rate \(sr\) Hz. Thời gian tương ứng:
\[
t = \\frac{n}{sr}
\]

#### (B) STFT (Short-Time Fourier Transform)

Âm nhạc biến đổi theo thời gian → ta dùng STFT trên từng frame:
\[
X(m,k) = \\sum_{n=0}^{N-1} x[n+mH] \\, w[n] \\, e^{-j2\\pi kn/N}
\]
- \(w[n]\): window (Hann…)
- \(N\): FFT size
- \(H\): hop length

Spectrogram magnitude/power:
\[
S(m,k) = |X(m,k)|^2
\]

#### (C) Mel filter bank → Mel spectrogram

Mel scale (một công thức phổ biến):
\[
m = 2595 \\, \\log_{10}\\left(1 + \\frac{f}{700}\\right)
\]

Mel filter bank \(M\) gom năng lượng theo dải mel:
\[
S_{mel} = M S
\]

Đưa về dB (log-compression):
\[
S_{dB} = 10\\log_{10}(S_{mel} + \\epsilon)
\]

**Ý nghĩa**: Mel spectrogram gần với cảm nhận thính giác; log giúp “nén” dynamic range và ổn định cho model.

#### (D) Split theo track group (chống leakage segment)

Một track 30s được cắt thành nhiều segment 3s. Để tránh leakage, pipeline gán `group_id = track_id` và split theo group:
- tất cả segment của 1 track chỉ thuộc 1 split.

---

## 5) Các mô hình phân loại (Model zoo) và ý nghĩa

### 5.1 Classical ML baselines (tabular)

- **KNN**: dự đoán theo đa số láng giềng gần nhất (nhạy scaling).
- **SVM (RBF/Linear)**: tìm siêu phẳng phân tách; RBF cho biên phi tuyến.
- **Logistic Regression (multinomial)**: baseline tuyến tính mạnh khi feature tốt.
- **Naive Bayes / LDA / QDA**: giả định phân phối (Gaussian), nhanh và hay dùng làm baseline.
- **RandomForest / ExtraTrees / AdaBoost / GBDT**: mô hình cây + ensemble, mạnh trên tabular.
- **XGBoost / LightGBM**: gradient boosting tối ưu (optional dependency).

### 5.2 Neural models (PyTorch)

- **MLP (tabular)**: tương đương logic notebook (Dense + Dropout). Loss dùng Cross Entropy.
- **CNN on mel**: dùng convolution 2D học pattern theo tần số–thời gian.
- **LSTM on mel**: xem mel như chuỗi theo thời gian \(t\), học phụ thuộc dài hạn.

---

## 6) Training objective (toán học)

### 6.1 Softmax

Với logits \(z \\in \\mathbb{R}^C\):
\[
p_i = \\frac{e^{z_i}}{\\sum_{j=1}^{C} e^{z_j}}
\]

### 6.2 Cross Entropy (multi-class)

Với nhãn đúng \(y\):
\[
\\mathcal{L} = -\\log(p_y)
\]

Notebook dùng `sparse_categorical_crossentropy`; trong pipeline PyTorch dùng `CrossEntropyLoss` (tương đương softmax + negative log-likelihood).

### 6.3 Adam optimizer (tóm tắt)

Với gradient \(g_t\):
\[
m_t = \\beta_1 m_{t-1} + (1-\\beta_1) g_t,\quad
v_t = \\beta_2 v_{t-1} + (1-\\beta_2) g_t^2
\]
\[
\\theta_t = \\theta_{t-1} - \\alpha \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t}+\\epsilon}
\]

---

## 7) Evaluation metrics (ý nghĩa + công thức)

### 7.1 Confusion matrix

Ma trận \(C\\) với \(C_{ij}\) = số mẫu lớp thật \(i\) được dự đoán thành \(j\).

### 7.2 Accuracy
\[
\\text{Acc} = \\frac{1}{N}\\sum_{i=1}^{N} \\mathbb{1}[\\hat{y}_i = y_i]
\]

### 7.3 Precision / Recall / F1 (1 lớp)
\[
P = \\frac{TP}{TP+FP},\\quad R = \\frac{TP}{TP+FN},\\quad
F1 = \\frac{2PR}{P+R}
\]

### 7.4 Macro-F1 vs Weighted-F1

- **Macro-F1**: trung bình F1 theo lớp (đối xử công bằng với lớp hiếm).
- **Weighted-F1**: weighted theo số mẫu mỗi lớp (phản ánh phân bố dữ liệu).

### 7.5 ROC-AUC (OvR)

Với multi-class, pipeline tính ROC-AUC theo **one-vs-rest** (nếu model có xác suất `predict_proba`).

---

## Quickstart (tóm tắt)

### Dataset layout trong repo

Repo hiện chứa GTZAN mirror trong `Data/` (nhắc lại nhanh):
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

## Chọn device (CPU/CUDA/MPS)

Mặc định pipeline auto-detect theo ưu tiên `cuda > mps > cpu`.

Override nhanh bằng CLI:

```bash
python src/main.py --stage train --run-name exp01 --device mps
python src/main.py --stage train --run-name exp01 --device cuda
python src/main.py --stage train --run-name exp01 --device cpu
```

## Tuỳ chỉnh tham số train qua CLI

YAML trong `configs/` là mặc định; CLI flags dùng để override nhanh cho từng run.

Ví dụ:

```bash
python src/main.py --stage train --run-name exp02 \
  --model mlp --epochs 200 --batch-size 512 --lr 1e-4 --weight-decay 1e-4 \
  --early-stopping on --patience 20 --monitor val_loss \
  --ckpt-metric val_loss --ckpt-mode min
```

Các flag chính:
- `--epochs`, `--batch-size`, `--lr`, `--weight-decay`, `--num-workers`
- `--early-stopping on|off`, `--patience`, `--monitor val_loss|val_accuracy`
- `--ckpt-metric val_loss|val_accuracy`, `--ckpt-mode min|max`

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
