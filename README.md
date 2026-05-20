# Music Genre Classification (GTZAN) — Modular Research Pipeline

Repo này refactor notebook `music-genre-classification.ipynb` thành một pipeline modular, có thể chạy từng stage độc lập, có cache feature, logging, checkpointing, seed reproducibility và visualization chuyên sâu.

---

## 1) Bài toán là gì? (Problem statement)

**Music Genre Classification**: cho một đoạn audio (hoặc feature trích xuất từ audio), dự đoán nhãn thể loại nhạc (genre) trong tập hữu hạn $C$ lớp. Với GTZAN (mirror phổ biến), $C=10$:

- blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

Ta xem đây là bài toán **multi-class classification**:

- Input: $x$ (audio waveform hoặc vector feature)
- Output: $y \in \{0,1,\dots,C-1\}$

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

### Hai "granularity" dữ liệu — Trade-off giữa số lượng và chất lượng

**1) Track-level (30s) — Ít nhưng Sạch**

- **Số lượng**: 1000 mẫu (100 mỗi genre)
- **Ưu điểm**:
  - Ít mẫu → model train chậm hơn, nhưng dễ overfit trên dataset nhỏ.
  - Không leakage nếu split theo track (mỗi row = 1 track độc lập).
  - Biểu diễn đầy đủ 1 bài nhạc (30s = 1,320,000 sample ở 44kHz).
- **Nhược điểm**:
  - Quá ít cho deep learning (CNN/LSTM cần hàng ngàn mẫu để train tốt).
  - K-fold cross-validation khó (10 fold × 10 genre = 1 mẫu/fold/genre).

**2) Segment-level (3s) — Nhiều nhưng Rủi Ro**

- **Số lượng**: ~10,000 mẫu (10 segment/track × 1000 track)
- **Ưu điểm**:
  - Đủ mẫu cho deep learning, hỗ trợ batch training, data augmentation hiệu quả.
  - Tăng khả năng phát hiện local pattern (ví dụ solo guitar 3s).
- **Nhược điểm**:
  - **Leakage nguy hiểm nếu split sai**: segment của cùng track dễ vào train+test → model "cheat" học track ID.
  - Mất thông tin global (solo/bridge ở 1 segment không đủ để phân loại chính xác).

Trong pipeline:
- `features.kind=tabular_csv` (mặc định): dùng `features_3_sec.csv` và **split theo track group** (tất cả segment của cùng bài nhạc chỉ xuất hiện trong một split duy nhất) — đây là cách đúng về mặt phương pháp luận.
- `features.kind=mel_from_audio` (end-to-end): cắt mel spectrogram từ raw audio, tự split theo track group.

---

## 3) Dữ liệu/feature trong `features_3_sec.csv` là gì?

File `Data/features_3_sec.csv` là dataset tabular đã trích xuất sẵn (thường bằng Librosa). Mỗi row gồm:

- `filename`: tên segment (ví dụ `blues.00000.0.wav`)
- `length`: độ dài tính theo frames/samples (tuỳ mirror)
- Nhóm feature **spectral/temporal** dạng *mean/variance* (mean và variance tính trên toàn bộ khung 3s):
   - **Chroma STFT** (`chroma_stft_mean`, `chroma_stft_var`): đại diện cho năng lượng theo 12 nốt (C..B). Mô tả sự hiện diện và phân bố hòa âm (chords, key, pitch class). Những thể loại có cấu trúc hòa âm rõ (jazz, classical) thường thể hiện patterns chroma rõ rệt; thể loại tập trung vào nhịp/percussion (hiphop, techno) có chroma kém rõ.
   - **RMS Energy** (`rms_mean`, `rms_var`): năng lượng trung bình và biến thiên (âm lượng). Thể loại "loud"/"dense" (metal, rock, disco) thường có `rms_mean` cao; thể loại có động lực lớn theo nhịp (dance/disco) có `rms_var` lớn do biến thiên mạnh giữa phần đệm và đoạn cao trào.
   - **Spectral Centroid** (`spectral_centroid_mean`, `spectral_centroid_var`): "trọng tâm" tần số (centroid đơn giản như trung bình tần số có trọng số năng lượng). Cao = phổ nghiêng về tần số cao (brighter timbre). Thể loại như metal, pop (vocal rõ, nhiều presence ở dải trung-cao) có centroid cao hơn so với blues/classical khi so sánh phần nhạc không có vocal mạnh.
   - **Spectral Bandwidth** (`spectral_bandwidth_mean/var`): đo độ rộng phổ xung quanh centroid. Cao = phổ phân bố rộng (nhiều harmonic/percussive content). Thể loại có nhiều lớp âm thanh (metal, disco) thường có bandwidth lớn; nhạc cổ điển solo hoặc acoustic có bandwidth hẹp hơn.
   - **Spectral Rolloff** (`rolloff_mean`, `rolloff_var`): tần số dưới đó chiếm 85% năng lượng. Phân biệt nội dung nhiều năng lượng cao (hi-hats, cymbals) vs nhiều năng lượng trầm (bass, kick). EDM/disco/metal có rolloff cao; reggae/folk có rolloff thấp.
   - **Zero Crossing Rate (ZCR)** (`zero_crossing_rate_mean`, `zero_crossing_rate_var`): tỷ lệ số lần tín hiệu đổi dấu. Cao gợi ý nhiều thành phần noisy / percussive / high-frequency content. Hiphop (drum-heavy) và rock có ZCR cao hơn so với classical hoặc ambient.
   - **Harmony / Percussive (HPSS)** (`harmony_mean/var`, `perceptr_mean/var`): tách tần số harmonic (sustained, tonal) và percussive (transients). Genres với focus vào nhạc cụ melody/harmony (classical, jazz) có harmony lớn; genres nhấn nhịp (rock, hiphop, disco) có percussive lớn.
   - **Tempo**: ước lượng BPM trung bình trong 3s. Giá trị trung bình và biến thiên hữu dụng: disco/edm thường 110–140 BPM, metal/fast-rock có BPM cao hơn, blues/balada thường thấp (60–90 BPM). Lưu ý: 3s ngắn nên tempo ước lượng có độ tin cậy hạn chế, nhưng phân bố tổng vẫn informative.
   - **MFCC 1..20** (`mfcc{i}_mean`, `mfcc{i}_var`): các hệ số Mel-frequency cepstral. MFCC1 thường tương quan với overall energy / log-spectrum bias; các MFCC tiếp theo mã hoá hình dạng phổ (timbre). MFCC là dạng feature rất mạnh cho phân loại timbre và thường khác nhau rõ giữa các genre: vocal-centric genres (pop, hiphop) và acoustic genres (classical, blues) có tập MFCC đặc trưng.

**Tổng cộng**: ~58 features numeric (56 từ phổ + tempo). Mỗi mẫu có:
- Input: vector 58D → các model tabular (SVM/RF/MLP)
- Output: genre (0-9)

### 3.1 Tại sao features này hoạt động tốt?

1. Signal-processing + perceptual design: những feature như MFCC, mel-scale, chroma trực tiếp phản ánh cách con người cảm nhận âm thanh (mel scale) hoặc cấu trúc hòa âm (chroma), do đó chúng nén thông tin hữu dụng cho phân loại.
2. Mean/variance trên cửa sổ 3s cung cấp mô tả phân bố — giúp robustness với biến đổi nhỏ trong signal.
3. Các feature này kết hợp cả thông tin tần số (timbre), thời gian (percussive/transient) và thống kê (tempo, RMS) — nên chịu trách nhiệm cho nhiều yếu tố phân biệt genre.

### 3.2 Giải thích kỹ từng feature và khác biệt giá trị giữa các thể loại

Phần này mở rộng các điểm trên với chi tiết về ý nghĩa vật lý/toán học của feature, phạm vi giá trị điển hình và cách các genre thường khác nhau theo feature đó.

- **MFCC (Mel-Frequency Cepstral Coefficients)**
   - Ý nghĩa: biểu diễn envelope của spectrum trên thang mel; các hệ số thấp (1–3) chứa thông tin chung về shape và năng lượng; các hệ số cao hơn chứa chi tiết timbre.
   - Phạm vi: tuỳ implementation; thường MFCC1 có giá trị lớn hơn các MFCC tiếp theo; mean/var của MFCCs là chỉ báo timbre ổn định.
   - Genre khác biệt: vocal-heavy genres (pop, hiphop) thường có pattern MFCC đặc trưng (do con sự hiện diện của formant vocal), guitar-driven genres (blues, country) có MFCC pattern khác do harmonic content khác.

- **Chroma STFT**
   - Ý nghĩa: tóm tắt năng lượng theo 12 pitch classes (bỏ qua octave) — tốt để phát hiện key, chord progression.
   - Phạm vi: giá trị normalized; mean gần 0..1.
   - Genre khác biệt: jazz/classical hiển thị phân bố chroma phong phú và biến đổi, pop thường có chord progression lặp lại (các cromas cụ thể mạnh), hiphop less harmonic => cromas mờ.

- **RMS Energy**
   - Ý nghĩa: năng lượng (âm lượng) trung bình; variance cho biết dynamic range.
   - Phạm vi: 0..1 (normalized) hoặc tuỳ scale của frame.
   - Genre khác biệt: metal/rock/disco => `rms_mean` cao; classical có `rms_var` lớn do nhiều đổi động từ pianissimo tới fortissimo; lo-fi hoặc acoustic có `rms_mean` thấp.

- **Spectral Centroid**
   - Ý nghĩa: trung tâm quỹ đạo tần số (weighted mean frequency) — biểu thị brightness.
   - Phạm vi: 0..sr/2 (Hz) nhưng thường scaled/normalized trong features.
   - Genre khác biệt: metal và pop (presence nhiều energy cao) có centroid cao; reggae/blues có centroid thấp hơn.

- **Spectral Bandwidth**
   - Ý nghĩa: độ phân tán quanh trung tâm — biểu thị complexity/tone color.
   - Genre khác biệt: electronic/metal có bandwidth lớn; solo-instrument pieces (classical solo) có bandwidth nhỏ hơn.

- **Spectral Rolloff**
   - Ý nghĩa: tần số dưới đó chứa X% năng lượng (thường 85%).
   - Genre khác biệt: tracks with bright hi-frequency content (electronic, disco) có rolloff cao; bass-heavy genres có rolloff thấp.

- **Zero Crossing Rate (ZCR)**
   - Ý nghĩa: tần suất signal đổi dấu — proxy cho noisiness hoặc high-frequency transient content.
   - Genre khác biệt: percussive-heavy (hiphop, rock) có ZCR cao; classical hoặc ambient có ZCR thấp.

- **Harmony / Percussive (HPSS)**
   - Ý nghĩa: tách phần harmonic (sustained tones) và percussive (transients). Thông số mean/var cho biết sự thống trị harmonic hay rhythmic.
   - Genre khác biệt: classical/jazz → harmony cao; rock/hiphop/disco → percussive cao.

- **Tempo (BPM)**
   - Ý nghĩa: nhịp độ ước lượng. Trên 3s có giới hạn về độ tin cậy nhưng phân bố global vẫn hữu ích.
   - Genre khác biệt: disco/edm: ~110–140 BPM; pop: ~100–130 BPM; hiphop: ~70–110 BPM (tuỳ subgenre); metal: có thể vượt 140 BPM; blues/classical: thường chậm hơn.

Những khác biệt trên là xu hướng chung — không phải luật tuyệt đối. Trong dataset nhỏ như GTZAN, quality/recording/timbre cá nhân của track có thể dẫn tới outlier; do đó model học kết hợp nhiều feature (MFCC + chroma + RMS + HPSS) để phân biệt.

### 3.3 Lưu ý khi dùng các feature (practical tips)

- Luôn chuẩn hóa (`StandardScaler`) trên train rồi áp dụng cho val/test.
- Tránh dùng `filename` hoặc các ID raw như feature đầu vào (risk of leakage).
- Khi dùng `features_3_sec.csv`, luôn split theo group (track_id) — không được random row.

---

## 3.1) Giải thích chi tiết Leakage — tại sao nó xảy ra?

Ví dụ đơn giản:
- Track 0 (blues) cắt thành segment 0.0, 0.1, 0.2, ..., 0.9 (10 cái)
- Track 1 (blues) cắt thành segment 1.0, 1.1, ..., 1.9
- ...
- Total: 1000 × 10 = 10,000 segment

**Nguy hiểm**:
```
# WRONG (random shuffle):
train: [0.0, 0.3, 1.2, 0.7, ...]  # mixed segments từ track 0,1,...
test:  [0.1, 0.5, 1.1, 2.0, ...]  # CÓ 0.1 (cùng track với 0.0, 0.3 trong train!)
```

→ Model học: "segment có MFCC pattern thế này + RMS pattern thế này → blues". Nhưng thực ra học luôn "track ID 0 → blues", vì segment cùng track đều cùng recording.

**Đúng** (group-stratified split theo track ID — cách pipeline hiện dùng):
```
train: tracks 0, 1, 3, 4, 5, 7, 8, 9   (800 segs = 8 tracks × 10 segs)
val:   tracks 2, 6                       (200 segs = 2 tracks × 10 segs)
test:  tracks từ partition riêng         (tracks không xuất hiện ở train/val)
```

→ Model không bao giờ thấy cùng track ở train+val/test. Đánh giá **thật**.

**Cơ chế trong code**: `src/data/split.py::make_group_splits_from_filenames` trích xuất track group từ tên file (`blues.00000.0.wav` → group `blues.00000`), sau đó split theo group. Split được lưu tại `Data/splits/split_by_track_v2.csv`.

---



## 4) Tiền xử lý (Preprocessing) — kèm toán học và ý nghĩa

Pipeline có 2 track preprocessing tương ứng 2 kiểu feature.

### 4.1 Track A: Tabular CSV (giống notebook)

#### (A) Label encoding

Chuyển `label` (string) → chỉ số lớp:

$$f: \{\text{genre strings}\} \to \{0,1,\dots,C-1\}$$

**Ý nghĩa**: Các model machine learning yêu cầu đầu ra là số (integer hoặc float), không phải string. `LabelEncoder` tạo mapping:
- blues → 0, classical → 1, ..., rock → 9

**Quan trọng**: Fit LabelEncoder **trên train** rồi áp dụng cho val/test. Nếu fit trên toàn bộ dữ liệu → test set bị "leak" thông tin từ train.

#### (B) Drop cột không dùng cho model

`filename` không mang thông tin âm học trực tiếp (và có thể gây leakage theo id) nên loại bỏ khỏi $X$.

#### (C) Standardization (z-score scaling)

Nhiều thuật toán (SVM, Logistic Regression, MLP) **rất nhạy** với thang đo feature. Ví dụ:
- MFCC range: [0, 100]
- RMS range: [0, 1]
- Spectral centroid range: [0, 22050] (Hz)

Nếu không chuẩn hóa, features có range lớn → model sẽ ignore features có range nhỏ.

Ta chuẩn hóa từng feature $j$:

$$\mu_j = \frac{1}{N_{train}}\sum_{i \in \text{train}} x_{ij}, \quad \sigma_j = \sqrt{\frac{1}{N_{train}}\sum_{i \in \text{train}} (x_{ij}-\mu_j)^2}$$

$$\tilde{x}_{ij} = \frac{x_{ij}-\mu_j}{\sigma_j + \epsilon}$$

**Quan trọng**: $\mu_j, \sigma_j$ chỉ được fit trên **train**, rồi áp dụng cho val/test để tránh "peek" vào test.

**Ý nghĩa**: Sau chuẩn hóa, mỗi feature có mean ≈ 0, std ≈ 1 → model "công bằng" với tất cả features → hội tụ nhanh hơn, kết quả ổn định.

#### (D) Split train/val/test (stratified)

Giữ tỷ lệ lớp gần giống nhau giữa các split bằng stratification. Mục tiêu:
- **train** (60-70%): học tham số
- **val** (10-15%): chọn hyperparameter / early stopping (không dùng để update tham số)
- **test** (15-20%): ước lượng performance cuối cùng (chỉ chạy 1 lần, cuối cùng)

**Stratified** = mỗi genre chiếm ~10% trong mỗi split. Nếu split ngẫu nhiên, có thể train thiếu classical (0%) nhưng test có 20% → không đánh giá được.

Pipeline lưu split vào `data/splits/*.csv` để reproducible (chạy 2 lần lại được train/val/test giống nhau).

**Đặc biệt với `tabular_csv`**: pipeline dùng `make_group_splits_from_filenames` để extract `track_id` từ `filename` column và split theo group. Kết quả split được lưu ở `Data/splits/split_by_track_v2.csv`.

### 4.2 Track B: End-to-end raw audio → mel spectrogram

#### (A) Waveform

Audio rời rạc $x[n]$ với sample rate $sr$ Hz. Thời gian tương ứng:

$$t = \frac{n}{sr}$$

**Ý nghĩa**: 44.1 kHz = 44,100 sample/giây → 1 giây audio = 44,100 số. 3 giây = 132,300 số. Quá nhiều để model xử lý trực tiếp. Cần feature extraction.

#### (B) STFT (Short-Time Fourier Transform)

Âm nhạc biến đổi theo thời gian → ta dùng STFT trên từng frame:

$$X(m,k) = \sum_{n=0}^{N-1} x[n+mH] \, w[n] \, e^{-j2\pi kn/N}$$

- $w[n]$: window (Hann, Hamming) — giảm rò rỉ spectral ở biên frame
- $N$: FFT size (thường 2048) — độ phân giải tần số
- $H$: hop length (thường 512) — khoảng cách giữa các frame (50% overlap = $H = N/2$)

Spectrogram magnitude/power:

$$S(m,k) = |X(m,k)|^2$$

**Ý nghĩa**: Từ time-domain raw audio → frequency-domain spectrogram. Mỗi cell $(m, k)$ = năng lượng ở tần số $k$ ở khung thời gian $m$. Dễ nhìn pattern: bass (row dưới), treble (row trên), drum beat (vertical lines).

#### (C) Mel filter bank → Mel spectrogram

Phổ tần số không được perceived **tuyến tính** bởi tai người. Tần số cao được compress hơn.

Mel scale (công thức phổ biến):

$$m = 2595 \, \log_{10}\left(1 + \frac{f}{700}\right)$$

Mel filter bank $M$ (triangular filters) gom năng lượng theo dải mel:

$$S_{\text{mel}} = M S$$

Đưa về dB (log-compression):

$$S_{\text{dB}} = 10\log_{10}(S_{\text{mel}} + \epsilon)$$

**Ý nghĩa**: 
- **Mel scale**: Gần với cách tai người nghe (100 Hz ≈ 200 Hz khoảng cách nhỏ; 8000 Hz ≈ 9000 Hz khoảng cách lớn trong Mel scale). Giúp model không "lãng phí" capacity học frequency resolution ở region mà tai không phân biệt tốt.
- **Log compression**: Dynamic range audio rất lớn (quiet violin vs. drums) → log nén lại → model training ổn định hơn. Con người cũng perceive volume theo log scale (mỗi +10dB nghe "gấp đôi" to).

**Kết quả**: Mel spectrogram shape $(T, 128)$ (T ≈ 130 timeframes, 128 mel bands) → CNN/LSTM có thể xử lý.

#### (D) Split theo track group (chống leakage segment)

Một track 30s được cắt thành nhiều segment 3s. Để tránh leakage, pipeline gán `group_id = track_id` và split theo group:
- tất cả segment của 1 track chỉ thuộc 1 split.

---

## 5) Các mô hình phân loại (Model zoo) và ý nghĩa

### 5.1 Classical ML baselines (tabular)

Những model này được thiết kế cho dữ liệu **tabular** — vector feature 58D. Chúng nhanh, dễ giải thích, và thường là baseline tốt trước khi thử deep learning:

- **KNN (k-Nearest Neighbors)**: dự đoán theo đa số $k$ láng giềng gần nhất trong không gian feature. Đơn giản nhưng nhạy với scaling và khoảng cách metric. Tốt cho dataset nhỏ.
- **SVM (Support Vector Machine)**: tìm siêu phẳng (hyperplane) phân tách các lớp với lề cực đại. Kernel **RBF** (Radial Basis Function) giúp xử lý biên phi tuyến (non-linear). **Linear** cho biên thẳng. Mạnh với dữ liệu cao chiều.
- **Logistic Regression (multinomial)**: baseline tuyến tính — học một hyperplane cho mỗi lớp. Nhanh, dễ tuỳ chỉnh regularization, kết quả dễ diễn giải.
- **Naive Bayes / LDA / QDA**: giả định dữ liệu tuân theo phân phối Gaussian. Nhanh, ít tham số, tốt khi giả định này đúng.
- **RandomForest / ExtraTrees / AdaBoost / GBDT**: **ensemble** — kết hợp nhiều cây quyết định. Không cần tuỳ chỉnh scaling, tự học tương tác giữa features, rất mạnh trên tabular.
- **XGBoost / LightGBM**: gradient boosting tối ưu cao — tăng cường và cải thiện cây lỗi. Thường cho kết quả tốt nhất trên tabular, nhưng cần tuỳ chỉnh nhiều hyperparameter.

### 5.2 Neural models (PyTorch)

Dành cho **dữ liệu phức tạp** (raw spectrogram):

- **MLP (Multi-Layer Perceptron)** — tabular: tương đương logic notebook. Dùng Dense layers + Dropout. Loss = Cross Entropy. Linh hoạt nhưng cần chuẩn hóa đầu vào tốt.
- **CNN (Convolutional Neural Network)** — mel spectrogram: convolution 2D học **spatial pattern** trên tần số (vertical) × thời gian (horizontal). Giúp phát hiện motif (ví dụ "drum pattern", "chord progression"). Tốt cho dữ liệu grid-like.
- **LSTM (Long Short-Term Memory)** — mel spectrogram: xem mel như **chuỗi** theo thời gian $t$, học **phụ thuộc dài hạn** (long-range dependency). Giúp model nhớ context xa → tốt cho cấu trúc bài nhạc (intro → verse → chorus).

---

## 6) Training objective (toán học)

### 6.1 Softmax

Với logits $z \in \mathbb{R}^C$:

$$p_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$$

### 6.2 Cross Entropy (multi-class)

Với nhãn đúng $y$:

$$\mathcal{L} = -\log(p_y)$$

Notebook dùng `sparse_categorical_crossentropy`; trong pipeline PyTorch dùng `CrossEntropyLoss` (tương đương softmax + negative log-likelihood).

### 6.3 Adam optimizer (tóm tắt)

Với gradient $g_t$:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

$$\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$$

---

## 7) Evaluation metrics (ý nghĩa + công thức)

### 7.1 Confusion matrix

Ma trận $C$ với $C_{ij}$ = số mẫu lớp thật $i$ được dự đoán thành $j$.

### 7.2 Accuracy

$$\text{Acc} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = y_i]$$

### 7.3 Precision / Recall / F1 (1 lớp)

$$P = \frac{TP}{TP+FP}, \quad R = \frac{TP}{TP+FN}, \quad F1 = \frac{2PR}{P+R}$$

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
python src/main.py --stage feature --run-name mel01 --feature-kind mel_from_audio
python src/main.py --stage train --run-name mel01 --model cnn_mel --feature-kind mel_from_audio
python src/main.py --stage evaluate --run-name mel01 --model cnn_mel --feature-kind mel_from_audio
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

---

## 8) Tóm tắt luồng dữ liệu (Data Flow Summary)

Để hiểu rõ cách hệ thống hoạt động, hãy theo dõi luồng dữ liệu từ raw audio đến dự đoán:

### Track A: Tabular Features (CSV)

```
1. Raw audio .wav (30s, 44.1 kHz)
   ↓
2. Feature extraction offline (Librosa)
   - MFCC, Spectral Centroid, RMS, ...
   ↓
3. features_3_sec.csv / features_30_sec.csv
   - 58D vector per segment/track
   ↓
4. Preprocessing: Label encode → Drop filename → Standardize
   ↓
5. Split (stratified, GROUP by track_id để tránh leakage)
   ↓
6. Train: KNN/SVM/RandomForest/MLP
   - Input: 58D vector
   - Output: logits (10 classes)
   ↓
7. Softmax + argmax → predicted genre
```

### Track B: End-to-End (Mel Spectrogram)

```
1. Raw audio .wav (30s, 44.1 kHz)
   ↓
2. On-the-fly feature extraction:
   - Waveform: 44.1k samples/s
   ↓
3. STFT: Chuyển từ time-domain → frequency-domain
   - Window size N=2048, hop H=512
   - Tạo spectrogram $(m, k)$: time × frequency
   ↓
4. Mel filter bank + log compression
   - Chuẩn hóa theo cách tai người nghe
   - Kết quả: $(T, 128)$ mel spectrogram
   ↓
5. Segment cắt 3s, Group split theo track_id
   ↓
6. Train: CNN/LSTM (PyTorch)
   - Input: $(1, T, 128)$ image / sequence
   - Convolution/LSTM → tìm spatial/temporal patterns
   ↓
7. Softmax + argmax → predicted genre
```

### Các bước quan trọng để tránh lỗi

| Bước | Sai lầm | Giải pháp |
|------|--------|----------|
| Label Encoding | Fit trên toàn bộ dữ liệu | Fit **chỉ trên train** |
| Standardization | Fit trên toàn bộ dữ liệu | Fit **chỉ trên train**, apply đến val/test |
| Train/Val/Test Split | Random shuffle toàn bộ | Stratified split, **group by track_id** |
| Mel Spectrogram | Không normalize log | Bắt buộc log-compression để tránh model instability |
| Feature Selection | Dùng filename làm feature | Drop filename (potential leakage) |

---

## 9) Quick Reference: Features và ý nghĩa

| Feature | Range | Ý nghĩa | Phân biệt |
|---------|-------|----------|----------|
| **MFCC** | [0, 100]+ | Hệ số cepstral (voice-like), **mạnh nhất** | Giọng hát, timbre |
| **Chroma** | [0, 1] | Nốt nhạc (C, C#, ..., B) | Hòa âm, key của bài |
| **RMS Energy** | [0, 1] | Năng lượng trung bình (volume) | Vocal vs Instrumental |
| **Spectral Centroid** | [0, 22050] Hz | "Vị trí trọng tâm" tần số | Bright vs Dark |
| **Spectral Bandwidth** | [0, 22050] Hz | Độ "rộng" của phổ | Timbre complexity |
| **Spectral Rolloff** | [0, 22050] Hz | 85% năng lượng nằm dưới tần số nào | Harmonic content |
| **Zero Crossing Rate** | [0, 1] | Số lần cắt qua 0 / frame | Voiced vs Unvoiced |
| **Tempo** | [0, 300] BPM | Nhịp độ | Genre rhythm |
| **Harmony** | Real | Phần harmonic (string/wind) | Instrument type |
| **Percussive** | Real | Phần percussion (drums) | Rhythm presence |

---


