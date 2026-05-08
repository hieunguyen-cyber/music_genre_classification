# Visualization Guide

Tài liệu này mô tả các visualization được sinh bởi stage `visualize` và ý nghĩa của chúng.

## 1) Waveform

**Câu hỏi trả lời**
- Âm lượng có bị clipping không?
- Có đoạn silence dài không?

**Cách đọc**
- Trục X: thời gian (s)
- Trục Y: biên độ mẫu âm thanh

## 2) STFT spectrogram (dB)

**Câu hỏi trả lời**
- Năng lượng theo tần số thay đổi theo thời gian thế nào?
- Có nhiều transient (percussive) hay harmonic (sustained) không?

**Cách đọc**
- X: thời gian
- Y: tần số (log)
- Màu: cường độ (dB)

## 3) Mel spectrogram

**Vì sao hữu ích**
- Mel scale gần với cảm nhận thính giác: độ phân giải cao ở tần số thấp, thấp hơn ở tần số cao.
- Rất phổ biến làm input cho CNN.

## 4) MFCC heatmap

**Vì sao hữu ích**
- MFCC tóm tắt “spectral envelope” (timbre) — yếu tố quan trọng cho phân loại thể loại.
- Dễ dùng cho các model đơn giản (SVM/KNN/MLP).

## 5) Chromagram

**Vì sao hữu ích**
- Thể hiện năng lượng theo 12 pitch classes (C, C#, ..., B).
- Gợi ý về hòa âm (chords) và tonal content.

## 6) Spectral contrast

**Vì sao hữu ích**
- Đo chênh lệch năng lượng giữa peak và valley trên các sub-bands.
- Hữu ích phân biệt chất âm, nhạc cụ.

## 7) Tempogram

**Vì sao hữu ích**
- Mô tả nhịp điệu / periodicity.
- Một số genre phân biệt mạnh theo rhythm.

## 8) HPSS (Harmonic/Percussive)

**Vì sao hữu ích**
- Tách phần harmonic (giai điệu/hòa âm) và percussive (trống, transient).
- Từ đó phân tích genre overlap theo “texture”.

## 9) Embeddings (PCA / t-SNE / UMAP)

**Câu hỏi trả lời**
- Feature space có “tách lớp” theo genre không?
- Genre nào overlap mạnh?

**Lưu ý**
- PCA: tuyến tính, preserve variance.
- t-SNE/UMAP: phi tuyến, tốt cho visualization nhưng không phải thước đo định lượng.

## 10) Feature correlation heatmap

**Câu hỏi trả lời**
- Feature nào trùng thông tin (correlated) mạnh?
- Có dấu hiệu leakage / redundant feature không?
