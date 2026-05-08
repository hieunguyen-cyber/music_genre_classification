# Math & Signal Processing Notes

Tài liệu này giải thích các khối toán học chính trong pipeline.

## 1) Fourier Transform (FT)

Với tín hiệu rời rạc \(x[n]\), biến đổi Fourier rời rạc (DFT):

\[
X[k] = \sum_{n=0}^{N-1} x[n] \\, e^{-j 2\pi kn/N}
\]

**Intuition**
- DFT “phân rã” tín hiệu thành các sóng sin/cos với các tần số rời rạc.
- \(|X[k]|\) cho biết năng lượng tại tần số \(k\).

## 2) STFT (Short-Time Fourier Transform)

Âm nhạc thay đổi theo thời gian, nên ta dùng STFT:

\[
X(m, k) = \sum_{n=0}^{N-1} x[n+mH] \\, w[n] \\, e^{-j2\pi kn/N}
\]

- \(w[n]\): window (Hann, Hamming, ...)
- \(H\): hop length
- \(m\): frame index

**Trực quan**

```
signal:  [..............audio..............]
window:       [----frame----]
shift:             H
frames:  [----][----][----][----]...
```

## 3) Mel scale & Mel Spectrogram

Mel scale (một công thức phổ biến):

\[
m = 2595 \\, \\log_{10}\\left(1 + \\frac{f}{700}\\right)
\]

**Mel spectrogram**
1) STFT -> power spectrogram \(S\)
2) Apply mel filter bank \(M\):

\[
S_{mel} = M S
\]

## 4) MFCC

MFCC thường được tính theo pipeline:

1) Mel spectrogram \(S_{mel}\)
2) Log:
\[
L = \\log(S_{mel} + \\epsilon)
\]
3) DCT:
\[
\\text{MFCC} = \\text{DCT}(L)
\]

**Ý nghĩa**
- MFCC giữ “shape” của phổ (timbre) và bỏ bớt chi tiết cao tần.

## 5) Softmax & Cross Entropy

Với logits \(z \\in \\mathbb{R}^C\):

\[
p_i = \\frac{e^{z_i}}{\\sum_{j=1}^C e^{z_j}}
\]

Cross Entropy cho nhãn đúng \(y\):

\[
\\mathcal{L} = -\\log(p_y)
\]

Notebook dùng `sparse_categorical_crossentropy`; trong PyTorch ta dùng `CrossEntropyLoss` (tương đương: softmax + NLL).

## 6) Backpropagation (high-level)

Ta tính gradient:
\[
\\nabla_\\theta \\mathcal{L}
\]
và cập nhật tham số theo optimizer.

## 7) Adam optimizer

Với gradient \(g_t\):

\[
m_t = \\beta_1 m_{t-1} + (1-\\beta_1) g_t
\]
\[
v_t = \\beta_2 v_{t-1} + (1-\\beta_2) g_t^2
\]

Bias-correction:
\[
\\hat{m}_t = \\frac{m_t}{1-\\beta_1^t}, \\quad \\hat{v}_t = \\frac{v_t}{1-\\beta_2^t}
\]

Update:
\[
\\theta_t = \\theta_{t-1} - \\alpha \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t}+\\epsilon}
\]

## 8) Confusion Matrix, Precision/Recall/F1

Confusion matrix \(C\\) có phần tử \(C_{ij}\) = số mẫu lớp thật \(i\) được dự đoán thành \(j\).

Với binary case:
- Precision \(= \\frac{TP}{TP+FP}\)
- Recall \(= \\frac{TP}{TP+FN}\)
- F1 \(= \\frac{2PR}{P+R}\)

Với multi-class:
- Macro-F1: trung bình F1 của từng lớp (đối xử công bằng với lớp hiếm).
- Weighted-F1: weighted theo support (phản ánh phân bố dữ liệu).
