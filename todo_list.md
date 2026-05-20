# TODO List — Music Genre Classification Pipeline

> **Ngày tạo:** 2026-05-19  
> **Trạng thái:** Đang thực hiện

---

## ✅ Đã hoàn thành

### [FIX-1] Data Leakage — Track A (tabular CSV)
**Vấn đề:** `split_v1.csv` được tạo bằng cách split theo từng row/segment ngẫu nhiên, khiến 966/1000 tracks có segments ở nhiều splits (train+test).

**Đã sửa:**
- [x] `src/data/split.py` — Thêm `make_group_splits_from_filenames()`: trích xuất track ID từ filename (`blues.00000.0.wav` → `blues.00000`) và split theo group.
- [x] `src/main.py` → `stage_feature()` — Thay `make_splits()` bằng `make_group_splits_from_filenames()` cho `tabular_csv` track.
- [x] `configs/feature.yaml` — Đổi tên split/cache sang `v2` (`split_by_track_v2.csv`, `tabular_features_v2.npz`) để pipeline tự tạo lại split sạch.
- [x] Verified: 0/1000 tracks bị overlap trong split mới.

**File liên quan:**
- `src/data/split.py`
- `src/main.py` (lines 156–175)
- `configs/feature.yaml`

---

### [IMPL-1] Mixup Augmentation (giải quyết Genre Overlap)
**Vấn đề:** Nhiều genre chia sẻ nhạc cụ/nhịp điệu → model học boundary quá cứng → nhầm lẫn ở vùng giao thoa.

**Đã implement:**
- [x] `src/data/augmentation.py` — `MixupCfg` dataclass và `mixup_batch()` function.
- [x] `configs/config.yaml` — `data.augmentation.mixup` config block.
- [x] Unit test: `x_mix.shape`, `y_soft.sum(axis=1) ≈ 1.0` — PASS.

---

### [IMPL-2] SpecAugment (giải quyết Domain Shift)
**Vấn đề:** Dataset nhỏ, thu âm/production khác nhau → model overfit đặc trưng recording-specific.

**Đã implement:**
- [x] `src/data/augmentation.py` — `SpecAugmentCfg` dataclass và `apply_spec_augment()` function.
- [x] `configs/config.yaml` — `data.augmentation.spec_augment` config block.
- [x] Unit test: shape in = shape out, masking hoạt động — PASS.

---

## 🔲 Còn lại — Cần làm

### [TODO-1] Wire Mixup vào Training Loop
**Priority:** 🔴 High  
**File cần sửa:** `src/models/trainer.py`

**Việc cần làm:**
- [ ] Trong training loop (hàm `train()`), đọc `cfg.data.augmentation.mixup`.
- [ ] Nếu `enabled=True` và `rng.random() < prob`: gọi `mixup_batch(x_batch, y_batch, cfg=mixup_cfg)`.
- [ ] Thay `CrossEntropyLoss` bằng soft-label compatible loss khi mixup bật:
  ```python
  # Cách A — dùng soft cross-entropy trực tiếp:
  loss = -(y_soft * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
  
  # Cách B — dùng formulation gốc của mixup (2 forward pass):
  loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
  ```
- [ ] Chỉ apply mixup trên train split (không áp dụng cho val/test).
- [ ] Thêm config flag vào `parse_args()` nếu cần override từ CLI.

---

### [TODO-2] Wire SpecAugment vào MelDataset
**Priority:** 🔴 High  
**File cần sửa:** `src/data/dataset.py`

**Việc cần làm:**
- [ ] Thêm `spec_augment_cfg: Optional[SpecAugmentCfg] = None` vào `MelDataset.__init__()`.
- [ ] Trong `__getitem__()`, nếu `spec_augment_cfg.enabled` và đang ở training phase:
  ```python
  from src.data.augmentation import apply_spec_augment
  x_np = apply_spec_augment(self.x[idx], self.spec_augment_cfg, rng=self._rng)
  ```
- [ ] Truyền config từ `stage_feature()` → `make_mel_dataloaders()` → `MelDataset` (chỉ cho train loader).
- [ ] Lưu ý: RNG phải là per-worker để tránh same augmentation trong mọi worker.

---

### [TODO-3] Track-level Prediction Aggregation
**Priority:** 🟡 Medium  
**File cần tạo:** `src/evaluation/track_aggregation.py`

**Việc cần làm:**
- [ ] Implement `aggregate_by_track(groups, y_true, y_pred, y_proba, mode)`:
  - `mode="majority_vote"`: lấy label xuất hiện nhiều nhất trong các segments của cùng track.
  - `mode="mean_proba"`: tính mean của `y_proba` rồi argmax.
- [ ] Tích hợp vào `stage_evaluate()` trong `main.py` — báo cáo cả segment-level và track-level accuracy.
- [ ] Lưu kết quả vào `outputs/<run>/reports/track_level_evaluation.md`.

**Lý do quan trọng:** Segment-level accuracy có thể không phản ánh đúng hiệu năng thực tế (1 bài nhạc = nhiều segments, nhưng người dùng chỉ cần 1 dự đoán/bài).

---

### [TODO-4] Re-run Tabular Experiments với Split v2
**Priority:** 🟡 Medium

**Việc cần làm:**
- [ ] Xóa cache cũ (bị tạo từ split v1):
  ```bash
  rm -f data/features/tabular_features_v1.npz
  rm -f data/features/tabular_scaler.pkl
  ```
- [ ] Re-run toàn bộ tabular models:
  ```bash
  python -m src.main --stage feature --feature-kind tabular_csv
  python -m src.main --stage train --model lightgbm
  python -m src.main --stage train --model svm_rbf
  python -m src.main --stage train --model xgboost
  python -m src.main --stage train --model mlp
  # ... các model khác
  ```
- [ ] Cập nhật bảng kết quả trong `SCIENTIFIC_REPORT.md` Section 7.1.
- [ ] So sánh v1 (leaky) vs v2 (clean) để định lượng mức độ inflation.

---

### [TODO-5] Stronger Mel Architectures (CRNN)
**Priority:** 🟢 Low  
**File cần tạo:** `src/models/crnn.py`

**Việc cần làm:**
- [ ] Implement `CRNNMel`: CNN front-end (3 conv blocks) + BiLSTM + attention pooling over time.
- [ ] Đăng ký vào `src/models/model.py::create_model()`.
- [ ] Thêm config vào `configs/model.yaml`.

---

### [TODO-6] Waveform Augmentation trong Mel Pipeline
**Priority:** 🟢 Low  
**File liên quan:** `src/data/audio_features.py`, `src/data/augmentation.py`

**Việc cần làm:**
- [ ] Trong `extract_mel_dataset()`, enable `AugmentCfg` và gọi `apply_augmentations()` trên waveform trước khi compute mel — chỉ trong training set.
- [ ] Thêm config: `features.mel_from_audio.augment.enabled`, `noise_std`, `time_stretch_*`, `pitch_shift_*`.
- [ ] Lưu ý: augmentation waveform cần chạy trước khi segment → tránh boundary artifact.

---

## Tóm tắt trạng thái

| ID | Vấn đề | Trạng thái | Priority |
|---|---|---|---|
| FIX-1 | Data leakage (Track A) | ✅ Done | — |
| IMPL-1 | Mixup implementation | ✅ Done | — |
| IMPL-2 | SpecAugment implementation | ✅ Done | — |
| TODO-1 | Wire Mixup → trainer | 🔲 Pending | 🔴 High |
| TODO-2 | Wire SpecAugment → MelDataset | 🔲 Pending | 🔴 High |
| TODO-3 | Track-level aggregation | 🔲 Pending | 🟡 Medium |
| TODO-4 | Re-run tabular (v2 split) | 🔲 Pending | 🟡 Medium |
| TODO-5 | CRNN architecture | 🔲 Pending | 🟢 Low |
| TODO-6 | Waveform aug in mel pipeline | 🔲 Pending | 🟢 Low |
