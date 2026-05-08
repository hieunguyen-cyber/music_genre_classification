# Experiments

File này là “lab notebook” dạng markdown để ghi lại các lần chạy (run) và kết quả.

## Default run

- Config: `configs/config.yaml`
- Features: `Data/features_3_sec.csv` (tabular, standardized)
- Model: MLP (giống logic notebook, nhưng implement bằng PyTorch để hỗ trợ CPU/CUDA/MPS nhất quán)

### Commands

```bash
make full RUN_NAME=default
```

### Artifacts

- Checkpoint: `outputs/default/checkpoints/best.pt`
- History: `outputs/default/reports/history.json`
- Eval report: `outputs/default/reports/evaluation_report.md`
- Figures: `outputs/default/figures/*.png`

## Gợi ý mở rộng

1) Baseline classical ML (đã hỗ trợ trong pipeline):

```bash
python src/main.py --stage train --run-name knn01 --model knn
python src/main.py --stage evaluate --run-name knn01 --model knn

python src/main.py --stage train --run-name svm01 --model svm_rbf
python src/main.py --stage evaluate --run-name svm01 --model svm_rbf
```

2) Thử `features_30_sec.csv` (1000 mẫu, ít hơn nhưng mỗi sample đầy đủ 30s).
3) Thử mel-spectrogram + CNN (nếu chuyển sang feature extraction từ raw audio).
4) Error analysis theo nhóm genre hay bị nhầm (dựa vào confusion matrix) và nghe lại samples.
