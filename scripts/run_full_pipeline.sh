#!/usr/bin/env bash
# =============================================================================
# run_full_pipeline.sh
# Full end-to-end pipeline for Music Genre Classification (GTZAN).
#
# Usage:
#   bash scripts/run_full_pipeline.sh [OPTIONS]
#
# Options (passed through to every python call):
#   --device    cpu|cuda|mps|auto   (default: auto)
#   --seed      <int>               (default: 42)
#   --epochs    <int>               override training epochs
#   --num-workers <int>             DataLoader workers
#
# Examples:
#   bash scripts/run_full_pipeline.sh                    # full run, all models
#   bash scripts/run_full_pipeline.sh --device cpu       # force CPU
#   bash scripts/run_full_pipeline.sh --epochs 50        # quick test
#
# Pipeline stages run:
#   [0] Preprocess      — scan raw audio, build metadata.csv
#   [1] Feature vis     — waveform/spectrogram/MFCC/chroma/HPSS visualizations
#   [2] Tabular (Track A, v2 group-split — no leakage)
#       feature → train → evaluate → visualize
#       Models: mlp, knn, svm_rbf, linear_svm, logreg, naive_bayes,
#               lda, qda, random_forest, extra_trees, ada_boost, gbdt,
#               xgboost, lightgbm
#   [3] Mel (Track B, track-disjoint)
#       feature → train → evaluate → visualize
#       Models: cnn_mel, lstm_mel, crnn_mel
#   [4] Summary         — print final accuracy table
#
# New features integrated in this version:
#   - Track A uses group-level split (make_group_splits_from_filenames)
#     → zero track-level leakage (split_by_track_v2.csv)
#   - Mixup augmentation (data.augmentation.mixup in config.yaml)
#   - SpecAugment for mel models (data.augmentation.spec_augment)
#   - Track-level prediction aggregation saved to reports/
#   - CRNN (CNN + BiLSTM + Attention) as a third mel model
# =============================================================================

set -euo pipefail

# ── Resolve project root (parent of scripts/) ────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
MAIN="src/main.py"
# Capture any extra CLI args forwarded to every python call.
# Use ${*:-} so the variable is never "unbound" under set -u.
EXTRA_ARGS="${*:-}"

# ── Color helpers ─────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; NC='\033[0m'
log_section() { echo -e "\n${BLUE}══════════════════════════════════════════════${NC}"; \
                echo -e "${BLUE}  $1${NC}"; \
                echo -e "${BLUE}══════════════════════════════════════════════${NC}"; }
log_ok()      { echo -e "${GREEN}  ✔ $1${NC}"; }
log_skip()    { echo -e "${YELLOW}  ⚡ SKIPPED: $1${NC}"; }

run() {
    # $@ = stage-specific args; $EXTRA_ARGS = global pass-through args
    echo "  $ $PYTHON $MAIN $* ${EXTRA_ARGS:-}"
    # shellcheck disable=SC2086
    $PYTHON "$MAIN" "$@" ${EXTRA_ARGS:-}
}

# ── Helper: run a model pipeline (train → evaluate → visualize) ───────────────
run_model() {
    local RUN_NAME="$1"; local MODEL="$2"; shift 2
    local EXTRA="$*"  # e.g. --feature-kind mel_from_audio
    run --stage train    --run-name "$RUN_NAME" --model "$MODEL" $EXTRA
    run --stage evaluate --run-name "$RUN_NAME" --model "$MODEL" $EXTRA
    run --stage visualize --run-name "$RUN_NAME" $EXTRA
    log_ok "$RUN_NAME complete"
}

START_TIME=$(date +%s)

# =============================================================================
# [0] PREPROCESS — scan genres_original/, write Data/processed/metadata.csv
# =============================================================================
log_section "[0] Preprocess"
run --stage preprocess --run-name preprocessing
log_ok "Preprocessing done"

# =============================================================================
# [1] FEATURE VISUALIZATION — waveforms, spectrograms, embeddings (once each)
# =============================================================================
log_section "[1] Feature Visualization"

echo "  Tabular feature visualizations..."
run --stage visualize-features --run-name visualize-features-tabular
log_ok "Tabular feature vis done"

echo "  Mel feature visualizations..."
run --stage visualize-features --run-name visualize-features-mel \
    --feature-kind mel_from_audio
log_ok "Mel feature vis done"

# =============================================================================
# [2] TRACK A: TABULAR (group-split v2, no leakage)
#
# The feature stage generates:
#   Data/splits/split_by_track_v2.csv   — group-level (track-disjoint)
#   data/features/tabular_features_v2.npz
#   data/features/tabular_scaler_v2.pkl
# =============================================================================
log_section "[2] Track A — Tabular (group-split v2)"

echo "  Building tabular feature cache (v2 group split)..."
run --stage feature --run-name tabular-features
log_ok "Tabular feature cache ready"

# ── Sklearn baselines ─────────────────────────────────────────────────────────
TABULAR_SKLEARN_MODELS=(
    "knn"
    "svm_rbf"
    "linear_svm"
    "logreg"
    "naive_bayes"
    "lda"
    "qda"
    "random_forest"
    "extra_trees"
    "ada_boost"
    "gbdt"
)
for MODEL in "${TABULAR_SKLEARN_MODELS[@]}"; do
    log_section "  [2] tabular_${MODEL}"
    run_model "tabular_${MODEL}" "$MODEL"
done

# ── Optional: XGBoost / LightGBM (install separately if needed) ───────────────
for MODEL in "xgboost" "lightgbm"; do
    log_section "  [2] tabular_${MODEL} (optional)"
    if $PYTHON -c "import ${MODEL}" 2>/dev/null; then
        run_model "tabular_${MODEL}" "$MODEL"
    else
        log_skip "${MODEL} not installed — pip install ${MODEL}"
    fi
done

# ── PyTorch MLP ───────────────────────────────────────────────────────────────
log_section "  [2] tabular_mlp"
run_model "tabular_mlp" "mlp"

# =============================================================================
# [3] TRACK B: MEL SPECTROGRAM (track-disjoint, SpecAugment + CRNN)
#
# The feature stage generates:
#   Data/splits/mel_split_by_track_v1.csv   — group-level (unchanged)
#   data/features/mel_features_v1.npz
# =============================================================================
log_section "[3] Track B — Mel (track-disjoint)"

echo "  Extracting mel spectrogram features (this may take several minutes)..."
run --stage feature --run-name mel-features --feature-kind mel_from_audio
log_ok "Mel feature cache ready"

# ── CNN ───────────────────────────────────────────────────────────────────────
log_section "  [3] mel_cnn"
run_model "mel_cnn" "cnn_mel" --feature-kind mel_from_audio

# ── LSTM ──────────────────────────────────────────────────────────────────────
log_section "  [3] mel_lstm"
run_model "mel_lstm" "lstm_mel" --feature-kind mel_from_audio

# ── CRNN (CNN + BiLSTM + Attention) — new model ───────────────────────────────
log_section "  [3] mel_crnn"
run_model "mel_crnn" "crnn_mel" --feature-kind mel_from_audio

# =============================================================================
# [4] SUMMARY — collect accuracy from all evaluation reports
# =============================================================================
log_section "[4] Summary"

$PYTHON - << 'PYEOF'
import pathlib, re, sys

outputs = pathlib.Path("outputs")
rows = []
for report in sorted(outputs.glob("*/reports/evaluation_report.md")):
    run = report.parent.parent.name
    text = report.read_text(encoding="utf-8")
    acc  = re.search(r"Accuracy:\s*\*\*([0-9.]+)\*\*", text)
    f1   = re.search(r"Macro F1:\s*\*\*([0-9.]+)\*\*", text)
    roc  = re.search(r"ROC-AUC.*?:\s*\*\*([0-9.]+)\*\*", text)
    if acc:
        rows.append((
            run,
            float(acc.group(1)),
            float(f1.group(1)) if f1 else None,
            float(roc.group(1)) if roc else None,
        ))

rows.sort(key=lambda r: r[1], reverse=True)
print(f"\n{'Run':<30} {'Accuracy':>10} {'Macro-F1':>10} {'ROC-AUC':>10}")
print("-" * 65)
for run, acc, f1, roc in rows:
    f1s  = f"{f1:.4f}" if f1  is not None else "    —"
    rocs = f"{roc:.4f}" if roc is not None else "    —"
    print(f"{run:<30} {acc:>10.4f} {f1s:>10} {rocs:>10}")

# Also report track-level accuracy for mel runs
print("\n── Track-level accuracy (mel models, mean_proba aggregation) ──")
for trep in sorted(outputs.glob("*/reports/track_level_mean_proba.md")):
    run = trep.parent.parent.name
    text = trep.read_text(encoding="utf-8")
    m = re.search(r"Track-level Accuracy:\s*\*\*([0-9.]+)\*\*", text)
    if m:
        print(f"  {run:<28} track_acc={float(m.group(1)):.4f}")
PYEOF

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINS=$(( ELAPSED / 60 )); SECS=$(( ELAPSED % 60 ))

echo ""
log_ok "Full pipeline complete in ${MINS}m ${SECS}s"
echo ""
echo "  Outputs: outputs/"
echo "  To view a run:  ls outputs/<run_name>/reports/"
echo "  Track-level reports: outputs/<mel_run>/reports/track_level_*.md"
