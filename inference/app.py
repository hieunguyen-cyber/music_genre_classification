import io
import json
import os
import pickle
import sys
import tempfile
import warnings
from pathlib import Path

import joblib
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FEATURES_DIR = PROJECT_ROOT / "Data" / "features"
SCALER_PATH = FEATURES_DIR / "tabular_scaler_v2.pkl"
LABEL_MAP_PATH = FEATURES_DIR / "label_map.json"

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 22050
SEGMENT_SECONDS = 3.0
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 20
FMAX = 8000
N_CLASSES = 10
INPUT_DIM = 58

# ── Genre labels ──────────────────────────────────────────────────────────────
with open(LABEL_MAP_PATH) as f:
    LABEL_MAP = json.load(f)
IDX_TO_GENRE = {v: k for k, v in LABEL_MAP.items()}
GENRES = [IDX_TO_GENRE[i] for i in range(N_CLASSES)]

# ── Scaler ────────────────────────────────────────────────────────────────────
with open(SCALER_PATH, "rb") as f:
    SCALER = pickle.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# Model definitions (must match training architectures)
# ═══════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, n_classes=N_CLASSES,
                 hidden_sizes=None, dropout=0.2):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128, 64, 32]
        layers = []
        prev = input_dim
        layers.append(nn.Dropout(dropout))
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True),
                           nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNNMel(nn.Module):
    def __init__(self, n_mels=N_MELS, n_classes=N_CLASSES,
                 channels=None, dropout=0.3):
        super().__init__()
        if channels is None:
            channels = (32, 64, 128)
        c1, c2, c3 = channels
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(dropout),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(dropout),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(c3, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


class LSTMMel(nn.Module):
    def __init__(self, n_mels=N_MELS, n_classes=N_CLASSES,
                 hidden_size=128, num_layers=2,
                 bidirectional=True, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_mels, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim), nn.Dropout(dropout),
            nn.Linear(out_dim, n_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.MaxPool2d(pool), nn.Dropout2d(p=dropout),
        )

    def forward(self, x):
        return self.block(x)


class AttentionPool(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        scores = self.attn(x).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        return (x * weights.unsqueeze(-1)).sum(dim=1)


class CRNNMel(nn.Module):
    def __init__(self, n_mels=N_MELS, n_classes=N_CLASSES,
                 cnn_channels=None, lstm_hidden=128,
                 lstm_layers=2, dropout=0.3):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = (32, 64, 128)
        c1, c2, c3 = cnn_channels
        self.cnn = nn.Sequential(
            ConvBlock(1, c1, pool=(2, 1), dropout=dropout),
            ConvBlock(c1, c2, pool=(2, 1), dropout=dropout),
            ConvBlock(c2, c3, pool=(2, 1), dropout=dropout),
        )
        freq_out = n_mels // 8
        rnn_input_size = c3 * freq_out
        self.lstm = nn.LSTM(
            input_size=rnn_input_size, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.attn_pool = AttentionPool(lstm_hidden * 2)
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_hidden * 2), nn.Dropout(p=dropout),
            nn.Linear(lstm_hidden * 2, n_classes),
        )

    def forward(self, x):
        x = self.cnn(x)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)
        x, _ = self.lstm(x)
        x = self.attn_pool(x)
        return self.head(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Model registry: name -> (model_class, kwargs, checkpoint_path)
# ═══════════════════════════════════════════════════════════════════════════════

def _find_ckpt(run_dir, names):
    for n in names:
        p = OUTPUTS_DIR / run_dir / "checkpoints" / n
        if p.exists():
            return p
    return None


SKLEARN_MODELS = {
    "knn":              (OUTPUTS_DIR / "tabular_knn" / "checkpoints" / "model.joblib"),
    "svm_rbf":          (OUTPUTS_DIR / "tabular_svm_rbf" / "checkpoints" / "model.joblib"),
    "linear_svm":       (OUTPUTS_DIR / "tabular_linear_svm" / "checkpoints" / "model.joblib"),
    "logreg":           (OUTPUTS_DIR / "tabular_logreg" / "checkpoints" / "model.joblib"),
    "naive_bayes":      (OUTPUTS_DIR / "tabular_naive_bayes" / "checkpoints" / "model.joblib"),
    "lda":              (OUTPUTS_DIR / "tabular_lda" / "checkpoints" / "model.joblib"),
    "qda":              (OUTPUTS_DIR / "tabular_qda" / "checkpoints" / "model.joblib"),
    "random_forest":    (OUTPUTS_DIR / "tabular_random_forest" / "checkpoints" / "model.joblib"),
    "extra_trees":      (OUTPUTS_DIR / "tabular_extra_trees" / "checkpoints" / "model.joblib"),
    "ada_boost":        (OUTPUTS_DIR / "tabular_ada_boost" / "checkpoints" / "model.joblib"),
    "gbdt":             (OUTPUTS_DIR / "tabular_gbdt" / "checkpoints" / "model.joblib"),
    "xgboost":          (OUTPUTS_DIR / "tabular_xgboost" / "checkpoints" / "model.joblib"),
    "lightgbm":         (OUTPUTS_DIR / "tabular_lightgbm" / "checkpoints" / "model.joblib"),
}

PYTORCH_MODELS = {
    "mlp": {
        "class": MLP,
        "kwargs": {"input_dim": INPUT_DIM, "n_classes": N_CLASSES},
        "ckpt": OUTPUTS_DIR / "tabular_mlp" / "checkpoints" / "best.pt",
    },
    "cnn_mel": {
        "class": CNNMel,
        "kwargs": {"n_mels": N_MELS, "n_classes": N_CLASSES},
        "ckpt": OUTPUTS_DIR / "mel_cnn" / "checkpoints" / "best.pt",
    },
    "lstm_mel": {
        "class": LSTMMel,
        "kwargs": {"n_mels": N_MELS, "n_classes": N_CLASSES},
        "ckpt": OUTPUTS_DIR / "mel_lstm" / "checkpoints" / "best.pt",
    },
    "crnn_mel": {
        "class": CRNNMel,
        "kwargs": {"n_mels": N_MELS, "n_classes": N_CLASSES},
        "ckpt": OUTPUTS_DIR / "mel_crnn" / "checkpoints" / "best.pt",
    },
}

MODEL_GROUPS = {
    "Deep Learning (Mel Spectrogram)": ["cnn_mel", "lstm_mel", "crnn_mel", "mlp"],
    "Ensemble / Tree": ["random_forest", "extra_trees", "ada_boost",
                        "gbdt", "xgboost", "lightgbm"],
    "Linear / Other": ["knn", "svm_rbf", "linear_svm", "logreg",
                       "naive_bayes", "lda", "qda"],
}

ALL_MODEL_NAMES = list(PYTORCH_MODELS.keys()) + list(SKLEARN_MODELS.keys())

# ── Training accuracy (segment-level) for weighted ensemble ──────────────────
MODEL_ACCURACY = {
    "lightgbm": 0.7540,
    "xgboost": 0.7535,
    "gbdt": 0.7440,
    "extra_trees": 0.7305,
    "random_forest": 0.7155,
    "svm_rbf": 0.7155,
    "mlp": 0.6785,
    "crnn_mel": 0.6670,
    "linear_svm": 0.6590,
    "logreg": 0.6580,
    "qda": 0.6560,
    "knn": 0.6435,
    "lda": 0.6345,
    "cnn_mel": 0.6320,
    "naive_bayes": 0.5065,
    "ada_boost": 0.4665,
    "lstm_mel": 0.2755,
}


def safe_float(x):
    """Convert to float, replacing NaN/Inf with 0.0."""
    if x is None:
        return 0.0
    try:
        v = float(x)
        if v != v or v == float("inf") or v == -float("inf"):
            return 0.0
        return v
    except (ValueError, TypeError):
        return 0.0

# Lazy-loaded caches
_sklearn_cache = {}
_pytorch_cache = {}


def load_sklearn(name):
    if name not in _sklearn_cache:
        path = SKLEARN_MODELS[name]
        obj = joblib.load(path)
        _sklearn_cache[name] = obj["model"]
    return _sklearn_cache[name]


def load_pytorch(name):
    if name not in _pytorch_cache:
        cfg = PYTORCH_MODELS[name]
        model = cfg["class"](**cfg["kwargs"])
        ckpt = torch.load(cfg["ckpt"], map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.to(DEVICE)
        model.eval()
        _pytorch_cache[name] = model
    return _pytorch_cache[name]


# ═══════════════════════════════════════════════════════════════════════════════
# Audio processing
# ═══════════════════════════════════════════════════════════════════════════════

def segment_audio(y, sr, seg_sec=SEGMENT_SECONDS):
    seg_len = int(round(seg_sec * sr))
    if seg_len <= 0:
        return []
    n = max(1, len(y) // seg_len)
    segs = []
    for i in range(n):
        start = i * seg_len
        end = start + seg_len
        if end > len(y):
            break
        segs.append(y[start:end])
    return segs


def extract_tabular_features(audio_path):
    """Extract 58-dim feature vector(s) matching the training CSV."""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    segs = segment_audio(y, sr)
    rows = []
    for seg in segs:
        stft = np.abs(librosa.stft(seg, n_fft=N_FFT, hop_length=HOP_LENGTH))
        row = {
            "length": SEGMENT_SECONDS,
            "chroma_stft_mean": np.mean(librosa.feature.chroma_stft(S=stft, sr=sr)),
            "chroma_stft_var": np.var(librosa.feature.chroma_stft(S=stft, sr=sr)),
            "rms_mean": np.mean(librosa.feature.rms(S=stft)),
            "rms_var": np.var(librosa.feature.rms(S=stft)),
            "spectral_centroid_mean": np.mean(librosa.feature.spectral_centroid(S=stft, sr=sr)),
            "spectral_centroid_var": np.var(librosa.feature.spectral_centroid(S=stft, sr=sr)),
            "spectral_bandwidth_mean": np.mean(librosa.feature.spectral_bandwidth(S=stft, sr=sr)),
            "spectral_bandwidth_var": np.var(librosa.feature.spectral_bandwidth(S=stft, sr=sr)),
            "rolloff_mean": np.mean(librosa.feature.spectral_rolloff(S=stft, sr=sr)),
            "rolloff_var": np.var(librosa.feature.spectral_rolloff(S=stft, sr=sr)),
            "zero_crossing_rate_mean": np.mean(librosa.feature.zero_crossing_rate(seg)),
            "zero_crossing_rate_var": np.var(librosa.feature.zero_crossing_rate(seg)),
        }
        harmony, perceptr = librosa.effects.hpss(seg)
        row["harmony_mean"] = np.mean(harmony)
        row["harmony_var"] = np.var(harmony)
        row["perceptr_mean"] = np.mean(perceptr)
        row["perceptr_var"] = np.var(perceptr)
        tempo_arr, _ = librosa.beat.beat_track(y=seg, sr=sr)
        row["tempo"] = float(tempo_arr.item()) if tempo_arr is not None and tempo_arr.size > 0 else 120.0
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(stft), sr=sr, n_mfcc=20)
        for i in range(1, 21):
            row[f"mfcc{i}_mean"] = np.mean(mfcc[i - 1])
            row[f"mfcc{i}_var"] = np.var(mfcc[i - 1])
        rows.append(row)
    if not rows:
        raise ValueError("Audio too short for any 3-second segment")
    return rows


def extract_mel_features(audio_path):
    """Extract mel spectrograms for CNN/LSTM/CRNN models."""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    segs = segment_audio(y, sr)
    mels = []
    for seg in segs:
        S = librosa.feature.melspectrogram(
            y=seg, sr=sr, n_mels=N_MELS, n_fft=N_FFT,
            hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX, power=2.0,
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        mels.append(S_db[None, :, :])
    if not mels:
        raise ValueError("Audio too short for any 3-second segment")
    return np.stack(mels, axis=0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Predictors
# ═══════════════════════════════════════════════════════════════════════════════

def predict_sklearn(model, features_df):
    """features_df: single-row DataFrame with 58 feature columns."""
    x = features_df.values.astype(np.float32)
    x_scaled = SCALER.transform(x)
    pred_idx = model.predict(x_scaled)[0]
    pred_genre = IDX_TO_GENRE[pred_idx]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_scaled)[0]
    else:
        proba = None
    return pred_genre, pred_idx, proba


def predict_pytorch_tabular(model, features_df):
    """MLP: tabular input."""
    x = features_df.values.astype(np.float32)
    x_scaled = SCALER.transform(x)
    x_t = torch.from_numpy(x_scaled).float().to(DEVICE)
    with torch.no_grad():
        logits = model(x_t)
        probs = torch.softmax(logits, dim=1)
    avg_probs = probs.mean(dim=0).cpu().numpy()
    pred_idx = int(avg_probs.argmax())
    return IDX_TO_GENRE[pred_idx], pred_idx, avg_probs


def predict_pytorch_mel(model, mel_spec, model_name):
    """CNN/LSTM/CRNN: mel spectrogram input."""
    x = torch.from_numpy(mel_spec).float()
    if model_name == "lstm_mel":
        x = x.squeeze(1).transpose(1, 2)
    x = x.to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
    avg_probs = probs.mean(dim=0).cpu().numpy()
    pred_idx = int(avg_probs.argmax())
    return IDX_TO_GENRE[pred_idx], pred_idx, avg_probs


# ═══════════════════════════════════════════════════════════════════════════════
# Conclusion
# ═══════════════════════════════════════════════════════════════════════════════

def compute_conclusion(results):
    valid = {n: r for n, r in results.items() if "error" not in r}

    best_by_acc_name = max(valid, key=lambda n: MODEL_ACCURACY.get(n, 0))
    best_by_acc = valid[best_by_acc_name]

    best_by_conf_name = max(
        valid,
        key=lambda n: (
            valid[n].get("confidence", 0)
            if isinstance(valid[n].get("confidence"), (int, float))
            else 0
        ),
    )
    best_by_conf = valid[best_by_conf_name]

    weighted_probs = np.zeros(N_CLASSES, dtype=np.float64)
    total_weight = 0.0
    for name, r in valid.items():
        if r.get("probabilities") is None:
            continue
        w = MODEL_ACCURACY.get(name, 0.0)
        if w <= 0:
            continue
        probs = r["probabilities"]
        for i, g in enumerate(GENRES):
            weighted_probs[i] += w * probs.get(g, 0.0)
        total_weight += w

    if total_weight > 0:
        weighted_probs /= total_weight
        ensemble_idx = int(np.argmax(weighted_probs))
        ensemble_genre = IDX_TO_GENRE[ensemble_idx]
        ensemble_conf = safe_float(weighted_probs[ensemble_idx])
    else:
        ensemble_genre = "N/A"
        ensemble_conf = 0.0

    matching_models = sorted(
        n for n, r in valid.items() if r.get("genre") == ensemble_genre
    )

    return {
        "final": {
            "genre": ensemble_genre,
            "confidence": ensemble_conf,
            "matching_models": matching_models,
            "total_models": len(valid),
        },
        "best_by_accuracy": {
            "model": best_by_acc_name,
            "genre": best_by_acc["genre"],
            "confidence": best_by_acc["confidence"],
            "accuracy": MODEL_ACCURACY.get(best_by_acc_name, 0),
        },
        "best_by_confidence": {
            "model": best_by_conf_name,
            "genre": best_by_conf["genre"],
            "confidence": best_by_conf["confidence"],
            "accuracy": MODEL_ACCURACY.get(best_by_conf_name, 0),
        },
        "weighted_ensemble": {
            "genre": ensemble_genre,
            "confidence": ensemble_conf,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Flask app
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024  # 30 MB


@app.route("/")
def index():
    return render_template("index.html", groups=MODEL_GROUPS, genres=GENRES)


@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"):
        return jsonify({"error": f"Unsupported format: {suffix}"}), 400

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp)
        tmp_path = tmp.name

    try:
        # ── Feature extraction ────────────────────────────────────────────
        tabular_rows = extract_tabular_features(tmp_path)
        mel_spec = extract_mel_features(tmp_path)

        import pandas as pd
        tabular_df = pd.DataFrame(tabular_rows)

        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

        # ── Inference ─────────────────────────────────────────────────────
        results = {}

        # Tabular MLP (PyTorch)
        mlp_model = load_pytorch("mlp")
        genre, idx, proba = predict_pytorch_tabular(mlp_model, tabular_df)
        results["mlp"] = {
            "genre": genre, "confidence": safe_float(proba[idx]),
            "probabilities": {GENRES[i]: safe_float(proba[i]) for i in range(N_CLASSES)},
        }

        # Mel models (PyTorch)
        for name in ("cnn_mel", "lstm_mel", "crnn_mel"):
            model = load_pytorch(name)
            genre, idx, proba = predict_pytorch_mel(model, mel_spec, name)
            results[name] = {
                "genre": genre, "confidence": safe_float(proba[idx]),
                "probabilities": {GENRES[i]: safe_float(proba[i]) for i in range(N_CLASSES)},
            }

        # Sklearn models (tabular)
        for name, ckpt_path in SKLEARN_MODELS.items():
            if not ckpt_path.exists():
                results[name] = {"genre": "N/A", "confidence": 0, "error": "checkpoint not found"}
                continue
            model = load_sklearn(name)
            genre, idx, proba = predict_sklearn(model, tabular_df)
            if proba is not None:
                results[name] = {
                    "genre": genre, "confidence": safe_float(proba[idx]),
                    "probabilities": {GENRES[i]: safe_float(proba[i]) for i in range(N_CLASSES)},
                }
            else:
                results[name] = {
                    "genre": genre, "confidence": 1.0,
                    "probabilities": None,
                }

        conclusion = compute_conclusion(results)
        return jsonify({"success": True, "results": results, "conclusion": conclusion})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 1234))
    app.run(host="0.0.0.0", port=port, debug=True)
