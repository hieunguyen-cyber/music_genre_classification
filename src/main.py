from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.dataset import make_dataloaders, make_mel_dataloaders
from src.data.feature_extraction import (
    build_tabular_cache,
    load_features_csv,
    load_mel_cache,
    load_tabular_cache,
    save_mel_cache,
)
from src.data.preprocessing import build_metadata
from src.data.split import load_split, make_group_splits, make_splits, save_split
from src.evaluation.confusion_matrix import plot_confusion_matrix
from src.evaluation.evaluate import evaluate_model
from src.models.model import create_model
from src.models.trainer import EarlyStoppingCfg, load_checkpoint, train
from src.utils.device import get_device
from src.utils.io import deep_merge, load_yaml, save_json
from src.utils.logger import setup_logger
from src.utils.paths import make_run_paths, project_root
from src.utils.seed import set_global_seed
from src.visualization.dataset_stats import plot_duration_distribution, plot_genre_distribution, plot_sample_rate_distribution
from src.visualization.embedding import plot_pca, plot_tsne, plot_umap
from src.visualization.feature_analysis import plot_class_distribution, plot_feature_correlation
from src.visualization.spectrogram import (
    plot_chromagram,
    plot_hpss,
    plot_mel_spectrogram,
    plot_mfcc,
    plot_spectral_contrast,
    plot_stft,
    plot_tempogram,
)
from src.visualization.training_curves import plot_training_curves
from src.visualization.waveform import plot_waveform
from src.visualization.error_analysis import build_misclassified_table
from src.visualization.curves import plot_multiclass_pr, plot_multiclass_roc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Music Genre Classification pipeline (GTZAN)")
    p.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    p.add_argument("--model-config", type=str, default="configs/model.yaml", help="Path to model.yaml")
    p.add_argument("--train-config", type=str, default="configs/train.yaml", help="Path to train.yaml")
    p.add_argument("--feature-config", type=str, default="configs/feature.yaml", help="Path to feature.yaml")
    p.add_argument("--stage", type=str, required=True, choices=["preprocess", "feature", "train", "evaluate", "visualize", "full"], help="Pipeline stage")
    p.add_argument("--run-name", type=str, default=None, help="Override outputs run name")
    p.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda", "mps"], help="Override device")
    p.add_argument("--seed", type=int, default=None, help="Override seed")
    p.add_argument("--epochs", type=int, default=None, help="Override training epochs (train stage)")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size (train/eval stages)")
    p.add_argument("--model", type=str, default=None, help="Override model name (e.g., mlp, knn, svm_rbf)")
    p.add_argument("--lr", type=float, default=None, help="Override learning rate (torch models)")
    p.add_argument("--weight-decay", type=float, default=None, help="Override weight decay (torch models)")
    p.add_argument("--num-workers", type=int, default=None, help="Override DataLoader num_workers")
    p.add_argument("--early-stopping", type=str, default=None, choices=["on", "off"], help="Enable/disable early stopping")
    p.add_argument("--patience", type=int, default=None, help="Early stopping patience (epochs)")
    p.add_argument("--monitor", type=str, default=None, choices=["val_loss", "val_accuracy"], help="Early stopping monitor")
    p.add_argument("--ckpt-metric", type=str, default=None, choices=["val_loss", "val_accuracy"], help="Checkpoint selection metric")
    p.add_argument("--ckpt-mode", type=str, default=None, choices=["min", "max"], help="Checkpoint mode for ckpt-metric")
    return p.parse_args()


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    root = project_root()
    cfg = load_yaml(root / args.config)
    cfg = deep_merge(cfg, {"model": {"_file": args.model_config}})
    # Merge extra yamls (kept separate so you can extend without touching config.yaml)
    cfg = deep_merge(cfg, {"_model_yaml": load_yaml(root / args.model_config)})
    cfg = deep_merge(cfg, {"_train_yaml": load_yaml(root / args.train_config)})
    cfg = deep_merge(cfg, {"_feature_yaml": load_yaml(root / args.feature_config)})

    if args.run_name is not None:
        cfg["paths"]["run_name"] = args.run_name
    if args.device is not None:
        cfg["project"]["device"] = args.device
    if args.seed is not None:
        cfg["project"]["seed"] = int(args.seed)
        cfg.setdefault("data", {}).setdefault("split", {})["random_state"] = int(args.seed)
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg.setdefault("train", {})["batch_size"] = int(args.batch_size)
        cfg.setdefault("eval", {})["batch_size"] = int(args.batch_size)
    if args.model is not None:
        cfg.setdefault("model", {})["name"] = str(args.model)
    if args.lr is not None:
        cfg.setdefault("train", {})["lr"] = float(args.lr)
    if args.weight_decay is not None:
        cfg.setdefault("train", {})["weight_decay"] = float(args.weight_decay)
    if args.num_workers is not None:
        cfg.setdefault("project", {})["num_workers"] = int(args.num_workers)
    if args.early_stopping is not None:
        cfg.setdefault("train", {}).setdefault("early_stopping", {})["enabled"] = args.early_stopping == "on"
    if args.patience is not None:
        cfg.setdefault("train", {}).setdefault("early_stopping", {})["patience"] = int(args.patience)
    if args.monitor is not None:
        cfg.setdefault("train", {}).setdefault("early_stopping", {})["monitor"] = str(args.monitor)
        # auto-set mode for common monitors unless user overrides ckpt-mode
        if args.ckpt_mode is None and str(args.monitor) == "val_accuracy":
            cfg.setdefault("train", {}).setdefault("early_stopping", {})["mode"] = "max"
    if args.ckpt_metric is not None:
        cfg.setdefault("train", {}).setdefault("checkpoint", {})["metric"] = str(args.ckpt_metric)
    if args.ckpt_mode is not None:
        cfg.setdefault("train", {}).setdefault("checkpoint", {})["mode"] = str(args.ckpt_mode)

    # Resolve simple references (keep config lightweight; no Hydra/OmegaConf dependency)
    split_cfg = cfg.setdefault("data", {}).setdefault("split", {})
    if not isinstance(split_cfg.get("random_state"), int):
        split_cfg["random_state"] = int(cfg["project"]["seed"])
    return cfg


def stage_preprocess(cfg: Dict[str, Any], run_paths) -> None:
    raw_root = project_root() / cfg["paths"]["raw_root"]
    raw_genres = raw_root / "genres_original"
    out_csv = project_root() / cfg["paths"]["processed_dir"] / "metadata.csv"
    logger = setup_logger("preprocess", run_paths.logs_dir / "preprocess.log")
    logger.info(f"raw_genres_dir={raw_genres}")
    logger.info(f"out_csv={out_csv}")
    df = build_metadata(raw_genres, out_csv)
    logger.info(f"metadata_rows={len(df)}")


def stage_feature(cfg: Dict[str, Any], run_paths) -> None:
    logger = setup_logger("feature", run_paths.logs_dir / "feature.log")
    root = project_root()
    features_csv = root / cfg["paths"]["features_csv"]
    kind = cfg["features"]["kind"]
    split_cfg = cfg["data"]["split"]
    out_dir = root / cfg["paths"]["features_dir"]

    if kind == "tabular_csv":
        df = load_features_csv(features_csv)
        feature_yaml = cfg["_feature_yaml"]["tabular_csv"]
        split_path = root / cfg["paths"]["splits_dir"] / feature_yaml["split_name"]

        if split_path.exists():
            logger.info(f"Loading existing split: {split_path}")
            split_df = load_split(split_path)
        else:
            logger.info("Creating new split (stratified)")
            split_res = make_splits(
                df,
                label_column=cfg["features"]["tabular_csv"]["label_column"],
                test_size=float(split_cfg["test_size"]),
                val_size=float(split_cfg["val_size"]),
                random_state=int(split_cfg["random_state"]),
                stratify=bool(split_cfg["stratify"]),
            )
            split_df = split_res.split_df
            save_split(split_df, split_path)
            logger.info(f"Saved split to: {split_path}")

        cache_path = out_dir / feature_yaml["cache_name"]
        scaler_path = out_dir / feature_yaml["scaler_name"]
        label_map_path = out_dir / feature_yaml["label_map_name"]
        if cache_path.exists():
            logger.info(f"Feature cache exists: {cache_path}")
            return

        logger.info(f"Building feature cache: {cache_path}")
        build_tabular_cache(
            df,
            split_df,
            label_column=cfg["features"]["tabular_csv"]["label_column"],
            filename_column=cfg["features"]["tabular_csv"]["filename_column"],
            drop_columns=cfg["features"]["tabular_csv"]["drop_columns"],
            standardize=bool(cfg["features"]["tabular_csv"]["standardize"]),
            out_npz=cache_path,
            out_scaler=scaler_path,
            out_label_map=label_map_path,
        )
        logger.info("Done")
        return

    if kind == "mel_from_audio":
        from src.data.audio_features import MelCfg, extract_mel_dataset
        from src.data.augmentation import AugmentCfg

        feature_yaml = cfg["_feature_yaml"]["mel_from_audio"]
        cache_path = out_dir / feature_yaml["cache_name"]
        split_path = root / cfg["paths"]["splits_dir"] / feature_yaml["split_name"]
        if cache_path.exists() and split_path.exists():
            logger.info(f"Mel cache exists: {cache_path}")
            return

        raw_root = root / cfg["paths"]["raw_root"]
        raw_genres = raw_root / "genres_original"
        mel_cfg = cfg["features"]["mel_from_audio"]
        X, y, groups, classes = extract_mel_dataset(
            raw_genres,
            cfg=MelCfg(
                sample_rate=int(mel_cfg["sample_rate"]),
                segment_seconds=float(mel_cfg["segment_seconds"]),
                n_mels=int(mel_cfg["n_mels"]),
                n_fft=int(mel_cfg["n_fft"]),
                hop_length=int(mel_cfg["hop_length"]),
                fmin=int(mel_cfg["fmin"]),
                fmax=int(mel_cfg["fmax"]),
            ),
            augment=AugmentCfg(enabled=False),
            seed=int(cfg["project"]["seed"]),
        )

        # group split by track
        split_by_group = make_group_splits(
            groups=groups.astype(str),
            labels=y,
            test_size=float(split_cfg["test_size"]),
            val_size=float(split_cfg["val_size"]),
            random_state=int(split_cfg["random_state"]),
            stratify=bool(split_cfg["stratify"]),
        )
        # Save split mapping (track -> split)
        split_df = pd.DataFrame({"group": list(split_by_group.keys()), "split": list(split_by_group.values())})
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_df.to_csv(split_path, index=False)

        save_mel_cache(
            x=X,
            y=y,
            groups=groups,
            split_by_group=split_by_group,
            classes=classes,
            out_npz=cache_path,
        )
        logger.info(f"Saved mel cache: {cache_path}")
        logger.info(f"Saved mel split: {split_path}")
        return

    raise ValueError(f"Unknown features.kind: {kind}")


def stage_train(cfg: Dict[str, Any], run_paths) -> None:
    logger = setup_logger("train", run_paths.logs_dir / "train.log")
    root = project_root()
    kind = cfg["features"]["kind"]

    set_global_seed(int(cfg["project"]["seed"]))
    device_info = get_device(cfg["project"]["device"])
    logger.info(f"device={device_info.name}")

    model_name = cfg["model"]["name"]
    model_cfg = cfg["_model_yaml"][model_name]
    from src.models.sklearn_models import SKLEARN_MODEL_NAMES
    if model_name in SKLEARN_MODEL_NAMES:
        from src.models.sklearn_models import create_sklearn_model
        from src.models.sklearn_trainer import fit_and_save

        feature_yaml = cfg["_feature_yaml"]["tabular_csv"]
        cache_path = root / cfg["paths"]["features_dir"] / feature_yaml["cache_name"]
        cache = load_tabular_cache(cache_path)
        logger.info(f"model={model_name} (sklearn) input_dim={cache.x_train.shape[1]} classes={len(cache.classes)}")
        m = create_sklearn_model(model_name, model_cfg)
        res = fit_and_save(
            m,
            x_train=cache.x_train,
            y_train=cache.y_train,
            x_val=cache.x_val,
            y_val=cache.y_val,
            run_dir=run_paths.run_dir,
            model_name=model_name,
        )
        logger.info(f"checkpoint={res.checkpoint_path} val_accuracy={res.val_accuracy:.4f} val_log_loss={res.val_log_loss}")
        return

    if kind == "tabular_csv":
        feature_yaml = cfg["_feature_yaml"]["tabular_csv"]
        cache_path = root / cfg["paths"]["features_dir"] / feature_yaml["cache_name"]
        cache = load_tabular_cache(cache_path)
        loaders = make_dataloaders(
            x_train=cache.x_train,
            y_train=cache.y_train,
            x_val=cache.x_val,
            y_val=cache.y_val,
            x_test=cache.x_test,
            y_test=cache.y_test,
            batch_size=int(cfg["train"]["batch_size"]),
            num_workers=int(cfg["project"]["num_workers"]),
            seed=int(cfg["project"]["seed"]),
        )
        model = create_model(model_name, model_cfg, input_dim=int(cache.x_train.shape[1]), n_classes=len(cache.classes))
        logger.info(f"model={model_name} (torch) input_dim={cache.x_train.shape[1]} classes={len(cache.classes)}")
        n_mels = None
    elif kind == "mel_from_audio":
        feature_yaml = cfg["_feature_yaml"]["mel_from_audio"]
        cache_path = root / cfg["paths"]["features_dir"] / feature_yaml["cache_name"]
        cache = load_mel_cache(cache_path)
        as_sequence = model_name == "lstm_mel"
        loaders = make_mel_dataloaders(
            x_train=cache.x_train,
            y_train=cache.y_train,
            x_val=cache.x_val,
            y_val=cache.y_val,
            x_test=cache.x_test,
            y_test=cache.y_test,
            batch_size=int(cfg["train"]["batch_size"]),
            num_workers=int(cfg["project"]["num_workers"]),
            seed=int(cfg["project"]["seed"]),
            as_sequence=as_sequence,
        )
        mel_cfg = cfg["features"]["mel_from_audio"]
        model = create_model(
            model_name,
            model_cfg,
            input_dim=1,
            n_classes=len(cache.classes),
            n_mels=int(mel_cfg["n_mels"]),
        )
        logger.info(f"model={model_name} (torch, mel) classes={len(cache.classes)}")
    else:
        raise ValueError(f"Unknown features.kind: {kind}")

    es_cfg = cfg["train"]["early_stopping"]
    _, _, best_path = train(
        model=model,
        train_loader=loaders.train,
        val_loader=loaders.val,
        device=device_info.device,
        epochs=int(cfg["train"]["epochs"]),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        run_dir=run_paths.run_dir,
        early_stopping=EarlyStoppingCfg(
            enabled=bool(es_cfg["enabled"]),
            patience=int(es_cfg["patience"]),
            monitor=str(es_cfg["monitor"]),
            mode=str(es_cfg["mode"]),
        ),
        checkpoint_metric=str(cfg["train"]["checkpoint"]["metric"]),
        checkpoint_mode=str(cfg["train"]["checkpoint"]["mode"]),
    )
    logger.info(f"best_checkpoint={best_path}")

    hist_json = run_paths.reports_dir / "history.json"
    curves_path = run_paths.figures_dir / "training_curves.png"
    if hist_json.exists():
        plot_training_curves(hist_json, curves_path)
        logger.info(f"Saved curves: {curves_path}")


def stage_evaluate(cfg: Dict[str, Any], run_paths) -> None:
    logger = setup_logger("evaluate", run_paths.logs_dir / "evaluate.log")
    root = project_root()
    kind = cfg["features"]["kind"]
    if kind == "tabular_csv":
        feature_yaml = cfg["_feature_yaml"]["tabular_csv"]
        cache_path = root / cfg["paths"]["features_dir"] / feature_yaml["cache_name"]
        cache = load_tabular_cache(cache_path)
        loaders = make_dataloaders(
            x_train=cache.x_train,
            y_train=cache.y_train,
            x_val=cache.x_val,
            y_val=cache.y_val,
            x_test=cache.x_test,
            y_test=cache.y_test,
            batch_size=int(cfg["eval"]["batch_size"]),
            num_workers=int(cfg["project"]["num_workers"]),
            seed=int(cfg["project"]["seed"]),
        )
    elif kind == "mel_from_audio":
        feature_yaml = cfg["_feature_yaml"]["mel_from_audio"]
        cache_path = root / cfg["paths"]["features_dir"] / feature_yaml["cache_name"]
        cache = load_mel_cache(cache_path)
        model_name = cfg["model"]["name"]
        as_sequence = model_name == "lstm_mel"
        loaders = make_mel_dataloaders(
            x_train=cache.x_train,
            y_train=cache.y_train,
            x_val=cache.x_val,
            y_val=cache.y_val,
            x_test=cache.x_test,
            y_test=cache.y_test,
            batch_size=int(cfg["eval"]["batch_size"]),
            num_workers=int(cfg["project"]["num_workers"]),
            seed=int(cfg["project"]["seed"]),
            as_sequence=as_sequence,
        )
    else:
        raise ValueError(f"Unknown features.kind: {kind}")

    device_info = get_device(cfg["project"]["device"])
    logger.info(f"device={device_info.name}")

    model_name = cfg["model"]["name"]
    model_cfg = cfg["_model_yaml"][model_name]
    from src.models.sklearn_models import SKLEARN_MODEL_NAMES
    if model_name in SKLEARN_MODEL_NAMES:
        from src.models.sklearn_trainer import load_sklearn_checkpoint
        from sklearn.metrics import accuracy_score

        ckpt_path = run_paths.checkpoints_dir / "model.joblib"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run --stage train first.")
        _, sk_model = load_sklearn_checkpoint(ckpt_path)
        y_true = cache.y_test  # type: ignore[attr-defined]
        y_pred = sk_model.predict(cache.x_test)  # type: ignore[attr-defined]
        y_proba = None
        if hasattr(sk_model, "predict_proba"):
            try:
                y_proba = sk_model.predict_proba(cache.x_test)  # type: ignore[attr-defined]
            except Exception:
                y_proba = None

        from src.evaluation.metrics import compute_metrics
        res = compute_metrics(y_true, y_pred, class_names=cache.classes, y_proba=y_proba)  # type: ignore[arg-type]
        preds = type("Preds", (), {"y_true": y_true, "y_pred": y_pred, "y_proba": y_proba})
        logger.info(f"accuracy={res.accuracy:.4f} macro_f1={res.macro_f1:.4f} weighted_f1={res.weighted_f1:.4f} roc_auc_ovr_macro={res.roc_auc_ovr_macro}")
    else:
        if kind == "tabular_csv":
            model = create_model(model_name, model_cfg, input_dim=int(cache.x_train.shape[1]), n_classes=len(cache.classes))  # type: ignore[arg-type]
            ckpt_path = run_paths.checkpoints_dir / "best.pt"
        else:
            mel_cfg = cfg["features"]["mel_from_audio"]
            # input_dim is unused for cnn/lstm, but factory requires it; pass a safe value
            model = create_model(
                model_name,
                model_cfg,
                input_dim=1,
                n_classes=len(cache.classes),  # type: ignore[arg-type]
                n_mels=int(mel_cfg["n_mels"]),
            )
            ckpt_path = run_paths.checkpoints_dir / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run --stage train first.")
        load_checkpoint(ckpt_path, model)
        model.to(device_info.device)
        preds, res = evaluate_model(model, loaders.test, device_info.device, class_names=cache.classes)
        logger.info(f"accuracy={res.accuracy:.4f} macro_f1={res.macro_f1:.4f} weighted_f1={res.weighted_f1:.4f} roc_auc_ovr_macro={res.roc_auc_ovr_macro}")

    # Save report
    report_path = run_paths.reports_dir / "evaluation_report.md"
    report_text = (
        "# Evaluation Report\n\n"
        f"- Run: `{run_paths.run_dir.name}`\n"
        f"- Timestamp: {datetime.now().isoformat(timespec='seconds')}\n"
        f"- Device: `{device_info.name}`\n\n"
        "## Summary\n\n"
        f"- Accuracy: **{res.accuracy:.4f}**\n"
        f"- Macro F1: **{res.macro_f1:.4f}**\n"
        f"- Weighted F1: **{res.weighted_f1:.4f}**\n"
        + (f"- ROC-AUC (OvR, macro): **{res.roc_auc_ovr_macro:.4f}**\n" if res.roc_auc_ovr_macro is not None else "")
        + "\n## Classification Report\n\n```\n"
        + res.report
        + "\n```\n"
    )
    report_path.write_text(report_text, encoding="utf-8")
    logger.info(f"Saved report: {report_path}")

    cm_path = run_paths.figures_dir / "confusion_matrix.png"
    plot_confusion_matrix(res.confusion, cache.classes, cm_path, normalize=True)
    logger.info(f"Saved confusion matrix: {cm_path}")

    mis_csv = run_paths.reports_dir / "misclassified.csv"
    if hasattr(cache, "filenames_test"):
        build_misclassified_table(cache.filenames_test, preds.y_true, preds.y_pred, cache.classes, mis_csv)
        logger.info(f"Saved misclassified table: {mis_csv}")
    elif hasattr(cache, "groups_test"):
        # For mel_from_audio cache: store track group id instead of filename.
        build_misclassified_table(cache.groups_test, preds.y_true, preds.y_pred, cache.classes, mis_csv)
        logger.info(f"Saved misclassified table (groups): {mis_csv}")
    else:
        logger.warning("Misclassified table skipped (no identifiers available).")

    if getattr(preds, "y_proba", None) is not None:
        try:
            plot_multiclass_roc(preds.y_true, preds.y_proba, cache.classes, run_paths.figures_dir / "roc_ovr.png")
            plot_multiclass_pr(preds.y_true, preds.y_proba, cache.classes, run_paths.figures_dir / "pr_ovr.png")
            logger.info("Saved ROC/PR curves")
        except Exception as e:
            logger.warning(f"ROC/PR skipped: {e}")


def stage_visualize(cfg: Dict[str, Any], run_paths) -> None:
    logger = setup_logger("visualize", run_paths.logs_dir / "visualize.log")
    root = project_root()

    meta_csv = root / cfg["paths"]["processed_dir"] / "metadata.csv"
    if meta_csv.exists():
        plot_genre_distribution(meta_csv, run_paths.figures_dir / "raw_genre_distribution.png")
        plot_duration_distribution(meta_csv, run_paths.figures_dir / "raw_duration_distribution.png")
        plot_sample_rate_distribution(meta_csv, run_paths.figures_dir / "raw_sample_rate_distribution.png")
        logger.info("Saved raw dataset stats")
    else:
        logger.warning(f"Metadata not found: {meta_csv} (run --stage preprocess)")

    kind = cfg["features"]["kind"]
    emb_cfg = cfg.get("visualize", {}).get("embedding", {})

    if kind == "tabular_csv":
        feature_yaml = cfg["_feature_yaml"]["tabular_csv"]
        cache_path = root / cfg["paths"]["features_dir"] / feature_yaml["cache_name"]
        if not cache_path.exists():
            logger.warning(f"Feature cache not found: {cache_path} (run --stage feature)")
            return

        cache = load_tabular_cache(cache_path)
        x_all = np.concatenate([cache.x_train, cache.x_val, cache.x_test], axis=0)
        y_all = np.concatenate([cache.y_train, cache.y_val, cache.y_test], axis=0)

        plot_class_distribution(
            y_all,
            cache.classes,
            run_paths.figures_dir / "class_distribution_segments.png",
            title="Class distribution (3-sec segments)",
        )

        df = load_features_csv(root / cfg["paths"]["features_csv"])
        feat_names = [c for c in df.columns if c not in cfg["features"]["tabular_csv"]["drop_columns"]]
        plot_feature_correlation(x_all, feat_names, run_paths.figures_dir / "feature_correlation.png")

        if emb_cfg.get("pca", True):
            plot_pca(x_all, y_all, cache.classes, run_paths.figures_dir / "embedding_pca.png")
        if emb_cfg.get("tsne", True):
            plot_tsne(x_all, y_all, cache.classes, run_paths.figures_dir / "embedding_tsne.png")
        if emb_cfg.get("umap", True):
            try:
                plot_umap(x_all, y_all, cache.classes, run_paths.figures_dir / "embedding_umap.png")
            except Exception as e:
                logger.warning(f"UMAP skipped: {e}")
    elif kind == "mel_from_audio":
        feature_yaml = cfg["_feature_yaml"]["mel_from_audio"]
        cache_path = root / cfg["paths"]["features_dir"] / feature_yaml["cache_name"]
        if not cache_path.exists():
            logger.warning(f"Mel cache not found: {cache_path} (run --stage feature)")
            return
        cache = load_mel_cache(cache_path)
        x_all = np.concatenate([cache.x_train, cache.x_val, cache.x_test], axis=0)
        y_all = np.concatenate([cache.y_train, cache.y_val, cache.y_test], axis=0)

        plot_class_distribution(
            y_all,
            cache.classes,
            run_paths.figures_dir / "class_distribution_segments.png",
            title="Class distribution (mel segments)",
        )

        # Flatten mel to 2D for embeddings
        x_flat = x_all.reshape(x_all.shape[0], -1)
        if emb_cfg.get("pca", True):
            plot_pca(x_flat, y_all, cache.classes, run_paths.figures_dir / "embedding_pca.png")
        if emb_cfg.get("tsne", True):
            plot_tsne(x_flat, y_all, cache.classes, run_paths.figures_dir / "embedding_tsne.png")
        if emb_cfg.get("umap", True):
            try:
                plot_umap(x_flat, y_all, cache.classes, run_paths.figures_dir / "embedding_umap.png")
            except Exception as e:
                logger.warning(f"UMAP skipped: {e}")
    else:
        logger.warning(f"Unknown features.kind={kind} for visualization")
        return

    raw_root = root / cfg["paths"]["raw_root"]
    raw_genres = raw_root / "genres_original"
    if not raw_genres.exists():
        logger.warning(f"Raw genres dir not found: {raw_genres}")
        return

    audio_files = sorted(raw_genres.rglob("*.wav"))
    max_n = int(cfg.get("visualize", {}).get("max_audio_files", 20))
    rng = np.random.default_rng(int(cfg["project"]["seed"]))
    pick = (
        audio_files
        if len(audio_files) <= max_n
        else [audio_files[i] for i in rng.choice(len(audio_files), size=max_n, replace=False)]
    )
    for p in pick:
        stem = f"{p.parent.name}_{p.stem}"
        plot_waveform(p, run_paths.figures_dir / f"waveform_{stem}.png")
        plot_stft(p, run_paths.figures_dir / f"stft_{stem}.png")
        plot_mel_spectrogram(p, run_paths.figures_dir / f"mel_{stem}.png")
        plot_mfcc(p, run_paths.figures_dir / f"mfcc_{stem}.png")
        plot_chromagram(p, run_paths.figures_dir / f"chroma_{stem}.png")
        plot_spectral_contrast(p, run_paths.figures_dir / f"contrast_{stem}.png")
        plot_tempogram(p, run_paths.figures_dir / f"tempogram_{stem}.png")
        plot_hpss(p, run_paths.figures_dir / f"hpss_{stem}.png")
    logger.info(f"Saved audio visualizations for {len(pick)} files")


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    root = project_root()

    outputs_root = root / cfg["paths"]["outputs_root"]
    run_name = cfg["paths"]["run_name"]
    run_paths = make_run_paths(outputs_root, run_name)

    (run_paths.reports_dir / "config_snapshot.json").write_text(
        json.dumps(cfg, indent=2, default=str),
        encoding="utf-8",
    )

    set_global_seed(int(cfg["project"]["seed"]))

    if args.stage == "preprocess":
        stage_preprocess(cfg, run_paths)
    elif args.stage == "feature":
        stage_feature(cfg, run_paths)
    elif args.stage == "train":
        stage_train(cfg, run_paths)
    elif args.stage == "evaluate":
        stage_evaluate(cfg, run_paths)
    elif args.stage == "visualize":
        stage_visualize(cfg, run_paths)
    elif args.stage == "full":
        stage_preprocess(cfg, run_paths)
        stage_feature(cfg, run_paths)
        stage_train(cfg, run_paths)
        stage_evaluate(cfg, run_paths)
        stage_visualize(cfg, run_paths)
    else:
        raise ValueError(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    main()
