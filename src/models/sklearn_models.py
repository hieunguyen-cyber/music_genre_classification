from __future__ import annotations

from typing import Any, Dict

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC


SKLEARN_MODEL_NAMES = {
    "knn",
    "svm_rbf",
    "linear_svm",
    "logreg",
    "naive_bayes",
    "lda",
    "qda",
    "random_forest",
    "extra_trees",
    "ada_boost",
    "gbdt",
    "xgboost",
    "lightgbm",
}


def create_sklearn_model(model_name: str, model_cfg: Dict[str, Any]):
    """
    Factory for sklearn baselines.

    Notes:
      - `linear_svm` uses `LinearSVC` (no predict_proba).
      - `svm_rbf` uses `SVC(probability=True)` for ROC/PR curves (slower but useful).
    """
    if model_name == "knn":
        return KNeighborsClassifier(
            n_neighbors=int(model_cfg.get("n_neighbors", 3)),
            weights=str(model_cfg.get("weights", "uniform")),
        )
    if model_name == "svm_rbf":
        return SVC(
            kernel="rbf",
            C=float(model_cfg.get("C", 10.0)),
            gamma=model_cfg.get("gamma", "scale"),
            probability=True,
        )
    if model_name == "linear_svm":
        return LinearSVC(C=float(model_cfg.get("C", 1.0)))
    if model_name == "logreg":
        return LogisticRegression(
            C=float(model_cfg.get("C", 1.0)),
            max_iter=int(model_cfg.get("max_iter", 2000)),
            n_jobs=-1,
            multi_class="auto",
        )
    if model_name == "naive_bayes":
        return GaussianNB()
    if model_name == "lda":
        return LinearDiscriminantAnalysis(solver=str(model_cfg.get("solver", "svd")))
    if model_name == "qda":
        return QuadraticDiscriminantAnalysis(reg_param=float(model_cfg.get("reg_param", 0.0)))
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(model_cfg.get("n_estimators", 500)),
            max_depth=model_cfg.get("max_depth", None),
            min_samples_split=int(model_cfg.get("min_samples_split", 2)),
            min_samples_leaf=int(model_cfg.get("min_samples_leaf", 1)),
            n_jobs=-1,
            random_state=int(model_cfg.get("random_state", 42)),
        )
    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=int(model_cfg.get("n_estimators", 800)),
            max_depth=model_cfg.get("max_depth", None),
            n_jobs=-1,
            random_state=int(model_cfg.get("random_state", 42)),
        )
    if model_name == "ada_boost":
        return AdaBoostClassifier(
            n_estimators=int(model_cfg.get("n_estimators", 400)),
            learning_rate=float(model_cfg.get("learning_rate", 0.05)),
            random_state=int(model_cfg.get("random_state", 42)),
        )
    if model_name == "gbdt":
        return GradientBoostingClassifier(
            n_estimators=int(model_cfg.get("n_estimators", 500)),
            learning_rate=float(model_cfg.get("learning_rate", 0.05)),
            max_depth=int(model_cfg.get("max_depth", 3)),
            random_state=int(model_cfg.get("random_state", 42)),
        )
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Model 'xgboost' requires `xgboost`. Install it to use this model.") from e
        return XGBClassifier(
            n_estimators=int(model_cfg.get("n_estimators", 800)),
            learning_rate=float(model_cfg.get("learning_rate", 0.05)),
            max_depth=int(model_cfg.get("max_depth", 6)),
            subsample=float(model_cfg.get("subsample", 0.8)),
            colsample_bytree=float(model_cfg.get("colsample_bytree", 0.8)),
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=int(model_cfg.get("num_class", 10)),
            n_jobs=-1,
            random_state=int(model_cfg.get("random_state", 42)),
        )
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Model 'lightgbm' requires `lightgbm`. Install it to use this model.") from e
        return LGBMClassifier(
            n_estimators=int(model_cfg.get("n_estimators", 1200)),
            learning_rate=float(model_cfg.get("learning_rate", 0.05)),
            num_leaves=int(model_cfg.get("num_leaves", 63)),
            subsample=float(model_cfg.get("subsample", 0.8)),
            colsample_bytree=float(model_cfg.get("colsample_bytree", 0.8)),
            objective="multiclass",
            n_jobs=-1,
            random_state=int(model_cfg.get("random_state", 42)),
        )
    raise ValueError(f"Unknown sklearn model name: {model_name}")
