"""
train_models.py
===============
Optimised for Render free tier (512 MB RAM, 0.1 CPU).
Strategy: NO GridSearchCV — train each model ONCE with pre-tuned params.
GridSearchCV kept as a wrapper for API compatibility but does zero grid search.
Total fits: 6 (one per model). Training completes in ~60-90 seconds on free tier.
"""

import warnings, gc
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

# ── Pre-tuned best params (determined offline) ────────────────────────────────
# These are the best params for this dataset — no grid search needed.
MODELS_CONFIG = [
    {
        "name": "Decision Tree",
        "model_class": DecisionTreeClassifier,
        "default_params": {
            "random_state": 42, "class_weight": "balanced",
            "max_depth": 10, "min_samples_split": 5,
        },
        "grid_params": {},  # empty — no grid search
    },
    {
        "name": "Random Forest",
        "model_class": RandomForestClassifier,
        "default_params": {
            "random_state": 42, "class_weight": "balanced",
            "n_estimators": 50, "max_depth": 10, "n_jobs": 1,
        },
        "grid_params": {},
    },
    {
        "name": "Bagging Classifier",
        "model_class": BaggingClassifier,
        "default_params": {
            "n_estimators": 20, "random_state": 42, "n_jobs": 1,
        },
        "grid_params": {},
    },
    {
        "name": "AdaBoost",
        "model_class": AdaBoostClassifier,
        "default_params": {
            "n_estimators": 50, "learning_rate": 0.1, "random_state": 42,
        },
        "grid_params": {},
    },
    {
        "name": "Gradient Boosting",
        "model_class": GradientBoostingClassifier,
        "default_params": {
            "n_estimators": 50, "learning_rate": 0.1,
            "max_depth": 3, "subsample": 0.8, "random_state": 42,
        },
        "grid_params": {},
    },
    {
        "name": "XGBoost",
        "model_class": XGBClassifier,
        "default_params": {
            "n_estimators": 50, "learning_rate": 0.1,
            "max_depth": 3, "subsample": 0.8,
            "use_label_encoder": False, "eval_metric": "mlogloss",
            "random_state": 42, "n_jobs": 1, "tree_method": "hist",
        },
        "grid_params": {},
    },
]


def train_default_models(X_train, y_train) -> dict:
    trained = {}
    for cfg in MODELS_CONFIG:
        name = cfg["name"]
        print(f"[train] Training {name} ...")
        model = cfg["model_class"](**cfg["default_params"])
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"  → {name} done ✓")
        gc.collect()
    return trained


def cross_validate_model(model_name: str, X_train, y_train, cv: int = 2) -> dict:
    cfg = next(c for c in MODELS_CONFIG if c["name"] == model_name)
    model = cfg["model_class"](**cfg["default_params"])
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=1)
    result = {
        "model": model_name, "cv": cv,
        "scores": scores.round(4).tolist(),
        "mean": round(float(scores.mean()), 4),
        "std":  round(float(scores.std()),  4),
    }
    print(f"[train] CV {cv}-fold {model_name}: {result['mean']:.4f} ± {result['std']:.4f}")
    gc.collect()
    return result


def grid_search_all(X_train, y_train, cv: int = 2) -> dict:
    """
    No actual grid search — just trains each model once with pre-tuned params.
    Kept for API compatibility with app.py and main.py.
    ~60-90 seconds total on Render free tier.
    """
    best_results = {}
    for cfg in MODELS_CONFIG:
        name = cfg["name"]
        print(f"[train] Training {name} (pre-tuned params)...")
        model = cfg["model_class"](**cfg["default_params"])
        model.fit(X_train, y_train)
        best_results[name] = {
            "best_params": {},
            "best_score":  0.0,
            "_trained_model": model,  # cache so retrain doesn't fit again
        }
        print(f"  → {name} done ✓")
        gc.collect()
    return best_results


def retrain_best_models(best_results: dict, X_train, y_train) -> dict:
    """Returns already-trained models from grid_search_all (no re-fitting)."""
    best_models = {}
    for cfg in MODELS_CONFIG:
        name = cfg["name"]
        if "_trained_model" in best_results[name]:
            # Reuse cached model — zero extra fitting!
            best_models[name] = best_results[name]["_trained_model"]
            print(f"[train] {name} — reusing trained model ✓")
        else:
            model = cfg["model_class"](**cfg["default_params"])
            model.fit(X_train, y_train)
            best_models[name] = model
            print(f"[train] {name} — retrained ✓")
        gc.collect()
    return best_models