"""
train_models.py
===============
Trains 6 ML models — optimised for Render free tier (512 MB RAM, 0.1 CPU).
GridSearchCV kept with slim grids + cv=2 to stay within memory limits.
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
from sklearn.model_selection import GridSearchCV, cross_val_score
from xgboost import XGBClassifier


MODELS_CONFIG = [
    {
        "name": "Decision Tree",
        "model_class": DecisionTreeClassifier,
        "default_params": {"random_state": 42, "class_weight": "balanced", "max_depth": 10},
        "grid_params": {"max_depth": [8, 12]},
    },
    {
        "name": "Random Forest",
        "model_class": RandomForestClassifier,
        "default_params": {"random_state": 42, "class_weight": "balanced", "n_estimators": 50, "n_jobs": 1, "max_depth": 10},
        "grid_params": {"n_estimators": [30, 50]},
    },
    {
        "name": "Bagging Classifier",
        "model_class": BaggingClassifier,
        "default_params": {"n_estimators": 20, "random_state": 42, "n_jobs": 1},
        "grid_params": {"n_estimators": [15, 25]},
    },
    {
        "name": "AdaBoost",
        "model_class": AdaBoostClassifier,
        "default_params": {"n_estimators": 50, "learning_rate": 0.1, "random_state": 42},
        "grid_params": {"n_estimators": [30, 50]},
    },
    {
        "name": "Gradient Boosting",
        "model_class": GradientBoostingClassifier,
        "default_params": {"n_estimators": 50, "learning_rate": 0.1, "random_state": 42, "max_depth": 3, "subsample": 0.8},
        "grid_params": {"n_estimators": [30, 50]},
    },
    {
        "name": "XGBoost",
        "model_class": XGBClassifier,
        "default_params": {
            "n_estimators": 50, "learning_rate": 0.1,
            "use_label_encoder": False, "eval_metric": "mlogloss",
            "random_state": 42, "n_jobs": 1,
            "max_depth": 3, "subsample": 0.8, "tree_method": "hist",
        },
        "grid_params": {"n_estimators": [30, 50]},
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
        print(f"  → {name} trained ✓")
        gc.collect()
    return trained


def cross_validate_model(model_name: str, X_train, y_train, cv: int = 2) -> dict:
    cfg = next(c for c in MODELS_CONFIG if c["name"] == model_name)
    model = cfg["model_class"](**cfg["default_params"])
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=1)
    result = {
        "model":  model_name,
        "cv":     cv,
        "scores": scores.round(4).tolist(),
        "mean":   round(float(scores.mean()), 4),
        "std":    round(float(scores.std()), 4),
    }
    print(f"[train] CV {cv}-fold — {model_name}: mean={result['mean']:.4f} ± {result['std']:.4f}")
    gc.collect()
    return result


def grid_search_all(X_train, y_train, cv: int = 2) -> dict:
    """cv=2, n_jobs=1, slim grid — safe for 512 MB RAM."""
    best_results = {}
    for cfg in MODELS_CONFIG:
        name = cfg["name"]
        print(f"\n[train] GridSearchCV — {name} ...")
        model = cfg["model_class"](**cfg["default_params"].copy())
        gs = GridSearchCV(
            model, cfg["grid_params"],
            cv=cv, scoring="accuracy",
            n_jobs=1, verbose=0, refit=True,
        )
        gs.fit(X_train, y_train)
        best_results[name] = {
            "best_params": gs.best_params_,
            "best_score":  round(gs.best_score_, 4),
        }
        print(f"  Best params : {gs.best_params_}")
        print(f"  CV accuracy : {gs.best_score_:.4f}")
        del gs
        gc.collect()
    return best_results


def retrain_best_models(best_results: dict, X_train, y_train) -> dict:
    best_models = {}
    for cfg in MODELS_CONFIG:
        name = cfg["name"]
        best_p = best_results[name]["best_params"]
        final_p = {**cfg["default_params"].copy(), **best_p}
        print(f"[train] Retraining {name} ...")
        model = cfg["model_class"](**final_p)
        model.fit(X_train, y_train)
        best_models[name] = model
        print(f"  → {name} retrained ✓")
        gc.collect()
    return best_models