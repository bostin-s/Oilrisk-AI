"""
train_models.py
===============
Trains 6 machine learning models using the preprocessed dataset:

  Bagging family:
    1. Decision Tree (base/single model)
    2. Random Forest
    3. BaggingClassifier

  Boosting family:
    4. AdaBoost
    5. Gradient Boosting
    6. XGBoost
"""

import warnings
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


# ── Model registry ────────────────────────────────────────────────────────────

MODELS_CONFIG = [
    {
        "name": "Decision Tree",
        "model_class": DecisionTreeClassifier,
        "default_params": {"random_state": 42},
        "grid_params": {
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
        },
    },
    {
        "name": "Random Forest",
        "model_class": RandomForestClassifier,
        "default_params": {"n_estimators": 100, "random_state": 42},
        "grid_params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, None],
        },
    },
    {
        "name": "Bagging Classifier",
        "model_class": BaggingClassifier,
        "default_params": {"n_estimators": 50, "random_state": 42},
        "grid_params": {
            "n_estimators": [10, 50, 100],
        },
    },
    {
        "name": "AdaBoost",
        "model_class": AdaBoostClassifier,
        "default_params": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42},
        "grid_params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 1.0],
        },
    },
    {
        "name": "Gradient Boosting",
        "model_class": GradientBoostingClassifier,
        "default_params": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42},
        "grid_params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 1.0],
        },
    },
    {
        "name": "XGBoost",
        "model_class": XGBClassifier,
        "default_params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "use_label_encoder": False,
            "eval_metric": "mlogloss",
            "random_state": 42,
        },
        "grid_params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 1.0],
        },
    },
]


def train_default_models(X_train, y_train) -> dict:
    trained = {}
    for cfg in MODELS_CONFIG:
        name = cfg["name"]
        print(f"[train] Training {name} (default params) ...")
        model = cfg["model_class"](**cfg["default_params"])
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"  → {name} trained ✓")
    return trained


def cross_validate_model(model_name: str, X_train, y_train, cv: int = 5) -> dict:
    cfg = next(c for c in MODELS_CONFIG if c["name"] == model_name)
    model = cfg["model_class"](**cfg["default_params"])
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    result = {
        "model":  model_name,
        "cv":     cv,
        "scores": scores.round(4).tolist(),
        "mean":   round(float(scores.mean()), 4),
        "std":    round(float(scores.std()), 4),
    }
    print(f"[train] CV {cv}-fold — {model_name}: mean={result['mean']:.4f} ± {result['std']:.4f}")
    return result


def grid_search_all(X_train, y_train, cv: int = 3) -> dict:
    best_results = {}
    for cfg in MODELS_CONFIG:
        name = cfg["name"]
        print(f"\n[train] GridSearchCV — {name} ...")
        base_params = cfg["default_params"].copy()
        model = cfg["model_class"](**base_params)
        gs = GridSearchCV(
            model, cfg["grid_params"], cv=cv, scoring="accuracy",
            n_jobs=-1, verbose=0, refit=True,
        )
        gs.fit(X_train, y_train)
        best_results[name] = {
            "best_params": gs.best_params_,
            "best_score":  round(gs.best_score_, 4),
        }
        print(f"  Best params : {gs.best_params_}")
        print(f"  CV accuracy : {gs.best_score_:.4f}")
    return best_results


def retrain_best_models(best_results: dict, X_train, y_train) -> dict:
    best_models = {}
    for cfg in MODELS_CONFIG:
        name = cfg["name"]
        best_p = best_results[name]["best_params"]
        fixed_p = cfg["default_params"].copy()
        final_p = {**fixed_p, **best_p}
        print(f"[train] Retraining {name} with best params: {best_p}")
        model = cfg["model_class"](**final_p)
        model.fit(X_train, y_train)
        best_models[name] = model
        print(f"  → {name} retrained ✓")
    return best_models


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.data_generator import generate_dataset
    from src.preprocess import full_pipeline

    df = generate_dataset()
    X_tr, X_te, y_tr, y_te, scaler, le_dict, df_enc = full_pipeline(df)
    default_models = train_default_models(X_tr, y_tr)
    cross_validate_model("Decision Tree", X_tr, y_tr)
    best_results = grid_search_all(X_tr, y_tr)
    best_models = retrain_best_models(best_results, X_tr, y_tr)
    print("\nAll models trained and tuned ✓")
