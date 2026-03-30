"""
app.py  —  Flask Web Application
Global Oil Supply Risk Prediction (Israel–Iran + Worldwide)
"""

import os, sys, json, threading, base64, traceback

sys.path.insert(0, os.path.dirname(__file__))

# ── Graceful xgboost fallback ─────────────────────────────────────────────────
try:
    import xgboost  # noqa: F401
except ImportError:
    import types
    from sklearn.ensemble import GradientBoostingClassifier
    _xgb_stub = types.ModuleType("xgboost")
    _xgb_stub.XGBClassifier = GradientBoostingClassifier
    sys.modules["xgboost"] = _xgb_stub

from flask import (
    Flask, render_template, request, jsonify
)
import pandas as pd

app = Flask(__name__)
app.config["SECRET_KEY"]      = "oil-risk-global-2025"
app.config["DATA_DIR"]        = os.path.join(os.path.dirname(__file__), "data")
app.config["OUTPUT_DIR"]      = os.path.join(os.path.dirname(__file__), "outputs")
# ── Global state ─────────────────────────────────────────────────────────────

_state = {
    "running":         False,
    "done":            False,
    "error":           None,
    "models":          {},
    "scaler":          None,
    "le_dict":         {},
    "results_df":      None,
    "df_encoded":      None,
    "best_model_name": None,
}


def _log(msg):
    print(msg, flush=True)


def _b64_chart(filename):
    path = os.path.join(app.config["OUTPUT_DIR"], filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _dataset_stats():
    p = os.path.join(app.config["DATA_DIR"], "Global_Oil_Risk_dataset.csv")
    if not os.path.exists(p):
        return {}
    df = pd.read_csv(p)
    stats = {
        "total_events":   len(df),
        "risk_dist":      df["oil_supply_risk"].value_counts().to_dict(),
        "event_types":    df["event_type"].value_counts().to_dict(),
        "oil_hit_rate":   round(df["oil_infrastructure_hit"].mean() * 100, 1),
        "avg_casualties": round(df[["reported_casualties_min","reported_casualties_max"]].mean().mean(), 1),
        "locations":      int(df["location_name"].nunique()),
        "regions":        int(df["region"].nunique()) if "region" in df.columns else 1,
        "region_dist":    df["region"].value_counts().to_dict() if "region" in df.columns else {},
        "oil_targets":    df[df["oil_infrastructure_hit"]==1]["target_description"].value_counts().head(6).to_dict(),
    }
    return stats


# ── Pipeline ──────────────────────────────────────────────────────────────────

def _run_pipeline():
    _state["running"] = True
    _state["done"]    = False
    _state["error"]   = None
    try:
        from src.data_generator import generate_dataset, save_dataset
        from src.preprocess import (
            clean, engineer_features, encode, select_features,
            split, scale, save_splits, ALL_FEATURES
        )
        from src.train_models import grid_search_all, retrain_best_models
        from src.evaluate import (
            plot_eda, plot_correlation_heatmap, evaluate_all_models,
            plot_feature_importance, plot_model_comparison,
            plot_confusion_matrices, export_excel
        )

        DD = app.config["DATA_DIR"]
        OD = app.config["OUTPUT_DIR"]
        os.makedirs(DD, exist_ok=True)
        os.makedirs(OD, exist_ok=True)

        _log("▶ [1/10] Generating 5,000-row global synthetic dataset…")
        df_raw = generate_dataset(n=5000, seed=42)
        save_dataset(df_raw, out_dir=DD)
        _log(f"   Shape: {df_raw.shape} | Regions: {df_raw['region'].nunique()}")

        _log("▶ [2/10] Cleaning data…")
        df_clean = clean(df_raw)

        _log("▶ [3/10] Feature engineering…")
        df_feat = engineer_features(df_clean)

        _log("▶ [4/10] Generating EDA visualisations…")
        plot_eda(df_feat, out_dir=OD)
        plot_correlation_heatmap(df_feat, out_dir=OD)

        _log("▶ [5/10] Encoding categorical features…")
        df_enc, le_dict = encode(df_feat)
        _state["le_dict"]    = le_dict
        _state["df_encoded"] = df_enc

        _log("▶ [6/10] Feature selection & 80/20 stratified split…")
        X, y = select_features(df_enc)
        X_train, X_test, y_train, y_test = split(X, y, seed=42)
        save_splits(X_train, X_test, y_train, y_test, out_dir=DD)

        _log("▶ [7/10] StandardScaler feature scaling…")
        X_train_s, X_test_s, scaler = scale(X_train, X_test)
        _state["scaler"] = scaler

        _log("▶ [8/10] GridSearchCV tuning (3-fold, 6 models)…")
        best_results = grid_search_all(X_train_s, y_train, cv=2)
        best_models  = retrain_best_models(best_results, X_train_s, y_train)
        _state["models"] = best_models

        _log("▶ [9/10] Evaluating all models on test set…")
        results_df = evaluate_all_models(best_models, X_test_s, y_test)
        _state["results_df"]      = results_df
        _state["best_model_name"] = results_df.iloc[0]["Model"]
        best_model = best_models[_state["best_model_name"]]

        _log("▶ [10/10] Charts & Excel export…")
        rf = best_models.get("Random Forest")
        if rf:
            plot_feature_importance(rf, ALL_FEATURES, out_dir=OD)
        plot_model_comparison(results_df, out_dir=OD)
        plot_confusion_matrices(best_models, X_test_s, y_test, out_dir=OD)

        df_tr = X_train.copy(); df_tr["oil_supply_risk_enc"] = y_train.values
        df_te = X_test.copy();  df_te["oil_supply_risk_enc"] = y_test.values
        export_excel(df_full=df_enc, df_train=df_tr, df_test=df_te,
                     results_df=results_df, best_model=best_model,
                     scaler=scaler, le_dict=le_dict, out_dir=OD)

        _log("✅ All 6 models trained on global dataset.")
        _state["done"] = True

    except Exception as e:
        msg = traceback.format_exc()
        _state["error"] = msg
        _log(f"❌ Error: {e}")
    finally:
        _state["running"] = False


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    results = None
    if _state["results_df"] is not None:
        results = _state["results_df"].to_dict(orient="records")
    return render_template("dashboard.html",
        best_model_name  = _state["best_model_name"],
        results          = results,
        dataset_stats    = _dataset_stats(),
    )



@app.route("/predict", methods=["GET", "POST"])
def predict_view():
    from src.data_generator import ACTORS_ATTACKER, ACTORS_TARGET, EVENT_TYPES, TARGET_DESCS
    if request.method == "GET":
        return render_template("predict.html",
            actors_attacker = ACTORS_ATTACKER,
            actors_target   = ACTORS_TARGET,
            event_types     = EVENT_TYPES,
            target_descs    = TARGET_DESCS,
        )
    if not _state["models"]:
        return jsonify({"error": "Models are loading, please wait."}), 400
    data = request.get_json(silent=True) or request.form.to_dict()
    try:
        from src.predict import predict_event
        risk, proba = predict_event(
            model               = _state["models"][_state["best_model_name"]],
            scaler              = _state["scaler"],
            le_dict             = _state["le_dict"],
            actor_attacker      = str(data["actor_attacker"]),
            actor_target        = str(data["actor_target"]),
            event_type          = str(data["event_type"]),
            target_description  = str(data["target_description"]),
            casualty_confidence = str(data["casualty_confidence"]),
            latitude            = float(data["latitude"]),
            longitude           = float(data["longitude"]),
            oil_infrastructure_hit = int(data["oil_infrastructure_hit"]),
            casualties_min      = float(data["casualties_min"]),
            casualties_max      = float(data["casualties_max"]),
            month               = int(data["month"]),
            day                 = int(data["day"]),
        )
        return jsonify({"risk": risk, "probabilities": proba,
                        "model_used": _state["best_model_name"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/batch-predict", methods=["GET", "POST"])
def batch_predict():
    from src.data_generator import ACTORS_ATTACKER, ACTORS_TARGET, EVENT_TYPES, TARGET_DESCS
    if request.method == "GET":
        from src.predict import SAMPLE_EVENTS
        samples = [{k: v for k, v in e.items()} for e in SAMPLE_EVENTS]
        return render_template("batch_predict.html",
            sample_events   = json.dumps(samples, indent=2),
            actors_attacker = ACTORS_ATTACKER,
            actors_target   = ACTORS_TARGET,
            event_types     = EVENT_TYPES,
            target_descs    = TARGET_DESCS,
        )
    if not _state["models"]:
        return jsonify({"error": "Models are loading, please wait."}), 400
    events = request.get_json(silent=True)
    if not isinstance(events, list):
        return jsonify({"error": "Body must be a JSON array."}), 400
    from src.predict import predict_batch
    try:
        clean_events = [{k: v for k, v in e.items() if k not in ("_label","label")} for e in events]
        df_r = predict_batch(_state["models"][_state["best_model_name"]],
                             _state["scaler"], _state["le_dict"], clean_events)
        return jsonify({"count": len(df_r), "results": df_r.to_dict(orient="records")})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/visualizations")
def visualizations():
    charts = {
        "EDA Overview":        _b64_chart("eda_visualization.png"),
        "Correlation Heatmap": _b64_chart("correlation_heatmap.png"),
        "Feature Importance":  _b64_chart("feature_importance.png"),
        "Model Comparison":    _b64_chart("model_comparison.png"),
        "Confusion Matrices":  _b64_chart("confusion_matrices.png"),
    }
    return render_template("visualizations.html", charts=charts)


@app.route("/dataset")
def dataset_explorer():
    p        = os.path.join(app.config["DATA_DIR"], "Global_Oil_Risk_dataset.csv")
    page     = int(request.args.get("page", 1))
    per_page = 25
    search   = request.args.get("q", "").strip()
    risk_f   = request.args.get("risk", "").strip()
    region_f = request.args.get("region", "").strip()
    columns, rows, total, total_pages = [], [], 0, 1
    if os.path.exists(p):
        df = pd.read_csv(p)
        if search:
            mask = df.apply(lambda r: r.astype(str).str.contains(search, case=False).any(), axis=1)
            df = df[mask]
        if risk_f:
            df = df[df["oil_supply_risk"] == risk_f]
        if region_f and "region" in df.columns:
            df = df[df["region"] == region_f]
        total       = len(df)
        total_pages = max(1, (total + per_page - 1) // per_page)
        page        = min(page, total_pages)
        df_page     = df.iloc[(page-1)*per_page : page*per_page]
        columns     = df.columns.tolist()
        rows        = df_page.fillna("").values.tolist()
    return render_template("dataset.html",
        columns=columns, rows=rows, page=page,
        total_pages=total_pages, total=total,
        search=search, risk_filter=risk_f, region_filter=region_f)


@app.route("/sustainability")
def sustainability():
    return render_template("sustainability.html")


# ── JSON API ──────────────────────────────────────────────────────────────────

@app.route("/api/stats")
def api_stats():
    return jsonify(_dataset_stats())


@app.route("/api/model-results")
def api_model_results():
    if _state["results_df"] is None:
        return jsonify({"error": "Models are loading, please wait."}), 404
    return jsonify(_state["results_df"].to_dict(orient="records"))


@app.route("/api/sustainability")
def api_sustainability():
    from src.evaluate import SUSTAINABILITY_SCENARIOS
    rows = []
    for sc in SUSTAINABILITY_SCENARIOS:
        eff  = sc["daily_consumption"] * (1 - sc["disruption_pct"] / 100)
        days = round(sc["total_stock"] / eff, 1) if eff > 0 else None
        rows.append({
            "scenario":         sc["label"],
            "disruption_pct":   sc["disruption_pct"],
            "effective_supply": round(eff),
            "days_sustainable": days,
        })
    return jsonify(rows)


@app.route("/api/risk-distribution")
def api_risk_dist():
    p = os.path.join(app.config["DATA_DIR"], "Global_Oil_Risk_dataset.csv")
    if not os.path.exists(p):
        return jsonify({})
    df = pd.read_csv(p)
    return jsonify(df["oil_supply_risk"].value_counts().to_dict())


@app.route("/api/region-distribution")
def api_region_dist():
    p = os.path.join(app.config["DATA_DIR"], "Global_Oil_Risk_dataset.csv")
    if not os.path.exists(p):
        return jsonify({})
    df = pd.read_csv(p)
    if "region" not in df.columns:
        return jsonify({})
    return jsonify(df["region"].value_counts().to_dict())


@app.route("/api/event-type-distribution")
def api_event_dist():
    p = os.path.join(app.config["DATA_DIR"], "Global_Oil_Risk_dataset.csv")
    if not os.path.exists(p):
        return jsonify({})
    df = pd.read_csv(p)
    return jsonify(df["event_type"].value_counts().to_dict())


@app.route("/api/live-events")
def api_live_events():
    """Returns simulated live risk events for the map ticker."""
    events = [
        {"region": "Middle East",  "location": "Strait of Hormuz",   "risk": "CRITICAL", "event": "Naval_Strike",      "time": "2m ago"},
        {"region": "Red Sea",      "location": "Bab-el-Mandeb",      "risk": "HIGH",     "event": "Missile_Strike",     "time": "7m ago"},
        {"region": "South Asia",   "location": "Indian Ocean Route",  "risk": "HIGH",     "event": "Mine_Attack",        "time": "12m ago"},
        {"region": "Middle East",  "location": "Kharg Island",       "risk": "CRITICAL", "event": "Airstrike",          "time": "22m ago"},
        {"region": "South Asia",   "location": "Jamnagar Refinery",  "risk": "MEDIUM",   "event": "Cyber_Attack",       "time": "35m ago"},
        {"region": "Europe",       "location": "Kyiv Pipeline",      "risk": "HIGH",     "event": "Pipeline_Sabotage",  "time": "41m ago"},
        {"region": "Africa",       "location": "Niger Delta",        "risk": "MEDIUM",   "event": "Ground_Assault",     "time": "1h ago"},
        {"region": "Asia-Pacific", "location": "S.China Sea",        "risk": "HIGH",     "event": "Naval_Strike",       "time": "1h ago"},
        {"region": "South Asia",   "location": "Mumbai Offshore",    "risk": "MEDIUM",   "event": "Drone_Attack",       "time": "2h ago"},
        {"region": "Americas",     "location": "Maracaibo",          "risk": "MEDIUM",   "event": "Pipeline_Sabotage",  "time": "3h ago"},
    ]
    return jsonify(events)


@app.route("/health")
def health():
    return jsonify({
        "status":        "ok",
        "models_loaded": list(_state["models"].keys()),
    })


@app.route("/api/chart/<n>")
def api_chart(n):
    """Return a single chart as base64 — frontend fetches individually, no page reload."""
    allowed = {
        "eda_visualization", "correlation_heatmap",
        "feature_importance", "model_comparison", "confusion_matrices"
    }
    if n not in allowed:
        return jsonify({"error": "unknown chart"}), 404
    data = _b64_chart(n + ".png")
    if not data:
        return jsonify({"error": "not ready"}), 404
    # Strip the data:image/png;base64, prefix — frontend adds it
    b64 = data.split(",", 1)[1]
    return jsonify({"data": b64})


@app.route("/api/status")
def api_status():
    """Frontend polls this to know when models are ready."""
    if _state["error"]:
        return jsonify({
            "ready":       False,
            "running":     False,
            "error":       _state["error"][:300],
        })
    return jsonify({
        "ready":       _state["done"],
        "running":     _state["running"],
        "model_count": len(_state["models"]),
        "best_model":  _state["best_model_name"],
    })


# ── Load pre-trained models on startup ───────────────────────────────────────
def _load_pretrained():
    """
    Load pre-trained models if available (Render build / after running main.py).
    If not found, automatically run the training pipeline in background.
    """
    import joblib
    OD = app.config["OUTPUT_DIR"]
    os.makedirs(OD, exist_ok=True)

    required = ["models.pkl", "scaler.pkl", "le_dict.pkl", "results_df.pkl"]
    missing  = [f for f in required if not os.path.exists(os.path.join(OD, f))]

    if missing:
        _log("⚠️  Model files not found — starting background training pipeline...")
        _log("   This takes ~5 min. /api/status will update when ready.")
        threading.Thread(target=_run_pipeline, daemon=True).start()
        return

    try:
        _state["models"]          = joblib.load(os.path.join(OD, "models.pkl"))
        _state["scaler"]          = joblib.load(os.path.join(OD, "scaler.pkl"))
        _state["le_dict"]         = joblib.load(os.path.join(OD, "le_dict.pkl"))
        _state["results_df"]      = joblib.load(os.path.join(OD, "results_df.pkl"))
        _state["best_model_name"] = _state["results_df"].iloc[0]["Model"]
        _state["done"]            = True
        _log(f"✅ Pre-trained models loaded instantly: {list(_state['models'].keys())}")
        _log(f"   Best model: {_state['best_model_name']}")
    except Exception as e:
        _log(f"❌ Failed to load model files: {e}")
        _log("   Falling back to background training...")
        threading.Thread(target=_run_pipeline, daemon=True).start()

threading.Thread(target=_load_pretrained, daemon=True).start()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)