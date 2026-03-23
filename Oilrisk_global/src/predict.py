"""
predict.py
==========
Real-time prediction functions for the global oil supply risk model.
"""

import numpy as np
import pandas as pd

RISK_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def predict_event(
    model, scaler, le_dict: dict,
    actor_attacker: str, actor_target: str,
    event_type: str, target_description: str,
    casualty_confidence: str,
    latitude: float, longitude: float,
    oil_infrastructure_hit: int,
    casualties_min: float, casualties_max: float,
    month: int, day: int,
) -> tuple:
    from src.preprocess import NUMERICAL_COLS, CATEGORICAL_COLS

    casualties_avg = (casualties_min + casualties_max) / 2

    cat_encoded = []
    for col, val in zip(
        CATEGORICAL_COLS,
        [actor_attacker, actor_target, event_type, casualty_confidence, target_description],
    ):
        le = le_dict[col]
        if val in le.classes_:
            cat_encoded.append(le.transform([val])[0])
        else:
            cat_encoded.append(0)

    num_row = np.array([[
        latitude, longitude,
        casualties_min, casualties_max, casualties_avg,
        oil_infrastructure_hit, month, day,
    ]])
    num_scaled = scaler.transform(num_row)
    full_row = np.hstack([num_scaled, np.array([cat_encoded])])

    pred_enc = int(model.predict(full_row)[0])
    proba    = model.predict_proba(full_row)[0]

    risk_label = RISK_ORDER[pred_enc]
    proba_dict = {label: round(float(p), 4) for label, p in zip(RISK_ORDER, proba)}
    return risk_label, proba_dict


def predict_batch(model, scaler, le_dict: dict, events: list) -> pd.DataFrame:
    records = []
    for event in events:
        risk, proba = predict_event(model=model, scaler=scaler, le_dict=le_dict, **event)
        row = event.copy()
        row["predicted_risk"] = risk
        for label, p in proba.items():
            row[f"prob_{label}"] = p
        records.append(row)
    return pd.DataFrame(records)


SAMPLE_EVENTS = [
    {
        "actor_attacker": "Israel", "actor_target": "Iran",
        "event_type": "Airstrike", "target_description": "Oil_Refinery",
        "casualty_confidence": "High", "latitude": 29.25, "longitude": 50.33,
        "oil_infrastructure_hit": 1, "casualties_min": 45, "casualties_max": 60,
        "month": 4, "day": 15, "_label": "Kharg Island Refinery Strike",
    },
    {
        "actor_attacker": "Houthis", "actor_target": "International_Shipping",
        "event_type": "Missile_Strike", "target_description": "Oil_Tanker",
        "casualty_confidence": "High", "latitude": 12.58, "longitude": 43.42,
        "oil_infrastructure_hit": 1, "casualties_min": 10, "casualties_max": 20,
        "month": 1, "day": 12, "_label": "Red Sea Oil Tanker Attack",
    },
    {
        "actor_attacker": "Russia", "actor_target": "Ukraine",
        "event_type": "Missile_Strike", "target_description": "Pipeline",
        "casualty_confidence": "Medium", "latitude": 49.99, "longitude": 36.23,
        "oil_infrastructure_hit": 1, "casualties_min": 5, "casualties_max": 15,
        "month": 6, "day": 22, "_label": "Ukraine Pipeline Strike",
    },
    {
        "actor_attacker": "IRGC", "actor_target": "US_Base",
        "event_type": "Drone_Attack", "target_description": "Pipeline",
        "casualty_confidence": "High", "latitude": 26.57, "longitude": 56.25,
        "oil_infrastructure_hit": 1, "casualties_min": 20, "casualties_max": 35,
        "month": 10, "day": 7, "_label": "Strait of Hormuz Pipeline Drone Attack",
    },
    {
        "actor_attacker": "IDF", "actor_target": "Iran",
        "event_type": "Cyber_Attack", "target_description": "Power_Plant",
        "casualty_confidence": "Low", "latitude": 33.72, "longitude": 51.93,
        "oil_infrastructure_hit": 0, "casualties_min": 0, "casualties_max": 2,
        "month": 3, "day": 12, "_label": "Natanz Power Grid Cyber Attack",
    },
]


def run_sample_predictions(model, scaler, le_dict: dict) -> None:
    print(f"\n{'='*65}")
    print(f"  REAL-TIME GLOBAL OIL SUPPLY RISK PREDICTIONS")
    print(f"  Model: {type(model).__name__}")
    print(f"{'='*65}")
    for event in SAMPLE_EVENTS:
        label = event.pop("_label")
        risk, proba = predict_event(model=model, scaler=scaler, le_dict=le_dict, **event)
        event["_label"] = label
        print(f"\n  Event     : {label}")
        print(f"  Type      : {event['event_type']}  →  Target: {event['target_description']}")
        print(f"  Oil Hit   : {'YES' if event['oil_infrastructure_hit'] else 'NO'}  |  Casualties: {event['casualties_min']}–{event['casualties_max']}")
        print(f"  ► Prediction : {risk}")
        print(f"  ► Probabilities : {proba}")
    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.data_generator import generate_dataset
    from src.preprocess import full_pipeline
    from src.train_models import train_default_models
    df = generate_dataset()
    X_tr, X_te, y_tr, y_te, scaler, le_dict, df_enc = full_pipeline(df)
    models = train_default_models(X_tr, y_tr)
    run_sample_predictions(models["Random Forest"], scaler, le_dict)
