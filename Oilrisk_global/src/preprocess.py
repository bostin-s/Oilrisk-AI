"""
preprocess.py
=============
All data pre-processing steps:
  1. Data cleaning  — fill nulls, type conversion
  2. Feature engineering — casualties_avg, year/month/day
  3. Encoding — LabelEncoder per categorical column
  4. Feature selection — defines X (features) and y (target)
  5. Train-test split (80/20 stratified)
  6. Feature scaling — StandardScaler on numerical columns only
  7. Saves train_dataset.csv and test_dataset.csv
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# ── Column definitions ───────────────────────────────────────────────────────

CATEGORICAL_COLS = [
    "actor_attacker",
    "actor_target",
    "event_type",
    "casualty_confidence",
    "target_description",
]

NUMERICAL_COLS = [
    "latitude",
    "longitude",
    "reported_casualties_min",
    "reported_casualties_max",
    "casualties_avg",
    "oil_infrastructure_hit",
    "month",
    "day",
]

ALL_FEATURES = NUMERICAL_COLS + CATEGORICAL_COLS

RISK_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
RISK_MAP   = {r: i for i, r in enumerate(RISK_ORDER)}


# ── Step 1 — Data cleaning ───────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["reported_casualties_min"] = df["reported_casualties_min"].fillna(0)
    df["reported_casualties_max"] = df["reported_casualties_max"].fillna(0)
    mode_val = df["casualty_confidence"].mode()
    if len(mode_val) > 0:
        df["casualty_confidence"] = df["casualty_confidence"].fillna(mode_val[0])
    else:
        df["casualty_confidence"] = df["casualty_confidence"].fillna("Medium")
    df["date_utc"] = pd.to_datetime(df["date_utc"], errors="coerce")
    print("[preprocess] Cleaning complete — missing values remaining:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    return df


# ── Step 2 — Feature engineering ────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["casualties_avg"] = (
        df["reported_casualties_min"] + df["reported_casualties_max"]
    ) / 2
    df["year"]  = pd.to_datetime(df["date_utc"]).dt.year
    df["month"] = pd.to_datetime(df["date_utc"]).dt.month
    df["day"]   = pd.to_datetime(df["date_utc"]).dt.day
    print("[preprocess] Feature engineering done — new cols: casualties_avg, year, month, day")
    return df


# ── Step 3 — Encoding ────────────────────────────────────────────────────────

def encode(df: pd.DataFrame):
    df = df.copy()
    le_dict = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    df["oil_supply_risk_enc"] = df["oil_supply_risk"].map(RISK_MAP)
    print("[preprocess] Encoding complete — categorical columns encoded:")
    for col, le in le_dict.items():
        print(f"  {col}: {list(le.classes_)}")
    return df, le_dict


# ── Step 4 — Feature selection ───────────────────────────────────────────────

def select_features(df: pd.DataFrame):
    X = df[ALL_FEATURES].copy()
    y = df["oil_supply_risk_enc"].copy()
    print(f"[preprocess] Feature selection — X: {X.shape}, y: {y.shape}")
    return X, y


# ── Step 5 — Train-test split ────────────────────────────────────────────────

def split(X, y, test_size: float = 0.20, seed: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    print(f"[preprocess] Split — train: {X_train.shape[0]}, test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


# ── Step 6 — Feature scaling ─────────────────────────────────────────────────

def scale(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = X_train.copy()
    X_test_s  = X_test.copy()
    X_train_s[NUMERICAL_COLS] = scaler.fit_transform(X_train[NUMERICAL_COLS])
    X_test_s[NUMERICAL_COLS]  = scaler.transform(X_test[NUMERICAL_COLS])
    print("[preprocess] Scaling complete — StandardScaler fitted on train set")
    return X_train_s, X_test_s, scaler


# ── Step 7 — Save split CSVs ─────────────────────────────────────────────────

def save_splits(X_train, X_test, y_train, y_test, out_dir: str = "data"):
    os.makedirs(out_dir, exist_ok=True)
    train_df = X_train.copy()
    train_df["oil_supply_risk_enc"] = y_train.values
    train_path = os.path.join(out_dir, "train_dataset.csv")
    train_df.to_csv(train_path, index=False)
    test_df = X_test.copy()
    test_df["oil_supply_risk_enc"] = y_test.values
    test_path = os.path.join(out_dir, "test_dataset.csv")
    test_df.to_csv(test_path, index=False)
    print(f"[preprocess] Saved → {train_path}  ({len(train_df)} rows)")
    print(f"[preprocess] Saved → {test_path}  ({len(test_df)} rows)")


# ── Full pipeline convenience function ──────────────────────────────────────

def full_pipeline(df: pd.DataFrame, out_dir: str = "data"):
    df = clean(df)
    df = engineer_features(df)
    df_enc, le_dict = encode(df)
    X, y = select_features(df_enc)
    X_train, X_test, y_train, y_test = split(X, y)
    save_splits(X_train, X_test, y_train, y_test, out_dir)
    X_train_s, X_test_s, scaler = scale(X_train, X_test)
    return X_train_s, X_test_s, y_train, y_test, scaler, le_dict, df_enc


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.data_generator import generate_dataset
    df_raw = generate_dataset()
    X_tr, X_te, y_tr, y_te, scaler, le_dict, df_enc = full_pipeline(df_raw)
    print("\nX_train (scaled) head:")
    print(X_tr.head())
