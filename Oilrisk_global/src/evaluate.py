"""
evaluate.py
===========
All evaluation and visualisation functions for the Global Oil Supply Risk project.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── Constants ────────────────────────────────────────────────────────────────

RISK_ORDER  = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
RISK_COLORS = ["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]


# ── 1. EDA Visualisation ─────────────────────────────────────────────────────

def plot_eda(df: pd.DataFrame, out_dir: str = "outputs") -> str:
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Global Oil Supply Risk Dataset — Exploratory Data Analysis",
        fontsize=16, fontweight="bold"
    )

    sns.countplot(
        x="oil_supply_risk", data=df, order=RISK_ORDER,
        palette=dict(zip(RISK_ORDER, RISK_COLORS)), ax=axes[0, 0]
    )
    axes[0, 0].set_title("Oil Supply Risk Distribution")
    axes[0, 0].set_xlabel("Risk Level")
    axes[0, 0].set_ylabel("Count")
    for bar in axes[0, 0].patches:
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            int(bar.get_height()),
            ha="center", va="bottom", fontsize=10
        )

    sns.countplot(
        y="event_type", data=df,
        order=df["event_type"].value_counts().index,
        palette="Blues_r", ax=axes[0, 1]
    )
    axes[0, 1].set_title("Event Type Frequency")
    axes[0, 1].set_xlabel("Count")

    sc = axes[0, 2].scatter(
        df["longitude"], df["latitude"],
        c=df["oil_infrastructure_hit"],
        cmap="RdYlGn_r", alpha=0.4, s=5
    )
    plt.colorbar(sc, ax=axes[0, 2], label="Oil Hit (1=Yes)")
    axes[0, 2].set_title("Global Strike Locations (Red = Oil Infrastructure Hit)")
    axes[0, 2].set_xlabel("Longitude")
    axes[0, 2].set_ylabel("Latitude")

    sns.histplot(df["casualties_avg"], bins=40, kde=True, color="#e74c3c", ax=axes[1, 0])
    axes[1, 0].set_title("Average Casualties Distribution")
    axes[1, 0].set_xlabel("Avg Casualties")

    hit_by_event = (
        df.groupby("event_type")["oil_infrastructure_hit"]
        .mean().sort_values(ascending=False)
    )
    hit_by_event.plot(kind="bar", ax=axes[1, 1], color="#e67e22", edgecolor="black")
    axes[1, 1].set_title("Oil Infrastructure Hit Rate by Event Type")
    axes[1, 1].set_ylabel("Hit Rate (0–1)")
    axes[1, 1].tick_params(axis="x", rotation=45)

    if "region" in df.columns:
        region_risk = (
            df.groupby(["region", "oil_supply_risk"])
            .size().unstack(fill_value=0)
        )
        for r in RISK_ORDER:
            if r not in region_risk.columns:
                region_risk[r] = 0
        region_risk[RISK_ORDER].plot(
            kind="bar", stacked=True,
            color=RISK_COLORS, ax=axes[1, 2]
        )
        axes[1, 2].set_title("Risk Level by Region")
    else:
        attacker_risk = (
            df.groupby(["actor_attacker", "oil_supply_risk"])
            .size().unstack(fill_value=0)
        )
        for r in RISK_ORDER:
            if r not in attacker_risk.columns:
                attacker_risk[r] = 0
        attacker_risk[RISK_ORDER].plot(
            kind="bar", stacked=True,
            color=RISK_COLORS, ax=axes[1, 2]
        )
        axes[1, 2].set_title("Risk Level by Attacker")

    axes[1, 2].tick_params(axis="x", rotation=45)
    axes[1, 2].legend(title="Risk Level", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    path = os.path.join(out_dir, "eda_visualization.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] EDA chart saved → {path}")
    return path


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: str = "outputs") -> str:
    os.makedirs(out_dir, exist_ok=True)
    num_cols = [
        "latitude", "longitude",
        "reported_casualties_min", "reported_casualties_max",
        "casualties_avg", "oil_infrastructure_hit",
        "month", "day",
    ]
    available = [c for c in num_cols if c in df.columns]
    plt.figure(figsize=(10, 7))
    corr = df[available].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Correlation heatmap saved → {path}")
    return path


# ── 2. Model evaluation ───────────────────────────────────────────────────────

def evaluate_model(model_name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*55}")
    print(f"  Model: {model_name}")
    print(f"{'='*55}")
    print(f"  Accuracy : {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=RISK_ORDER))
    return {"name": model_name, "acc": round(acc, 4), "y_pred": y_pred}


def evaluate_all_models(models: dict, X_test, y_test) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        result = evaluate_model(name, model, X_test, y_test)
        rows.append({"Model": result["name"], "Accuracy": result["acc"]})
    df_res = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    print("\n=== Model Comparison Summary ===")
    print(df_res.to_string(index=False))
    return df_res


# ── 3. Feature importance chart ───────────────────────────────────────────────

def plot_feature_importance(rf_model, feature_names: list, out_dir: str = "outputs") -> str:
    os.makedirs(out_dir, exist_ok=True)
    importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)
    plt.figure(figsize=(12, 5))
    importances.plot(kind="bar", color="steelblue", edgecolor="black")
    plt.title("Random Forest — Feature Importance", fontsize=14, fontweight="bold")
    plt.ylabel("Importance Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(out_dir, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Feature importance chart saved → {path}")
    return path


# ── 4. Model comparison chart ─────────────────────────────────────────────────

def plot_model_comparison(results_df: pd.DataFrame, out_dir: str = "outputs") -> str:
    os.makedirs(out_dir, exist_ok=True)
    max_acc = results_df["Accuracy"].max()
    colors = ["#8e44ad" if acc == max_acc else "#3498db" for acc in results_df["Accuracy"]]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results_df["Model"], results_df["Accuracy"], color=colors, edgecolor="black")
    plt.ylim(0, 1.10)
    for bar, val in zip(bars, results_df["Accuracy"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", fontsize=11, fontweight="bold"
        )
    plt.title("Model Accuracy Comparison (Best Hyperparameters)", fontsize=14, fontweight="bold")
    plt.ylabel("Test Accuracy")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = os.path.join(out_dir, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Model comparison chart saved → {path}")
    return path


# ── 5. Confusion matrix grid ──────────────────────────────────────────────────

def plot_confusion_matrices(models: dict, X_test, y_test, out_dir: str = "outputs") -> str:
    os.makedirs(out_dir, exist_ok=True)
    model_list = list(models.items())
    n = len(model_list)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 5 + 1))
    axes = axes.flatten()
    for i, (name, model) in enumerate(model_list):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
            xticklabels=RISK_ORDER, yticklabels=RISK_ORDER
        )
        axes[i].set_title(name, fontweight="bold", fontsize=12)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Confusion Matrices — All Models", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Confusion matrix grid saved → {path}")
    return path


# ── 6. Oil stock sustainability ───────────────────────────────────────────────

SUSTAINABILITY_SCENARIOS = [
    {"label": "Baseline (No Disruption)",  "total_stock": 400_000_000, "daily_consumption": 20_000_000, "disruption_pct": 0},
    {"label": "LOW disruption",            "total_stock": 400_000_000, "daily_consumption": 20_000_000, "disruption_pct": 10},
    {"label": "MEDIUM disruption",         "total_stock": 400_000_000, "daily_consumption": 20_000_000, "disruption_pct": 20},
    {"label": "HIGH disruption",           "total_stock": 400_000_000, "daily_consumption": 20_000_000, "disruption_pct": 40},
    {"label": "CRITICAL disruption",       "total_stock": 400_000_000, "daily_consumption": 20_000_000, "disruption_pct": 65},
]


def oil_sustainability_table() -> pd.DataFrame:
    rows = []
    for sc in SUSTAINABILITY_SCENARIOS:
        eff_supply = sc["daily_consumption"] * (1 - sc["disruption_pct"] / 100)
        days = sc["total_stock"] / eff_supply if eff_supply > 0 else float("inf")
        rows.append({
            "Scenario":              sc["label"],
            "Total_Stock_BBL":       sc["total_stock"],
            "Disruption_%":          sc["disruption_pct"],
            "Effective_Supply_Day":  round(eff_supply),
            "Days_Sustainable":      round(days, 1),
        })
    df = pd.DataFrame(rows)
    print("\n[evaluate] Oil Stock Sustainability:")
    print(df.to_string(index=False))
    return df


# ── 7. Excel export ───────────────────────────────────────────────────────────

def export_excel(df_full, df_train, df_test, results_df, best_model,
                 scaler, le_dict, out_dir="outputs") -> str:
    os.makedirs(out_dir, exist_ok=True)
    from src.preprocess import NUMERICAL_COLS, ALL_FEATURES, RISK_ORDER
    RISK_MAP_INV = {i: r for i, r in enumerate(RISK_ORDER)}
    df_export = df_full.copy()
    X_all = df_export[ALL_FEATURES].copy()
    X_all[NUMERICAL_COLS] = scaler.transform(X_all[NUMERICAL_COLS])
    df_export["predicted_risk_enc"] = best_model.predict(X_all)
    df_export["predicted_risk"]     = df_export["predicted_risk_enc"].map(RISK_MAP_INV)
    df_export["prediction_correct"] = (
        df_export["oil_supply_risk"] == df_export["predicted_risk"]
    )
    path = os.path.join(out_dir, "global_oil_risk_project.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        cols_to_drop = [c for c in ["date_utc", "time_utc"] if c in df_export.columns]
        df_export.drop(columns=cols_to_drop).to_excel(
            writer, sheet_name="Full_Dataset_Predictions", index=False
        )
        df_train.to_excel(writer, sheet_name="Train_Dataset", index=False)
        df_test.to_excel(writer, sheet_name="Test_Dataset", index=False)
        results_df.to_excel(writer, sheet_name="Model_Results", index=False)
        oil_sustainability_table().to_excel(writer, sheet_name="Oil_Sustainability", index=False)
    print(f"[evaluate] Excel workbook saved → {path}")
    return path


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.data_generator import generate_dataset
    from src.preprocess import full_pipeline
    df_raw = generate_dataset()
    X_tr, X_te, y_tr, y_te, scaler, le_dict, df_enc = full_pipeline(df_raw)
    df_raw["casualties_avg"] = (
        df_raw["reported_casualties_min"].fillna(0) +
        df_raw["reported_casualties_max"].fillna(0)
    ) / 2
    plot_eda(df_raw)
    plot_correlation_heatmap(df_raw)
    oil_sustainability_table()
    print("\nEDA complete ✓")
