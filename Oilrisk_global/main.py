"""
main.py
=======
End-to-end CLI pipeline for the Global Oil Supply Risk Prediction project.
Covers worldwide conflict zones: Israel–Iran, Red Sea, Russia–Ukraine,
Libya, Nigeria, Venezuela, South China Sea, Sudan, Azerbaijan, and more.

Run:
    python main.py
"""

import os, sys, time
import joblib
sys.path.insert(0, os.path.dirname(__file__))

from src.data_generator import generate_dataset, save_dataset
from src.preprocess     import (
    clean, engineer_features, encode, select_features,
    split, scale, save_splits, RISK_MAP, ALL_FEATURES, NUMERICAL_COLS
)
from src.train_models   import (
    train_default_models, cross_validate_model,
    grid_search_all, retrain_best_models, MODELS_CONFIG
)
from src.evaluate       import (
    plot_eda, plot_correlation_heatmap,
    evaluate_all_models, plot_feature_importance,
    plot_model_comparison, plot_confusion_matrices,
    export_excel, oil_sustainability_table,
)
from src.predict        import run_sample_predictions


DATA_DIR    = "data"
OUTPUT_DIR  = "outputs"
N_ROWS      = 5000
RANDOM_SEED = 42


def main():
    start = time.time()
    print("\n" + "=" * 65)
    print("  Global Oil Supply Risk Prediction — Full Pipeline")
    print("  Coverage: Israel–Iran · Red Sea · Russia–Ukraine")
    print("            Libya · Nigeria · Venezuela · S.China Sea")
    print("=" * 65 + "\n")

    print("── STEP 1: Generating global dataset ──")
    df_raw = generate_dataset(n=N_ROWS, seed=RANDOM_SEED)
    save_dataset(df_raw, out_dir=DATA_DIR)
    print(f"  Shape  : {df_raw.shape}")
    print(f"  Regions: {df_raw['region'].value_counts().to_string()}")
    print(f"  Risk   :\n{df_raw['oil_supply_risk'].value_counts().to_string()}\n")

    print("── STEP 2: Data cleaning ──")
    df_clean = clean(df_raw)

    print("\n── STEP 3: Feature engineering ──")
    df_feat = engineer_features(df_clean)

    print("\n── STEP 4: EDA Visualisations ──")
    plot_eda(df_feat, out_dir=OUTPUT_DIR)
    plot_correlation_heatmap(df_feat, out_dir=OUTPUT_DIR)

    print("\n── STEP 5: Encoding ──")
    df_enc, le_dict = encode(df_feat)

    print("\n── STEP 6: Feature selection ──")
    X, y = select_features(df_enc)

    print("\n── STEP 7: Train-test split ──")
    X_train, X_test, y_train, y_test = split(X, y, seed=RANDOM_SEED)
    save_splits(X_train, X_test, y_train, y_test, out_dir=DATA_DIR)
    df_train_raw = X_train.copy(); df_train_raw["oil_supply_risk_enc"] = y_train.values
    df_test_raw  = X_test.copy();  df_test_raw["oil_supply_risk_enc"]  = y_test.values

    print("\n── STEP 8: Feature scaling ──")
    X_train_s, X_test_s, scaler = scale(X_train, X_test)

    print("\n── STEP 9: Train all models (default params) ──")
    default_models = train_default_models(X_train_s, y_train)

    print("\n── STEP 10: Cross-validation (Decision Tree, 5-fold) ──")
    cv_result = cross_validate_model("Decision Tree", X_train_s, y_train, cv=5)
    print(f"  CV mean accuracy: {cv_result['mean']:.4f} ± {cv_result['std']:.4f}")

    print("\n── STEP 11: GridSearchCV (all models, 3-fold) ──")
    best_results = grid_search_all(X_train_s, y_train, cv=3)

    print("\n── STEP 12: Retrain with best hyperparameters ──")
    best_models = retrain_best_models(best_results, X_train_s, y_train)

    print("\n── STEP 12b: Saving models, scaler & encoders to outputs/ ──")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(best_models, os.path.join(OUTPUT_DIR, "models.pkl"))
    joblib.dump(scaler,      os.path.join(OUTPUT_DIR, "scaler.pkl"))
    joblib.dump(le_dict,     os.path.join(OUTPUT_DIR, "le_dict.pkl"))
    # Save results_df after evaluation — patch below
    print("  → models.pkl, scaler.pkl, le_dict.pkl saved ✓")

    print("\n── STEP 13: Evaluate all best models on test set ──")
    results_df = evaluate_all_models(best_models, X_test_s, y_test)
    joblib.dump(results_df, os.path.join(OUTPUT_DIR, "results_df.pkl"))
    print("  → results_df.pkl saved ✓")

    print("\n── STEP 14: Generate output charts ──")
    rf_model = best_models.get("Random Forest")
    if rf_model:
        plot_feature_importance(rf_model, ALL_FEATURES, out_dir=OUTPUT_DIR)
    plot_model_comparison(results_df, out_dir=OUTPUT_DIR)
    plot_confusion_matrices(best_models, X_test_s, y_test, out_dir=OUTPUT_DIR)

    print("\n── STEP 15: Real-time global sample predictions ──")
    best_model_name = results_df.iloc[0]["Model"]
    best_model      = best_models[best_model_name]
    print(f"  Using best model: {best_model_name} (accuracy={results_df.iloc[0]['Accuracy']:.4f})")
    run_sample_predictions(best_model, scaler, le_dict)

    print("\n── STEP 16: Oil stock sustainability calculator ──")
    oil_sustainability_table()

    print("\n── STEP 17: Export Excel workbook ──")
    export_excel(
        df_full=df_enc, df_train=df_train_raw, df_test=df_test_raw,
        results_df=results_df, best_model=best_model,
        scaler=scaler, le_dict=le_dict, out_dir=OUTPUT_DIR,
    )

    elapsed = time.time() - start
    print(f"\n{'='*65}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"{'='*65}")
    for root, _, files in os.walk(DATA_DIR):
        for f in sorted(files): print(f"    {os.path.join(root, f)}")
    for root, _, files in os.walk(OUTPUT_DIR):
        for f in sorted(files): print(f"    {os.path.join(root, f)}")
    print()


if __name__ == "__main__":
    main()