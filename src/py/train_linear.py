# src/py/train_linear.py
"""
Elastic-net linear regression for NFL point margin prediction.
Python equivalent of src/r/train_linear.R
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Select rolling stat columns + optional betting/game columns.
    Only includes columns that have at least some non-NaN values.
    """
    roll_patterns = ["points_", "pass_yards_", "rush_yards_", "to_diff_"]
    roll_cols = [
        c for c in df.columns
        if any(p in c for p in roll_patterns)
        and c.startswith(("home_", "away_"))
        and df[c].dtype in [np.float64, np.int64, float, int]
        and df[c].notna().any()
    ]
    opt_cols = [c for c in ["spread_close", "total_close", "is_divisional"] if c in df.columns]
    return roll_cols + opt_cols


def print_header(title: str):
    print("=" * 80)
    print(f"{title:^80}")
    print("=" * 80)


def print_metrics_table(metrics: dict):
    print("\nRESULTS")
    print("-" * 80)
    print(f"{'Split':<12} | {'MAE':>10} | {'RMSE':>10} |")
    print("-" * 80)
    for split, (mae, rmse) in metrics.items():
        print(f"{split:<12} | {mae:>10.3f} | {rmse:>10.3f} |")
    print("-" * 80)


def print_example_predictions(test_df: pd.DataFrame, n: int = 5):
    print(f"\nEXAMPLE PREDICTIONS (last {n} games)")
    print("-" * 80)
    print(f"{'Game':<35} | {'Predicted':>10} | {'Actual':>10} | {'Error':>10}")
    print("-" * 80)

    samples = test_df.tail(n)
    for _, row in samples.iterrows():
        game = f"{row['away']} @ {row['home']}"
        pred = row['pred_margin']
        actual = row['margin']
        error = pred - actual

        # Interpret prediction
        if pred > 0:
            winner = f"{row['home']} by {abs(pred):.1f}"
        else:
            winner = f"{row['away']} by {abs(pred):.1f}"

        print(f"{game:<35} | {pred:>+10.1f} | {actual:>+10.0f} | {error:>+10.1f}")

    print("-" * 80)
    print("\nHOW TO READ: Positive margin = Home team wins by X points")
    print("             Negative margin = Away team wins by X points")


def main():
    print_header("POINT SPREAD MODEL (Linear Regression)")

    df = pd.read_parquet("data/features/model_df.parquet")
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    x_cols = get_feature_columns(df)
    if not x_cols:
        raise ValueError("No feature columns found. Run build_dataset.py first.")

    df = df.dropna(subset=["margin"] + x_cols)

    train = df[df["season"] <= 2019].copy()
    val = df[df["season"] == 2020].copy()
    test = df[df["season"] >= 2021].copy()

    print(f"\nFeatures: {len(x_cols)} columns")
    print(f"Data splits: Train={len(train):,} | Val={len(val):,} | Test={len(test):,}")
    print("\nTraining model...")

    X_train, y_train = train[x_cols].values, train["margin"].values
    X_val, y_val = val[x_cols].values, val["margin"].values

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    np.random.seed(42)
    model = ElasticNetCV(l1_ratio=0.5, cv=10, random_state=42, max_iter=5000)
    model.fit(X_train_sc, y_train)

    # Train metrics (for overfitting check)
    train_pred = model.predict(X_train_sc)
    train_mae = mean_absolute_error(y_train, train_pred)
    train_rmse = root_mean_squared_error(y_train, train_pred)

    # Validation metrics
    val_pred = model.predict(X_val_sc)
    val_mae = mean_absolute_error(y_val, val_pred)
    val_rmse = root_mean_squared_error(y_val, val_pred)

    # Refit on train+val, evaluate on test
    train_val = pd.concat([train, val])
    X_trv, y_trv = train_val[x_cols].values, train_val["margin"].values

    scaler2 = StandardScaler()
    X_trv_sc = scaler2.fit_transform(X_trv)
    X_test_sc = scaler2.transform(test[x_cols].values)

    model2 = ElasticNetCV(l1_ratio=0.5, cv=10, random_state=42, max_iter=5000)
    model2.fit(X_trv_sc, y_trv)

    # Final train metrics (on combined train+val)
    trv_pred = model2.predict(X_trv_sc)
    trv_mae = mean_absolute_error(y_trv, trv_pred)
    trv_rmse = root_mean_squared_error(y_trv, trv_pred)

    # Test metrics
    test["pred_margin"] = model2.predict(X_test_sc)
    test_mae = mean_absolute_error(test["margin"], test["pred_margin"])
    test_rmse = root_mean_squared_error(test["margin"], test["pred_margin"])

    # Print results table
    metrics = {
        "Train": (trv_mae, trv_rmse),
        "Test": (test_mae, test_rmse),
    }
    print_metrics_table(metrics)

    # Overfitting check
    mae_diff = test_mae - trv_mae
    mae_pct = (mae_diff / trv_mae) * 100
    print("\nOVERFITTING CHECK")
    print("-" * 80)
    print(f"Train MAE: {trv_mae:.3f} | Test MAE: {test_mae:.3f} | Difference: {mae_diff:+.3f} ({mae_pct:+.1f}%)")
    if abs(mae_pct) < 10:
        print("Status: LOW overfitting risk - model generalizes well")
    elif abs(mae_pct) < 20:
        print("Status: MODERATE overfitting risk - consider regularization")
    else:
        print("Status: HIGH overfitting risk - model may not generalize")

    # Example predictions
    print_example_predictions(test, n=5)

    # Save artifacts
    Path("artifacts").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    joblib.dump({"model": model2, "scaler": scaler2, "features": x_cols},
                "artifacts/linear_elasticnet.pkl")
    test[["game_id", "season", "week", "home", "away", "pred_margin", "margin"]].to_csv(
        "reports/linear_test_predictions.csv", index=False
    )

    print(f"\nSaved: artifacts/linear_elasticnet.pkl")
    print(f"Saved: reports/linear_test_predictions.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
