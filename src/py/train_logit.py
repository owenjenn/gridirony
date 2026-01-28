# src/py/train_logit.py
"""
Elastic-net logistic regression for NFL win probability prediction.
Python equivalent of src/r/train_logit.R
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
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
    print(f"{'Split':<12} | {'Brier':>10} | {'LogLoss':>10} | {'ROC-AUC':>10} |")
    print("-" * 80)
    for split, (brier, logloss, auc) in metrics.items():
        print(f"{split:<12} | {brier:>10.4f} | {logloss:>10.4f} | {auc:>10.4f} |")
    print("-" * 80)


def print_example_predictions(test_df: pd.DataFrame, n: int = 5):
    print(f"\nEXAMPLE PREDICTIONS (last {n} games)")
    print("-" * 80)
    print(f"{'Game':<35} | {'Home Win %':>10} | {'Predicted':>10} | {'Actual':>10}")
    print("-" * 80)

    samples = test_df.tail(n)
    for _, row in samples.iterrows():
        game = f"{row['away']} @ {row['home']}"
        prob = row['pred_prob']
        predicted = "HOME" if prob >= 0.5 else "AWAY"
        actual = "HOME" if row['home_win'] == 1 else "AWAY"
        correct = "Y" if predicted == actual else "X"

        print(f"{game:<35} | {prob*100:>9.1f}% | {predicted:>10} | {actual:>10} {correct}")

    print("-" * 80)
    print("\nHOW TO READ: Home Win % = probability home team wins")
    print("             Y = correct prediction, X = incorrect prediction")


def main():
    print_header("WIN PROBABILITY MODEL (Logistic Regression)")

    df = pd.read_parquet("data/features/model_df.parquet")
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    x_cols = get_feature_columns(df)
    if not x_cols:
        raise ValueError("No feature columns found. Run build_dataset.py first.")

    df = df.dropna(subset=["home_win"] + x_cols)

    train = df[df["season"] <= 2019].copy()
    val = df[df["season"] == 2020].copy()
    test = df[df["season"] >= 2021].copy()

    print(f"\nFeatures: {len(x_cols)} columns")
    print(f"Data splits: Train={len(train):,} | Val={len(val):,} | Test={len(test):,}")
    print("\nTraining model...")

    X_train, y_train = train[x_cols].values, train["home_win"].values
    X_val, y_val = val[x_cols].values, val["home_win"].values

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    np.random.seed(42)
    model = LogisticRegressionCV(
        penalty="elasticnet",
        l1_ratios=[0.5],
        solver="saga",
        cv=10,
        random_state=42,
        max_iter=5000
    )
    model.fit(X_train_sc, y_train)

    # Train metrics (for overfitting check)
    train_prob = model.predict_proba(X_train_sc)[:, 1]
    train_brier = brier_score_loss(y_train, train_prob)
    train_logloss = log_loss(y_train, train_prob)
    train_auc = roc_auc_score(y_train, train_prob)

    # Validation metrics
    val_prob = model.predict_proba(X_val_sc)[:, 1]
    val_brier = brier_score_loss(y_val, val_prob)
    val_logloss = log_loss(y_val, val_prob)
    val_auc = roc_auc_score(y_val, val_prob)

    # Refit on train+val, evaluate on test
    train_val = pd.concat([train, val])
    X_trv, y_trv = train_val[x_cols].values, train_val["home_win"].values

    scaler2 = StandardScaler()
    X_trv_sc = scaler2.fit_transform(X_trv)
    X_test_sc = scaler2.transform(test[x_cols].values)

    model2 = LogisticRegressionCV(
        penalty="elasticnet",
        l1_ratios=[0.5],
        solver="saga",
        cv=10,
        random_state=42,
        max_iter=5000
    )
    model2.fit(X_trv_sc, y_trv)

    # Final train metrics (on combined train+val)
    trv_prob = model2.predict_proba(X_trv_sc)[:, 1]
    trv_brier = brier_score_loss(y_trv, trv_prob)
    trv_logloss = log_loss(y_trv, trv_prob)
    trv_auc = roc_auc_score(y_trv, trv_prob)

    # Test metrics
    test["pred_prob"] = model2.predict_proba(X_test_sc)[:, 1]
    test_brier = brier_score_loss(test["home_win"], test["pred_prob"])
    test_logloss = log_loss(test["home_win"], test["pred_prob"])
    test_auc = roc_auc_score(test["home_win"], test["pred_prob"])

    # Print results table
    metrics = {
        "Train": (trv_brier, trv_logloss, trv_auc),
        "Test": (test_brier, test_logloss, test_auc),
    }
    print_metrics_table(metrics)

    # Overfitting check (using AUC - higher is better, so train > test = overfitting)
    auc_diff = trv_auc - test_auc
    auc_pct = (auc_diff / trv_auc) * 100
    print("\nOVERFITTING CHECK")
    print("-" * 80)
    print(f"Train AUC: {trv_auc:.4f} | Test AUC: {test_auc:.4f} | Difference: {auc_diff:+.4f} ({auc_pct:+.1f}%)")
    if abs(auc_pct) < 5:
        print("Status: LOW overfitting risk - model generalizes well")
    elif abs(auc_pct) < 15:
        print("Status: MODERATE overfitting risk - consider regularization")
    else:
        print("Status: HIGH overfitting risk - model may not generalize")

    # Accuracy summary
    test["pred_win"] = (test["pred_prob"] >= 0.5).astype(int)
    accuracy = (test["pred_win"] == test["home_win"]).mean()
    print(f"\nTest Accuracy: {accuracy:.1%} ({int(accuracy * len(test))}/{len(test)} correct)")

    # Example predictions
    print_example_predictions(test, n=5)

    # Save artifacts
    Path("artifacts").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    joblib.dump({"model": model2, "scaler": scaler2, "features": x_cols},
                "artifacts/logit_elasticnet.pkl")
    test[["game_id", "season", "week", "home", "away", "pred_prob", "home_win"]].to_csv(
        "reports/logit_test_predictions.csv", index=False
    )

    print(f"\nSaved: artifacts/logit_elasticnet.pkl")
    print(f"Saved: reports/logit_test_predictions.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
